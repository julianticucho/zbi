import json
import os

import numpy as np
import torch
from torch.distributions import Uniform
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

from zbi.data.zarr_store import ZarrStore
from zbi.inference.posterior import Posterior
from zbi.neural_nets.factory import build_maf_estimator
from zbi.pipeline._data import SlicedDataset, load_to_ram
from zbi.pipeline._resolve import build_prior_from_meta, resolve_class
from zbi.pipeline.setup import DEFAULT_MAF
from zbi.pipeline.simulate import simulate_round
from zbi.utils.checkpoint import save_checkpoint
from zbi.utils.plotting import kl_matrix
from zbi.utils.preprocessing import compute_z_scores_streaming


HP_KEYS = [
    "batch_size", "lr", "max_epochs", "stop_after_epochs",
]


def _max_offdiag(K):
    off = K[~np.eye(K.shape[0], dtype=bool)]
    off = off[np.isfinite(off)]
    return float(off.max()) if off.size else float("nan")


def _build_model(config, dim_maf, theta_mean, theta_std, x_mean, x_std, device):
    emb_config = config["embedding"]
    emb_class = resolve_class(emb_config["class"])
    emb_kwargs = {k: v for k, v in emb_config.items() if k != "class"}
    embedding_net = emb_class(**emb_kwargs)

    maf_config = config.get("maf", DEFAULT_MAF)
    maf = build_maf_estimator(
        dim_theta=dim_maf, dim_x=config["dim_x"],
        theta_mean=theta_mean, theta_std=theta_std,
        x_mean=x_mean, x_std=x_std,
        embedding_net=embedding_net,
        z_score_theta="independent", z_score_x="independent",
        hidden_features=maf_config["hidden_features"],
        num_transforms=maf_config["num_transforms"],
        num_blocks=maf_config["num_blocks"],
    )
    return maf.to(device)


def _make_loaders(dataset, n_sims, batch_size, seed):
    num_train = int(0.9 * n_sims)
    perm = torch.randperm(n_sims, generator=torch.Generator().manual_seed(seed))

    train_indices = perm[:num_train].tolist()
    val_indices = perm[num_train:].tolist()

    train_loader = DataLoader(
        dataset,
        batch_size=min(batch_size, num_train),
        sampler=SubsetRandomSampler(train_indices),
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=min(batch_size, n_sims - num_train),
        sampler=SubsetRandomSampler(val_indices),
        drop_last=True,
    )
    return train_loader, val_loader


def _train_one_epoch(maf, loader, optimizer, device):
    maf.train()
    total_loss = 0.0
    n = 0
    for theta_batch, x_batch in loader:
        theta_batch = theta_batch.to(device)
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        loss = maf.loss(theta_batch, x_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(maf.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * theta_batch.shape[0]
        n += theta_batch.shape[0]
    return total_loss / n


def _val_loss(maf, loader, device):
    maf.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for theta_batch, x_batch in loader:
            theta_batch = theta_batch.to(device)
            x_batch = x_batch.to(device)
            loss = maf.loss(theta_batch, x_batch)
            total_loss += loss.item() * theta_batch.shape[0]
            n += theta_batch.shape[0]
    return total_loss / n


def _save_ensemble_checkpoint(path, maf, config, n_sims, tag):
    emb_config = config["embedding"]
    maf_config = config.get("maf", DEFAULT_MAF)
    torch.save(
        {
            "round_id": 0,
            "maf_state_dict": maf.state_dict(),
            "maf_build_kwargs": dict(
                dim_theta=config["dim_theta"],
                dim_x=config["dim_x"],
                num_transforms=maf_config["num_transforms"],
                hidden_features=maf_config["hidden_features"],
                num_blocks=maf_config["num_blocks"],
            ),
            "config": config,
            "embedding_net_config": emb_config,
            "prior_meta": config["prior_meta"],
        },
        path,
    )


def train_round(
    round_id: int,
    n_sims: int,
    offset: int,
    zarr_store: ZarrStore,
    config: dict,
    x_o: torch.Tensor,
    original_prior,
    proposal,
    device: str = "cpu",
    tag: str | None = None,
    load_to_ram_flag: bool = False,
):
    val_fraction = 0.1
    num_training = int((1 - val_fraction) * n_sims)
    permuted = torch.randperm(n_sims) + offset
    train_indices = permuted[:num_training]
    val_indices = permuted[num_training:]

    interest_dims = config.get("interest_dims")
    dim_maf = config.get("dim_interest", config["dim_theta"])

    train_idx_list = train_indices.tolist()
    theta_mean, theta_std = compute_z_scores_streaming(zarr_store, train_idx_list, "theta")
    x_mean, x_std = compute_z_scores_streaming(zarr_store, train_idx_list, "x")
    if interest_dims:
        theta_mean = theta_mean[interest_dims]
        theta_std = theta_std[interest_dims]

    emb_config = config.get("embedding", {"class": "EmbeddingNet"})
    emb_class = resolve_class(emb_config.get("class", "EmbeddingNet"))
    emb_kwargs = {k: v for k, v in emb_config.items() if k != "class"}
    embedding_net = emb_class(**emb_kwargs)

    maf_config = config.get("maf", DEFAULT_MAF)
    maf = build_maf_estimator(
        dim_theta=dim_maf, dim_x=config["dim_x"],
        theta_mean=theta_mean, theta_std=theta_std,
        x_mean=x_mean, x_std=x_std,
        embedding_net=embedding_net,
        z_score_theta="independent", z_score_x="independent",
        hidden_features=maf_config["hidden_features"],
        num_transforms=maf_config["num_transforms"],
        num_blocks=maf_config["num_blocks"],
    )
    maf.to(device)

    is_marginal = interest_dims is not None and len(interest_dims) < config["dim_theta"]
    if is_marginal:
        marginal_proposal = Uniform(
            original_prior.low[interest_dims],
            original_prior.high[interest_dims],
        )
        posterior = Posterior(maf, marginal_proposal, device=device)
    else:
        posterior = Posterior(maf, proposal, device=device)

    if load_to_ram_flag:
        dataset = load_to_ram(zarr_store, permuted.tolist(), interest_dims)
        train_indices_loader = torch.arange(num_training)
        val_indices_loader = torch.arange(num_training, n_sims)
    else:
        dataset = zarr_store.get_dataset()
        if is_marginal:
            dataset = SlicedDataset(dataset, interest_dims)
        train_indices_loader = train_indices
        val_indices_loader = val_indices

    train_batch_size = min(config["batch_size"], len(train_indices))
    train_loader = DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=SubsetRandomSampler(train_indices_loader.tolist()), drop_last=True,
    )
    val_batch_size = min(config["batch_size"], len(val_indices))
    val_loader = DataLoader(
        dataset, batch_size=val_batch_size,
        sampler=SubsetRandomSampler(val_indices_loader.tolist()), drop_last=True,
    )

    posterior.train(
        train_loader, val_loader,
        max_epochs=config["max_epochs"],
        stop_after_epochs=config["stop_after_epochs"],
        learning_rate=config["lr"],
    )

    full_emb = config.get("embedding", {"class": "EmbeddingNet", "dim_in": 100, "dim_out": 5})
    suffix = f"_{tag}" if tag else ""
    save_checkpoint(
        f"{config['run_dir']}/models/round_{round_id:05d}_{n_sims}{suffix}.pt",
        round_id=round_id,
        maf=maf,
        maf_build_kwargs=dict(
            dim_theta=dim_maf, dim_x=config["dim_x"],
            num_transforms=maf_config["num_transforms"],
            hidden_features=maf_config["hidden_features"],
            num_blocks=maf_config["num_blocks"],
        ),
        config=config,
        embedding_net_config=full_emb,
        prior_meta=config["prior_meta"],
    )

    return maf


def run_round(
    round_id: int,
    n_sims: int,
    simulator,
    proposal,
    zarr_store: ZarrStore,
    config: dict,
    x_o: torch.Tensor,
    original_prior,
    device: str = "cpu",
):
    offset = simulate_round(round_id, n_sims, simulator, proposal, zarr_store)
    return train_round(round_id, n_sims, offset, zarr_store, config, x_o,
                       original_prior, proposal, device)


def train(
    run_dir: str,
    round: int,
    n_sims: int,
    offset: int | None = None,
    device: str = "cpu",
    batch_size: int = 64,
    lr: float = 5e-4,
    max_epochs: int = 2_147_483_647,
    stop_after_epochs: int = 20,
    tag: str | None = None,
    interest_dims: list[int] | None = None,
    load_to_ram: bool = False,
):
    with open(f"{run_dir}/config.json") as f:
        config = json.load(f)

    if offset is None:
        with open(f"{run_dir}/sim_round_{round}.json") as f:
            offset = json.load(f)["offset"]
    print(f"Using simulations: offset={offset}, n_sims={n_sims}")

    config["batch_size"] = batch_size
    config["lr"] = lr
    config["max_epochs"] = max_epochs
    config["stop_after_epochs"] = stop_after_epochs
    if interest_dims is not None:
        config["interest_dims"] = interest_dims
        config["dim_interest"] = len(interest_dims)

    zarr_store = ZarrStore(f"{run_dir}/data")
    x_o = torch.load(f"{run_dir}/x_o.pt", weights_only=True).to(device)

    original_prior = build_prior_from_meta(config["prior_meta"])
    proposal = original_prior

    maf = train_round(
        round, n_sims, offset, zarr_store, config, x_o,
        original_prior, proposal, device,
        tag=tag, load_to_ram_flag=load_to_ram,
    )

    suffix = f"_{tag}" if tag else ""
    ckpt_name = f"round_{round:05d}_{n_sims}{suffix}.pt"
    with open(f"{run_dir}/last_checkpoint.json", "w") as f:
        json.dump({"checkpoint": ckpt_name}, f)

    hp_log = {key: config[key] for key in HP_KEYS}
    hp_log["n_sims"] = n_sims
    hp_log["offset"] = offset
    hp_log["tag"] = tag
    hp_log["interest_dims"] = interest_dims
    maf_config = config.get("maf", DEFAULT_MAF)
    hp_log["hidden_features"] = maf_config["hidden_features"]
    hp_log["num_transforms"] = maf_config["num_transforms"]
    hp_log["num_blocks"] = maf_config["num_blocks"]
    config_path = f"{run_dir}/models/{ckpt_name.replace('.pt', '_config.json')}"
    with open(config_path, "w") as f:
        json.dump(hp_log, f, indent=2)

    print(f"Round {round} trained. Checkpoint: models/{ckpt_name}")


def train_ensemble_kl(
    run_dir: str,
    round: int,
    n_sims: int,
    offset: int | None = None,
    n_members: int = 3,
    kl_every: int | None = 10,
    n_samples_kl: int = 5_000,
    device: str = "cpu",
    batch_size: int = 64,
    lr: float = 5e-4,
    max_epochs: int = 200,
    stop_after_epochs: int = 20,
    tag: str | None = None,
    interest_dims: list[int] | None = None,
    load_to_ram: bool = False,
) -> tuple[list, dict]:
    with open(f"{run_dir}/config.json") as f:
        config = json.load(f)

    if offset is None:
        with open(f"{run_dir}/sim_round_{round}.json") as f:
            offset = json.load(f)["offset"]
    print(f"Using simulations: offset={offset}, n_sims={n_sims}")

    config["batch_size"] = batch_size
    config["lr"] = lr
    config["max_epochs"] = max_epochs
    config["stop_after_epochs"] = stop_after_epochs
    if interest_dims is not None:
        config["interest_dims"] = interest_dims
        config["dim_interest"] = len(interest_dims)

    zarr_store = ZarrStore(f"{run_dir}/data")
    x_o = torch.load(f"{run_dir}/x_o.pt", weights_only=True).to(device)

    all_indices = list(range(offset, offset + n_sims))
    theta_mean, theta_std = compute_z_scores_streaming(zarr_store, all_indices, "theta")
    x_mean, x_std = compute_z_scores_streaming(zarr_store, all_indices, "x")

    dim_maf = config.get("dim_interest", config["dim_theta"])
    if interest_dims is not None:
        theta_mean = theta_mean[interest_dims]
        theta_std = theta_std[interest_dims]

    print(f"Building ensemble of {n_members} models with {n_sims} sims")
    if interest_dims is not None:
        print(f"  interest_dims={interest_dims}, dim_maf={dim_maf}")

    models = []
    optimizers = []
    for i in range(n_members):
        torch.manual_seed(i * 42)
        maf = _build_model(
            config, dim_maf, theta_mean, theta_std, x_mean, x_std, device,
        )
        models.append(maf)
        optimizers.append(torch.optim.Adam(maf.parameters(), lr=lr))

    if load_to_ram:
        print("Loading data to RAM...")
        perm = torch.randperm(n_sims) + offset
        indices = perm.tolist()
        base_dataset = load_to_ram(zarr_store, indices, interest_dims)
        loaders = []
        for i in range(n_members):
            train_loader, val_loader = _make_loaders(
                base_dataset, n_sims, batch_size, seed=i,
            )
            loaders.append((train_loader, val_loader))
    else:
        loaders = []
        for i in range(n_members):
            perm = torch.randperm(n_sims, generator=torch.Generator().manual_seed(i)) + offset
            indices = perm.tolist()
            theta_all = torch.from_numpy(zarr_store.data["theta"][indices]).float()
            x_all = torch.from_numpy(zarr_store.data["x"][indices]).float()
            if interest_dims is not None:
                theta_all = theta_all[:, interest_dims]
            dataset = TensorDataset(theta_all, x_all)
            train_loader, val_loader = _make_loaders(
                dataset, n_sims, batch_size, seed=i,
            )
            loaders.append((train_loader, val_loader))

    epochs_history = []
    max_kl_history = []

    best_val_losses = [float("inf")] * n_members
    epochs_no_improve = [0] * n_members
    best_states = [None] * n_members
    stopped = [False] * n_members

    for epoch in range(max_epochs):
        epoch_losses = []
        for i in range(n_members):
            if not stopped[i]:
                loss = _train_one_epoch(models[i], loaders[i][0], optimizers[i], device)
                epoch_losses.append(loss)
            else:
                epoch_losses.append(float("nan"))

        for i in range(n_members):
            vl = _val_loss(models[i], loaders[i][1], device)
            if vl < best_val_losses[i]:
                best_val_losses[i] = vl
                best_states[i] = {k: v.detach().cpu() for k, v in models[i].state_dict().items()}
                epochs_no_improve[i] = 0
            else:
                epochs_no_improve[i] += 1
            if epochs_no_improve[i] >= stop_after_epochs and not stopped[i]:
                print(f"  model {i+1} early stopping at epoch {epoch}")
                stopped[i] = True

        loss_str = "  ".join(f"m{i+1}={epoch_losses[i]:.4f}" for i in range(n_members))
        print(f"Epoch {epoch:3d}  {loss_str}")

        if kl_every is not None and epoch % kl_every == 0:
            mafs_for_kl = [m.eval() for m in models]
            K = kl_matrix(
                mafs_for_kl, x_o,
                n_samples=n_samples_kl,
                norm_posterior=False,
            )
            mk = _max_offdiag(K)
            epochs_history.append(epoch)
            max_kl_history.append(mk)
            print(f"  -> max KL = {mk:.4g}")

        if all(stopped):
            print(f"\nEarly stopping triggered for all {n_members} models. Exiting.")
            break

    os.makedirs(f"{run_dir}/models", exist_ok=True)
    hp_keys = ["batch_size", "lr", "max_epochs", "stop_after_epochs"]
    maf_config = config.get("maf", DEFAULT_MAF)
    for i in range(n_members):
        if best_states[i] is not None:
            models[i].load_state_dict(best_states[i])
        suffix = f"_{tag}{i+1}" if tag else f"_ens{i+1}"
        ckpt_name = f"round_00000_{n_sims}{suffix}.pt"
        _save_ensemble_checkpoint(
            f"{run_dir}/models/{ckpt_name}", models[i], config, n_sims, tag,
        )

        hp_log = {key: config[key] for key in hp_keys}
        hp_log["n_sims"] = n_sims
        hp_log["offset"] = offset
        hp_log["tag"] = f"{tag}{i+1}" if tag else f"ens{i+1}"
        hp_log["interest_dims"] = interest_dims
        hp_log["hidden_features"] = maf_config["hidden_features"]
        hp_log["num_transforms"] = maf_config["num_transforms"]
        hp_log["num_blocks"] = maf_config["num_blocks"]
        config_path = f"{run_dir}/models/{ckpt_name.replace('.pt', '_config.json')}"
        with open(config_path, "w") as f:
            json.dump(hp_log, f, indent=2)

        print(f"checkpoint saved: models/{ckpt_name}")

    history = {"epochs": epochs_history, "max_kl": max_kl_history}

    return models, history
