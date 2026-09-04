import json
import os

import torch
from torch.distributions import Uniform
from torch.utils.data import DataLoader, SubsetRandomSampler

from zbi.data.zarr_store import ZarrStore
from zbi.inference.posterior import Posterior
from zbi.neural_nets.factory import build_maf_estimator
from zbi.pipeline._data import SlicedDataset, load_to_ram
from zbi.pipeline._resolve import resolve_prior, resolve_class
from zbi.pipeline.setup import DEFAULT_MAF
from zbi.utils.checkpoint import save_checkpoint
from zbi.utils.preprocessing import compute_z_scores_streaming


HP_KEYS = [
    "batch_size", "lr", "max_epochs", "stop_after_epochs",
]


def _train_round(
    round_id: int,
    n_sims: int,
    offset: int,
    zarr_store: ZarrStore,
    config: dict,
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

    original_prior = resolve_prior(config["prior_meta"])
    proposal = original_prior

    maf = _train_round(
        round, n_sims, offset, zarr_store, config,
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
