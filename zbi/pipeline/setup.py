import json, os
import torch
from zbi.data.zarr_store import ZarrStore


DEFAULT_MAF = {
    "hidden_features": 32,
    "num_transforms": 5,
    "num_blocks": 2,
    "dropout_probability": 0.0,
    "use_batch_norm": False,
}


def init(
    run_dir: str,
    x_o: torch.Tensor,
    simulator_class,
    embedding_class,
    prior_low,
    prior_high,
    dim_theta,
    dim_x,
    zarr_N=2000,
    zarr_chunk_size=128,
    embedding_kwargs: dict | None = None,
    simulator_kwargs: dict | None = None,
    maf_kwargs: dict | None = None,
):
    config_path = f"{run_dir}/config.json"
    if os.path.exists(config_path):
        return

    os.makedirs(f"{run_dir}/models", exist_ok=True)
    config = dict(
        run_dir=run_dir,
        dim_theta=dim_theta, dim_x=dim_x,
        zarr_N=zarr_N, zarr_chunk_size=zarr_chunk_size,
        simulator={"class": f"{simulator_class.__module__}.{simulator_class.__qualname__}", **(simulator_kwargs or {})},
        embedding={"class": f"{embedding_class.__module__}.{embedding_class.__qualname__}", **(embedding_kwargs or {})},
        prior_meta={
            "type": "Uniform",
            "low": list(prior_low),
            "high": list(prior_high),
        },
        maf={**DEFAULT_MAF, **(maf_kwargs or {})},
    )
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    ZarrStore(f"{run_dir}/data").init(
        N=zarr_N, chunk_size=zarr_chunk_size,
        dim_theta=dim_theta, dim_x=dim_x,
    )
    torch.save(x_o, f"{run_dir}/x_o.pt")


def update_store(run_dir: str, new_N: int):
    config_path = f"{run_dir}/config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No config.json found in {run_dir}. Run init() first."
        )
    with open(config_path) as f:
        config = json.load(f)
    if new_N <= config["zarr_N"]:
        raise ValueError(
            f"new_N ({new_N}) must be greater than current zarr_N ({config['zarr_N']})."
        )
    ZarrStore(f"{run_dir}/data").resize(new_N)
    config["zarr_N"] = new_N
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Store expanded: {config['zarr_N']} -> {new_N}")


def update_embedding(run_dir, embedding_class, **embedding_kwargs):
    config_path = f"{run_dir}/config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No config.json found in {run_dir}. Run init() first."
        )
    with open(config_path) as f:
        config = json.load(f)
    config["embedding"] = {
        "class": f"{embedding_class.__module__}.{embedding_class.__qualname__}",
        **embedding_kwargs,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Embedding updated: {config['embedding']['class']}")


def update_maf(run_dir, **maf_kwargs):
    config_path = f"{run_dir}/config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No config.json found in {run_dir}. Run init() first."
        )
    with open(config_path) as f:
        config = json.load(f)
    if "maf" not in config:
        config["maf"] = dict(DEFAULT_MAF)
    config["maf"].update(maf_kwargs)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"MAF config updated: {config['maf']}")
