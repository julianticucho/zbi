from typing import Optional
import torch
from torch import nn
import yaml

from zbi.utils.truncation import TruncatedBoxPrior


CHECKPOINT_KEYS = [
    "round_id",
    "maf_state_dict",
    "maf_build_kwargs",
    "embedding_net_config",
    "prior_bounds",
    "prior_meta",
    "prior_type",
    "config",
    "optimizer_state_dict",
]


def save_checkpoint(
    path: str,
    round_id: int,
    maf: nn.Module,
    maf_build_kwargs: dict,
    config: dict,
    embedding_net_config: Optional[dict] = None,
    prior_meta: Optional[dict] = None,
    prior_bounds: Optional[torch.Tensor] = None,
    prior_type: Optional[str] = None,
    optimizer_state_dict: Optional[dict] = None,
) -> None:
    torch.save(
        {
            "round_id": round_id,
            "maf_state_dict": maf.state_dict(),
            "maf_build_kwargs": maf_build_kwargs,
            "embedding_net_config": embedding_net_config,
            "prior_bounds": prior_bounds,
            "prior_meta": prior_meta,
            "prior_type": prior_type,
            "config": config,
            "optimizer_state_dict": optimizer_state_dict,
        },
        path,
    )


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def save_config_yaml(path: str, config: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_maf_from_checkpoint(
    ckpt: dict, embedding_net: nn.Module
) -> nn.Module:
    from zbi.neural_nets.factory import build_maf_estimator

    build_kwargs = ckpt["maf_build_kwargs"]
    dim_theta = build_kwargs.get("dim_theta", 2)
    dim_x = build_kwargs.get("dim_x", 100)

    theta_mean = torch.zeros(dim_theta)
    theta_std = torch.ones(dim_theta)
    x_mean = torch.zeros(dim_x)
    x_std = torch.ones(dim_x)

    maf = build_maf_estimator(
        dim_theta=dim_theta, dim_x=dim_x,
        embedding_net=embedding_net,
        z_score_theta="independent", z_score_x="independent",
        theta_mean=theta_mean, theta_std=theta_std,
        x_mean=x_mean, x_std=x_std,
        **{k: v for k, v in build_kwargs.items()
           if k not in ("dim_theta", "dim_x")},
    )

    state_dict = {}
    for k, v in ckpt["maf_state_dict"].items():
        if k.startswith("net."):
            new_k = "_flow." + k[4:]
        else:
            new_k = k
        state_dict[new_k] = v
    maf.load_state_dict(state_dict)
    maf.eval()
    return maf


def load_prior_from_checkpoint(
    ckpt: dict, prior_class=None, **prior_kwargs
):
    import torch.distributions as dist

    if ckpt.get("prior_meta") is not None:
        pm = ckpt["prior_meta"]
        prior_class = getattr(dist, pm["type"])
        prior_kwargs = {k: torch.tensor(v) for k, v in pm.items() if k != "type"}

    original_prior = prior_class(**prior_kwargs)

    if ckpt.get("prior_type") == "truncated" and ckpt.get("prior_bounds") is not None:
        return TruncatedBoxPrior(original_prior, ckpt["prior_bounds"])
    return original_prior
