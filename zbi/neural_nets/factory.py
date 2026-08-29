from typing import Optional
import torch
from torch import nn

from zbi.neural_nets.maf import MAF
from zbi.utils.preprocessing import build_standardize_net, Standardize


def build_maf_estimator(
    theta: Optional[torch.Tensor] = None,
    x: Optional[torch.Tensor] = None,
    dim_theta: Optional[int] = None,
    dim_x: Optional[int] = None,
    embedding_net: nn.Module = nn.Identity(),
    z_score_theta: str = "independent",
    z_score_x: str = "independent",
    theta_mean: Optional[torch.Tensor] = None,
    theta_std: Optional[torch.Tensor] = None,
    x_mean: Optional[torch.Tensor] = None,
    x_std: Optional[torch.Tensor] = None,
    **maf_kwargs,
) -> MAF:
    if dim_theta is None:
        assert theta is not None, "theta or dim_theta is required"
        dim_theta = theta.shape[-1]
    if dim_x is None:
        assert x is not None, "x or dim_x is required"
        dim_x = x.shape[-1]

    if theta_mean is not None and theta_std is not None:
        z_theta = Standardize(theta_mean, theta_std)
    elif z_score_theta != "none":
        assert theta is not None, "theta is required to compute z-scoring"
        z_theta = build_standardize_net(
            theta, structured=(z_score_theta == "structured")
        )
    else:
        z_theta = None

    if x_mean is not None and x_std is not None:
        z_x = Standardize(x_mean, x_std)
    elif z_score_x != "none":
        assert x is not None, "x is required to compute z-scoring"
        z_x = build_standardize_net(
            x, structured=(z_score_x == "structured")
        )
    else:
        z_x = None

    return MAF(
        dim_theta=dim_theta,
        dim_x=dim_x,
        embedding_net=embedding_net,
        z_score_theta=z_theta,
        z_score_x=z_x,
        **maf_kwargs,
    )
