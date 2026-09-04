import numpy as np
import torch
from typing import Any
from zbi.inference.posterior import Posterior


def _sample_posterior(
    est: Any,
    n: int,
    x_o: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if isinstance(est, Posterior):
        return est.sample(
            (n,), x_o, reject_outside_prior=False, max_sampling_batch_size=batch_size
        )
    parts = []
    remaining = n
    while remaining > 0:
        k = min(batch_size, remaining)
        parts.append(est.sample(k, x_o))
        remaining -= k
    return torch.cat(parts, dim=0)[:n]


def _posterior_log_prob(
    est: Any,
    theta: torch.Tensor,
    x_o: torch.Tensor,
    norm_posterior: bool,
    batch_size: int,
) -> torch.Tensor:
    device = x_o.device
    parts = []
    for k in range(0, theta.shape[0], batch_size):
        batch = theta[k : k + batch_size].to(device)
        with torch.no_grad():
            if isinstance(est, Posterior):
                lp = est.log_prob(batch, x_o, norm_posterior=norm_posterior)
            else:
                lp = est.forward(batch, x_o.expand(batch.shape[0], -1))
        parts.append(lp.to(device))
    return torch.cat(parts, dim=0)


def kl_matrix(
    estimators: list[Any],
    x_o: torch.Tensor,
    n_samples: int = 10_000,
    norm_posterior: bool = True,
    sample_batch_size: int = 2_000,
    log_prob_batch_size: int = 2_000,
) -> np.ndarray:
    if not estimators:
        raise ValueError("estimators cannot be empty")
    if isinstance(x_o, torch.Tensor):
        x_o = x_o.detach().float()
    else:
        x_o = torch.as_tensor(np.asarray(x_o), dtype=torch.float32)
    if x_o.ndim == 1:
        x_o = x_o.unsqueeze(0)

    m = len(estimators)
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            theta = _sample_posterior(estimators[i], n_samples, x_o, sample_batch_size)
            lpi = _posterior_log_prob(estimators[i], theta, x_o, norm_posterior, log_prob_batch_size)
            lpj = _posterior_log_prob(estimators[j], theta, x_o, norm_posterior, log_prob_batch_size)
            diff = (lpi - lpj).detach().float()
            mask = torch.isfinite(diff)
            if not bool(mask.all()):
                diff = diff[mask]
            K[i, j] = diff.mean().item() if diff.numel() > 0 else np.nan
    return K
