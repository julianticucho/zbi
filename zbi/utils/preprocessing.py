from typing import Tuple
import torch
from torch import nn


class Standardize(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_mean", mean.clone())
        self.register_buffer("_std", std.clone().clamp(min=1e-14))

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        return self._std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self._mean) / self._std

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self._std + self._mean


def compute_z_scores(
    batch: torch.Tensor, 
    structured: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if structured:
        mean = batch.mean()
        std = batch.std()
    else:
        mean = batch.mean(dim=0)
        std = batch.std(dim=0)
    std = std.clamp(min=1e-14)
    return mean, std


def build_standardize_net(
    batch: torch.Tensor, structured: bool = False
) -> Standardize:
    mean, std = compute_z_scores(batch, structured)
    return Standardize(mean, std)


def compute_z_scores_streaming(
    store,
    indices,
    key: str,
    batch_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import numpy as np

    idx = torch.as_tensor(indices)
    n = len(idx)
    if n == 0:
        raise ValueError("At least 1 index is required to compute z-scoring")

    first_batch = idx[: min(batch_size, n)]
    arr = store.data[key][first_batch.tolist()]
    dim = arr.shape[-1]

    sum_x = torch.zeros(dim, dtype=torch.float64)
    sum_x2 = torch.zeros(dim, dtype=torch.float64)
    count = 0

    for i in range(0, n, batch_size):
        batch_idx = idx[i : i + batch_size]
        arr = store.data[key][batch_idx.tolist()]
        tensor = torch.from_numpy(arr).float()
        sum_x += tensor.sum(dim=0).double()
        sum_x2 += (tensor ** 2).sum(dim=0).double()
        count += tensor.shape[0]

    mean = (sum_x / count).float()
    std = ((sum_x2 / count - mean.double() ** 2).clamp(min=0).sqrt()).float()
    std = std.clamp(min=1e-14)
    return mean, std


def warn_if_invalid_for_zscoring(
    theta: torch.Tensor, x: torch.Tensor
) -> None:
    for name, tensor in [("theta", theta), ("x", x)]:
        if torch.isnan(tensor).any():
            print(f"WARNING: {name} contains NaN.")
        if torch.isinf(tensor).any():
            print(f"WARNING: {name} contains Inf.")
        if (tensor.std(dim=0) == 0).any():
            print(f"WARNING: {name} has zero variance.")
