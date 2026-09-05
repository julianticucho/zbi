# test fixtures — used by test_* files

import torch
from torch import nn

from zbi.simulators.base import Simulator


class SimuladorLineal(Simulator):
    def __init__(self):
        super().__init__()
        self.x_grid = torch.linspace(-5, 5, 100)

    def simulate(self, theta: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        a, b = theta[0].item(), theta[1].item()
        y = a * self.x_grid + b + 0.1 * torch.randn(100)
        return y


class EmbeddingNet(nn.Module):
    def __init__(self, dim_in: int = 100, dim_out: int = 5):
        super().__init__()
        self.net = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
