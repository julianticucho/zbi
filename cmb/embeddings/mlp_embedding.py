import torch
from torch import nn


class PlanckLiteMLPEmbedding(nn.Module):
    def __init__(self, dim_out: int = 32):
        super().__init__()
        self.dim_out = dim_out
        self.net = nn.Sequential(
            nn.Linear(613, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlanckLiteMLPEmbeddingV2(nn.Module):
    def __init__(self, dim_out: int = 10):
        super().__init__()
        self.dim_out = dim_out
        self.net = nn.Sequential(
            nn.Linear(613, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
