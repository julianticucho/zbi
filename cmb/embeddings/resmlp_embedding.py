import torch
from torch import nn


class PlanckLiteResMLPEmbedding(nn.Module):
    def __init__(self, dim_out: int = 10):
        super().__init__()
        self.dim_out = dim_out

        self.in_proj = nn.Linear(613, 256)

        self.block1 = nn.Sequential(
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
        )

        self.out_proj = nn.Linear(256, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out_proj(h)
