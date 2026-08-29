import torch
from torch import nn


class BmodeCNNEmbedding(nn.Module):
    """CNN embedding para mapas B-mode de 256x256.

    Arquitectura:
        - 3 bloques Conv2D + BatchNorm + ReLU + MaxPool
        - Global Avg Pooling
        - Capa linear a dim_out

    Args:
        dim_out: Dimension del embedding de salida.
    """

    def __init__(self, dim_out: int = 16):
        super().__init__()
        self.dim_out = dim_out

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # 64
        self.fc = nn.Linear(64, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor (batch, 65536) o (batch, 1, 256, 256).

        Returns:
            Tensor (batch, dim_out).
        """
        if x.ndim == 2:
            x = x.view(-1, 1, 256, 256)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.gap(h).squeeze(-1).squeeze(-1)
        return self.fc(h)


class BmodeCNNEmbedding128(nn.Module):
    """CNN embedding para mapas B-mode de 128x128.

    Misma arquitectura que BmodeCNNEmbedding pero para entrada de 128x128.

    Args:
        dim_out: Dimension del embedding de salida.
    """

    def __init__(self, dim_out: int = 16):
        super().__init__()
        self.dim_out = dim_out

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # 64
        self.fc = nn.Linear(64, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor (batch, 16384) o (batch, 1, 128, 128).

        Returns:
            Tensor (batch, dim_out).
        """
        if x.ndim == 2:
            x = x.view(-1, 1, 128, 128)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.gap(h).squeeze(-1).squeeze(-1)
        return self.fc(h)
