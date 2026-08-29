import torch
from torch import nn


class PlanckLiteCNN1DEmbedding(nn.Module):
    """CNN 1D para espectros C_l concatenados (TT+TE+EE, 613 dims).

    Trata los C_l como secuencia 1D y aplica convoluciones para
    capturar estructura local (picos acústicos, damping tail).

    Arquitectura:
        (batch, 613) -> reshape a (batch, 1, 613)
        -> Conv1d(1→32, k=7, s=2) -> BN -> ReLU   (304)
        -> Conv1d(32→64, k=5, s=2) -> BN -> ReLU   (151)
        -> Conv1d(64→128, k=3, s=2) -> BN -> ReLU  (75)
        -> AdaptiveAvgPool1d(1) -> Flatten -> Linear(128, dim_out)

    Args:
        dim_out: Dimension del embedding de salida.
    """

    def __init__(self, dim_out: int = 12):
        super().__init__()
        self.dim_out = dim_out

        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.net(x)
