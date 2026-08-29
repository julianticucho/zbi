from typing import Optional
import torch
from torch import nn
from nflows.flows import Flow
from nflows.transforms import (
    MaskedAffineAutoregressiveTransform,
    RandomPermutation,
    CompositeTransform,
    PointwiseAffineTransform,
)
from nflows.distributions import StandardNormal


class MAF(nn.Module):
    def __init__(
        self,
        dim_theta: int,
        dim_x: int,
        num_transforms: int = 5,
        hidden_features: int = 50,
        num_blocks: int = 2,
        embedding_net: nn.Module = nn.Identity(),
        z_score_theta: Optional[nn.Module] = None,
        z_score_x: Optional[nn.Module] = None,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.dim_theta = dim_theta
        self.dim_x = dim_x
        self.z_score_theta_transform = z_score_theta if z_score_theta is not None else nn.Identity()
        self.z_score_x_transform = z_score_x if z_score_x is not None else nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, dim_x)
            if z_score_x is not None:
                dummy = z_score_x(dummy)
            context_dim = embedding_net(dummy).shape[-1]

        transform_list = []
        if z_score_theta is not None:
            mean = z_score_theta.mean.to(dtype=torch.float32)
            std = z_score_theta.std.to(dtype=torch.float32).clamp(min=1e-14)
            transform_list.append(
                PointwiseAffineTransform(
                    shift=(-mean / std).detach(),
                    scale=(1.0 / std).detach()
                )
            )

        for _ in range(num_transforms):
            transform_list.append(
                MaskedAffineAutoregressiveTransform(
                    features=dim_theta,
                    hidden_features=hidden_features,
                    context_features=context_dim,
                    num_blocks=num_blocks,
                    use_residual_blocks=False,
                    random_mask=False,
                    activation=nn.functional.tanh,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            transform_list.append(RandomPermutation(features=dim_theta))
        flow_transform = CompositeTransform(transform_list)
        
        if z_score_x is not None:
            embedding_net = nn.Sequential(z_score_x, embedding_net)

        self._flow = Flow(
            transform=flow_transform,
            distribution=StandardNormal((dim_theta,)),
            embedding_net=embedding_net,
        )

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._flow.log_prob(theta, context=x)

    def sample(
        self,
        num_samples: int,
        x: torch.Tensor,
        sample_batch_size: int = 256
    ) -> torch.Tensor:
        if sample_batch_size is None or sample_batch_size >= num_samples:
            samples = self._flow.sample(num_samples, context=x)
        else:
            parts = []
            for i in range(0, num_samples, sample_batch_size):
                n = min(sample_batch_size, num_samples - i)
                part = self._flow.sample(n, context=x)
                if part.ndim == 3 and part.shape[0] == 1:
                    part = part.squeeze(0)
                parts.append(part)
            samples = torch.cat(parts, dim=0)
        if samples.ndim == 3 and samples.shape[0] == 1:
            samples = samples.squeeze(0)
        return samples

    def loss(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -self.forward(theta, x).mean()
