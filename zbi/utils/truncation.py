import math
import torch
from torch.distributions import Distribution


def compute_bounding_box(
    samples: torch.Tensor,
    log_probs: torch.Tensor,
    threshold: float,
    original_bounds: torch.Tensor,
) -> torch.Tensor:
    if threshold <= 0:
        selected = samples
    elif threshold >= 1:
        return original_bounds
    else:
        mask = log_probs > log_probs.max() + math.log(threshold)
        selected = samples[mask]
    if len(selected) == 0:
        return original_bounds
    
    bounds = torch.stack([selected.min(dim=0)[0], selected.max(dim=0)[0]])
    return torch.clamp(bounds, original_bounds[0], original_bounds[1])


class TruncatedBoxPrior(Distribution):
    arg_constraints: dict = {}
    def __init__(
        self, prior: Distribution, box_bounds: torch.Tensor
    ) -> None:
        super().__init__()
        self.prior = prior
        self.box_bounds = box_bounds
        self._u_low = prior.cdf(box_bounds[0])
        self._u_high = prior.cdf(box_bounds[1])

    @property
    def batch_shape(self) -> torch.Size:
        return self.box_bounds.shape[1:]
    
    @property
    def event_shape(self) -> torch.Size:
        return self.box_bounds.shape[1:]

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        u = torch.rand(
            sample_shape + self._u_low.shape,
            device=self._u_low.device,
            dtype=self._u_low.dtype,
        )
        u = u * (self._u_high - self._u_low) + self._u_low
        return self.prior.icdf(u)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        low, high = self.box_bounds[0], self.box_bounds[1]
        in_box = (theta >= low).all(dim=-1) & (theta <= high).all(dim=-1)
        volume = (high - low).prod().clamp(min=1e-14)
        batch = theta.shape[0]
        result = torch.full((batch,), float("-inf"), device=theta.device, dtype=theta.dtype)
        result[in_box] = -volume.log()
        return result

    @property
    def bounds(self) -> torch.Tensor:
        return self.box_bounds
