from typing import Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.distributions.constraints import Constraint


def _within_support(prior, theta: torch.Tensor) -> torch.Tensor:
    try:
        support = prior.support
    except NotImplementedError:
        support = None
    if isinstance(support, Constraint):
        return support.check(theta).all(dim=-1)
    log_prob = getattr(prior, "log_prob", None)
    if log_prob is not None:
        return torch.isfinite(log_prob(theta))
    raise TypeError("Prior must implement support or log_prob.")


class Posterior:
    def __init__(
        self,
        estimator: nn.Module,
        proposal,
        device: str = "cpu",
        enable_leakage_correction: bool = True,
    ) -> None:
        self.estimator = estimator.to(device)
        self.proposal = proposal
        self.device = device
        self.enable_leakage_correction = enable_leakage_correction
        self._leakage_factor: Optional[float] = None
        self._default_x: Optional[torch.Tensor] = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_epochs: int = 2_147_483_647,
        stop_after_epochs: int = 20,
        learning_rate: float = 5e-4,
        clip_grad_norm: float = 5.0,
    ):
        optimizer = optim.Adam(self.estimator.parameters(), lr=learning_rate)
        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            self.estimator.train()
            train_loss = 0.0
            for theta_batch, x_batch in train_loader:
                theta_batch = theta_batch.to(self.device)
                x_batch = x_batch.to(self.device)
                optimizer.zero_grad()
                loss = self.estimator.loss(theta_batch, x_batch)
                loss.backward()
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.estimator.parameters(), clip_grad_norm)
                optimizer.step()
                train_loss += loss.item() * theta_batch.shape[0]
            train_loss /= len(train_loader) * train_loader.batch_size

            if val_loader is not None:
                self.estimator.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for theta_batch, x_batch in val_loader:
                        theta_batch = theta_batch.to(self.device)
                        x_batch = x_batch.to(self.device)
                        loss = self.estimator.loss(theta_batch, x_batch)
                        val_loss += loss.item() * theta_batch.shape[0]
                val_loss /= len(val_loader) * val_loader.batch_size

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu() for k, v in self.estimator.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
                if epochs_no_improve >= stop_after_epochs:
                    print(f"Early stopping at epoch {epoch}")
                    if best_state is not None:
                        self.estimator.load_state_dict(best_state)
                    break
            else:
                print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}")

        if val_loader is not None and best_state is not None:
            self.estimator.load_state_dict(best_state)
        self.estimator.eval()

    def sample(
        self,
        sample_shape: tuple,
        x_o: torch.Tensor,
        reject_outside_prior: bool = True,
        max_sampling_batch_size: int = 10_000,
        sample_batch_size: int = 64,
    ) -> torch.Tensor:
        x_o = x_o.to(self.device)
        total = 1
        for s in sample_shape:
            total *= s
        all_samples = []
        num_collected = 0
        while num_collected < total:
            n_remaining = total - num_collected
            outer_batch = min(max_sampling_batch_size, n_remaining)
            inner_parts = []
            for j in range(0, outer_batch, sample_batch_size):
                n = min(sample_batch_size, outer_batch - j)
                part = self.estimator.sample(n, x_o)
                inner_parts.append(part)
            batch = torch.cat(inner_parts, dim=0)
            if reject_outside_prior:
                in_prior = _within_support(self.proposal, batch)
                batch = batch[in_prior]
            all_samples.append(batch)
            num_collected += batch.shape[0]
            if len(all_samples) > 1:
                all_samples = [torch.cat(all_samples, dim=0)]
            if all_samples[0].shape[0] >= total:
                break
        return torch.cat(all_samples, dim=0)[:total].reshape(sample_shape + (-1,))

    def log_prob(
        self,
        theta: torch.Tensor,
        x_o: torch.Tensor,
        norm_posterior: bool = True,
    ) -> torch.Tensor:
        x_o = x_o.to(self.device)
        theta = theta.to(self.device)
        with torch.no_grad():
            lp = self.estimator.forward(theta, x_o.expand(theta.shape[0], -1))
            if norm_posterior and self.enable_leakage_correction:
                lc = self.leakage_correction(x_o)
                if lc > 0:
                    lp = lp - torch.log(torch.tensor(lc, device=lp.device))
        return lp

    def leakage_correction(
        self,
        x_o: torch.Tensor,
        num_rejection_samples: int = 10_000,
        force_update: bool = False,
        batch_size: int = 256,
    ) -> float:
        x_o = x_o.to(self.device)
        if (
            not force_update
            and self._default_x is not None
            and self._leakage_factor is not None
            and torch.equal(x_o, self._default_x)
        ):
            return self._leakage_factor
        with torch.no_grad():
            all_samples = []
            for i in range(0, num_rejection_samples, batch_size):
                n = min(batch_size, num_rejection_samples - i)
                batch = self.estimator.sample(n, x_o)
                all_samples.append(batch)
            samples = torch.cat(all_samples, dim=0)
        in_prior = _within_support(self.proposal, samples)
        factor = in_prior.float().mean().item()
        self._default_x = x_o
        self._leakage_factor = factor
        return factor

    def map(
        self,
        x_o: torch.Tensor,
        num_init_samples: int = 1_000,
        num_to_optimize: int = 100,
        num_iter: int = 1_000,
        learning_rate: float = 0.01,
    ) -> torch.Tensor:
        x_o = x_o.to(self.device)
        with torch.no_grad():
            init_samples = self.sample(
                (num_init_samples,), x_o, reject_outside_prior=True
            ).detach()
            log_probs = self.estimator.forward(init_samples, x_o.expand(init_samples.shape[0], -1))
        best_idx = log_probs.topk(min(num_to_optimize, num_init_samples)).indices
        theta = init_samples[best_idx].detach().requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=learning_rate)
        for _ in range(num_iter):
            optimizer.zero_grad()
            lp = self.estimator.forward(theta, x_o.expand(theta.shape[0], -1))
            loss = -lp.mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            lp = self.estimator.forward(theta, x_o.expand(theta.shape[0], -1))
        return theta[lp.argmax().item()].detach()
