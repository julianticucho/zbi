import torch
from torch import nn
from zbi.neural_nets.maf import MAF
from zbi.utils.preprocessing import build_standardize_net


def test_maf_forward():
    maf = MAF(dim_theta=2, dim_x=5)
    theta = torch.randn(32, 2)
    x = torch.randn(32, 5)
    lp = maf.forward(theta, x)
    assert lp.shape == (32,), f"shape: {lp.shape}"
    assert lp.isfinite().all(), "non-finite log_prob"


def test_maf_sample():
    maf = MAF(dim_theta=2, dim_x=5)
    x = torch.randn(1, 5)
    samples = maf.sample(100, x)
    assert samples.shape == (100, 2), f"shape: {samples.shape}"
    assert samples.isfinite().all()


def test_maf_loss():
    maf = MAF(dim_theta=2, dim_x=5)
    theta = torch.randn(32, 2)
    x = torch.randn(32, 5)
    loss = maf.loss(theta, x)
    assert loss.ndim == 0, f"ndim: {loss.ndim}"
    assert loss.isfinite(), "non-finite loss"
    assert loss > 0, "loss should be positive for random data"


def test_maf_gradient_flow():
    maf = MAF(dim_theta=2, dim_x=5)
    theta = torch.randn(32, 2)
    x = torch.randn(32, 5)
    loss = maf.loss(theta, x)
    loss.backward()
    has_grad = False
    for p in maf.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "no gradients flowing"


def test_maf_with_embedding():
    class EmbeddingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )
        def forward(self, x):
            return self.net(x)

    maf = MAF(dim_theta=2, dim_x=10, embedding_net=EmbeddingNet())
    theta = torch.randn(32, 2)
    x = torch.randn(32, 10)
    lp = maf.forward(theta, x)
    assert lp.shape == (32,)
    assert lp.isfinite().all()


def test_maf_with_z_scoring():
    theta = torch.randn(1000, 2)
    x = torch.randn(1000, 5)
    z_theta = build_standardize_net(theta)
    z_x = build_standardize_net(x)
    maf = MAF(
        dim_theta=2, dim_x=5,
        z_score_theta=z_theta,
        z_score_x=z_x,
    )
    lp = maf.forward(theta, x)
    assert lp.shape == (1000,)
    assert lp.isfinite().all()


def test_maf_sample_with_z_scoring():
    theta_batch = torch.randn(1000, 2)
    x_batch = torch.randn(1000, 5)
    z_theta = build_standardize_net(theta_batch)
    z_x = build_standardize_net(x_batch)
    maf = MAF(
        dim_theta=2, dim_x=5,
        z_score_theta=z_theta,
        z_score_x=z_x,
    )
    # Un solo x para condicionar
    x_single = torch.randn(1, 5)
    samples = maf.sample(100, x_single)
    assert samples.shape == (100, 2), f"shape: {samples.shape}"
    assert samples.isfinite().all(), "non-finite samples"


def test_maf_loss_decreases():
    maf = MAF(dim_theta=2, dim_x=5)
    opt = torch.optim.Adam(maf.parameters(), lr=1e-2)
    theta = torch.randn(128, 2)
    x = torch.randn(128, 5)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = maf.loss(theta, x)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
