import torch
from torch import nn
from zbi.neural_nets.factory import build_maf_estimator
from zbi.utils.preprocessing import Standardize


def test_build_with_tensors():
    theta = torch.randn(100, 3)
    x = torch.randn(100, 5)
    maf = build_maf_estimator(theta=theta, x=x)
    assert isinstance(maf.z_score_theta_transform, Standardize)
    assert isinstance(maf.z_score_x_transform, Standardize)
    out = maf(theta[:4], x[:4])
    assert out.shape == (4,)


def test_build_with_precomputed_stats():
    theta = torch.randn(100, 3)
    x = torch.randn(100, 5)
    t_mean = theta.mean(0)
    t_std = theta.std(0).clamp(min=1e-14)
    x_mean = x.mean(0)
    x_std = x.std(0).clamp(min=1e-14)
    maf = build_maf_estimator(
        theta_mean=t_mean, theta_std=t_std,
        x_mean=x_mean, x_std=x_std,
        dim_theta=3, dim_x=5,
    )
    assert torch.equal(maf.z_score_theta_transform.mean, t_mean)
    out = maf(theta[:4], x[:4])
    assert out.shape == (4,)


def test_build_with_z_score_none():
    theta = torch.randn(100, 2)
    x = torch.randn(100, 4)
    maf = build_maf_estimator(theta=theta, x=x, z_score_theta="none", z_score_x="none")
    assert isinstance(maf.z_score_theta_transform, nn.Identity)
    assert isinstance(maf.z_score_x_transform, nn.Identity)
    out = maf(theta[:4], x[:4])
    assert out.shape == (4,)


def test_build_with_embedding_net():
    theta = torch.randn(50, 3)
    x = torch.randn(50, 10)
    emb = nn.Sequential(nn.Linear(10, 4), nn.ReLU())
    maf = build_maf_estimator(theta=theta, x=x, embedding_net=emb)
    out = maf(theta[:2], x[:2])
    assert out.shape == (2,)


def test_build_structured_z_score():
    theta = torch.randn(100, 3)
    x = torch.randn(100, 5)
    maf = build_maf_estimator(theta=theta, x=x, z_score_theta="structured", z_score_x="structured")
    # structured uses single scalar mean/std
    assert maf.z_score_theta_transform.mean.ndim == 0
    assert maf.z_score_x_transform.mean.ndim == 0


def test_build_forward_backward():
    theta = torch.randn(32, 3)
    x = torch.randn(32, 5)
    maf = build_maf_estimator(theta=theta, x=x, num_transforms=2, hidden_features=32)
    loss = maf.loss(theta, x)
    loss.backward()
    assert loss.ndim == 0
    for p in maf.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break


def test_build_with_theta_and_x_only():
    # Should infer dims from tensors
    theta = torch.randn(50, 4)
    x = torch.randn(50, 6)
    maf = build_maf_estimator(theta=theta, x=x)
    assert maf.dim_theta == 4
    assert maf.dim_x == 6
