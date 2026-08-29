import torch
from zbi.utils.preprocessing import (
    Standardize,
    compute_z_scores,
    build_standardize_net,
    warn_if_invalid_for_zscoring,
)


def test_standardize_shapes():
    x = torch.randn(1000, 5)
    mean, std = compute_z_scores(x)
    assert mean.shape == (5,)
    assert std.shape == (5,)


def test_standardize_values():
    x = torch.randn(1000, 5)
    s = build_standardize_net(x)
    z = s(x)
    assert torch.allclose(z.mean(0), torch.zeros(5), atol=1e-2)
    assert torch.allclose(z.std(0), torch.ones(5), atol=1e-2)


def test_inverse():
    x = torch.randn(100, 10)
    s = build_standardize_net(x)
    z = s(x)
    xr = s.inverse(z)
    assert torch.allclose(xr, x, atol=1e-5)


def test_structured():
    x = torch.randn(100, 3, 32, 32)
    mean, std = compute_z_scores(x, structured=True)
    assert mean.ndim == 0
    assert std.ndim == 0


def test_std_clamp():
    x_const = torch.ones(100, 5)
    _, std = compute_z_scores(x_const)
    assert (std == 1e-14).all()


def test_state_dict():
    x = torch.randn(100, 5)
    s = build_standardize_net(x)
    sd = s.state_dict()
    assert "_mean" in sd
    assert "_std" in sd
    s2 = Standardize(torch.zeros(5), torch.ones(5))
    s2.load_state_dict(sd)
    assert torch.allclose(s2.mean, s.mean)
    assert torch.allclose(s2.std, s.std)


def test_warn_no_crash():
    x = torch.randn(100, 5)
    warn_if_invalid_for_zscoring(x, x)
