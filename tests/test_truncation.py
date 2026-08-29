import math
import torch
from zbi.utils.truncation import compute_bounding_box, TruncatedBoxPrior


def test_compute_bounding_box_basic():
    samples = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0]])
    log_probs = torch.tensor([-1.0, -0.5, -0.1, -100.0])
    orig = torch.tensor([[-5.0, -5.0], [15.0, 15.0]])
    bounds = compute_bounding_box(samples, log_probs, 1e-6, orig)
    # Points with log_prob > -0.1 + log(1e-6) = -0.1 - 13.8 = -13.9
    # Only first 3 points qualify
    expected = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    assert torch.allclose(bounds, expected), f"Got {bounds}"


def test_compute_bounding_box_empty_mask():
    samples = torch.randn(10, 2)
    log_probs = torch.full((10,), -1.0)
    orig = torch.tensor([[-5.0, -5.0], [5.0, 5.0]])
    bounds = compute_bounding_box(samples, log_probs, 1.0, orig)
    # threshold=1 → mask = log_probs > max + 0 → none
    assert torch.allclose(bounds, orig)


def test_compute_bounding_box_all_points():
    samples = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    log_probs = torch.tensor([-1.0, -2.0])
    orig = torch.tensor([[-5.0, -5.0], [5.0, 5.0]])
    bounds = compute_bounding_box(samples, log_probs, 0.0, orig)
    # threshold=0 → log(0)=-inf → mask = log_probs > -inf → all
    expected = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    assert torch.allclose(bounds, expected)


def test_compute_bounding_box_clamped():
    samples = torch.tensor([[-10.0, 0.0], [1.0, 20.0]])
    log_probs = torch.tensor([-0.1, -0.1])
    orig = torch.tensor([[-5.0, -5.0], [5.0, 5.0]])
    bounds = compute_bounding_box(samples, log_probs, 1e-6, orig)
    # Both points qualify, but clamped to original
    expected = torch.tensor([[-5.0, 0.0], [1.0, 5.0]])
    assert torch.allclose(bounds, expected), f"Got {bounds}"


def test_truncated_box_prior_sample():
    prior = torch.distributions.Uniform(-3.0, 3.0)
    bounds = torch.tensor([[-1.0, -2.0], [1.0, 2.0]])
    tbp = TruncatedBoxPrior(prior, bounds)
    samples = tbp.sample((1000,))
    assert samples.shape == (1000, 2)
    assert (samples >= bounds[0]).all()
    assert (samples <= bounds[1]).all()


def test_truncated_box_prior_log_prob():
    prior = torch.distributions.Uniform(-3.0, 3.0)
    bounds = torch.tensor([[-1.0], [1.0]])
    tbp = TruncatedBoxPrior(prior, bounds)
    theta_in = torch.tensor([[0.0]])
    theta_out = torch.tensor([[2.0]])
    lp_in = tbp.log_prob(theta_in)
    lp_out = tbp.log_prob(theta_out)
    volume = 2.0
    assert torch.allclose(lp_in, torch.tensor([math.log(1.0 / volume)]))
    assert lp_out == float("-inf")


def test_truncated_box_prior_deterministic():
    prior = torch.distributions.Uniform(-3.0, 3.0)
    bounds = torch.tensor([[-1.0, -2.0], [1.0, 2.0]])
    tbp = TruncatedBoxPrior(prior, bounds)
    torch.manual_seed(0)
    s1 = tbp.sample((5,))
    torch.manual_seed(0)
    s2 = tbp.sample((5,))
    assert torch.allclose(s1, s2)
