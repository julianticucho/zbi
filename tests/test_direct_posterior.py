import torch
from torch import nn
from torch.distributions import Uniform

from zbi.inference.posterior import Posterior
from zbi.neural_nets.factory import build_maf_estimator
from zbi.utils.truncation import TruncatedBoxPrior


def _make_prior_and_maf(device="cpu"):
    prior = Uniform(torch.zeros(3), torch.ones(3))
    theta = prior.sample((200,))
    x = torch.randn(200, 5)
    maf = build_maf_estimator(theta=theta, x=x, num_transforms=2, hidden_features=32)
    maf = maf.to(device)
    return prior, maf


def test_sample_basic():
    prior, maf = _make_prior_and_maf()
    post = Posterior(maf, prior, device="cpu")
    x_o = torch.randn(1, 5)
    samples = post.sample((50,), x_o, reject_outside_prior=False)
    assert samples.shape == (50, 3)


def test_sample_reject_outside_prior():
    prior, maf = _make_prior_and_maf()
    box = TruncatedBoxPrior(prior, torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    post = Posterior(maf, box, device="cpu")
    x_o = torch.randn(1, 5)
    samples = post.sample((100,), x_o, reject_outside_prior=True)
    assert samples.shape == (100, 3)
    assert (samples >= 0).all() and (samples <= 1).all()


def test_sample_shape_tuple():
    prior, maf = _make_prior_and_maf()
    post = Posterior(maf, prior, device="cpu")
    x_o = torch.randn(1, 5)
    samples = post.sample((3, 10), x_o, reject_outside_prior=False)
    assert samples.shape == (3, 10, 3)


def test_log_prob_basic():
    prior, maf = _make_prior_and_maf()
    post = Posterior(maf, prior, device="cpu", enable_leakage_correction=False)
    x_o = torch.randn(1, 5)
    theta = torch.randn(4, 3)
    lp = post.log_prob(theta, x_o, norm_posterior=False)
    assert lp.shape == (4,)


def test_leakage_correction_caching():
    prior, maf = _make_prior_and_maf()
    post = Posterior(maf, prior, device="cpu")
    x_o = torch.randn(1, 5)
    lc1 = post.leakage_correction(x_o, num_rejection_samples=500)
    lc2 = post.leakage_correction(x_o, num_rejection_samples=500)
    assert lc1 == lc2  # cached


def test_leakage_correction_force_update():
    prior, maf = _make_prior_and_maf()
    post = Posterior(maf, prior, device="cpu")
    x_o = torch.randn(1, 5)
    lc1 = post.leakage_correction(x_o, num_rejection_samples=500, force_update=False)
    lc2 = post.leakage_correction(x_o, num_rejection_samples=500, force_update=True)
    # force_update should recalc (may differ due to randomness but should be valid)
    assert 0 <= lc1 <= 1
    assert 0 <= lc2 <= 1


def test_log_prob_with_leakage():
    prior, maf = _make_prior_and_maf()
    post = Posterior(maf, prior, device="cpu", enable_leakage_correction=True)
    x_o = torch.randn(1, 5)
    theta = torch.randn(4, 3)
    lp_norm = post.log_prob(theta, x_o, norm_posterior=True)
    lp_raw = post.log_prob(theta, x_o, norm_posterior=False)
    assert lp_norm.shape == (4,)
    # normalized should be <= raw (log(fraction) <= 0, so subtract = more negative)
    assert (lp_norm >= lp_raw - 1e-6).all()


def test_map():
    prior, maf = _make_prior_and_maf()
    post = Posterior(maf, prior, device="cpu")
    x_o = torch.randn(1, 5)
    theta_map = post.map(x_o, num_init_samples=200, num_to_optimize=10, num_iter=100)
    assert theta_map.dim() == 1
    assert theta_map.shape[0] == 3
