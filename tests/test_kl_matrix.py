import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Uniform

from zbi.inference import kl_matrix
from zbi.utils.plotting import plot_kl_matrix
from zbi.inference.posterior import Posterior


# estimador sintético: posterior gaussiana con log-prob exacta, expone el
# mismo contrato duck-typed que MAF (sample(num, x) + forward(theta, x))
class GaussianPosterior(nn.Module):
    def __init__(self, mean, cov):
        super().__init__()
        self.dist = MultivariateNormal(
            torch.as_tensor(mean, dtype=torch.float32),
            torch.as_tensor(cov, dtype=torch.float32),
        )
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, theta, x):
        return self.dist.log_prob(theta)

    def sample(self, n, x, sample_batch_size=None):
        return self.dist.sample((n,))


def analytic_kl(mu1, cov1, mu2, cov2):
    """KL(N1 || N2) analítica (papel Alvey et al. 2026, Ec. 6)."""
    mu1 = torch.as_tensor(mu1, dtype=torch.float64)
    mu2 = torch.as_tensor(mu2, dtype=torch.float64)
    c1 = torch.as_tensor(cov1, dtype=torch.float64)
    c2 = torch.as_tensor(cov2, dtype=torch.float64)
    inv2 = torch.linalg.inv(c2)
    t = (mu2 - mu1) @ inv2 @ (mu2 - mu1)
    logdet = torch.logdet(c2) - torch.logdet(c1)
    tr = torch.trace(inv2 @ c1)
    return float(0.5 * (t + logdet + tr - mu1.shape[0]))


def make_estimators():
    q0 = GaussianPosterior([0.0], [[1.0]])
    q1 = GaussianPosterior([0.5], [[2.25]])
    q2 = GaussianPosterior([0.0], [[4.0]])
    return [q0, q1, q2]


X_O = torch.zeros(1, 1)


def test_kl_matrix_diagonal_zero():
    est = make_estimators()
    K = kl_matrix(est, X_O, n_samples=5_000)
    assert K.shape == (3, 3)
    assert np.allclose(np.diag(K), 0.0)
    assert np.all(K >= -1e-6)


def test_kl_matrix_matches_analytic():
    est = make_estimators()
    K = kl_matrix(est, X_O, n_samples=50_000)
    k01 = analytic_kl([0.0], [[1.0]], [0.5], [[2.25]])
    k02 = analytic_kl([0.0], [[1.0]], [0.0], [[4.0]])
    assert K[0, 1] == pytest.approx(k01, abs=0.02)
    assert K[0, 2] == pytest.approx(k02, abs=0.02)


def test_kl_matrix_asymmetric():
    est = make_estimators()
    K = kl_matrix(est, X_O, n_samples=50_000)
    k01 = analytic_kl([0.0], [[1.0]], [0.5], [[2.25]])
    k10 = analytic_kl([0.5], [[2.25]], [0.0], [[1.0]])
    assert K[0, 1] == pytest.approx(k01, abs=0.02)
    assert K[1, 0] == pytest.approx(k10, abs=0.02)
    assert abs(K[0, 1] - K[1, 0]) > 1e-2


def test_kl_matrix_batched_matches_unbatched():
    est = make_estimators()
    torch.manual_seed(0)
    K_full = kl_matrix(est, X_O, n_samples=3_000, sample_batch_size=10_000, log_prob_batch_size=10_000)
    torch.manual_seed(0)
    K_batched = kl_matrix(est, X_O, n_samples=3_000, sample_batch_size=128, log_prob_batch_size=256)
    assert np.allclose(K_full, K_batched, atol=1e-8)


def test_kl_matrix_mc_convergence():
    est = make_estimators()
    ref = analytic_kl([0.0], [[1.0]], [0.5], [[2.25]])

    torch.manual_seed(1)
    K_small = kl_matrix(est, X_O, n_samples=500)
    torch.manual_seed(1)
    K_large = kl_matrix(est, X_O, n_samples=50_000)
    err_small = abs(K_small[0, 1] - ref)
    err_large = abs(K_large[0, 1] - ref)
    assert err_large < err_small


def test_kl_matrix_empty_raises():
    with pytest.raises(ValueError):
        kl_matrix([], X_O)


def test_kl_matrix_norm_posterior_presence():
    # una posterior amplia (leakage fuera del prior) y una angosta: la
    # corrección de leakage (norm_posterior=True) modifica la matriz KL
    wide = GaussianPosterior([0.0], [[4.0]])
    narrow = GaussianPosterior([0.2], [[0.25]])
    prior = Uniform(-1.0, 1.0)
    p_wide = Posterior(wide, prior, device="cpu")
    p_narrow = Posterior(narrow, prior, device="cpu")
    x_o = torch.zeros(1, 1)

    K_raw = kl_matrix([wide, narrow], x_o, n_samples=4_000, norm_posterior=False)
    K_norm = kl_matrix([p_wide, p_narrow], x_o, n_samples=4_000, norm_posterior=True)
    assert np.isfinite(K_norm).all()
    assert abs(K_norm[0, 1] - K_raw[0, 1]) > 1e-3


def test_plot_kl_matrix_renders_and_saves(tmp_path):
    est = make_estimators()
    K = kl_matrix(est, X_O, n_samples=5_000)
    out = tmp_path / "kl_matrix.pdf"
    fig, ax = plot_kl_matrix(
        K,
        sample_labels=["r1", "r2", "r3"],
        title="KL matrix test",
        output_path=str(out),
    )
    assert os.path.exists(out)
    assert out.stat().st_size > 0
    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.parametrize("log_scale", [True, False])
@pytest.mark.parametrize("annotate", [True, False])
def test_plot_kl_matrix_options(tmp_path, log_scale, annotate):
    K = np.array([[0.0, 0.05, 2.0], [0.03, 0.0, 0.8], [1.5, 0.1, 0.0]])
    fig, ax = plot_kl_matrix(
        K, log_scale=log_scale, annotate=annotate, dim_theta=6,
        output_path=str(tmp_path / "opt.pdf"),
    )
    assert os.path.exists(tmp_path / "opt.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_kl_matrix_non_square_raises():
    with pytest.raises(ValueError):
        plot_kl_matrix(np.ones((2, 3)))