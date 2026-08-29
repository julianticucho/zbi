import torch
from zbi.simulators.base import Simulator


class SimCuadratico(Simulator):
    """Test simulator: x = theta^2 + ruido."""

    def simulate(self, theta, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return theta**2 + 0.1 * torch.randn_like(theta)


def test_not_implemented():
    s = Simulator()
    try:
        s.simulate(torch.tensor([1.0, 2.0]))
        assert False, "should raise"
    except NotImplementedError:
        pass


def test_batch():
    s = SimCuadratico()
    theta = torch.randn(100, 3)
    x = s(theta, show_progress=False)
    assert x.shape == (100, 3)


def test_single_1d():
    s = SimCuadratico()
    theta = torch.tensor([1.0, 2.0, 3.0])
    x = s(theta)
    assert x.shape == (3,)


def test_empty_batch():
    s = SimCuadratico()
    theta = torch.empty(0, 3)
    x = s(theta, show_progress=False)
    assert x.shape == (0,)


def test_reproducible():
    s = SimCuadratico()
    theta = torch.tensor([1.0, 2.0])
    x1 = s.simulate(theta, seed=42)
    x2 = s.simulate(theta, seed=42)
    assert torch.allclose(x1, x2)


def test_different_seeds_different():
    s = SimCuadratico()
    theta = torch.tensor([1.0, 2.0])
    x1 = s.simulate(theta, seed=1)
    x2 = s.simulate(theta, seed=2)
    assert not torch.allclose(x1, x2)


def test_default_seed_is_random():
    s = SimCuadratico()
    theta = torch.tensor([1.0, 2.0])
    x1 = s.simulate(theta)
    x2 = s.simulate(theta)
    # seed=None → aleatorio, muy improbable que sean iguales
    assert not torch.allclose(x1, x2)
