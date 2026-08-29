import torch
import pickle
from joblib import Parallel, delayed
from zbi.simulators.base import Simulator


class SimCuadratico(Simulator):
    def simulate(self, theta, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return theta**2 + 0.1 * torch.randn_like(theta)


def test_parallel_basic():
    s = SimCuadratico()
    theta = torch.randn(10, 3)
    x = s(theta, show_progress=False, n_jobs=2)
    assert x.shape == (10, 3)


def test_parallel_equals_sequential():
    s = SimCuadratico()
    theta = torch.randn(10, 3)
    x_seq = s(theta, show_progress=False, n_jobs=1)
    x_par = s(theta, show_progress=False, n_jobs=2)
    assert x_seq.shape == x_par.shape == (10, 3)


def test_parallel_empty():
    s = SimCuadratico()
    x = s(torch.empty(0, 3), show_progress=False, n_jobs=2)
    assert x.shape == (0,)


def test_parallel_single_1d():
    s = SimCuadratico()
    x = s(torch.tensor([1.0, 2.0]), n_jobs=2)
    assert x.shape == (2,)


def test_parallel_large_n_jobs():
    s = SimCuadratico()
    theta = torch.randn(3, 3)
    x = s(theta, show_progress=False, n_jobs=8)
    assert x.shape == (3, 3)


def test_simulator_pickleable():
    s = SimCuadratico()
    data = pickle.dumps(s)
    s2 = pickle.loads(data)
    theta = torch.tensor([1.0, 2.0, 3.0])
    x1 = s.simulate(theta, seed=42)
    x2 = s2.simulate(theta, seed=42)
    assert torch.allclose(x1, x2)


def test_parallel_joblib():
    s = SimCuadratico()
    theta = torch.randn(10, 3)
    batches = [theta[i].numpy() for i in range(10)]
    seeds = list(range(10))

    def _run(t, seed):
        return s.simulate(torch.from_numpy(t), seed)

    results = Parallel(n_jobs=2)(
        delayed(_run)(b, sd) for b, sd in zip(batches, seeds)
    )
    x = torch.stack(results)
    assert x.shape == (10, 3)


def test_parallel_no_show_progress():
    s = SimCuadratico()
    theta = torch.randn(5, 3)
    x = s(theta, show_progress=False, n_jobs=2)
    assert x.shape == (5, 3)
