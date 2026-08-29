import tempfile, os, shutil
import torch
import numpy as np
from zbi.data.zarr_store import ZarrStore, ZarrDataset


def _make_store():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "test.zarr")
    store = ZarrStore(path)
    store.init(N=100, chunk_size=16, dim_theta=2, dim_x=5)
    return store, tmp


def test_init_and_len():
    store, tmp = _make_store()
    assert len(store) == 100
    assert store.chunk_size == 16
    assert "theta" in store.keys()
    assert "x" in store.keys()
    shutil.rmtree(tmp)


def test_append_and_getitem():
    store, tmp = _make_store()
    theta = torch.randn(10, 2)
    x = torch.randn(10, 5)
    store.append(theta, x)
    assert store.sims_required == 90
    entry = store[0]
    assert torch.allclose(entry["theta"], theta[0:1])
    assert torch.allclose(entry["x"], x[0:1])
    shutil.rmtree(tmp)


def test_append_full():
    store, tmp = _make_store()
    theta = torch.randn(100, 2)
    x = torch.randn(100, 5)
    store.append(theta, x)
    assert store.sims_required == 0
    try:
        store.append(torch.randn(1, 2), torch.randn(1, 5))
        assert False, "should raise"
    except RuntimeError:
        pass
    shutil.rmtree(tmp)


def test_append_too_many():
    store, tmp = _make_store()
    try:
        store.append(torch.randn(200, 2), torch.randn(200, 5))
        assert False, "should raise"
    except RuntimeError:
        pass
    shutil.rmtree(tmp)


def test_zarr_dataset():
    store, tmp = _make_store()
    theta = torch.randn(50, 2)
    x = torch.randn(50, 5)
    store.append(theta, x)
    ds = ZarrDataset(store)
    assert len(ds) == 100
    t, x_ = ds[3]
    assert torch.allclose(t, theta[3])
    assert torch.allclose(x_, x[3])
    shutil.rmtree(tmp)


def test_zarr_dataset_random_access():
    store, tmp = _make_store()
    theta = torch.randn(50, 2)
    x = torch.randn(50, 5)
    store.append(theta, x)
    ds = ZarrDataset(store)
    indices = torch.randperm(50)
    for i in indices[:5]:
        t, x_ = ds[i]
        assert torch.allclose(t, theta[i])
        assert torch.allclose(x_, x[i])
    shutil.rmtree(tmp)


def test_multiple_appends():
    store, tmp = _make_store()
    for i in range(5):
        theta = torch.randn(10, 2)
        x = torch.randn(10, 5)
        store.append(theta, x)
    assert store.sims_required == 50
    shutil.rmtree(tmp)


def test_resize_basic():
    store, tmp = _make_store()
    assert len(store) == 100
    store.resize(200)
    assert len(store) == 200
    assert store.sims_required == 200
    shutil.rmtree(tmp)


def test_resize_preserves_data():
    store, tmp = _make_store()
    theta = torch.randn(30, 2)
    x = torch.randn(30, 5)
    store.append(theta, x)
    t0, x0 = store[0]["theta"], store[0]["x"]
    store.resize(200)
    assert torch.allclose(store[0]["theta"], t0)
    assert torch.allclose(store[0]["x"], x0)
    assert store.sims_required == 170
    shutil.rmtree(tmp)


def test_resize_allows_more_appends():
    store, tmp = _make_store()
    store.append(torch.randn(100, 2), torch.randn(100, 5))
    assert store.sims_required == 0
    store.resize(200)
    assert store.sims_required == 100
    store.append(torch.randn(100, 2), torch.randn(100, 5))
    assert store.sims_required == 0
    shutil.rmtree(tmp)


def test_resize_shrink_raises():
    store, tmp = _make_store()
    try:
        store.resize(50)
        assert False, "deberia haber lanzado ValueError"
    except ValueError:
        pass
    shutil.rmtree(tmp)


def test_resize_twice():
    store, tmp = _make_store()
    theta = torch.randn(30, 2)
    x = torch.randn(30, 5)
    store.append(theta, x)
    store.resize(200)
    store.resize(300)
    assert len(store) == 300
    assert store.sims_required == 270
    shutil.rmtree(tmp)
