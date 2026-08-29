import json, os, shutil, tempfile
import numpy as np
import torch

from zbi.data.zarr_store import ZarrStore
from zbi.pipeline.simulate import delete_last_round


def _make_experiment(run_dir):
    os.makedirs(f"{run_dir}/models", exist_ok=True)
    config = {
        "run_dir": run_dir,
        "dim_theta": 2,
        "dim_x": 10,
        "zarr_N": 50,
        "zarr_chunk_size": 16,
        "simulator": {"class": "zbi.examples.simple.SimuladorLineal"},
        "embedding": {
            "class": "zbi.examples.simple.EmbeddingNet",
            "dim_in": 10,
            "dim_out": 4,
        },
        "prior_meta": {
            "type": "Uniform",
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        },
    }
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    ZarrStore(f"{run_dir}/data").init(N=50, chunk_size=16, dim_theta=2, dim_x=10)
    torch.save(torch.randn(1, 10), f"{run_dir}/x_o.pt")


def _write_sim_round(run_dir, round_id, n_sims, offset):
    meta = {"round_id": round_id, "n_sims": n_sims, "offset": offset}
    with open(f"{run_dir}/sim_round_{round_id}.json", "w") as f:
        json.dump(meta, f)


def _fill_store(run_dir, start, count):
    store = ZarrStore(f"{run_dir}/data")
    theta = torch.randn(count, 2)
    x = torch.randn(count, 10)
    store.data["theta"][start:start + count] = theta.numpy()
    store.data["x"][start:start + count] = x.numpy()
    store.meta["sim_status"][start:start + count] = 1


def test_delete_last_round_clears_data():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        _write_sim_round(tmp, round_id=0, n_sims=20, offset=0)
        _fill_store(tmp, start=0, count=20)

        store = ZarrStore(f"{tmp}/data")
        assert store.num_filled == 20

        delete_last_round(tmp)

        store = ZarrStore(f"{tmp}/data")
        assert store.num_filled == 0
        assert not os.path.exists(f"{tmp}/sim_round_0.json")
    finally:
        shutil.rmtree(tmp)


def test_delete_last_round_multiple_rounds():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        _write_sim_round(tmp, round_id=0, n_sims=20, offset=0)
        _fill_store(tmp, start=0, count=20)
        _write_sim_round(tmp, round_id=1, n_sims=15, offset=20)
        _fill_store(tmp, start=20, count=15)

        store = ZarrStore(f"{tmp}/data")
        assert store.num_filled == 35

        delete_last_round(tmp)

        store = ZarrStore(f"{tmp}/data")
        assert store.num_filled == 20
        assert not os.path.exists(f"{tmp}/sim_round_1.json")
        assert os.path.exists(f"{tmp}/sim_round_0.json")
    finally:
        shutil.rmtree(tmp)


def test_delete_last_round_no_files_raises():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        try:
            delete_last_round(tmp)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(tmp)


def test_delete_last_round_cleared_slots_reusable():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        _write_sim_round(tmp, round_id=0, n_sims=20, offset=0)
        _fill_store(tmp, start=0, count=20)

        delete_last_round(tmp)

        store = ZarrStore(f"{tmp}/data")
        theta = torch.randn(10, 2)
        x = torch.randn(10, 10)
        store.append(theta, x)
        assert store.num_filled == 10
    finally:
        shutil.rmtree(tmp)
