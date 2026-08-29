import json, os, shutil, tempfile
import torch

from zbi.data.zarr_store import ZarrStore
from zbi.pipeline.simulate import simulate, delete_last_round


def _make_experiment(run_dir):
    os.makedirs(f"{run_dir}/models", exist_ok=True)
    config = {
        "run_dir": run_dir,
        "dim_theta": 2,
        "dim_x": 100,
        "zarr_N": 50,
        "zarr_chunk_size": 16,
        "simulator": {"class": "zbi.examples.simple.SimuladorLineal"},
        "embedding": {
            "class": "zbi.examples.simple.EmbeddingNet",
            "dim_in": 100,
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
    ZarrStore(f"{run_dir}/data").init(N=50, chunk_size=16, dim_theta=2, dim_x=100)
    torch.save(torch.randn(1, 100), f"{run_dir}/x_o.pt")


def _write_sim_round(run_dir, round_id, n_sims, offset):
    meta = {"round_id": round_id, "n_sims": n_sims, "offset": offset}
    with open(f"{run_dir}/sim_round_{round_id}.json", "w") as f:
        json.dump(meta, f)


def _fill_store(run_dir, start, count):
    store = ZarrStore(f"{run_dir}/data")
    theta = torch.randn(count, 2)
    x = torch.randn(count, 100)
    store.data["theta"][start:start + count] = theta.numpy()
    store.data["x"][start:start + count] = x.numpy()
    store.meta["sim_status"][start:start + count] = 1


def _write_proposal(run_dir, round_id):
    proposal = {
        "prior_meta": {
            "type": "Uniform",
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        },
        "bounds": {
            "low": [-0.5, -0.5],
            "high": [0.5, 0.5],
        },
        "threshold": 1e-3,
        "checkpoints": [f"round_{round_id:05d}_500.pt"],
    }
    with open(f"{run_dir}/proposal.json", "w") as f:
        json.dump(proposal, f, indent=2)
    return proposal


def test_simulate_saves_proposal():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        _write_sim_round(tmp, round_id=0, n_sims=20, offset=0)
        _fill_store(tmp, start=0, count=20)
        _write_proposal(tmp, round_id=1)

        simulate(tmp, round=1, n_sims=10)

        assert os.path.exists(f"{tmp}/proposals/round_1.json")
    finally:
        shutil.rmtree(tmp)


def test_simulate_round_0_no_proposal():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)

        simulate(tmp, round=0, n_sims=10)

        assert not os.path.exists(f"{tmp}/proposals")
    finally:
        shutil.rmtree(tmp)


def test_simulate_proposal_content_matches():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        _write_sim_round(tmp, round_id=0, n_sims=20, offset=0)
        _fill_store(tmp, start=0, count=20)
        proposal = _write_proposal(tmp, round_id=1)

        simulate(tmp, round=1, n_sims=10)

        with open(f"{tmp}/proposals/round_1.json") as f:
            saved = json.load(f)
        assert saved == proposal
    finally:
        shutil.rmtree(tmp)


def test_simulate_overwrites_proposal_file():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        _write_sim_round(tmp, round_id=0, n_sims=20, offset=0)
        _fill_store(tmp, start=0, count=20)
        _write_proposal(tmp, round_id=1)

        simulate(tmp, round=1, n_sims=5)
        _write_proposal(tmp, round_id=1)
        simulate(tmp, round=1, n_sims=5)

        with open(f"{tmp}/proposals/round_1.json") as f:
            saved = json.load(f)
        assert saved["threshold"] == 1e-3
    finally:
        shutil.rmtree(tmp)


def test_delete_last_round_removes_proposal():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        _write_sim_round(tmp, round_id=0, n_sims=20, offset=0)
        _fill_store(tmp, start=0, count=20)
        os.makedirs(f"{tmp}/proposals", exist_ok=True)
        _write_proposal(tmp, round_id=0)
        shutil.copy(f"{tmp}/proposal.json", f"{tmp}/proposals/round_0.json")

        delete_last_round(tmp)

        assert not os.path.exists(f"{tmp}/proposals/round_0.json")
    finally:
        shutil.rmtree(tmp)


def test_delete_last_round_no_proposal_no_error():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        _write_sim_round(tmp, round_id=0, n_sims=20, offset=0)
        _fill_store(tmp, start=0, count=20)

        delete_last_round(tmp)

        assert not os.path.exists(f"{tmp}/sim_round_0.json")
    finally:
        shutil.rmtree(tmp)


def test_delete_last_round_only_removes_last():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        _write_sim_round(tmp, round_id=0, n_sims=20, offset=0)
        _fill_store(tmp, start=0, count=20)
        _write_sim_round(tmp, round_id=1, n_sims=15, offset=20)
        _fill_store(tmp, start=20, count=15)
        os.makedirs(f"{tmp}/proposals", exist_ok=True)
        _write_proposal(tmp, round_id=0)
        shutil.copy(f"{tmp}/proposal.json", f"{tmp}/proposals/round_0.json")
        _write_proposal(tmp, round_id=1)
        shutil.copy(f"{tmp}/proposal.json", f"{tmp}/proposals/round_1.json")

        delete_last_round(tmp)

        assert not os.path.exists(f"{tmp}/proposals/round_1.json")
        assert os.path.exists(f"{tmp}/proposals/round_0.json")
    finally:
        shutil.rmtree(tmp)


def test_proposal_lifecycle():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)

        simulate(tmp, round=0, n_sims=20)
        _write_proposal(tmp, round_id=1)

        simulate(tmp, round=1, n_sims=15)
        assert os.path.exists(f"{tmp}/proposals/round_1.json")

        delete_last_round(tmp)
        assert not os.path.exists(f"{tmp}/proposals/round_1.json")
        assert not os.path.exists(f"{tmp}/sim_round_1.json")
    finally:
        shutil.rmtree(tmp)
