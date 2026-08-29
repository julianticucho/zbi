import json, os, shutil, tempfile
import torch

from zbi.data.zarr_store import ZarrStore
from zbi.pipeline.train import train_ensemble_kl


def _make_experiment(run_dir, n_sims=50):
    os.makedirs(f"{run_dir}/models", exist_ok=True)
    config = {
        "run_dir": run_dir,
        "dim_theta": 2,
        "dim_x": 100,
        "zarr_N": n_sims,
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
    ZarrStore(f"{run_dir}/data").init(N=n_sims, chunk_size=16, dim_theta=2, dim_x=100)

    store = ZarrStore(f"{run_dir}/data")
    theta = torch.randn(n_sims, 2)
    x = torch.randn(n_sims, 100)
    store.data["theta"][:] = theta.numpy()
    store.data["x"][:] = x.numpy()
    store.meta["sim_status"][:] = 1

    with open(f"{run_dir}/sim_round_0.json", "w") as f:
        json.dump({"round_id": 0, "n_sims": n_sims, "offset": 0}, f)

    torch.save(torch.randn(1, 100), f"{run_dir}/x_o.pt")


def test_ensemble_kl_none():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        models, history = train_ensemble_kl(
            tmp, round=0, n_sims=50,
            n_members=2, kl_every=None,
            max_epochs=3, stop_after_epochs=5,
            batch_size=16,
        )
        assert len(models) == 2
        assert history["epochs"] == []
        assert history["max_kl"] == []
    finally:
        shutil.rmtree(tmp)


def test_ensemble_kl_tracks_history():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        models, history = train_ensemble_kl(
            tmp, round=0, n_sims=50,
            n_members=2, kl_every=1,
            max_epochs=3, stop_after_epochs=5,
            batch_size=16,
        )
        assert len(models) == 2
        assert len(history["epochs"]) > 0
        assert len(history["max_kl"]) > 0
    finally:
        shutil.rmtree(tmp)


def test_ensemble_stops_early():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp, n_sims=100)
        models, history = train_ensemble_kl(
            tmp, round=0, n_sims=100,
            n_members=2, kl_every=None,
            max_epochs=200, stop_after_epochs=3,
            batch_size=16,
        )
        ckpts = [f for f in os.listdir(f"{tmp}/models") if f.endswith(".pt")]
        assert len(ckpts) == 2
    finally:
        shutil.rmtree(tmp)
