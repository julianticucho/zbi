import json, os, shutil, tempfile
import torch
from torch import nn

from zbi.pipeline.setup import update_embedding
from zbi.pipeline.train import train
from zbi.utils.checkpoint import load_checkpoint, load_maf_from_checkpoint
from zbi.pipeline._resolve import resolve_class
from zbi.data.zarr_store import ZarrStore


class _EmbeddingA(nn.Module):
    def __init__(self, dim_out: int = 4):
        super().__init__()
        self.net = nn.Linear(10, dim_out)
    def forward(self, x):
        return self.net(x)


class _EmbeddingB(nn.Module):
    def __init__(self, dim_out: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, dim_out),
        )
    def forward(self, x):
        return self.net(x)


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
            "class": f"{_EmbeddingA.__module__}._EmbeddingA",
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


def test_update_embedding_changes_config():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        with open(f"{tmp}/config.json") as f:
            cfg = json.load(f)
        assert cfg["embedding"]["class"].endswith("_EmbeddingA")
        assert cfg["embedding"]["dim_out"] == 4

        update_embedding(tmp, _EmbeddingB, dim_out=8)
        with open(f"{tmp}/config.json") as f:
            cfg = json.load(f)
        assert cfg["embedding"]["class"].endswith("_EmbeddingB")
        assert cfg["embedding"]["dim_out"] == 8
    finally:
        shutil.rmtree(tmp)


def test_update_embedding_revert():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        update_embedding(tmp, _EmbeddingB, dim_out=8)
        update_embedding(tmp, _EmbeddingA, dim_out=4)
        with open(f"{tmp}/config.json") as f:
            cfg = json.load(f)
        assert cfg["embedding"]["class"].endswith("_EmbeddingA")
        assert cfg["embedding"]["dim_out"] == 4
    finally:
        shutil.rmtree(tmp)


def test_update_embedding_sin_config_raise():
    tmp = tempfile.mkdtemp()
    try:
        try:
            update_embedding(tmp, _EmbeddingB, dim_out=8)
            assert False, "deberia haber lanzado FileNotFoundError"
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(tmp)


def test_update_embedding_nuevo_modelo_usa_nueva_embedding():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        from zbi.data.zarr_store import ZarrStore
        store = ZarrStore(f"{tmp}/data")
        store.append(torch.randn(20, 2), torch.randn(20, 10))

        train(tmp, round=0, n_sims=20, offset=0, tag="old", max_epochs=2, stop_after_epochs=2)

        update_embedding(tmp, _EmbeddingB, dim_out=8)
        train(tmp, round=0, n_sims=20, offset=0, tag="new", max_epochs=2, stop_after_epochs=2)

        old_ckpt = load_checkpoint(f"{tmp}/models/round_00000_20_old.pt")
        new_ckpt = load_checkpoint(f"{tmp}/models/round_00000_20_new.pt")

        assert old_ckpt["embedding_net_config"]["class"].endswith("_EmbeddingA")
        assert new_ckpt["embedding_net_config"]["class"].endswith("_EmbeddingB")
    finally:
        shutil.rmtree(tmp)


def test_update_embedding_checkpoints_viejos_intactos():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        from zbi.data.zarr_store import ZarrStore
        store = ZarrStore(f"{tmp}/data")
        store.append(torch.randn(20, 2), torch.randn(20, 10))

        train(tmp, round=0, n_sims=20, offset=0, tag="old", max_epochs=2, stop_after_epochs=2)

        update_embedding(tmp, _EmbeddingB, dim_out=8)
        train(tmp, round=0, n_sims=20, offset=0, tag="new", max_epochs=2, stop_after_epochs=2)

        ckpt = load_checkpoint(f"{tmp}/models/round_00000_20_old.pt")
        emb_config = ckpt["embedding_net_config"]
        emb_class = resolve_class(emb_config["class"])
        emb_kwargs = {k: v for k, v in emb_config.items() if k != "class"}
        emb = emb_class(**emb_kwargs)
        maf = load_maf_from_checkpoint(ckpt, embedding_net=emb)
        theta = torch.randn(5, 2)
        x = torch.randn(5, 10)
        lp = maf(theta, x)
        assert lp.shape == (5,)
    finally:
        shutil.rmtree(tmp)
