import json, os, shutil, tempfile
import torch

from zbi.pipeline.setup import update_maf, init
from zbi.pipeline.train import train
from zbi.utils.checkpoint import load_checkpoint
from zbi.data.zarr_store import ZarrStore


def _make_experiment(run_dir, with_maf_key=True):
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
    if with_maf_key:
        config["maf"] = {
            "hidden_features": 32,
            "num_transforms": 5,
            "num_blocks": 2,
            "dropout_probability": 0.0,
            "use_batch_norm": False,
        }
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    ZarrStore(f"{run_dir}/data").init(N=50, chunk_size=16, dim_theta=2, dim_x=10)
    torch.save(torch.randn(1, 10), f"{run_dir}/x_o.pt")


def test_update_maf_changes_config():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        update_maf(tmp, hidden_features=100, num_transforms=8)
        with open(f"{tmp}/config.json") as f:
            cfg = json.load(f)
        assert cfg["maf"]["hidden_features"] == 100
        assert cfg["maf"]["num_transforms"] == 8
        assert cfg["maf"]["num_blocks"] == 2
    finally:
        shutil.rmtree(tmp)


def test_update_maf_creates_key_if_missing():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp, with_maf_key=False)
        with open(f"{tmp}/config.json") as f:
            cfg = json.load(f)
        assert "maf" not in cfg

        update_maf(tmp, hidden_features=100)
        with open(f"{tmp}/config.json") as f:
            cfg = json.load(f)
        assert "maf" in cfg
        assert cfg["maf"]["hidden_features"] == 100
        assert cfg["maf"]["num_blocks"] == 2
    finally:
        shutil.rmtree(tmp)


def test_update_maf_no_config_raises():
    tmp = tempfile.mkdtemp()
    try:
        try:
            update_maf(tmp, hidden_features=100)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(tmp)


def test_update_maf_revert():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        update_maf(tmp, hidden_features=100)
        update_maf(tmp, hidden_features=32)
        with open(f"{tmp}/config.json") as f:
            cfg = json.load(f)
        assert cfg["maf"]["hidden_features"] == 32
    finally:
        shutil.rmtree(tmp)


def test_update_maf_partial_update():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        update_maf(tmp, num_transforms=10)
        with open(f"{tmp}/config.json") as f:
            cfg = json.load(f)
        assert cfg["maf"]["num_transforms"] == 10
        assert cfg["maf"]["hidden_features"] == 32
        assert cfg["maf"]["num_blocks"] == 2
    finally:
        shutil.rmtree(tmp)


def test_train_uses_maf_config():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp)
        store = ZarrStore(f"{tmp}/data")
        store.append(torch.randn(20, 2), torch.randn(20, 10))

        update_maf(tmp, hidden_features=64, num_transforms=3)
        train(tmp, round=0, n_sims=20, offset=0, tag="test", max_epochs=2, stop_after_epochs=2)

        ckpt = load_checkpoint(f"{tmp}/models/round_00000_20_test.pt")
        assert ckpt["maf_build_kwargs"]["hidden_features"] == 64
        assert ckpt["maf_build_kwargs"]["num_transforms"] == 3
    finally:
        shutil.rmtree(tmp)


def test_train_backward_compat_no_maf_key():
    tmp = tempfile.mkdtemp()
    try:
        _make_experiment(tmp, with_maf_key=False)
        store = ZarrStore(f"{tmp}/data")
        store.append(torch.randn(20, 2), torch.randn(20, 10))

        train(tmp, round=0, n_sims=20, offset=0, tag="test", max_epochs=2, stop_after_epochs=2)

        ckpt = load_checkpoint(f"{tmp}/models/round_00000_20_test.pt")
        assert ckpt["maf_build_kwargs"]["hidden_features"] == 32
    finally:
        shutil.rmtree(tmp)


def test_init_creates_maf_default():
    tmp = tempfile.mkdtemp()
    try:
        from zbi.examples.simple import SimuladorLineal, EmbeddingNet
        x_o = torch.randn(1, 10)
        init(
            run_dir=tmp,
            x_o=x_o,
            simulator_class=SimuladorLineal,
            embedding_class=EmbeddingNet,
            embedding_kwargs=dict(dim_in=10, dim_out=4),
            prior_low=(-1.0, -1.0),
            prior_high=(1.0, 1.0),
            dim_theta=2,
            dim_x=10,
        )
        with open(f"{tmp}/config.json") as f:
            cfg = json.load(f)
        assert "maf" in cfg
        assert cfg["maf"]["hidden_features"] == 32
        assert cfg["maf"]["num_transforms"] == 5
        assert cfg["maf"]["num_blocks"] == 2
    finally:
        shutil.rmtree(tmp)


def test_init_maf_kwargs_override():
    tmp = tempfile.mkdtemp()
    try:
        from zbi.examples.simple import SimuladorLineal, EmbeddingNet
        x_o = torch.randn(1, 10)
        init(
            run_dir=tmp,
            x_o=x_o,
            simulator_class=SimuladorLineal,
            embedding_class=EmbeddingNet,
            embedding_kwargs=dict(dim_in=10, dim_out=4),
            prior_low=(-1.0, -1.0),
            prior_high=(1.0, 1.0),
            dim_theta=2,
            dim_x=10,
            maf_kwargs={"hidden_features": 100, "num_transforms": 8},
        )
        with open(f"{tmp}/config.json") as f:
            cfg = json.load(f)
        assert cfg["maf"]["hidden_features"] == 100
        assert cfg["maf"]["num_transforms"] == 8
        assert cfg["maf"]["num_blocks"] == 2
    finally:
        shutil.rmtree(tmp)
