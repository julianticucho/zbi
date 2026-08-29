import os
import tempfile
import torch
from torch import nn
from torch.distributions import Uniform

from zbi.utils.checkpoint import (
    CHECKPOINT_KEYS,
    save_checkpoint,
    load_checkpoint,
    save_config_yaml,
    load_config_yaml,
    load_maf_from_checkpoint,
    load_prior_from_checkpoint,
)
from zbi.neural_nets.factory import build_maf_estimator
from zbi.utils.truncation import TruncatedBoxPrior


def test_save_load_checkpoint_cycle():
    theta = torch.randn(100, 3)
    x = torch.randn(100, 5)
    maf = build_maf_estimator(theta=theta, x=x, num_transforms=2, hidden_features=32)
    build_kwargs = {"dim_theta": 3, "dim_x": 5, "num_transforms": 2, "hidden_features": 32}
    config = {"n_rounds": 3, "batch_size": 64}

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(
            path=path,
            round_id=0,
            maf=maf,
            maf_build_kwargs=build_kwargs,
            prior_bounds=None,
            prior_type="original",
            config=config,
            embedding_net_config={"class": "Identity", "dim_in": 5},
            prior_meta={"type": "Uniform", "low": [0.0, 0.0, 0.0], "high": [1.0, 1.0, 1.0]},
        )
        ckpt = load_checkpoint(path)
        for key in CHECKPOINT_KEYS:
            assert key in ckpt, f"Falta clave {key}"
        assert ckpt["round_id"] == 0
        assert ckpt["prior_type"] == "original"
    finally:
        os.unlink(path)


def test_load_maf_from_checkpoint():
    theta = torch.randn(100, 4)
    x = torch.randn(100, 6)
    maf = build_maf_estimator(theta=theta, x=x, num_transforms=2, hidden_features=32)
    build_kwargs = {"dim_theta": 4, "dim_x": 6, "num_transforms": 2, "hidden_features": 32}
    emb = nn.Identity()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(
            path=path,
            round_id=0,
            maf=maf,
            maf_build_kwargs=build_kwargs,
            prior_bounds=None,
            prior_type="original",
            config={},
        )
        ckpt = load_checkpoint(path)
        maf_loaded = load_maf_from_checkpoint(ckpt, embedding_net=emb)
        assert maf_loaded.dim_theta == 4
        assert maf_loaded.dim_x == 6
        with torch.no_grad():
            out_orig = maf(theta[:4], x[:4])
            out_loaded = maf_loaded(theta[:4], x[:4])
        assert torch.allclose(out_orig, out_loaded, atol=1e-6)
    finally:
        os.unlink(path)


def test_load_prior_from_checkpoint_original():
    theta = torch.randn(50, 2)
    x = torch.randn(50, 3)
    maf = build_maf_estimator(theta=theta, x=x, num_transforms=2, hidden_features=16)
    build_kwargs = {"dim_theta": 2, "dim_x": 3, "num_transforms": 2, "hidden_features": 16}

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(
            path=path,
            round_id=0,
            maf=maf,
            maf_build_kwargs=build_kwargs,
            prior_bounds=None,
            prior_type="original",
            config={},
            prior_meta={"type": "Uniform", "low": [-1.0, -1.0], "high": [1.0, 1.0]},
        )
        ckpt = load_checkpoint(path)
        prior = load_prior_from_checkpoint(ckpt)
        assert isinstance(prior, Uniform)
    finally:
        os.unlink(path)


def test_load_prior_from_checkpoint_truncated():
    theta = torch.randn(50, 2)
    x = torch.randn(50, 3)
    maf = build_maf_estimator(theta=theta, x=x, num_transforms=2, hidden_features=16)
    build_kwargs = {"dim_theta": 2, "dim_x": 3, "num_transforms": 2, "hidden_features": 16}
    bounds = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(
            path=path,
            round_id=1,
            maf=maf,
            maf_build_kwargs=build_kwargs,
            prior_bounds=bounds,
            prior_type="truncated",
            config={},
            prior_meta={"type": "Uniform", "low": [-1.0, -1.0], "high": [1.0, 1.0]},
        )
        ckpt = load_checkpoint(path)
        prior = load_prior_from_checkpoint(ckpt)
        assert isinstance(prior, TruncatedBoxPrior)
        assert torch.equal(prior.box_bounds, bounds)
    finally:
        os.unlink(path)


def test_save_load_config_yaml():
    config = {"n_rounds": 3, "learning_rate": 0.001, "prior": {"low": [-3, -3], "high": [3, 3]}}
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        path = f.name
    try:
        save_config_yaml(path, config)
        loaded = load_config_yaml(path)
        assert loaded == config
    finally:
        os.unlink(path)


def test_checkpoint_keys_unchanged():
    expected = sorted(CHECKPOINT_KEYS)
    theta = torch.randn(20, 2)
    x = torch.randn(20, 3)
    maf = build_maf_estimator(theta=theta, x=x, num_transforms=1, hidden_features=16)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(
            path=path,
            round_id=0,
            maf=maf,
            maf_build_kwargs={"dim_theta": 2, "dim_x": 3},
            prior_bounds=None,
            prior_type="original",
            config={},
        )
        ckpt = load_checkpoint(path)
        assert sorted(ckpt.keys()) == expected
    finally:
        os.unlink(path)
