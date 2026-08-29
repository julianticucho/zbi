import torch
from torch import nn
from torch.utils.data import DataLoader

from zbi.neural_nets.factory import build_maf_estimator
from zbi.data.dataloader import get_train_val_loaders
from zbi.inference.posterior import Posterior
from torch.utils.data import TensorDataset
from zbi.examples.simple import EmbeddingNet


def train(
    maf: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    max_epochs: int = 2_147_483_647,
    stop_after_epochs: int = 20,
    learning_rate: float = 5e-4,
    device: str = "cpu",
):
    post = Posterior(maf, torch.distributions.Uniform(0, 1), device=device)
    post.train(
        train_loader, val_loader,
        max_epochs=max_epochs,
        stop_after_epochs=stop_after_epochs,
        learning_rate=learning_rate,
    )


def test_train_early_stopping():
    theta = torch.randn(200, 3)
    x = torch.randn(200, 5)
    maf = build_maf_estimator(theta=theta, x=x, num_transforms=2, hidden_features=32)
    ds = TensorDataset(theta, x)
    train_loader, val_loader = get_train_val_loaders(ds, batch_size=32, val_fraction=0.1)
    train(maf, train_loader, val_loader, max_epochs=5, stop_after_epochs=3, device="cpu")
    assert not maf.training


def test_train_no_val():
    theta = torch.randn(100, 2)
    x = torch.randn(100, 4)
    maf = build_maf_estimator(theta=theta, x=x, num_transforms=2, hidden_features=16)
    ds = TensorDataset(theta, x)
    train_loader, _ = get_train_val_loaders(ds, batch_size=32, val_fraction=0.0)
    train(maf, train_loader, None, max_epochs=3, device="cpu")
    assert not maf.training


def test_train_loss_decreases():
    theta = torch.randn(300, 2)
    x = torch.randn(300, 3)
    maf = build_maf_estimator(theta=theta, x=x, num_transforms=2, hidden_features=16)
    ds = TensorDataset(theta, x)
    train_loader, val_loader = get_train_val_loaders(ds, batch_size=32, val_fraction=0.1)
    maf.train()
    with torch.no_grad():
        init_loss = sum(
            maf.loss(t, x_val).item() * t.shape[0]
            for t, x_val in val_loader
        ) / len(val_loader.dataset)
    train(maf, train_loader, val_loader, max_epochs=20, stop_after_epochs=5, device="cpu")
    maf.eval()
    with torch.no_grad():
        final_loss = sum(
            maf.loss(t, x_val).item() * t.shape[0]
            for t, x_val in val_loader
        ) / len(val_loader.dataset)
    assert final_loss < init_loss


def test_embedding_net_shapes():
    net = EmbeddingNet(dim_in=100, dim_out=5)
    x = torch.randn(10, 100)
    out = net(x)
    assert out.shape == (10, 5)
