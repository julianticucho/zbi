import torch
from torch.utils.data import Dataset
from zbi.data.dataloader import get_train_val_loaders


class DummyDataset(Dataset):
    def __init__(self, n: int):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return torch.tensor([idx]), torch.tensor([idx])


def test_basic_split():
    ds = DummyDataset(100)
    train, val = get_train_val_loaders(ds, batch_size=10, val_fraction=0.2)
    assert train is not None
    assert val is not None
    n_train = sum(1 for _ in train)
    n_val = sum(1 for _ in val)
    assert n_train == 8  # 80 train / 10 batch = 8
    assert n_val == 2    # 20 val / 10 batch = 2


def test_no_val():
    ds = DummyDataset(100)
    train, val = get_train_val_loaders(ds, batch_size=10, val_fraction=0.0)
    assert train is not None
    assert val is None


def test_indices_do_not_overlap():
    ds = DummyDataset(50)
    train, val = get_train_val_loaders(ds, batch_size=5, val_fraction=0.2)
    train_indices = set()
    for b in train:
        for t in b[0]:
            train_indices.add(t.item())
    val_indices = set()
    for b in val:
        for t in b[0]:
            val_indices.add(t.item())
    assert len(train_indices & val_indices) == 0
    assert len(train_indices | val_indices) == 50


def test_batch_size_larger_than_dataset():
    ds = DummyDataset(10)
    train, val = get_train_val_loaders(ds, batch_size=100, val_fraction=0.2)
    # batch_size should be clamped to dataset size
    for b in train:
        assert b[0].shape[0] <= 10


def test_empty_dataset():
    ds = DummyDataset(0)
    try:
        get_train_val_loaders(ds, batch_size=10)
        assert False, "should raise"
    except ValueError:
        pass


def test_drop_last():
    ds = DummyDataset(105)
    train, val = get_train_val_loaders(ds, batch_size=10, val_fraction=0.1, drop_last=True)
    # 105 * 0.9 = 94 train / 10 = 9 batches (drop last incomplete)
    n_train = sum(1 for _ in train)
    n_val = sum(1 for _ in val)
    assert n_train == 9
    assert n_val == 1  # 11 val / 10 = 1 batch (drop last)


def test_num_workers():
    ds = DummyDataset(100)
    train, val = get_train_val_loaders(ds, batch_size=10, num_workers=2)
    assert train.num_workers == 2
    assert val.num_workers == 2
