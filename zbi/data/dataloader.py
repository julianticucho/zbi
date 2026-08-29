from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


def get_train_val_loaders(
    dataset: Dataset,
    batch_size: int,
    val_fraction: float = 0.1,
    drop_last: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, Optional[DataLoader]]:
    num_examples = len(dataset)
    if num_examples == 0:
        raise ValueError("Empty dataset.")

    num_training = int((1 - val_fraction) * num_examples)
    num_validation = num_examples - num_training

    permuted = torch.randperm(num_examples)
    train_indices = permuted[:num_training].tolist()
    val_indices = permuted[num_training:].tolist()

    train_loader = DataLoader(
        dataset,
        batch_size=min(batch_size, num_training) if num_training > 0 else batch_size,
        sampler=SubsetRandomSampler(train_indices),
        drop_last=drop_last,
        num_workers=num_workers,
    )

    val_loader = None
    if val_fraction > 0 and num_validation > 0:
        val_loader = DataLoader(
            dataset,
            batch_size=min(batch_size, num_validation),
            sampler=SubsetRandomSampler(val_indices),
            drop_last=drop_last,
            num_workers=num_workers,
        )

    return train_loader, val_loader
