import torch
from torch.utils.data import Dataset, TensorDataset


class SlicedDataset(Dataset):
    def __init__(self, dataset: Dataset, interest_dims: list[int]):
        self.dataset = dataset
        self.interest_dims = interest_dims

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        theta, x = self.dataset[idx]
        return theta[self.interest_dims], x


def load_to_ram(zarr_store, indices, interest_dims):
    dataset = zarr_store.get_dataset()
    theta_list, x_list = [], []
    for idx in indices:
        t, x = dataset[idx]
        theta_list.append(t.unsqueeze(0))
        x_list.append(x.unsqueeze(0))
    all_theta = torch.cat(theta_list)
    all_x = torch.cat(x_list)
    if interest_dims is not None:
        all_theta = all_theta[:, interest_dims]
    return TensorDataset(all_theta, all_x)
