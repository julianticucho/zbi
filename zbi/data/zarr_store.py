from typing import Optional
import numpy as np
import torch
import zarr
import fasteners


class ZarrStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self.synchronizer = zarr.ProcessSynchronizer(path + ".sync")
        self.root = zarr.open_group(
            path, mode="a", synchronizer=self.synchronizer,
        )
        self.lock = fasteners.InterProcessLock(path + ".lock")
        self._N = 0
        self._chunk_size = 512

    @property
    def data(self):
        return self.root["data"]

    @property
    def meta(self):
        return self.root["meta"]

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def num_filled(self) -> int:
        return int(np.sum(self.meta["sim_status"][:] == 1))

    @property
    def sims_required(self) -> int:
        return int(np.sum(self.meta["sim_status"][:] == 0))

    def init(
        self,
        N: int,
        chunk_size: int,
        dim_theta: int,
        dim_x: int,
        dtype=np.float32,
    ) -> "ZarrStore":
        self._N = N
        self._chunk_size = chunk_size
        dt = np.dtype(dtype)
        if "data" in self.root:
            del self.root["data"]
        if "meta" in self.root:
            del self.root["meta"]
        self.root.create_group("data")
        self.root.create_group("meta")
        self.data.zeros(
            "theta", shape=(N, dim_theta), chunks=(chunk_size, dim_theta), dtype=dt,
        )
        self.data.zeros(
            "x", shape=(N, dim_x), chunks=(chunk_size, dim_x), dtype=dt,
        )
        self.meta.zeros(
            "sim_status", shape=(N,), chunks=(chunk_size,), dtype="i4",
        )
        return self

    def __len__(self) -> int:
        return self._N

    def keys(self) -> list[str]:
        return list(self.data.array_keys())

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.item()
        theta = torch.from_numpy(self.data["theta"][i : i + 1])
        x = torch.from_numpy(self.data["x"][i : i + 1])
        return {"theta": theta, "x": x}

    def append(self, theta: torch.Tensor, x: torch.Tensor) -> None:
        batch = theta.shape[0]
        with self.lock:
            sim_status = self.meta["sim_status"][:]
            empty = np.where(sim_status == 0)[0]
            if len(empty) < batch:
                raise RuntimeError(
                    f"Need {batch} empty slots but only have {len(empty)}. "
                    f"Store has N={self._N} slots."
                )
            idx = empty[:batch]
            self.data["theta"][idx] = theta.detach().numpy()
            self.data["x"][idx] = x.detach().numpy()
            self.meta["sim_status"][idx] = 1

    def resize(self, new_N: int) -> "ZarrStore":
        if new_N <= self._N:
            raise ValueError(
                f"new_N ({new_N}) must be greater than current N ({self._N})."
            )
        self.data["theta"].resize((new_N, self.data["theta"].shape[1]))
        self.data["x"].resize((new_N, self.data["x"].shape[1]))
        self.meta["sim_status"].resize((new_N,))
        self._N = new_N
        return self

    def clear(self, start: int, count: int) -> None:
        with self.lock:
            self.meta["sim_status"][start:start + count] = 0
            self.data["theta"][start:start + count] = 0
            self.data["x"][start:start + count] = 0

    def get_dataset(self) -> "ZarrDataset":
        return ZarrDataset(self)


class ZarrDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_store: ZarrStore) -> None:
        self.store = zarr_store

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.item()
        theta = torch.from_numpy(self.store.data["theta"][idx]).float()
        x = torch.from_numpy(self.store.data["x"][idx]).float()
        return theta, x
