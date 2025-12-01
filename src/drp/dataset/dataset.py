import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose, ToTensor

from iapytoo.utils.config import Config

from drp.dataset.memmap import DataModality, load_cube
from drp.dataset.sqlite_dataset import SqliteDataset
from drp.dataset.config import Drp3dDatasetConfig


class MeanNormalize:
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, y):
        return (y - self.mean) / self.std


class MinMaxNormalize:
    def __init__(self, y_min, y_max) -> None:
        self.y_min = y_min
        self.y_max = y_max

    def __call__(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min)


class Drp3dSqliteDataset(Dataset):

    def __init__(self,
                 config: Config,
                 train_flag: str = "Train",
                 x_transform=None,
                 y_transform=None):
        dataset_config: Drp3dDatasetConfig = config.dataset
        self.root = dataset_config.path
        self.db = dataset_config.db
        self.train_flag = train_flag.lower()
        self.dim = dataset_config.dim
        self.x_transform = x_transform
        self.y_transform = y_transform

        with SqliteDataset(db_path=dataset_config.db) as db:
            self.cubes = db.get_subcubes(
                kind=self.train_flag, volumes=dataset_config.volumes)

    def __len__(self):
        return len(self.cubes)

    def cube_id(self, idx):
        cube = self.cubes[idx]
        return cube[4]

    def infos(self, idx):
        cube = self.cubes[idx]
        print(f"cube {idx} cube_id {cube[4]}  permeability {cube[3]}")

    def __getitem__(self, idx):
        cube = self.cubes[idx]
        offset = cube[0:3]
        cube_id = cube[4]

        minicube = load_cube(
            os.path.join(self.root, str(cube_id)),
            DataModality.GLV,
            offset=offset,
            subshape=[self.dim, self.dim, self.dim],
        ).astype(np.float32)

        permeability = np.array([cube[3]], dtype=np.float32)

        if self.x_transform is not None:
            minicube = self.x_transform(minicube)

        if self.y_transform is not None:
            permeability = self.y_transform(permeability)

        return torch.tensor(np.expand_dims(minicube, axis=0), dtype=torch.float32), \
            torch.tensor(permeability, dtype=torch.float32)
