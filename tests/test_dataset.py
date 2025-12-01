import unittest
import torch
from pathlib import Path
from iapytoo.utils.config import Config, ConfigFactory
from drp.dataset.dataset import Drp3dSqliteDataset, MinMaxNormalize, MeanNormalize
from drp.dataset.config import Drp3dDatasetConfig


class TestDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        root_path = Path(__file__).parent
        cls.config: Config = ConfigFactory.from_yaml(
            root_path / "config.yml")
        cls.dataset_config: Drp3dDatasetConfig = cls.config.dataset

    def test_sqlite_dataset(self):
        dataset = Drp3dSqliteDataset(self.config, train_flag="train")
        minicube, permeability = dataset[0]
        print(f"{len(dataset)} cubes in training")
        print(minicube.shape, torch.min(minicube), torch.max(
            minicube), minicube.dtype, permeability)

        dataset = Drp3dSqliteDataset(self.config, train_flag="valid")
        minicube, permeability = dataset[0]
        print(f"{len(dataset)} cubes in validation")
        print(minicube.shape, torch.min(minicube), torch.max(
            minicube), minicube.dtype, permeability)

    def test_normalize(self):
        x_transform = MeanNormalize(self.dataset_config.mean,
                                    self.dataset_config.std)
        y_transform = MinMaxNormalize(0.2, 23)
        dataset = Drp3dSqliteDataset(self.config, train_flag="train",
                                     x_transform=x_transform, y_transform=y_transform)
        minicube, permeability = dataset[0]
        print(f"Normalisation")
        print(minicube.shape, torch.min(minicube), torch.max(
            minicube), minicube.dtype, permeability)


if __name__ == "__main__":
    unittest.main()
