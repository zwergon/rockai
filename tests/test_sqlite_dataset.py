import unittest
import numpy as np
from pathlib import Path
from iapytoo.utils.config import Config, ConfigFactory

from drp.dataset.config import Drp3dDatasetConfig
from drp.dataset.sqlite_dataset import SqliteDataset


class TestSqliteDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        root_path = Path(__file__).parent
        cls.config: Config = ConfigFactory.from_yaml(
            root_path / "config.yml")

    def test_value_by_group(self):
        perm_by_groups = {}

        dataset_config: Drp3dDatasetConfig = self.config.dataset
        with SqliteDataset(db_path=dataset_config.db) as db:
            perm_by_groups = db.get_values_by_group()

        for k, v in perm_by_groups.items():
            hist, bins = np.histogram(v, 128)
            print(
                f"Group: {k}, Num values: {len(v)}, Histogram bins: {bins}, counts: {hist}")


if __name__ == "__main__":
    unittest.main()
