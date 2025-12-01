from iapytoo.utils.config import DatasetConfig, DatasetConfigFactory


class Drp3dDatasetConfig(DatasetConfig):
    db: str = ""
    dim: int = 32
    volumes: list[int] = [4419, 4420]
    mean: float = 0.0
    std: float = 1.0


DatasetConfigFactory().register_dataset_config(
    "drp3d_sqlite", Drp3dDatasetConfig)
