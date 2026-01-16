import numpy as np


from iapytoo.utils.config import Config
from iapytoo.dataset.scaling import MeanNormalize
from iapytoo.train.mlflow_model import MlflowModelProvider
from iapytoo.predictions.predictors import Predictor

from rockai.dataset.dataset import (
    Drp3dDatasetConfig
)
from rockai.models.resnet18 import Resnet18Model


class RockaiProvider(MlflowModelProvider):

    def __init__(self, config: Config) -> None:
        self._config:  Drp3dDatasetConfig = config.dataset
        i_dim = self._config.dim
        self._input_example = np.random.rand(
            1, i_dim, i_dim, i_dim).astype(np.float32)
        self._transform = MeanNormalize(config)
        self._model = Resnet18Model(config)
        self._predictor = Predictor()

    def code_definition(self) -> dict:
        from pathlib import Path
        return {
            "path": str(Path(__file__).parent),
            "config": {
                "module": "rockai.dataset.config",
                "dataset": "Drp3dDatasetConfig"
            },
            "provider": {
                "module": "rockai.provider",
                "class": "RockaiProvider"
            }
        }
