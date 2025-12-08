import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image


from iapytoo.predictions.plotters import ScatterPlotter
from iapytoo.train.factories import Model, Scheduler, Factory
from iapytoo.utils.config import ConfigFactory, Config
from iapytoo.train.training import Training
from iapytoo.train.mlflow_model import MlflowTransform, IMlfowModelProvider


from mlflow.types.schema import TensorSpec, Schema
from mlflow.models import ModelSignature


from rockai.models.resnet18 import Resnet18Model
from rockai.dataset.dataset import (
    Drp3dSqliteDataset,
    MinMaxNormalize,
    MeanNormalize,
    Drp3dDatasetConfig
)


class RockaiTransform(MlflowTransform):
    def __init__(self, config: Config) -> None:
        super().__init__(transform=None)
        dataset_config: Drp3dDatasetConfig = config.dataset
        self.transform = MeanNormalize(
            dataset_config.mean,
            dataset_config.std
        )

    def __call__(self, model_input, *args, **kwds) -> torch.Tensor:
        # input is numpy array
        model_input = self.transform(model_input)
        tensor = torch.from_numpy(model_input)
        return tensor.float()


class RockaiMlfowModelProvider(IMlfowModelProvider):

    def __init__(self, config: Config) -> None:
        self._config:  Drp3dDatasetConfig = config.dataset
        i_dim = self._config.dim
        self._input_example = np.random.rand(
            1, 1, i_dim, i_dim, i_dim).astype(np.float32)
        self._transform = RockaiTransform(config)

    def get_input_example(self) -> np.array:
        return self._input_example

    def get_transform(self) -> MlflowTransform:
        return self._transform

    def get_signature(self) -> ModelSignature:
        input_schema = Schema([
            TensorSpec(np.dtype(np.float32),
                       shape=self._input_example.shape)
        ])
        output_schema = Schema([
            TensorSpec(np.dtype(np.float32), shape=(-1, 1))
        ])
        return ModelSignature(inputs=input_schema, outputs=output_schema)


class RockaiTraining(Training):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.predictions.add_plotter(ScatterPlotter(config))
        self.mlflow_model_provider = RockaiMlfowModelProvider(config)


if __name__ == "__main__":
    from iapytoo.utils.arguments import parse_args

    factory = Factory()
    factory.register_model("resnet18", Resnet18Model)

    args = parse_args()

    if args.run_id is not None:
        config = ConfigFactory.from_run_id(args.run_id, args.tracking_uri)
        config.training.epochs = args.epochs
    else:
        # INPUT Parameters
        config = ConfigFactory.from_yaml(args.yaml)

    Training.seed(config)

    y_transform = MinMaxNormalize(0.2, 23)

    training = RockaiTraining(config)
    mlflow_transform: MlflowTransform = training.get_transform()
    train_dataset = Drp3dSqliteDataset(
        config,
        train_flag="train",
        x_transform=mlflow_transform.transform,
        y_transform=y_transform)

    valid_dataset = Drp3dSqliteDataset(
        config,
        train_flag="valid",
        x_transform=mlflow_transform.transform,
        y_transform=y_transform
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers
    )

    training.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        run_id=args.run_id
    )
