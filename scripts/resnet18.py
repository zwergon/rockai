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


from drp.models.resnet18 import Resnet18Model
from drp.dataset.dataset import Drp3dSqlite, MinMaxNormalize, MeanNormalize


class Resnet18Training(Training):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.predictions.add_plotter(ScatterPlotter(config))


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

    x_transform = MeanNormalize(config.dataset.mean, config.dataset.std)
    y_transform = MinMaxNormalize(0.2, 23)

    training = Resnet18Training(config)
    train_dataset = Drp3dSqlite(
        config,
        train_flag="train",
        x_transform=x_transform,
        y_transform=y_transform)

    valid_dataset = Drp3dSqlite(
        config,
        train_flag="valid",
        x_transform=x_transform,
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
