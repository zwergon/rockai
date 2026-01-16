import numpy as np
from torch.utils.data import DataLoader


from iapytoo.dataset.scaling import MinMaxNormalize
from iapytoo.predictions.plotters import ScatterPlotter
from iapytoo.train.factories import Factory
from iapytoo.utils.config import ConfigFactory, Config
from iapytoo.train.training import Training


from rockai.dataset.dataset import (
    Drp3dSqliteDataset
)
from rockai.provider import RockaiProvider


class RockaiTraining(Training):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.predictions.add_plotter(ScatterPlotter(config))


if __name__ == "__main__":
    from iapytoo.utils.arguments import parse_args

    factory = Factory()
    factory.register_provider(RockaiProvider)

    args = parse_args()

    if args.run_id is not None:
        config = ConfigFactory.from_run_id(args.run_id, args.tracking_uri)
        config.training.epochs = args.epochs
    else:
        # INPUT Parameters
        config = ConfigFactory.from_yaml(args.yaml)

    Training.seed(config)

    y_transform = MinMaxNormalize(config, name="permeability")

    training: RockaiTraining = RockaiTraining(config)

    train_dataset = Drp3dSqliteDataset(
        config,
        train_flag="train",
        x_transform=training.transform,
        y_transform=y_transform)

    valid_dataset = Drp3dSqliteDataset(
        config,
        train_flag="valid",
        x_transform=training.transform,
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
