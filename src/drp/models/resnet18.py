from iapytoo.train.factories import Model
from iapytoo.utils.config import Config
from drp.models.backbone import ResNet, generate_model


class Resnet18Model(Model):
    def __init__(self, loader, config: Config):
        super(Resnet18Model, self).__init__(loader, config)
        self.model: ResNet = generate_model(model_depth=18, n_classes=1)

    def forward(self, x):
        return self.model(x)
