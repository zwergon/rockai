import unittest
import torch
from rockai.models.backbone import ResNet, generate_model


class TestModels(unittest.TestCase):

    def test_resnet18(self):

        model: ResNet = generate_model(model_depth=18, n_classes=1)
        # print(model)

        # Create a dummy input with the specified size (batch_size, channels, depth, height, width)
        dummy_input = torch.randn(1, 1, 100, 100, 100)

        # Perform a forward pass with the dummy input
        output = model(dummy_input)
        print(output)


if __name__ == "__main__":
    unittest.main()
