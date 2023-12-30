"""NN architectures implemented in PyTorch."""

import torch
from torch import nn


class FullyConnectedNN(nn.Module):
    """
    Implements a feed-forward fully connected PyTorch neural network. The number of
    hidden layers is dynamic and based on the size of the `hidden_units` parameter, which is a list
    that indicates the size of each hidden layer.
    """

    def __init__(
            self,
            input_size: int,
            hidden_layers: list[int],
            output_size: int):
        """
        Args:
            input_size: The number of input features.
            hidden_layers: A list of integers indicating the size of each hidden layer. At least
                one hidden layer is required (i.e. list of one item).
            output_size: The number of output features.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Flatten())
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(input_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return x


class ConvNet2L(nn.Module):
    """Convolutional neural network (two convolutional layers)."""

    def __init__(self, kernel_0: int, kernel_1: int, classes: int = 10):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernel_0, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(kernel_0, kernel_1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernel_1, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)
