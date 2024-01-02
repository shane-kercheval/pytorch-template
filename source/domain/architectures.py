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


# class ConvNet2L(nn.Module):
#     """Convolutional neural network (two convolutional layers)."""

#     def __init__(self, kernel_0: int, kernel_1: int, classes: int = 10):
#         super().__init__()

#         self.layer1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=kernel_0,
#                 kernel_size=5,
#                 stride=1,
#                 padding=2,
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(kernel_0, kernel_1, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc = nn.Linear(7 * 7 * kernel_1, classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass."""
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.reshape(out.size(0), -1)
#         return self.fc(out)


def calculate_same_padding(kernel_size: int, stride: int) -> int:
    """Calculate padding size to achieve 'same' padding, considering stride."""
    return ((kernel_size - 1) * stride) // 2


class ConvNet2L(nn.Module):
    """
    Convolutional neural network with two convolutional layers e.g.
    `ConvNet2L(l1_filters=8, l2_filters=16, l1_kernel_size=5, l2_kernel_size=5, classes=10)`.

    The `filters` (i.e. out_channels in Pyotorch) parameter specifies the number of filters the
    convolutional layer will use. This determines the depth of the output feature map, i.e., how
    many distinct feature maps will be produced by the layer. Using more filters in deeper layers
    of a convolutional neural network (CNN) is a common practice. In a CNN, the initial layers
    often capture basic, low-level features such as edges and simple textures. As we move deeper
    into the network, the layers are designed to capture more complex and abstract features. By
    increasing the number of filters, we enable the network to create a richer, more diverse set
    of these high-level features, which are crucial for tasks like image classification, object
    detection, etc.

    The kernel_size parameter in a convolutional layer (nn.Conv2d) refers to the size of the
    convolutional filter or kernel. This size is typically a small number like 3, 5, or 7. The
    kernel size determines how many input values are considered simultaneously in each convolution
    operation. It is unrelated to the number of output channels (filters).

    Using the same kernel size in multiple layers can provide consistency in how features are
    extracted at different levels. For instance, 3x3 kernels are very common in popular
    architectures like VGG networks.

    Alternatively, varying the kernel size can help the network capture features at different
    scales. A smaller kernel in the first layer can capture fine details, while a larger kernel
    in deeper layers might capture more global, abstract features.

    Smaller kernels generally result in lower computational costs. So, using smaller kernels in
    initial layers (where the spatial dimensions are larger) and larger kernels in deeper layers
    (where spatial dimensions are reduced due to pooling) can be computationally efficient.
    """

    def __init__(
            self,
            dimensions: tuple[int, int],
            l1_out_channels: int,
            l2_out_channels: int,
            l1_kernel_size: int,
            l2_kernel_size: int,
            classes: int):
        super().__init__()
        stride = 1
        pool_kernel_size = 2
        pool_stride = 2

        padding_1 = calculate_same_padding(l1_kernel_size, stride=stride)
        padding_2 = calculate_same_padding(l2_kernel_size, stride=stride)
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=l1_out_channels,
                kernel_size=l1_kernel_size,
                stride=stride,
                padding=padding_1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=l1_out_channels,
                out_channels=l2_out_channels,
                kernel_size=l2_kernel_size,
                stride=stride,
                padding=padding_2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self._get_linear_input_size(dimensions), classes)

    def _get_linear_input_size(self, dimensions: tuple[int, int]) -> int:
        dummy_input = torch.zeros(1, 1, dimensions[0], dimensions[1])
        with torch.no_grad():
            dummy_output = self.layer1(dummy_input)
            dummy_output = self.layer2(dummy_output)
            dummy_output = self.flatten(dummy_output)
        return dummy_output.size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        return self.fc(out)
