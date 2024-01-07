"""NN architectures implemented in PyTorch."""

from enum import Enum
import inspect
import torch
from torch import nn


class Architecture(Enum):
    """Available architectures."""

    FULLY_CONNECTED = 'fully_connected'
    CONVOLUTIONAL = 'convolutional'

    @staticmethod
    def to_enum(architecture: str) -> 'Architecture':
        """Get an architecture from a string."""
        if isinstance(architecture, Architecture):
            return architecture
        for enum_value in Architecture:
            if enum_value.value.lower() == architecture.lower():
                return enum_value
        raise ValueError(f"{architecture} is not a valid Architecture")


class ModelRegistry:
    """Registry for models."""

    def __init__(self):
        self._registry: dict[str, nn.Module] = {}

    def register(self, name: str, cls: nn.Module) -> None:
        """Register a model with the registry."""
        if name in self._registry:
            raise ValueError(f"A model with name '{name}' is already registered.")
        self._registry[name] = cls

    def get_parameters(self, architecture: Architecture) -> list:
        """Get the list of parameters of a function."""
        return _get_parameters(self._registry[architecture].__init__)

    def create_instance(
            self,
            architecture: Architecture,
            data_dimensions: tuple[int, int],
            in_channels: int,
            output_size: int,
            model_parameters: dict,
            ) -> nn.Module:
        """Create an instance of a model."""
        if architecture not in self._registry:
            raise ValueError(f"Model '{architecture}' not found in registry.")
        # parameter for fully connected models
        model_parameters['input_size'] = data_dimensions[0] * data_dimensions[1] * in_channels
        # parameter for convolutional models
        model_parameters['dimensions'] = data_dimensions
        # parameter for all models
        model_parameters['output_size'] = output_size
        model_parameters = {
            k: v for k, v in model_parameters.items()
            if k in self.get_parameters(architecture)
        }
        return self._registry[architecture](**model_parameters)

    def list_models(self) -> dict[str, nn.Module]:
        """List all registered models."""
        return self._registry

    def __contains__(self, model_name: str) -> bool:
        """Check if a model is registered."""
        return model_name in self._registry


def register_model(architecture: Architecture) -> callable:
    """Decorator to register a model with a custom architecture."""
    def decorator(cls: nn.Module) -> nn.Module:
        assert issubclass(cls, nn.Module), \
            f"Model '{architecture}' ({cls.__name__}) must extend Pytorch model"
        assert (architecture not in MODEL_REGISTRY), \
            f"Model architecture '{architecture}' already registered."
        MODEL_REGISTRY.register(architecture, cls)
        return cls
    return decorator


def _get_parameters(func: callable) -> list:
    """Get the list of parameters of a function."""
    return [k for k in inspect.signature(func).parameters if k != 'self']


MODEL_REGISTRY = ModelRegistry()


@register_model(Architecture.FULLY_CONNECTED)
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


def calculate_same_padding(kernel_size: int, stride: int) -> int:
    """Calculate padding size to achieve 'same' padding, considering stride."""
    return ((kernel_size - 1) * stride) // 2


@register_model(Architecture.CONVOLUTIONAL)
class ConvolutionalNet(nn.Module):
    """
    Convolutional neural network with a dynamic number of convolutional layers.

    For example:

    ```
    ConvNetDynamic(
        dimensions=(28, 28),
        output_size=10,
        out_channels=(8, 16, 32),
        kernel_sizes=(5, 5, 3),
        input_channels=1
    )
    ```

    The `out_channels` parameter is a tuple specifying the number of filters for each convolutional
    layer.
    The `kernel_sizes` parameter is a tuple specifying the kernel size for each convolutional
    layer.
    The length of these tuples determines the number of convolutional layers in the network.

    The out_channels parameter specifies the number of filters the
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
            output_size: int,
            out_channels: tuple[int, ...],
            kernel_sizes: tuple[int, ...],
            input_channels: int = 1,
            use_batch_norm: bool = False,
            conv_dropout_p: float | None = None,
            fc_dropout_p: float | None = None,
            activation_type: str = 'relu',
            include_second_fc_layer: bool = False,
            ):
        super().__init__()
        if len(out_channels) != len(kernel_sizes):
            raise ValueError("out_channels and kernel_sizes must be of the same length")

        self.convs = nn.ModuleList()
        in_channels = input_channels

        for c, k in zip(out_channels, kernel_sizes):
            padding = calculate_same_padding(k, stride=1)
            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=c,
                    kernel_size=k,
                    stride=1,
                    padding=padding,
                ),
            ]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(c))
            layers.append(self._get_activation(activation_type))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if conv_dropout_p:
                layers.append(nn.Dropout2d(p=conv_dropout_p))

            self.convs.append(nn.Sequential(*layers))
            in_channels = c

        fc_input_size = self._get_linear_input_size(dimensions, input_channels)
        fc_layers = [nn.Flatten()]
        if include_second_fc_layer:
            second_fc_size = fc_input_size // 2
            fc_layers.extend([
                nn.Linear(fc_input_size, second_fc_size),
                self._get_activation(activation_type),
            ])
            if fc_dropout_p:
                fc_layers.append(nn.Dropout(p=fc_dropout_p))
            fc_layers.append(nn.Linear(second_fc_size, output_size))
        else:
            fc_layers.append(nn.Linear(fc_input_size, output_size))
        self.fc = nn.Sequential(*fc_layers)

    def _get_linear_input_size(self, dimensions: tuple[int, int], input_channels: int) -> int:
        """Calculate the input size of the linear layer."""
        dummy_input = torch.zeros(1, input_channels, *dimensions)
        with torch.no_grad():
            for conv in self.convs:
                dummy_input = conv(dummy_input)
        return nn.Flatten()(dummy_input).size(1)

    @staticmethod
    def _get_activation(activation_type: str) -> nn.Module:
        if activation_type == 'leaky_relu':
            return nn.LeakyReLU()
        if activation_type == 'relu':
            return nn.ReLU()
        raise ValueError(f'Unknown activation type: {activation_type}')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for conv in self.convs:
            x = conv(x)
        return self.fc(x)
