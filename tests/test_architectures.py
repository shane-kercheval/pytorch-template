import pytest
import torch
import torch.nn as nn
from source.domain.architectures import ConvNet2L  # replace 'your_module' with the actual module name

# Define constants for tests
INPUT_CHANNELS = 1  # MNIST is grayscale, so 1 channel
MNIST_IMAGE_SIZE = (28, 28)  # Standard MNIST image size
NUM_CLASSES = 10  # MNIST has 10 classes (digits 0-9)



@pytest.mark.parametrize("input_channels", [1, 3])
@pytest.mark.parametrize("use_batch_norm", [True, False])
@pytest.mark.parametrize("dropout_p", [None, 0, 0.5])
@pytest.mark.parametrize("activation_type", ['relu', 'leaky_relu'])
@pytest.mark.parametrize("include_second_fc", [True, False])
def test_initialization(input_channels, use_batch_norm, dropout_p, activation_type, include_second_fc):
    model = ConvNet2L(
        dimensions=MNIST_IMAGE_SIZE,
        l1_out_channels=16,
        l2_out_channels=32,
        l1_kernel_size=3,
        l2_kernel_size=5,
        classes=NUM_CLASSES,
        input_channels=input_channels,
        use_batch_norm=use_batch_norm,
        dropout_p=dropout_p,
        activation_type=activation_type,
        include_second_fc=include_second_fc,
    )

    # Basic checks
    assert model.layer1 is not None
    assert model.layer2 is not None
    assert model.fc is not None

    # Specific checks
    assert isinstance(model.layer1[0], nn.Conv2d)
    assert model.layer1[0].in_channels == input_channels

    if use_batch_norm:
        assert any(isinstance(layer, nn.BatchNorm2d) for layer in model.layer1)

test that dropout of 0 does not add dropout layers

    if dropout_p is not None:
        assert any(isinstance(layer, nn.Dropout2d) for layer in model.layer1)

    if include_second_fc:
        assert len(list(model.fc.children())) > 1


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_forward_pass(model, batch_size):
    dummy_input = torch.randn(batch_size, INPUT_CHANNELS, *MNIST_IMAGE_SIZE)
    output = model(dummy_input)

    assert output is not None
    assert output.shape[0] == batch_size
    assert output.shape[1] == NUM_CLASSES



def test_output_shape(model):
    dummy_input = torch.randn(1, INPUT_CHANNELS, *MNIST_IMAGE_SIZE)

    # Check output shape at each layer
    out = model.layer1(dummy_input)
    assert out.shape[1] == model.layer1[0].out_channels

    out = model.layer2(out)
    assert out.shape[1] == model.layer2[0].out_channels

    out = model.fc(model.flatten(out))
    assert out.shape[1] == NUM_CLASSES



def test_activation_function(model):
    if isinstance(model.layer1[1], nn.ReLU):
        assert isinstance(model.layer1[1], nn.ReLU)
    elif isinstance(model.layer1[1], nn.LeakyReLU):
        assert isinstance(model.layer1[1], nn.LeakyReLU)
    else:
        raise AssertionError("Unknown activation function in layer 1")

    if isinstance(model.layer2[1], nn.ReLU):
        assert isinstance(model.layer2[1], nn.ReLU)
    elif isinstance(model.layer2[1], nn.LeakyReLU):
        assert isinstance(model.layer2[1], nn.LeakyReLU)
    else:
        raise AssertionError("Unknown activation function in layer 2")



def test_batch_normalization(model):
    use_batch_norm = any(isinstance(layer, nn.BatchNorm2d) for layer in model.layer1)
    assert use_batch_norm == model.use_batch_norm

    use_batch_norm = any(isinstance(layer, nn.BatchNorm2d) for layer in model.layer2)
    assert use_batch_norm == model.use_batch_norm



def test_dropout(model):
    dropout_present_layer1 = any(isinstance(layer, nn.Dropout2d) for layer in model.layer1)
    dropout_present_layer2 = any(isinstance(layer, nn.Dropout2d) for layer in model.layer2)

    if model.dropout_p:
        assert dropout_present_layer1
        assert dropout_present_layer2
    else:
        assert not dropout_present_layer1
        assert not dropout_present_layer2


def test_fully_connected_layers(model):
    fc_layers = list(model.fc.children())
    assert len(fc_layers) > 0  # At least one layer should be present

    if model.include_second_fc:
        assert len(fc_layers) >= 3  # More layers expected with a second FC layer
        assert isinstance(fc_layers[0], nn.Linear)
        assert isinstance(fc_layers[-1], nn.Linear)
    else:
        assert len(fc_layers) == 1
        assert isinstance(fc_layers[0], nn.Linear)
