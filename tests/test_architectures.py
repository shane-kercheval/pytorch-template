"""Tests for the architectures module."""

import pytest
import torch
from torch import nn
from source.library.architectures import Architecture, ConvNet2L, MODEL_REGISTRY, ModelRegistry


class ExampleModel(nn.Module):
    """Example model for testing."""

    def __init__(self, foo: str | None = None):
        self.foo = foo


@pytest.mark.parametrize('input_channels', [1, 3])
@pytest.mark.parametrize('image_size', [(28, 28), (20, 100)])
@pytest.mark.parametrize('out_channels', [(16, 32), (32, 64)])
@pytest.mark.parametrize('kernel_sizes', [(3, 5), (5, 3)])
@pytest.mark.parametrize('num_classes', [5, 10])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('conv_dropout_p', [None, 0, 0.5])
@pytest.mark.parametrize('fc_dropout_p', [None, 0, 0.5])
@pytest.mark.parametrize('activation_type', ['relu', 'leaky_relu'])
@pytest.mark.parametrize('include_second_fc', [True, False])
def test_convnet2l_initialization(  # noqa
        input_channels: int,
        image_size: tuple,
        out_channels: tuple,
        kernel_sizes: tuple,
        num_classes: int,
        use_batch_norm: bool,
        conv_dropout_p: float | None,
        fc_dropout_p: float | None,
        activation_type: str,
        include_second_fc: bool):
    model = ConvNet2L(
        dimensions=image_size,
        input_channels=input_channels,
        out_channels=out_channels,
        kernel_sizes=kernel_sizes,
        output_size=num_classes,
        use_batch_norm=use_batch_norm,
        conv_dropout_p=conv_dropout_p,
        fc_dropout_p=fc_dropout_p,
        activation_type=activation_type,
        include_second_fc_layer=include_second_fc,
    )
    assert model.layer1 is not None
    assert model.layer2 is not None
    assert model.fc is not None
    assert isinstance(model.layer1[0], nn.Conv2d)
    assert model.layer1[0].in_channels == input_channels
    assert model.layer1[0].out_channels == out_channels[0]
    if use_batch_norm:
        next_index = 2
        assert isinstance(model.layer1[1], nn.BatchNorm2d)
    else:
        next_index = 1
    assert isinstance(model.layer1[next_index], type(ConvNet2L._get_activation(activation_type)))
    assert isinstance(model.layer1[next_index + 1], nn.MaxPool2d)
    if conv_dropout_p:
        assert isinstance(model.layer1[next_index + 2], nn.Dropout2d)
        assert model.layer1[next_index + 2].p == conv_dropout_p
    else:
        assert not any(isinstance(layer, nn.Dropout2d) for layer in model.layer1)

    assert isinstance(model.layer2[0], nn.Conv2d)
    assert model.layer2[0].in_channels == out_channels[0]
    assert model.layer2[0].out_channels == out_channels[1]
    if use_batch_norm:
        next_index = 2
        assert isinstance(model.layer2[1], nn.BatchNorm2d)
    else:
        next_index = 1
    assert isinstance(model.layer2[next_index], type(ConvNet2L._get_activation(activation_type)))
    assert isinstance(model.layer2[next_index + 1], nn.MaxPool2d)
    if conv_dropout_p:
        assert isinstance(model.layer2[next_index + 2], nn.Dropout2d)
        assert model.layer2[next_index + 2].p == conv_dropout_p
    else:
        assert not any(isinstance(layer, nn.Dropout2d) for layer in model.layer2)

    assert isinstance(model.fc, nn.Sequential)
    assert isinstance(model.fc[0], nn.Flatten)
    if include_second_fc:
        assert isinstance(model.fc[1], nn.Linear)
        assert isinstance(model.fc[2], type(ConvNet2L._get_activation(activation_type)))
        if fc_dropout_p:
            assert isinstance(model.fc[3], nn.Dropout)
            assert model.fc[3].p == fc_dropout_p
            assert isinstance(model.fc[4], nn.Linear)
            assert model.fc[4].out_features == num_classes
        else:
            assert isinstance(model.fc[3], nn.Linear)
            assert model.fc[3].out_features == num_classes
            assert not any(isinstance(layer, nn.Dropout) for layer in model.fc)
    else:
        assert isinstance(model.fc[1], nn.Linear)
        assert model.fc[1].out_features == num_classes

    with torch.no_grad():
        out = model(torch.randn(20, input_channels, *image_size))
    assert out.size() == (20, num_classes)


def test_get_model_parameters():  # noqa
    params = MODEL_REGISTRY.get_parameters(architecture=Architecture.FULLY_CONNECTED)
    assert 'self' not in params
    assert 'input_size' in params
    assert 'output_size' in params
    assert 'hidden_layers' in params
    params = MODEL_REGISTRY.get_parameters(architecture=Architecture.CONVOLUTIONAL)
    assert 'self' not in params
    assert 'dimensions' in params
    assert 'output_size' in params
    assert 'out_channels' in params
    assert 'kernel_sizes' in params


@pytest.fixture()
def registry() -> ModelRegistry:
    """Fixture to provide a fresh instance of ModelRegistry for each test."""
    return ModelRegistry()


def test_register_model_success(registry):  # noqa
    """Test successful registration of a model."""
    registry.register('TestModel', ExampleModel)
    assert 'TestModel' in registry.list_models()


def test_register_model_duplicate(registry):  # noqa
    """Test registering a model with a duplicate name raises an error."""
    registry.register('TestModel', ExampleModel)
    with pytest.raises(ValueError):  # noqa
        registry.register('TestModel', ExampleModel)


def test_create_instance_success(registry):  # noqa
    """Test successful creation of a model instance."""
    registry.register('TestModel', ExampleModel)
    instance = registry.create_instance(
        'TestModel', input_size=28*28, output_size=10, model_parameters={},
    )
    assert isinstance(instance, ExampleModel)


# test with kwargs
def test_create_instance_kwargs(registry):  # noqa
    """Test that kwargs are passed to model constructor."""
    registry.register('TestModel', ExampleModel)
    instance = registry.create_instance(
        'TestModel', input_size=28*28, output_size=10, model_parameters={'foo': 'bar'},
    )
    assert instance.foo == 'bar'


def test_create_instance_unregistered(registry):  # noqa
    """Test error when creating an instance of an unregistered model."""
    with pytest.raises(ValueError):  # noqa
        registry.create_instance('UnregisteredModel', input_size=28*28, output_size=10, model_parameters={})  # noqa


def test_list_models(registry):  # noqa
    """Test listing all registered models."""
    registry.register('Model1', ExampleModel)
    registry.register('Model2', ExampleModel)
    models = registry.list_models()
    assert 'Model1' in models
    assert 'Model2' in models
    assert len(models) == 2
