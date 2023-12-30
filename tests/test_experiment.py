"""Test the experiment module."""
import torch
from torch import nn
from source.domain.architectures import FullyConnectedNN, ConvNet2L
from source.domain.experiment import make_objects


def test_make_objects__fc__sgd__cpu(mnist_fc):  # noqa
    model, train_loader, validation_loader, test_loader, criterion, optimizer_creator = \
        make_objects(
            *mnist_fc,
            model_type='FC',
            batch_size=55,
            kernels=None,
            layers=[64, 32],
            optimizer='sgd',
            device='cpu',
        )
    # test model
    expected_output_size = 10
    assert isinstance(model, FullyConnectedNN)
    expected_sizes = [(784, 64), (64, 32), (32, expected_output_size)]
    actual_sizes = [
        (layer.in_features, layer.out_features)
        for layer in model.layers if isinstance(layer, nn.Linear)
    ]
    assert expected_sizes == actual_sizes
    # check that the model was sent to the correct device
    assert model.layers[1].weight.device.type == 'cpu'
    # ensure model can do forward pass given the correct dataset (fc)
    x_batch, y_batch = next(iter(train_loader))
    y_pred = model(x_batch)
    assert list(y_pred.shape) == [len(y_batch), expected_output_size]

    # test loaders
    x_batch, y_batch = next(iter(train_loader))
    assert list(x_batch.size()) == [55, 784]
    assert len(y_batch) == 55
    x_batch, y_batch = next(iter(validation_loader))
    assert list(x_batch.size()) == [55, 784]
    assert len(y_batch) == 55
    x_batch, y_batch = next(iter(test_loader))
    assert list(x_batch.size()) == [55, 784]
    assert len(y_batch) == 55

    # test criterion
    assert isinstance(criterion, nn.CrossEntropyLoss)

    # test optimizer
    optimizer = optimizer_creator(lr=0.01)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == 0.01


def test_make_objects__cnn__adam__cuda(mnist_cnn):  # noqa
    model, train_loader, validation_loader, test_loader, criterion, optimizer_creator = \
        make_objects(
            *mnist_cnn,
            model_type='CNN',
            batch_size=55,
            kernels=[8, 16],
            layers=None,
            optimizer='adam',
            device='cuda',
        )
    # test model
    expected_output_size = 10
    assert isinstance(model, ConvNet2L)
    expected_sizes = [(1, 8, 5, 5), (8, 16, 5, 5), (784, expected_output_size)]
    actual_sizes = [
        (model.layer1[0].weight.shape[1], model.layer1[0].weight.shape[0], model.layer1[0].kernel_size[0], model.layer1[0].kernel_size[1]),  # noqa
        (model.layer2[0].weight.shape[1], model.layer2[0].weight.shape[0], model.layer2[0].kernel_size[0], model.layer2[0].kernel_size[1]),  # noqa
        (model.fc.weight.shape[1], model.fc.weight.shape[0]),
    ]
    assert expected_sizes == actual_sizes
    # check that the model was sent to the correct device
    assert model.layer1[0].weight.device.type == 'cuda'
    # ensure model can do forward pass given the correct dataset (cnn)
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to('cuda')
    y_pred = model(x_batch)
    assert list(y_pred.shape) == [len(y_batch), expected_output_size]

    # test loaders
    x_batch, y_batch = next(iter(train_loader))
    assert list(x_batch.size()) == [55, 1, 28, 28]
    assert len(y_batch) == 55
    x_batch, y_batch = next(iter(validation_loader))
    assert list(x_batch.size()) == [55, 1, 28, 28]
    assert len(y_batch) == 55
    x_batch, y_batch = next(iter(test_loader))
    assert list(x_batch.size()) == [55, 1, 28, 28]
    assert len(y_batch) == 55

    # test criterion
    assert isinstance(criterion, nn.CrossEntropyLoss)

    # test optimizer
    optimizer = optimizer_creator(lr=0.01)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults['lr'] == 0.01
