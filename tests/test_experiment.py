"""Test the experiment module."""
import torch
from torch import nn
from source.library.architectures import Architecture, FullyConnectedNN, ConvolutionalNet
from source.library.data import CHANNELS, DIMENSIONS, OUTPUT_SIZE
from source.library.experiment import (
    train,
    evaluate,
    make_model,
    make_optimizer,
    make_loader,
    get_available_device,
)


def test__make_objects__fc__sgd__cpu(mnist_fc):  # noqa
    device = get_available_device()
    x_train, x_val, _, y_train, y_val, _ = mnist_fc
    train_loader = make_loader(x_train, y_train, batch_size=55)
    validation_loader = make_loader(x_val, y_val, batch_size=55)
    model = make_model(
        data_dimensions=DIMENSIONS,
        in_channels=CHANNELS,
        output_size=OUTPUT_SIZE,
        config = {
            'architecture': Architecture.FULLY_CONNECTED,
            'hidden_layers': [64, 32],
            'device': device,
        },
    )
    criterion = nn.CrossEntropyLoss()
    optimizer_creator = make_optimizer(optimizer='sgd', model=model)
    # test model
    assert isinstance(model, FullyConnectedNN)
    expected_sizes = [(784, 64), (64, 32), (32, OUTPUT_SIZE)]
    actual_sizes = [
        (layer.in_features, layer.out_features)
        for layer in model.layers if isinstance(layer, nn.Linear)
    ]
    assert expected_sizes == actual_sizes
    # check that the model was sent to the correct device
    assert model.layers[1].weight.device.type == device
    # ensure model can do forward pass given the correct dataset (fc)
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(device)
    y_pred = model(x_batch)
    assert list(y_pred.shape) == [len(y_batch), OUTPUT_SIZE]

    # test loaders
    x_batch, y_batch = next(iter(train_loader))
    assert list(x_batch.size()) == [55, 784]
    assert len(y_batch) == 55
    x_batch, y_batch = next(iter(validation_loader))
    assert list(x_batch.size()) == [55, 784]
    assert len(y_batch) == 55

    # test criterion
    assert isinstance(criterion, nn.CrossEntropyLoss)

    # test optimizer
    optimizer = optimizer_creator(lr=0.01)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == 0.01


def test__make_objects__cnn__adam__cuda(mnist_cnn):  # noqa
    device = get_available_device()
    x_train, x_val, _, y_train, y_val, _ = mnist_cnn
    train_loader = make_loader(x_train, y_train, batch_size=55)
    validation_loader = make_loader(x_val, y_val, batch_size=55)
    model = make_model(
        data_dimensions=DIMENSIONS,
        in_channels=CHANNELS,
        output_size=OUTPUT_SIZE,
        config={
            'architecture': Architecture.CONVOLUTIONAL,
            'out_channels': [8, 16],
            'kernel_sizes': [3, 7],
            'device': device,
        },
    )
    criterion = nn.CrossEntropyLoss()
    optimizer_creator = make_optimizer(optimizer='adam', model=model)
    # test model
    assert isinstance(model, ConvolutionalNet)
    expected_sizes = [(1, 8, 3, 3), (8, 16, 7, 7), (784, OUTPUT_SIZE)]
    actual_sizes = [
        (model.convs[0][0].weight.shape[1], model.convs[0][0].weight.shape[0], model.convs[0][0].kernel_size[0], model.convs[0][0].kernel_size[1]),  # noqa
        (model.convs[1][0].weight.shape[1], model.convs[1][0].weight.shape[0], model.convs[1][0].kernel_size[0], model.convs[1][0].kernel_size[1]),  # noqa
        (model.fc[1].weight.shape[1], model.fc[1].weight.shape[0]),
    ]
    assert expected_sizes == actual_sizes
    # check that the model was sent to the correct device
    assert model.convs[0][0].weight.device.type == device
    # ensure model can do forward pass given the correct dataset (cnn)
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(device)
    y_pred = model(x_batch)
    assert list(y_pred.shape) == [len(y_batch), OUTPUT_SIZE]

    # test loaders
    x_batch, y_batch = next(iter(train_loader))
    assert list(x_batch.size()) == [55, CHANNELS, DIMENSIONS[0], DIMENSIONS[1]]
    assert len(y_batch) == 55
    x_batch, y_batch = next(iter(validation_loader))
    assert list(x_batch.size()) == [55, CHANNELS, DIMENSIONS[0], DIMENSIONS[1]]
    assert len(y_batch) == 55

    # test criterion
    assert isinstance(criterion, nn.CrossEntropyLoss)

    # test optimizer
    optimizer = optimizer_creator(lr=0.01)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults['lr'] == 0.01


def test__train__fc(mnist_fc):  # noqa
    device = get_available_device()
    x_train, x_val, x_test, y_train, y_val, y_test = mnist_fc
    train_loader = make_loader(x_train, y_train, batch_size=128)
    validation_loader = make_loader(x_val, y_val, batch_size=128)
    model = make_model(
        data_dimensions=DIMENSIONS,
        in_channels=CHANNELS,
        output_size=OUTPUT_SIZE,
        config={
            'architecture': Architecture.FULLY_CONNECTED,
            'hidden_layers': [64],
            'device': device,
        },
    )
    criterion = nn.CrossEntropyLoss()
    optimizer_creator = make_optimizer(optimizer='sgd', model=model)
    train(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer_creator=optimizer_creator,
        epochs=2,
        learning_rate=0.01,
        device=device,
        num_reduce_learning_rate=1,
        log_wandb=False,
    )
    evaluate(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        x_test=x_test,
        y_test=y_test,
        criterion=criterion,
        device=device,
        log_wandb=False,
    )

def test__train__cnn(mnist_cnn):  # noqa
    device = get_available_device()
    x_train, x_val, x_test, y_train, y_val, y_test = mnist_cnn
    train_loader = make_loader(x_train, y_train, batch_size=128)
    validation_loader = make_loader(x_val, y_val, batch_size=128)
    model = make_model(
        data_dimensions=DIMENSIONS,
        in_channels=CHANNELS,
        output_size=OUTPUT_SIZE,
        config={
            'architecture': Architecture.CONVOLUTIONAL,
            'out_channels': [8, 16],
            'kernel_sizes': [3, 7],
            'device': device,
        },
    )
    criterion = nn.CrossEntropyLoss()
    optimizer_creator = make_optimizer(optimizer='adam', model=model)
    train(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer_creator=optimizer_creator,
        epochs=2,
        learning_rate=0.01,
        device=device,
        num_reduce_learning_rate=1,
        log_wandb=False,
    )
    evaluate(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        x_test=x_test,
        y_test=y_test,
        criterion=criterion,
        device=device,
        log_wandb=False,
    )


def test__make_model__fc__default_config(default_fc_run_config, mnist_fc):  # noqa
    device = get_available_device()
    x_train, _, _, _, _, _ = mnist_fc
    default_fc_run_config['device'] = device
    model = make_model(
        data_dimensions=DIMENSIONS,
        in_channels=CHANNELS,
        output_size=OUTPUT_SIZE,
        config=default_fc_run_config,
    )
    # test forward pass
    out = model.forward(x_train[0:100].to(device))
    assert list(out.shape) == [100, OUTPUT_SIZE]

def test__make_model__cnn__default_config(default_cnn_run_config, mnist_cnn):  # noqa
    device = get_available_device()
    x_train, _, _, _, _, _ = mnist_cnn
    default_cnn_run_config['device'] = device
    model = make_model(
        data_dimensions=DIMENSIONS,
        in_channels=CHANNELS,
        output_size=OUTPUT_SIZE,
        config=default_cnn_run_config,
    )
    # test forward pass
    out = model.forward(x_train[0:100].to(device))
    assert list(out.shape) == [100, OUTPUT_SIZE]
