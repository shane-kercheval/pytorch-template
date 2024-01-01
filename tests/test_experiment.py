"""Test the experiment module."""
import torch
from torch import nn
from source.domain.architectures import FullyConnectedNN, ConvNet2L
from source.domain.experiment import train, evaluate, make_model, make_optimizer, make_loader


def test__make_objects__fc__sgd__cpu(mnist_fc):  # noqa
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train, x_val, _, y_train, y_val, _ = mnist_fc
    train_loader = make_loader(x_train, y_train, batch_size=55)
    validation_loader = make_loader(x_val, y_val, batch_size=55)
    model = make_model(
        architecture='FC',
        input_size=28*28,
        layers=[64, 32],
        out_channels=None,  # for cnn
        kernel_sizes=None,  # for cnn
        device=device,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer_creator = make_optimizer(optimizer='sgd', model=model)
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
    assert model.layers[1].weight.device.type == device
    # ensure model can do forward pass given the correct dataset (fc)
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(device)
    y_pred = model(x_batch)
    assert list(y_pred.shape) == [len(y_batch), expected_output_size]

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train, x_val, _, y_train, y_val, _ = mnist_cnn
    train_loader = make_loader(x_train, y_train, batch_size=55)
    validation_loader = make_loader(x_val, y_val, batch_size=55)
    model = make_model(
        architecture='CNN',
        layers=None,  # for fc
        input_size=28*28,
        out_channels=[8, 16],
        kernel_sizes=[3, 7],
        device=device,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer_creator = make_optimizer(optimizer='adam', model=model)
    # test model
    expected_output_size = 10
    assert isinstance(model, ConvNet2L)
    expected_sizes = [(1, 8, 3, 3), (8, 16, 7, 7), (784, expected_output_size)]
    actual_sizes = [
        (model.layer1[0].weight.shape[1], model.layer1[0].weight.shape[0], model.layer1[0].kernel_size[0], model.layer1[0].kernel_size[1]),  # noqa
        (model.layer2[0].weight.shape[1], model.layer2[0].weight.shape[0], model.layer2[0].kernel_size[0], model.layer2[0].kernel_size[1]),  # noqa
        (model.fc.weight.shape[1], model.fc.weight.shape[0]),
    ]
    assert expected_sizes == actual_sizes
    # check that the model was sent to the correct device
    assert model.layer1[0].weight.device.type == device
    # ensure model can do forward pass given the correct dataset (cnn)
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(device)
    y_pred = model(x_batch)
    assert list(y_pred.shape) == [len(y_batch), expected_output_size]

    # test loaders
    x_batch, y_batch = next(iter(train_loader))
    assert list(x_batch.size()) == [55, 1, 28, 28]
    assert len(y_batch) == 55
    x_batch, y_batch = next(iter(validation_loader))
    assert list(x_batch.size()) == [55, 1, 28, 28]
    assert len(y_batch) == 55

    # test criterion
    assert isinstance(criterion, nn.CrossEntropyLoss)

    # test optimizer
    optimizer = optimizer_creator(lr=0.01)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults['lr'] == 0.01


def test__train__fc(mnist_fc):  # noqa
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train, x_val, x_test, y_train, y_val, y_test = mnist_fc
    train_loader = make_loader(x_train, y_train, batch_size=128)
    validation_loader = make_loader(x_val, y_val, batch_size=128)
    model = make_model(
        architecture='FC',
        input_size=28*28,
        layers=[64],
        out_channels=None,  # for cnn
        kernel_sizes=None,  # for cnn
        device=device,
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train, x_val, x_test, y_train, y_val, y_test = mnist_cnn
    train_loader = make_loader(x_train, y_train, batch_size=128)
    validation_loader = make_loader(x_val, y_val, batch_size=128)
    model = make_model(
        architecture='CNN',
        input_size=28*28,
        layers=None,  # for fc
        out_channels=[8, 16],
        kernel_sizes=[3, 7],
        device=device,
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
