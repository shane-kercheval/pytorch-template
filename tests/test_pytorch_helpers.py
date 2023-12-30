"""Test the pytorch_wrappers module."""

import numpy as np
from torch import nn
from torch import optim
from source.domain.pytorch_helpers import PyTorchTrainer
from source.domain.architectures import FullyConnectedNN


def assert_states_are_same(state_a: dict, state_b: dict) -> None:  # noqa: D103
    assert all(x == y for x, y in zip(state_a.keys(), state_b.keys()))
    assert all((x == y).all() for x, y in zip(state_a.values(), state_b.values()))


def test_FullyConnectedNN_builds_correct_layers():  # noqa
    input_size = 10
    output_size = 3
    model = FullyConnectedNN(
        input_size=input_size,
        hidden_layers=[50],
        output_size=output_size,
    )
    assert isinstance(model.layers[0], nn.Flatten)
    assert isinstance(model.layers[1], nn.modules.linear.Linear)
    assert model.layers[1].in_features == input_size
    assert model.layers[1].out_features == 50
    assert isinstance(model.layers[2], nn.modules.ReLU)
    assert isinstance(model.layers[3], nn.modules.linear.Linear)
    assert model.layers[3].in_features == 50
    assert model.layers[3].out_features == output_size

    input_size = 100
    output_size = 10
    hidden_layers = [60, 30]
    model = FullyConnectedNN(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
    )
    assert isinstance(model.layers[0], nn.Flatten)
    assert isinstance(model.layers[1], nn.modules.linear.Linear)
    assert model.layers[1].in_features == input_size
    assert model.layers[1].out_features == hidden_layers[0]
    assert isinstance(model.layers[2], nn.modules.ReLU)
    assert isinstance(model.layers[3], nn.modules.linear.Linear)
    assert model.layers[3].in_features == hidden_layers[0]
    assert model.layers[3].out_features == hidden_layers[1]
    assert isinstance(model.layers[4], nn.modules.ReLU)
    assert isinstance(model.layers[5], nn.modules.linear.Linear)
    assert model.layers[5].in_features == hidden_layers[1]
    assert model.layers[5].out_features == output_size


def test_FullyConnectedNN_training(dummy_x_y):  # noqa
    x, y = dummy_x_y
    model = FullyConnectedNN(
        input_size=x.shape[1],
        hidden_layers=[8],
        output_size=1,
    )
    trainer = PyTorchTrainer(
        model=model,
        loss_func=nn.MSELoss(),
        early_stopping_patience=5,
        verbose=False,
    )
    train_loss, validation_loss = trainer.train(
        x=x,
        y=y,
        epochs=10,
        optimizer=optim.Adam(model.parameters(), lr=0.001),
    )
    assert len(train_loss) == trainer.early_stopping._index + 1
    assert len(validation_loss) == trainer.early_stopping._index + 1
    assert (~np.isnan(train_loss)).all()
    assert (~np.isnan(validation_loss)).all()
    lowest_loss = trainer.best_validation_loss
    assert lowest_loss == np.array(validation_loss).min()
    best_state_dict = model.state_dict()

    # train again, but with learning rate so high that we won't improve the loss
    # make sure we retain the same lowest loss and state dict
    train_loss_new, validation_loss_new = trainer.train(
        x=x,
        y=y,
        epochs=10,
        optimizer=optim.Adam(model.parameters(), lr=1000),
    )
    assert len(train_loss) + len(train_loss_new) == trainer.early_stopping._index + 1
    assert len(validation_loss) + len(validation_loss_new) == trainer.early_stopping._index + 1
    assert (~np.isnan(train_loss_new)).all()
    assert (~np.isnan(validation_loss_new)).all()
    # lowest loss should not have changed
    assert trainer.best_validation_loss == lowest_loss
    assert_states_are_same(model.state_dict(), best_state_dict)
    assert_states_are_same(trainer.early_stopping.best_state, best_state_dict)


def test_FullyConnectedNN_training_no_early_stopping(dummy_x_y):  # noqa
    x, y = dummy_x_y
    model = FullyConnectedNN(
        input_size=x.shape[1],
        hidden_layers=[8],
        output_size=1,
    )
    trainer = PyTorchTrainer(
        model=model,
        loss_func=nn.MSELoss(),
        early_stopping_patience=None,
    )
    train_loss, validation_loss = trainer.train(
        x=x,
        y=y,
        optimizer=optim.Adam(model.parameters(), lr=0.001),
    )
    assert len(train_loss) > 0
    assert len(validation_loss) == len(train_loss)
    assert (~np.isnan(train_loss)).all()
    assert (~np.isnan(validation_loss)).all()


def test_FullyConnectedNN_training_verbose(dummy_x_y):  # noqa
    x, y = dummy_x_y
    model = FullyConnectedNN(
        input_size=x.shape[1],
        hidden_layers=[8],
        output_size=1,
    )
    trainer = PyTorchTrainer(
        model=model,
        loss_func=nn.MSELoss(),
        validation_size=0.1,
        verbose=True,
    )
    train_loss, validation_loss = trainer.train(
        x=x,
        y=y,
        optimizer=optim.Adam(model.parameters(), lr=0.001),
    )
    assert len(train_loss) > 0
    assert len(train_loss) == trainer.early_stopping._index + 1
    assert len(validation_loss) == trainer.early_stopping._index + 1
    assert (~np.isnan(train_loss)).all()
    assert (~np.isnan(validation_loss)).all()
