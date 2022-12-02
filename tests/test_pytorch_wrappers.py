import numpy as np
import torch.nn as nn
from source.domain.pytorch_wrappers import FullyConnectedNN


def test_FullyConnectedNN_builds_correct_layers():
    input_size = 10
    output_size = 3
    model = FullyConnectedNN(
        input_size=input_size,
        output_size=output_size,
        loss_func=nn.MSELoss()
    )
    assert model.hidden_units == [50, 20]

    assert isinstance(model._model[0], nn.modules.linear.Linear)
    assert model._model[0].in_features == input_size
    assert model._model[0].out_features == model.hidden_units[0]

    assert isinstance(model._model[1], nn.modules.ReLU)

    assert isinstance(model._model[2], nn.modules.linear.Linear)
    assert model._model[2].in_features == model.hidden_units[0]
    assert model._model[2].out_features == model.hidden_units[1]

    assert isinstance(model._model[3], nn.modules.ReLU)

    assert isinstance(model._model[4], nn.modules.linear.Linear)
    assert model._model[4].in_features == model.hidden_units[1]
    assert model._model[4].out_features == output_size

    input_size = 100
    output_size = 10
    hidden_units = [60, 30]
    model = FullyConnectedNN(
        input_size=input_size,
        output_size=output_size,
        hidden_units=hidden_units,
        loss_func=nn.MSELoss
    )
    assert model.hidden_units == hidden_units

    assert isinstance(model._model[0], nn.modules.linear.Linear)
    assert model._model[0].in_features == input_size
    assert model._model[0].out_features == model.hidden_units[0]

    assert isinstance(model._model[1], nn.modules.ReLU)

    assert isinstance(model._model[2], nn.modules.linear.Linear)
    assert model._model[2].in_features == model.hidden_units[0]
    assert model._model[2].out_features == model.hidden_units[1]

    assert isinstance(model._model[3], nn.modules.ReLU)

    assert isinstance(model._model[4], nn.modules.linear.Linear)
    assert model._model[4].in_features == model.hidden_units[1]
    assert model._model[4].out_features == output_size


def test_FullyConnectedNN_training(dummy_x_y):
    X, y = dummy_x_y
    model = FullyConnectedNN(
        input_size=X.shape[1],
        output_size=1,
        loss_func=nn.MSELoss(),
        verbose=False,
    )
    train_loss, validation_loss = model.train(X=X, y=y)
    assert len(train_loss) == model._early_stopping._index + 1
    assert len(validation_loss) == model._early_stopping._index + 1


def test_FullyConnectedNN_training_no_early_stopping_no_validation(dummy_x_y):
    X, y = dummy_x_y
    model = FullyConnectedNN(
        input_size=X.shape[1],
        output_size=1,
        loss_func=nn.MSELoss(),
        early_stopping_patience=None,
        verbose=False,
    )
    train_loss, validation_loss = model.train(X=X, y=y, validation_size=0)
    assert len(train_loss) > 0
    assert len(validation_loss) == len(train_loss)
    assert np.isnan(validation_loss).all()


def test_FullyConnectedNN_training_verbose(dummy_x_y):
    X, y = dummy_x_y
    model = FullyConnectedNN(
        input_size=X.shape[1],
        output_size=1,
        loss_func=nn.MSELoss(),
        early_stopping_patience=None,
        verbose=True,
    )
    train_loss, validation_loss = model.train(X=X, y=y, validation_size=0)
    assert len(train_loss) > 0
    assert len(validation_loss) == len(train_loss)
    assert np.isnan(validation_loss).all()
