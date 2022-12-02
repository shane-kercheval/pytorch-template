import torch.nn as nn
from source.domain.pytorch_wrappers import FullyConnectedNN


def test_FullyConnectedNN_builds_correct_layers(dummy_x_y):
    input_size = 10
    output_size = 3
    model = FullyConnectedNN(input_size=input_size, output_size=output_size, loss_func=nn.MSELoss)
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
