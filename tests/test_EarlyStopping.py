import logging
import pytest
import torch.nn as nn
import numpy as np

from source.domain.pytorch_wrappers import EarlyStopping, FullyConnectedNN
from tests.helpers import get_test_file_path


logging.config.fileConfig(get_test_file_path("logging/test_logging.conf"),
                          defaults={'logfilename': get_test_file_path("logging/log.log")},
                          disable_existing_loggers=False)


def copy_state(state: dict):
    return {k: np.array(v).copy() for k, v in state.items()}


def assert_states_are_same(state_a, state_b):
    assert all([x == y for x, y in zip(state_a.keys(), state_b.keys())])
    assert all([(x == y).all() for x, y in zip(state_a.values(), state_b.values())])


def assert_state_values_are_different(state_a, state_b):
    assert all([x == y for x, y in zip(state_a.keys(), state_b.keys())])
    assert all([(x != y).any() for x, y in zip(state_a.values(), state_b.values())])


def test_early_stopping_standard_usecase(dummy_x_y):
    """
    This function tests the EarlyStopping class outside of the PyTorchNN classes so that we
    can control the validation loss values that are used.
    """
    X, y = dummy_x_y
    test_model = FullyConnectedNN(
        input_size=2,
        output_size=1,
        hidden_units=[8, 4],
        loss_func=nn.MSELoss(),
        learning_rate=0.001,
        early_stopping_patience=None
    )
    # capture original state of the model to make sure best_state gets updated
    original_state = copy_state(test_model._model.state_dict())
    early_stopping = EarlyStopping(model=test_model._model, verbose=False)
    assert not early_stopping.early_stop
    assert early_stopping._counter == 0
    assert early_stopping.min_validation_loss is np.NaN
    # assert that the default best_state is equal to the original state of the model passed in
    current_state = copy_state(early_stopping.best_state)
    assert_states_are_same(original_state, current_state)

    _, validation_loss = test_model.train(
        X=X, y=y, num_epochs=1, validation_size=0.1, batch_size=2, random_seed=3
    )

    early_stopping(validation_loss=validation_loss[0])
    assert not early_stopping.early_stop
    assert early_stopping._counter == 0
    assert early_stopping.best_index == 0
    assert early_stopping.min_validation_loss == validation_loss[0]

    # the state should have been updated since we have a lower validation score
    current_state = copy_state(early_stopping.best_state)
    assert_state_values_are_different(original_state, current_state)

    model_state = copy_state(test_model._model.state_dict())
    assert_states_are_same(model_state, current_state)

    for index in range(6):
        early_stopping(validation_loss=validation_loss[0] + 1)
        assert not early_stopping.early_stop
        assert early_stopping._counter == index + 1
        assert early_stopping.best_index == 0
        assert early_stopping.min_validation_loss == validation_loss[0]

    early_stopping(validation_loss=validation_loss[0] + 1)
    assert early_stopping.early_stop
    assert early_stopping._counter == 7
    assert early_stopping.best_index == 0
    assert early_stopping.min_validation_loss == validation_loss[0]

    # this should fail, we should not be able to keep calling early_stopping if we are already
    # past the point where we should have stopped
    with pytest.raises(AssertionError):
        early_stopping(validation_loss=validation_loss[0] + 1)


def test_early_stopping_within_pytorch_wrapper(dummy_x_y):
    X, y = dummy_x_y
    test_model = FullyConnectedNN(
        input_size=2,
        output_size=1,
        hidden_units=[8, 4],
        loss_func=nn.MSELoss(),
        learning_rate=0.001,
        # early_stopping_patience=None
    )
    assert test_model._early_stopping is None
    # capture original state of the model to make sure best_state gets updated
    original_state = copy_state(test_model._model.state_dict())

    # this validation loss will be counted as lower since it is the first
    training_loss, validation_loss = test_model.train(
        X=X, y=y, num_epochs=1000, validation_size=0.2, batch_size=2, random_seed=3
    )
    assert test_model._early_stopping.early_stop
    assert test_model._early_stopping.best_index < 1000
    assert test_model._early_stopping.best_index + test_model._early_stopping_patience == test_model._early_stopping._index
    assert len(validation_loss) == test_model._early_stopping._index + 1
    assert len(training_loss) == len(validation_loss)
    assert not (validation_loss == 0).any()
    assert validation_loss.argmin() == test_model._early_stopping.best_index
    assert test_model._early_stopping._no_improvement_counter == test_model._early_stopping_patience
    # the state should have been updated since we have a lower validation score
    current_state = copy_state(test_model._early_stopping.best_state)
    assert_state_values_are_different(original_state, current_state)
    # ensure the saved best_state in the early_stopping object is the same as the model
    model_state = copy_state(test_model._model.state_dict())
    assert_states_are_same(model_state, current_state)
