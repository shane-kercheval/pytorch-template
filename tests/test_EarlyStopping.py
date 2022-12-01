import logging
import pytest
import torch.nn as nn
import numpy as np

from source.domain.pytorch_wrappers import EarlyStopping, FullyConnectedNN
from tests.helpers import get_test_file_path


logging.config.fileConfig(
    get_test_file_path("logging/test_logging.conf"),
    defaults={'logfilename': get_test_file_path("logging/log.log")},
    disable_existing_loggers=False
)


def copy_state(state: dict):
    return {k: np.array(v).copy() for k, v in state.items()}


def assert_states_are_same(state_a, state_b):
    assert all([x == y for x, y in zip(state_a.keys(), state_b.keys())])
    assert all([(x == y).all() for x, y in zip(state_a.values(), state_b.values())])


def assert_state_values_are_different(state_a, state_b):
    assert all([x == y for x, y in zip(state_a.keys(), state_b.keys())])
    assert all([(x != y).any() for x, y in zip(state_a.values(), state_b.values())])


def test_early_stopping_delta_0():
    """
    This function tests the EarlyStopping class outside of the PyTorchNN classes so that we
    can control the validation loss values that are used.
    """

    class MockPytorchModel():
        def __init__(self) -> None:
            self._state = -1

        def state_dict(self) -> dict:
            self._state += 1
            return {'state': self._state}

    mock_model = MockPytorchModel()
    early_stopping = EarlyStopping(model=mock_model, verbose=False, delta=0)
    assert not early_stopping.early_stop
    assert early_stopping._counter == 0
    assert early_stopping.lowest_loss is np.Inf
    assert early_stopping.best_state == {'state': 0}
    assert early_stopping.best_index is None

    # initial validation will automatically be captured because it is less than the default of Inf
    early_stopping(validation_loss=100)
    assert not early_stopping.early_stop
    assert early_stopping._counter == 0
    assert early_stopping._index == 0
    assert early_stopping.best_index == 0
    assert early_stopping.lowest_loss == 100
    assert early_stopping.best_state == {'state': 1}

    # now we are going to test out a lower/better loss that is beyond our delta of 0
    # a value of 99.99 should be captured as a better loss and should not increment the early
    # stopping counter
    early_stopping(validation_loss=99.99)
    assert not early_stopping.early_stop
    assert early_stopping._counter == 0
    assert early_stopping._index == 1
    assert early_stopping.best_index == 1
    assert early_stopping.lowest_loss == 99.99
    assert early_stopping.best_state == {'state': 2}

    # should *not* be captured as a better loss and *should* increment the early stopping counter
    model_state = early_stopping._model.state_dict()  # mimic state transition
    assert model_state == {'state': 3}
    early_stopping(validation_loss=99.99)
    assert not early_stopping.early_stop
    assert early_stopping._counter == 1
    assert early_stopping._index == 2
    assert early_stopping.best_index == 1
    assert early_stopping.lowest_loss == 99.99
    assert early_stopping.best_state == {'state': 2}

    current_lowest_loss = 99.99
    for index in range(5):
        model_state = early_stopping._model.state_dict()  # mimic state transition
        assert model_state == {'state': 4 + index}
        early_stopping(validation_loss=current_lowest_loss)
        assert not early_stopping.early_stop
        assert early_stopping._counter == index + 2
        assert early_stopping._index == index + 3
        assert early_stopping.best_index == 1
        assert early_stopping.lowest_loss == current_lowest_loss
        assert early_stopping.best_state == {'state': 2}

    model_state = early_stopping._model.state_dict()  # mimic state transition
    assert model_state == {'state': 9}
    early_stopping(validation_loss=current_lowest_loss)
    assert early_stopping.early_stop
    assert early_stopping._counter == 7
    assert early_stopping._index == 8
    assert early_stopping.best_index == 1
    assert early_stopping.lowest_loss == current_lowest_loss
    assert early_stopping.best_state == {'state': 2}

    # this should fail, we should not be able to keep calling early_stopping if we are already
    # past the point where we should have stopped
    with pytest.raises(AssertionError):
        early_stopping(validation_loss=current_lowest_loss - 1)


def test_early_stopping_delta_1():
    """
    This function tests the EarlyStopping class outside of the PyTorchNN classes so that we
    can control the validation loss values that are used.
    """

    class MockPytorchModel():
        def __init__(self) -> None:
            self._state = -1

        def state_dict(self) -> dict:
            self._state += 1
            return {'state': self._state}

    mock_model = MockPytorchModel()
    early_stopping = EarlyStopping(model=mock_model, verbose=False, delta=1)
    assert not early_stopping.early_stop
    assert early_stopping._counter == 0
    assert early_stopping.lowest_loss is np.Inf
    assert early_stopping.best_state == {'state': 0}
    assert early_stopping.best_index is None

    # initial validation will automatically be captured because it is less than the default of Inf
    early_stopping(validation_loss=100)
    assert not early_stopping.early_stop
    assert early_stopping._counter == 0
    assert early_stopping._index == 0
    assert early_stopping.best_index == 0
    assert early_stopping.lowest_loss == 100
    assert early_stopping.best_state == {'state': 1}

    # now we are going to test out a lower/better loss that is beyond our delta of 1
    # a value of 98.99 should be captured as a better loss and should not increment the early
    # stopping counter
    early_stopping(validation_loss=98.99)
    assert not early_stopping.early_stop
    assert early_stopping._counter == 0
    assert early_stopping._index == 1
    assert early_stopping.best_index == 1
    assert early_stopping.lowest_loss == 98.99
    assert early_stopping.best_state == {'state': 2}

    # now we are going to test out a lower/better loss that is *not* beyond our delta of 1
    # a value of 97.99 (exactly 1 less than our current lowest loss of 98.99) should *not* be
    # captured as a better loss and *should* increment the early stopping counter
    model_state = early_stopping._model.state_dict()  # mimic state transition
    assert model_state == {'state': 3}
    early_stopping(validation_loss=97.99)
    assert not early_stopping.early_stop
    assert early_stopping._counter == 1
    assert early_stopping._index == 2
    assert early_stopping.best_index == 1
    assert early_stopping.lowest_loss == 98.99
    assert early_stopping.best_state == {'state': 2}

    current_lowest_loss = 98.99
    for index in range(5):
        model_state = early_stopping._model.state_dict()  # mimic state transition
        assert model_state == {'state': 4 + index}
        early_stopping(validation_loss=current_lowest_loss - 1)
        assert not early_stopping.early_stop
        assert early_stopping._counter == index + 2
        assert early_stopping._index == index + 3
        assert early_stopping.best_index == 1
        assert early_stopping.lowest_loss == current_lowest_loss
        assert early_stopping.best_state == {'state': 2}

    model_state = early_stopping._model.state_dict()  # mimic state transition
    assert model_state == {'state': 9}
    early_stopping(validation_loss=current_lowest_loss - 1)
    assert early_stopping.early_stop
    assert early_stopping._counter == 7
    assert early_stopping._index == 8
    assert early_stopping.best_index == 1
    assert early_stopping.lowest_loss == current_lowest_loss
    assert early_stopping.best_state == {'state': 2}

    # this should fail, we should not be able to keep calling early_stopping if we are already
    # past the point where we should have stopped
    with pytest.raises(AssertionError):
        early_stopping(validation_loss=current_lowest_loss - 1)


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
    assert test_model._early_stopping.best_index + test_model._early_stopping_patience == test_model._early_stopping._index  # noqa
    assert len(validation_loss) == test_model._early_stopping._index + 1
    assert len(training_loss) == len(validation_loss)
    assert not (validation_loss == 0).any()
    assert validation_loss.argmin() == test_model._early_stopping.best_index
    assert test_model._early_stopping._counter == test_model._early_stopping_patience
    # the state should have been updated since we have a lower validation score
    current_state = copy_state(test_model._early_stopping.best_state)
    assert_state_values_are_different(original_state, current_state)
    # ensure the saved best_state in the early_stopping object is the same as the model
    model_state = copy_state(test_model._model.state_dict())
    assert_states_are_same(model_state, current_state)
