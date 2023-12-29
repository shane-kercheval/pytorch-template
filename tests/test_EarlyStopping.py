"""Tests early stopping functionality."""

import logging
import pytest
from torch import nn
from torch import optim
import numpy as np

from source.domain.pytorch_wrappers import EarlyStopping, FullyConnectedNN, PyTorchTrainer
from tests.helpers import get_test_file_path


logging.config.fileConfig(
    get_test_file_path("logging/test_logging.conf"),
    defaults={'logfilename': get_test_file_path("logging/log.log")},
    disable_existing_loggers=False,
)


def copy_state(state: dict) -> dict:  # noqa: D103
    return {k: np.array(v).copy() for k, v in state.items()}


def assert_states_are_same(state_a: dict, state_b: dict) -> None:  # noqa: D103
    assert all(x == y for x, y in zip(state_a.keys(), state_b.keys()))
    assert all((x == y).all() for x, y in zip(state_a.values(), state_b.values()))


def assert_state_values_are_different(state_a: dict, state_b: dict) -> None:  # noqa: D103
    assert all(x == y for x, y in zip(state_a.keys(), state_b.keys()))
    assert all((x != y).any() for x, y in zip(state_a.values(), state_b.values()))


def test_early_stopping_delta_0():  # noqa
    """
    Tests the EarlyStopping class using a mock model so that we can control the validation loss
    values that are used.
    """

    class MockPytorchModel:
        def __init__(self) -> None:
            self._state = -1

        def state_dict(self) -> dict:
            self._state += 1
            return {'state': self._state}

    mock_model = MockPytorchModel()
    early_stopping = EarlyStopping(model=mock_model, verbose=False, delta=0)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 0
    assert early_stopping.lowest_loss is np.Inf
    assert early_stopping.best_state == {'state': 0}
    assert early_stopping.best_index is None

    # initial validation will automatically be captured because it is less than the default of Inf
    assert not early_stopping(loss=100)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 0
    assert early_stopping._index == 0
    assert early_stopping.best_index == 0
    assert early_stopping.lowest_loss == 100
    assert early_stopping.best_state == {'state': 1}

    # now we are going to test out a lower/better loss that is beyond our delta of 0
    # a value of 99.99 should be captured as a better loss and should not increment the early
    # stopping counter
    assert not early_stopping(loss=99.99)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 0
    assert early_stopping._index == 1
    assert early_stopping.best_index == 1
    assert early_stopping.lowest_loss == 99.99
    assert early_stopping.best_state == {'state': 2}

    # should *not* be captured as a better loss and *should* increment the early stopping counter
    model_state = early_stopping._model.state_dict()  # mimic state transition
    assert model_state == {'state': 3}
    assert not early_stopping(loss=99.99)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 1
    assert early_stopping._index == 2
    assert early_stopping.best_index == 1
    assert early_stopping.lowest_loss == 99.99
    assert early_stopping.best_state == {'state': 2}

    current_lowest_loss = 99.99
    for index in range(5):
        model_state = early_stopping._model.state_dict()  # mimic state transition
        assert model_state == {'state': 4 + index}
        assert not early_stopping(loss=current_lowest_loss)
        assert not early_stopping.is_stopped
        assert early_stopping._counter == index + 2
        assert early_stopping._index == index + 3
        assert early_stopping.best_index == 1
        assert early_stopping.lowest_loss == current_lowest_loss
        assert early_stopping.best_state == {'state': 2}

    model_state = early_stopping._model.state_dict()  # mimic state transition
    assert model_state == {'state': 9}
    assert early_stopping(loss=current_lowest_loss)
    assert early_stopping.is_stopped
    assert early_stopping._counter == 7
    assert early_stopping._index == 8
    assert early_stopping.best_index == 1
    assert early_stopping.lowest_loss == current_lowest_loss
    assert early_stopping.best_state == {'state': 2}

    # this should fail, we should not be able to keep calling early_stopping if we are already
    # past the point where we should have stopped
    with pytest.raises(AssertionError):
        early_stopping(loss=current_lowest_loss - 1)


def test_early_stopping_delta_absolute_1():  # noqa
    """
    Tests the EarlyStopping class using a mock model so that we can control the validation loss
    values that are used.
    """
    class MockPytorchModel:
        def __init__(self) -> None:
            self._state = -1

        def state_dict(self) -> dict:
            self._state += 1
            return {'state': self._state}

    mock_model = MockPytorchModel()
    early_stopping = EarlyStopping(model=mock_model, verbose=False, delta=1, delta_type='absolute')
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 0
    assert early_stopping.lowest_loss is np.Inf
    assert early_stopping.best_state == {'state': 0}
    assert early_stopping.best_index is None

    # initial validation will automatically be captured because it is less than the default of Inf
    assert not early_stopping(loss=100)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 0
    assert early_stopping._index == 0
    assert early_stopping.best_index == 0
    assert early_stopping.lowest_loss == 100
    assert early_stopping.best_state == {'state': 1}

    # now we are going to test out a lower/better loss that is beyond our delta of 1
    # a value of 98.99 should be captured as a better loss and should not increment the early
    # stopping counter
    assert not early_stopping(loss=98.99)
    assert not early_stopping.is_stopped
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
    assert not early_stopping(loss=97.99)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 1
    assert early_stopping._index == 2
    assert early_stopping.best_index == 1
    assert early_stopping.lowest_loss == 98.99
    assert early_stopping.best_state == {'state': 2}

    current_lowest_loss = 98.99
    for index in range(5):
        model_state = early_stopping._model.state_dict()  # mimic state transition
        assert model_state == {'state': 4 + index}
        assert not early_stopping(loss=current_lowest_loss - 1)
        assert not early_stopping.is_stopped
        assert early_stopping._counter == index + 2
        assert early_stopping._index == index + 3
        assert early_stopping.best_index == 1
        assert early_stopping.lowest_loss == current_lowest_loss
        assert early_stopping.best_state == {'state': 2}

    model_state = early_stopping._model.state_dict()  # mimic state transition
    assert model_state == {'state': 9}
    assert early_stopping(loss=current_lowest_loss - 1)
    assert early_stopping.is_stopped
    assert early_stopping._counter == 7
    assert early_stopping._index == 8
    assert early_stopping.best_index == 1
    assert early_stopping.lowest_loss == current_lowest_loss
    assert early_stopping.best_state == {'state': 2}

    # this should fail, we should not be able to keep calling early_stopping if we are already
    # past the point where we should have stopped
    with pytest.raises(AssertionError):
        early_stopping(loss=current_lowest_loss - 1)


def test_early_stopping_delta_percent_5():  # noqa
    """
    Tests the EarlyStopping class using a mock model so that we can control the validation loss
    values that are used.
    """
    class MockPytorchModel:
        def __init__(self) -> None:
            self._state = -1

        def state_dict(self) -> dict:
            self._state += 1
            return {'state': self._state}

    mock_model = MockPytorchModel()
    early_stopping = EarlyStopping(
        model=mock_model,
        verbose=False,
        delta=0.05,  # 5%
        delta_type='percent',
    )
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 0
    assert early_stopping.lowest_loss is np.Inf
    assert early_stopping.best_state == {'state': 0}
    assert early_stopping.best_index is None

    # initial validation will automatically be captured because it is less than the default of Inf
    assert not early_stopping(loss=100)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 0
    assert early_stopping._index == 0
    assert early_stopping.best_index == 0
    assert early_stopping.lowest_loss == 100
    assert early_stopping.best_state == {'state': 1}

    # a drop of 5 from 100 is 95, which is exactly 5%, so it should not be captured as a better
    # because it is not **beyond** our delta of 5%
    assert not early_stopping(loss=95)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 1
    assert early_stopping._index == 1
    assert early_stopping.best_index == 0
    assert early_stopping.lowest_loss == 100
    assert early_stopping.best_state == {'state': 1}


    # now we are going to test out a lower/better loss that is beyond our delta of 5%
    # a value of 94.99 should be captured as a better loss and should not increment the early
    # stopping counter
    assert not early_stopping(loss=94.99)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 0
    assert early_stopping._index == 2
    assert early_stopping.best_index == 2
    assert early_stopping.lowest_loss == 94.99
    assert early_stopping.best_state == {'state': 2}

    # now we are going to test out a lower/better loss that is *not* beyond our delta of 1
    # a value of 97.99 (exactly 1 less than our current lowest loss of 94.99) should *not* be
    # captured as a better loss and *should* increment the early stopping counter
    model_state = early_stopping._model.state_dict()  # mimic state transition
    assert model_state == {'state': 3}
    assert not early_stopping(loss=93.99)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 1
    assert early_stopping._index == 3
    assert early_stopping.best_index == 2
    assert early_stopping.lowest_loss == 94.99
    assert early_stopping.best_state == {'state': 2}

    current_lowest_loss = 94.99
    for index in range(5):
        model_state = early_stopping._model.state_dict()  # mimic state transition
        assert model_state == {'state': 4 + index}
        assert not early_stopping(loss=current_lowest_loss * 0.95)
        assert not early_stopping.is_stopped
        assert early_stopping._counter == index + 2
        assert early_stopping._index == index + 4
        assert early_stopping.best_index == 2
        assert early_stopping.lowest_loss == current_lowest_loss
        assert early_stopping.best_state == {'state': 2}

    model_state = early_stopping._model.state_dict()  # mimic state transition
    assert model_state == {'state': 9}
    assert early_stopping(loss=current_lowest_loss * 0.95)
    assert early_stopping.is_stopped
    assert early_stopping._counter == 7
    assert early_stopping._index == 9
    assert early_stopping.best_index == 2
    assert early_stopping.lowest_loss == current_lowest_loss
    assert early_stopping.best_state == {'state': 2}

    # this should fail, we should not be able to keep calling early_stopping if we are already
    # past the point where we should have stopped
    with pytest.raises(AssertionError):
        early_stopping(loss=current_lowest_loss - 1)



def test_early_stopping_with_reset():  # noqa
    class MockPytorchModel:
        def __init__(self) -> None:
            self._state = -1

        def state_dict(self) -> dict:
            self._state += 1
            return {'state': self._state}

    mock_model = MockPytorchModel()
    early_stopping = EarlyStopping(
        model=mock_model,
        patience=2,
        delta=0,
        verbose=False,
    )
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 0
    assert early_stopping.lowest_loss == np.Inf
    assert early_stopping.best_state == {'state': 0}
    assert early_stopping.best_index is None

    # initial validation will automatically be captured because it is less than the default of Inf
    assert not early_stopping(loss=2)
    assert not early_stopping.is_stopped
    # this will not increment because loss of 2 is lower than the default of Inf
    assert early_stopping._counter == 0
    assert early_stopping._index == 0
    assert early_stopping.best_index == 0
    assert early_stopping.lowest_loss == 2
    assert early_stopping.best_state == {'state': 1}  # mock state captured (and incremented)

    # this will not increment because loss of 3 is not lower than the current lowest loss of 2
    assert not early_stopping(loss=3)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 1
    assert early_stopping._index == 1
    assert early_stopping.best_index == 0
    assert early_stopping.lowest_loss == 2
    assert early_stopping.best_state == {'state': 1}  # mock state not changed

    # this will not increment because loss of 2 is not lower than the current lowest loss of 2
    # this will stop the early stopping because we have reached the patience limit of 2
    assert early_stopping(loss=2)
    assert early_stopping.is_stopped
    assert early_stopping._counter == 2
    assert early_stopping._index == 2
    assert early_stopping.best_index == 0
    assert early_stopping.lowest_loss == 2
    assert early_stopping.best_state == {'state': 1}  # mock state not changed

    # this should fail, we should not be able to keep calling early_stopping if we are already
    # past the point where we should have stopped
    with pytest.raises(AssertionError):
        early_stopping(loss=1)
    # everything should remain the unchanged
    assert early_stopping.is_stopped
    assert early_stopping._counter == 2
    assert early_stopping._index == 2
    assert early_stopping.best_index == 0
    assert early_stopping.lowest_loss == 2
    assert early_stopping.best_state == {'state': 1}  # mock state not changed

    # now we are going to reset the early stopping object
    early_stopping.reset()
    assert not early_stopping.is_stopped  # should not be stopped after reset
    assert early_stopping._counter == 0  # counter should be reset to 0
    assert early_stopping._index == 2  # index should not be reset
    assert early_stopping.best_index == 0  # best index should not be reset
    assert early_stopping.lowest_loss == 2  # lowest loss should not be reset
    assert early_stopping.best_state == {'state': 1}  # mock state not changed

    # now we should be able to call early_stopping again
    # this will not increment because loss of 2 is not lower than the current lowest loss of 2
    assert not early_stopping(loss=2)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 1
    assert early_stopping._index == 3
    assert early_stopping.best_index == 0
    assert early_stopping.lowest_loss == 2
    assert early_stopping.best_state == {'state': 1}  # mock state not changed

    # call with lower loss should reset the counter and save the new best state
    assert not early_stopping(loss=1)
    assert not early_stopping.is_stopped
    assert early_stopping._counter == 0
    assert early_stopping._index == 4
    assert early_stopping.best_index == 4
    assert early_stopping.lowest_loss == 1
    assert early_stopping.best_state == {'state': 2}  # mock state captured (and incremented)


def test_early_stopping_within_pytorch_wrapper(dummy_x_y):  # noqa
    x, y = dummy_x_y
    test_model = FullyConnectedNN(
        input_size=2,
        hidden_layers=[8, 4],
        output_size=1,
    )
    trainer = PyTorchTrainer(
        model=test_model,
        loss_func=nn.MSELoss(),
        validation_size=0.2,
        early_stopping_patience=5,
        device='cpu',
        verbose=False,
    )
    assert trainer.early_stopping is None
    # capture original state of the model to make sure best_state gets updated
    original_state = copy_state(test_model.state_dict())

    # this validation loss will be counted as lower since it is the first
    training_loss, validation_loss = trainer.train(
        x=x,
        y=y,
        optimizer=optim.Adam(test_model.parameters(), lr=0.001),
        epochs=1000,
        batch_size=2,
        random_seed=3,
    )
    assert trainer.early_stopping.is_stopped
    assert trainer.early_stopping.best_index < 1000
    assert len(validation_loss) == trainer.early_stopping._index + 1
    assert len(training_loss) == len(validation_loss)
    assert not any(loss == 0 for loss in validation_loss)
    assert np.array(validation_loss).argmin() == trainer.early_stopping.best_index
    # the state should have been updated since we have a lower validation score
    assert trainer.early_stopping.lowest_loss == trainer.best_validation_loss
    current_state = copy_state(trainer.early_stopping.best_state)
    assert_state_values_are_different(original_state, current_state)
    # ensure the saved best_state in the early_stopping object is the same as the model
    model_state = copy_state(trainer.model.state_dict())
    assert_states_are_same(model_state, current_state)
