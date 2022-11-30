from abc import ABC
import logging
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Altered from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, model, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self._model = model
        self._patience = patience
        self._verbose = verbose
        self._counter = 0
        self._delta = delta
        self._index = -1
        self.best_index = None
        self.early_stop = False
        self.best_score = -np.Inf
        self.min_validation_loss = np.Inf
        self.best_state = None
        self.save_checkpoint(validation_loss=np.NaN)

    def __call__(self, validation_loss):
        assert not self.early_stop
        self._index += 1
        score = -validation_loss
        if score < self.best_score + self._delta:
            # if the score has not improved (i.e. loss has not decreased) then increment counter
            self._counter += 1
            if self._verbose:
                message = f'Early Stopping counter: {self._counter} out of {self._patience}'
                logging.info(message)
            if self._counter >= self._patience:
                self.early_stop = True
        else:
            # loss decreased; restart counter and save the model's state
            self.best_score = score
            self.save_checkpoint(validation_loss)
            self._counter = 0

    def save_checkpoint(self, validation_loss):
        '''Saves model when validation loss decrease.'''
        if self._verbose:
            message = f'Validation loss decreased ({self.min_validation_loss:.6f} --> ' \
                f'{validation_loss:.6f}). Saving model ...'
            logging.info(message)
        self.best_state = self._model.state_dict()
        self.best_index = self._index
        self.min_validation_loss = validation_loss


class PyTorchNN(ABC):
    """Base class that wraps pytorch training, early stopping logic"""
    def __init__(
            self,
            model,
            loss_func,
            optimizer,
            early_stopping_patience=10,
            early_stopping_verbose=False,
            ) -> None:
        super().__init__()
        self._model = model
        self._loss_func = loss_func
        self._optimizer = optimizer
        self._early_stopping = None
        self._early_stopping_patience = early_stopping_patience
        self._early_stopping_verbose = early_stopping_verbose

    def _train_epoch(self, data_loader: DataLoader):
        """
        Training over a single epoch.
        """
        self._model.train()
        for x_batch, y_batch in data_loader:
            pred = self._model(x_batch)[:, 0]
            loss = self._loss_func(pred, y_batch)
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

    def _eval_epoch(self, X_train, y_train, X_validation, y_validation):
        """
        Evaluation over a single epoch.
        """
        pred = self.predict(X=X_train)
        loss = self._loss_func(pred, y_train)
        train_loss = loss.item()

        if X_validation is None:
            validation_loss = np.NaN
        else:
            pred = self.predict(X=X_validation)
            loss = self._loss_func(pred, y_validation)
            validation_loss = loss.item()

        return train_loss, validation_loss

    def train(self, X, y, num_epochs=200, batch_size=8, validation_size=0.1, random_seed=1):
        """
        Train a specific number of epochs

        Args:
            X: tensor containing training data
            y: tensor containing training target values
            num_epochs: the number of epochs
            batch_size: the number records per batches for each epoch iteration
            validation_size: the percent of the data to use for validation e.g. 0.1 means 10% of
                the data (`X`) will be used for validation. A validation set is required for early
                stopping.
            random_seed: random seed/state
        """
        assert self._early_stopping_patience is None or validation_size > 0
        if self._early_stopping_patience:
            self._early_stopping = EarlyStopping(
                model=self._model,
                patience=self._early_stopping_patience,
                verbose=self._early_stopping_verbose,
            )
        else:
            self._early_stopping = None

        if validation_size > 0:
            X_t, X_v, y_t, y_v = train_test_split(
                X, y, train_size=1-validation_size, random_state=random_seed
            )
            assert len(X_t) + len(X_v) == len(X)
            assert len(y_t) + len(y_v) == len(y)

        else:
            X_t, X_v, y_t, y_v = X, None, y, None

        torch.manual_seed(1)

        loss_hist_train = np.zeros(num_epochs)
        loss_hist_validation = np.zeros(num_epochs)

        assert len(X_t) == len(y_t)
        training_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            self._train_epoch(data_loader=training_loader)

            train_loss, validation_loss = self._eval_epoch(
                X_train=X_t, y_train=y_t, X_validation=X_v, y_validation=y_v
            )

            loss_hist_train[epoch] = train_loss
            loss_hist_validation[epoch] = validation_loss

            if self._early_stopping:
                self._early_stopping(validation_loss=validation_loss)
                if self._early_stopping.early_stop:
                    self._model.load_state_dict(self._early_stopping.best_state)
                    loss_hist_train = loss_hist_train[0:self._early_stopping._index + 1]
                    loss_hist_validation = loss_hist_validation[0:self._early_stopping._index + 1]
                    break

        return loss_hist_train, loss_hist_validation

    def predict(self, X: torch.Tensor):
        """Use the model to generate predictions from `X`."""
        self._model.eval()
        with torch.no_grad():
            pred = self._model(X)[:, 0]
        return pred


class FullyConnectedNN(PyTorchNN):
    """
    This class implements a feed-forward fully connected PyTorch neural network. The number of
    hidden layers is dynamic and based on the size of the `hidden_units` parameter, which is a list
    that indicates the size of each hidden layer.
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_units: Optional[Tuple[int]],
            loss_func: Callable,
            learning_rate: float,
            early_stopping_patience: int = 10,
            early_stopping_verbose: bool = True,
            ) -> None:

        if hidden_units is None:
            hidden_units = [50, 20]

        all_layers = []
        for hidden_unit in hidden_units:
            # print(f"Creating Linear layer with {input_size} input units and {hidden_unit} hidden units.")  # noqa
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.ReLU())
            input_size = hidden_unit

        # print(f"Creating output layer Linear layer with {input_size} input units and {hidden_units[-1]} output units.")  # noqa
        all_layers.append(nn.Linear(hidden_units[-1], output_size))
        model = nn.Sequential(*all_layers)
        # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        super().__init__(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            early_stopping_patience=early_stopping_patience,
            early_stopping_verbose=early_stopping_verbose,
        )
