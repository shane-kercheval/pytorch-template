"""Convenient wrappers around pytorch for training."""

import logging
import math
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience. Saves the
    model state associated with the best/lowest validation loss and restores that state when early
    stopping is triggered.

    Altered from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(
            self,
            model: nn.Module,
            patience: int = 7,
            delta: float = 0,
            delta_type: str = 'relative',
            verbose: bool = False,
            ):
        """
        Args:
            model: PyTorch model to save state of.
            patience: How long (in calls) to wait after last time validation loss improved.
            delta: Minimum change in the monitored quantity to qualify as an improvement.
            delta_type:
                The type of delta to use. Either 'relative' or 'absolute'. If 'relative', the
                delta is a percentage of the previous lowest loss. For example, a value of 0.10
                indicates a 10% change in the new loss from the previous lowest loss will trigger
                early stopping. If 'absolute', the delta is an absolute value.
            previous_lowest_loss: The lowest validation loss score from a previous training run.
            verbose: If True, prints a message for each validation loss improvement.
        """
        self._model = model
        self._patience = patience
        self._verbose = verbose
        self._counter = 0
        self._delta = delta
        if delta_type not in ('relative', 'absolute'):
            raise ValueError(f"Invalid delta_type: {delta_type}")
        self._delta_type = delta_type
        self._index = -1
        self.is_stopped = False
        self.best_index = None
        self.lowest_loss = np.Inf
        self.best_state = model.state_dict()

    def __call__(self, loss: float) -> bool:
        """
        Runs the early stopping logic and saves the model's state if the validation loss score
        improved. If the validation loss score did not improve, the counter is incremented.

        Args:
            loss: The validation loss score to monitor.
        """
        assert not self.is_stopped
        self._index += 1

        loss_decrease = False
        if self._delta_type == 'relative':
            if loss < self.lowest_loss * (1 - self._delta):
                loss_decrease = True
        elif self._delta_type == 'absolute':
            if loss < self.lowest_loss - abs(self._delta):
                loss_decrease = True
        else:
            raise ValueError(f"Invalid delta_type: {self._delta_type}")

        if loss_decrease:
        # if loss < self.lowest_loss - abs(self._delta):
            # loss decreased; restart counter and save the model's state
            if self._verbose:
                logging.info(
                    f'Validation loss decreased ({self.lowest_loss:.6f} --> '
                    f'{loss:.6f}). Caching model state.',
                )
            self.best_state = self._model.state_dict()
            self.best_index = self._index
            self.lowest_loss = loss
            self._counter = 0
        else:
            # if the score has not improved (i.e. loss has not decreased) then increment counter
            self._counter += 1
            if self._verbose:
                logging.info(
                    f'Early Stopping counter: {self._counter} out of {self._patience}',
                )
            if self._counter >= self._patience:
                self.is_stopped = True

        return self.is_stopped

    def reset(self) -> None:
        """Reset the early stopping counter and restore the model's best state."""
        self.is_stopped = False
        self._counter = 0


def calculate_average_loss(
        data_loader: DataLoader,
        model: nn.Module,
        loss_func: callable,
        device: str) -> float:
    """
    Calculate the average loss for a given data set. Calculates the loss for each batch and then
    takes the weighted average of the loss adjusted for the batch size. Uses batches to avoid
    memory issues.

    Args:
        data_loader: PyTorch DataLoader containing data.
        model: PyTorch model to use for prediction.
        loss_func: PyTorch loss function.
        device: The device e.g. 'cpu' or 'cuda'.
    """
    running_loss = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)  # noqa: PLW2901
            loss = loss_func(model(x), y)
            # weighted average of the loss adjusted for the batch size
            running_loss += loss.item() * x.shape[0]
            total_samples += x.shape[0]
    return running_loss / total_samples



class PyTorchTrainer:
    """Base class that wraps pytorch training, early stopping logic."""

    def __init__(
            self,
            model: nn.Module,
            loss_func: callable,
            device: str | None = None,
            validation_size: float = 0.1,
            early_stopping_patience: int | None = 10,
            early_stopping_delta: float = 0,
            verbose: bool = False,
            ) -> None:
        """
        Args:
            model:
                PyTorch model to train.
            loss_func:
                PyTorch loss function.
            device:
                Device to use for training. If None, will use GPU if available, otherwise CPU.
            validation_size: the percent of the data to use for validation e.g. 0.1 means 10% of
                the data (x) will be used for validation. A validation set is required for early
                stopping.
            early_stopping_patience:
                How long (in calls) to wait after last time validation loss improved. If None,
                early stopping is disabled.
            early_stopping_delta:
                Minimum change in the monitored quantity to qualify as an improvement.
            verbose:
                If True, prints a message for each validation loss improvement.
        """
        super().__init__()
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.device = device
        self.model = model.to(device)
        assert 0 < validation_size < 1
        self.validation_size = validation_size
        self.best_state = None
        self.early_stopping = None
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.verbose = verbose
        self._loss_func = loss_func

    def _train_epoch(
            self,
            epoch: int,
            optimizer: torch.optim.Optimizer,
            training_loader: DataLoader,
            validation_loader: DataLoader | None = None,
            log_interval: int | None = 20) -> None:
        """
        Training over a single epoch.

        Args:
            epoch: The current epoch.
            optimizer: PyTorch optimizer.
            training_loader: PyTorch DataLoader containing training data.
            validation_loader: PyTorch DataLoader containing validation data. Only used when
                logging data.
            log_interval: How often to log training progress (i.e. number of equal sized batches).
                If None, no logging will occur.

        """
        running_training_loss = 0
        total_train_samples = 0

        if log_interval:
            total_batches = len(training_loader)
            log_interval = max(1, math.floor(total_batches / log_interval))

        self.model.train()
        optimizer.zero_grad()
        for batch_index, (x_batch, y_batch) in enumerate(training_loader):
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)  # noqa: PLW2901
            # forward pass
            pred = self.model(x_batch)
            if self.model.layers[-1].out_features == 1:  # if output is 1 dim, extract array
                pred = pred[:, 0]
            loss = self._loss_func(pred, y_batch)
            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if log_interval:
                # weighted average of the training loss
                running_training_loss += loss.item() * x_batch.shape[0]
                total_train_samples += x_batch.shape[0]
                # log at specified intervals
                if batch_index % log_interval == 0:
                    # training loss
                    avg_training_loss = running_training_loss / total_train_samples
                    running_training_loss = 0
                    total_train_samples = 0
                    # validation loss
                    self.model.eval()
                    avg_validation_loss = calculate_average_loss(
                        data_loader=validation_loader, model=self.model,
                        loss_func=self._loss_func, device=self.device,
                    )
                    self.model.train()
                    logging.info(
                        f"Epoch {epoch} - Batch {batch_index}/{total_batches}: "
                        f"Training Loss={avg_training_loss:.3f}; "
                        f"Validation Loss={avg_validation_loss:.3f}",
                    )

    def train(
            self,
            x: torch.tensor,
            y: torch.tensor,
            optimizer: torch.optim.Optimizer,
            epochs: int = 200,
            batch_size: int = 32,
            log_batches: int | None = None,
            random_seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Train a specific number of epochs.

        Args:
            x: tensor containing training data
            y: tensor containing training target values
            optimizer: PyTorch optimizer.
            epochs: the number of epochs
            batch_size: the number records per batches for each epoch iteration
            log_batches: How often to log training progress (i.e. number of equal sized batches).
                If None, no logging will occur.
            random_seed: random seed/state
        """
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        random.seed(random_seed)
        np.random.seed(random_seed + 1)  # noqa: NPY002
        torch.manual_seed(random_seed + 2)
        torch.cuda.manual_seed_all(random_seed + 3)

        # if we have previous early stopping state, restore it
        if self.early_stopping_patience:
            # if we have already initialized early stopping, reset it
            if self.early_stopping:
                self.early_stopping.reset()
            else:
                self.early_stopping = EarlyStopping(
                    model=self.model,
                    patience=self.early_stopping_patience,
                    delta=self.early_stopping_delta,
                    verbose=self.verbose,
            )
        else:
            self.early_stopping = None

        data_loader = lambda x, y: DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)  # noqa
        x_train, x_validation, y_train, y_validation = train_test_split(
            x, y,
            train_size=1-self.validation_size,
            random_state=random_seed,
        )
        assert len(x_train) + len(x_validation) == len(x)
        assert len(y_train) + len(y_validation) == len(y)
        training_loader = data_loader(x_train, y_train)
        validation_loader = data_loader(x_validation, y_validation)

        def calc_avg_loss(loader: DataLoader) -> float:
            return calculate_average_loss(
                data_loader=loader, model=self.model,
                loss_func=self._loss_func, device=self.device,
            )

        loss_hist_train = []
        loss_hist_validation = []
        for epoch in range(epochs):
            self._train_epoch(
                epoch=epoch,
                optimizer=optimizer,
                training_loader=training_loader,
                validation_loader=validation_loader,
                log_interval=log_batches,
            )
            self.model.eval()
            average_training_loss = calc_avg_loss(training_loader)
            average_validation_loss = calc_avg_loss(validation_loader)
            loss_hist_train.append(average_training_loss)
            loss_hist_validation.append(average_validation_loss)

            if self.verbose:
                logging.info(
                    f"Epoch {epoch}: Training Loss={average_training_loss:.3f}; "
                    f"Validation Loss={average_validation_loss:.3f}",
                )
            if self.early_stopping and self.early_stopping(loss=average_validation_loss):
                self.model.load_state_dict(self.early_stopping.best_state)
                loss_hist_train = loss_hist_train[0:self.early_stopping._index + 1]
                loss_hist_validation = loss_hist_validation[0:self.early_stopping._index + 1]
                break

        return loss_hist_train, loss_hist_validation

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Use the model to generate predictions from `X`."""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            pred = self.model(x)
            if self.model.layers[-1].out_features == 1:  # if output is 1 dim, extract array
                pred = pred[:, 0]
        return pred

    @property
    def best_validation_loss(self) -> float:
        """The lowest validation loss score."""
        return self.early_stopping.lowest_loss if self.early_stopping else None


class FullyConnectedNN(nn.Module):
    """
    Implements a feed-forward fully connected PyTorch neural network. The number of
    hidden layers is dynamic and based on the size of the `hidden_units` parameter, which is a list
    that indicates the size of each hidden layer.
    """

    def __init__(
            self,
            input_size: int,
            hidden_layers: list[int],
            output_size: int):
        """
        Args:
            input_size: The number of input features.
            hidden_layers: A list of integers indicating the size of each hidden layer. At least
                one hidden layer is required (i.e. list of one item).
            output_size: The number of output features.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Flatten())
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(input_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return x




# class FullyConnectedNN(PyTorchNN):
#     """
#     Implements a feed-forward fully connected PyTorch neural network. The number of
#     hidden layers is dynamic and based on the size of the `hidden_units` parameter, which is a
#     list that indicates the size of each hidden layer.
#     """

#     def __init__(
#             self,
#             input_size: int,
#             output_size: int,
#             loss_func: callable,
#             hidden_units: tuple[int] | None = None,
#             optimizer_func: torch.optim.Optimizer = torch.optim.SGD,
#             learning_rate: float = 0.001,
#             initial_layers: list | None = None,
#             early_stopping_patience: int | None = 10,
#             early_stopping_delta: float = 0,
#             device: str | None = None,
#             verbose: bool = False,
#             ) -> None:

#         if hidden_units is None:
#             hidden_units = [50, 20]

#         self.hidden_units = hidden_units
#         if initial_layers:
#             assert isinstance(initial_layers, list)
#             self.layers = initial_layers
#         else:
#             self.layers = []
#         for hidden_unit in hidden_units:
#             # print(f"Creating Linear layer with {input_size} input units and {hidden_unit} hidden units.")  # noqa
#             layer = nn.Linear(input_size, hidden_unit)
#             self.layers.append(layer)
#             self.layers.append(nn.ReLU())
#             input_size = hidden_unit

#         # print(f"Creating output layer Linear layer with {input_size} input units and {hidden_units[-1]} output units.")  # noqa
#         self.layers.append(nn.Linear(hidden_units[-1], output_size))
#         model = nn.Sequential(*self.layers)
#         # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
#         optimizer = optimizer_func(model.parameters(), lr=learning_rate)
#         super().__init__(
#             model=model,
#             loss_func=loss_func,
#             optimizer=optimizer,
#             early_stopping_patience=early_stopping_patience,
#             early_stopping_delta=early_stopping_delta,
#             device=device,
#             verbose=verbose,
#         )
