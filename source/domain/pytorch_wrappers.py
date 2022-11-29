
from abc import ABC
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class PyTorchNN(ABC):
    def __init__(self, model, loss_func, optimizer) -> None:
        super().__init__()
        self._model = model
        self._loss_func = loss_func
        self._optimizer = optimizer

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

        pred = self.predict(X=X_validation)
        loss = self._loss_func(pred, y_validation)
        validation_loss = loss.item()

        return train_loss, validation_loss

    def train(self, X, y, num_epochs=200, batch_size=8, validation_size=0.1, random_seed=1):

        X_t, X_v, y_t, y_v = train_test_split(
            X, y, train_size=1-validation_size, random_state=random_seed
        )
        assert len(X_t) + len(X_v) == len(X)
        assert len(y_t) + len(y_v) == len(y)

        torch.manual_seed(1)

        loss_hist_train = np.empty(num_epochs)
        loss_hist_validation = np.empty(num_epochs)

        training_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            self._train_epoch(data_loader=training_loader)

            train_loss, validation_loss = self._eval_epoch(
                X_train=X_t, y_train=y_t, X_validation=X_v, y_validation=y_v
            )

            loss_hist_train[epoch] = train_loss
            loss_hist_validation[epoch] = validation_loss

        return loss_hist_train, loss_hist_validation

    def predict(self, X: torch.Tensor):
        self._model.eval()
        with torch.no_grad():
            pred = self._model(X)[:, 0]
        return pred


class FullyConnectedNN(PyTorchNN):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_units: Optional[Tuple[int]],
            loss_func,
            learning_rate,
            ) -> None:

        if hidden_units is None:
            hidden_units = [50, 20]

        # print(f"Number of features: {input_size}")

        all_layers = []
        for hidden_unit in hidden_units:
            # print(f"Creating Linear layer with {input_size} input units and {hidden_unit} hidden units.")
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.ReLU())
            input_size = hidden_unit

        # print(f"Creating output layer Linear layer with {input_size} input units and {hidden_units[-1]} output units.")
        all_layers.append(nn.Linear(hidden_units[-1], output_size))
        model = nn.Sequential(*all_layers)
        # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        super().__init__(model=model, loss_func=loss_func, optimizer=optimizer)
