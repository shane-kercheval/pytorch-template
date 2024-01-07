"""Defines functions and variables used for loading data."""
import logging
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch

from source.library.architectures import Architecture


DIMENSIONS = (28, 28)
INPUT_SIZE = DIMENSIONS[0] * DIMENSIONS[1]
OUTPUT_SIZE = 10


def get_data(architecture: Architecture):  # noqa
    """Function is required by and called from `model_pipeline()`."""
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    x, y = transform_data(architecture=architecture, x=x, y=y)
    # 80% train; 10% validation; 10% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    logging.info(f"Training set  : X-{x_train.shape}, y-{y_train.shape}")
    logging.info(f"Validation set: X-{x_val.shape}, y-{y_val.shape}")
    logging.info(f"Test set      : X-{x_test.shape}, y-{y_test.shape}")
    return x_train, x_val, x_test, y_train, y_val, y_test


def transform_data(
        architecture: Architecture,
        x: pd.DataFrame,
        y: pd.Series | None = None) -> tuple[torch.tensor, torch.tensor]:
    """
    Transforms the data. Returns a tuple of (x, y) where x is a tensor of the features and y is a
    tensor of the labels.

    Args:
        architecture: The architecture to use.
        x: A dataframe of the features.
        y: A series of the labels.
    """
    x = torch.tensor(x.values, dtype=torch.float32)
    # Normalize the tensor
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)
    assert x.min() == 0
    assert x.max() == 1
    if architecture == Architecture.CONVOLUTIONAL:
        # Reshape data to have channel dimension
        # MNIST images are 28x28, so we reshape them to [batch_size, 1, 28, 28]
        x = x.reshape(-1, 1, DIMENSIONS[0], DIMENSIONS[1])
    if y is not None:
        y = torch.tensor(y.astype(int).values, dtype=torch.long)
    return x, y
