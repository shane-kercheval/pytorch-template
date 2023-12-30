"""
TODO:
"""

import pprint
import numpy as np
import wandb
import yaml
import logging.config
import logging
import os
import click
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from source.domain.experiment import model_pipeline


logging.config.fileConfig(
    os.path.join(os.getcwd(), '/code/source/config/logging.conf'),
    disable_existing_loggers=False,
)


def get_data():  # noqa
    """Function is required by and called from `model_pipeline()`."""
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    x = torch.tensor(x.values, dtype=torch.float32)
    y = torch.tensor(y.astype(int).values, dtype=torch.long)
    # need to make this dynamic based on Fully Connected vs Convolutional
    # Reshape data to have channel dimension
    # MNIST images are 28x28, so we reshape them to [batch_size, 1, 28, 28]
    x = x.reshape(-1, 1, 28, 28)
    # 80% train; 10% validation; 10% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    logging.info(f"Training set  : X-{x_train.shape}, y-{y_train.shape}")
    logging.info(f"Validation set: X-{x_val.shape}, y-{y_val.shape}")
    logging.info(f"Test set      : X-{x_test.shape}, y-{y_test.shape}")
    return x_train, x_val, x_test, y_train, y_val, y_test


@click.group()
def main() -> None:
    """Command Line Logic."""
    pass


@main.command()
@click.option('--config_file', '-c', type=str)
@click.option('--device', type=str, default=None)
def run(config_file: str, device: str | None = None) -> None:
    """Execute a single 'run' on Weights and Biases."""
    with open(config_file) as f:
        config = yaml.safe_load(f)

    pprint.pprint(config)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")
    config['device'] = device
    _ = model_pipeline(config)


@main.command()
@click.option('--config_file', '-c', type=str)
@click.option('--device', type=str, default=None)
@click.option('--count', type=int, default=None)
@click.option('--sweep_id', type=str, default=None)
def sweep(
        config_file: str,
        device: str | None = None,
        count: int | None = None,
        sweep_id: str | None = None) -> None:
    """
    Execute a 'sweep' (of multiple runs) on Weights and Biases.

    Args:
        config_file:
            Path to the YAML configuration file.
        device:
            Device to use for training. If None, will use GPU if available.
        count:
            Number of runs to execute. If None, will execute all runs. Ignored if
            config['method'] == 'grid'.
        sweep_id:
            ID of the sweep to execute. If provided, this function assumes a sweep is already
            running with the provided ID. If None, will create a new sweep.
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)

    pprint.pprint(config)
    logging.info(f"Number of grid combinations: {np.cumprod([len(v['values']) for v in config['parameters'].values() if 'values' in v])[-1]}")  # noqa: E501
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")
    config['parameters']['device'] = {'value': device}

    if sweep_id is None:
        # start a new sweep if one is not already running (i.e. sweep_id is None)
        sweep_id = wandb.sweep(config)
    count = None if config['method'] == 'grid' else count
    logging.info(f"Run count: {count}")
    wandb.agent(sweep_id, model_pipeline, count=count)


if __name__ == '__main__':
    main()
