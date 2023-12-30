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

# from helpsk.logging import log_function_call, log_timer
# import source.config.config as config


logging.config.fileConfig(
    os.path.join(os.getcwd(), '/code/source/config/logging.conf'),
    disable_existing_loggers=False,
)


def get_data():  # noqa
    """This is called by model_pipeline."""
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
def main():
    """
    Logic For Extracting and Transforming Datasets
    """
    pass


@main.command()
@click.option('--config_file', '-c', type=str)
@click.option('--device', type=str, default=None)
def run(config_file: str, device: str | None = None) -> None:
    """Execute a 'run' on Weights and Biases."""
    with open(config_file) as f:
        config = yaml.safe_load(f)

    pprint.pprint(config)
    logging.info(f"Number of grid combinations: {np.cumprod([len(v['values']) for v in config['parameters'].values() if 'values' in v])[-1]}")  # noqa: E501
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")
    config['device'] = device
    _ = model_pipeline(config)


@main.command()
@click.option('--config_file', '-c', type=str)
@click.option('--device', type=str, default=None)
@click.option('--count', type=int, default=None)
def sweep(config_file: str, device: str | None = None, count: int | None = None) -> None:
    """Execute a 'sweep' on Weights and Biases."""
    with open(config_file) as f:
        config = yaml.safe_load(f)

    pprint.pprint(config)
    logging.info(f"Number of grid combinations: {np.cumprod([len(v['values']) for v in config['parameters'].values() if 'values' in v])[-1]}")  # noqa: E501
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")
    config['parameters']['device'] = {'value': device}

    sweep_id = wandb.sweep(config, project=config['parameters']['project']['value'])
    count = None if config['method'] == 'grid' else count
    logging.info(f"Run count: {count}")
    wandb.agent(sweep_id, model_pipeline, count=count)


if __name__ == '__main__':
    main()
