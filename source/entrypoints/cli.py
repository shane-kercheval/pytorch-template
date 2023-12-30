"""Command line interface for running experiments with Weights and Biases."""

import pprint
import numpy as np
import wandb
import yaml
import logging.config
import logging
import os
import click
import torch
from source.domain.experiment import model_pipeline

from dotenv import load_dotenv
load_dotenv()  # EXPECTS WANDB_API_KEY TO BE SET IN .env FILE


logging.config.fileConfig(
    os.path.join(os.getcwd(), '/code/source/config/logging.conf'),
    disable_existing_loggers=False,
)


@click.group()
def main() -> None:
    """Command Line Logic."""
    pass


@main.command()
@click.option('-config_file', '-c', type=str)
@click.option('-device', type=str, default=None)
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
@click.option('-config_file', type=str)
@click.option('-device', type=str, default=None)
@click.option('-count', type=int, default=None)
def sweep(
        config_file: str,
        device: str | None = None,
        count: int | None = None) -> None:
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
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)

    pprint.pprint(config)
    logging.info(f"Number of grid combinations: {np.cumprod([len(v['values']) for v in config['parameters'].values() if 'values' in v])[-1]}")  # noqa: E501
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")
    config['parameters']['device'] = {'value': device}

    sweep_id = wandb.sweep(config)
    count = None if config['method'] == 'grid' else count
    logging.info(f"Run count: {count}")
    wandb.agent(sweep_id, model_pipeline, count=count)


if __name__ == '__main__':
    main()
