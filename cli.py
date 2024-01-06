"""Command line interface for running experiments with Weights and Biases."""

import pprint
import numpy as np
import wandb
import yaml
import logging.config
import logging
import os
import click
from source.domain.experiment import model_pipeline, get_available_device

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
    logging.info(f"Configuration file: {config_file}")
    with open(config_file) as f:
        config = yaml.safe_load(f)
        pprint.pprint(config)
    device = get_available_device() if device is None else device
    logging.info(f"Device: {device}")
    config['device'] = device
    _ = model_pipeline(config)


@main.command()
@click.option('-config_file', type=str)
@click.option('-device', type=str, default=None)
@click.option('-runs', type=int, default=None)
def sweep(
        config_file: str,
        device: str | None = None,
        runs: int | None = None) -> None:
    """
    Execute a 'sweep' (of multiple runs) on Weights and Biases.

    Args:
        config_file:
            Path to the YAML configuration file.
        device:
            Device to use for training. If None, will use 'cuda' if available, 'mps' if available,
            and 'cpu' otherwise.
        runs:
            Number of runs to execute. If None, will execute all runs. Ignored if
            config['method'] == 'grid'.
    """
    logging.info(f"Configuration file: {config_file}")
    with open(config_file) as f:
        config = yaml.safe_load(f)
    pprint.pprint(config)
    num_combinations = _num_combinations(config)
    logging.info(f"Number of parameter combinations: {num_combinations}")
    if device is None:
        device = get_available_device()
    logging.info(f"Device: {device}")
    config['parameters']['device'] = {'value': device}

    sweep_id = wandb.sweep(config)
    if config['method'] == 'grid':
        runs = None
        logging.info(f"Running grid search with {num_combinations} combinations.")
    else:
        logging.info(f"Running {config['method']} search with {runs} runs.")

    wandb.agent(sweep_id, model_pipeline, count=runs)


@main.command()
@click.option('-config_file', type=str)
def num_combinations(config_file: str) -> None:
    """Print the number of grid combinations."""
    print(f"Configuration file: {config_file}")
    with open(config_file) as f:
        config = yaml.safe_load(f)
    print(f"Number of parameter combinations: {_num_combinations(config)}")


def _num_combinations(config: dict) -> int:
    return np.cumprod([len(v['values']) for v in config['parameters'].values() if 'values' in v])[-1]  # noqa

if __name__ == '__main__':
    main()
