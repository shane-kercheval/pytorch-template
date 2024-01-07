"""Command line interface for running experiments with Weights and Biases."""

import pprint
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
import wandb
import yaml
import logging.config
import logging
import os
import click
from source.library.experiment import (
    make_model,
    model_pipeline,
    get_available_device,
    transform_data,
    predict as pred,
)
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


@main.command()
@click.option('-x_parquet_path', type=str)
@click.option('-predictions_path', type=str)
@click.option('-w_and_b_run_id', type=str)
@click.option('-w_and_b_user', type=str, default='shane-kercheval')
@click.option('-w_and_b_project', type=str, default='pytorch-demo')
@click.option('-w_and_b_model_name', type=str, default='model_state.pth')
@click.option('-y_parquet_path', type=str, default=None)
def predict(
        x_parquet_path: str,
        predictions_path: str,
        w_and_b_run_id: str,
        w_and_b_user: str = 'shane-kercheval',
        w_and_b_project: str = 'pytorch-demo',
        w_and_b_model_name: str = 'model_state.pth',
        y_parquet_path: str | None = None) -> None:
    """
    Predict on a dataset loaded from parquet file. Uses the model state and config from a
    particular project/run.

    NOTE: a better way to implement this would be to use the Weights and Biases model registry.

    Args:
        x_parquet_path:
            Path to parquet file containing x data.
        predictions_path:
            Path to parquet file where predictions will be saved.
        arch:
            Architecture to use for prediction. Must be one of ['fc', 'cnn'].
        w_and_b_run_id:
            Weights and Biases run id. Model state is assumed to be `model_state.pt`.
        w_and_b_user:
            Weights and Biases user name.
        w_and_b_project:
            Weights and Biases project name.
        w_and_b_model_name:
            Name of model state file.
        y_parquet_path:
            Path to parquet file containing y data. If provided, accuracy will be printed.
            If None, will not be used.
    """
    logging.info(f"Loading model from run {w_and_b_run_id}.")
    run_path = f'{w_and_b_user}/{w_and_b_project}/{w_and_b_run_id}'
    model_state = torch.load(wandb.restore(w_and_b_model_name, run_path=run_path).name)
    model_config = wandb.restore('config.yaml',run_path=run_path)
    with open(model_config.name) as f:
        model_config = yaml.safe_load(f)
    # model config is stored in a different format in wandb than in the config file
    model_config = {
        k: v['value'] for k, v in model_config.items()
        if isinstance(v, dict) and 'value' in v
    }
    model = make_model(input_size=28*28, output_size=10, config=model_config)
    model.load_state_dict(model_state)
    # load data
    x = pd.read_parquet(x_parquet_path)
    y = pd.read_parquet(y_parquet_path).iloc[:, 0] if y_parquet_path is not None else None
    x, y = transform_data(x=x, y=y, architecture=model_config['architecture'])
    assert len(x) == len(y)
    # predict
    device = get_available_device()
    predictions = pred(model=model, x=x, device=device, probs=False).cpu()
    # save predictions
    pd.DataFrame(predictions).to_parquet(predictions_path)
    if y is not None:
        print(f"Accuracy: {(predictions.numpy() == y.numpy()).mean():.2%}")


####
# functions used for testing
####
@main.command()
def test_data() -> None:
    """Function used to create a test dataset used to mimic predicting on a new dataset."""
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_test, _, y_test, _ = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    # save x_test as parquet
    assert len(x) == len(y)
    pd.DataFrame(x_test).to_parquet('data/external/x_test.parquet')
    # save y_test as parquet
    pd.DataFrame(y_test).to_parquet('data/external/y_test.parquet')


if __name__ == '__main__':
    main()
