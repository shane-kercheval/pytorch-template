"""
This file contains the functions for the command line interface. The makefile calls the commands
defined in this file.

For help in terminal, navigate to the project directory, run the docker container, and from within
the container run the following examples:
    - `python3.9 source/scripts/commands.py --help`
    - `python3.9 source/scripts/commands.py extract --help`
"""
import logging.config
import logging
import os
import click
import pandas as pd

from helpsk.logging import log_function_call, log_timer
import source.config.config as config

logging.config.fileConfig(
    "source/config/logging_to_file.conf",
    defaults={'logfilename': 'output/log.log'},
    disable_existing_loggers=False
)


@click.group()
def main():
    """
    Logic For Extracting and Transforming Datasets
    """
    pass


@log_function_call
@log_timer
def extract_auto_mpg(output_directory: str):
    """This function downloads the auto-mpg dataset and saves it to `output_directory`."""
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    logging.info(f"Downloading auto-mpg data from {url}")
    df = pd.read_csv(
        url,
        names=[
            'MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration',
            'Model Year', 'Origin'
        ],
        na_values="?",
        comment='\t',
        sep=" ",
        skipinitialspace=True
    )
    logging.info(
        f"Credit data downloaded with {df.shape[0]} "
        f"rows and {df.shape[1]} columns."
    )
    output_file = os.path.join(output_directory, 'auto_mpg.pkl')
    logging.info(f"Saving data to `{output_file}`")
    df.to_pickle(output_file)


@main.command()
def extract():
    """This function downloads the credit data from openml.org."""
    extract_auto_mpg(output_directory=config.dir_data_raw())


if __name__ == '__main__':
    main()
