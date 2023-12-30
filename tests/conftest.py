"""Defines test fixtures for pytest unit-tests."""
from typing import Tuple
import pytest
import torch
import numpy as np
import os

import logging.config

from source.domain.experiment import get_data

logging.config.fileConfig(
    os.path.join(os.getcwd(), 'source/config/logging_to_file.conf'),
    defaults={'logfilename': os.path.join(os.getcwd(), 'tests/test_files/log.log')},
    disable_existing_loggers=False,
)


@pytest.fixture(scope='session')
def dummy_x_y() -> Tuple[torch.Tensor, torch.Tensor]:  # noqa
    x = np.arange(20, dtype='float32').reshape((10, 2))
    y = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype='float32')
    x = torch.from_numpy((x - np.mean(x)) / np.std(x)).float()
    y = torch.from_numpy(y).float()
    return x, y


@pytest.fixture(scope='session')
def mnist_fc():  # noqa
    return get_data(architecture='FC')


@pytest.fixture(scope='session')
def mnist_cnn():  # noqa
    return get_data(architecture='CNN')
