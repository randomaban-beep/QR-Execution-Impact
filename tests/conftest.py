"""Pytest configuration and shared fixtures for exec-impact test suite."""

import numpy as np
import pytest

from exec_impact.config import Config


@pytest.fixture(autouse=True)
def set_random_seeds() -> None:
    """Ensure determinism across tests by fixing the numpy and random seed."""
    np.random.seed(42)


@pytest.fixture
def test_config() -> Config:
    """Fixture providing standard configuration."""
    return Config()
