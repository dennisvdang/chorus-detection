"""Shared fixtures for the chorus detection test suite."""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def config():
    """Load the default model/training configuration."""
    with open(os.path.join(REPO_ROOT, "config", "default.yaml")) as f:
        return yaml.safe_load(f)
