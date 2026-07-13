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


@pytest.fixture
def processed_audio_path():
    """Path to one processed training audio file, skipping if none exist."""
    audio_dir = os.path.join(REPO_ROOT, "data", "audio", "processed")
    if os.path.isdir(audio_dir):
        for name in sorted(os.listdir(audio_dir)):
            if name.endswith(".mp3"):
                return os.path.join(audio_dir, name)
    pytest.skip("no processed audio files available")
