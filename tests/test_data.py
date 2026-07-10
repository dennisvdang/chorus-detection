"""Tests for dataset padding behavior."""

import numpy as np

from pytorch_core.data.dataset import ChorusDataset


def _empty_dataset(config):
    """Build a dataset with no songs so the padding helpers can be tested directly."""
    return ChorusDataset([], segments_dir=".", labels_dir=".", config=config)


def test_pad_labels_pads_short_with_minus_one(config):
    ds = _empty_dataset(config)
    labels = np.array([1, 0, 1])

    padded = ds._pad_labels(labels)

    assert len(padded) == config["data"]["max_meters"]
    assert list(padded[:3]) == [1, 0, 1]
    assert np.all(padded[3:] == -1)  # padding marker


def test_pad_labels_truncates_long(config):
    ds = _empty_dataset(config)
    labels = np.ones(config["data"]["max_meters"] + 50)

    padded = ds._pad_labels(labels)

    assert len(padded) == config["data"]["max_meters"]


def test_pad_segments_shapes(config):
    ds = _empty_dataset(config)
    n_features = config["data"]["n_features"]
    # Two meters with different frame counts
    segments = [np.random.rand(120, n_features), np.random.rand(400, n_features)]

    padded = ds._pad_segments(segments)

    assert padded.shape == (config["data"]["max_meters"],
                            config["data"]["max_frames"], n_features)
    # First meter padded to max_frames, extra frames are zero
    assert np.all(padded[0, 120:] == 0)
    # Second meter (400 frames) truncated to max_frames (300)
    assert not np.all(padded[1, :] == 0)
