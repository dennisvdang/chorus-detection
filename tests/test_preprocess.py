"""Integration test for the preprocessing pipeline."""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import preprocess  # noqa: E402


def test_generate_and_align_labels_marks_chorus_meters():
    """A meter with >=25% chorus frames is labeled 1, otherwise 0."""
    df = pd.DataFrame({
        "label": ["chorus"],
        "start_time": [0.0],
        "end_time": [1.0],
    })
    # 4 meters of 100 frames each; chorus covers frames 0..~93 (1.0s at 12kHz/128 hop)
    meter_grid = np.array([0, 100, 200, 300, 400])
    labels = preprocess.generate_and_align_labels(
        df, sr=12000, hop_length=128, n_frames=400, meter_grid_frames=meter_grid)

    assert labels[0] == 1  # first meter is mostly chorus
    assert labels[-1] == 0  # last meter has no chorus


def test_positional_encoding_shape():
    pe = preprocess.positional_encoding(10, 15)
    assert pe.shape == (10, 15)


@pytest.mark.slow
def test_process_one_song(tmp_path):
    """End-to-end: process a real audio file and verify output shapes/alignment."""
    audio_path = os.path.join(REPO_ROOT, "data", "audio", "processed", "1.mp3")
    csv_path = os.path.join(REPO_ROOT, "data", "clean_labeled.csv")
    if not os.path.exists(audio_path) or not os.path.exists(csv_path):
        pytest.skip("audio/CSV data not available")

    df = pd.read_csv(csv_path)
    song_df = df.loc[df["SongID"] == 1]

    seg_dir, lab_dir = tmp_path / "seg", tmp_path / "lab"
    seg_dir.mkdir()
    lab_dir.mkdir()

    n_meters, max_frames = preprocess.process_song(
        1, song_df, audio_path, str(seg_dir), str(lab_dir))

    with open(seg_dir / "1_data.pkl", "rb") as f:
        segments = pickle.load(f)
    with open(lab_dir / "1_labels.pkl", "rb") as f:
        labels = pickle.load(f)

    assert len(segments) == len(labels) == n_meters
    assert all(seg.shape[1] == 15 for seg in segments)  # 15 combined features
    assert set(np.unique(labels)).issubset({0, 1})
