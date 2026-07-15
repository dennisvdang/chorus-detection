import librosa
import numpy as np

from pytorch_core.downbeats import create_downbeat_meter_grid


def test_grid_starts_at_zero_ends_at_nframes_and_increasing():
    downbeats = np.array([0.5, 2.5, 4.5])  # seconds
    sr, hop, n_frames = 12000, 128, 500
    grid = create_downbeat_meter_grid(downbeats, n_frames, sr, hop)
    assert grid[0] == 0
    assert grid[-1] == n_frames
    assert np.all(np.diff(grid) > 0)


def test_grid_matches_downbeat_frames_between_ends():
    downbeats = np.array([0.5, 2.5, 4.5])
    sr, hop, n_frames = 12000, 128, 500
    grid = create_downbeat_meter_grid(downbeats, n_frames, sr, hop)
    expected = librosa.time_to_frames(downbeats, sr=sr, hop_length=hop)
    for e in expected:
        assert e in grid
