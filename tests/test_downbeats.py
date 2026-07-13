"""Tests for downbeat snapping (pytorch_core.downbeats)."""

import numpy as np
import pytest

from pytorch_core.downbeats import (
    is_available,
    snap_chorus_segments,
    snap_to_downbeats,
    track_downbeats,
)


class TestSnapToDownbeats:
    def test_snaps_to_nearest_downbeat(self):
        downbeats = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        times = [1.9, 4.4, 7.2]
        snapped = snap_to_downbeats(times, downbeats)
        np.testing.assert_allclose(snapped, [2.0, 4.0, 8.0])

    def test_exact_downbeat_unchanged(self):
        downbeats = np.array([0.0, 2.0, 4.0])
        snapped = snap_to_downbeats([2.0], downbeats)
        np.testing.assert_allclose(snapped, [2.0])

    def test_max_shift_keeps_distant_times(self):
        downbeats = np.array([0.0, 10.0])
        # 5.0 is 5 s from both downbeats; with max_shift=1 it must not move
        snapped = snap_to_downbeats([5.0, 9.5], downbeats, max_shift=1.0)
        np.testing.assert_allclose(snapped, [5.0, 10.0])

    def test_empty_times(self):
        snapped = snap_to_downbeats([], np.array([1.0, 2.0]))
        assert snapped.size == 0

    def test_empty_downbeats_returns_times_unchanged(self):
        snapped = snap_to_downbeats([1.5, 3.0], np.array([]))
        np.testing.assert_allclose(snapped, [1.5, 3.0])


class TestSnapChorusSegments:
    def test_snaps_starts_and_ends(self):
        downbeats = np.arange(0.0, 20.0, 2.0)
        starts, ends = snap_chorus_segments([3.9], [9.8], downbeats)
        assert starts == [4.0]
        assert ends == [10.0]

    def test_drops_collapsed_segment(self):
        # Both boundaries snap to the same downbeat -> degenerate segment
        downbeats = np.array([0.0, 10.0, 20.0])
        starts, ends = snap_chorus_segments([9.0], [11.0], downbeats)
        assert starts == []
        assert ends == []

    def test_merges_overlapping_segments(self):
        downbeats = np.arange(0.0, 30.0, 2.0)
        # Second segment starts where the first ends after snapping
        starts, ends = snap_chorus_segments([2.1, 10.1], [10.2, 18.0], downbeats)
        assert starts == [2.0]
        assert ends == [18.0]

    def test_preserves_separate_segments(self):
        downbeats = np.arange(0.0, 40.0, 2.0)
        starts, ends = snap_chorus_segments([2.1, 20.3], [10.2, 30.1], downbeats)
        assert starts == [2.0, 20.0]
        assert ends == [10.0, 30.0]


@pytest.mark.slow
@pytest.mark.skipif(not is_available(), reason="beat_this not installed")
def test_track_downbeats_on_audio(processed_audio_path):
    """Integration test: Beat This! produces a plausible downbeat grid."""
    beats, downbeats = track_downbeats(processed_audio_path)
    assert len(beats) > 0
    assert len(downbeats) > 0
    # Downbeats are a subset of the metrical structure: increasing, sensible spacing
    assert np.all(np.diff(downbeats) > 0)
    intervals = np.diff(downbeats)
    assert 1.0 < np.median(intervals) < 6.0  # one bar at 40-240 bpm in 4/4
