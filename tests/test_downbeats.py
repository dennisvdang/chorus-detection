"""Tests for downbeat snapping (pytorch_core.downbeats)."""

import numpy as np
import pytest

from pytorch_core.downbeats import (
    energy_snap_to_downbeats,
    is_available,
    snap_chorus_segments,
    snap_to_downbeats,
    track_downbeats,
)


def step_energy(rise_at, fall_at, duration=40.0, dt=0.05):
    """Energy envelope that is low outside [rise_at, fall_at) and high inside."""
    energy_times = np.arange(0.0, duration, dt)
    energy = np.where((energy_times >= rise_at) & (energy_times < fall_at), 1.0, 0.1)
    return energy, energy_times


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


class TestEnergySnapToDownbeats:
    downbeats = np.arange(0.0, 40.0, 2.0)

    def test_snaps_to_energy_rise_a_bar_away(self):
        # Boundary predicted a bar early; nearest downbeat is 8.0 but the
        # energy rise (the drop) is at 10.0, within the +/-2 bar window
        energy, energy_times = step_energy(rise_at=10.0, fall_at=30.0)
        snapped = energy_snap_to_downbeats([8.9], self.downbeats, energy, energy_times,
                                           direction=1, search_bars=2.0)
        np.testing.assert_allclose(snapped, [10.0])

    def test_snaps_end_to_energy_fall(self):
        energy, energy_times = step_energy(rise_at=10.0, fall_at=30.0)
        snapped = energy_snap_to_downbeats([31.1], self.downbeats, energy, energy_times,
                                           direction=-1, search_bars=2.0)
        np.testing.assert_allclose(snapped, [30.0])

    def test_flat_energy_falls_back_to_nearest(self):
        energy_times = np.arange(0.0, 40.0, 0.05)
        energy = np.ones_like(energy_times)
        snapped = energy_snap_to_downbeats([3.9], self.downbeats, energy, energy_times,
                                           direction=1, search_bars=2.0)
        np.testing.assert_allclose(snapped, [4.0])

    def test_rise_outside_window_not_chosen(self):
        # Energy rise at 20.0 is 5 bars from the boundary; window is 2 bars,
        # so the boundary snaps to the nearest downbeat instead
        energy, energy_times = step_energy(rise_at=20.0, fall_at=30.0)
        snapped = energy_snap_to_downbeats([9.9], self.downbeats, energy, energy_times,
                                           direction=1, search_bars=2.0)
        np.testing.assert_allclose(snapped, [10.0])


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

    def test_energy_aware_segment_snap(self):
        # Start predicted a bar before the drop, end a bar after the fall:
        # with energy the segment locks onto [10.0, 30.0]
        downbeats = np.arange(0.0, 40.0, 2.0)
        energy, energy_times = step_energy(rise_at=10.0, fall_at=30.0)
        starts, ends = snap_chorus_segments([8.9], [31.2], downbeats,
                                            energy=energy, energy_times=energy_times)
        assert starts == [10.0]
        assert ends == [30.0]


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
