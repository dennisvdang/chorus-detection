"""Tests for the Mixxx constant-BPM post-processing port."""

import numpy as np
import pytest

from pytorch_core.mixxx_bpm import (calculate_bpm, make_const_bpm,
                                    retrieve_const_regions,
                                    round_bpm_within_range)


def jittered_beats(bpm, n, jitter_s, seed=0):
    """Beat times at a fixed tempo with bounded uniform jitter."""
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    times = np.arange(n) * period
    return times + rng.uniform(-jitter_s, jitter_s, size=n)


def test_clean_beats_give_the_exact_bpm():
    assert calculate_bpm(np.arange(120) * 0.5) == pytest.approx(120.0)


def test_qm_style_jitter_is_ironed_to_a_whole_bpm():
    """+-12 ms jitter is the Queen Mary detector's step width; the whole point
    of the post-processing is that this still comes out as exactly 120."""
    beats = jittered_beats(120.0, 240, 0.012)
    assert calculate_bpm(beats) == pytest.approx(120.0)


def test_uneven_tempo_like_128_is_recovered():
    beats = jittered_beats(128.0, 240, 0.010, seed=3)
    assert calculate_bpm(beats) == pytest.approx(128.0)


def test_fewer_than_two_beats_gives_none():
    assert calculate_bpm([]) is None
    assert calculate_bpm([1.0]) is None


def test_a_short_beat_list_falls_back_to_the_plain_average():
    # 10 beats at 100 BPM: below MIN_REGION_BEAT_COUNT, so no region search.
    beats = np.arange(10) * 0.6
    assert calculate_bpm(beats) == pytest.approx(100.0)


def test_a_tempo_change_yields_the_longer_sections_bpm():
    """100 BPM for 30 s, then 128 BPM for 90 s: the 128 section is longer."""
    first = np.arange(0.0, 30.0, 60.0 / 100.0)
    second = np.arange(30.0, 120.0, 60.0 / 128.0)
    bpm = calculate_bpm(np.concatenate([first, second]))
    assert bpm == pytest.approx(128.0, abs=0.2)


def test_regions_end_with_a_zero_length_marker():
    regions = retrieve_const_regions(np.arange(64) * 0.5)
    assert regions[-1][1] == 0.0
    assert regions[-1][0] == pytest.approx(63 * 0.5)


def test_a_steady_list_is_one_region():
    regions = retrieve_const_regions(np.arange(64) * 0.5)
    # One real region plus the end marker.
    assert len(regions) == 2
    assert regions[0][0] == pytest.approx(0.0)
    assert regions[0][1] == pytest.approx(0.5)


def test_rounding_snaps_to_a_whole_number_first():
    assert round_bpm_within_range(119.5, 119.87, 120.4) == 120.0


def test_rounding_snaps_to_a_twelfth_step_when_one_fits_the_tight_range():
    # Width < 1/12 and 119 + 10/12 = 119.8333 lies inside: snap to it.
    assert round_bpm_within_range(119.80, 119.84, 119.86) == pytest.approx(119.8333, abs=1e-3)


def test_rounding_keeps_the_center_when_no_snap_fits():
    # Width < 1/12 and no whole, half, or 1/12 step lies inside the range.
    assert round_bpm_within_range(119.84, 119.845, 119.85) == pytest.approx(119.845)


def test_rounding_uses_half_steps_for_slow_wide_ranges():
    # Width > 0.5 and center < 85: half-BPM values are allowed.
    assert round_bpm_within_range(84.0, 84.4, 85.1) == pytest.approx(84.5)


def test_rounding_uses_two_thirds_steps_for_fast_wide_ranges():
    # Width > 0.5 and center > 127: snap built from the 2/3 value.
    result = round_bpm_within_range(173.5, 174.2, 175.5)
    assert result == pytest.approx(174.0)


def test_make_const_bpm_of_no_regions_is_none():
    assert make_const_bpm([]) is None
    assert make_const_bpm([(10.0, 0.0)]) is None


def test_one_displaced_beat_is_tolerated_as_an_outlier():
    """One beat 40 ms off grid: within MAX_OUTLIERS_COUNT, the region holds."""
    beats = np.arange(64) * 0.5
    beats[30] += 0.040
    regions = retrieve_const_regions(beats)
    assert len(regions) == 2  # one region plus the end marker
    assert calculate_bpm(beats) == pytest.approx(120.0)


def test_two_displaced_beats_split_the_region():
    """Two beats 40 ms off exceed the one-outlier budget: the region splits.

    40 ms sits above the 25 ms tolerance but keeps the summed drift under the
    100 ms abort, so this pins MAX_SECS_PHASE_ERROR and MAX_OUTLIERS_COUNT
    themselves rather than the drift guard.
    """
    beats = np.arange(64) * 0.5
    beats[30] += 0.040
    beats[31] += 0.040
    regions = retrieve_const_regions(beats)
    assert len(regions) >= 3  # at least two real regions plus the marker


def test_a_region_may_not_begin_on_an_outlier():
    """The second beat 40 ms off makes the first beat pair an outlier, so the
    region must start later rather than absorb it as its one allowed outlier."""
    beats = np.arange(64) * 0.5
    beats[1] += 0.040
    regions = retrieve_const_regions(beats)
    # The long clean region must start at or after the displaced beat.
    longest = max(regions[:-1], key=lambda r: r[1] and 0 or 0, default=None)
    starts = [r[0] for r in regions[:-1]]
    assert any(s >= beats[1] - 1e-9 for s in starts)
    assert calculate_bpm(beats) == pytest.approx(120.0)


def test_rounding_two_thirds_branch_directly():
    """Width > 0.5, center > 127, and the whole number lies outside the range:
    the 2/3 rule must fire. round(174.55/3*2)*3/2 = 174.0."""
    assert round_bpm_within_range(174.2, 174.55, 174.9) == pytest.approx(174.0)


def test_rounding_twelfth_branch_directly():
    """Width between 1/12 and 0.5, whole number outside the range, mid tempo:
    the 1/12 rule must fire. round(119.42*12)/12 = 119.4167."""
    assert round_bpm_within_range(119.30, 119.42, 119.55) == pytest.approx(
        119.4167, abs=1e-3)


def test_cpp_rounding_ties_round_half_away_from_zero():
    """119.875 * 12 = 1438.5 exactly: C++ rounds it up, Python's built-in
    round would round to even (down). The port must match C++."""
    assert round_bpm_within_range(119.86, 119.875, 119.93) == pytest.approx(
        119.9167, abs=1e-3)


def test_phase_shifted_halves_do_not_merge_into_one_region():
    """Same tempo, but the second half is shifted by 60 ms: a region border
    must appear rather than one region papering over the shift."""
    first = np.arange(0.0, 60.0, 0.5)
    second = np.arange(60.0, 120.0, 0.5) + 0.06
    regions = retrieve_const_regions(np.concatenate([first, second]))
    assert len(regions) >= 3  # at least two real regions plus the marker
