"""Python port of Mixxx's constant-BPM post-processing (BeatUtils).

Mixxx's Queen Mary analyzer produces a raw beat list whose single beats jitter
by up to +-12 ms around their true positions. Mixxx does not report the raw
inter-beat tempo; it searches the beat list for long regions of constant beat
spacing, merges compatible regions from the start, middle, and end of the
track, and rounds the resulting BPM to a musically likely value (whole number
first, then 0.5, then 1/12 steps). That post-processing is why Mixxx shows
"120.00" where a raw average would show "119.87".

This module ports that logic - retrieveConstRegions(), makeConstBpm(), and
roundBpmWithinRange() from src/track/beatutils.cpp - to plain NumPy over beat
times in seconds. It is tracker-independent: feed it beats from any source
(the Queen Mary port in scripts/script.py, Beat This!, librosa) and it returns
one stable BPM for the track, or None when no constant region exists.

Constants match Mixxx exactly; comments give the original rationale.
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _round(x: float) -> float:
    """Round half away from zero, matching C++ round().

    Python's built-in round() rounds half to even, so the two disagree on
    exact ties (e.g. 1438.5). All values here are positive.
    """
    return math.floor(x + 0.5)

# A beat within 25 ms of its ironed position is "on grid": this small of a
# difference is inaudible, and it is > 2 * 12 ms, the step width of the QM
# beat detector.
MAX_SECS_PHASE_ERROR = 0.025
# Abort a region when the accumulated signed error drifts beyond this, which
# happens during an offset shift (for instance when the beat instrument
# changes).
MAX_SECS_PHASE_ERROR_SUM = 0.1
MAX_OUTLIERS_COUNT = 1
MIN_REGION_BEAT_COUNT = 16


def retrieve_const_regions(beat_times: Sequence[float]
                           ) -> List[Tuple[float, float]]:
    """Split a jittery beat list into regions of constant beat length.

    Walks the beats from the left, testing ever-shorter candidate regions until
    one fits: every beat within MAX_SECS_PHASE_ERROR of the ironed grid, at
    most one outlier, and no one-directional drift. Each accepted region is
    stored as (first_beat_time, mean_beat_length).

    Returns the accepted regions plus a final (last_beat_time, 0.0) marker,
    mirroring the C++ contract. Fewer than two beats give an empty list.
    """
    beats = np.asarray(beat_times, dtype=float)
    if beats.size < 2:
        return []

    left = 0
    right = beats.size - 1
    regions: List[Tuple[float, float]] = []

    while left < beats.size - 1:
        mean_beat_length = (beats[right] - beats[left]) / (right - left)
        outliers = 0
        ironed_beat = beats[left]
        phase_error_sum = 0.0
        completed = True
        for i in range(left + 1, right + 1):
            ironed_beat += mean_beat_length
            phase_error = ironed_beat - beats[i]
            phase_error_sum += phase_error
            if abs(phase_error) > MAX_SECS_PHASE_ERROR:
                outliers += 1
                # The first beat of a region must not be an outlier.
                if outliers > MAX_OUTLIERS_COUNT or i == left + 1:
                    completed = False
                    break
            if abs(phase_error_sum) > MAX_SECS_PHASE_ERROR_SUM:
                # Drifting away in one direction: mean_beat_length is off.
                completed = False
                break

        if completed:
            # Reject regions whose first and last beat lengths are correction
            # beats in the same direction, which bends the mean away from the
            # optimum.
            region_border_error = 0.0
            if right > left + 2:
                first_beat_length = beats[left + 1] - beats[left]
                last_beat_length = beats[right] - beats[right - 1]
                region_border_error = abs(first_beat_length + last_beat_length
                                          - 2 * mean_beat_length)
            if region_border_error < MAX_SECS_PHASE_ERROR / 2:
                regions.append((float(beats[left]), float(mean_beat_length)))
                left = right
                right = beats.size - 1
                continue
        right -= 1

    regions.append((float(beats[-1]), 0.0))
    return regions


def round_bpm_within_range(min_bpm: float, center_bpm: float,
                           max_bpm: float) -> float:
    """Round a BPM to the most musically likely value the range allows.

    Tries a whole number first. If the range is wide, allows 0.5 steps for
    slow tempos (halved fast tempos) and 2/3-based steps for fast ones, then
    1/12 steps, matching Mixxx's assumption of a metronome at a full BPM.
    """
    snap = _round(center_bpm)
    if min_bpm < snap < max_bpm:
        return float(snap)

    width = max_bpm - min_bpm
    if width > 0.5:
        if center_bpm < 85.0:
            # Can actually be up to 175 BPM: allow half-BPM values.
            return _round(center_bpm * 2) / 2
        if center_bpm > 127.0:
            # Optimize for the 2/3 value going down to 85.
            return _round(center_bpm / 3 * 2) * 3 / 2

    if width > 1.0 / 12:
        # Covers the 1/2, 2/3, and 3/4 multipliers.
        return _round(center_bpm * 12) / 12

    # More than ~75 beats and ~30 s: try a 1/12 snap, else keep the value.
    snap = _round(center_bpm * 12) / 12
    if min_bpm < snap < max_bpm:
        return snap
    return float(center_bpm)


def make_const_bpm(regions: List[Tuple[float, float]]) -> Optional[float]:
    """Derive one BPM from constant regions, favouring the longest evidence.

    Starts from the longest region, then tries to extend it with a compatible
    region near the start of the track and one near the end. Two regions are
    merged only when their tempo ranges overlap AND the beat count between
    them is unambiguous, meaning they share both tempo and phase. The final
    range feeds the rounding rule.

    Returns None when no region has nonzero length.
    """
    if not regions:
        return None

    mid_index = 0
    longest_length = 0.0
    longest_beat_length = 0.0
    for i in range(len(regions) - 1):
        length = regions[i + 1][0] - regions[i][0]
        if length > longest_length:
            longest_length = length
            longest_beat_length = regions[i][1]
            mid_index = i

    if longest_length == 0.0 or longest_beat_length <= 0.0:
        return None

    n_beats = int(longest_length / longest_beat_length + 0.5)
    beat_length_min = longest_beat_length - MAX_SECS_PHASE_ERROR / n_beats
    beat_length_max = longest_beat_length + MAX_SECS_PHASE_ERROR / n_beats

    start_index = mid_index

    # Extend with a compatible region near the start of the track.
    for i in range(mid_index):
        length = regions[i + 1][0] - regions[i][0]
        region_beats = int(length / regions[i][1] + 0.5) if regions[i][1] else 0
        if region_beats < MIN_REGION_BEAT_COUNT:
            continue  # too short, too unstable
        this_min = regions[i][1] - MAX_SECS_PHASE_ERROR / region_beats
        this_max = regions[i][1] + MAX_SECS_PHASE_ERROR / region_beats
        if this_min < longest_beat_length < this_max:
            new_length = regions[mid_index + 1][0] - regions[i][0]
            merged_min = max(beat_length_min, this_min)
            merged_max = min(beat_length_max, this_max)
            max_n = _round(new_length / merged_min)
            min_n = _round(new_length / merged_max)
            if min_n != max_n:
                continue  # ambiguous beat count: phases disagree
            new_beat_length = new_length / min_n
            if beat_length_min < new_beat_length < beat_length_max:
                longest_length = new_length
                longest_beat_length = new_beat_length
                n_beats = min_n
                beat_length_min = longest_beat_length - MAX_SECS_PHASE_ERROR / n_beats
                beat_length_max = longest_beat_length + MAX_SECS_PHASE_ERROR / n_beats
                start_index = i
                break

    # Extend with a compatible region near the end of the track.
    for i in range(len(regions) - 2, mid_index, -1):
        length = regions[i + 1][0] - regions[i][0]
        region_beats = int(length / regions[i][1] + 0.5) if regions[i][1] else 0
        if region_beats < MIN_REGION_BEAT_COUNT:
            continue
        this_min = regions[i][1] - MAX_SECS_PHASE_ERROR / region_beats
        this_max = regions[i][1] + MAX_SECS_PHASE_ERROR / region_beats
        if this_min < longest_beat_length < this_max:
            new_length = regions[i + 1][0] - regions[start_index][0]
            merged_min = max(beat_length_min, this_min)
            merged_max = min(beat_length_max, this_max)
            max_n = _round(new_length / merged_min)
            min_n = _round(new_length / merged_max)
            if min_n != max_n:
                continue
            new_beat_length = new_length / min_n
            if beat_length_min < new_beat_length < beat_length_max:
                longest_length = new_length
                longest_beat_length = new_beat_length
                n_beats = min_n
                break

    beat_length_min = longest_beat_length - MAX_SECS_PHASE_ERROR / n_beats
    beat_length_max = longest_beat_length + MAX_SECS_PHASE_ERROR / n_beats

    min_round_bpm = 60.0 / beat_length_max
    max_round_bpm = 60.0 / beat_length_min
    center_bpm = 60.0 / longest_beat_length
    return round_bpm_within_range(min_round_bpm, center_bpm, max_round_bpm)


def calculate_bpm(beat_times: Sequence[float]) -> Optional[float]:
    """One stable BPM for a track, from any tracker's beat times.

    Mirrors BeatUtils::calculateBpm: fewer than 2 beats give None; fewer than
    MIN_REGION_BEAT_COUNT beats fall back to the plain average; otherwise the
    constant-region pipeline runs.
    """
    beats = np.asarray(beat_times, dtype=float)
    if beats.size < 2:
        return None
    if beats.size < MIN_REGION_BEAT_COUNT:
        return 60.0 * (beats.size - 1) / (beats[-1] - beats[0])
    return make_const_bpm(retrieve_const_regions(beats))
