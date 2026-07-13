"""
Downbeat tracking and boundary snapping using Beat This! (CPJKU, ISMIR 2024).

The CRNN predicts chorus probabilities on a meter grid extrapolated from a
single global tempo estimate, so its boundaries can be off by a beat or a bar
when the grid phase drifts from the true downbeats. This module refines those
boundaries by snapping them onto downbeats detected by the Beat This! tracker
(https://github.com/CPJKU/beat_this), which handles metrical irregularity
without a strict 4/4 prior.

Beat This! is an optional dependency; use is_available() to check for it
before calling track_downbeats().
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

# Cached tracker instance keyed by device, since loading the checkpoint is slow
_trackers = {}


def is_available() -> bool:
    """Return True if the beat_this package is installed."""
    try:
        import beat_this  # noqa: F401
        return True
    except ImportError:
        return False


def track_downbeats(audio_path: str, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Track beats and downbeats in an audio file with Beat This!.

    Args:
        audio_path: Path to the audio file. Must be the same (silence-stripped)
            file the model predictions were computed on, so both share a timeline.
        device: Torch device for the tracker ("cpu" or "cuda").

    Returns:
        Tuple of (beat_times, downbeat_times) in seconds.
    """
    from beat_this.inference import File2Beats

    if device not in _trackers:
        _trackers[device] = File2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = _trackers[device](audio_path)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def snap_to_downbeats(times: Sequence[float], downbeats: np.ndarray,
                      max_shift: Optional[float] = None) -> np.ndarray:
    """Snap each time to its nearest downbeat.

    Args:
        times: Boundary times in seconds.
        downbeats: Downbeat times in seconds.
        max_shift: If set, times farther than this from every downbeat are
            left unchanged (guards against snapping across a tracker gap).

    Returns:
        Array of snapped times, same length as `times`.
    """
    times = np.asarray(times, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if times.size == 0 or downbeats.size == 0:
        return times.copy()

    nearest_idx = np.abs(downbeats[None, :] - times[:, None]).argmin(axis=1)
    snapped = downbeats[nearest_idx]
    if max_shift is not None:
        shift = np.abs(snapped - times)
        snapped = np.where(shift <= max_shift, snapped, times)
    return snapped


def energy_snap_to_downbeats(times: Sequence[float], downbeats: np.ndarray,
                             energy: np.ndarray, energy_times: np.ndarray,
                             direction: int = 1,
                             search_bars: float = 2.0) -> np.ndarray:
    """Snap each time to the best downbeat within +/- search_bars.

    Rather than snapping to the nearest downbeat, this searches all downbeats
    within a window of the boundary and picks the one with the largest energy
    change in the given direction, so a chorus/drop onset predicted a bar early
    or late still lands on the actual energy rise.

    Args:
        times: Boundary times in seconds.
        downbeats: Downbeat times in seconds.
        energy: Energy envelope (e.g. RMS), 1-D.
        energy_times: Time in seconds of each energy sample, parallel to energy.
        direction: +1 to favor energy rises (segment starts, drops),
            -1 to favor energy falls (segment ends).
        search_bars: Half-width of the search window, in bars.

    Returns:
        Array of snapped times, same length as `times`. Falls back to the
        nearest downbeat when no candidate shows an energy change in the
        requested direction.
    """
    times = np.asarray(times, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    energy = np.asarray(energy, dtype=float).ravel()
    energy_times = np.asarray(energy_times, dtype=float)
    if times.size == 0 or downbeats.size == 0:
        return times.copy()
    if downbeats.size < 2 or energy.size == 0:
        return snap_to_downbeats(times, downbeats)

    bar_length = float(np.median(np.diff(downbeats)))
    window = search_bars * bar_length

    snapped = np.empty_like(times)
    for i, t in enumerate(times):
        candidates = downbeats[(downbeats >= t - window) & (downbeats <= t + window)]
        if candidates.size == 0:
            snapped[i] = downbeats[np.abs(downbeats - t).argmin()]
            continue

        scores = np.full(candidates.size, -np.inf)
        for j, d in enumerate(candidates):
            before = energy[(energy_times >= d - bar_length) & (energy_times < d)]
            after = energy[(energy_times >= d) & (energy_times < d + bar_length)]
            if before.size and after.size:
                scores[j] = direction * (after.mean() - before.mean())

        if np.max(scores) > 0:
            snapped[i] = candidates[scores.argmax()]
        else:
            # No energy change in the requested direction; keep the nearest
            snapped[i] = candidates[np.abs(candidates - t).argmin()]
    return snapped


def snap_chorus_segments(start_times: Sequence[float], end_times: Sequence[float],
                         downbeats: np.ndarray,
                         max_shift: Optional[float] = None,
                         energy: Optional[np.ndarray] = None,
                         energy_times: Optional[np.ndarray] = None,
                         search_bars: float = 2.0) -> Tuple[List[float], List[float]]:
    """Snap chorus segment boundaries to downbeats.

    When an energy envelope is given, boundaries snap to the downbeat within
    +/- search_bars showing the strongest energy rise (starts) or fall (ends);
    otherwise they snap to the nearest downbeat. Segments that collapse
    (end <= start) after snapping are dropped, and segments that end up
    overlapping or touching are merged.

    Args:
        start_times: Chorus start times in seconds.
        end_times: Chorus end times in seconds, parallel to start_times.
        downbeats: Downbeat times in seconds.
        max_shift: Passed through to snap_to_downbeats() (nearest-snap only).
        energy: Optional energy envelope (e.g. RMS), 1-D.
        energy_times: Time of each energy sample, required with `energy`.
        search_bars: Half-width of the energy search window, in bars.

    Returns:
        Tuple of (snapped_start_times, snapped_end_times) as lists.
    """
    if energy is not None and energy_times is not None:
        starts = energy_snap_to_downbeats(start_times, downbeats, energy, energy_times,
                                          direction=1, search_bars=search_bars)
        ends = energy_snap_to_downbeats(end_times, downbeats, energy, energy_times,
                                        direction=-1, search_bars=search_bars)
    else:
        starts = snap_to_downbeats(start_times, downbeats, max_shift=max_shift)
        ends = snap_to_downbeats(end_times, downbeats, max_shift=max_shift)

    out_starts: List[float] = []
    out_ends: List[float] = []
    for start, end in zip(starts, ends):
        if end <= start:
            continue
        if out_starts and start <= out_ends[-1]:
            out_ends[-1] = max(out_ends[-1], end)
        else:
            out_starts.append(float(start))
            out_ends.append(float(end))
    return out_starts, out_ends
