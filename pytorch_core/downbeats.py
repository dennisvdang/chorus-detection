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


def snap_chorus_segments(start_times: Sequence[float], end_times: Sequence[float],
                         downbeats: np.ndarray,
                         max_shift: Optional[float] = None) -> Tuple[List[float], List[float]]:
    """Snap chorus segment boundaries to downbeats.

    Segments that collapse (end <= start) after snapping are dropped, and
    segments that end up overlapping or touching are merged.

    Args:
        start_times: Chorus start times in seconds.
        end_times: Chorus end times in seconds, parallel to start_times.
        downbeats: Downbeat times in seconds.
        max_shift: Passed through to snap_to_downbeats().

    Returns:
        Tuple of (snapped_start_times, snapped_end_times) as lists.
    """
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
