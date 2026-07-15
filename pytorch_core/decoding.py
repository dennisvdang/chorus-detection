"""Viterbi decoding of per-meter chorus probabilities.

smooth_predictions() thresholds probability at 0.5 and drops runs shorter than
two meters, with no notion of how likely a section switch is or how long a
chorus should be. This decoder frames the meter sequence as a two-state HMM
(0 = non-chorus, 1 = chorus): each meter pays -log(prob) to be a chorus and
-log(1-prob) not to be, plus a fixed penalty for switching state between
adjacent meters. Viterbi finds the lowest-cost path, then runs shorter than
min_bars are dissolved, encoding the prior that choruses span whole phrases.
"""

import numpy as np


def viterbi_chorus(probs: np.ndarray, switch_penalty: float = 2.0,
                   min_bars: int = 4) -> np.ndarray:
    """Decode a 0/1 chorus path from per-meter probabilities."""
    probs = np.clip(np.asarray(probs, dtype=float).ravel(), 1e-6, 1 - 1e-6)
    n = probs.size
    if n == 0:
        return np.zeros(0, dtype=int)

    emit = np.stack([-np.log(1 - probs), -np.log(probs)], axis=1)  # [n, 2]
    cost = np.full((n, 2), np.inf)
    back = np.zeros((n, 2), dtype=int)
    cost[0] = emit[0]
    for i in range(1, n):
        for s in range(2):
            stay = cost[i - 1, s]
            switch = cost[i - 1, 1 - s] + switch_penalty
            if stay <= switch:
                cost[i, s], back[i, s] = stay + emit[i, s], s
            else:
                cost[i, s], back[i, s] = switch + emit[i, s], 1 - s

    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmin(cost[-1]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    return _dissolve_short_runs(path, min_bars)


def _dissolve_short_runs(path: np.ndarray, min_bars: int) -> np.ndarray:
    """Enforce the minimum-phrase-length prior on the decoded path.

    A chorus run shorter than min_bars is a blip and is dropped. A non-chorus
    run shorter than min_bars is filled only when it sits between two chorus
    runs (an interior gap that fragmented one chorus); a short run at the song's
    head or tail is left alone so intros/outros are not absorbed. Runs are read
    from the original path, so mutations never corrupt run detection.
    """
    if min_bars <= 1 or path.size == 0:
        return path
    out = path.copy()

    runs = []  # (start, end, label) on the original path
    start = 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start]:
            runs.append((start, i, int(path[start])))
            start = i

    for idx, (s, e, label) in enumerate(runs):
        if e - s >= min_bars:
            continue
        if label == 1:
            out[s:e] = 0
        else:
            left_chorus = idx > 0 and runs[idx - 1][2] == 1
            right_chorus = idx < len(runs) - 1 and runs[idx + 1][2] == 1
            if left_chorus and right_chorus:
                out[s:e] = 1
    return out
