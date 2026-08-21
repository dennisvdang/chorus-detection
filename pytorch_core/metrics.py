"""Canonical metric definitions for chorus boundary evaluation.

This module is the single source of truth for how chorus predictions are
scored; do not redefine these functions elsewhere.

- Anchor metrics score only the first chorus start.
- Interval metrics score every boundary and every frame: frame-level
  precision/recall/F1, boundary hit rates, and median boundary error.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

# A first-chorus prediction within this fraction of a bar counts as exact.
EXACT_FRACTION = 0.25

# Assumed beats per bar when converting beat periods to bar lengths.
BEATS_PER_BAR = 4


# --------------------------------------------------------------------------
# Anchor metrics: the first chorus start only
# --------------------------------------------------------------------------

def anchor_error(pred_starts: Sequence[float],
                 true_starts: Sequence[float]) -> Optional[float]:
    """Signed seconds from the first predicted chorus start to the labelled one.

    Positive means the prediction is late. Returns None when either list is
    empty.
    """
    if not len(pred_starts) or not len(true_starts):
        return None
    return float(pred_starts[0]) - float(true_starts[0])


def classify_anchor(error_s: Optional[float],
                    bar_length_s: Optional[float]) -> str:
    """Classify one first-chorus placement as exact, within_1_bar, or off.

    "exact" is within EXACT_FRACTION of a bar; "within_1_bar" is within one
    bar. A missing error or a non-positive bar length classifies as "off".
    """
    if error_s is None or bar_length_s is None or bar_length_s <= 0:
        return "off"
    size = abs(float(error_s))
    if size <= EXACT_FRACTION * bar_length_s:
        return "exact"
    if size <= bar_length_s + 1e-6:
        return "within_1_bar"
    return "off"


def summarize_classifications(records: Sequence[str]) -> dict:
    """Counts and rates for a list of classify_anchor() results."""
    total = len(records)
    exact = sum(1 for r in records if r == "exact")
    within = exact + sum(1 for r in records if r == "within_1_bar")
    if total == 0:
        return {"songs": 0, "exact": 0, "within_1_bar": 0,
                "exact_rate": 0.0, "within_1_bar_rate": 0.0}
    return {"songs": total, "exact": exact, "within_1_bar": within,
            "exact_rate": exact / total, "within_1_bar_rate": within / total}


def merge_adjacent_segments(starts: Sequence[float], ends: Sequence[float],
                            tol: float = 1e-6) -> Tuple[List[float], List[float]]:
    """Merge labelled segments whose start touches the previous segment's end.

    Adjacent label rows with equal end/start times are one continuous chorus
    and score as one segment.
    """
    order = sorted(zip((float(s) for s in starts), (float(e) for e in ends)))
    merged_starts: List[float] = []
    merged_ends: List[float] = []
    for start, end in order:
        if merged_ends and abs(start - merged_ends[-1]) < tol:
            merged_ends[-1] = end
        else:
            merged_starts.append(start)
            merged_ends.append(end)
    return merged_starts, merged_ends


# --------------------------------------------------------------------------
# Interval metrics: every boundary and every frame
# --------------------------------------------------------------------------

def rasterize_intervals(intervals: Sequence[Tuple[float, float]],
                        duration: float, hop: float = 0.1) -> np.ndarray:
    """Boolean frame vector, True where a frame center falls inside an interval."""
    n = int(np.ceil(duration / hop))
    frames = np.zeros(n, dtype=bool)
    centers = (np.arange(n) + 0.5) * hop
    for start, end in intervals:
        frames |= (centers >= start) & (centers < end)
    return frames


def frame_metrics(pred: Sequence[Tuple[float, float]],
                  true: Sequence[Tuple[float, float]],
                  duration: float, hop: float = 0.1) -> dict:
    """Frame-level precision/recall/F1 on a fixed hop grid."""
    p = rasterize_intervals(pred, duration, hop)
    t = rasterize_intervals(true, duration, hop)
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def match_boundaries(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Signed error (pred - nearest true) for each predicted boundary."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if pred.size == 0 or true.size == 0:
        return np.array([])
    nearest = true[np.abs(true[None, :] - pred[:, None]).argmin(axis=1)]
    return pred - nearest


def boundary_hit_rate(pred: np.ndarray, true: np.ndarray, tol: float) -> float:
    """Fraction of predicted boundaries within `tol` seconds of a true boundary."""
    err = match_boundaries(pred, true)
    if err.size == 0:
        return 0.0
    return float(np.mean(np.abs(err) <= tol))


def score_song(pred_starts: Sequence[float], pred_ends: Sequence[float],
               true_starts: Sequence[float], true_ends: Sequence[float],
               duration: float, beat_period: Optional[float] = None) -> dict:
    """Frame metrics plus boundary error stats for one song."""
    pred_intervals = list(zip(pred_starts, pred_ends))
    true_intervals = list(zip(true_starts, true_ends))
    out = frame_metrics(pred_intervals, true_intervals, duration)

    pred_b = (np.concatenate([np.asarray(pred_starts, float), np.asarray(pred_ends, float)])
              if len(pred_starts) else np.array([]))
    true_b = (np.concatenate([np.asarray(true_starts, float), np.asarray(true_ends, float)])
              if len(true_starts) else np.array([]))
    err = match_boundaries(pred_b, true_b)
    abs_err = np.abs(err)
    out["median_abs_err_s"] = float(np.median(abs_err)) if abs_err.size else float("nan")
    out["hit_rate_70ms"] = boundary_hit_rate(pred_b, true_b, 0.070)
    out["hit_rate_500ms"] = boundary_hit_rate(pred_b, true_b, 0.500)
    out["n_pred_boundaries"] = int(pred_b.size)
    if beat_period:
        out["median_abs_err_beats"] = (float(np.median(abs_err) / beat_period)
                                       if abs_err.size else float("nan"))
    return out


def aggregate(rows: List[dict]) -> dict:
    """Mean and median of each numeric metric across songs (NaNs ignored)."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_median"] = float(np.median(vals))
    return agg
