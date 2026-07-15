"""Time-domain evaluation for chorus boundary predictions.

Per-meter metrics are not comparable across meter grids, so every trial is
scored here in the time domain: predicted chorus intervals are rasterized onto
a fixed hop grid for frame-level F1, and boundaries are matched to the nearest
ground-truth boundary for hit-rate and median error. Ground-truth boundaries
are the annotated chorus start/end times, which sit on true downbeats.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np


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
