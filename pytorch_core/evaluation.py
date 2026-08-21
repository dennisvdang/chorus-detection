"""Alias kept for existing imports. The definitions live in pytorch_core.metrics.

This file re-exports the interval metrics so code and notebooks written
against pytorch_core.evaluation keep working. Do not add definitions here.
"""

from pytorch_core.metrics import (  # noqa: F401
    aggregate,
    boundary_hit_rate,
    frame_metrics,
    match_boundaries,
    rasterize_intervals,
    score_song,
)

__all__ = ["aggregate", "boundary_hit_rate", "frame_metrics",
           "match_boundaries", "rasterize_intervals", "score_song"]
