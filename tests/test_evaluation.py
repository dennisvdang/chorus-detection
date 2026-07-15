import numpy as np

from pytorch_core import evaluation as ev


def test_rasterize_marks_covered_frames():
    frames = ev.rasterize_intervals([(1.0, 2.0)], duration=3.0, hop=0.5)
    assert frames.tolist() == [False, False, True, True, False, False]


def test_frame_metrics_perfect_match():
    m = ev.frame_metrics([(1.0, 2.0)], [(1.0, 2.0)], duration=3.0, hop=0.1)
    assert m["f1"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0


def test_frame_metrics_half_overlap():
    # pred [1,3], true [2,4] on 5s -> TP=1s, FP=1s, FN=1s -> P=R=F1=0.5
    m = ev.frame_metrics([(1.0, 3.0)], [(2.0, 4.0)], duration=5.0, hop=0.1)
    assert abs(m["precision"] - 0.5) < 1e-9
    assert abs(m["recall"] - 0.5) < 1e-9
    assert abs(m["f1"] - 0.5) < 1e-9


def test_match_boundaries_signed_error():
    err = ev.match_boundaries(np.array([1.1, 5.0]), np.array([1.0, 4.0]))
    assert abs(err[0] - 0.1) < 1e-9 and abs(err[1] - 1.0) < 1e-9


def test_boundary_hit_rate_tolerance():
    hr = ev.boundary_hit_rate(np.array([1.05, 3.0]), np.array([1.0, 2.0]), tol=0.1)
    assert hr == 0.5  # only 1.05 is within 0.1 of a true boundary


def test_score_song_reports_beats_when_beat_period_given():
    s = ev.score_song([1.0], [2.0], [1.0], [2.0], duration=3.0, beat_period=0.5)
    assert s["f1"] == 1.0 and s["median_abs_err_beats"] == 0.0


def test_empty_predictions_do_not_crash():
    s = ev.score_song([], [], [1.0], [2.0], duration=3.0)
    assert s["recall"] == 0.0
