"""Tests pinning the behaviour of pytorch_core.metrics, the single metric module.

The values pinned here (the 0.25-bar exactness tolerance, the classification
boundaries) are the ones data/trials.csv was scored with.
"""

import json
import os

from pytorch_core import evaluation, manifest, metrics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_exact_tolerance_is_a_quarter_bar():
    assert metrics.EXACT_FRACTION == 0.25
    assert metrics.BEATS_PER_BAR == 4


def test_anchor_error_is_signed_first_start_difference():
    assert metrics.anchor_error([10.0, 60.0], [12.0, 61.0]) == -2.0
    assert metrics.anchor_error([12.0], [10.0]) == 2.0
    assert metrics.anchor_error([], [10.0]) is None
    assert metrics.anchor_error([10.0], []) is None


def test_classify_anchor_boundaries():
    bar = 2.0  # seconds
    assert metrics.classify_anchor(0.0, bar) == "exact"
    assert metrics.classify_anchor(0.5, bar) == "exact"        # exactly 0.25 bar
    assert metrics.classify_anchor(-0.5, bar) == "exact"
    assert metrics.classify_anchor(0.51, bar) == "within_1_bar"
    assert metrics.classify_anchor(2.0, bar) == "within_1_bar"  # exactly one bar
    assert metrics.classify_anchor(2.01, bar) == "off"
    assert metrics.classify_anchor(None, bar) == "off"          # no prediction
    assert metrics.classify_anchor(0.0, 0.0) == "off"           # no bar grid
    assert metrics.classify_anchor(0.0, None) == "off"


def test_summarize_classifications_counts_and_rates():
    summary = metrics.summarize_classifications(
        ["exact", "exact", "within_1_bar", "off"])
    assert summary["songs"] == 4
    assert summary["exact"] == 2
    assert summary["within_1_bar"] == 3          # exact placements count as within
    assert summary["exact_rate"] == 0.5
    assert summary["within_1_bar_rate"] == 0.75
    empty = metrics.summarize_classifications([])
    assert empty["songs"] == 0 and empty["exact_rate"] == 0.0


def test_merge_adjacent_segments_joins_touching_and_keeps_gaps():
    starts, ends = metrics.merge_adjacent_segments(
        [10.0, 20.0, 40.0], [20.0, 30.0, 50.0])
    assert starts == [10.0, 40.0]
    assert ends == [30.0, 50.0]


def test_evaluation_module_is_an_alias_not_a_second_definition():
    assert evaluation.score_song is metrics.score_song
    assert evaluation.frame_metrics is metrics.frame_metrics
    assert evaluation.match_boundaries is metrics.match_boundaries
    assert evaluation.aggregate is metrics.aggregate


def test_manifest_records_command_git_checkpoint_and_versions(tmp_path):
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"not a real checkpoint")
    out = tmp_path / "manifest.json"

    started = manifest.utc_now_iso()
    written = manifest.write_manifest(
        str(out),
        command=["prog", "--flag", "value"],
        repo_dir=REPO_ROOT,
        checkpoint=str(checkpoint),
        seed=42,
        dataset_paths={"labels": "data/clean_labeled.csv"},
        split_files={"test": "data/splits/test_songs.txt"},
        started_at=started,
        extra={"decoder": "viterbi"},
    )

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == written
    assert loaded["command"] == ["prog", "--flag", "value"]
    assert loaded["git"]["commit"], "the repo's commit must be recorded"
    assert loaded["checkpoint"]["sha256"] == manifest.file_sha256(str(checkpoint))
    assert loaded["checkpoint"]["bytes"] == len(b"not a real checkpoint")
    assert "beat_this" in loaded["packages"]
    assert loaded["seed"] == 42
    assert loaded["started_at"] == started
    assert loaded["finished_at"]
    assert loaded["extra"]["decoder"] == "viterbi"
