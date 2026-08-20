"""The shipped inference defaults must stay the configuration that measured best.

A grid ablation over 51 held-out songs (data/trials.csv, run
2026-08-17) scored ten configurations on how accurately each places the first
chorus start. Tracked downbeats + Viterbi + snapping won with 76.5% exact,
against 11.8% for the extrapolated grid with the same decoder.

These tests read that CSV rather than hard-coding the winner, so changing a
default means either reproducing a better measurement or deleting the evidence.
"""

import argparse
import csv
import inspect
import os

import pytest

from pytorch_core import defaults
from pytorch_core.audio_processor import process_audio
from pytorch_core.model import MODEL_PATH, make_predictions

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIALS_CSV = os.path.join(REPO_ROOT, "data", "trials.csv")


def _trials():
    """Every scored configuration, as dictionaries."""
    if not os.path.exists(TRIALS_CSV):
        pytest.skip("data/trials.csv not available")
    with open(TRIALS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _best_trial():
    """The configuration that placed the most first chorus starts exactly."""
    return max(_trials(), key=lambda row: float(row["anchor_exact_rate"]))


def test_defaults_are_the_configuration_that_measured_best():
    """The defaults must name the same grid, decoder, and snapping as the winner."""
    best = _best_trial()
    assert best["grid"] == defaults.GRID_SOURCE
    assert best["decode"] == defaults.DECODE
    assert bool(int(best["snap"])) == defaults.SNAP_DOWNBEATS


def test_winning_trial_still_beats_every_alternative():
    """Guards against a CSV edit that quietly reshuffles which row wins."""
    best = _best_trial()
    others = [r for r in _trials() if r["trial"] != best["trial"]]
    assert others, "trials.csv holds only one configuration"
    assert all(float(r["anchor_exact_rate"]) < float(best["anchor_exact_rate"])
               for r in others)
    # The measured headline: 76.5% exact, 80.4% within one bar.
    assert float(best["anchor_exact_rate"]) == pytest.approx(0.765, abs=0.001)
    assert float(best["anchor_within_1_bar_rate"]) == pytest.approx(0.804, abs=0.001)


def test_tracked_downbeat_grid_beats_extrapolated_grid_with_same_decoder():
    """The grid, not the decoder, is what moves the number."""
    rows = {r["trial"]: r for r in _trials()}
    extrapolated = float(rows["T3a"]["anchor_exact_rate"])  # librosa grid, viterbi
    tracked = float(rows["T3b"]["anchor_exact_rate"])       # beat_this grid, viterbi
    assert tracked > extrapolated
    assert extrapolated == pytest.approx(0.118, abs=0.001)


def test_process_audio_reads_the_tracked_downbeat_grid_by_default():
    grid_default = inspect.signature(process_audio).parameters["grid_source"].default
    assert grid_default == "beat_this"


def test_make_predictions_decodes_with_viterbi_and_snaps_by_default():
    params = inspect.signature(make_predictions).parameters
    assert params["decode"].default == "viterbi"
    assert params["snap_downbeats"].default is True


def test_viterbi_parameters_match_the_ones_the_experiment_used():
    assert defaults.VITERBI_SWITCH_PENALTY == 2.0
    assert defaults.VITERBI_MIN_BARS == 4
    assert defaults.SNAP_WINDOW_BARS == 2.0


def _parse_with_no_options(module, required_args):
    """Return the namespace a script's own parser produces from required args only.

    Intercepts parse_args inside main(), so this reads the real parser the
    script declares rather than a copy of it, then stops main() before it
    touches the filesystem.
    """
    captured = {}
    real_parse_args = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        parsed = real_parse_args(self, required_args, namespace)
        captured.update(vars(parsed))
        raise SystemExit(0)

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = capture
    try:
        try:
            module.main()
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured


def test_training_data_is_built_on_the_same_grid_inference_reads():
    """Training and inference must default to the same bar grid.

    Following the README in order -- preprocess, train, infer -- must not
    produce a model trained on one grid and run on the other.
    """
    import scripts.preprocess as preprocess

    signature_default = inspect.signature(
        preprocess.process_song).parameters["grid_source"].default
    assert signature_default == defaults.GRID_SOURCE

    cli_default = _parse_with_no_options(preprocess, [])["grid_source"]
    assert cli_default == defaults.GRID_SOURCE
    assert cli_default == inspect.signature(
        process_audio).parameters["grid_source"].default


def test_default_checkpoint_is_the_one_trained_on_the_tracked_downbeat_grid():
    """The grid and the weights must change together or the numbers do not hold."""
    assert os.path.basename(MODEL_PATH) == "crnn_beatthis_v1.pt"


def test_inference_cli_defaults_match_the_shipped_configuration():
    """scripts/inference.py must not reintroduce the losing configuration."""
    from scripts.inference import detect_chorus, main  # noqa: F401

    params = inspect.signature(detect_chorus).parameters
    assert params["grid_source"].default == "beat_this"
    assert params["decode"].default == "viterbi"
    assert params["snap_downbeats"].default is True


def test_inference_cli_arguments_default_to_the_shipped_configuration(monkeypatch):
    """Parse an argument list with only --audio given and check what it produces."""
    import scripts.inference as inference

    captured = {}

    real_parse_args = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        parsed = real_parse_args(self, ["--audio", "song.mp3"], namespace)
        captured.update(vars(parsed))
        raise SystemExit(0)  # stop main() before it touches the filesystem

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(SystemExit):
        inference.main()

    assert captured["grid_source"] == "beat_this"
    assert captured["decode"] == "viterbi"
    assert captured["no_snap"] is False
    assert captured["snap_window"] == 2.0
    assert os.path.basename(captured["checkpoint"]) == "crnn_beatthis_v1.pt"
