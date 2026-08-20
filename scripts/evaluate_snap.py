#!/usr/bin/env python
"""
Evaluate downbeat snapping across the labeled dataset.

For every labeled song, runs the model to get raw meter-grid chorus
boundaries, snaps them to Beat This! downbeats, and measures:
  - the shift each boundary moved (snapped - raw)
  - the error of raw and snapped boundaries against the annotated
    ground-truth chorus boundaries (predicted - truth, matched to the
    nearest true boundary of the same kind)

Appends one row per boundary to the output CSV as it goes, so an
interrupted run keeps its partial results. Prints a summary at the end.

Usage:
    python scripts/evaluate_snap.py --checkpoint models/pytorch/best_model.pt \
        --output output/snap_evaluation.csv [--limit N] [--device cuda]
"""

import argparse
import csv
import os
import sys

import librosa
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_core.audio_processor import process_audio
from pytorch_core import defaults
from pytorch_core.downbeats import snap_chorus_segments, track_downbeats
from pytorch_core.model import smooth_predictions
from scripts.inference import load_config, load_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(REPO_ROOT, "data", "audio", "processed")
LABELS_CSV = os.path.join(REPO_ROOT, "data", "clean_labeled.csv")

CSV_FIELDS = ["song_id", "kind", "raw", "snapped", "shift",
              "truth", "raw_error", "snapped_error", "bar_length"]


def true_chorus_segments(song_df):
    """Merge the song's contiguous labeled chorus rows into (starts, ends)."""
    rows = song_df[song_df["label"] == "chorus"].sort_values("start_time")
    starts, ends = [], []
    for _, row in rows.iterrows():
        if ends and abs(row["start_time"] - ends[-1]) < 1e-6:
            ends[-1] = row["end_time"]
        else:
            starts.append(row["start_time"])
            ends.append(row["end_time"])
    return starts, ends


def chorus_segments_from_predictions(smoothed, meter_grid_times):
    """Group consecutive positive meters into (start_times, end_times)."""
    starts, ends = [], []
    indices = np.where(smoothed == 1)[0]
    if len(indices) == 0:
        return starts, ends
    group = [indices[0]]
    for i in indices[1:]:
        if i == group[-1] + 1:
            group.append(i)
        else:
            starts.append(meter_grid_times[group[0]])
            ends.append(meter_grid_times[group[-1] + 1])
            group = [i]
    starts.append(meter_grid_times[group[0]])
    ends.append(meter_grid_times[group[-1] + 1])
    return starts, ends


def nearest(value, candidates):
    """Nearest candidate to value, or None if candidates is empty."""
    if not candidates:
        return None
    candidates = np.asarray(candidates, dtype=float)
    return float(candidates[np.abs(candidates - value).argmin()])


def evaluate_song(song_id, song_df, model, config, device,
                  grid_source=defaults.GRID_SOURCE):
    """Return per-boundary result dicts for one song."""
    audio_path = os.path.join(AUDIO_DIR, f"{song_id}.mp3")
    if not os.path.exists(audio_path):
        return None

    # Labels were annotated on the processed audio, so do not re-trim.
    # Tempo/time signature come from the CSV like the training preprocessing.
    bpm = song_df["sp_tempo"].iloc[0]
    bpm = None if pd.isna(bpm) or bpm == 0 else float(bpm)
    time_signature = song_df["sp_time_signature"].iloc[0]
    time_signature = 4 if pd.isna(time_signature) or time_signature == 0 else int(time_signature)
    processed_audio, audio_features = process_audio(
        audio_path, trim_silence=False,
        sr=config["data"]["sr"], hop_length=config["data"]["hop_length"],
        bpm=bpm, time_signature=time_signature,
        grid_source=grid_source, device=device)
    if processed_audio is None:
        return None

    audio_tensor = torch.tensor(processed_audio, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(audio_tensor).cpu().numpy().squeeze()
    n_meters = min(len(audio_features.meter_grid) - 1, len(outputs))
    smoothed = smooth_predictions(outputs[:n_meters])

    meter_grid_times = librosa.frames_to_time(
        audio_features.meter_grid, sr=audio_features.sr,
        hop_length=audio_features.hop_length)
    raw_starts, raw_ends = chorus_segments_from_predictions(smoothed, meter_grid_times)
    if not raw_starts:
        return []

    _, downbeats = track_downbeats(audio_path, device=device)
    if len(downbeats) < 2:
        return []
    bar_length = float(np.median(np.diff(downbeats)))

    rms = np.asarray(audio_features.rms).ravel()
    rms_times = librosa.frames_to_time(
        np.arange(rms.size), sr=audio_features.sr,
        hop_length=audio_features.hop_length)
    snapped_starts, snapped_ends = snap_chorus_segments(
        raw_starts, raw_ends, downbeats, energy=rms, energy_times=rms_times)

    true_starts, true_ends = true_chorus_segments(song_df)

    # Snapping can drop or merge segments; match each snapped boundary back
    # to its nearest raw boundary of the same kind for the shift measurement.
    results = []
    for kind, snapped_list, raw_list, truth_list in (
            ("start", snapped_starts, raw_starts, true_starts),
            ("end", snapped_ends, raw_ends, true_ends)):
        for snapped in snapped_list:
            raw = nearest(snapped, raw_list)
            truth = nearest(snapped, truth_list)
            results.append({
                "song_id": song_id,
                "kind": kind,
                "raw": round(raw, 3),
                "snapped": round(float(snapped), 3),
                "shift": round(float(snapped) - raw, 3),
                "truth": round(truth, 3) if truth is not None else "",
                "raw_error": round(raw - truth, 3) if truth is not None else "",
                "snapped_error": round(float(snapped) - truth, 3) if truth is not None else "",
                "bar_length": round(bar_length, 3),
            })
    return results


def summarize(csv_path):
    """Print summary statistics from the per-boundary results CSV."""
    df = pd.read_csv(csv_path)
    if df.empty:
        print("No results to summarize.")
        return
    print(f"\n=== Snap evaluation summary ({df['song_id'].nunique()} songs, "
          f"{len(df)} boundaries) ===")

    shift = df["shift"]
    print(f"\nShift (snapped - raw), seconds:")
    print(f"  mean {shift.mean():+.3f}  median {shift.median():+.3f}  "
          f"mean|.| {shift.abs().mean():.3f}  median|.| {shift.abs().median():.3f}")
    print(f"  moved later: {(shift > 0).mean():.1%}   "
          f"moved earlier: {(shift < 0).mean():.1%}   "
          f"unmoved: {(shift == 0).mean():.1%}")

    scored = df.dropna(subset=["raw_error", "snapped_error"])
    if scored.empty:
        return
    for kind in ("start", "end", None):
        sub = scored if kind is None else scored[scored["kind"] == kind]
        label = kind or "all"
        raw_abs = sub["raw_error"].abs()
        snap_abs = sub["snapped_error"].abs()
        bars = sub["bar_length"]
        print(f"\nError vs ground truth [{label}] ({len(sub)} boundaries):")
        print(f"  raw     mean|err| {raw_abs.mean():.3f}s  median {raw_abs.median():.3f}s  "
              f"within 1 bar {(raw_abs <= bars).mean():.1%}  within 0.5s {(raw_abs <= 0.5).mean():.1%}")
        print(f"  snapped mean|err| {snap_abs.mean():.3f}s  median {snap_abs.median():.3f}s  "
              f"within 1 bar {(snap_abs <= bars).mean():.1%}  within 0.5s {(snap_abs <= 0.5).mean():.1%}")
        signed = sub["snapped_error"]
        print(f"  snapped signed error: mean {signed.mean():+.3f}s  median {signed.median():+.3f}s  "
              f"late {(signed > 0).mean():.1%} / early {(signed < 0).mean():.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate downbeat snapping on the dataset")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--grid-source", choices=["librosa", "beat_this"],
                        default=defaults.GRID_SOURCE,
                        help="Bar grid the checkpoint was trained on. A checkpoint "
                             "read on the other grid is a train/inference mismatch.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--output", type=str, default="output/snap_evaluation.csv")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N songs")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip evaluation and summarize an existing CSV")
    args = parser.parse_args()

    if args.summary_only:
        summarize(args.output)
        return

    config = load_config(args.config)
    model = load_model(args.checkpoint, config).to(args.device)
    labels = pd.read_csv(LABELS_CSV)
    song_ids = sorted(labels["SongID"].unique())
    if args.limit:
        song_ids = song_ids[:args.limit]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done_ids = set()
    if os.path.exists(args.output):
        try:
            done_ids = set(pd.read_csv(args.output)["song_id"].unique())
        except Exception:
            pass
    write_header = not os.path.exists(args.output) or not done_ids

    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for i, song_id in enumerate(song_ids):
            if song_id in done_ids:
                continue
            try:
                results = evaluate_song(song_id, labels[labels["SongID"] == song_id],
                                        model, config, args.device,
                                        grid_source=args.grid_source)
            except Exception as e:
                print(f"[{i+1}/{len(song_ids)}] song {song_id} FAILED: {e}", flush=True)
                continue
            if results is None:
                print(f"[{i+1}/{len(song_ids)}] song {song_id} skipped (no audio)", flush=True)
                continue
            for row in results:
                writer.writerow(row)
            f.flush()
            print(f"[{i+1}/{len(song_ids)}] song {song_id}: {len(results)} boundaries", flush=True)

    summarize(args.output)


if __name__ == "__main__":
    main()
