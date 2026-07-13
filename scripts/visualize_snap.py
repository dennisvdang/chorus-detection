#!/usr/bin/env python
"""
Visualize raw vs downbeat-snapped chorus boundaries on the waveform,
with the Beat This! downbeat grid overlaid.
"""

import argparse
import os
import sys

import librosa
import numpy as np
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_core.audio_processor import process_audio
from pytorch_core.downbeats import snap_chorus_segments, track_downbeats
from pytorch_core.model import smooth_predictions
from pytorch_core.visualization import plot_snap_comparison
from scripts.inference import load_config, load_model


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


def main():
    parser = argparse.ArgumentParser(
        description="Plot raw vs downbeat-snapped chorus boundaries")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the comparison plot")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run inference on")
    parser.add_argument("--snap-window", type=float, default=2.0,
                        help="Half-width in bars of the downbeat snap search window")
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_model(args.checkpoint, config).to(args.device)

    processed_audio, audio_features = process_audio(
        args.audio, trim_silence=True,
        sr=config["data"]["sr"], hop_length=config["data"]["hop_length"])
    if processed_audio is None:
        print(f"Error processing audio: {args.audio}")
        return

    audio_tensor = torch.tensor(processed_audio, dtype=torch.float32).to(args.device)
    with torch.no_grad():
        outputs = model(audio_tensor).cpu().numpy().squeeze()
    n_meters = min(len(audio_features.meter_grid) - 1, len(outputs))
    smoothed = smooth_predictions(outputs[:n_meters])

    meter_grid_times = librosa.frames_to_time(
        audio_features.meter_grid, sr=audio_features.sr,
        hop_length=audio_features.hop_length)
    raw_starts, raw_ends = chorus_segments_from_predictions(smoothed, meter_grid_times)
    if not raw_starts:
        print("No choruses detected; nothing to plot.")
        return

    _, downbeat_times = track_downbeats(args.audio, device=args.device)
    rms = np.asarray(audio_features.rms).ravel()
    rms_times = librosa.frames_to_time(
        np.arange(rms.size), sr=audio_features.sr, hop_length=audio_features.hop_length)
    snapped_starts, snapped_ends = snap_chorus_segments(
        raw_starts, raw_ends, downbeat_times,
        energy=rms, energy_times=rms_times, search_bars=args.snap_window)

    print("raw    :", [f"{t:7.2f}" for t in raw_starts], [f"{t:7.2f}" for t in raw_ends])
    print("snapped:", [f"{t:7.2f}" for t in snapped_starts], [f"{t:7.2f}" for t in snapped_ends])

    plot_snap_comparison(audio_features, raw_starts, raw_ends,
                         snapped_starts, snapped_ends, downbeat_times,
                         save_path=args.output)
    if args.output:
        print(f"Comparison plot saved to {args.output}")


if __name__ == "__main__":
    main()
