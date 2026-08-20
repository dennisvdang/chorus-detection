"""
Model loading and prediction for chorus detection.
"""

import os
import urllib.request

import librosa
import numpy as np
import torch

from pytorch_core.models.crnn import CRNN
from pytorch_core import defaults
from pytorch_core import downbeats as downbeat_tracking
from pytorch_core.decoding import viterbi_chorus

# Default model location; downloaded from the GitHub release when missing.
# This checkpoint was trained on the tracked-downbeat grid, so callers must
# read audio with grid_source=defaults.GRID_SOURCE. See pytorch_core.defaults.
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "CRNN_pytorch", "crnn_beatthis_v1.pt")
MODEL_URL = ("https://github.com/dennisvdang/chorus-detection/"
             "releases/download/beatthis-v1.0/crnn_beatthis_v1.pt")

# The previous checkpoint, trained on the extrapolated librosa grid. Kept so
# the older configuration stays reproducible; it is no longer the default.
LEGACY_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "models", "CRNN_pytorch", "crnn_v1.pt")
LEGACY_MODEL_URL = ("https://github.com/dennisvdang/chorus-detection/"
                    "releases/download/pytorch-v1.0/crnn_v1.pt")


def download_model(model_path: str = MODEL_PATH, url: str = MODEL_URL) -> str:
    """Download the pretrained model from the GitHub release if not present."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Downloading pretrained model to {model_path}...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    return model_path


def load_CRNN_model(model_path: str = MODEL_PATH, url: str = MODEL_URL):
    """Load the pre-trained CRNN model from the specified path.

    Pass LEGACY_MODEL_PATH and LEGACY_MODEL_URL together to load the older
    checkpoint; the two must match or the download writes the wrong weights.
    """
    try:
        if not os.path.exists(model_path):
            download_model(model_path, url)
        checkpoint = torch.load(model_path, map_location="cpu")
        model = CRNN(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def smooth_predictions(data: np.ndarray) -> np.ndarray:
    """Apply smoothing to model predictions to reduce jitter."""
    # First pass: Moving average
    window_size = 3
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(data), i + window_size // 2 + 1)
        smoothed[i] = np.mean(data[window_start:window_end])

    # Second pass: Eliminate short segments
    min_segment_length = 2
    current_segment_length = 1
    current_value = smoothed[0] > 0.5
    binary_smoothed = np.zeros_like(smoothed, dtype=int)
    binary_smoothed[0] = int(current_value)

    for i in range(1, len(smoothed)):
        new_value = smoothed[i] > 0.5
        if new_value == current_value:
            current_segment_length += 1
        else:
            # If segment is too short, revert to previous value
            if current_segment_length < min_segment_length:
                for j in range(i - current_segment_length, i):
                    binary_smoothed[j] = int(new_value)
            current_value = new_value
            current_segment_length = 1
        binary_smoothed[i] = int(current_value)

    # Third pass: Fix final segment if too short
    if current_segment_length < min_segment_length:
        for j in range(len(smoothed) - current_segment_length, len(smoothed)):
            binary_smoothed[j] = int(not current_value)

    return binary_smoothed


def make_predictions(model, processed_audio, audio_features,
                     snap_downbeats=defaults.SNAP_DOWNBEATS,
                     decode=defaults.DECODE):
    """Make chorus predictions using the loaded model.

    Args:
        decode: "viterbi" reads the per-bar probabilities as a two-state HMM
            and takes the lowest-cost path; "smooth" thresholds at 0.5 and
            drops runs shorter than two bars. Viterbi is the default because
            it places 58.8% of first chorus starts exactly against 54.9% for
            smoothing on this grid (results/vast-run/trials.csv, T3b vs T2b).
        snap_downbeats: Move each decoded boundary to a nearby downbeat
            tracked by Beat This!, which raises exact placement from 58.8% to
            76.5% (T4b vs T3b). Requires the beat_this package; without it the
            boundaries are returned unsnapped.

    The audio must have been read with grid_source=defaults.GRID_SOURCE, the
    grid the shipped checkpoint was trained on.
    """
    # Generate predictions
    audio_tensor = torch.tensor(processed_audio, dtype=torch.float32)
    with torch.no_grad():
        raw_predictions = model(audio_tensor).numpy().squeeze()

    # Limit predictions to actual meters
    n_meters = min(len(audio_features.meter_grid) - 1, len(raw_predictions))
    predictions = raw_predictions[:n_meters]

    # Turn per-bar probabilities into a 0/1 path
    if decode == "viterbi":
        smoothed_predictions = viterbi_chorus(
            predictions,
            switch_penalty=defaults.VITERBI_SWITCH_PENALTY,
            min_bars=defaults.VITERBI_MIN_BARS)
    elif decode == "smooth":
        smoothed_predictions = smooth_predictions(predictions)
    else:
        raise ValueError(f"unknown decode {decode!r}; use 'viterbi' or 'smooth'")

    # Calculate time values for display
    meter_grid_times = librosa.frames_to_time(
        audio_features.meter_grid, sr=audio_features.sr, hop_length=audio_features.hop_length)

    # Find chorus segments
    chorus_indices = np.where(smoothed_predictions == 1)[0]
    chorus_start_times = []
    chorus_end_times = []

    if len(chorus_indices) > 0:
        # Group consecutive indices
        groups = []
        current_group = [chorus_indices[0]]

        for i in range(1, len(chorus_indices)):
            if chorus_indices[i] == chorus_indices[i-1] + 1:
                current_group.append(chorus_indices[i])
            else:
                groups.append(current_group)
                current_group = [chorus_indices[i]]
        groups.append(current_group)

        for group in groups:
            chorus_start_times.append(meter_grid_times[group[0]])
            chorus_end_times.append(meter_grid_times[group[-1] + 1])

        if snap_downbeats:
            chorus_start_times, chorus_end_times = snap_boundaries_to_downbeats(
                chorus_start_times, chorus_end_times, audio_features.audio_path,
                audio_features=audio_features,
                search_bars=defaults.SNAP_WINDOW_BARS)

        # Display chorus segments
        print("\nDetected chorus sections:")
        for i, (start_time, end_time) in enumerate(zip(chorus_start_times, chorus_end_times)):
            start_min, start_sec = divmod(start_time, 60)
            end_min, end_sec = divmod(end_time, 60)

            print(f"Chorus {i+1}: {int(start_min)}:{start_sec:05.2f} - {int(end_min)}:{end_sec:05.2f}")
    else:
        print("No choruses detected in this audio file.")

    return smoothed_predictions, chorus_start_times, chorus_end_times


def snap_boundaries_to_downbeats(chorus_start_times, chorus_end_times, audio_path,
                                 device="cpu", audio_features=None,
                                 search_bars=defaults.SNAP_WINDOW_BARS):
    """Snap chorus boundaries to Beat This! downbeats, if the tracker is available.

    When audio_features carries an RMS envelope, each boundary snaps to the
    downbeat within +/- search_bars with the strongest energy rise (starts) or
    fall (ends), correcting boundaries that are off by up to a full bar or two.
    Without RMS it snaps to the nearest downbeat. Returns the boundaries
    unchanged when beat_this is not installed or the tracker fails, so
    snapping never breaks plain chorus detection.
    """
    if not chorus_start_times:
        return chorus_start_times, chorus_end_times
    if not downbeat_tracking.is_available():
        print("beat_this not installed; skipping downbeat snapping "
              "(pip install https://github.com/CPJKU/beat_this/archive/main.zip)")
        return chorus_start_times, chorus_end_times
    try:
        _, downbeat_times = downbeat_tracking.track_downbeats(audio_path, device=device)
        if len(downbeat_times) == 0:
            print("No downbeats detected; skipping downbeat snapping.")
            return chorus_start_times, chorus_end_times

        energy = energy_times = None
        rms = getattr(audio_features, "rms", None) if audio_features is not None else None
        if rms is not None:
            energy = np.asarray(rms).ravel()
            energy_times = librosa.frames_to_time(
                np.arange(energy.size), sr=audio_features.sr,
                hop_length=audio_features.hop_length)

        return downbeat_tracking.snap_chorus_segments(
            chorus_start_times, chorus_end_times, downbeat_times,
            energy=energy, energy_times=energy_times, search_bars=search_bars)
    except Exception as e:
        print(f"Downbeat snapping failed ({e}); using unsnapped boundaries.")
        return chorus_start_times, chorus_end_times
