"""
Model loading and prediction for chorus detection.
"""

import os
import urllib.request

import librosa
import numpy as np
import torch

from pytorch_core.models.crnn import CRNN

# Default model location; downloaded from the GitHub release when missing
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "CRNN_pytorch", "crnn_v1.pt")
MODEL_URL = ("https://github.com/dennisvdang/chorus-detection/"
             "releases/download/pytorch-v1.0/crnn_v1.pt")


def download_model(model_path: str = MODEL_PATH, url: str = MODEL_URL) -> str:
    """Download the pretrained model from the GitHub release if not present."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Downloading pretrained model to {model_path}...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    return model_path


def load_CRNN_model(model_path: str = MODEL_PATH):
    """Load the pre-trained CRNN model from the specified path."""
    try:
        if not os.path.exists(model_path):
            download_model(model_path)
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


def make_predictions(model, processed_audio, audio_features):
    """Make chorus predictions using the loaded model."""
    # Generate predictions
    audio_tensor = torch.tensor(processed_audio, dtype=torch.float32)
    with torch.no_grad():
        raw_predictions = model(audio_tensor).numpy().squeeze()

    # Limit predictions to actual meters
    n_meters = min(len(audio_features.meter_grid) - 1, len(raw_predictions))
    predictions = raw_predictions[:n_meters]

    # Apply smoothing
    smoothed_predictions = smooth_predictions(predictions)

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

        # Display chorus segments
        print("\nDetected chorus sections:")
        for i, group in enumerate(groups):
            start_time = meter_grid_times[group[0]]
            end_time = meter_grid_times[group[-1] + 1]
            chorus_start_times.append(start_time)
            chorus_end_times.append(end_time)

            start_min, start_sec = divmod(start_time, 60)
            end_min, end_sec = divmod(end_time, 60)

            print(f"Chorus {i+1}: {int(start_min)}:{start_sec:05.2f} - {int(end_min)}:{end_sec:05.2f}")
    else:
        print("No choruses detected in this audio file.")

    return smoothed_predictions, chorus_start_times, chorus_end_times
