#!/usr/bin/env python
"""
Inference script for chorus detection model.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_core.models.crnn import CRNN
from pytorch_core.audio_processor import process_audio


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, config):
    """Load model from checkpoint."""
    # Create model with config
    model = CRNN(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def smooth_predictions(predictions, window_size=3, min_segment_length=2):
    """
    Apply smoothing to predictions to reduce jitter.
    This is the PyTorch equivalent of the function in model.py.
    """
    # First pass: Moving average
    smoothed = medfilt(predictions, window_size)
    
    # Second pass: Eliminate short segments
    binary_smoothed = np.zeros_like(smoothed, dtype=int)
    current_value = int(smoothed[0] > 0.5)
    current_segment_length = 1
    binary_smoothed[0] = current_value
    
    for i in range(1, len(smoothed)):
        new_value = int(smoothed[i] > 0.5)
        if new_value == current_value:
            current_segment_length += 1
        else:
            # If segment is too short, revert to previous value
            if current_segment_length < min_segment_length:
                for j in range(i - current_segment_length, i):
                    binary_smoothed[j] = new_value
            current_value = new_value
            current_segment_length = 1
        binary_smoothed[i] = current_value
    
    # Third pass: Fix final segment if too short
    if current_segment_length < min_segment_length:
        for j in range(len(smoothed) - current_segment_length, len(smoothed)):
            binary_smoothed[j] = int(not current_value)
    
    return binary_smoothed


def detect_chorus(model, audio_path, config, device='cpu', bpm=None, time_signature=None):
    """
    Detect chorus segments in audio file using the trained model.

    Args:
        model: Trained CRNN model
        audio_path: Path to audio file
        config: Configuration dictionary
        device: Device to run inference on
        bpm: Optional known tempo; beat-tracker estimate is used when None
        time_signature: Optional known time signature (beats per meter)

    Returns:
        Tuple of (predictions, chorus_start_times, chorus_end_times, audio_features)
    """
    # Process audio with the same pipeline used to build the training data
    processed_audio, audio_features = process_audio(audio_path, trim_silence=True,
                                                    sr=config["data"]["sr"],
                                                    hop_length=config["data"]["hop_length"],
                                                    bpm=bpm, time_signature=time_signature)
    
    if processed_audio is None or audio_features is None:
        print(f"Error processing audio: {audio_path}")
        return None, None, None, None
    
    # Move model to device
    model = model.to(device)
    
    # Convert processed audio to PyTorch tensor
    processed_audio_tensor = torch.tensor(processed_audio, dtype=torch.float32).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(processed_audio_tensor).cpu().numpy().squeeze()
    
    # Limit predictions to actual meters
    n_meters = min(len(audio_features.meter_grid) - 1, len(outputs))
    predictions = outputs[:n_meters]
    
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
        
        # Display chorus sections
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
    
    return smoothed_predictions, chorus_start_times, chorus_end_times, audio_features


def plot_predictions(audio_path, predictions, meter_grid, chorus_start_times, chorus_end_times, output_path=None):
    """Plot predictions against audio waveform."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Create time array for waveform
    t = np.arange(len(y)) / sr
    
    # Create time array for predictions
    pred_times = np.linspace(0, t[-1], len(predictions))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(t, y, color='#1f77b4', alpha=0.7)
    plt.title('Audio Waveform with Chorus Sections')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Highlight chorus sections
    for start, end in zip(chorus_start_times, chorus_end_times):
        plt.axvspan(start, end, color='green', alpha=0.3)
    
    # Set limits
    plt.xlim(0, t[-1])
    
    # Plot predictions
    plt.subplot(2, 1, 2)
    plt.plot(pred_times, predictions, color='blue', marker='.', linestyle='-')
    plt.title('Chorus Detection Predictions')
    plt.xlabel('Time (s)')
    plt.ylabel('Prediction')
    plt.grid(True)
    
    # Highlight chorus sections
    for start, end in zip(chorus_start_times, chorus_end_times):
        plt.axvspan(start, end, color='green', alpha=0.3)
    
    # Set limits
    plt.xlim(0, t[-1])
    plt.ylim(-0.1, 1.1)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Prediction plot saved to {output_path}")
    
    # Show figure
    plt.show()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference with chorus detection model")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio file")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output plot")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run inference on")
    parser.add_argument("--bpm", type=float, default=None,
                        help="Known tempo of the song (otherwise estimated)")
    parser.add_argument("--time-signature", type=int, default=None,
                        help="Known time signature (otherwise 4/4)")
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Audio file not found: {args.audio}")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    model = load_model(args.checkpoint, config)
    print(f"Loaded model from {args.checkpoint}")
    
    # Detect chorus
    print(f"Detecting chorus in {args.audio}...")
    predictions, chorus_start_times, chorus_end_times, audio_features = detect_chorus(
        model, args.audio, config, device=args.device,
        bpm=args.bpm, time_signature=args.time_signature)
    
    if predictions is None:
        print("Error detecting chorus.")
        return
    
    # Plot predictions
    plot_predictions(args.audio, predictions, audio_features.meter_grid, 
                     chorus_start_times, chorus_end_times, output_path=args.output)


if __name__ == "__main__":
    main() 