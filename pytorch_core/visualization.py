"""
Visualization utilities for chorus detection.
"""

import os
import numpy as np
import librosa
from matplotlib import pyplot as plt


def plot_meter_lines(ax: plt.Axes, meter_grid_times: np.ndarray) -> None:
    """Draw meter grid lines on the plot."""
    for time in meter_grid_times:
        ax.axvline(x=time, color='grey', linestyle='--', linewidth=1, alpha=0.6)


def plot_predictions(audio_features, binary_predictions, title=None, save_path=None):
    """
    Plot the audio waveform and overlay the predicted chorus locations.
    
    Parameters:
    - audio_features: AudioFeature object containing audio data
    - binary_predictions: Array of binary predictions (1=chorus, 0=not chorus)
    - title: Optional title for the plot (default: based on audio filename)
    - save_path: Optional path to save the plot image (default: don't save)
    
    Returns:
    - fig: The matplotlib figure object
    """
    meter_grid_times = librosa.frames_to_time(
        audio_features.meter_grid, sr=audio_features.sr, hop_length=audio_features.hop_length)
    fig, ax = plt.subplots(figsize=(12.5, 3), dpi=96)

    # Display waveform components
    librosa.display.waveshow(audio_features.y_harm, sr=audio_features.sr, 
                             alpha=0.8, ax=ax, color='deepskyblue')
    librosa.display.waveshow(audio_features.y_perc, sr=audio_features.sr, 
                             alpha=0.7, ax=ax, color='plum')
    plot_meter_lines(ax, meter_grid_times)

    # Highlight chorus sections
    first_chorus = True
    for i, prediction in enumerate(binary_predictions):
        if i < len(meter_grid_times) - 1 and prediction == 1:
            start_time = meter_grid_times[i]
            end_time = meter_grid_times[i + 1]
            ax.axvspan(start_time, end_time, color='green', alpha=0.3,
                      label='Predicted Chorus' if first_chorus else None)
            first_chorus = False

    # Configure plot appearance
    ax.set_xlim([0, len(audio_features.y) / audio_features.sr])
    ax.set_ylabel('Amplitude')
    
    # Set plot title
    if title:
        ax.set_title(title)
    else:
        audio_file_name = os.path.basename(audio_features.audio_path)
        ax.set_title(f'Chorus Predictions for {os.path.splitext(audio_file_name)[0]}')

    # Add legend
    chorus_patch = plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(chorus_patch)
    labels.append('Chorus')
    ax.legend(handles=handles, labels=labels)

    # Set time-based x-axis labels
    duration = len(audio_features.y) / audio_features.sr
    xticks = np.arange(0, duration, 10)
    xlabels = [f"{int(tick // 60)}:{int(tick % 60):02d}" for tick in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show(block=False)
    return fig


def plot_chorus_segments(audio_features, chorus_start_times, chorus_end_times, 
                         title=None, save_path=None):
    """
    Plot only the chorus segments as a timeline.
    
    Parameters:
    - audio_features: AudioFeature object containing audio data
    - chorus_start_times: List of chorus start times in seconds
    - chorus_end_times: List of chorus end times in seconds
    - title: Optional title for the plot
    - save_path: Optional path to save the plot image
    
    Returns:
    - fig: The matplotlib figure object
    """
    duration = len(audio_features.y) / audio_features.sr
    fig, ax = plt.subplots(figsize=(12, 1.5), dpi=96)
    
    # Create timeline
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    
    # Mark chorus sections
    for i, (start, end) in enumerate(zip(chorus_start_times, chorus_end_times)):
        ax.axvspan(start, end, ymin=0.4, ymax=0.6, color='green', alpha=0.7)
        ax.text((start + end) / 2, 0.1, f"Chorus {i+1}", 
                horizontalalignment='center', fontsize=10)
    
    # Configure plot appearance
    ax.set_xlim([0, duration])
    ax.set_ylim([-0.2, 0.2])
    ax.set_yticks([])
    
    # Set plot title
    if title:
        ax.set_title(title)
    else:
        audio_file_name = os.path.basename(audio_features.audio_path)
        ax.set_title(f'Chorus Timeline for {os.path.splitext(audio_file_name)[0]}')
    
    # Set time-based x-axis labels
    xticks = np.arange(0, duration, 10)
    xlabels = [f"{int(tick // 60)}:{int(tick % 60):02d}" for tick in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Time')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show(block=False)
    return fig 