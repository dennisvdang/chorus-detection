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

def plot_snap_comparison(audio_features, raw_starts, raw_ends,
                         snapped_starts, snapped_ends, downbeat_times,
                         title=None, save_path=None, zoom_bars=4.0):
    """
    Plot the waveform with the downbeat grid and raw vs snapped chorus
    boundaries, plus zoomed views around each snapped boundary.

    Parameters:
    - audio_features: AudioFeature object containing audio data
    - raw_starts, raw_ends: Chorus boundaries from the meter-grid predictions
    - snapped_starts, snapped_ends: Boundaries after downbeat snapping
    - downbeat_times: Downbeat times in seconds (e.g. from Beat This!)
    - title: Optional title for the plot
    - save_path: Optional path to save the plot image
    - zoom_bars: Half-width of each zoom panel, in bars

    Returns:
    - fig: The matplotlib figure object
    """
    downbeat_times = np.asarray(downbeat_times, dtype=float)
    bar_length = float(np.median(np.diff(downbeat_times))) if downbeat_times.size > 1 else 2.0
    duration = len(audio_features.y) / audio_features.sr

    n_zoom = len(snapped_starts) + len(snapped_ends)
    zoom_cols = max(n_zoom, 1)
    fig = plt.figure(figsize=(14, 6 if n_zoom else 3.5), dpi=96)
    grid = fig.add_gridspec(2 if n_zoom else 1, zoom_cols, height_ratios=[2, 1] if n_zoom else [1])

    # Full-song overview
    ax = fig.add_subplot(grid[0, :])
    librosa.display.waveshow(audio_features.y, sr=audio_features.sr,
                             alpha=0.5, ax=ax, color='steelblue')
    for t in downbeat_times:
        ax.axvline(x=t, color='grey', linestyle='--', linewidth=0.6, alpha=0.5)
    for i, (s, e) in enumerate(zip(raw_starts, raw_ends)):
        ax.axvspan(s, e, color='red', alpha=0.12, label='Raw chorus' if i == 0 else None)
        ax.axvline(s, color='red', linestyle='--', linewidth=1.4)
        ax.axvline(e, color='red', linestyle='--', linewidth=1.4)
    for i, (s, e) in enumerate(zip(snapped_starts, snapped_ends)):
        ax.axvspan(s, e, color='green', alpha=0.18, label='Snapped chorus' if i == 0 else None)
        ax.axvline(s, color='green', linestyle='-', linewidth=1.4)
        ax.axvline(e, color='green', linestyle='-', linewidth=1.4)
    ax.set_xlim([0, duration])
    ax.set_ylabel('Amplitude')
    xticks = np.arange(0, duration, 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(t // 60)}:{int(t % 60):02d}" for t in xticks])
    if title is None:
        name = os.path.splitext(os.path.basename(audio_features.audio_path))[0]
        title = f'Raw vs downbeat-snapped chorus boundaries: {name}'
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

    # Zoomed views around each snapped boundary
    boundaries = ([(t, f'Start {i+1}') for i, t in enumerate(snapped_starts)] +
                  [(t, f'End {i+1}') for i, t in enumerate(snapped_ends)])
    boundaries.sort(key=lambda b: b[0])
    raw_all = list(raw_starts) + list(raw_ends)
    half = zoom_bars * bar_length
    for col, (t, label) in enumerate(boundaries):
        axz = fig.add_subplot(grid[1, col])
        lo, hi = max(0.0, t - half), min(duration, t + half)
        i0, i1 = int(lo * audio_features.sr), int(hi * audio_features.sr)
        seg = audio_features.y[i0:i1]
        axz.plot(np.linspace(lo, hi, len(seg)), seg, color='steelblue',
                 alpha=0.6, linewidth=0.4)
        for d in downbeat_times[(downbeat_times >= lo) & (downbeat_times <= hi)]:
            axz.axvline(d, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
        for r in [r for r in raw_all if lo <= r <= hi]:
            axz.axvline(r, color='red', linestyle='--', linewidth=1.6)
        axz.axvline(t, color='green', linestyle='-', linewidth=1.8)
        axz.set_title(f'{label}  {int(t // 60)}:{t % 60:05.2f}', fontsize=8)
        axz.set_xlim([lo, hi])
        axz.set_yticks([])
        axz.tick_params(axis='x', labelsize=6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show(block=False)
    return fig
