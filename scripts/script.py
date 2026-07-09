#!/usr/bin/env python3
"""
Chorus Detection Model Training Pipeline

This script provides a complete pipeline for training chorus detection models
using either TensorFlow/Keras or PyTorch implementations.
"""

import os
import sys
import argparse
import json
import pickle
import gzip
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Audio processing
import librosa
import scipy.signal
from sklearn.decomposition import NMF
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Deep learning frameworks
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Advanced Beat Detection Classes
# ============================================================================

@dataclass
class BeatResult:
    """Beat analysis results"""
    beats_seconds: np.ndarray    # Beat positions in seconds
    beats_frames: np.ndarray     # Beat positions in frames  
    tempo_bpm: float            # Estimated BPM
    confidence: float           # Analysis confidence (0-1)


class ComplexSpectralDifference:
    """Complex Spectral Difference onset detection (DF_COMPLEXSD from Queen Mary DSP)"""
    
    def __init__(self, frame_length: int, hop_length: int):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.half_length = frame_length // 2 + 1
        
        # History buffers
        self.mag_history = np.zeros(self.half_length)
        self.phase_history = np.zeros(self.half_length) 
        self.phase_history_old = np.zeros(self.half_length)
        self.window = np.hanning(frame_length)
        
    def process_frame(self, frame: np.ndarray) -> float:
        """Process frame and return onset strength"""
        windowed = frame * self.window
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        
        detection_value = 0.0
        for i in range(self.half_length):
            phase_diff = (phase[i] - 2 * self.phase_history[i] + 
                         self.phase_history_old[i])
            phase_dev = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
            diff = abs(self.mag_history[i] - magnitude[i] * np.exp(1j * phase_dev))
            detection_value += diff
            
            # Update history
            self.phase_history_old[i] = self.phase_history[i]
            self.phase_history[i] = phase[i]
            self.mag_history[i] = magnitude[i]
            
        return detection_value


class TempoTracker:
    """Tempo tracking using Viterbi decoding (from Queen Mary TempoTrackV2)"""
    
    def __init__(self, sr: int, hop_length: int):
        self.sr = sr
        self.hop_length = hop_length
        
    def analyze(self, detection_function: np.ndarray, 
                input_tempo: float = 120.0) -> Tuple[np.ndarray, float]:
        """Analyze detection function and return beat periods and tempo"""
        
        # Parameters from Mixxx
        wv_len = 128
        winlen = 512
        step = 128
        
        # Rayleigh weighting for tempo preference
        rayparam = (60 * self.sr / self.hop_length) / input_tempo
        weight_vector = np.zeros(wv_len)
        for i in range(wv_len):
            weight_vector[i] = (i / (rayparam**2)) * np.exp(-(i**2) / (2 * rayparam**2))
        
        # Process in overlapping frames
        rcf_matrix = []
        for i in range(0, len(detection_function) - winlen, step):
            df_frame = detection_function[i:i + winlen]
            rcf = self._get_rcf(df_frame, weight_vector)
            rcf_matrix.append(rcf)
        
        if not rcf_matrix:
            return np.array([60]), 120.0
            
        rcf_matrix = np.array(rcf_matrix).T
        
        # Viterbi decoding
        beat_periods = self._viterbi_decode(rcf_matrix, weight_vector)
        
        # Calculate tempo
        valid_periods = beat_periods[beat_periods > 0]
        if len(valid_periods) > 0:
            avg_period = np.median(valid_periods)
            tempo = (60.0 * self.sr / self.hop_length) / avg_period
        else:
            tempo = 120.0
            
        return beat_periods, tempo
    
    def track_beats(self, detection_function: np.ndarray, 
                   beat_periods: np.ndarray) -> np.ndarray:
        """Track beat positions using dynamic programming"""
        
        if len(detection_function) == 0 or len(beat_periods) == 0:
            return np.array([])
            
        alpha = 0.9
        tightness = 4.0
        df_len = len(detection_function)
        
        cumscore = np.zeros(df_len)
        backlink = np.full(df_len, -1, dtype=int)
        local_score = detection_function.copy()
        
        for i in range(df_len):
            beat_period = beat_periods[min(i, len(beat_periods) - 1)]
            if beat_period <= 0:
                beat_period = 60
            
            prange_min = int(-2 * beat_period)
            prange_max = int(-0.5 * beat_period)
            
            if prange_max >= prange_min:
                transition_length = prange_max - prange_min + 1
                score_candidates = np.zeros(transition_length)
                
                for j in range(transition_length):
                    mu = beat_period
                    offset = (2 * mu - j)
                    weight = np.exp(-0.5 * (tightness * np.log(offset / mu))**2) if mu > 0 else 0
                    
                    prev_idx = i + prange_min + j
                    if 0 <= prev_idx < df_len:
                        score_candidates[j] = weight * cumscore[prev_idx]
                
                if len(score_candidates) > 0:
                    best_idx = np.argmax(score_candidates)
                    cumscore[i] = alpha * np.max(score_candidates) + (1 - alpha) * local_score[i]
                    backlink[i] = i + prange_min + best_idx
                else:
                    cumscore[i] = local_score[i]
            else:
                cumscore[i] = local_score[i]
        
        # Backtrack
        search_range = min(int(beat_periods[-1]) if len(beat_periods) > 0 else 60, df_len)
        start_idx = max(0, df_len - search_range)
        start_point = start_idx + np.argmax(cumscore[start_idx:]) if start_idx < df_len else df_len - 1
        
        beats = []
        current = start_point
        while current >= 0 and current < df_len and current not in beats:
            beats.append(current)
            current = backlink[current]
                
        return np.array(beats[::-1])
    
    def _get_rcf(self, df_frame: np.ndarray, weight_vector: np.ndarray) -> np.ndarray:
        """Get resonant comb filter output"""
        df_frame = self._adaptive_threshold(df_frame.copy())
        acf = np.correlate(df_frame, df_frame, mode='full')[len(df_frame)-1:]
        
        rcf = np.zeros(len(weight_vector))
        for i in range(2, len(weight_vector)):
            for a in range(1, 5):
                for b in range(1 - a, a):
                    idx = a * i + b - 1
                    if 0 <= idx < len(acf):
                        rcf[i] += (acf[idx] * weight_vector[i]) / (2.0 * a - 1.0)
        
        rcf = self._adaptive_threshold(rcf)
        rcf_sum = np.sum(rcf)
        if rcf_sum > 0:
            rcf /= rcf_sum
            
        return rcf
    
    def _adaptive_threshold(self, data: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding"""
        if len(data) < 3:
            return data
        filtered = scipy.signal.medfilt(data, kernel_size=3)
        result = data - filtered
        result[result < 0] = 0
        return result
    
    def _viterbi_decode(self, rcf_matrix: np.ndarray, weight_vector: np.ndarray) -> np.ndarray:
        """Viterbi decoding for beat period estimation"""
        if rcf_matrix.shape[1] < 2:
            return np.array([60])
            
        num_states, num_frames = rcf_matrix.shape
        sigma = 8.0
        
        # Transition matrix
        transition_matrix = np.zeros((num_states, num_states))
        for i in range(20, num_states - 20):
            for j in range(20, num_states - 20):
                transition_matrix[i, j] = np.exp(-((j - i)**2) / (2 * sigma**2))
        
        # Viterbi
        delta = np.zeros((num_frames, num_states))
        psi = np.zeros((num_frames, num_states), dtype=int)
        
        delta[0, :] = weight_vector * rcf_matrix[:, 0]
        delta[0, :] /= (np.sum(delta[0, :]) + 1e-10)
        
        for t in range(1, num_frames):
            for j in range(num_states):
                temp = delta[t-1, :] * transition_matrix[j, :]
                psi[t, j] = np.argmax(temp)
                delta[t, j] = np.max(temp) * rcf_matrix[j, t]
            delta[t, :] /= (np.sum(delta[t, :]) + 1e-10)
        
        # Backtrack
        best_path = np.zeros(num_frames, dtype=int)
        best_path[-1] = np.argmax(delta[-1, :])
        for t in range(num_frames - 2, -1, -1):
            best_path[t] = psi[t + 1, best_path[t + 1]]
        
        # Expand to full length
        beat_periods = []
        for period_idx in best_path:
            for _ in range(128):  # step size
                beat_periods.append(period_idx)
        
        return np.array(beat_periods)


class BeatAnalyzer:
    """Beat analyzer with fixed 22050 Hz resampling"""
    
    def __init__(self):
        self.sr = 22050  # Fixed sample rate
        self.step_secs = 0.01161  # 11.61ms steps (86 Hz)
        self.hop_length = int(self.sr * self.step_secs)  # 256 samples
        self.frame_length = 512  # Fixed frame length for 22050 Hz
        
    def analyze(self, audio_path: str) -> BeatResult:
        """Analyze audio file and return beat information"""
        
        # Load and resample to 22050 Hz
        y, _ = librosa.load(audio_path, sr=self.sr)
        y = librosa.util.normalize(y)
        
        # Onset detection
        onset_detector = ComplexSpectralDifference(self.frame_length, self.hop_length)
        detection_function = self._compute_detection_function(y, onset_detector)
        
        # Skip first 2 frames (noise reduction)
        if len(detection_function) > 2:
            detection_function = detection_function[2:]
        else:
            return BeatResult(np.array([]), np.array([]), 120.0, 0.0)
        
        # Tempo tracking
        tempo_tracker = TempoTracker(self.sr, self.hop_length)
        beat_periods, tempo = tempo_tracker.analyze(detection_function)
        
        # Beat tracking
        beat_frames = tempo_tracker.track_beats(detection_function, beat_periods)
        
        # Adjust for skipped frames and convert to time
        beat_frames = beat_frames + 2
        beat_frames_adjusted = beat_frames * self.hop_length + self.hop_length // 2
        beats_seconds = beat_frames_adjusted / self.sr
        
        # Calculate confidence
        confidence = self._calculate_confidence(beats_seconds, tempo, detection_function)
        
        return BeatResult(
            beats_seconds=beats_seconds,
            beats_frames=beat_frames_adjusted.astype(int),
            tempo_bpm=tempo,
            confidence=confidence
        )
    
    def analyze_array(self, y: np.ndarray) -> BeatResult:
        """Analyze audio array directly (for preprocessed audio)"""
        y = librosa.util.normalize(y)
        
        # Onset detection
        onset_detector = ComplexSpectralDifference(self.frame_length, self.hop_length)
        detection_function = self._compute_detection_function(y, onset_detector)
        
        # Skip first 2 frames (noise reduction)
        if len(detection_function) > 2:
            detection_function = detection_function[2:]
        else:
            return BeatResult(np.array([]), np.array([]), 120.0, 0.0)
        
        # Tempo tracking
        tempo_tracker = TempoTracker(self.sr, self.hop_length)
        beat_periods, tempo = tempo_tracker.analyze(detection_function)
        
        # Beat tracking
        beat_frames = tempo_tracker.track_beats(detection_function, beat_periods)
        
        # Adjust for skipped frames and convert to time
        beat_frames = beat_frames + 2
        beat_frames_adjusted = beat_frames * self.hop_length + self.hop_length // 2
        beats_seconds = beat_frames_adjusted / self.sr
        
        # Calculate confidence
        confidence = self._calculate_confidence(beats_seconds, tempo, detection_function)
        
        return BeatResult(
            beats_seconds=beats_seconds,
            beats_frames=beat_frames_adjusted.astype(int),
            tempo_bpm=tempo,
            confidence=confidence
        )
    
    def _compute_detection_function(self, y: np.ndarray, 
                                  onset_detector: ComplexSpectralDifference) -> np.ndarray:
        """Compute detection function using overlapping frames"""
        detection_values = []
        buffer = np.zeros(self.frame_length)
        buffer_pos = self.frame_length // 2
        audio_pos = 0
        
        while audio_pos < len(y):
            samples_needed = self.frame_length - buffer_pos
            samples_available = len(y) - audio_pos
            samples_to_copy = min(samples_needed, samples_available)
            
            if samples_to_copy > 0:
                buffer[buffer_pos:buffer_pos + samples_to_copy] = y[audio_pos:audio_pos + samples_to_copy]
                buffer_pos += samples_to_copy
                audio_pos += samples_to_copy
            
            if buffer_pos == self.frame_length:
                detection_value = onset_detector.process_frame(buffer)
                detection_values.append(detection_value)
                
                shift_amount = self.hop_length
                if shift_amount < self.frame_length:
                    buffer[:-shift_amount] = buffer[shift_amount:]
                    buffer_pos = self.frame_length - shift_amount
                else:
                    buffer.fill(0)
                    buffer_pos = 0
            else:
                buffer[buffer_pos:] = 0
                detection_value = onset_detector.process_frame(buffer)
                detection_values.append(detection_value)
                break
        
        return np.array(detection_values)
    
    def _calculate_confidence(self, beats: np.ndarray, tempo: float, 
                            onset_env: np.ndarray) -> float:
        """Calculate analysis confidence score"""
        if len(beats) < 2:
            return 0.0
            
        # Tempo consistency
        beat_intervals = np.diff(beats)
        expected_interval = 60.0 / tempo
        tempo_consistency = max(0, 1.0 - np.std(beat_intervals) / expected_interval)
        
        return float(np.clip(tempo_consistency, 0, 1))


def analyze_audio(audio_path: str) -> BeatResult:
    """Convenience function to analyze an audio file"""
    analyzer = BeatAnalyzer()
    return analyzer.analyze(audio_path)

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class AudioConfig:
    """Audio processing configuration optimized for beat detection"""
    sr: int = 22050  # Fixed sample rate for beat detection algorithm
    hop_length: int = 256  # Optimized hop length (11.61ms steps)
    n_mels: int = 128
    n_components: int = 3
    n_fft: int = 2048
    

@dataclass
class DataConfig:
    """Data configuration"""
    max_frames: int = 348
    max_meters: int = 203
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    

@dataclass
class ModelConfig:
    """Model configuration"""
    dropout: float = 0.3
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[Tuple[int, int]] = None
    cnn_pool_sizes: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [(3, 3), (3, 3)]
        if self.cnn_pool_sizes is None:
            self.cnn_pool_sizes = [(1, 2), (1, 2)]


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 16
    learning_rate: float = 0.001
    epochs: int = 50
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "reduce_lr_on_plateau"
    early_stopping_patience: int = 10
    

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    audio: AudioConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    experiment_name: str = "pytorch_melspec"
    version: str = "1"
    output_dir: str = "./models"
    

# ============================================================================
# Feature Extraction
# ============================================================================

def normalize_tempo(tempo: float) -> float:
    """Normalize tempo to reasonable range (60-160 BPM)"""
    if tempo <= 0:
        return 120.0
    
    # If below 60 bpm, double it
    while tempo < 60:
        tempo *= 2
    
    # If above 160 bpm, halve it  
    while tempo > 160:
        tempo /= 2
        
    return tempo

class FeatureExtractor:
    """Extract various audio features for chorus detection"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.beat_analyzer = BeatAnalyzer()
        
    def extract_melspectrogram(self, y: np.ndarray) -> np.ndarray:
        """Extract and normalize mel-spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.config.sr,
            n_mels=self.config.n_mels,
            hop_length=self.config.hop_length,
            n_fft=self.config.n_fft
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        return self._normalize_features(log_mel)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization"""
        mean = features.mean()
        std = features.std()
        return (features - mean) / (std + 1e-6)
    
    def _sort_by_peak_frequency(self, components: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sort components by peak frequency and return sorting indices"""
        peak_frequencies = []
        for component in components:
            # Find the frequency bin with maximum energy
            peak_bin = np.argmax(np.abs(component))
            peak_frequencies.append(peak_bin)
            
        # Get sorting indices
        sort_idx = np.argsort(peak_frequencies)
        
        # Sort components
        sorted_components = components[sort_idx]
        
        return sorted_components, sort_idx
    
    def extract_nmf_components(self, mel_spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Apply NMF decomposition to mel-spectrogram with sorted components"""
        # Ensure non-negative input for NMF
        positive_mel = mel_spec - mel_spec.min() + 1e-6
        
        # Transpose input to match sklearn's expected shape (n_samples, n_features)
        S = positive_mel.T
        
        # Initialize and fit NMF
        model = NMF(
            n_components=self.config.n_components,
            init='nndsvda',
            random_state=42,
            max_iter=500,
            tol=1e-3
        )
        
        # Track convergence warnings
        has_convergence_warning = False
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Get activations (W) and components (H)
            activations = model.fit_transform(S)  # Shape: (n_samples, n_components)
            components = model.components_        # Shape: (n_components, n_features)
            
            # Check for convergence warnings
            for warning in w:
                if "did not converge" in str(warning.message).lower():
                    has_convergence_warning = True
                    break
        
        # Sort components by peak frequency
        sorted_components, sort_idx = self._sort_by_peak_frequency(components)
        
        # Sort activations according to component sorting
        sorted_activations = activations[:, sort_idx]
        
        # Return in the expected format: (W, H, convergence_warning)
        return sorted_activations, sorted_components, has_convergence_warning
    
    def create_meter_grid(self, y: np.ndarray, song_data: dict = None, tempo: float = None) -> np.ndarray:
        """Create meter-based segmentation grid using advanced beat detection"""
        # Get beat analysis using the advanced beat detection
        beat_result = self.beat_analyzer.analyze_array(y)
        
        # Use detected tempo or fall back to provided/default values
        detected_tempo = normalize_tempo(beat_result.tempo_bpm)
        if song_data and 'sp_tempo' in song_data and not pd.isna(song_data['sp_tempo']):
            preferred_tempo = normalize_tempo(float(song_data['sp_tempo']))
            # Use Spotify tempo if it's reasonable, otherwise use detected
            if 60 <= preferred_tempo <= 160:
                bpm = preferred_tempo
            else:
                bpm = detected_tempo
        elif tempo is not None:
            bpm = normalize_tempo(tempo)
        else:
            bpm = detected_tempo
        
        # Ensure BPM is valid
        if bpm <= 0:
            bpm = 120.0
        
        # Get time signature (default to 4 if not available)
        time_signature = 4
        if song_data and 'sp_time_signature' in song_data and not pd.isna(song_data['sp_time_signature']):
            time_signature = int(song_data['sp_time_signature'])
            if time_signature == 0:
                time_signature = 4
        
        # Use detected beats if available and confident, otherwise generate grid
        if len(beat_result.beats_seconds) > 0 and beat_result.confidence > 0.3:
            beat_times = beat_result.beats_seconds
            
            # Create meter grid using time signature
            meter_indices = np.arange(0, len(beat_times), time_signature)
            meter_grid = beat_times[meter_indices]
            
            # Ensure grid starts at 0
            if len(meter_grid) == 0 or meter_grid[0] > 0.5:
                meter_grid = np.insert(meter_grid, 0, 0.0)
                
            # Ensure grid covers full duration
            total_duration = len(y) / self.config.sr
            if len(meter_grid) == 0 or meter_grid[-1] < total_duration - 1.0:
                # Add final meter at end
                meter_grid = np.append(meter_grid, total_duration)
        else:
            # Fall back to regular grid based on tempo
            total_duration = len(y) / self.config.sr
            seconds_per_beat = 60.0 / bpm
            seconds_per_meter = seconds_per_beat * time_signature
            
            num_meters = int(total_duration / seconds_per_meter) + 1
            meter_grid = np.arange(num_meters) * seconds_per_meter
            
            # Ensure grid doesn't exceed duration
            meter_grid = meter_grid[meter_grid <= total_duration]
            if len(meter_grid) == 0 or meter_grid[-1] < total_duration:
                meter_grid = np.append(meter_grid, total_duration)
        
        # Convert to frames using the beat analyzer's sample rate and hop length
        meter_frames = librosa.time_to_frames(
            meter_grid, 
            sr=self.config.sr, 
            hop_length=self.config.hop_length
        )
        
        return meter_frames


# ============================================================================
# PyTorch Dataset
# ============================================================================

class ChorusDataset(Dataset):
    """PyTorch dataset for chorus detection"""
    
    def __init__(
        self, 
        audio_files: List[str],
        labels_dir: str,
        config: ExperimentConfig,
        feature_extractor: FeatureExtractor,
        labels_csv: str = None
    ):
        self.audio_files = audio_files
        self.labels_dir = labels_dir
        self.config = config
        self.feature_extractor = feature_extractor
        
        # Load song data if CSV is provided
        self.song_data_dict = {}
        if labels_csv and os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            for song_id in df['song_id'].unique():
                song_rows = df[df['song_id'] == song_id]
                if not song_rows.empty:
                    self.song_data_dict[song_id] = song_rows.iloc[0].to_dict()
        
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_file = self.audio_files[idx]
        song_id = Path(audio_file).stem
        
        # Process audio and load labels
        features = self._process_audio(audio_file, song_id)
        labels = self._load_labels(song_id)
        
        return torch.FloatTensor(features), torch.FloatTensor(labels)
    
    def _process_audio(self, audio_file: str, song_id: str) -> np.ndarray:
        """Process audio file and extract features"""
        # Load audio
        y, _ = librosa.load(audio_file, sr=self.config.audio.sr)
        
        # Get song data if available
        song_data = self.song_data_dict.get(song_id, None)
        
        # Extract features in sequence
        mel_spec = self.feature_extractor.extract_melspectrogram(y)
        W, H, _ = self.feature_extractor.extract_nmf_components(mel_spec)
        meter_grid = self.feature_extractor.create_meter_grid(y, song_data)
        
        # Create segments 
        segments = []
        for i in range(len(meter_grid) - 1):
            start, end = int(meter_grid[i]), int(meter_grid[i + 1])
            if start < W.shape[0] and end <= W.shape[0] and start < end:
                segments.append(W[start:end])
        
        return self._pad_segments(segments)
    
    def _pad_segments(self, segments: List[np.ndarray]) -> np.ndarray:
        """Pad segments to fixed dimensions"""
        max_meters = self.config.data.max_meters
        max_frames = self.config.data.max_frames
        n_components = self.config.audio.n_components
        
        padded = np.zeros((max_meters, max_frames, n_components))
        
        for i, segment in enumerate(segments[:max_meters]):
            frames = min(segment.shape[0], max_frames)
            padded[i, :frames, :] = segment[:frames, :]
            
        return padded
    
    def _load_labels(self, song_id: str) -> np.ndarray:
        """Load and process labels"""
        label_file = os.path.join(
            self.labels_dir,
            f"{song_id}_labels_{self.config.experiment_name}_v{self.config.version}.pkl.gz"
        )
        
        with gzip.open(label_file, 'rb') as f:
            labels = pickle.load(f)
            
        # Pad or truncate labels
        target_size = self.config.data.max_meters
        if len(labels) > target_size:
            labels = labels[:target_size]
        else:
            labels = np.pad(labels, (0, target_size - len(labels)), constant_values=-1)
            
        return labels.reshape(-1, 1)


# ============================================================================
# PyTorch Model
# ============================================================================

class ChorusDetectionModel(nn.Module):
    """CNN-based model for chorus detection"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        # Build CNN layers efficiently
        layers = []
        in_channels = 1
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(zip(
            config.model.cnn_channels,
            config.model.cnn_kernel_sizes,
            config.model.cnn_pool_sizes
        )):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size)
            ])
            in_channels = out_channels
        
        self.cnn_layers = nn.Sequential(*layers)
        
        # Calculate output size after convolutions
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Meter-level processing layers
        self.meter_processor = nn.Sequential(
            nn.Linear(self.conv_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after conv layers"""
        dummy_input = torch.zeros(1, 1, self.config.audio.n_components, self.config.data.max_frames)
        
        with torch.no_grad():
            x = self.cnn_layers(dummy_input)
            return x.numel() // x.size(0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        batch_size, max_meters = x.size(0), x.size(1)
        
        # Process all meters in parallel
        # Reshape: (batch, meters, frames, components) -> (batch*meters, 1, components, frames)
        x_reshaped = x.transpose(2, 3).unsqueeze(2).view(-1, 1, self.config.audio.n_components, self.config.data.max_frames)
            
            # Apply CNN layers
        features = self.cnn_layers(x_reshaped)
        features = features.view(batch_size * max_meters, -1)
        
        # Apply meter processor
        outputs = self.meter_processor(features)
        
        # Reshape back to (batch, meters, 1)
        return outputs.view(batch_size, max_meters, 1)

class MaskedBCELoss(nn.Module):
    """Binary cross-entropy loss with masking for padding"""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = (targets != -1).float()
        valid_targets = torch.clamp(targets, 0, 1)
        loss = self.bce(predictions, valid_targets)
        masked_loss = loss * mask
        num_valid = torch.sum(mask)
        
        if num_valid > 0:
            return torch.sum(masked_loss) / num_valid
        else:
            return torch.tensor(0.0, device=predictions.device)


class Trainer:
    """Model trainer with logging and checkpointing"""
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup loss and optimizer
        self.criterion = MaskedBCELoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config"""
        params = self.model.parameters()
        lr = self.config.training.learning_rate
        wd = self.config.training.weight_decay
        
        if self.config.training.optimizer == "adam":
            return Adam(params, lr=lr, weight_decay=wd)
        elif self.config.training.optimizer == "sgd":
            return SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        elif self.config.training.optimizer == "adamw":
            return AdamW(params, lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
            
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.config.training.scheduler == "reduce_lr_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            )
        else:
            return None
            
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                # Collect predictions
                mask = targets != -1
                all_predictions.extend(outputs[mask].cpu().numpy().flatten())
                all_targets.extend(targets[mask].cpu().numpy().flatten())
                
        val_loss /= len(val_loader)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        return val_loss, metrics
    
    def _calculate_metrics(
        self, 
        predictions: List[float], 
        targets: List[float]
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        if not predictions:
            return {}
            
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Binarize predictions
        binary_preds = (predictions > 0.5).astype(int)
        binary_targets = targets.astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_targets, binary_preds, average='binary', zero_division=0
        )
        accuracy = accuracy_score(binary_targets, binary_preds)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = None
    ) -> None:
        """Full training loop"""
        epochs = epochs or self.config.training.epochs
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Using device: {self.device}")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(metrics)
            
            # Log progress
            logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val F1: {metrics.get('f1', 0):.4f}"
            )
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
                    
    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': asdict(self.config),
            'history': self.history
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")


# ============================================================================
# Data Pipeline
# ============================================================================

def create_data_loaders(
    song_ids: List[str],
    features_dir: str,
    labels_dir: str,
    config: ExperimentConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create optimized data loaders for preprocessed data"""
    
    # Shuffle and split song IDs
    random.shuffle(song_ids)
    n_files = len(song_ids)
    n_train = int(n_files * config.data.train_split)
    n_val = int(n_files * config.data.val_split)
    
    train_ids = song_ids[:n_train]
    val_ids = song_ids[n_train:n_train + n_val]
    test_ids = song_ids[n_train + n_val:]
    
    logger.info(f"Data split - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Create datasets
    train_dataset = PreprocessedChorusDataset(train_ids, features_dir, labels_dir, config)
    val_dataset = PreprocessedChorusDataset(val_ids, features_dir, labels_dir, config)
    test_dataset = PreprocessedChorusDataset(test_ids, features_dir, labels_dir, config)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Evaluation and Visualization
# ============================================================================

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on test set"""
    trainer = Trainer(model, config, device)
    test_loss, metrics = trainer.validate(test_loader)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    for metric, value in metrics.items():
        logger.info(f"Test {metric}: {value:.4f}")
        
    return metrics


def visualize_results(
    model: nn.Module,
    test_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    num_samples: int = 3
) -> None:
    """Visualize model predictions"""
    model.eval()
    samples_shown = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            if samples_shown >= num_samples:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            # Plot first sample in batch
            sample_outputs = outputs[0].cpu().numpy().flatten()
            sample_targets = targets[0].cpu().numpy().flatten()
            
            # Filter valid positions
            valid_mask = sample_targets != -1
            valid_outputs = sample_outputs[valid_mask]
            valid_targets = sample_targets[valid_mask]
            
            if len(valid_outputs) == 0:
                continue
                
            plt.figure(figsize=(12, 4))
            plt.plot(valid_outputs, label='Predictions', alpha=0.8)
            plt.plot(valid_targets, label='Ground Truth', alpha=0.8)
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
            plt.ylim(-0.1, 1.1)
            plt.xlabel('Time (meters)')
            plt.ylabel('Chorus Probability')
            plt.title(f'Chorus Detection - Sample {samples_shown + 1}')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            samples_shown += 1


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main(args):
    """Main training pipeline"""
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    base_data_dir = "../data" 
    dirs = create_data_structure(base_data_dir, args.experiment_name, args.version)
    
    # Load dataset statistics
    stats_file = os.path.join(dirs['experiment_results'], f"dataset_stats_{args.experiment_name}_v{args.version}.json")
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Create experiment config
    config = ExperimentConfig(
        audio=AudioConfig(
            sr=args.sample_rate,
            hop_length=args.hop_length,
            n_components=args.n_components
        ),
        data=DataConfig(
            max_frames=stats['max_frames'],
            max_meters=stats['max_meters'],
            train_split=args.train_split,
            val_split=args.val_split
        ),
        model=ModelConfig(
            dropout=args.dropout
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            optimizer=args.optimizer
        ),
        experiment_name=args.experiment_name,
        version=args.version,
        output_dir=dirs['experiment_models']
    )
    
    # Save config
    config_path = os.path.join(dirs['experiment_results'], "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Saved config to {config_path}")
    
    # Get all preprocessed song IDs
    experiment_features_dir = dirs['experiment_features']
    song_ids = []
    for file in os.listdir(experiment_features_dir):
        if file.endswith('.pkl.gz') and 'features' in file:
            song_id = file.split('_features_')[0]
            song_ids.append(song_id)
    
    logger.info(f"Found {len(song_ids)} preprocessed songs")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        song_ids, experiment_features_dir, dirs['experiment_labels'], config
    )
    
    # Create model
    model = ChorusDetectionModel(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, config, trainer.device)
    
    # Save final results
    results_path = os.path.join(dirs['experiment_results'], "results.json")
    with open(results_path, "w") as f:
        json.dump({
            'config': asdict(config),
            'test_metrics': test_metrics,
            'history': trainer.history
        }, f, indent=2)
    
    # Visualize results
    if args.visualize:
        visualize_results(model, test_loader, config, trainer.device)
        
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {dirs['experiment_results']}")
    logger.info(f"Models saved to: {dirs['experiment_models']}")


def preprocess_dataset(args):
    """Preprocess entire dataset and save features/labels as pkl.gz files"""
    logger.info("Starting dataset preprocessing...")
    
    # Create unified data structure
    base_data_dir = "../data" 
    dirs = create_data_structure(base_data_dir, args.experiment_name, args.version)
    
    # Load labels CSV
    df = pd.read_csv(args.labels_csv)
    
    # Get unique songs with their data
    song_data_dict = {}
    for song_id in df['song_id'].unique():
        song_data = df[df['song_id'] == song_id].iloc[0].to_dict()
        audio_path = os.path.join(args.audio_dir, f"{song_id}.mp3")
        if os.path.exists(audio_path):
            song_data_dict[song_id] = {
                'audio_path': audio_path,
                'song_data': song_data
            }
    
    logger.info(f"Found {len(song_data_dict)} audio files to process")
    
    # Initialize feature extractor
    audio_config = AudioConfig(
        sr=args.sample_rate,
        hop_length=args.hop_length,
        n_components=args.n_components
    )
    feature_extractor = FeatureExtractor(audio_config)
    
    # Track dataset statistics and convergence warnings
    all_frame_counts = []
    all_meter_counts = []
    convergence_warnings = []
    skipped_count = 0
    processed_count = 0
    
    for i, (song_id, song_info) in enumerate(song_data_dict.items()):
        # Check if files already exist
        features_file = os.path.join(
            dirs['experiment_features'],
            f"{song_id}_features_{args.experiment_name}_v{args.version}.pkl.gz"
        )
        labels_file = os.path.join(
            dirs['experiment_labels'],
            f"{song_id}_labels_{args.experiment_name}_v{args.version}.pkl.gz"
        )
        
        if os.path.exists(features_file) and os.path.exists(labels_file):
            logger.info(f"Skipping {song_id} ({i + 1}/{len(song_data_dict)}) - files already exist")
            skipped_count += 1
            continue
            
        logger.info(f"Processing {song_id} ({i + 1}/{len(song_data_dict)})")
        
        audio_file = song_info['audio_path']
        song_data = song_info['song_data']
        
        # Load and process audio
        y, _ = librosa.load(audio_file, sr=audio_config.sr)
        
        # Extract features
        mel_spec = feature_extractor.extract_melspectrogram(y)
        W, H, has_convergence_warning = feature_extractor.extract_nmf_components(mel_spec)
        
        # Track convergence warnings
        if has_convergence_warning:
            convergence_warnings.append(song_id)
            logger.warning(f"NMF convergence warning for {song_id}")
            
        meter_grid = feature_extractor.create_meter_grid(y, song_data)
        
        # Segment features by meters
        segments = []
        segment_frame_counts = []
        
        for j in range(len(meter_grid) - 1):
            start, end = int(meter_grid[j]), int(meter_grid[j + 1])
            if start < W.shape[0] and end <= W.shape[0] and start < end:
                segment = W[start:end]
                segments.append(segment)
                segment_frame_counts.append(segment.shape[0])
        
        # Track statistics
        if segment_frame_counts:
            all_frame_counts.extend(segment_frame_counts)
            all_meter_counts.append(len(segments))
        
        # Generate labels for this song
        labels = generate_labels_for_song(song_id, df, len(segments))
        
        # Save features
        with gzip.open(features_file, 'wb') as f:
            pickle.dump({
                'segments': segments,
                'mel_spec': mel_spec,
                'nmf_components': (W, H),
                'meter_grid': meter_grid,
                'frame_counts': segment_frame_counts,
                'song_data': song_data,
                'convergence_warning': has_convergence_warning
            }, f)
        
        # Save labels
        with gzip.open(labels_file, 'wb') as f:
            pickle.dump(labels, f)
            
        processed_count += 1
    
    # Calculate dataset statistics
    max_frames = max(all_frame_counts) if all_frame_counts else 300
    max_meters = max(all_meter_counts) if all_meter_counts else 200
    
    stats = {
        'total_files': len(song_data_dict),
        'processed_files': processed_count,
        'skipped_files': skipped_count,
        'max_frames': max_frames,
        'max_meters': max_meters,
        'avg_frames': np.mean(all_frame_counts) if all_frame_counts else 0,
        'avg_meters': np.mean(all_meter_counts) if all_meter_counts else 0,
        'frame_counts': all_frame_counts,
        'meter_counts': all_meter_counts,
        'convergence_warnings': convergence_warnings,
        'convergence_warning_count': len(convergence_warnings)
    }
    
    # Save statistics
    stats_file = os.path.join(dirs['experiment_results'], f"dataset_stats_{args.experiment_name}_v{args.version}.json")
    with open(stats_file, 'w') as f:
        stats_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()}
        json.dump(stats_json, f, indent=2)
    
    logger.info(f"Preprocessing completed!")
    logger.info(f"Processed: {processed_count} files")
    logger.info(f"Skipped: {skipped_count} files (already existed)")
    logger.info(f"Convergence warnings: {len(convergence_warnings)} files")
    if convergence_warnings:
        logger.info(f"Files with convergence warnings: {convergence_warnings}")
    logger.info(f"Max frames per segment: {max_frames}")
    logger.info(f"Max meters per song: {max_meters}")
    logger.info(f"Features saved to: {dirs['experiment_features']}")
    logger.info(f"Labels saved to: {dirs['experiment_labels']}")
    logger.info(f"Statistics saved to: {stats_file}")


def generate_labels_for_song(song_id: str, df: pd.DataFrame, num_segments: int) -> np.ndarray:
    """Generate binary labels for song segments based on chorus annotations"""
    song_data = df[df['song_id'] == song_id]
    labels = np.zeros(num_segments)
    
    # Process each labeled segment
    for _, row in song_data.iterrows():
        if row['label'] == 'chorus':
            start_frame = int(row['start_frame'])
            end_frame = int(row['end_frame'])
            
            # Convert frame indices to segment indices
            frames_per_segment = 348  # Based on your max_frames
            start_segment = max(0, start_frame // frames_per_segment)
            end_segment = min(num_segments, end_frame // frames_per_segment + 1)
            
            labels[start_segment:end_segment] = 1
    
    return labels


def load_preprocessed_features(song_id: str, features_dir: str, experiment_name: str, version: str) -> Dict:
    """Load preprocessed features from pkl.gz file"""
    features_file = os.path.join(
        features_dir,
        f"{song_id}_features_{experiment_name}_v{version}.pkl.gz"
    )
    
    with gzip.open(features_file, 'rb') as f:
        return pickle.load(f)


class PreprocessedChorusDataset(Dataset):
    """Optimized dataset that loads preprocessed features"""
    
    def __init__(
        self,
        song_ids: List[str],
        features_dir: str,
        labels_dir: str,
        config: ExperimentConfig
    ):
        self.song_ids = song_ids
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.config = config
        
    def __len__(self) -> int:
        return len(self.song_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        song_id = self.song_ids[idx]
        
        # Load preprocessed features
        features_data = load_preprocessed_features(
            song_id, self.features_dir, self.config.experiment_name, self.config.version
        )
        
        features = self._pad_segments(features_data['segments'])
        
        # Load labels
        labels = self._load_labels(song_id)
        
        return torch.FloatTensor(features), torch.FloatTensor(labels)
    
    def _pad_segments(self, segments: List[np.ndarray]) -> np.ndarray:
        """Pad segments to fixed dimensions"""
        max_meters = self.config.data.max_meters
        max_frames = self.config.data.max_frames
        n_components = self.config.audio.n_components
        
        padded = np.zeros((max_meters, max_frames, n_components))
        
        for i, segment in enumerate(segments[:max_meters]):
            frames = min(segment.shape[0], max_frames)
            padded[i, :frames, :] = segment[:frames, :]
            
        return padded
    
    def _load_labels(self, song_id: str) -> np.ndarray:
        """Load preprocessed labels"""
        label_file = os.path.join(
            self.labels_dir,
            f"{song_id}_labels_{self.config.experiment_name}_v{self.config.version}.pkl.gz"
        )
        
        with gzip.open(label_file, 'rb') as f:
            labels = pickle.load(f)
            
        # Pad or truncate labels
        if len(labels) > self.config.data.max_meters:
            labels = labels[:self.config.data.max_meters]
        else:
            labels = np.pad(
                labels,
                (0, self.config.data.max_meters - len(labels)),
                constant_values=-1
            )
            
        return labels.reshape(-1, 1)


def create_data_structure(base_data_dir: str, experiment_name: str, version: str) -> Dict[str, str]:
    """Create unified data directory structure and return paths"""
    
    # Define directory structure
    dirs = {
        # Base directories
        'audio_raw': os.path.join(base_data_dir, 'audio', 'raw'),
        'audio_processed': os.path.join(base_data_dir, 'audio', 'processed'),
        'metadata': os.path.join(base_data_dir, 'metadata'),
        
        # Experiment-specific directories (all data goes here)
        'experiment_base': os.path.join(base_data_dir, 'experiments', experiment_name, f'v{version}'),
        'experiment_features': os.path.join(base_data_dir, 'experiments', experiment_name, f'v{version}', 'features'),
        'experiment_labels': os.path.join(base_data_dir, 'experiments', experiment_name, f'v{version}', 'labels'),
        'experiment_models': os.path.join(base_data_dir, 'experiments', experiment_name, f'v{version}', 'models'),
        'experiment_results': os.path.join(base_data_dir, 'experiments', experiment_name, f'v{version}', 'results'),
    }
    
    # Create all directories
    for dir_name, dir_path in dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chorus detection model")
    
    # Data arguments
    parser.add_argument("--audio-dir", type=str, 
                        default="../data/audio/processed",
                        help="Directory containing audio files")
    parser.add_argument("--labels-csv", type=str, 
                        default="../data/metadata/metadata.csv",
                        help="CSV file with chorus labels and song metadata")
    parser.add_argument("--labels-dir", type=str, 
                        default="../data/experiments",
                        help="Directory to save/load processed labels and features")
    
    # Audio arguments
    parser.add_argument("--sample-rate", type=int, default=22050,
                        help="Audio sample rate (optimized for beat detection)")
    parser.add_argument("--hop-length", type=int, default=256,
                        help="Hop length for feature extraction (optimized for beat detection)")
    parser.add_argument("--n-components", type=int, default=3,
                        help="Number of NMF components")
    
    # Model arguments
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd", "adamw"],
                        help="Optimizer type")
    
    # Data split arguments
    parser.add_argument("--train-split", type=float, default=0.7,
                        help="Training data split ratio")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Validation data split ratio")
    
    # Experiment arguments
    parser.add_argument("--experiment-name", type=str, default="pytorch_melspec",
                        help="Experiment name")
    parser.add_argument("--version", type=str, default="1",
                        help="Experiment version")
    parser.add_argument("--output-dir", type=str, 
                        default="../data/experiments/pytorch_melspec/v1/models",
                        help="Output directory for models and results")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization plots")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only run preprocessing, don't train")
    
    args = parser.parse_args()
    
    if args.preprocess_only:
        preprocess_dataset(args)
    else:
        main(args)