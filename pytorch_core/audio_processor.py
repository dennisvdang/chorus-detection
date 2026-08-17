"""
PyTorch audio processing functionality for chorus detection.
Built upon the original AudioFeature class with modular feature extraction.
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import librosa
from sklearn.preprocessing import StandardScaler

from pytorch_core.utils import strip_silence

# Constants
SR = 12000
HOP_LENGTH = 128
MAX_FRAMES = 300
MAX_METERS = 201
N_FEATURES = 15


def fold_tempo(tempo: float, low: float = 70.0, high: float = 180.0) -> float:
    """Fold a tempo into [low, high] by octave doubling or halving.

    A beat tracker reporting 200 BPM has most likely counted double time on a
    100 BPM song; halving preserves the bar structure, whereas clipping would
    build a grid at a tempo the song never has. The ceiling is 180 so that
    genuine fast tempos (drum and bass at 174, for instance) survive unfolded.
    A tempo of 0 or below cannot be folded and is returned unchanged for the
    caller to reject.

    This is the one tempo-range rule shared by training-data preprocessing and
    inference; change it here and both sides move together.
    """
    if tempo <= 0:
        return float(tempo)
    while tempo > high:
        tempo /= 2.0
    while tempo < low:
        tempo *= 2.0
    return float(tempo)


class AudioFeature:
    def __init__(self, audio_path, sr=SR, hop_length=HOP_LENGTH):
        """Initialize the AudioFeature with an audio file."""
        self.audio_path = audio_path
        self.sr = sr
        self.hop_length = hop_length
        
        # Audio data
        self.y = None
        self.y_harm = self.y_perc = None
        
        # Features
        self.beats = None
        self.chromagram = self.chroma_acts = None
        self.combined_features = None
        self.key = self.mode = None
        self.mel_acts = self.melspectrogram = None
        self.meter_grid = None
        self.mfccs = self.mfcc_acts = None
        self.n_frames = None
        self.onset_env = None
        self.rms = None
        self.spectrogram = None
        self.tempo = None
        self.tempogram = self.tempogram_acts = None
        self.time_signature = 4
        
        # Feature extraction tracking
        self._extracted_features = set()

    def load_audio(self):
        """Load audio and separate harmonic/percussive components."""
        self.y, self.sr = librosa.load(self.audio_path, sr=self.sr)
        self.y_harm, self.y_perc = librosa.effects.hpss(self.y)
        return self.y
    
    def extract_spectrogram(self):
        """Extract spectrogram from the loaded audio."""
        if self.y is None:
            self.load_audio()
            
        self.spectrogram, _ = librosa.magphase(librosa.stft(self.y, hop_length=self.hop_length))
        self._extracted_features.add('spectrogram')
        return self.spectrogram
    
    def extract_rms(self):
        """Extract RMS energy from the spectrogram."""
        if 'spectrogram' not in self._extracted_features:
            self.extract_spectrogram()
            
        self.rms = librosa.feature.rms(S=self.spectrogram, hop_length=self.hop_length).astype(np.float32)
        self._extracted_features.add('rms')
        return self.rms
    
    def extract_melspectrogram(self):
        """Extract mel spectrogram from the audio."""
        if self.y is None:
            self.load_audio()
            
        self.melspectrogram = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_mels=128, hop_length=self.hop_length).astype(np.float32)
        self._extracted_features.add('melspectrogram')
        return self.melspectrogram
    
    def extract_mel_components(self, n_components=3):
        """Extract NMF components from the mel spectrogram."""
        if 'melspectrogram' not in self._extracted_features:
            self.extract_melspectrogram()
            
        self.mel_acts = librosa.decompose.decompose(
            self.melspectrogram, n_components=n_components, sort=True)[1].astype(np.float32)
        self._extracted_features.add('mel_acts')
        return self.mel_acts
    
    def detect_key(self, chroma_vals: np.ndarray) -> Tuple[str, str]:
        """Detect the key and mode (major or minor) of the audio segment."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize profiles
        major_profile /= np.linalg.norm(major_profile)
        minor_profile /= np.linalg.norm(minor_profile)

        # Calculate correlations for all possible keys
        major_correlations = [np.corrcoef(chroma_vals, np.roll(major_profile, i))[0, 1] for i in range(12)]
        minor_correlations = [np.corrcoef(chroma_vals, np.roll(minor_profile, i))[0, 1] for i in range(12)]

        # Find best match
        max_major_idx = np.argmax(major_correlations)
        max_minor_idx = np.argmax(minor_correlations)

        self.mode = 'major' if major_correlations[max_major_idx] > minor_correlations[max_minor_idx] else 'minor'
        self.key = note_names[max_major_idx if self.mode == 'major' else max_minor_idx]
        return self.key, self.mode
    
    def calculate_ki_chroma(self, waveform: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
        """Calculate a normalized, key-invariant chromagram."""
        chromagram = librosa.feature.chroma_cqt(y=waveform, sr=sr, hop_length=hop_length, bins_per_octave=24)
        chromagram = (chromagram - chromagram.min()) / (chromagram.max() - chromagram.min())
        
        chroma_vals = np.sum(chromagram, axis=1)
        key, mode = self.detect_key(chroma_vals)
        
        key_idx = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'].index(key)
        shift_amount = -key_idx if mode == 'major' else -(key_idx + 3) % 12
        
        return librosa.util.normalize(np.roll(chromagram, shift_amount, axis=0), axis=1)
    
    def extract_chromagram(self):
        """Extract key-invariant chromagram from the harmonic component."""
        if self.y_harm is None:
            self.load_audio()
            
        self.chromagram = self.calculate_ki_chroma(
            self.y_harm, self.sr, self.hop_length).astype(np.float32)
        self._extracted_features.add('chromagram')
        return self.chromagram
    
    def extract_chroma_components(self, n_components=4):
        """Extract NMF components from the chromagram."""
        if 'chromagram' not in self._extracted_features:
            self.extract_chromagram()
            
        self.chroma_acts = librosa.decompose.decompose(
            self.chromagram, n_components=n_components, sort=True)[1].astype(np.float32)
        self._extracted_features.add('chroma_acts')
        return self.chroma_acts
    
    def extract_onset_envelope(self):
        """Extract onset envelope from the percussive component."""
        if self.y_perc is None:
            self.load_audio()
            
        self.onset_env = librosa.onset.onset_strength(
            y=self.y_perc, sr=self.sr, hop_length=self.hop_length)
        self._extracted_features.add('onset_env')
        return self.onset_env
    
    def extract_tempogram(self):
        """Extract tempogram from the onset envelope."""
        if 'onset_env' not in self._extracted_features:
            self.extract_onset_envelope()
            
        self.tempogram = np.clip(librosa.feature.tempogram(
            onset_envelope=self.onset_env, sr=self.sr, hop_length=self.hop_length), 0, None)
        self._extracted_features.add('tempogram')
        return self.tempogram
    
    def extract_tempogram_components(self, n_components=3):
        """Extract NMF components from the tempogram."""
        if 'tempogram' not in self._extracted_features:
            self.extract_tempogram()
            
        self.tempogram_acts = librosa.decompose.decompose(
            self.tempogram, n_components=n_components, sort=True)[1]
        self._extracted_features.add('tempogram_acts')
        return self.tempogram_acts
    
    def extract_mfccs(self, n_mfcc=20):
        """Extract MFCCs from the audio."""
        if self.y is None:
            self.load_audio()
            
        self.mfccs = librosa.feature.mfcc(
            y=self.y, sr=self.sr, n_mfcc=n_mfcc, hop_length=self.hop_length)
        self.mfccs += abs(np.min(self.mfccs))  # Make all values positive for NMF
        self._extracted_features.add('mfccs')
        return self.mfccs
    
    def extract_mfcc_components(self, n_components=4):
        """Extract NMF components from the MFCCs."""
        if 'mfccs' not in self._extracted_features:
            self.extract_mfccs()
            
        self.mfcc_acts = librosa.decompose.decompose(
            self.mfccs, n_components=n_components, sort=True)[1].astype(np.float32)
        self._extracted_features.add('mfcc_acts')
        return self.mfcc_acts
    
    def combine_all_features(self):
        """Combine all extracted features with weighted normalization."""
        # Extract all features if not already done
        self.extract_rms()
        self.extract_mel_components()
        self.extract_chroma_components()
        self.extract_tempogram_components()
        self.extract_mfcc_components()
        
        # Collect features for combination
        features = [self.rms, self.mel_acts, self.chroma_acts, self.tempogram_acts, self.mfcc_acts]
        feature_names = ['rms', 'mel_acts', 'chroma_acts', 'tempogram_acts', 'mfcc_acts']
        
        # Calculate weights for each feature type
        dims = {name: feature.shape[0] for feature, name in zip(features, feature_names)}
        total_inv_dim = sum(1 / dim for dim in dims.values())
        weights = {name: 1 / (dims[name] * total_inv_dim) for name in feature_names}
        
        # Standardize and weight features
        std_weighted_features = [
            StandardScaler().fit_transform(feature.T).T * weights[name]
            for feature, name in zip(features, feature_names)
        ]
        
        self.combined_features = np.concatenate(std_weighted_features, axis=0).T.astype(np.float32)
        self.n_frames = len(self.combined_features)
        self._extracted_features.add('combined_features')
        return self.combined_features
    
    def extract_features(self):
        """
        Extract all audio features (maintained for compatibility).
        This is now a wrapper that calls individual extraction methods.
        """
        # Load audio if not already done
        if self.y is None:
            self.load_audio()
            
        # Extract all features
        self.extract_spectrogram()
        self.extract_rms()
        self.extract_melspectrogram()
        self.extract_mel_components()
        self.extract_chromagram()
        self.extract_chroma_components()
        self.extract_onset_envelope()
        self.extract_tempogram()
        self.extract_tempogram_components()
        self.extract_mfccs()
        self.extract_mfcc_components()
        self.combine_all_features()
        
        return self.combined_features
    
    def detect_beats(self):
        """Detect tempo and beat locations."""
        if 'onset_env' not in self._extracted_features:
            self.extract_onset_envelope()
            
        self.tempo, self.beats = librosa.beat.beat_track(
            onset_envelope=self.onset_env, sr=self.sr, hop_length=self.hop_length)
        
        # Fold into the shared tempo range
        self.tempo = fold_tempo(float(np.atleast_1d(self.tempo)[0]))
            
        self._extracted_features.add('beats')
        return self.tempo, self.beats
    
    def create_meter_grid(self, bpm: Optional[float] = None,
                          time_signature: Optional[int] = None,
                          grid_source: str = "librosa",
                          device: str = "cpu") -> np.ndarray:
        """Create a grid based on the meter of the song, using tempo and beats.

        Args:
            bpm: Optional known tempo (e.g. from dataset metadata). Folded by
                octave into the shared range (see fold_tempo). When None, the
                tempo detected by detect_beats() is used.
            time_signature: Optional known time signature (beats per meter).
            grid_source: "librosa" extrapolates a single tempo over the song
                (original behavior); "beat_this" builds the grid from downbeats
                tracked by Beat This!. Inference must use the same grid_source
                the model was trained on.
            device: Torch device for the Beat This! tracker.
        """
        if 'beats' not in self._extracted_features:
            self.detect_beats()

        if bpm is not None:
            self.tempo = fold_tempo(bpm)
        if time_signature is not None:
            self.time_signature = int(time_signature)

        if grid_source == "beat_this":
            from pytorch_core.downbeats import create_downbeat_meter_grid, track_downbeats
            _, downbeats = track_downbeats(self.audio_path, device=device)
            if len(downbeats) < 2:
                raise ValueError("beat_this returned <2 downbeats")
            self.meter_grid = create_downbeat_meter_grid(
                downbeats, self.n_frames, self.sr, self.hop_length)
        else:
            self.meter_grid = self._create_meter_grid()
        return self.meter_grid

    def _create_meter_grid(self) -> np.ndarray:
        """Helper function to create a meter grid for the song.

        Mirrors the meter grid used to build the training data
        (notebooks/Automated-Chorus-Detection-V2.ipynb / scripts/preprocess.py).
        """
        first_beat_frame = self.beats[0] if len(self.beats) > 0 else 0
        first_beat_time = librosa.frames_to_time(first_beat_frame, sr=self.sr, hop_length=self.hop_length)
        time_duration = librosa.frames_to_time(self.n_frames, sr=self.sr, hop_length=self.hop_length)
        seconds_per_beat = 60.0 / self.tempo

        # Calculate beats forward and backward from the first detected beat
        num_beats_forward = int((time_duration - first_beat_time) / seconds_per_beat)
        num_beats_backward = int(first_beat_time / seconds_per_beat) + 1
        beat_times_forward = first_beat_time + np.arange(num_beats_forward) * seconds_per_beat
        beat_times_backward = first_beat_time - np.arange(1, num_beats_backward) * seconds_per_beat

        # Combine and segment by meter
        beat_grid = np.concatenate((np.array([0.0]), beat_times_backward[::-1], beat_times_forward))
        meter_indices = np.arange(0, len(beat_grid), self.time_signature)
        meter_grid = beat_grid[meter_indices]

        # Ensure grid starts at 0 and ends at the final frame
        if meter_grid[0] != 0.0:
            meter_grid = np.insert(meter_grid, 0, 0.0)
        meter_grid_frames = librosa.time_to_frames(meter_grid, sr=self.sr, hop_length=self.hop_length)
        if meter_grid_frames[-1] != self.n_frames:
            meter_grid_frames = np.append(meter_grid_frames, self.n_frames)

        return meter_grid_frames


def segment_data_meters(data: np.ndarray, meter_grid: List[int]) -> List[np.ndarray]:
    """Segment input data into chunks based on a meter grid."""
    return [data[meter_grid[i]:meter_grid[i+1]] for i in range(len(meter_grid) - 1)]


def positional_encoding(position: int, d_model: int) -> np.ndarray:
    """Generate positional encodings for the given number of positions.

    Same encoding used to build the training data
    (notebooks/Automated-Chorus-Detection-V2.ipynb / scripts/preprocess.py).
    """
    angle_rads = (
        np.arange(position)[:, np.newaxis] /
        np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    )
    return np.concatenate([np.sin(angle_rads[:, 0::2]), np.cos(angle_rads[:, 1::2])], axis=-1)


def apply_hierarchical_positional_encoding(segments: List[np.ndarray]) -> List[np.ndarray]:
    """Apply positional encoding at the meter and frame levels to a list of segments."""
    n_features = segments[0].shape[1]
    meter_level_encodings = positional_encoding(len(segments), n_features)
    return [
        seg + positional_encoding(len(seg), n_features) + meter_level_encodings[i]
        for i, seg in enumerate(segments)
    ]


def pad_song(encoded_segments: List[np.ndarray], max_frames: int = MAX_FRAMES, 
             max_meters: int = MAX_METERS, n_features: int = N_FEATURES) -> np.ndarray:
    """
    Pad a list of encoded segments to create a uniform 3D array.
    
    Parameters:
    - encoded_segments (list): List of encoded data segments
    - max_frames (int): Maximum number of frames per segment
    - max_meters (int): Maximum number of meters
    - n_features (int): Number of features per frame
    
    Returns:
    - np.ndarray: Padded 3D array of shape (max_meters, max_frames, n_features)
    """
    padded_song = np.zeros((max_meters, max_frames, n_features))
    
    for i, segment in enumerate(encoded_segments):
        if i >= max_meters:
            break  # Only consider up to max_meters segments
            
        segment_frames = segment.shape[0]
        if segment_frames <= max_frames:
            # If segment fits, copy it directly
            padded_song[i, :segment_frames, :] = segment
        else:
            # If segment is too long, sample frames evenly
            indices = np.linspace(0, segment_frames - 1, max_frames, dtype=int)
            padded_song[i, :, :] = segment[indices, :]
            
    return padded_song


def process_audio(audio_path, trim_silence=True, sr=SR, hop_length=HOP_LENGTH,
                  extract_spectrogram_only=False, bpm=None, time_signature=None,
                  grid_source="librosa", device="cpu"):
    """
    Process an audio file for chorus detection.

    Args:
        audio_path: Path to audio file
        trim_silence: Whether to strip silence from the audio
        sr: Sample rate
        hop_length: Hop length for feature extraction
        extract_spectrogram_only: If True, only extract raw spectrograms for the spectrogram-based model
        bpm: Optional known tempo; falls back to beat-tracker estimate when None
        time_signature: Optional known time signature (beats per meter)
        grid_source: Meter grid source, "librosa" or "beat_this". Must match the
            grid the model was trained on.
        device: Torch device for the Beat This! tracker (beat_this grid only)

    Returns:
        Tuple of (padded_song, audio_features)
    """
    try:
        # Optionally strip silence
        if trim_silence:
            strip_silence(audio_path)

        # Extract audio features
        audio_features = AudioFeature(audio_path, sr=sr, hop_length=hop_length)
        
        if extract_spectrogram_only:
            # For spectrogram-based model
            audio_features.extract_spectrogram()
            # We need the combined features format even if we only use the spectrogram
            audio_features.n_frames = audio_features.spectrogram.shape[1]
            audio_features.combined_features = audio_features.spectrogram.T
        else:
            # Extract all features
            audio_features.extract_features()
        
        # Create meter grid and segment
        meter_grid = audio_features.create_meter_grid(bpm=bpm, time_signature=time_signature,
                                                      grid_source=grid_source, device=device)
        feature_segments = segment_data_meters(audio_features.combined_features, meter_grid)
        encoded_segments = apply_hierarchical_positional_encoding(feature_segments)
        
        # Calculate actual feature dimension
        n_features = audio_features.combined_features.shape[1]
        
        # Pad song
        padded_song = pad_song(encoded_segments, n_features=n_features)

        # Add batch dimension for model
        padded_song = np.expand_dims(padded_song, axis=0)
        return padded_song, audio_features
    
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None 