#!/usr/bin/env python
"""
Preprocessing script for chorus detection model training.

Generates per-song segment and label pickles from processed audio files and
the labeled dataset CSV. This is the reproducible-script version of the
feature pipeline in notebooks/Automated-Chorus-Detection-V2.ipynb and produces
the data expected by pytorch_core/data/dataset.py and scripts/train.py:

    data/segments_V2/{song_id}_data.pkl    list of [n_frames, 15] float32 arrays
    data/labels_V2/{song_id}_labels.pkl    int32 array of per-meter labels

Pipeline per song:
    1. Load audio at 12 kHz, split harmonic/percussive components
    2. Extract RMS, mel/chroma/tempogram/MFCC NMF activations (3+4+3+4 comps)
    3. Standardize each feature group and weight inversely to its dimension
    4. Build a meter grid from the CSV tempo/time signature (librosa fallback)
    5. Align chorus labels to meters (chorus if >=25% of frames are chorus)
    6. Segment features by meter and apply hierarchical positional encoding

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --limit 3
    python scripts/preprocess.py --song-ids 101 102 103
"""

import argparse
import os
import pickle
import sys
import warnings
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TARGET_SR = 12000
HOP_LENGTH = 128

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def detect_key(chroma_vals: np.ndarray) -> Tuple[str, str]:
    """Detect the key and mode (major or minor) of the audio segment."""
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_profile /= np.linalg.norm(major_profile)
    minor_profile /= np.linalg.norm(minor_profile)
    major_correlations = [np.corrcoef(chroma_vals, np.roll(major_profile, i))[0, 1] for i in range(12)]
    minor_correlations = [np.corrcoef(chroma_vals, np.roll(minor_profile, i))[0, 1] for i in range(12)]
    max_major_idx = int(np.argmax(major_correlations))
    max_minor_idx = int(np.argmax(minor_correlations))
    mode = 'major' if major_correlations[max_major_idx] > minor_correlations[max_minor_idx] else 'minor'
    key = NOTE_NAMES[max_major_idx if mode == 'major' else max_minor_idx]
    return key, mode


def calculate_ki_chroma(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Calculate a normalized, key-invariant chromagram for the given audio waveform."""
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=24)
    chromagram = (chromagram - chromagram.min()) / (chromagram.max() - chromagram.min())
    chroma_vals = np.sum(chromagram, axis=1)
    key, mode = detect_key(chroma_vals)
    key_idx = NOTE_NAMES.index(key)
    shift_amount = -key_idx if mode == 'major' else -(key_idx + 3) % 12
    return librosa.util.normalize(np.roll(chromagram, shift_amount, axis=0), axis=1)


def fold_tempo(tempo: float, low: float = 70.0, high: float = 140.0) -> float:
    """Fold a tempo into [low, high] by octave doubling or halving.

    A beat tracker reporting 174 BPM has most likely counted double time on an
    87 BPM song; halving preserves the bar structure, whereas clipping to 140
    would build a grid at a tempo the song never has. A tempo of 0 or below
    cannot be folded and is returned unchanged for the caller to reject.
    """
    if tempo <= 0:
        return float(tempo)
    while tempo > high:
        tempo /= 2.0
    while tempo < low:
        tempo *= 2.0
    return float(tempo)


def create_meter_grid(beats: np.ndarray, tempo: float, sr: int, hop_length: int,
                      frame_duration: int, time_signature: int = 4) -> np.ndarray:
    """Generate a meter grid (in frames) spanning the duration of a song."""
    first_beat_time = librosa.frames_to_time(beats[0], sr=sr, hop_length=hop_length)
    time_duration = librosa.frames_to_time(frame_duration, sr=sr, hop_length=hop_length)
    seconds_per_beat = 60.0 / tempo
    num_beats_forward = int((time_duration - first_beat_time) / seconds_per_beat)
    num_beats_backward = int(first_beat_time / seconds_per_beat) + 1
    beat_times_forward = first_beat_time + np.arange(num_beats_forward) * seconds_per_beat
    beat_times_backward = first_beat_time - np.arange(1, num_beats_backward) * seconds_per_beat
    beat_grid = np.concatenate((np.array([0.0]), beat_times_backward[::-1], beat_times_forward))
    meter_indices = np.arange(0, len(beat_grid), time_signature)
    meter_grid = beat_grid[meter_indices]
    if meter_grid[0] != 0.0:
        meter_grid = np.insert(meter_grid, 0, 0.0)
    meter_grid = librosa.time_to_frames(meter_grid, sr=sr, hop_length=hop_length)
    if meter_grid[-1] != frame_duration:
        meter_grid = np.append(meter_grid, frame_duration)
    return meter_grid


def generate_and_align_labels(df: pd.DataFrame, sr: int, hop_length: int, n_frames: int,
                              meter_grid_frames: np.ndarray) -> np.ndarray:
    """Create a binary label sequence for 'chorus' in a song and align it to a meter grid."""
    binary_label_sequence = np.zeros(n_frames, dtype=int)
    chorus_rows = df[df['label'] == 'chorus']
    start_frames = librosa.time_to_frames(chorus_rows['start_time'].to_numpy(), sr=sr, hop_length=hop_length)
    end_frames = librosa.time_to_frames(chorus_rows['end_time'].to_numpy(), sr=sr, hop_length=hop_length)
    for start_frame, end_frame in zip(start_frames, end_frames):
        binary_label_sequence[start_frame:end_frame] = 1
    aligned_labels = [
        int(np.mean(binary_label_sequence[s:e]) >= 0.25)  # Meter is chorus if >=25% of its frames are
        for s, e in zip(meter_grid_frames[:-1], meter_grid_frames[1:])
    ]
    return np.array(aligned_labels)


def segment_data_meters(data: np.ndarray, meter_grid_frames: np.ndarray) -> List[np.ndarray]:
    """Divide song data into segments based on meter grid frames."""
    return [data[s:e] for s, e in zip(meter_grid_frames[:-1], meter_grid_frames[1:])]


def positional_encoding(position: int, d_model: int) -> np.ndarray:
    """Generate a positional encoding for a given position and model dimension."""
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


def extract_combined_features(y: np.ndarray, y_harm: np.ndarray, y_perc: np.ndarray,
                              sr: int, hop_length: int,
                              nmf_device: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extract, decompose, standardize, and weight all features for one song.

    Args:
        nmf_device: When set (e.g. "cuda"), run the four NMF fits with the
            torch implementation on that device instead of sklearn's CPU
            solver. Same init, loss, and component sorting.

    Returns:
        Tuple of (combined_features [n_frames, 15], onset_env)
    """
    if nmf_device:
        from pytorch_core.nmf_torch import decompose as _nmf
    else:
        def _nmf(S, n_components, sort=True, device=None):
            return librosa.decompose.decompose(S, n_components=n_components, sort=sort)

    S = np.abs(librosa.stft(y, hop_length=hop_length))
    rms = librosa.feature.rms(S=S).astype(np.float32)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_acts = _nmf(mel, 3, sort=True, device=nmf_device)[1].astype(np.float32)

    chromagram = calculate_ki_chroma(y_harm, sr, hop_length).astype(np.float32)
    chroma_acts = _nmf(chromagram, 4, sort=True, device=nmf_device)[1].astype(np.float32)

    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)
    tempogram = np.clip(librosa.feature.tempogram(onset_envelope=onset_env,
                                                  sr=sr, hop_length=hop_length), 0, None)
    tempogram_acts = _nmf(tempogram, 3, sort=True, device=nmf_device)[1].astype(np.float32)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    mfccs += abs(np.min(mfccs))
    mfcc_acts = _nmf(mfccs, 4, sort=True, device=nmf_device)[1].astype(np.float32)

    features = [rms, mel_acts, chroma_acts, tempogram_acts, mfcc_acts]
    feature_names = ['rms', 'mel_acts', 'chroma_acts', 'tempogram_acts', 'mfcc_acts']

    dims = {name: feature.shape[0] for feature, name in zip(features, feature_names)}
    total_inv_dim = sum(1.0 / dim for dim in dims.values())
    weights = {name: 1.0 / (dims[name] * total_inv_dim) for name in feature_names}

    standardized_weighted_features = [StandardScaler().fit_transform(feature.T).T * weights[name]
                                      for feature, name in zip(features, feature_names)]

    combined_features = np.concatenate(standardized_weighted_features, axis=0).T.astype(np.float32)
    return combined_features, onset_env


def process_song(song_id, song_df: pd.DataFrame, audio_path: str,
                 segments_dir: str, labels_dir: str,
                 grid_source: str = "librosa", device: str = "cpu",
                 nmf_device: str = None,
                 tempo_source: str = "dataset") -> Tuple[int, int]:
    """Process a single song and write its segment/label pickles.

    Args:
        grid_source: "librosa" builds the meter grid by extrapolating a single
            tempo over the song (original behavior); "beat_this" builds it from
            downbeats tracked by Beat This!, which handles metrical irregularity
            without a 4/4 prior.
        device: Torch device for the Beat This! tracker when grid_source is
            "beat_this".
        nmf_device: Torch device for the NMF fits; None keeps sklearn on CPU.
        tempo_source: Where the extrapolated grid's tempo comes from.
            "dataset" reads the per-song Spotify tempo and time signature from
            the CSV (original behavior); "estimated" uses the tempo measured
            from the audio and assumes 4/4, so no outside metadata is used.
            Only affects the librosa grid.

    Returns:
        Tuple of (n_meters, max_frames_per_meter) for dataset statistics.
    """
    y, _ = librosa.load(audio_path, sr=TARGET_SR)
    y_harm, y_perc = librosa.effects.hpss(y)

    combined_features, onset_env = extract_combined_features(y, y_harm, y_perc, TARGET_SR, HOP_LENGTH,
                                                             nmf_device=nmf_device)

    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=TARGET_SR, hop_length=HOP_LENGTH)
    tempo = float(np.atleast_1d(tempo)[0])

    if tempo_source == "estimated":
        # No outside metadata: tempo measured from the audio, 4/4 assumed.
        bpm = fold_tempo(tempo)
        time_signature = 4
    else:
        # Tempo and time signature come from the dataset CSV (Spotify metadata),
        # with the librosa estimate as fallback — matching the original training data.
        bpm = song_df['sp_tempo'].fillna(tempo).replace(0, tempo).clip(lower=70, upper=140).values[0]
        time_signature = int(song_df['sp_time_signature'].fillna(4).replace(0, 4).values[0])

    if grid_source == "beat_this":
        from pytorch_core.downbeats import create_downbeat_meter_grid, track_downbeats
        _, downbeats = track_downbeats(audio_path, device=device)
        if len(downbeats) < 2:
            raise ValueError("beat_this returned <2 downbeats")
        meter_grid = create_downbeat_meter_grid(
            downbeats, len(combined_features), TARGET_SR, HOP_LENGTH)
    else:
        meter_grid = create_meter_grid(beats, bpm, TARGET_SR, HOP_LENGTH,
                                       len(combined_features), time_signature)
    aligned_labels = generate_and_align_labels(song_df, sr=TARGET_SR, hop_length=HOP_LENGTH,
                                               n_frames=len(combined_features),
                                               meter_grid_frames=meter_grid).astype(np.int32)

    meter_segments = segment_data_meters(combined_features, meter_grid)
    meter_segments = [segment.astype(np.float32) for segment in meter_segments]
    encoded_segments = apply_hierarchical_positional_encoding(meter_segments)
    encoded_segments = [segment.astype(np.float32) for segment in encoded_segments]

    assert len(encoded_segments) == len(aligned_labels), (
        f"Song {song_id}: {len(encoded_segments)} segments vs {len(aligned_labels)} labels")

    with open(os.path.join(segments_dir, f"{song_id}_data.pkl"), "wb") as f:
        pickle.dump(encoded_segments, f)
    with open(os.path.join(labels_dir, f"{song_id}_labels.pkl"), "wb") as f:
        pickle.dump(aligned_labels, f)

    max_frames = max(segment.shape[0] for segment in encoded_segments)
    return len(encoded_segments), max_frames


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio into training segments and labels")
    parser.add_argument("--csv", type=str, default="data/clean_labeled.csv",
                        help="Path to labeled dataset CSV")
    parser.add_argument("--audio-dir", type=str, default="data/audio/processed",
                        help="Directory with processed audio files ({SongID}.mp3)")
    parser.add_argument("--segments-dir", type=str, default="data/segments_V2",
                        help="Output directory for segment pickles")
    parser.add_argument("--labels-dir", type=str, default="data/labels_V2",
                        help="Output directory for label pickles")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N songs (for smoke testing)")
    parser.add_argument("--song-ids", type=str, nargs="*", default=None,
                        help="Only process these song IDs")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess songs whose output pickles already exist")
    parser.add_argument("--grid-source", choices=["librosa", "beat_this"], default="librosa",
                        help="Meter grid source: librosa tempo extrapolation or Beat This! downbeats")
    parser.add_argument("--device", default="cpu",
                        help="Torch device for the Beat This! tracker (beat_this grid only)")
    parser.add_argument("--nmf-device", default=None,
                        help="Torch device for NMF fits (e.g. cuda); default keeps sklearn on CPU")
    parser.add_argument("--tempo-source", choices=["dataset", "estimated"], default="dataset",
                        help="Tempo for the extrapolated grid: 'dataset' reads the per-song "
                             "Spotify tempo from the CSV (original behavior); 'estimated' "
                             "measures it from the audio and assumes 4/4, using no outside "
                             "metadata. Only affects --grid-source librosa.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.segments_dir, exist_ok=True)
    os.makedirs(args.labels_dir, exist_ok=True)

    song_ids = list(df['SongID'].unique())
    if args.song_ids:
        wanted = {type(song_ids[0])(s) for s in args.song_ids}
        song_ids = [s for s in song_ids if s in wanted]
    if args.limit:
        song_ids = song_ids[:args.limit]

    processed, skipped, failed = 0, 0, []
    max_meters_seen, max_frames_seen = 0, 0

    for song_id in tqdm(song_ids, desc="Preprocessing"):
        segments_path = os.path.join(args.segments_dir, f"{song_id}_data.pkl")
        labels_path = os.path.join(args.labels_dir, f"{song_id}_labels.pkl")
        if not args.overwrite and os.path.exists(segments_path) and os.path.exists(labels_path):
            skipped += 1
            continue

        audio_path = os.path.join(args.audio_dir, f"{song_id}.mp3")
        if not os.path.exists(audio_path):
            failed.append((song_id, "audio file not found"))
            continue

        try:
            song_df = df.loc[df['SongID'] == song_id]
            n_meters, max_frames = process_song(song_id, song_df, audio_path,
                                                args.segments_dir, args.labels_dir,
                                                grid_source=args.grid_source, device=args.device,
                                                nmf_device=args.nmf_device,
                                                tempo_source=args.tempo_source)
            max_meters_seen = max(max_meters_seen, n_meters)
            max_frames_seen = max(max_frames_seen, max_frames)
            processed += 1
        except Exception as e:
            failed.append((song_id, f"{type(e).__name__}: {e}"))

    print(f"\nProcessed: {processed}, skipped (already done): {skipped}, failed: {len(failed)}")
    if processed:
        print(f"Max meters per song: {max_meters_seen}, max frames per meter: {max_frames_seen}")
    for song_id, reason in failed:
        print(f"  FAILED {song_id}: {reason}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
