# Automated Chorus Detection

## Introduction

"Automated Chorus Detection" leverages a Convolutional Recurrent Neural Network (CRNN) to predict chorus locations in songs. This project aims to automate chorus detection, potentially enhancing music recommendation systems and can be used to generate "highlights" for music to enhance music discovery processes.

## Objective

The project focuses on automating the identification of choruses within songs to support various applications in the music industry and improve user listening experiences.

## Methodology

### Data Preprocessing

The preprocessing phase involves several key steps:

- **Data Loading**: The dataset, comprising labeled songs, is loaded using Librosa to identify chorus segments.
- **Feature Extraction**: Key features extracted from the audio include:
  - Separation of harmonic and percussive components.
  - Extraction of onset strength, Mel spectrograms, Chromagrams, Tempograms, and MFCCs.
  - Dimensionality reduction on selected features to capture essential aspects.
- **Standardization and Weighting**: Features are standardized and weighted to ensure scale consistency and balanced influence on the model.
- **Temporal Alignment**: Features are aligned with the song's temporal structure through tempo and beat tracking.

### CRNN Architecture

The project employs a CRNN architecture, combining convolutional layers for feature extraction with recurrent layers to capture temporal dependencies, facilitating accurate chorus location predictions.

## Preprocessing Code Overview

```python
# Load the audio file
audio_path = f'../data/audio_files/processed/{song_id}.mp3'
y, _ = librosa.load(audio_path, sr=TARGET_SR)

# Harmonic-percussive source separation
y_harm, y_perc = librosa.effects.hpss(y)

onset_env = librosa.onset.onset_strength(y=y_perc, sr=TARGET_SR, hop_length=HOP_LENGTH)

# Compute RMS energy from spectrogram to give a more accurate representation of energy over time because its frames can be windowed
S = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
rms = librosa.feature.rms(S=S)

# Compute Mel Spectrogram and decompose into 4 components (4 chosen from EDA)
mel = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=128, hop_length=HOP_LENGTH)
mel_acts = librosa.decompose.decompose(mel, n_components=4, sort=True)[1]

# Compute chromagram, make it key invariant, and decompose 
chromagram = librosa.feature.chroma_cqt(y=y_harm, sr=TARGET_SR, hop_length=HOP_LENGTH)
chroma_ki = make_key_invariant(chromagram)
chroma_acts = librosa.decompose.decompose(chroma_ki, n_components=3, sort=True)[1]

# Compute tempogram, ensure non-negative b/c tempograms are finicky, and decompose 
tempogram = np.clip(librosa.feature.tempogram(onset_envelope=onset_env, sr=TARGET_SR, hop_length=HOP_LENGTH), 0, None)
tempogram_acts = librosa.decompose.decompose(tempogram, n_components=3, sort=True)[1]

# Compute MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=13, hop_length=HOP_LENGTH)

# Standardize features, stack, and segment by beats
features = [rms, mel_acts, chroma_ki, chroma_acts, tempogram_acts, mfccs]
total_inv_dim = sum(1.0 / dim for dim in dims.values()) # Calculate the total sum of inverse dimensions to normalize weights
weights = {feature: (1.0 / dims[feature]) / total_inv_dim for feature in dims} # Normalize weights so each feature weighs the same despite dimensionality

# Apply StandardScaler and weights to each feature
standardized_weighted_features = [StandardScaler().fit_transform(feature.T).T * weights[feature_name]
                                    for feature, feature_name in zip(features, dims)]

# Stack/concat features
combined_features = np.vstack(standardized_weighted_features)

# Tempo and beat tracking
tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=TARGET_SR, hop_length=HOP_LENGTH)
beat_times = librosa.frames_to_time(beats, sr=TARGET_SR, hop_length=HOP_LENGTH)
bpm = group['sp_tempo'].fillna(tempo).replace(0, tempo).clip(lower=70, upper=140).values[0]
time_signature = int(group['sp_time_signature'].fillna(4).replace(0, 4).values[0])
beat_interval_in_frames = librosa.time_to_frames(60/bpm, sr=sr)

# Measure grid creation and label alignment
anchor_frame = find_anchor_frame(beats, bpm, sr)
beat_grid, meter_grid = create_beat_grid(beats, anchor_frame, sr, beat_interval_in_frames, time_signature, len(combined_features))
aligned_labels = generate_and_align_labels(group, len(combined_features), meter_grid)
meter_segments = segment_data_measures(combined_features, meter_grid)
