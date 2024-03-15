# Automated Chorus Detection

## Introduction

This project leverages a Convolutional Recurrent Neural Network (CRNN) to predict chorus locations in songs. This project aims to automate chorus detection, potentially enhancing music recommendation systems and potentially music discovery processes where the "highlights" of a song can be identified more efficiently.

## Objective

The project focuses on automating the identification of choruses within songs to support various applications in the music industry and improve user listening experiences.

## Methodology

### Data Preprocessing

- **Data Loading**: The dataset, comprising 332 labeled songs, is loaded using Librosa to identify chorus segments.

- **Feature Extraction**: Key features extracted from the audio include:
  - Separation of harmonic and percussive components.
  - Extraction of onset strength, Mel spectrograms, Chromagrams, Tempograms, and MFCCs.
  - Non-negative Matrix Factorization activations on selected features to capture essential aspects.

- **Standardization and Weighting**: Features are standardized and weighted to ensure scale consistency and balanced influence on the model.

- **Hierarchical Positional Encodings**: To capture the rich structural nuances inherent in musical compositions, we employ a hierarchical positional encoding scheme. This approach adds two layers of positional information:
  1. **Meter-level Encoding**: Embeds the position of each meter within the song, acknowledging the structured progression typical in musical compositions. This allows the model to understand and leverage the macro-structure of songs, such as verses, choruses, and bridges, in their sequential context.
  2. **Frame-level Encoding**: Incorporates the position of each frame within its respective meter, providing fine-grained temporal context. This enables the model to discern the detailed nuances within meters, crucial for capturing the subtle dynamics and rhythms that characterize musical segments.

  The combination of meter-level and frame-level positional encodings (hopefully) equips the model with a deep understanding of both the global structure and the local dynamics of songs, enhancing its ability to predict chorus locations with high precision. Empirical testing is needed to confirm the effectiveness of this approach.

- **Temporal Alignment**: Features are aligned with the song's temporal structure through tempo, time signature (extracted from Spotify API), and beat tracking. This alignment is intuitive, aiming to synchronize the feature data with the musical content to help the model's predictions be temporally coherent with the actual song structure. Empirical testing with and without temporal alignment can confirm its benefits.

### CRNN Architecture

The project employs a CRNN architecture, combining convolutional layers for feature extraction with recurrent layers to capture temporal dependencies, facilitating accurate chorus location predictions.

## Preprocessing Code Overview

```python
TARGET_SR = 12000 # Target sample rate chosen to be 1/4 of the original 48kHz.
HOP_LENGTH = 128  # Hop length for short-time Fourier transform. Hop length of 128 at 12kHz gives a similar frame rate to a hop length of 512 at 48kHz.

# Load the audio file
audio_path = f'../data/audio_files/processed/{song_id}.mp3'
y, _ = librosa.load(audio_path, sr=TARGET_SR)

# Harmonic-percussive source separation
y_harm, y_perc = librosa.effects.hpss(y)

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
onset_env = librosa.onset.onset_strength(y=y_perc, sr=TARGET_SR, hop_length=HOP_LENGTH)
tempogram = np.clip(librosa.feature.tempogram(onset_envelope=onset_env, sr=TARGET_SR, hop_length=HOP_LENGTH), 0, None)
tempogram_acts = librosa.decompose.decompose(tempogram, n_components=3, sort=True)[1]

# Compute MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=13, hop_length=HOP_LENGTH)

# Calculate dimensions, total inverse dimension, and weights
features = [rms, mel_acts, chroma_ki, chroma_acts, tempogram_acts, mfccs]
feature_names = ['rms', 'mel_acts', 'chroma_ki', 'chroma_acts', 'tempogram_acts', 'mfccs']

# Calculate dimensions and weights
dims = {name: feature.shape[0] for feature, name in zip(features, feature_names)}
total_inv_dim = sum(1.0 / dim for dim in dims.values())
weights = {name: 1.0 / (dims[name] * total_inv_dim) for name in feature_names}

# Standardize and apply weights
standardized_weighted_features = [StandardScaler().fit_transform(feature.T).T * weights[name]
                                    for feature, name in zip(features, feature_names)]

# Stack/concat features
combined_features = np.concatenate(standardized_weighted_features, axis=0).T

# Tempo and beat tracking
tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=TARGET_SR, hop_length=HOP_LENGTH)
beat_times = librosa.frames_to_time(beats, sr=TARGET_SR, hop_length=HOP_LENGTH)
data = df.loc[df['SongID'] == song_id]
bpm = data['sp_tempo'].fillna(tempo).replace(0, tempo).clip(lower=70, upper=140).values[0]
time_signature = int(data['sp_time_signature'].fillna(4).replace(0, 4).values[0])
beat_interval_in_frames = librosa.time_to_frames(60/bpm, sr=TARGET_SR, hop_length=HOP_LENGTH)

# Measure grid creation and label alignment
anchor_frame = find_anchor_frame(beats, bpm, TARGET_SR, HOP_LENGTH)
beat_grid, meter_grid = create_beat_grid(beats, anchor_frame, beat_interval_in_frames, time_signature, len(combined_features))
aligned_labels = generate_and_align_labels(data, len(combined_features), meter_grid)
meter_segments = segment_data_measures(combined_features, meter_grid)

# Apply Hierarchical Positional Encoding
encoded_segments = apply_hierarchical_positional_encoding(meter_segments)
```

### Data Padding

Ensuring that our CRNN model receives uniformly structured input is crucial for achieving accurate and reliable chorus detection across a diverse range of songs. Given the inherent variability in song lengths and structures, I employed padding on both the meters and frames. 

#### Frame Padding

- To standardize the length of each meter across all songs and ensure consistent feature analysis by the model, each meter within a song is padded with zeros to match the length of the longest measure found in the dataset. This ensures that every measure has the same number of frames, aligning the temporal resolution across all songs.

#### Meter Padding

- To standardize the number of measures in each song and ensure that the model can process songs of varying lengths without bias, songs with fewer measures than the maximum found in the dataset are padded with measures of zeros. These padding measures are structured to have the same dimensions as real measures, allowing songs of varying structural complexity to be represented in a consistent format for model processing.

#### Label Padding

- Label sequences are padded with a special value (-1) to match the length of the padded song structures. Once masking is applied, this special value indicates to the model that these segments are not part of the original song data and should be ignored in the learning process.

### Data Splitting and Dataset Creation

#### Splitting Data

- **Partitioning**: The dataset, consisting of padded songs and their corresponding labels, is divided into training, validation, and test sets. 
  - **Training Set**: 70% of the data is reserved for training, allowing the model to learn the patterns associated with chorus locations within songs.
  - **Validation Set**: 15% of the data is allocated for validation, used to fine-tune the model's parameters and prevent overfitting.
  - **Test Set**: The remaining 15% serves as the test set, providing an unbiased evaluation of the final model's performance.

#### Dataset Creation

- **Batch Processing**: Data is further processed into batches, facilitating efficient training. Batches are dynamically generated using a custom data generator, ensuring that the model can handle varying song lengths and structures.
- **TensorFlow Datasets**: Utilizing TensorFlow's `Dataset` API, we construct datasets from the generator function for each of the training, validation, and test sets. These datasets are optimized for performance, supporting parallel data processing and prefetching.

### Model Architecture and Custom Functions

The core of our automated chorus detection system is the Convolutional Recurrent Neural Network (CRNN) model, designed to capture both the temporal dynamics and the intricate patterns present in musical compositions. The model architecture and its components are meticulously crafted to address the unique challenges of chorus detection in songs.

#### CRNN Model Architecture

- **Input Layer**: Receives the preprocessed and standardized feature arrays, segmented by measure and frame.
- **Convolutional Layers**: Three convolutional layers, each followed by max pooling, extract hierarchical features from the input data, capturing various aspects of the musical signal.
- **Recurrent Layer**: A bidirectional LSTM layer processes the time-distributed frame features, enabling the model to understand long-term dependencies and temporal patterns in the song data.
- **Output Layer**: A time-distributed dense layer with a sigmoid activation function makes binary predictions for each segment/meter, indicating the presence or absence of a chorus.

#### Custom Loss and Accuracy Functions

To accommodate the unique structure of our dataset, particularly the use of padding to standardize input lengths, we employ custom functions for calculating loss and accuracy:

- **Custom Binary Cross-Entropy Loss**: Modified to ignore padded values (labeled as -1) in the loss calculation, ensuring that the model's learning is focused on meaningful data segments only.
- **Custom Accuracy Metric**: Similarly, this metric disregards padded segments in accuracy calculations, providing a more accurate assessment of the model's performance on relevant song parts.

#### Model Compilation

The model is compiled with the Adam optimizer, utilizing the custom binary cross-entropy function for loss and including the custom accuracy metric for performance evaluation. This setup ensures that the model training process is optimally tuned to the specificities of our chorus detection task.


