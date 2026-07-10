# Automated Chorus Detection [![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/dennisvdang/chorus-detection)

![Chorus Prediction](./images/131.webp)

## Overview

A hierarchical convolutional recurrent neural network designed for detecting choruses in music recordings. The model was initially trained on 332 annotated songs from electronic music genres and achieved an F1 score of 0.864 (Precision: 0.831, Recall: 0.900) on unseen test data. For more details, scroll down to the [Project Technical Summary section](#project-technical-summary).

## Quick Links

- [Try the model on HuggingFace Spaces](https://huggingface.co/spaces/dennisvdang/Chorus-Detection)
- [Labeled training dataset of 332 songs (audio files not included)](data/clean_labeled.csv)
- [Pre-trained model file](models/CRNN/best_model_V3.h5)
- [Model training notebook](notebooks/Automated-Chorus-Detection.ipynb)
- [Music annotation process](docs/Data_Annotation_Guide.pdf)
- [Project PDF writeup](docs/Capstone_Final_Report.pdf)

## Quick Installation

```bash
# Clone repository
git clone https://github.com/dennisvdang/chorus-detection.git
cd chorus-detection

# Set up environment
conda env create -f environment.yml
conda activate chorus-detection
pip install -r requirements.txt

# With conda environment activated
# Run the web-app locally (recommended)
streamlit run web/app.py

# Or run the CLI
python cli/cli_app.py
```

## Project Structure

```
chorus-detection/
│
├── core/                 # Core functionality
│   ├── audio_processor.py   # Audio processing and feature extraction
│   ├── model.py             # Model loading and prediction
│   ├── utils.py             # Utility functions
│   └── visualization.py     # Plotting and visualization
│
├── cli/                  # Command-line interface
│   └── cli_app.py           # CLI application
│
├── web/                  # Web interface
│   └── app.py               # Streamlit web application
│
├── models/               # Pre-trained models
├── input/                # Input audio files
├── output/               # Output files and visualizations
│
├── setup.py              # Package setup
├── requirements.txt      # Package requirements
├── Dockerfile            # Docker configuration
└── docker-compose.yml    # Docker Compose configuration
```

## Project Technical Summary

### Data

The dataset consists of 332 manually labeled songs, predominantly from electronic music genres. Data preparation involved:

1. **Audio preprocessing**: Formatting songs uniformly, processing at a consistent sampling rate, trimming silence, and extracting metadata using Spotify's API. [Link to preprocessing notebook](notebooks/Preprocessing.ipynb)

2. **Manual Chorus Labeling**: Labeling the start and end timestamps of choruses following a set of guidelines. More details on the annotation process can be found in the [Annotation Guide pdf.](docs/Data_Annotation_Guide.pdf)

### Model Preprocessing

- Features such as Root Mean Squared energy, key-invariant chromagrams, Melspectrograms, MFCCs, and tempograms were extracted. These features were decomposed using Non-negative Matrix Factorization using an optimal number of components derived in our exploratory analysis.

- Songs were segmented into timesteps based on musical meters, with positional and grid encoding applied to every audio frame and meter, respectively. Songs and labels were uniformly padded and split into train/validation/test sets, processed into batch sizes of 32 using a custom generator.

Below are examples of audio feature visualizations of a song with 3 choruses (highlighted in green). The gridlines represent the musical meters, which are used to divide the song into segments; these segments then serve as the timesteps for the CRNN input.

![hspss](./images/hpss.png)
![rms_beat_synced](./images/rms_beat_synced.png)
![chromagram](./images/chromagram_stacked.png)
![tempogram](./images/tempogram.png)

### Model Architecture

The model employs a two-tier architecture that respects the heirarchical structure of music (frames → meters → song)

**Input Features:**

The model receives as input a song's feature vector (NMF-activated features derived from RMS, mel spectrogram, chromagram, tempogram, and MFCCs). These are computed per frame and grouped by musical meter.

**CNN Layers:**

The CNN layers (3 Conv1D + MaxPooling1D layers) apply a series of learnable filters to the input features, sliding across the time (frame) dimension within each meter segment, and outputs a single feature vector (embedding) that summarizes the temporal information found within that meter. Note that there is no information shared *between* meters at this stage; each meter's frames are processed in isolation.

**LSTM Layer:**

After the CNN layers, the sequence of meter embeddings that make up the input song is passed to a bidirectional LSTM, allowing the model to capture both past and future context across the song's structure. The LSTM outputs a sequence of hidden states, one for each meter, which are then used for final classification.

**Output Layer:**

A TimeDistributed dense layer with a sigmoid activation is applied to the LSTM outputs, producing a probability for each meter indicating the likelihood that it corresponds to a chorus section. The model is trained using a custom binary cross-entropy loss that masks out padded values, allowing the model to learn from variable-length songs.

``` python
def create_crnn_model(max_frames_per_meter, max_meters, n_features):
    """
    Args:
    max_frames_per_meter (int): Maximum number of frames per meter.
    max_meters (int): Maximum number of meters.
    n_features (int): Number of features per frame.
    """
    frame_input = layers.Input(shape=(max_frames_per_meter, n_features))
    conv1 = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(frame_input)
    pool1 = layers.MaxPooling1D(pool_size=2, padding='same')(conv1)
    conv2 = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling1D(pool_size=2, padding='same')(conv2)
    conv3 = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling1D(pool_size=2, padding='same')(conv3)
    frame_features = layers.Flatten()(pool3)
    frame_feature_model = Model(inputs=frame_input, outputs=frame_features)

    meter_input = layers.Input(shape=(max_meters, max_frames_per_meter, n_features))
    time_distributed = layers.TimeDistributed(frame_feature_model)(meter_input)
    masking_layer = layers.Masking(mask_value=0.0)(time_distributed)
    lstm_out = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(masking_layer)
    output = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(lstm_out)
    model = Model(inputs=meter_input, outputs=output)
    model.compile(optimizer='adam', loss=custom_binary_crossentropy, metrics=[custom_accuracy])
    return model
```

### Training

- Custom loss and accuracy functions handle padded values
- Callbacks to save best model based on minimal validation loss, reduce learning rate on plateau, and early stopping
- Trained for 50 epochs (stopped early after 18 epochs). Training/Validation Loss and Accuracy plotted below:
![Training History](images/training_history.png)

### Results

The model achieved strong results on the held-out test set as shown in the summary table. Visualizations of the predictions on sample test songs are also provided and can be found in the [test_predictions folder](images/test_predictions).

| Metric         | Score  |
|----------------|--------|
| Loss           | 0.278  |
| Accuracy       | 0.891  |
| Precision      | 0.831  |
| Recall         | 0.900  |
| F1 Score       | 0.864  |

![Confusion Matrix](./images/confusion_matrix.png)

## Works in progress

- Additional training data for other musical segments (e.g. intro, pre-chorus, bridge, verse)
- Music data labeling interface for contributions

## PyTorch Implementation (pytorch-modular branch)

This branch contains a PyTorch port of the chorus detection model with a modular
architecture that makes it easy to experiment with different model
configurations, feature sets, and training parameters. The port reproduces the
original TensorFlow model's performance (test F1 0.862 vs 0.864); see
[docs/pytorch_results.md](docs/pytorch_results.md) for the full comparison.

### Key Features

- **Configuration-Driven:** Model and training parameters are defined in a YAML config file
- **Modular Architecture:** Clean separation between data processing, model architecture, and training
- **Reproducible Results:** Consistent random seed initialization for reproducible experiments
- **Comprehensive Metrics:** Loss, accuracy, precision, recall, and F1 score
- **Checkpointing:** Save and resume training from checkpoints

### Project Structure

```
chorus-detection/
│
├── config/
│   └── default.yaml         # Model and training configuration
│
├── pytorch_core/            # Core PyTorch functionality
│   ├── data/                # Dataset classes
│   ├── models/              # Model architectures (crnn.py + spectrogram experiment)
│   ├── training/            # Trainer with evaluation and checkpointing
│   └── audio_processor.py   # Feature extraction and meter segmentation
│
├── scripts/
│   ├── preprocess.py        # Audio -> training segments and labels
│   ├── train.py             # Training entry point
│   └── inference.py         # Chorus detection on a new audio file
│
├── tests/                   # pytest suite
│
├── models/                  # Original TensorFlow model (models/CRNN/)
└── core/                    # Original TensorFlow inference code
```

### Getting Started with the PyTorch Implementation

1. **Set up the environment:**
   ```bash
   conda env create -f environment.yml
   conda activate chorus-detection
   pip install -r requirements.txt
   ```
   For CUDA training, install the matching PyTorch build, e.g.:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Preprocess the dataset** (audio files in `data/audio/processed/{SongID}.mp3`
   plus `data/clean_labeled.csv`). This writes per-song segment and label
   pickles to `data/segments_V2/` and `data/labels_V2/`:
   ```bash
   python scripts/preprocess.py
   # smoke-test on a few songs first:
   python scripts/preprocess.py --limit 3
   ```

3. **Train the model:**
   ```bash
   python scripts/train.py --config config/default.yaml --checkpoint_dir models/pytorch
   # on an 8 GB GPU, reduce the batch size:
   python scripts/train.py --checkpoint_dir models/pytorch --batch_size 16 --device cuda
   ```

4. **Run inference on a new song:**
   ```bash
   python scripts/inference.py --audio path/to/audio.mp3 \
       --checkpoint models/pytorch/best_model.pt --output prediction.png
   ```

### Running the Tests

```bash
pip install pytest
pytest tests/ -m "not slow"   # fast unit tests
pytest tests/                 # includes the audio preprocessing integration test
```

### Spectrogram Model (experimental)

`pytorch_core/models/spectrogram_model.py` and the accompanying
`scripts/train_spectrogram_model_v2.py` are an experimental 2D-CNN variant that
learns directly from spectrograms rather than NMF-decomposed features. It is a
work in progress and is not the validated CRNN port described above.

### Future Work

- Extend to multi-class classification for all song sections (verse, chorus, bridge, etc.)
- Implement feature importance analysis
- Add support for audio augmentation

## Contributing

If you found this project interesting or informative, feel free to star the repository! Issues, pull requests, and feedback are welcome.