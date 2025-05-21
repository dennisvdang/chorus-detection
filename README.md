# Automated Chorus Detection [![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/dennisvdang/chorus-detection)

![Chorus Prediction](./images/131.webp)

## Overview

A hierarchical convolutional recurrent neural network designed for musical structure analysis, specifically optimized for detecting choruses in music recordings. The model was initially trained on 332 annotated songs from electronic music genres and achieved an F1 score of 0.864 (Precision: 0.831, Recall: 0.900) on unseen test data. For more details, scroll down to the [Project Technical Summary section](#project-technical-summary).

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

This branch contains a PyTorch implementation of the chorus detection model with a modular architecture that makes it easy to experiment with different model configurations, feature sets, and training parameters.

### Key Features

- **Configuration-Driven:** All model and training parameters are defined in YAML configuration files
- **Modular Architecture:** Clean separation between data processing, model architecture, and training
- **Reproducible Results:** Consistent random seed initialization for reproducible experiments
- **Comprehensive Metrics:** Detailed training metrics including loss, accuracy, precision, recall, and F1 score
- **Visualization:** Automatic plotting of training history and confusion matrices
- **Checkpointing:** Save and resume training from checkpoints

### Project Structure

```
chorus-detection/
│
├── config/                # Configuration files
│   ├── default.yaml         # Default configuration
│   ├── models/              # Model-specific configs
│   └── features/            # Feature extraction configs
│
├── pytorch_core/          # Core PyTorch functionality
│   ├── data/                # Dataset and data processing
│   ├── models/              # Model architectures
│   ├── training/            # Training and evaluation
│   └── utils/               # Utility functions
│
├── scripts/               # Training and inference scripts
│   ├── train.py             # Script for training
│   └── inference.py         # Script for inference
│
├── experiments/           # For tracking experiments
│
├── models/                # Original TensorFlow models
├── core/                  # Original TensorFlow code
│
└── ... (existing directories)
```

### Getting Started with PyTorch Implementation

1. **Setup Environment:**
   ```bash
   conda env create -f environment.yml
   conda activate chorus-detection
   pip install torch torchvision torchaudio
   ```

2. **Train Model:**
   ```bash
   python scripts/train.py --config config/default.yaml --segments_dir data/segments_V2 --labels_dir data/labels_V2
   ```

3. **Run Inference:**
   ```bash
   python scripts/inference.py --audio path/to/audio.mp3 --checkpoint checkpoints/best_model.pt
   ```

### Customizing the Model

You can customize the model by creating new configuration files:

```yaml
# config/models/custom_model.yaml
data:
  sr: 12000
  hop_length: 128
  
model:
  name: "crnn"
  cnn:
    filters: [64, 128, 256]
    kernel_sizes: [5, 3, 3]
  
training:
  batch_size: 16
  epochs: 100
```

Then train with your custom configuration:
```bash
python scripts/train.py --config config/models/custom_model.yaml
```

### Future Work

- Extend to multi-class classification for all song sections (verse, chorus, bridge, etc.)
- Implement feature importance analysis
- Add support for audio augmentation

## Contributing

If you found this project interesting or informative, feel free to star the repository! Issues, pull requests, and feedback are welcome.