# Automated Chorus Detection

![Demo](images/chorus-detection-preview.gif)

## Project Overview

This project applies machine learning techniques from Digital Signal Processing, Music Information Retrieval, and Data Science to predict chorus locations in songs. The goal is to develop an accurate and efficient automated chorus detection model that can enhance user experience for a music streaming company's new product feature which plays reels of song choruses.

A Convolutional Recurrent Neural Network (CRNN) model is used to make binary predictions of whether a meter in a song belongs to a chorus or not. The CRNN performance on a holdout test set of 50 songs is summarized in the table below.

| Metric         | Score  |
|----------------|--------|
| Loss           | 0.234  |
| Accuracy       | 0.899  |
| Precision      | 0.884  |
| Recall         | 0.869  |
| F1 Score       | 0.876  |

## CLI Setup and Usage

This project repository includes a command-line tool that allows users to input a YouTube link and utilize the pre-trained CRNN model to detect the chorus sections of the corresponding audio file.

### Setup for Non-Conda Users

1. **Clone the repository or download the project files.**
2. **Navigate to the project directory in PowerShell or Terminal:** `cd chorus-detection`
3. **Create a virtual environment:** `make create_venv` or `python -m venv venv`
4. **Activate the virtual environment:**
   - On Windows: `venv\Scripts\activate` 
   - On Unix/Linux/macOS: `source venv/bin/activate`
5. **Install dependencies:** `pip install -r requirements.txt`
6. **Run the CLI:**
   - `make run_venv URL="{youtube url}"` Replace `{youtube url}` with the actual YouTube URL wrapped in quotes.
   - Example: `make run_venv URL="https://www.youtube.com/watch?v=dQw4w9WgXcQ"`

### Conda Users

1. **Clone the repository or download the project files.**
2. **Navigate to the project directory in cmd:** `cd chorus-detection`
3. **Create the conda environment:** `conda env create -f environment.yml`
4. **Activate the Conda environment:** `conda activate chorus-detection`
5. **Run the script:** 
   - `python src/chorus_finder.py {youtube url}` Replace `{youtube url}` with the actual YouTube URL (don't need to wrap in quotes).
   - Example: `python src/chorus_finder.py https://www.youtube.com/watch?v=dQw4w9WgXcQ`

### CLI Arguments

The `chorus_finder.py` script accepts the following command-line arguments:

- `url`: YouTube URL of the audio file (required)
- `--model_path`: Path to the pretrained model (default: `../models/CRNN/best_model.h5`)
- `--verbose`: Enable verbose output (default: `True`)
- `--output_plot`: Path to save the plot (default: `output/plots/`)

## Project Documentation and Resources

- **Final Project Write-up**: For a more in-depth analysis, see the [Final Project Write-up](docs/Capstone_Final_Report.pdf).

- **Data Annotation**: Details on the manual song labeling process are in the [Mixin Data Annotation Guide](docs/Mixin%20Data%20Annotation%20Guide.pdf).

- **Model Metrics**: Key performance metrics for the CRNN model are summarized in the [model_metrics.csv](docs/model_metrics.csv).

- **Notebooks**:
  - [Preprocessing](notebooks/Preprocessing.ipynb): Audio formatting, trimming, metadata extraction
  - [EDA](notebooks/Mixin_EDA.ipynb): Exploratory analysis and visualizations of audio features
  - [Modeling](notebooks/Automated-Chorus-Detection.ipynb): CRNN model preprocessing, architecture, training, evaluation

## Data

The dataset consists of 332 manually labeled songs, predominantly from electronic music genres. The data wrangling steps included:

- **Audio preprocessing** - formatting songs uniformly, processing at consistent sampling rate, trimming silence, extracting metadata using Spotify's API. [See Jupyter Notebook](notebooks/Preprocessing.ipynb)
- **Manual chorus labeling** - label start/end timestamp of choruses, skipping ambiguous songs. More details on the annotation process can be found in the [Mixin Annotation Guide](docs/Mixin%20Data%20Annotation%20Guide.pdf)

## Exploratory Data Analysis

The EDA aimed to uncover insights and patterns to inform model development. We did so through developing an in-depth data profile of the labels, songs, artists, and genres found in the dataset. The validity of labels was also assessed during this stage. Audio features such as spectrogram, tempogram, chromagram, RMS energy, MFCCs, etc. were extracted and visualized to better understand the characteristics of the audio signals that are relevant to the task of chorus detection. More details on the EDA process can be found in the [EDA Jupyter Notebook](notebooks/Mixin_EDA.ipynb).

Below are feature visualizations of a song with 3 choruses (highlighted in green) with the meter segmentation overlayed.

![hspss](./images/hpss.png)
![rms_beat_synced](./images/rms_beat_synced.png)
![chromagram](./images/chromagram_stacked.png)
![tempogram](./images/tempogram.png)

## Model Preprocessing

- Key features extracted include Mel spectrogram, chromagram, MFCCs, RMS energy, and tempogram. Features are decomposed using Non-negative Matrix Factorization.
- Songs are segmented and aligned into musical meters. This introduces an inductive bias to help the CRNN learn more relevant features and patterns.
- A novel positional encoding scheme is applied to the meter-segmented data to provide temporal context to the CRNN at both the meter and frame levels.
- Songs and labels are uniformly padded and split into train/validation/test sets. Data is processed into batches using a custom generator.

## Modeling

The CRNN model consists of:

- Three 1D convolutional layers with ReLU and max-pooling to extract local patterns 
- A Bidirectional LSTM layer to model long-range temporal dependencies
- A TimeDistributed Dense output layer with sigmoid activation for meter-wise predictions

``` python
def create_crnn_model(max_frames_per_measure, max_measures, feature_per_frame):
    """
    Args:
    max_frames_per_measure (int): Maximum number of frames per measure.
    max_measures (int): Maximum number of measures.
    feature_per_frame (int): Number of features per frame.
    """
    frame_input = layers.Input(shape=(max_frames_per_measure, feature_per_frame))
    conv1 = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(frame_input)
    pool1 = layers.MaxPooling1D(pool_size=2, padding='same')(conv1)
    conv2 = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling1D(pool_size=2, padding='same')(conv2)
    conv3 = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling1D(pool_size=2, padding='same')(conv3)
    frame_features = layers.Flatten()(pool3)
    frame_feature_model = Model(inputs=frame_input, outputs=frame_features)

    measure_input = layers.Input(shape=(max_measures, max_frames_per_measure, feature_per_frame))
    time_distributed = layers.TimeDistributed(frame_feature_model)(measure_input)
    masking_layer = layers.Masking(mask_value=0.0)(time_distributed)
    lstm_out = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(masking_layer)
    output = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(lstm_out)
    model = Model(inputs=measure_input, outputs=output)
    model.compile(optimizer='adam', loss=custom_binary_crossentropy, metrics=[custom_accuracy])
    return model
```

### Training

- Custom loss and accuracy functions handle padded values
- Callbacks save best model, reduce LR on plateau, and enable early stopping
- Trained for 20 epochs with plots of loss and accuracy tracked
![Training History](./images/training_history_model1.png)

## Results

The model achieved strong results on the held-out test set as shown in the summary table. Visualizations of the predictions on sample test songs are also provided and can be found in the [test_predictions folder](images/test_predictions).

| Metric         | Score  |
|----------------|--------|
| Loss           | 0.234  |
| Accuracy       | 0.899  |
| Precision      | 0.884  |
| Recall         | 0.869  |
| F1 Score       | 0.876  |

![Confusion Matrix](./images/confusion_matrix.png)

## Limitations, Implications, and Future Directions

While the model demonstrates promising results, it's important to note limitations such as its potential biases towards the predominantly electronic music genre in the dataset. Future work could explore the application of semi-supervised learning techniques to leverage unlabeled data, expand the dataset to include a wider variety of genres, and explore alternative architectures or attention mechanisms that could further enhance model performance, generalizeability, and interpretability. More empirical testing is needed to determine whether the hierarchical positional encoding and segmentation techniques are effective.