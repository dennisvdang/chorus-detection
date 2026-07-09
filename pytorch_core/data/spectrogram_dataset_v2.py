"""
PyTorch dataset for spectrogram-based chorus detection.
Uses the modular audio processor from pytorch_core/audio_processor.py.
"""

import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import librosa

from pytorch_core.audio_processor import AudioFeature, process_audio


class SpectrogramDataset(Dataset):
    """
    Dataset for spectrogram-based chorus detection.
    Uses AudioFeature to extract spectrograms.
    """
    
    def __init__(
        self,
        audio_files: List[str],
        labels_dir: str,
        config: Dict[str, Any],
        precomputed_dir: Optional[str] = None,
        cache_data: bool = True,
        transform: Optional[callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            audio_files: List of audio file paths
            labels_dir: Directory containing label files
            config: Configuration dictionary
            precomputed_dir: Directory for cached preprocessed data
            cache_data: Whether to cache data in memory
            transform: Optional transform to apply to samples
        """
        self.audio_files = audio_files
        self.labels_dir = labels_dir
        self.config = config
        self.precomputed_dir = precomputed_dir
        self.cache_data = cache_data
        self.transform = transform
        
        # Extract song IDs from file paths
        self.song_ids = [os.path.splitext(os.path.basename(file))[0] for file in audio_files]
        
        # Audio processing parameters
        self.sr = config["audio"]["sr"]
        self.hop_length = config["audio"]["hop_length"]
        self.n_fft = config["audio"]["n_fft"]
        self.max_frames = config["data"]["max_frames"]
        self.max_meters = config["data"]["max_meters"]
        
        # Cache for loaded data
        self.data_cache = {}
        
        # Pre-load data if caching is enabled
        if self.cache_data:
            self._preload_data()
            
    def _preload_data(self):
        """Preload and preprocess all data if caching is enabled."""
        for idx, audio_file in enumerate(self.audio_files):
            print(f"Preprocessing {idx+1}/{len(self.audio_files)}: {os.path.basename(audio_file)}")
            self._load_item(idx)
    
    def _load_item(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess a single item.
        
        Args:
            idx: Index of the item to load
            
        Returns:
            Tuple of (features, labels)
        """
        song_id = self.song_ids[idx]
        
        # Check if already in cache
        if song_id in self.data_cache:
            return self.data_cache[song_id]
        
        # Check if precomputed data exists
        if self.precomputed_dir:
            features_path = os.path.join(self.precomputed_dir, f"{song_id}_features.pkl")
            if os.path.exists(features_path):
                try:
                    with open(features_path, "rb") as f:
                        padded_song = pickle.load(f)
                        
                    # Load labels
                    labels_path = os.path.join(self.labels_dir, f"{song_id}_labels.pkl")
                    with open(labels_path, "rb") as f:
                        labels = pickle.load(f)
                    
                    padded_labels = self._pad_labels(labels)
                    
                    if self.cache_data:
                        self.data_cache[song_id] = (padded_song, padded_labels)
                        
                    return padded_song, padded_labels
                    
                except Exception as e:
                    print(f"Error loading precomputed data for {song_id}: {e}")
        
        # Process audio if not precomputed
        try:
            audio_file = self.audio_files[idx]
            
            # Process audio using AudioFeature
            padded_song, _ = process_audio(
                audio_file, 
                sr=self.sr, 
                hop_length=self.hop_length,
                extract_spectrogram_only=True
            )
            
            # Remove batch dimension
            padded_song = padded_song[0]
            
            # Load labels
            labels_path = os.path.join(self.labels_dir, f"{song_id}_labels.pkl")
            with open(labels_path, "rb") as f:
                labels = pickle.load(f)
            
            padded_labels = self._pad_labels(labels)
            
            # Save precomputed features if directory specified
            if self.precomputed_dir:
                os.makedirs(self.precomputed_dir, exist_ok=True)
                with open(features_path, "wb") as f:
                    pickle.dump(padded_song, f)
            
            # Cache result
            if self.cache_data:
                self.data_cache[song_id] = (padded_song, padded_labels)
                
            return padded_song, padded_labels
            
        except Exception as e:
            print(f"Error processing {song_id}: {e}")
            # Return a zero tensor as fallback
            padded_song = np.zeros((self.max_meters, self.max_frames, self.n_fft // 2 + 1))
            padded_labels = np.ones((self.max_meters, 1)) * -1  # All masked
            
            if self.cache_data:
                self.data_cache[song_id] = (padded_song, padded_labels)
                
            return padded_song, padded_labels
    
    def _pad_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Pad labels to match model expectations.
        
        Args:
            labels: Raw labels array
            
        Returns:
            Padded labels array
        """
        if len(labels) > self.max_meters:
            # Truncate if too many
            return labels[:self.max_meters].reshape(-1, 1)
        else:
            # Pad with -1 (masked values) if too few
            padded = np.pad(
                labels, 
                (0, self.max_meters - len(labels)), 
                mode='constant', 
                constant_values=-1
            )
            return padded.reshape(-1, 1)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, labels) as PyTorch tensors
        """
        # Load data
        features, labels = self._load_item(idx)
        
        # Convert to torch tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # Apply transform if specified
        if self.transform:
            features_tensor = self.transform(features_tensor)
            
        return features_tensor, labels_tensor


def create_data_loaders(
    config: Dict[str, Any],
    audio_dir: str,
    labels_dir: str,
    precomputed_dir: Optional[str] = None,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        config: Configuration dictionary
        audio_dir: Directory containing audio files
        labels_dir: Directory containing label files
        precomputed_dir: Directory for precomputed features
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Find all audio files with corresponding labels
    audio_files = []
    for filename in os.listdir(audio_dir):
        if filename.endswith((".wav", ".mp3", ".flac")):
            song_id = os.path.splitext(filename)[0]
            label_path = os.path.join(labels_dir, f"{song_id}_labels.pkl")
            
            if os.path.exists(label_path):
                audio_files.append(os.path.join(audio_dir, filename))
    
    # Shuffle and split the dataset
    np.random.seed(random_seed)
    np.random.shuffle(audio_files)
    
    n_samples = len(audio_files)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    
    train_files = audio_files[:n_train]
    val_files = audio_files[n_train:n_train + n_val]
    test_files = audio_files[n_train + n_val:]
    
    print(f"Dataset split - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Create datasets
    train_dataset = SpectrogramDataset(
        train_files, labels_dir, config, 
        precomputed_dir=precomputed_dir, 
        cache_data=True
    )
    
    val_dataset = SpectrogramDataset(
        val_files, labels_dir, config,
        precomputed_dir=precomputed_dir,
        cache_data=True
    )
    
    test_dataset = SpectrogramDataset(
        test_files, labels_dir, config,
        precomputed_dir=precomputed_dir,
        cache_data=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader 