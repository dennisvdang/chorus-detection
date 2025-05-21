"""
PyTorch dataset implementation for chorus detection.
"""

import os
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import librosa


class ChorusDataset(Dataset):
    """
    PyTorch Dataset for chorus detection.
    Handles loading and processing of audio data with padding.
    """
    
    def __init__(
        self,
        song_ids: List[str],
        segments_dir: str,
        labels_dir: str,
        config: Dict[str, Any],
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset with song IDs and directories.
        
        Args:
            song_ids: List of song IDs to include in the dataset
            segments_dir: Directory with pickled segments
            labels_dir: Directory with pickled labels
            config: Configuration dictionary
            transform: Optional transform to apply to samples
        """
        self.song_ids = song_ids
        self.segments_dir = segments_dir
        self.labels_dir = labels_dir
        self.config = config
        self.transform = transform
        
        # Cached data
        self.max_frames_per_meter = config["data"]["max_frames"]
        self.max_meters = config["data"]["max_meters"]
        self.n_features = config["data"]["n_features"]
        
        # Pre-load data
        self.data = self._load_data()
        
    def _load_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load and preprocess all song data and labels."""
        data = {}
        
        for song_id in self.song_ids:
            # Load segments and labels
            segments_path = os.path.join(self.segments_dir, f"{song_id}_data.pkl")
            labels_path = os.path.join(self.labels_dir, f"{song_id}_labels.pkl")
            
            try:
                with open(segments_path, "rb") as f:
                    segments = pickle.load(f)
                with open(labels_path, "rb") as f:
                    labels = pickle.load(f)
                    
                # Pad segments
                padded_segments = self._pad_segments(segments)
                padded_labels = self._pad_labels(labels)
                
                data[song_id] = (padded_segments, padded_labels)
            except Exception as e:
                print(f"Error loading data for song {song_id}: {e}")
                
        return data
    
    def _pad_segments(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Pad segments to uniform dimensions.
        
        Args:
            segments: List of segment arrays of shape [n_frames, n_features]
            
        Returns:
            Padded array of shape [max_meters, max_frames, n_features]
        """
        # Pad frames in each meter
        padded_meters = []
        for meter in segments:
            if len(meter) > self.max_frames_per_meter:
                # Truncate if too long
                padded_meter = meter[:self.max_frames_per_meter]
            else:
                # Pad if too short
                pad_width = ((0, self.max_frames_per_meter - meter.shape[0]), (0, 0))
                padded_meter = np.pad(meter, pad_width, mode='constant')
            padded_meters.append(padded_meter)
            
        # Pad meters
        if len(padded_meters) > self.max_meters:
            # Truncate if too many meters
            padded_meters = padded_meters[:self.max_meters]
        else:
            # Pad with zero meters if too few
            padding_meter = np.zeros((self.max_frames_per_meter, self.n_features))
            padded_meters.extend([padding_meter] * (self.max_meters - len(padded_meters)))
            
        return np.stack(padded_meters)
    
    def _pad_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Pad labels to match the number of meters.
        
        Args:
            labels: Label array of shape [n_meters]
            
        Returns:
            Padded label array of shape [max_meters]
        """
        if len(labels) > self.max_meters:
            # Truncate if too many
            return labels[:self.max_meters]
        else:
            # Pad with -1 (masked values) if too few
            return np.pad(labels, (0, self.max_meters - len(labels)), 
                          mode='constant', constant_values=-1)
    
    def __len__(self) -> int:
        """Return the number of songs in the dataset."""
        return len(self.song_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, labels) as PyTorch tensors
        """
        song_id = self.song_ids[idx]
        features, labels = self.data[song_id]
        
        # Convert to torch tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        
        if self.transform:
            features_tensor = self.transform(features_tensor)
            
        return features_tensor, labels_tensor


def create_data_loaders(
    config: Dict[str, Any],
    segments_dir: str = None,
    labels_dir: str = None,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        config: Configuration dictionary
        segments_dir: Directory with segment data (default: data/segments_V2)
        labels_dir: Directory with label data (default: data/labels_V2)
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set default directories if not provided
    segments_dir = segments_dir or "data/segments_V2"
    labels_dir = labels_dir or "data/labels_V2"
    
    # Get all song IDs
    song_ids = []
    for filename in os.listdir(segments_dir):
        if filename.endswith("_data.pkl"):
            song_id = filename.split("_")[0]
            # Check if labels also exist
            if os.path.exists(os.path.join(labels_dir, f"{song_id}_labels.pkl")):
                song_ids.append(song_id)
    
    # Shuffle and split
    np.random.seed(random_seed)
    np.random.shuffle(song_ids)
    
    n_train = int(len(song_ids) * train_split)
    n_val = int(len(song_ids) * val_split)
    
    train_ids = song_ids[:n_train]
    val_ids = song_ids[n_train:n_train + n_val]
    test_ids = song_ids[n_train + n_val:]
    
    # Create datasets
    train_dataset = ChorusDataset(train_ids, segments_dir, labels_dir, config)
    val_dataset = ChorusDataset(val_ids, segments_dir, labels_dir, config)
    test_dataset = ChorusDataset(test_ids, segments_dir, labels_dir, config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,  # Adjust based on available CPU cores
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