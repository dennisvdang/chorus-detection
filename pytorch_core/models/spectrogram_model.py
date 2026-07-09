"""
PyTorch implementation of a Spectrogram-based model for chorus detection.
Uses the modular audio processing from pytorch_core/audio_processor.py.
"""

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from pytorch_core.audio_processor import AudioFeature


class SpectrogramModel(nn.Module):
    """
    Spectrogram-based model for chorus detection.
    
    This model uses raw spectrograms instead of decomposed features:
    1. Extracts spectrograms using AudioFeature
    2. Processes them with a 2D CNN architecture
    3. Integrates across meters for chorus prediction
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SpectrogramModel.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(SpectrogramModel, self).__init__()
        self.config = config
        
        # Data parameters
        self.max_frames = config["data"]["max_frames"]
        self.max_meters = config["data"]["max_meters"]
        self.n_fft = config["audio"]["n_fft"]
        self.n_features = self.n_fft // 2 + 1  # Number of frequency bins
        self.dropout_rate = config["model"]["dropout"]
        
        # CNN layers for processing spectrograms
        cnn_config = config["model"]["cnn"]
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        # Create CNN layers
        in_channels = 1  # Grayscale spectrogram
        channels = cnn_config["channels"]
        kernel_sizes = cnn_config["kernel_sizes"]
        pool_sizes = cnn_config["pool_sizes"]
        
        for i in range(len(channels)):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channels[i],
                    kernel_size=kernel_sizes[i],
                    padding='same'
                )
            )
            self.norm_layers.append(nn.BatchNorm2d(channels[i]))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=pool_sizes[i]))
            in_channels = channels[i]
        
        # Calculate output size after convolutions
        freq_dim, time_dim = self._calculate_output_size()
        cnn_output_size = freq_dim * time_dim * channels[-1]
        
        # Meter-level processing
        self.meter_layer = nn.Sequential(
            nn.Linear(cnn_output_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Output layer
        self.output_layer = nn.Linear(128, 1)
    
    def _calculate_output_size(self) -> Tuple[int, int]:
        """
        Calculate output dimensions after convolutions and pooling.
        
        Returns:
            Tuple of (frequency dimension, time dimension)
        """
        freq_dim = self.n_features
        time_dim = self.max_frames
        
        # Apply pooling calculations
        for pool_layer in self.pool_layers:
            if isinstance(pool_layer.kernel_size, tuple):
                freq_dim //= pool_layer.kernel_size[0]
                time_dim //= pool_layer.kernel_size[1]
            else:
                freq_dim //= pool_layer.kernel_size
                time_dim //= pool_layer.kernel_size
        
        return freq_dim, time_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, max_meters, max_frames, n_features]
            
        Returns:
            Output tensor of shape [batch_size, max_meters, 1]
        """
        batch_size = x.size(0)
        outputs = []
        
        # Process each meter separately
        for meter_idx in range(self.max_meters):
            # Extract the current meter's spectrogram
            meter_spec = x[:, meter_idx, :, :]  # [batch_size, max_frames, n_features]
            
            # Add channel dimension and transpose for CNN
            meter_spec = meter_spec.unsqueeze(1)  # [batch_size, 1, max_frames, n_features]
            meter_spec = meter_spec.transpose(2, 3)  # [batch_size, 1, n_features, max_frames]
            
            # Pass through CNN layers
            for i in range(len(self.conv_layers)):
                meter_spec = self.conv_layers[i](meter_spec)
                meter_spec = self.norm_layers[i](meter_spec)
                meter_spec = F.relu(meter_spec)
                meter_spec = self.pool_layers[i](meter_spec)
            
            # Flatten and process with meter-level layers
            meter_flat = meter_spec.reshape(batch_size, -1)
            meter_features = self.meter_layer(meter_flat)
            
            # Output prediction for this meter
            meter_output = torch.sigmoid(self.output_layer(meter_features))
            outputs.append(meter_output)
        
        # Stack all meter outputs
        stacked_outputs = torch.stack(outputs, dim=1)  # [batch_size, max_meters, 1]
        
        # Create mask for padding (where input is all zeros)
        mask = (torch.sum(torch.abs(x), dim=(2, 3)) > 0).float().unsqueeze(2)
        
        # Apply mask
        masked_outputs = stacked_outputs * mask
        
        return masked_outputs


class BCEMaskedLoss(nn.Module):
    """
    Binary Cross Entropy loss with support for masked values.
    Ignores positions where target is -1 (padding).
    """
    
    def __init__(self):
        super(BCEMaskedLoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute masked BCE loss.
        
        Args:
            predictions: Model predictions [batch_size, max_meters, 1]
            targets: Target values [batch_size, max_meters, 1], with -1 for padding
            
        Returns:
            Loss value
        """
        # Create mask for valid positions (not -1)
        mask = (targets != -1).float()
        
        # Replace -1 with 0 to make it valid for BCE
        valid_targets = torch.clamp(targets, 0, 1)
        
        # Compute BCE loss
        loss = self.bce(predictions, valid_targets)
        
        # Apply mask and compute mean over valid positions
        masked_loss = loss * mask
        num_valid = torch.sum(mask)
        
        if num_valid > 0:
            return torch.sum(masked_loss) / num_valid
        else:
            return torch.tensor(0.0, device=predictions.device)


def extract_spectrogram(audio_path: str, sr: int = 12000, hop_length: int = 128,
                         n_fft: int = 2048) -> np.ndarray:
    """
    Extract spectrogram from audio file using AudioFeature.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        hop_length: Hop length for spectrogram
        n_fft: FFT size
        
    Returns:
        Spectrogram matrix
    """
    # Create audio feature extractor
    audio_feat = AudioFeature(audio_path, sr=sr, hop_length=hop_length)
    
    # Extract spectrogram
    spectrogram = audio_feat.extract_spectrogram()
    
    # Convert to decibels and normalize
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    spectrogram_norm = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min() + 1e-8)
    
    return spectrogram_norm


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration for the SpectrogramModel.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "data": {
            "max_frames": 300,
            "max_meters": 201
        },
        "audio": {
            "sr": 12000,
            "hop_length": 128,
            "n_fft": 2048
        },
        "model": {
            "dropout": 0.3,
            "cnn": {
                "channels": [32, 64, 128, 128],
                "kernel_sizes": [(3, 3), (3, 3), (3, 3), (3, 3)],
                "pool_sizes": [(2, 2), (2, 2), (2, 2), (2, 2)]
            }
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.001,
            "epochs": 50,
            "weight_decay": 1e-5
        }
    } 