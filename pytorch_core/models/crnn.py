"""
PyTorch implementation of the CRNN model for chorus detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for chorus detection.
    Processes audio at two time scales:
    1. CNN extracts features from frames within each meter
    2. Bidirectional LSTM models relationships between meters
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CRNN model with the given configuration.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(CRNN, self).__init__()
        self.config = config
        
        # CNN parameters
        cnn_config = config["model"]["cnn"]
        self.filters = cnn_config["filters"]
        self.kernel_sizes = cnn_config["kernel_sizes"]
        self.pool_sizes = cnn_config["pool_sizes"]
        self.padding = cnn_config["padding"]
        
        # LSTM parameters
        rnn_config = config["model"]["rnn"]
        self.hidden_size = rnn_config["hidden_size"]
        self.bidirectional = rnn_config["bidirectional"]
        
        # Data parameters
        self.max_frames = config["data"]["max_frames"]
        self.max_meters = config["data"]["max_meters"]
        self.n_features = config["data"]["n_features"]
        self.dropout_rate = config["model"]["dropout"]
        
        # Build CNN layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_channels = self.n_features
        for i in range(len(self.filters)):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=self.filters[i],
                    kernel_size=self.kernel_sizes[i],
                    padding='same'
                )
            )
            self.pool_layers.append(
                nn.MaxPool1d(
                    kernel_size=self.pool_sizes[i],
                    padding=0 if self.padding == 'valid' else self.pool_sizes[i] // 2
                )
            )
            in_channels = self.filters[i]
        
        # Calculate output size of CNN
        cnn_output_size = self._calculate_cnn_output_size()
        
        # Build LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Build output layer
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.dropout = nn.Dropout(self.dropout_rate)
        self.output_layer = nn.Linear(lstm_output_size, 1)
        
    def _calculate_cnn_output_size(self) -> int:
        """Calculate the output size of the CNN layers."""
        size = self.max_frames
        for pool_size in self.pool_sizes:
            size = size // pool_size
        return size * self.filters[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, max_meters, max_frames, n_features]
            
        Returns:
            Output tensor of shape [batch_size, max_meters, 1]
        """
        batch_size = x.size(0)
        
        # Process each meter through CNN
        meter_features = []
        for i in range(self.max_meters):
            # Extract frames for current meter
            meter_frames = x[:, i, :, :]  # [batch_size, max_frames, n_features]
            
            # Transpose for Conv1D (channel last -> channel first)
            meter_frames = meter_frames.transpose(1, 2)  # [batch_size, n_features, max_frames]
            
            # Pass through CNN layers
            conv_out = meter_frames
            for j in range(len(self.conv_layers)):
                conv_out = F.relu(self.conv_layers[j](conv_out))
                conv_out = self.pool_layers[j](conv_out)
            
            # Flatten CNN output
            flat_features = conv_out.reshape(batch_size, -1)
            meter_features.append(flat_features)
        
        # Stack meter features
        meters_sequence = torch.stack(meter_features, dim=1)  # [batch_size, max_meters, cnn_output_size]
        
        # Create mask for padding values
        mask = (torch.sum(torch.abs(meters_sequence), dim=2) != 0).float().unsqueeze(2)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(meters_sequence)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Pass through output layer
        outputs = torch.sigmoid(self.output_layer(lstm_out))
        
        # Apply mask to handle padding
        outputs = outputs * mask
        
        return outputs


class CustomBCELoss(nn.Module):
    """
    Custom Binary Cross Entropy loss that handles masked values.
    """
    
    def __init__(self):
        super(CustomBCELoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Predictions of shape [batch_size, max_meters, 1]
            targets: Ground truth of shape [batch_size, max_meters, 1]
            
        Returns:
            Mean loss over non-masked values
        """
        # Create mask for valid values (not -1)
        mask = (targets != -1).float()
        
        # Ensure masked targets are valid for BCE (between 0 and 1)
        valid_targets = targets * mask
        
        # Calculate BCE loss
        loss = self.bce(outputs, valid_targets)
        
        # Apply mask and calculate mean over non-masked values
        masked_loss = loss * mask
        n_valid = torch.sum(mask)
        
        if n_valid > 0:
            return torch.sum(masked_loss) / n_valid
        else:
            return torch.sum(masked_loss)  # Will be 0 if no valid elements 