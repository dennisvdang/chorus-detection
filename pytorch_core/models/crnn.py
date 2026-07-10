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
        
        # CNN for frame-level feature extraction
        self.cnn_layers = self._build_cnn_layers()
        
        # Calculate CNN output dimension
        self.cnn_output_dim = self._calculate_cnn_output_dim()
        
        # RNN for meter-level sequence modeling
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=config["model"]["rnn"]["hidden_size"],
            bidirectional=config["model"]["rnn"]["bidirectional"],
            batch_first=True
        )

        # Optional dropout between RNN and output layer
        # (the original TF model used none; config dropout defaults to 0.0)
        self.dropout = nn.Dropout(config["model"].get("dropout", 0.0))

        # Output layer
        rnn_output_dim = config["model"]["rnn"]["hidden_size"] * (2 if config["model"]["rnn"]["bidirectional"] else 1)
        self.output_layer = nn.Linear(rnn_output_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def _build_cnn_layers(self) -> nn.Sequential:
        """Build CNN layers based on configuration."""
        layers = []
        in_channels = self.config["data"]["n_features"]
        
        for i, (filters, kernel_size, pool_size) in enumerate(zip(
            self.config["model"]["cnn"]["filters"], 
            self.config["model"]["cnn"]["kernel_sizes"],
            self.config["model"]["cnn"]["pool_sizes"]
        )):
            padding = kernel_size // 2 if self.config["model"]["cnn"]["padding"] == "same" else 0
            layers.append(nn.Conv1d(in_channels, filters, kernel_size, padding=padding))
            layers.append(nn.ReLU())
            # ceil_mode matches Keras MaxPooling1D(padding='same'):
            # 300 -> 150 -> 75 -> 38 frames, keeping the flattened CNN
            # output dimension identical to the original TF model
            layers.append(nn.MaxPool1d(pool_size, ceil_mode=True))
            in_channels = filters
            
        return nn.Sequential(*layers)
    
    def _calculate_cnn_output_dim(self) -> int:
        """Calculate the output dimensions of the CNN for a single frame."""
        # Start with the maximum number of frames per meter
        max_frames = self.config["data"]["max_frames"]
        n_features = self.config["data"]["n_features"]
        
        # Create a dummy input tensor
        x = torch.zeros(1, n_features, max_frames)
        
        # Pass through CNN layers
        x = self.cnn_layers(x)
        
        # Return flattened dimension
        return x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CRNN model.
        
        Args:
            x: Input tensor of shape [batch_size, max_meters, max_frames, n_features]
            
        Returns:
            Output tensor of shape [batch_size, max_meters, 1]
        """
        batch_size, max_meters, max_frames, n_features = x.shape
        
        # Create mask for padding (meters with all zeros are padding)
        mask = (x.sum(dim=(2, 3)) != 0).float().unsqueeze(-1)
        
        # Reshape for TimeDistributed-like behavior
        x_reshaped = x.view(batch_size * max_meters, max_frames, n_features)
        
        # Transpose for Conv1D (expects [N, C, L])
        x_reshaped = x_reshaped.transpose(1, 2)
        
        # Apply CNN
        cnn_out = self.cnn_layers(x_reshaped)
        
        # Flatten CNN output
        cnn_out = cnn_out.reshape(cnn_out.size(0), -1)
        
        # Reshape back to [batch_size, max_meters, cnn_output_size]
        cnn_out = cnn_out.view(batch_size, max_meters, -1)
        
        # Apply masking before RNN to handle variable-length sequences
        # The RNN will process all time steps, but we'll mask the outputs
        
        # Apply RNN
        rnn_out, _ = self.rnn(cnn_out)
        rnn_out = self.dropout(rnn_out)

        # Apply output layer
        output = self.output_layer(rnn_out)
        output = self.sigmoid(output)
        
        # Apply mask to zero out padding
        output = output * mask
        
        return output


class CustomBCELoss(nn.Module):
    """Custom Binary Cross Entropy Loss with masking for padded values."""
    
    def __init__(self):
        super(CustomBCELoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute masked BCE loss.
        
        Args:
            predictions: Model predictions [batch_size, max_meters, 1]
            targets: Target values [batch_size, max_meters, 1] with -1 for padding
            
        Returns:
            Masked BCE loss value
        """
        # Create mask for valid targets (not -1)
        mask = (targets != -1).float()
        
        # Convert -1 to 0 for BCE calculation
        targets = torch.clamp(targets, min=0)
        
        # Calculate BCE
        bce_loss = self.bce(predictions, targets)
        
        # Apply mask
        masked_loss = bce_loss * mask

        # Return mean over valid (non-padded) values, guarding against an
        # all-padding batch (sum(mask) == 0 would otherwise give NaN)
        num_valid = torch.sum(mask)
        if num_valid > 0:
            return torch.sum(masked_loss) / num_valid
        return torch.tensor(0.0, device=predictions.device)