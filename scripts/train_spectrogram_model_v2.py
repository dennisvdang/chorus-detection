#!/usr/bin/env python
"""
Training script for the spectrogram-based chorus detection model.
Uses the modular audio processor and improved dataset implementation.
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model and dataset components
from pytorch_core.models.spectrogram_model import SpectrogramModel, BCEMaskedLoss, create_default_config
from pytorch_core.data.spectrogram_dataset_v2 import create_data_loaders


def train_model(config, data_config, output_dir):
    """
    Train the spectrogram-based model.
    
    Args:
        config: Model configuration
        data_config: Data paths and settings
        output_dir: Directory to save model and results
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config=config,
        audio_dir=data_config["audio_dir"],
        labels_dir=data_config["labels_dir"],
        precomputed_dir=data_config.get("precomputed_dir"),
        train_split=data_config.get("train_split", 0.7),
        val_split=data_config.get("val_split", 0.15),
        test_split=data_config.get("test_split", 0.15)
    )
    
    # Create model
    print("Creating model...")
    model = SpectrogramModel(config)
    model.to(device)
    
    # Setup loss function, optimizer and scheduler
    criterion = BCEMaskedLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training parameters
    epochs = config["training"]["epochs"]
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Training loop
    print(f"Starting training for {epochs} epochs")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        model.train()
        epoch_train_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_train_loss += loss.item()
            
            # Print batch progress
            if (i + 1) % 10 == 0 or i == len(train_loader) - 1:
                print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate average training loss
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        print(f"Training Loss: {epoch_train_loss:.4f}")
        
        # Validate
        model.eval()
        epoch_val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Update statistics
                epoch_val_loss += loss.item()
                
                # Store predictions and targets for metrics calculation
                mask = targets != -1
                val_predictions.extend(outputs[mask].cpu().numpy().flatten())
                val_targets.extend(targets[mask].cpu().numpy().flatten())
        
        # Calculate average validation loss
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        print(f"Validation Loss: {epoch_val_loss:.4f}")
        
        # Calculate metrics
        if val_predictions:
            binary_predictions = (np.array(val_predictions) > 0.5).astype(int)
            binary_targets = np.array(val_targets).astype(int)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                binary_targets, binary_predictions, average='binary'
            )
            accuracy = accuracy_score(binary_targets, binary_predictions)
            
            print(f"Validation Metrics - Accuracy: {accuracy:.4f}, "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Save model if it's the best so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            
            # Save model checkpoint
            model_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'train_loss': epoch_train_loss,
                'config': config
            }, model_path)
            print(f"Saved best model to {model_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Update statistics
            test_loss += loss.item()
            
            # Store predictions and targets for metrics calculation
            mask = targets != -1
            test_predictions.extend(outputs[mask].cpu().numpy().flatten())
            test_targets.extend(targets[mask].cpu().numpy().flatten())
    
    # Calculate average test loss
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Calculate test metrics
    if test_predictions:
        binary_predictions = (np.array(test_predictions) > 0.5).astype(int)
        binary_targets = np.array(test_targets).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_targets, binary_predictions, average='binary'
        )
        accuracy = accuracy_score(binary_targets, binary_predictions)
        
        print(f"Test Metrics - Accuracy: {accuracy:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'test_loss': test_loss,
        'config': config
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Save training history and metrics
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    }
    
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    
    print("Training completed!")
    return model


def main():
    """Parse command line arguments and run training."""
    parser = argparse.ArgumentParser(description="Train spectrogram-based chorus detection model")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory with label pickles")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and logs")
    parser.add_argument("--precomputed_dir", type=str, help="Directory for precomputed features")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = create_default_config()
        
    # Update config with command line args
    config["training"]["batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.lr
    config["training"]["epochs"] = args.epochs
    
    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Data configuration
    data_config = {
        "audio_dir": args.audio_dir,
        "labels_dir": args.labels_dir,
        "precomputed_dir": args.precomputed_dir
    }
    
    # Train model
    train_model(config, data_config, args.output_dir)


if __name__ == "__main__":
    main() 