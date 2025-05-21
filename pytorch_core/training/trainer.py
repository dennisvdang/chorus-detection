"""
Training functionality for chorus detection.
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Callable
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from pytorch_core.models.crnn import CRNN, CustomBCELoss


class Trainer:
    """
    Trainer class for chorus detection model.
    Handles training, validation, evaluation, and checkpoints.
    """
    
    def __init__(
        self,
        model: CRNN,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "./checkpoints",
        device: Optional[str] = None
    ):
        """
        Initialize trainer with model and data loaders.
        
        Args:
            model: CRNN model to train
            config: Configuration dictionary
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            checkpoint_dir: Directory to save checkpoints
            device: Device to use for training ('cpu' or 'cuda')
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Create criterion and optimizer
        self.criterion = CustomBCELoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.config["training"]["optimizer"]["name"].lower()
        lr = self.config["training"]["optimizer"]["lr"]
        
        if optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        if "scheduler" not in self.config["training"]:
            return None
            
        scheduler_name = self.config["training"]["scheduler"]["name"].lower()
        
        if scheduler_name == "reduce_lr_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config["training"]["scheduler"]["factor"],
                patience=self.config["training"]["scheduler"]["patience"],
                verbose=True
            )
        else:
            return None
    
    def train(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            epochs: Number of epochs to train for (default from config)
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config["training"]["epochs"]
        early_stopping_patience = self.config["training"]["early_stopping"]["patience"]
        min_delta = self.config["training"]["early_stopping"]["min_delta"]
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        print(f"Training on device: {self.device}")
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            
            # Validation phase
            if self.val_loader:
                val_loss, val_acc = self._validate_epoch()
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                
                # Update learning rate scheduler if using validation loss
                if self.scheduler:
                    self.scheduler.step(val_loss)
                
                # Check for improvement for early stopping
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Save best model
                    self.save_checkpoint(f"best_model.pt")
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
                
                print(f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")
            else:
                print(f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")
            
            # Save checkpoint periodically
            if epoch % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
                
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        return self.history
    
    def _train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (features, targets) in enumerate(self.train_loader):
            features, targets = features.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            
            # Calculate accuracy only on non-padded elements
            mask = (targets != -1).float()
            predicted = (outputs > 0.5).float()
            correct += ((predicted == targets) * mask).sum().item()
            total += mask.sum().item()
            
        # Calculate average metrics
        avg_loss = running_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def _validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(self.val_loader):
                features, targets = features.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                
                # Calculate accuracy only on non-padded elements
                mask = (targets != -1).float()
                predicted = (outputs > 0.5).float()
                correct += ((predicted == targets) * mask).sum().item()
                total += mask.sum().item()
                
        # Calculate average metrics
        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data (uses val_loader if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        loader = test_loader or self.val_loader
        if not loader:
            raise ValueError("No test_loader provided and no val_loader available")
        
        self.model.eval()
        
        all_targets = []
        all_predictions = []
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, targets in loader:
                features, targets = features.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                
                # Get binary predictions
                mask = (targets != -1).float()
                predicted = (outputs > 0.5).float()
                
                # Collect for metrics
                for i in range(targets.size(0)):
                    for j in range(targets.size(1)):
                        if mask[i, j, 0] > 0:  # Only include non-padded values
                            all_targets.append(targets[i, j, 0].cpu().item())
                            all_predictions.append(predicted[i, j, 0].cpu().item())
                
                # Accuracy
                correct += ((predicted == targets) * mask).sum().item()
                total += mask.sum().item()
        
        # Calculate metrics
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = running_loss / len(loader)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Print evaluation results
        print("\nTest results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        self._plot_confusion_matrix(cm)
        
        return metrics
    
    def save_checkpoint(self, filename: str) -> str:
        """
        Save model checkpoint.
        
        Args:
            filename: Name of checkpoint file
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, checkpoint_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training and validation loss/accuracy.
        
        Args:
            save_path: Optional path to save the plot
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        if self.val_loader:
            ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_accuracy'], label='Train Accuracy')
        if self.val_loader:
            ax2.plot(self.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix array
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = ['Not Chorus', 'Chorus']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Label the plot
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
                
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix plot saved to {save_path}")
            
        plt.show() 