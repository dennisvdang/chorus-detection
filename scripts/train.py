#!/usr/bin/env python
"""
Training script for chorus detection model.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_core.models.crnn import CRNN
from pytorch_core.data.dataset import create_data_loaders
from pytorch_core.training.trainer import Trainer


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train chorus detection model")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--segments_dir", type=str, default="data/segments_V2",
                        help="Directory with segment data")
    parser.add_argument("--labels_dir", type=str, default="data/labels_V2",
                        help="Directory with label data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override the batch size from the config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., 'cuda:0', 'cpu')")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    # Set device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config=config,
        segments_dir=args.segments_dir,
        labels_dir=args.labels_dir,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_seed=args.seed
    )
    print(f"Created data loaders with {len(train_loader.dataset)} training samples, "
          f"{len(val_loader.dataset)} validation samples, and "
          f"{len(test_loader.dataset)} test samples")
    
    # Create model
    model = CRNN(config)
    print(f"Created CRNN model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed training from {args.resume}")
    
    # Train model
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set:")
    metrics = trainer.evaluate(test_loader)
    
    # Plot training history
    trainer.plot_training_history(os.path.join(args.checkpoint_dir, "training_history.png"))
    
    # Print final metrics
    print("\nFinal test metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main() 