#!/usr/bin/env python3
"""
Example script for preprocessing the chorus detection dataset.

This script demonstrates how to preprocess all audio files and save
features/labels as pkl.gz files, while calculating dataset statistics.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script import preprocess_dataset
import argparse


def main():
    """Example usage of the preprocessing functionality"""
    parser = argparse.ArgumentParser(description="Preprocess chorus detection dataset")
    
    # Required arguments
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="Directory containing audio files (.mp3)")
    parser.add_argument("--labels-csv", type=str, required=True,
                        help="CSV file with chorus labels")
    parser.add_argument("--labels-dir", type=str, required=True,
                        help="Directory to save processed labels and features")
    
    # Optional audio processing parameters
    parser.add_argument("--sample-rate", type=int, default=12000,
                        help="Audio sample rate (default: 12000)")
    parser.add_argument("--hop-length", type=int, default=128,
                        help="Hop length for feature extraction (default: 128)")
    parser.add_argument("--n-components", type=int, default=3,
                        help="Number of NMF components (default: 3)")
    
    # Experiment configuration
    parser.add_argument("--experiment-name", type=str, default="chorus_detection",
                        help="Experiment name (default: chorus_detection)")
    parser.add_argument("--version", type=str, default="1",
                        help="Experiment version (default: 1)")
    
    args = parser.parse_args()
    
    print("Starting dataset preprocessing...")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Labels CSV: {args.labels_csv}")
    print(f"Output directory: {args.labels_dir}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Hop length: {args.hop_length}")
    print(f"NMF components: {args.n_components}")
    print("-" * 50)
    
    # Run preprocessing
    preprocess_dataset(args)
    
    print("-" * 50)
    print("Preprocessing completed!")
    print(f"Features saved to: {os.path.join(args.labels_dir, 'features')}")
    print(f"Labels saved to: {args.labels_dir}")
    print(f"Statistics saved to: {os.path.join(args.labels_dir, f'dataset_stats_{args.experiment_name}_v{args.version}.json')}")


if __name__ == "__main__":
    main() 