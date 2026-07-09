#!/usr/bin/env python3
"""
Tempo Comparison Script

This script compares three different tempo detection methods:
1. Spotify tempo (from metadata CSV)
2. Librosa tempo detection
3. Advanced tempo tracker (Complex Spectral Difference + Viterbi)

Usage:
    python tempo_comparison.py --audio-dir ../data/audio/processed --metadata-csv ../data/metadata/metadata.csv
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Audio processing
import librosa

# Import our advanced beat detection classes
sys.path.append('.')
from script import BeatAnalyzer, normalize_tempo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TempoComparator:
    """Compare different tempo detection methods"""
    
    def __init__(self, audio_dir: str, metadata_csv: str):
        self.audio_dir = audio_dir
        self.metadata_csv = metadata_csv
        self.beat_analyzer = BeatAnalyzer()
        
        # Load metadata
        self.metadata_df = pd.read_csv(metadata_csv)
        self.song_metadata = {}
        for song_id in self.metadata_df['song_id'].unique():
            song_rows = self.metadata_df[self.metadata_df['song_id'] == song_id]
            if not song_rows.empty:
                self.song_metadata[song_id] = song_rows.iloc[0].to_dict()
    
    def get_spotify_tempo(self, song_id: str) -> Optional[float]:
        """Get Spotify tempo for a song"""
        if song_id in self.song_metadata:
            sp_tempo = self.song_metadata[song_id].get('sp_tempo')
            if not pd.isna(sp_tempo):
                return normalize_tempo(float(sp_tempo))
        return None
    
    def get_librosa_tempo(self, audio_path: str) -> float:
        """Get tempo using librosa beat tracking"""
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, trim=False)
            return normalize_tempo(float(tempo))
        except Exception as e:
            logger.warning(f"Librosa tempo detection failed for {audio_path}: {e}")
            return 120.0
    
    def get_advanced_tempo(self, audio_path: str) -> Tuple[float, float]:
        """Get tempo using advanced beat analyzer"""
        try:
            beat_result = self.beat_analyzer.analyze(audio_path)
            return normalize_tempo(beat_result.tempo_bpm), beat_result.confidence
        except Exception as e:
            logger.warning(f"Advanced tempo detection failed for {audio_path}: {e}")
            return 120.0, 0.0
    
    def analyze_all_songs(self) -> pd.DataFrame:
        """Analyze tempo for all songs in the audio directory"""
        results = []
        
        # Get all audio files
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
            audio_files.extend(Path(self.audio_dir).glob(ext))
        
        logger.info(f"Found {len(audio_files)} audio files to analyze")
        
        for i, audio_file in enumerate(audio_files):
            song_id = audio_file.stem
            logger.info(f"Processing {song_id} ({i + 1}/{len(audio_files)})")
            
            result = {
                'song_id': song_id,
                'file_path': str(audio_file)
            }
            
            # Get Spotify tempo
            spotify_tempo = self.get_spotify_tempo(song_id)
            result['spotify_tempo'] = spotify_tempo
            result['has_spotify_tempo'] = spotify_tempo is not None
            
            # Get librosa tempo
            librosa_tempo = self.get_librosa_tempo(str(audio_file))
            result['librosa_tempo'] = librosa_tempo
            
            # Get advanced tempo
            advanced_tempo, confidence = self.get_advanced_tempo(str(audio_file))
            result['advanced_tempo'] = advanced_tempo
            result['advanced_confidence'] = confidence
            
            # Calculate differences if Spotify tempo is available
            if spotify_tempo is not None:
                result['spotify_librosa_diff'] = abs(spotify_tempo - librosa_tempo)
                result['spotify_advanced_diff'] = abs(spotify_tempo - advanced_tempo)
                result['librosa_advanced_diff'] = abs(librosa_tempo - advanced_tempo)
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def visualize_results(self, df: pd.DataFrame, output_dir: str = "tempo_analysis_results"):
        """Create comprehensive visualizations of tempo comparison results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter to songs with Spotify tempo for most analyses
        df_with_spotify = df[df['has_spotify_tempo']].copy()
        
        logger.info(f"Creating visualizations for {len(df)} total songs")
        logger.info(f"{len(df_with_spotify)} songs have Spotify tempo data")
        
        # 1. Tempo distribution comparison
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(df_with_spotify['spotify_tempo'], bins=30, alpha=0.7, label='Spotify', density=True)
        plt.hist(df['librosa_tempo'], bins=30, alpha=0.7, label='Librosa', density=True)
        plt.hist(df['advanced_tempo'], bins=30, alpha=0.7, label='Advanced', density=True)
        plt.xlabel('Tempo (BPM)')
        plt.ylabel('Density')
        plt.title('Tempo Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.boxplot([df_with_spotify['spotify_tempo'], df['librosa_tempo'], df['advanced_tempo']], 
                   labels=['Spotify', 'Librosa', 'Advanced'])
        plt.ylabel('Tempo (BPM)')
        plt.title('Tempo Distribution (Box Plot)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        tempo_stats = {
            'Spotify': df_with_spotify['spotify_tempo'].describe(),
            'Librosa': df['librosa_tempo'].describe(),
            'Advanced': df['advanced_tempo'].describe()
        }
        stats_df = pd.DataFrame(tempo_stats).round(2)
        plt.table(cellText=stats_df.values, rowLabels=stats_df.index, 
                 colLabels=stats_df.columns, loc='center', cellLoc='center')
        plt.axis('off')
        plt.title('Tempo Statistics')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tempo_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Scatter plots comparing methods (only for songs with Spotify data)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Spotify vs Librosa
        axes[0, 0].scatter(df_with_spotify['spotify_tempo'], df_with_spotify['librosa_tempo'], 
                          alpha=0.6, s=50)
        axes[0, 0].plot([60, 160], [60, 160], 'r--', alpha=0.8, label='Perfect Agreement')
        axes[0, 0].set_xlabel('Spotify Tempo (BPM)')
        axes[0, 0].set_ylabel('Librosa Tempo (BPM)')
        axes[0, 0].set_title('Spotify vs Librosa Tempo')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Spotify vs Advanced
        scatter = axes[0, 1].scatter(df_with_spotify['spotify_tempo'], df_with_spotify['advanced_tempo'], 
                                    c=df_with_spotify['advanced_confidence'], cmap='viridis', 
                                    alpha=0.6, s=50)
        axes[0, 1].plot([60, 160], [60, 160], 'r--', alpha=0.8, label='Perfect Agreement')
        axes[0, 1].set_xlabel('Spotify Tempo (BPM)')
        axes[0, 1].set_ylabel('Advanced Tempo (BPM)')
        axes[0, 1].set_title('Spotify vs Advanced Tempo')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='Confidence')
        
        # Librosa vs Advanced
        axes[1, 0].scatter(df['librosa_tempo'], df['advanced_tempo'], 
                          c=df['advanced_confidence'], cmap='viridis', alpha=0.6, s=50)
        axes[1, 0].plot([60, 160], [60, 160], 'r--', alpha=0.8, label='Perfect Agreement')
        axes[1, 0].set_xlabel('Librosa Tempo (BPM)')
        axes[1, 0].set_ylabel('Advanced Tempo (BPM)')
        axes[1, 0].set_title('Librosa vs Advanced Tempo')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confidence distribution
        axes[1, 1].hist(df['advanced_confidence'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(df['advanced_confidence'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["advanced_confidence"].mean():.3f}')
        axes[1, 1].set_xlabel('Advanced Tracker Confidence')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Advanced Tracker Confidence Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tempo_correlations.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Error analysis
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(df_with_spotify['spotify_librosa_diff'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Difference (BPM)')
        plt.ylabel('Count')
        plt.title('Spotify vs Librosa\nAbsolute Differences')
        plt.axvline(df_with_spotify['spotify_librosa_diff'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df_with_spotify["spotify_librosa_diff"].mean():.1f} BPM')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.hist(df_with_spotify['spotify_advanced_diff'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Difference (BPM)')
        plt.ylabel('Count')
        plt.title('Spotify vs Advanced\nAbsolute Differences')
        plt.axvline(df_with_spotify['spotify_advanced_diff'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df_with_spotify["spotify_advanced_diff"].mean():.1f} BPM')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.hist(df['librosa_advanced_diff'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Difference (BPM)')
        plt.ylabel('Count')
        plt.title('Librosa vs Advanced\nAbsolute Differences')
        plt.axvline(df['librosa_advanced_diff'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["librosa_advanced_diff"].mean():.1f} BPM')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tempo_differences.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Correlation matrix and statistics
        correlation_data = df_with_spotify[['spotify_tempo', 'librosa_tempo', 'advanced_tempo']].corr()
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('Tempo Method Correlations')
        
        plt.subplot(1, 2, 2)
        # Create accuracy metrics table
        metrics_data = {
            'Metric': ['Correlation with Spotify', 'MAE vs Spotify (BPM)', 'RMSE vs Spotify (BPM)', 
                      'Mean Confidence', 'Songs within 5 BPM', 'Songs within 10 BPM'],
            'Librosa': [
                f"{stats.pearsonr(df_with_spotify['spotify_tempo'], df_with_spotify['librosa_tempo'])[0]:.3f}",
                f"{mean_absolute_error(df_with_spotify['spotify_tempo'], df_with_spotify['librosa_tempo']):.1f}",
                f"{np.sqrt(mean_squared_error(df_with_spotify['spotify_tempo'], df_with_spotify['librosa_tempo'])):.1f}",
                "N/A",
                f"{(df_with_spotify['spotify_librosa_diff'] <= 5).sum()}/{len(df_with_spotify)} ({(df_with_spotify['spotify_librosa_diff'] <= 5).mean()*100:.1f}%)",
                f"{(df_with_spotify['spotify_librosa_diff'] <= 10).sum()}/{len(df_with_spotify)} ({(df_with_spotify['spotify_librosa_diff'] <= 10).mean()*100:.1f}%)"
            ],
            'Advanced': [
                f"{stats.pearsonr(df_with_spotify['spotify_tempo'], df_with_spotify['advanced_tempo'])[0]:.3f}",
                f"{mean_absolute_error(df_with_spotify['spotify_tempo'], df_with_spotify['advanced_tempo']):.1f}",
                f"{np.sqrt(mean_squared_error(df_with_spotify['spotify_tempo'], df_with_spotify['advanced_tempo'])):.1f}",
                f"{df['advanced_confidence'].mean():.3f}",
                f"{(df_with_spotify['spotify_advanced_diff'] <= 5).sum()}/{len(df_with_spotify)} ({(df_with_spotify['spotify_advanced_diff'] <= 5).mean()*100:.1f}%)",
                f"{(df_with_spotify['spotify_advanced_diff'] <= 10).sum()}/{len(df_with_spotify)} ({(df_with_spotify['spotify_advanced_diff'] <= 10).mean()*100:.1f}%)"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        plt.table(cellText=metrics_df[['Librosa', 'Advanced']].values, 
                 rowLabels=metrics_df['Metric'], 
                 colLabels=['Librosa', 'Advanced'],
                 loc='center', cellLoc='left')
        plt.axis('off')
        plt.title('Accuracy Metrics (vs Spotify)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tempo_metrics.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Confidence vs accuracy for advanced method
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df_with_spotify['advanced_confidence'], df_with_spotify['spotify_advanced_diff'], 
                   alpha=0.6, s=50)
        plt.xlabel('Advanced Tracker Confidence')
        plt.ylabel('Absolute Error vs Spotify (BPM)')
        plt.title('Confidence vs Accuracy (Advanced Tracker)')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df_with_spotify['advanced_confidence'], df_with_spotify['spotify_advanced_diff'], 1)
        p = np.poly1d(z)
        conf_range = np.linspace(df_with_spotify['advanced_confidence'].min(), 
                                df_with_spotify['advanced_confidence'].max(), 100)
        plt.plot(conf_range, p(conf_range), "r--", alpha=0.8, 
                label=f'Trend: slope={z[0]:.1f}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        # Binned confidence vs accuracy
        conf_bins = pd.cut(df_with_spotify['advanced_confidence'], bins=5)
        binned_accuracy = df_with_spotify.groupby(conf_bins)['spotify_advanced_diff'].mean()
        binned_accuracy.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.xlabel('Confidence Bins')
        plt.ylabel('Mean Absolute Error (BPM)')
        plt.title('Mean Error by Confidence Level')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"All visualizations saved to {output_dir}/")
    
    def save_results(self, df: pd.DataFrame, output_path: str):
        """Save results to CSV"""
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics"""
        df_with_spotify = df[df['has_spotify_tempo']]
        
        print("\n" + "="*60)
        print("TEMPO COMPARISON SUMMARY")
        print("="*60)
        
        print(f"Total songs analyzed: {len(df)}")
        print(f"Songs with Spotify tempo: {len(df_with_spotify)}")
        print(f"Coverage: {len(df_with_spotify)/len(df)*100:.1f}%")
        
        print(f"\nTempo Statistics:")
        print(f"Spotify tempo: {df_with_spotify['spotify_tempo'].mean():.1f} ± {df_with_spotify['spotify_tempo'].std():.1f} BPM")
        print(f"Librosa tempo: {df['librosa_tempo'].mean():.1f} ± {df['librosa_tempo'].std():.1f} BPM")
        print(f"Advanced tempo: {df['advanced_tempo'].mean():.1f} ± {df['advanced_tempo'].std():.1f} BPM")
        
        print(f"\nAccuracy vs Spotify (for songs with Spotify data):")
        lib_corr = stats.pearsonr(df_with_spotify['spotify_tempo'], df_with_spotify['librosa_tempo'])[0]
        adv_corr = stats.pearsonr(df_with_spotify['spotify_tempo'], df_with_spotify['advanced_tempo'])[0]
        print(f"Librosa correlation: {lib_corr:.3f}")
        print(f"Advanced correlation: {adv_corr:.3f}")
        
        lib_mae = mean_absolute_error(df_with_spotify['spotify_tempo'], df_with_spotify['librosa_tempo'])
        adv_mae = mean_absolute_error(df_with_spotify['spotify_tempo'], df_with_spotify['advanced_tempo'])
        print(f"Librosa MAE: {lib_mae:.1f} BPM")
        print(f"Advanced MAE: {adv_mae:.1f} BPM")
        
        print(f"\nAdvanced tracker confidence: {df['advanced_confidence'].mean():.3f} ± {df['advanced_confidence'].std():.3f}")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Compare tempo detection methods")
    
    parser.add_argument("--audio-dir", type=str, 
                        default="../data/audio/processed",
                        help="Directory containing audio files")
    parser.add_argument("--metadata-csv", type=str, 
                        default="../data/metadata/metadata.csv",
                        help="CSV file with song metadata including Spotify tempo")
    parser.add_argument("--output-dir", type=str, 
                        default="../output",
                        help="Output directory for results and visualizations")
    parser.add_argument("--save-csv", type=str, default="tempo_comparison_results.csv",
                        help="CSV file to save detailed results")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.audio_dir):
        logger.error(f"Audio directory not found: {args.audio_dir}")
        return
    
    if not os.path.exists(args.metadata_csv):
        logger.error(f"Metadata CSV not found: {args.metadata_csv}")
        return
    
    # Run comparison
    comparator = TempoComparator(args.audio_dir, args.metadata_csv)
    
    logger.info("Starting tempo analysis...")
    results_df = comparator.analyze_all_songs()
    
    # Save results
    if args.save_csv:
        comparator.save_results(results_df, args.save_csv)
    
    # Print summary
    comparator.print_summary(results_df)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    comparator.visualize_results(results_df, args.output_dir)
    
    logger.info("Tempo comparison analysis complete!")


if __name__ == "__main__":
    main() 