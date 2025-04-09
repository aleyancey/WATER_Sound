import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_features(processed_dir):
    """Load all processed features and create a DataFrame."""
    features_data = []
    
    # Load label mapping
    with open(os.path.join(processed_dir, 'label_map.json'), 'r') as f:
        label_map = json.load(f)
    
    # Process each feature file
    for file in tqdm(os.listdir(processed_dir)):
        if file.endswith('.npz'):
            try:
                data = np.load(os.path.join(processed_dir, file))
                
                # Extract surface type from filename
                surface_type = file.split('_')[0]
                
                # Extract scalar features
                features_data.append({
                    'surface_type': surface_type,
                    'filename': file,
                    'tempo': float(data['tempo'].item()),
                    'loudness': float(data['loudness'].item()),
                    'brightness': float(data['brightness'].item()),
                    'noisiness': float(data['noisiness'].item()),
                    'mel_spectrogram': data['mel_spectrogram'],
                    'mfcc': data['mfcc']
                })
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
    
    return pd.DataFrame(features_data), label_map

def plot_spectrograms(features_df, output_dir, num_samples=2):
    """Plot mel spectrograms and MFCCs for random samples from each surface type."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    for surface_type in features_df['surface_type'].unique():
        surface_samples = features_df[features_df['surface_type'] == surface_type].sample(n=min(num_samples, len(features_df[features_df['surface_type'] == surface_type])))
        
        for idx, (_, sample) in enumerate(surface_samples.iterrows()):
            plt.figure(figsize=(15, 5))
            
            # Plot mel spectrogram
            mel_spec = sample['mel_spectrogram']
            if len(mel_spec.shape) == 3:
                mel_spec = mel_spec[0]  # Take the first channel if multi-channel
            plt.imshow(mel_spec, aspect='auto', origin='lower')
            plt.title(f"{surface_type} - Mel Spectrogram")
            plt.colorbar(format='%+2.0f dB')
            plt.xlabel('Time')
            plt.ylabel('Mel Frequency')
            
            # Plot MFCC
            mfcc = sample['mfcc']
            if len(mfcc.shape) == 3:
                mfcc = mfcc[0]  # Take the first channel if multi-channel
            plt.imshow(mfcc, aspect='auto', origin='lower')
            plt.title(f"{surface_type} - MFCC")
            plt.colorbar()
            plt.xlabel('Time')
            plt.ylabel('MFCC Coefficients')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{surface_type}_sample_{idx+1}.png"))
            plt.close()

def plot_feature_distributions(features_df, output_dir):
    """Plot distributions of audio characteristics across surface types."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create violin plots for each feature
    features_to_plot = ['tempo', 'loudness', 'brightness', 'noisiness']
    
    plt.figure(figsize=(15, 10))
    for idx, feature in enumerate(features_to_plot, 1):
        plt.subplot(2, 2, idx)
        sns.violinplot(data=features_df, x='surface_type', y=feature)
        plt.xticks(rotation=45)
        plt.title(f'{feature.capitalize()} Distribution by Surface Type')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation = features_df[features_to_plot].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlations.png'))
    plt.close()

def plot_feature_scatter(features_df, output_dir):
    """Create scatter plots comparing pairs of features, colored by surface type."""
    os.makedirs(output_dir, exist_ok=True)
    
    features_to_plot = ['tempo', 'loudness', 'brightness', 'noisiness']
    
    # Create pairwise scatter plots
    plt.figure(figsize=(15, 15))
    for i, feature1 in enumerate(features_to_plot):
        for j, feature2 in enumerate(features_to_plot):
            if i < j:  # Only plot upper triangle
                plt.subplot(len(features_to_plot)-1, len(features_to_plot)-1, (i)*(len(features_to_plot)-1) + j)
                sns.scatterplot(data=features_df, x=feature1, y=feature2, hue='surface_type', alpha=0.6)
                plt.xticks(rotation=45)
                plt.title(f'{feature1.capitalize()} vs {feature2.capitalize()}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_scatter.png'))
    plt.close()

def main():
    # Set up directories
    processed_dir = 'data/processed'
    output_dir = 'data/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load features
    print("Loading features...")
    features_df, label_map = load_features(processed_dir)
    
    # Create visualizations
    print("Creating spectrograms...")
    plot_spectrograms(features_df, os.path.join(output_dir, 'spectrograms'))
    
    print("Creating feature distributions...")
    plot_feature_distributions(features_df, output_dir)
    
    print("Creating feature scatter plots...")
    plot_feature_scatter(features_df, output_dir)
    
    print(f"Visualizations saved to {output_dir}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print("-" * 50)
    print(f"Total samples: {len(features_df)}")
    print("\nSamples per surface type:")
    print(features_df['surface_type'].value_counts())
    print("\nFeature statistics:")
    print(features_df[['tempo', 'loudness', 'brightness', 'noisiness']].describe())

if __name__ == '__main__':
    main() 