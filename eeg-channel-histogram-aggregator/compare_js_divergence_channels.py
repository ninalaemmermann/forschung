"""
Calculate JS Divergence averaged over all 27 EEG channels.
Each channel is binned independently, then JS divergences are averaged.
"""

import pickle
import numpy as np
from scipy.spatial.distance import jensenshannon
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
BASE_PATH = Path('/home/data/ninalaemmermann/forschung')
OUTPUT_DIR = BASE_PATH / 'eeg-channel-histogram-aggregator' / 'data' / 'output'
N_BINS = 50  # Number of histogram bins per channel
SEIZURE_TYPES = ['absz', 'cpsz', 'fnsz', 'gnsz']

def load_eeg_data(seizure_type):
    """Load original EEG data (27 channels)."""
    pkl_path = BASE_PATH / f'eeg_results_{seizure_type}.pkl'
    print(f"Lade {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_histogram(channel_data, bins, range_bounds):
    """
    Calculate normalized histogram for a single channel.
    
    Args:
        channel_data: 1D array of EEG values
        bins: Number of bins
        range_bounds: (min, max) tuple for histogram range
    
    Returns:
        Normalized histogram (probabilities sum to 1)
    """
    hist, _ = np.histogram(channel_data, bins=bins, range=range_bounds)
    
    # Normalize to probability distribution
    hist = hist.astype(float)
    hist_sum = np.sum(hist)
    
    if hist_sum > 0:
        hist /= hist_sum
    else:
        # If all zeros, create uniform distribution
        hist = np.ones(bins) / bins
    
    # Add small epsilon to avoid log(0) in JS divergence
    hist = hist + 1e-10
    hist /= np.sum(hist)
    
    return hist

def calculate_js_divergence_per_channel(data1, data2, n_bins=50):
    """
    Calculate JS divergence for each of the 27 channels and return the mean.
    
    Args:
        data1: (27, n_timepoints) array
        data2: (27, n_timepoints) array
        n_bins: Number of histogram bins
    
    Returns:
        mean_js: Mean JS divergence over all channels
        per_channel_js: Array of JS divergence per channel (for debugging)
    """
    n_channels = data1.shape[0]
    js_divergences = []
    
    # Determine global range for consistent binning across all data
    global_min = min(np.min(data1), np.min(data2))
    global_max = max(np.max(data1), np.max(data2))
    range_bounds = (global_min, global_max)
    
    for channel in range(n_channels):
        channel_data1 = data1[channel, :]
        channel_data2 = data2[channel, :]
        
        # Create histograms
        hist1 = calculate_histogram(channel_data1, n_bins, range_bounds)
        hist2 = calculate_histogram(channel_data2, n_bins, range_bounds)
        
        # Calculate JS divergence
        js = jensenshannon(hist1, hist2, base=2)  # base=2 for bits
        js_divergences.append(js)
    
    return np.mean(js_divergences), np.array(js_divergences)

def main():
    print("=" * 70)
    print("JS DIVERGENCE (CHANNEL-WISE) AUF 27D ORIGINAL-DATEN")
    print("=" * 70)
    print(f"Bins pro Kanal: {N_BINS}")
    print(f"Methode: Mittelwert über alle 27 Kanäle")
    print()
    
    print(f"\n" + "=" * 70)
    print("DATEN LADEN")
    print("=" * 70)
    
    # Load all datasets
    datasets = {}
    
    for seizure_type in SEIZURE_TYPES:
        print(f"\n{seizure_type.upper()}:")
        data = load_eeg_data(seizure_type)
        
        # Convert to arrays if needed
        seizure_data = np.array(data['seizure_data']) if isinstance(data['seizure_data'], list) else data['seizure_data']
        non_seizure_data = np.array(data['non_seizure_data']) if isinstance(data['non_seizure_data'], list) else data['non_seizure_data']
        
        # Store as (27, n_timepoints) arrays
        datasets[f'{seizure_type}_seizure'] = seizure_data
        datasets[f'{seizure_type}_non_seizure'] = non_seizure_data
        
        print(f"  Seizure:     {seizure_data.shape}")
        print(f"  Non-seizure: {non_seizure_data.shape}")
    
    # Calculate JS divergence matrix
    print(f"\n" + "=" * 70)
    print("JS DIVERGENCE BERECHNEN (KANALWEISE)")
    print("=" * 70)
    print()
    
    # Y-axis (rows): Seizures first, then non-seizures
    # X-axis (cols): Non-seizures first, then seizures
    
    y_labels = []
    y_dataset_names = []
    
    # Y-axis: First all seizures
    for st in SEIZURE_TYPES:
        y_labels.append(f'{st.upper()}\nSeizure')
        y_dataset_names.append(f'{st}_seizure')
    
    # Y-axis: Then all non-seizures
    for st in SEIZURE_TYPES:
        y_labels.append(f'{st.upper()}\nNon-Sz')
        y_dataset_names.append(f'{st}_non_seizure')
    
    # X-axis: First all non-seizures
    x_labels = []
    x_dataset_names = []
    
    for st in SEIZURE_TYPES:
        x_labels.append(f'{st.upper()}\nNon-Sz')
        x_dataset_names.append(f'{st}_non_seizure')
    
    # X-axis: Then all seizures
    for st in SEIZURE_TYPES:
        x_labels.append(f'{st.upper()}\nSeizure')
        x_dataset_names.append(f'{st}_seizure')
    
    n = len(y_labels)
    distance_matrix = np.zeros((n, n))
    
    # Calculate pairwise JS divergences
    total_pairs = n * (n + 1) // 2
    with tqdm(total=total_pairs, desc="Berechne JS-Divergenzen") as pbar:
        for i, y_name in enumerate(y_dataset_names):
            for j, x_name in enumerate(x_dataset_names):
                if i <= j:
                    mean_js, per_channel = calculate_js_divergence_per_channel(
                        datasets[y_name], 
                        datasets[x_name],
                        n_bins=N_BINS
                    )
                    
                    distance_matrix[i, j] = mean_js
                    distance_matrix[j, i] = mean_js
                    
                    if i != j:
                        print(f"{y_labels[i]:15} vs {x_labels[j]:15}: {mean_js:.4f} bits")
                    
                    pbar.update(1)
    
    # Plot heatmap
    print(f"\n" + "=" * 70)
    print("HEATMAP ERSTELLEN")
    print("=" * 70)
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with limited color range
    sns.heatmap(distance_matrix, 
                annot=True, 
                fmt='.4f',
                cmap='YlOrRd',
                vmin=0.0,
                vmax=0.15,
                xticklabels=x_labels,
                yticklabels=y_labels,
                cbar_kws={'label': 'JS Divergence (bits)', 'extend': 'max'})
    
    plt.title(f'JS Divergence Matrix (27 Kanäle, Mittelwert)\n' + 
              f'{N_BINS} bins pro Kanal',
              fontsize=14, fontweight='bold')
    plt.xlabel('', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    output_path = OUTPUT_DIR / f'js_divergence_channels_27d.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap gespeichert: {output_path}")
    
    # Summary statistics
    print(f"\n" + "=" * 70)
    print("STATISTIK")
    print("=" * 70)
    
    # Within-type comparisons (seizure vs non-seizure)
    within_type = []
    for i, st in enumerate(SEIZURE_TYPES):
        idx_sz = i * 2
        idx_nsz = i * 2 + 1
        within_type.append(distance_matrix[idx_sz, idx_nsz])
    
    print(f"\nInnerhalb Seizure-Typ (Seizure vs Non-Seizure):")
    for st, val in zip(SEIZURE_TYPES, within_type):
        print(f"  {st.upper()}: {val:.4f} bits")
    print(f"  Durchschnitt: {np.mean(within_type):.4f} ± {np.std(within_type):.4f} bits")
    
    # Between-type comparisons (seizure vs seizure)
    between_type_sz = []
    for i in range(len(SEIZURE_TYPES)):
        for j in range(i+1, len(SEIZURE_TYPES)):
            idx1 = i * 2
            idx2 = j * 2
            between_type_sz.append(distance_matrix[idx1, idx2])
    
    print(f"\nZwischen Seizure-Typen (Seizure vs Seizure):")
    print(f"  Min: {np.min(between_type_sz):.4f} bits")
    print(f"  Max: {np.max(between_type_sz):.4f} bits")
    print(f"  Durchschnitt: {np.mean(between_type_sz):.4f} ± {np.std(between_type_sz):.4f} bits")
    
    # Between-type comparisons (non-seizure vs non-seizure)
    between_type_nsz = []
    for i in range(len(SEIZURE_TYPES)):
        for j in range(i+1, len(SEIZURE_TYPES)):
            idx1 = i * 2 + 1
            idx2 = j * 2 + 1
            between_type_nsz.append(distance_matrix[idx1, idx2])
    
    print(f"\nZwischen Seizure-Typen (Non-Seizure vs Non-Seizure):")
    print(f"  Min: {np.min(between_type_nsz):.4f} bits")
    print(f"  Max: {np.max(between_type_nsz):.4f} bits")
    print(f"  Durchschnitt: {np.mean(between_type_nsz):.4f} ± {np.std(between_type_nsz):.4f} bits")
    
    # Ratio analysis
    print(f"\n" + "=" * 70)
    print("ANALYSE")
    print("=" * 70)
    ratio = np.mean(within_type) / np.mean(between_type_sz)
    print(f"Verhältnis Within/Between (Seizure): {ratio:.2f}")
    print(f"  < 1.0: Seizure-Typen sind sich ähnlicher als Seizure/Non-Seizure innerhalb Typ")
    print(f"  > 1.0: Seizure/Non-Seizure innerhalb Typ sind sich ähnlicher")
    
    print("\n" + "=" * 70)
    print("✓ FERTIG!")
    print("=" * 70)

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
