"""
2D Analysis: Find 2 best global channels and compare seizure types.
Creates 2D histograms and calculates JS-Divergence.
"""

import pickle
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_PATH = Path('/home/data/ninalaemmermann/forschung')
OUTPUT_DIR = BASE_PATH / 'eeg-channel-histogram-aggregator' / 'data' / 'output'
SEIZURE_TYPES = ['absz', 'cpsz', 'fnsz', 'gnsz']
N_BINS = 50  # Bins per dimension for 2D histogram

def load_eeg_data(seizure_type):
    """Load EEG data from pickle file."""
    pkl_path = BASE_PATH / f'eeg_results_{seizure_type}.pkl'
    print(f"Lade {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def find_best_global_channels():
    """
    Find the 2 best channels that discriminate Seizure vs Non-Seizure
    across ALL seizure types.
    """
    print("=" * 70)
    print("SCHRITT 1: BESTE 2 KANÄLE GLOBAL FINDEN")
    print("=" * 70)
    
    # Load all data
    all_data = {}
    channel_names = None
    
    for seizure_type in SEIZURE_TYPES:
        data = load_eeg_data(seizure_type)
        all_data[seizure_type] = data
        if channel_names is None:
            channel_names = data['channel_names']
    
    n_channels = len(channel_names)
    print(f"\n{n_channels} Kanäle: {channel_names}")
    
    # For each channel, calculate discriminative power using JS-Divergence
    channel_scores = []
    
    print("\nBerechne Diskriminierungskraft pro Kanal (JS-Divergenz über alle Paarvergleiche)...")
    for ch_idx, ch_name in enumerate(channel_names):
        # Collect samples from all seizure types and conditions
        all_groups = []
        
        for seizure_type in SEIZURE_TYPES:
            data = all_data[seizure_type]
            
            # Get this channel's data for seizure and non-seizure separately
            sz_channel = np.array(data['seizure_data'][ch_idx])
            nsz_channel = np.array(data['non_seizure_data'][ch_idx])
            
            # Sample to limit size
            if len(sz_channel) > 10000:
                sz_channel = np.random.choice(sz_channel, 10000, replace=False)
            if len(nsz_channel) > 10000:
                nsz_channel = np.random.choice(nsz_channel, 10000, replace=False)
            
            all_groups.append(sz_channel)
            all_groups.append(nsz_channel)
        
        # Create 1D histograms for all 8 groups (4 types × 2 conditions)
        histograms = []
        for group in all_groups:
            hist, _ = np.histogram(group, bins=50, density=True)
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            histograms.append(hist)
        
        # Calculate average JS-Divergence across all pairs
        js_values = []
        for i in range(len(histograms)):
            for j in range(i+1, len(histograms)):
                # Use scipy's jensenshannon (returns JS distance, square it to get divergence)
                js_dist = jensenshannon(histograms[i], histograms[j])
                js_div = js_dist ** 2  # Convert distance to divergence
                js_values.append(js_div)
        
        score = np.mean(js_values)
        channel_scores.append((ch_idx, ch_name, score))
        print(f"  {ch_name:20s}: {score:.6f}")
    
    # Sort by score (descending)
    channel_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Select best and worst
    best_channel = channel_scores[0]
    worst_channel = channel_scores[-1]
    
    print(f"\n" + "=" * 70)
    print("BESTER UND SCHLECHTESTER KANAL:")
    print("=" * 70)
    print(f"BEST:  {best_channel[1]} (Index {best_channel[0]}): Score {best_channel[2]:.6f}")
    print(f"WORST: {worst_channel[1]} (Index {worst_channel[0]}): Score {worst_channel[2]:.6f}")
    print(f"Ratio: {best_channel[2] / worst_channel[2]:.2f}x")
    
    return best_channel[0], worst_channel[0], channel_names

def create_2d_histogram(samples_ch1, samples_ch2, n_bins):
    """
    Create 2D histogram from two channels.
    
    Returns:
        histogram: 2D array (n_bins, n_bins)
        edges_ch1, edges_ch2: Bin edges
    """
    # Create histogram
    hist, edges_ch1, edges_ch2 = np.histogram2d(
        samples_ch1, samples_ch2, bins=n_bins
    )
    
    return hist, edges_ch1, edges_ch2

def histogram_to_probability(hist):
    """Convert histogram counts to probability distribution."""
    total = hist.sum()
    if total == 0:
        return hist
    return hist / total

def calculate_js_divergence_2d(hist1, hist2):
    """
    Calculate JS-Divergence between two 2D histograms.
    """
    # Flatten to 1D
    p = histogram_to_probability(hist1).flatten()
    q = histogram_to_probability(hist2).flatten()
    
    # Calculate JS divergence
    js_div = jensenshannon(p, q)
    
    return js_div

def main():
    # Find best 2 channels
    ch1_idx, ch2_idx, channel_names = find_best_global_channels()
    ch1_name = channel_names[ch1_idx]
    ch2_name = channel_names[ch2_idx]
    
    # Create 2D histograms for all seizure types
    print(f"\n" + "=" * 70)
    print(f"SCHRITT 2: 2D-HISTOGRAMME ERSTELLEN")
    print(f"Kanäle: {ch1_name} (X) vs {ch2_name} (Y)")
    print("=" * 70)
    
    histograms = {}
    
    for seizure_type in SEIZURE_TYPES:
        print(f"\n{seizure_type.upper()}:")
        data = load_eeg_data(seizure_type)
        
        # Get the two channels
        sz_ch1 = np.array(data['seizure_data'][ch1_idx])
        sz_ch2 = np.array(data['seizure_data'][ch2_idx])
        
        nsz_ch1 = np.array(data['non_seizure_data'][ch1_idx])
        nsz_ch2 = np.array(data['non_seizure_data'][ch2_idx])
        
        # Sample to limit size
        max_samples = 500000
        if len(sz_ch1) > max_samples:
            indices = np.random.choice(len(sz_ch1), max_samples, replace=False)
            sz_ch1 = sz_ch1[indices]
            sz_ch2 = sz_ch2[indices]
        
        if len(nsz_ch1) > max_samples:
            indices = np.random.choice(len(nsz_ch1), max_samples, replace=False)
            nsz_ch1 = nsz_ch1[indices]
            nsz_ch2 = nsz_ch2[indices]
        
        # Create 2D histograms
        sz_hist, edges_ch1, edges_ch2 = create_2d_histogram(sz_ch1, sz_ch2, N_BINS)
        nsz_hist, _, _ = create_2d_histogram(nsz_ch1, nsz_ch2, N_BINS)
        
        histograms[f'{seizure_type}_seizure'] = sz_hist
        histograms[f'{seizure_type}_non_seizure'] = nsz_hist
        
        print(f"  Seizure:     {len(sz_ch1):,} samples → {N_BINS}×{N_BINS} histogram")
        print(f"  Non-seizure: {len(nsz_ch1):,} samples → {N_BINS}×{N_BINS} histogram")
        
        # Check overlap
        sz_prob = histogram_to_probability(sz_hist)
        nsz_prob = histogram_to_probability(nsz_hist)
        overlap = np.sum((sz_prob > 0) & (nsz_prob > 0))
        total_bins = N_BINS * N_BINS
        print(f"  Overlap: {overlap}/{total_bins} bins ({overlap/total_bins*100:.1f}%)")
    
    # Calculate JS divergence matrix
    print(f"\n" + "=" * 70)
    print("SCHRITT 3: JS-DIVERGENZ BERECHNEN")
    print("=" * 70)
    
    labels = []
    for st in SEIZURE_TYPES:
        labels.append(f'{st.upper()}\nSeizure')
        labels.append(f'{st.upper()}\nNon-Sz')
    
    n = len(labels)
    js_matrix = np.zeros((n, n))
    
    hist_names = []
    for st in SEIZURE_TYPES:
        hist_names.append(f'{st}_seizure')
        hist_names.append(f'{st}_non_seizure')
    
    print()
    for i, name1 in enumerate(hist_names):
        for j, name2 in enumerate(hist_names):
            if i <= j:
                js_div = calculate_js_divergence_2d(histograms[name1], histograms[name2])
                js_matrix[i, j] = js_div
                js_matrix[j, i] = js_div
                
                if i != j:
                    print(f"{labels[i]:15} vs {labels[j]:15}: {js_div:.4f}")
    
    # Plot heatmap
    print(f"\n" + "=" * 70)
    print("SCHRITT 4: VISUALISIERUNG")
    print("=" * 70)
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(js_matrix, 
                annot=True, 
                fmt='.4f',
                cmap='RdYlGn_r',
                xticklabels=labels,
                yticklabels=labels,
                vmin=0,
                vmax=0.7,
                cbar_kws={'label': 'JS Divergence'})
    
    plt.title(f'JS-Divergenz Matrix (2D: {ch1_name} vs {ch2_name})\n' + 
              f'{N_BINS}×{N_BINS} Bins',
              fontsize=14, fontweight='bold')
    plt.xlabel('', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f'js_divergence_2d_{ch1_name}_{ch2_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap gespeichert: {output_path}")
    
    # Create 2D density plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for idx, seizure_type in enumerate(SEIZURE_TYPES):
        # Seizure plot
        ax = axes[0, idx]
        sz_hist = histograms[f'{seizure_type}_seizure']
        im = ax.imshow(sz_hist.T, origin='lower', aspect='auto', cmap='viridis')
        ax.set_title(f'{seizure_type.upper()} Seizure')
        ax.set_xlabel(ch1_name)
        ax.set_ylabel(ch2_name)
        plt.colorbar(im, ax=ax)
        
        # Non-seizure plot
        ax = axes[1, idx]
        nsz_hist = histograms[f'{seizure_type}_non_seizure']
        im = ax.imshow(nsz_hist.T, origin='lower', aspect='auto', cmap='viridis')
        ax.set_title(f'{seizure_type.upper()} Non-Seizure')
        ax.set_xlabel(ch1_name)
        ax.set_ylabel(ch2_name)
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    density_path = OUTPUT_DIR / f'density_2d_{ch1_name}_{ch2_name}.png'
    plt.savefig(density_path, dpi=300, bbox_inches='tight')
    print(f"✓ Density Plots gespeichert: {density_path}")
    
    # Summary statistics
    print(f"\n" + "=" * 70)
    print("STATISTIK")
    print("=" * 70)
    
    within_type = []
    for i, st in enumerate(SEIZURE_TYPES):
        idx_sz = i * 2
        idx_nsz = i * 2 + 1
        within_type.append(js_matrix[idx_sz, idx_nsz])
    
    print(f"\nInnerhalb Seizure-Typ (Seizure vs Non-Seizure):")
    for st, val in zip(SEIZURE_TYPES, within_type):
        print(f"  {st.upper()}: {val:.4f}")
    print(f"  Durchschnitt: {np.mean(within_type):.4f} ± {np.std(within_type):.4f}")
    
    between_type_sz = []
    for i in range(len(SEIZURE_TYPES)):
        for j in range(i+1, len(SEIZURE_TYPES)):
            idx1 = i * 2
            idx2 = j * 2
            between_type_sz.append(js_matrix[idx1, idx2])
    
    print(f"\nZwischen Seizure-Typen (Seizure vs Seizure):")
    print(f"  Min: {np.min(between_type_sz):.4f}")
    print(f"  Max: {np.max(between_type_sz):.4f}")
    print(f"  Durchschnitt: {np.mean(between_type_sz):.4f} ± {np.std(between_type_sz):.4f}")
    
    print("\n" + "=" * 70)
    print("✓ FERTIG!")
    print("=" * 70)

if __name__ == '__main__':
    np.random.seed(42)
    main()
