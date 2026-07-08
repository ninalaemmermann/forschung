"""
Calculate Wasserstein Distance (Energy Distance) on PCA-reduced data.
No binning required - works directly with sample points.
"""

import pickle
import numpy as np
from scipy.stats import energy_distance
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
BASE_PATH = Path('/home/data/ninalaemmermann/forschung')
OUTPUT_DIR = BASE_PATH / 'eeg-channel-histogram-aggregator' / 'data' / 'output'
MAX_SAMPLES = 50000  # Increased for better statistical robustness
SEIZURE_TYPES = ['absz', 'cpsz', 'fnsz', 'gnsz']

def load_eeg_data(seizure_type):
    """Load original EEG data (27 channels)."""
    pkl_path = BASE_PATH / f'eeg_results_{seizure_type}.pkl'
    print(f"Lade {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def prepare_samples(data_array, max_samples):
    """
    Convert (27, n_timepoints) to (n_samples, 27) by transposing and sampling.
    """
    if isinstance(data_array, list):
        data_array = np.array(data_array)
    
    # Transpose: (27, n_timepoints) → (n_timepoints, 27)
    samples = data_array.T
    
    # Sample if too many datapoints
    if samples.shape[0] > max_samples:
        indices = np.random.choice(samples.shape[0], max_samples, replace=False)
        samples = samples[indices]
    
    return samples

def subsample_data(samples, max_samples):
    """
    Randomly subsample data if too large.
    
    Args:
        samples: (n_samples, n_dims) array
        max_samples: Maximum number of samples to keep
    
    Returns:
        subsampled: (max_samples, n_dims) array
    """
    if samples.shape[0] <= max_samples:
        return samples
    
    indices = np.random.choice(samples.shape[0], max_samples, replace=False)
    return samples[indices]

def calculate_energy_distance_multidim(samples1, samples2):
    """
    Calculate Energy Distance (generalized Wasserstein-2) for multidimensional data.
    
    Energy distance is a metric between distributions that generalizes
    to arbitrary dimensions and doesn't require binning.
    
    Formula: E(X,Y) = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
    
    Args:
        samples1: (n1, d) array
        samples2: (n2, d) array
    
    Returns:
        distance: Energy distance value
    """
    from scipy.spatial.distance import cdist
    
    n1, d = samples1.shape
    n2, _ = samples2.shape
    
    # Subsample for tractability if needed
    if n1 > 2000:
        idx1 = np.random.choice(n1, 2000, replace=False)
        samples1 = samples1[idx1]
        n1 = 2000
    if n2 > 2000:
        idx2 = np.random.choice(n2, 2000, replace=False)
        samples2 = samples2[idx2]
        n2 = 2000
    
    # Calculate pairwise Euclidean distances efficiently
    # E[||X-Y||]
    cross_dists = cdist(samples1, samples2, metric='euclidean')
    cross_term = np.mean(cross_dists)
    
    # E[||X-X'||] - exclude diagonal (distance to self = 0)
    within1_dists = cdist(samples1, samples1, metric='euclidean')
    np.fill_diagonal(within1_dists, np.nan)
    within1_term = np.nanmean(within1_dists)
    
    # E[||Y-Y'||] - exclude diagonal
    within2_dists = cdist(samples2, samples2, metric='euclidean')
    np.fill_diagonal(within2_dists, np.nan)
    within2_term = np.nanmean(within2_dists)
    
    # Energy distance formula
    energy_dist = 2 * cross_term - within1_term - within2_term
    
    return energy_dist

def main():
    print("=" * 70)
    print("WASSERSTEIN DISTANCE (ENERGY DISTANCE) AUF 27D ORIGINAL-DATEN")
    print("=" * 70)
    print(f"Max Samples pro Verteilung: {MAX_SAMPLES:,}")
    print(f"Dimensionen: 27 EEG-Kanäle (keine PCA)")
    print()
    
    print(f"\n" + "=" * 70)
    print("DATEN LADEN UND VORBEREITEN")
    print("=" * 70)
    
    # Prepare all datasets
    datasets = {}
    
    for seizure_type in SEIZURE_TYPES:
        print(f"\n{seizure_type.upper()}:")
        data = load_eeg_data(seizure_type)
        
        # Seizure samples: (27, n_timepoints) → (n_samples, 27)
        seizure_samples = prepare_samples(data['seizure_data'], MAX_SAMPLES)
        datasets[f'{seizure_type}_seizure'] = seizure_samples
        print(f"  Seizure:     {seizure_samples.shape[0]:,} samples × {seizure_samples.shape[1]} Kanäle")
        
        # Non-seizure samples
        non_seizure_samples = prepare_samples(data['non_seizure_data'], MAX_SAMPLES)
        datasets[f'{seizure_type}_non_seizure'] = non_seizure_samples
        print(f"  Non-seizure: {non_seizure_samples.shape[0]:,} samples × {non_seizure_samples.shape[1]} Kanäle")
    
    # Calculate Wasserstein distance matrix
    print(f"\n" + "=" * 70)
    print("WASSERSTEIN DISTANCE BERECHNEN")
    print("=" * 70)
    print("(Dies kann einige Minuten dauern...)")
    print()
    
    labels = []
    for st in SEIZURE_TYPES:
        labels.append(f'{st.upper()}\nSeizure')
        labels.append(f'{st.upper()}\nNon-Sz')
    
    n = len(labels)
    distance_matrix = np.zeros((n, n))
    
    dataset_names = []
    for st in SEIZURE_TYPES:
        dataset_names.append(f'{st}_seizure')
        dataset_names.append(f'{st}_non_seizure')
    
    # Calculate pairwise distances with progress bar
    total_pairs = n * (n + 1) // 2
    with tqdm(total=total_pairs, desc="Berechne Distanzen") as pbar:
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i <= j:
                    dist = calculate_energy_distance_multidim(
                        datasets[name1], 
                        datasets[name2]
                    )
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
                    
                    if i != j:
                        print(f"{labels[i]:15} vs {labels[j]:15}: {dist:.2e}")
                    
                    pbar.update(1)
    
    # Plot heatmap
    print(f"\n" + "=" * 70)
    print("HEATMAP ERSTELLEN")
    print("=" * 70)
    
    # Ensure diagonal is exactly 0
    np.fill_diagonal(distance_matrix, 0.0)
    
    # Scale matrix to have common exponent (× 10^5)
    scaled_matrix = distance_matrix * 1e5
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with scaled values
    sns.heatmap(scaled_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='YlOrRd',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Wasserstein Distance (× 10⁻⁵)'})
    
    plt.title(f'Wasserstein Distance Matrix (27D Original-Daten)\n' + 
              f'{MAX_SAMPLES:,} samples, alle EEG-Kanäle',
              fontsize=14, fontweight='bold')
    plt.xlabel('', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    output_path = OUTPUT_DIR / f'wasserstein_distance_27d.png'
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
        print(f"  {st.upper()}: {val:.2e}")
    print(f"  Durchschnitt: {np.mean(within_type):.2e} ± {np.std(within_type):.2e}")
    
    # Between-type comparisons (seizure vs seizure)
    between_type_sz = []
    for i in range(len(SEIZURE_TYPES)):
        for j in range(i+1, len(SEIZURE_TYPES)):
            idx1 = i * 2  # seizure of type i
            idx2 = j * 2  # seizure of type j
            between_type_sz.append(distance_matrix[idx1, idx2])
    
    print(f"\nZwischen Seizure-Typen (Seizure vs Seizure):")
    print(f"  Min: {np.min(between_type_sz):.2e}")
    print(f"  Max: {np.max(between_type_sz):.2e}")
    print(f"  Durchschnitt: {np.mean(between_type_sz):.2e} ± {np.std(between_type_sz):.2e}")
    
    # Between-type comparisons (non-seizure vs non-seizure)
    between_type_nsz = []
    for i in range(len(SEIZURE_TYPES)):
        for j in range(i+1, len(SEIZURE_TYPES)):
            idx1 = i * 2 + 1  # non-seizure of type i
            idx2 = j * 2 + 1  # non-seizure of type j
            between_type_nsz.append(distance_matrix[idx1, idx2])
    
    print(f"\nZwischen Seizure-Typen (Non-Seizure vs Non-Seizure):")
    print(f"  Min: {np.min(between_type_nsz):.2e}")
    print(f"  Max: {np.max(between_type_nsz):.2e}")
    print(f"  Durchschnitt: {np.mean(between_type_nsz):.2e} ± {np.std(between_type_nsz):.2e}")
    
    # Ratio analysis
    print(f"\n" + "=" * 70)
    print("ANALYSE")
    print("=" * 70)
    print(f"Verhältnis Within/Between (Seizure): {np.mean(within_type) / np.mean(between_type_sz):.2f}")
    print(f"  < 1.0: Seizure-Typen sind sich ähnlicher als Seizure/Non-Seizure innerhalb Typ")
    print(f"  > 1.0: Seizure/Non-Seizure innerhalb Typ sind sich ähnlicher")
    
    print("\n" + "=" * 70)
    print("✓ FERTIG!")
    print("=" * 70)

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
