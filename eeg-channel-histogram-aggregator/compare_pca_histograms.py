"""
Calculate JS-Divergence on PCA-reduced data (10D instead of 27D).
Creates histograms with fewer bins to avoid curse of dimensionality.
"""

import pickle
import numpy as np
from scipy.spatial.distance import jensenshannon
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
N_BINS = 5  # Bins per dimension (5^5 = 3,125 total bins for 5D)
BASE_PATH = Path('/home/data/ninalaemmermann/forschung')
OUTPUT_DIR = BASE_PATH / 'eeg-channel-histogram-aggregator' / 'data' / 'output'
PCA_FILE = OUTPUT_DIR / 'pca_data_5d.pkl'

def load_pca_data():
    """Load PCA-transformed data."""
    print(f"Lade PCA-Daten: {PCA_FILE}")
    with open(PCA_FILE, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ {len(data['seizure_types'])} Seizure-Typen geladen")
    print(f"  Komponenten: {data['n_components']}")
    print(f"  Erklärte Varianz: {data['explained_variance_ratio'].sum():.2%}")
    return data

def create_sparse_histogram(samples, n_bins):
    """
    Create sparse multidimensional histogram.
    
    Args:
        samples: (n_samples, n_dims) array
        n_bins: Number of bins per dimension
    
    Returns:
        sparse_hist: Dictionary {bin_tuple: count}
        bin_edges: List of bin edges per dimension
    """
    n_samples, n_dims = samples.shape
    
    # Calculate bin edges for each dimension
    bin_edges = []
    for dim in range(n_dims):
        edges = np.linspace(samples[:, dim].min(), samples[:, dim].max(), n_bins + 1)
        bin_edges.append(edges)
    
    # Digitize each dimension
    bin_indices = []
    for dim in range(n_dims):
        indices = np.digitize(samples[:, dim], bin_edges[dim]) - 1
        # Clip to valid range [0, n_bins-1]
        indices = np.clip(indices, 0, n_bins - 1)
        bin_indices.append(indices)
    
    # Create sparse histogram
    sparse_hist = {}
    for i in range(n_samples):
        bin_tuple = tuple(bin_indices[dim][i] for dim in range(n_dims))
        sparse_hist[bin_tuple] = sparse_hist.get(bin_tuple, 0) + 1
    
    return sparse_hist, bin_edges

def sparse_to_probability(sparse_hist, all_bins):
    """
    Convert sparse histogram to probability vector.
    
    Args:
        sparse_hist: Dictionary {bin_tuple: count}
        all_bins: Set of all bin tuples across both histograms
    
    Returns:
        prob_vector: Probability vector for all bins
    """
    total = sum(sparse_hist.values())
    prob_vector = np.array([sparse_hist.get(bin_tuple, 0) / total for bin_tuple in all_bins])
    return prob_vector

def calculate_js_divergence(hist1, hist2):
    """
    Calculate Jensen-Shannon divergence between two sparse histograms.
    
    Args:
        hist1, hist2: Sparse histograms (dict)
    
    Returns:
        js_div: JS divergence value
    """
    # Get union of all bins
    all_bins = sorted(set(hist1.keys()) | set(hist2.keys()))
    
    # Convert to probability vectors
    p = sparse_to_probability(hist1, all_bins)
    q = sparse_to_probability(hist2, all_bins)
    
    # Calculate JS divergence
    js_div = jensenshannon(p, q)
    
    return js_div

def main():
    print("=" * 70)
    print("JS-DIVERGENZ AUF PCA-DATEN (10D)")
    print("=" * 70)
    print()
    
    # Load PCA data
    pca_data = load_pca_data()
    seizure_types = list(pca_data['seizure_types'].keys())
    
    print(f"\n" + "=" * 70)
    print(f"HISTOGRAMME ERSTELLEN ({N_BINS} bins per dimension)")
    print("=" * 70)
    print(f"Theoretische Bins: {N_BINS}^{pca_data['n_components']} = {N_BINS**pca_data['n_components']:,}")
    print()
    
    # Create histograms for all seizure types
    histograms = {}
    
    for seizure_type in seizure_types:
        print(f"\n{seizure_type.upper()}:")
        
        # Seizure histogram
        seizure_samples = pca_data['seizure_types'][seizure_type]['seizure']
        sz_hist, sz_edges = create_sparse_histogram(seizure_samples, N_BINS)
        histograms[f'{seizure_type}_seizure'] = sz_hist
        print(f"  Seizure:     {len(sz_hist):,} besetzte Bins von {seizure_samples.shape[0]:,} Samples")
        
        # Non-seizure histogram
        non_seizure_samples = pca_data['seizure_types'][seizure_type]['non_seizure']
        nsz_hist, nsz_edges = create_sparse_histogram(non_seizure_samples, N_BINS)
        histograms[f'{seizure_type}_non_seizure'] = nsz_hist
        print(f"  Non-seizure: {len(nsz_hist):,} besetzte Bins von {non_seizure_samples.shape[0]:,} Samples")
    
    # Check overlap example
    print(f"\n" + "=" * 70)
    print("OVERLAP-TEST")
    print("=" * 70)
    absz_sz = histograms['absz_seizure']
    absz_nsz = histograms['absz_non_seizure']
    overlap = len(set(absz_sz.keys()) & set(absz_nsz.keys()))
    print(f"ABSZ Seizure vs Non-Seizure:")
    print(f"  Seizure Bins: {len(absz_sz):,}")
    print(f"  Non-Seizure Bins: {len(absz_nsz):,}")
    print(f"  Overlap: {overlap:,}")
    print(f"  Overlap %: {overlap/len(absz_sz)*100:.2f}%")
    
    # Calculate JS divergence matrix
    print(f"\n" + "=" * 70)
    print("JS-DIVERGENZ BERECHNEN")
    print("=" * 70)
    
    labels = []
    for st in seizure_types:
        labels.append(f'{st.upper()}\nSeizure')
        labels.append(f'{st.upper()}\nNon-Sz')
    
    n = len(labels)
    js_matrix = np.zeros((n, n))
    
    hist_names = []
    for st in seizure_types:
        hist_names.append(f'{st}_seizure')
        hist_names.append(f'{st}_non_seizure')
    
    print()
    for i, name1 in enumerate(hist_names):
        for j, name2 in enumerate(hist_names):
            if i <= j:
                js_div = calculate_js_divergence(histograms[name1], histograms[name2])
                js_matrix[i, j] = js_div
                js_matrix[j, i] = js_div
                
                if i != j:
                    print(f"{labels[i]:15} vs {labels[j]:15}: {js_div:.4f}")
    
    # Plot heatmap
    print(f"\n" + "=" * 70)
    print("HEATMAP ERSTELLEN")
    print("=" * 70)
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(js_matrix, 
                annot=True, 
                fmt='.4f',
                cmap='RdYlGn_r',
                xticklabels=labels,
                yticklabels=labels,
                vmin=0,
                vmax=0.7,  # ln(2) ≈ 0.693
                cbar_kws={'label': 'JS Divergence'})
    
    plt.title(f'JS-Divergenz Matrix (PCA {pca_data["n_components"]}D, {N_BINS} bins)\n' + 
              f'Erklärte Varianz: {pca_data["explained_variance_ratio"].sum():.1%}',
              fontsize=14, fontweight='bold')
    plt.xlabel('', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    output_path = OUTPUT_DIR / f'js_divergence_pca_{pca_data["n_components"]}d_{N_BINS}bins.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap gespeichert: {output_path}")
    
    # Summary statistics
    print(f"\n" + "=" * 70)
    print("STATISTIK")
    print("=" * 70)
    
    # Within-type comparisons (seizure vs non-seizure)
    within_type = []
    for i, st in enumerate(seizure_types):
        idx_sz = i * 2
        idx_nsz = i * 2 + 1
        within_type.append(js_matrix[idx_sz, idx_nsz])
    
    print(f"Innerhalb Seizure-Typ (Seizure vs Non-Seizure):")
    for st, val in zip(seizure_types, within_type):
        print(f"  {st.upper()}: {val:.4f}")
    print(f"  Durchschnitt: {np.mean(within_type):.4f} ± {np.std(within_type):.4f}")
    
    # Between-type comparisons
    between_type = []
    for i in range(len(seizure_types)):
        for j in range(i+1, len(seizure_types)):
            idx1 = i * 2  # seizure of type i
            idx2 = j * 2  # seizure of type j
            between_type.append(js_matrix[idx1, idx2])
    
    print(f"\nZwischen Seizure-Typen (Seizure vs Seizure):")
    print(f"  Min: {np.min(between_type):.4f}")
    print(f"  Max: {np.max(between_type):.4f}")
    print(f"  Durchschnitt: {np.mean(between_type):.4f} ± {np.std(between_type):.4f}")
    
    print("\n" + "=" * 70)
    print("✓ FERTIG!")
    print("=" * 70)

if __name__ == '__main__':
    main()
