"""
Statistische Analyse der Wasserstein-Distanz Matrix:
- Hierarchisches Clustering
- Multidimensionale Skalierung (MDS)
- Statistische Signifikanztests
- Detaillierte Visualisierungen
"""

import pickle
import numpy as np
from scipy.stats import energy_distance
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Configuration
BASE_PATH = Path('/home/data/ninalaemmermann/forschung')
OUTPUT_DIR = BASE_PATH / 'eeg-channel-histogram-aggregator' / 'data' / 'output'
MAX_SAMPLES = 5000
SEIZURE_TYPES = ['absz', 'cpsz', 'fnsz', 'gnsz']

def load_eeg_data(seizure_type):
    """Load original EEG data (27 channels)."""
    pkl_path = BASE_PATH / f'eeg_results_{seizure_type}.pkl'
    print(f"Lade {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def prepare_samples(data_array, max_samples):
    """Convert (27, n_timepoints) to (n_samples, 27)."""
    if isinstance(data_array, list):
        data_array = np.array(data_array)
    samples = data_array.T
    if samples.shape[0] > max_samples:
        indices = np.random.choice(samples.shape[0], max_samples, replace=False)
        samples = samples[indices]
    return samples

def calculate_energy_distance_multidim(samples1, samples2):
    """Calculate Energy Distance for multidimensional data."""
    from scipy.spatial.distance import cdist
    
    n1, d = samples1.shape
    n2, _ = samples2.shape
    
    # Subsample for tractability
    if n1 > 2000:
        idx1 = np.random.choice(n1, 2000, replace=False)
        samples1 = samples1[idx1]
        n1 = 2000
    if n2 > 2000:
        idx2 = np.random.choice(n2, 2000, replace=False)
        samples2 = samples2[idx2]
        n2 = 2000
    
    cross_dists = cdist(samples1, samples2, metric='euclidean')
    cross_term = np.mean(cross_dists)
    
    within1_dists = cdist(samples1, samples1, metric='euclidean')
    within1_term = np.mean(within1_dists)
    
    within2_dists = cdist(samples2, samples2, metric='euclidean')
    within2_term = np.mean(within2_dists)
    
    energy_dist = 2 * cross_term - within1_term - within2_term
    return energy_dist

def bootstrap_energy_distance(samples1, samples2, n_bootstrap=100):
    """Calculate bootstrap confidence interval for energy distance."""
    distances = []
    n1, n2 = samples1.shape[0], samples2.shape[0]
    
    for _ in range(n_bootstrap):
        # Bootstrap resample
        idx1 = np.random.choice(n1, n1, replace=True)
        idx2 = np.random.choice(n2, n2, replace=True)
        
        dist = calculate_energy_distance_multidim(samples1[idx1], samples2[idx2])
        distances.append(dist)
    
    return np.mean(distances), np.std(distances)

def plot_hierarchical_clustering(distance_matrix, labels, output_dir):
    """Create hierarchical clustering dendrogram."""
    print("\n" + "=" * 70)
    print("HIERARCHISCHES CLUSTERING")
    print("=" * 70)
    
    # Ensure diagonal is zero
    distance_matrix_copy = distance_matrix.copy()
    np.fill_diagonal(distance_matrix_copy, 0)
    
    # Convert to condensed distance matrix
    condensed_dist = squareform(distance_matrix_copy)
    
    # Perform hierarchical clustering with different methods
    methods = ['ward', 'average', 'complete', 'single']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        # Perform linkage
        linkage_matrix = linkage(condensed_dist, method=method)
        
        # Plot dendrogram
        dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=90)
        ax.set_title(f'Hierarchisches Clustering ({method.capitalize()})', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Wasserstein Distance', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'hierarchical_clustering.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Dendrogram gespeichert: {output_path}")
    plt.close()
    
    # Return ward linkage for further analysis
    return linkage(condensed_dist, method='ward')

def plot_mds(distance_matrix, labels, output_dir):
    """Create MDS visualization."""
    print("\n" + "=" * 70)
    print("MULTIDIMENSIONALE SKALIERUNG (MDS)")
    print("=" * 70)
    
    # Perform MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)
    
    # Separate seizure types and conditions
    colors = []
    markers = []
    for label in labels:
        # Extract type
        if 'ABSZ' in label:
            colors.append('red')
        elif 'CPSZ' in label:
            colors.append('blue')
        elif 'FNSZ' in label:
            colors.append('green')
        elif 'GNSZ' in label:
            colors.append('orange')
        
        # Extract condition
        if 'Seizure' in label and 'Non-Sz' not in label:
            markers.append('o')  # circle for seizure
        else:
            markers.append('s')  # square for non-seizure
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i, (x, y, color, marker, label) in enumerate(zip(coords[:, 0], coords[:, 1], 
                                                           colors, markers, labels)):
        ax.scatter(x, y, c=color, marker=marker, s=300, alpha=0.7, 
                   edgecolors='black', linewidths=2)
        ax.annotate(label, (x, y), fontsize=9, ha='center', va='center')
    
    ax.set_xlabel('MDS Dimension 1', fontsize=12)
    ax.set_ylabel('MDS Dimension 2', fontsize=12)
    ax.set_title('Multidimensionale Skalierung der Wasserstein-Distanzen\n' + 
                 '(Nähe = ähnliche Verteilungen)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='ABSZ'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=10, label='CPSZ'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=10, label='FNSZ'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
               markersize=10, label='GNSZ'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Seizure'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=10, label='Non-Seizure')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'mds_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ MDS-Plot gespeichert: {output_path}")
    print(f"  Stress: {mds.stress_:.4f} (< 0.1 ist gut)")
    plt.close()

def analyze_distance_statistics(distance_matrix, labels, dataset_names):
    """Perform detailed statistical analysis."""
    print("\n" + "=" * 70)
    print("DETAILLIERTE STATISTIK")
    print("=" * 70)
    
    results = {
        'within_type': {},
        'between_type_seizure': [],
        'between_type_non_seizure': [],
        'absz_vs_others': []
    }
    
    # 1. Within-type distances (Seizure vs Non-Seizure)
    print("\n1. INNERHALB SEIZURE-TYP (Seizure vs. Non-Seizure):")
    print("-" * 70)
    for i, st in enumerate(SEIZURE_TYPES):
        idx_sz = i * 2
        idx_nsz = i * 2 + 1
        dist = distance_matrix[idx_sz, idx_nsz]
        results['within_type'][st] = dist
        print(f"  {st.upper():6s}: {dist:.6f}")
    
    within_mean = np.mean(list(results['within_type'].values()))
    within_std = np.std(list(results['within_type'].values()))
    print(f"\n  Durchschnitt: {within_mean:.6f} ± {within_std:.6f}")
    
    # 2. Between-type distances (Seizure vs Seizure)
    print("\n2. ZWISCHEN SEIZURE-TYPEN (Seizure vs. Seizure):")
    print("-" * 70)
    pairs = []
    for i in range(len(SEIZURE_TYPES)):
        for j in range(i+1, len(SEIZURE_TYPES)):
            idx1 = i * 2
            idx2 = j * 2
            dist = distance_matrix[idx1, idx2]
            results['between_type_seizure'].append(dist)
            type1 = SEIZURE_TYPES[i].upper()
            type2 = SEIZURE_TYPES[j].upper()
            pairs.append((type1, type2, dist))
    
    # Sort by distance
    pairs.sort(key=lambda x: x[2])
    print("  Ähnlichste Paare:")
    for type1, type2, dist in pairs[:3]:
        print(f"    {type1} ↔ {type2}: {dist:.6f}")
    
    print("\n  Unterschiedlichste Paare:")
    for type1, type2, dist in pairs[-3:]:
        print(f"    {type1} ↔ {type2}: {dist:.6f}")
    
    between_sz_mean = np.mean(results['between_type_seizure'])
    between_sz_std = np.std(results['between_type_seizure'])
    print(f"\n  Durchschnitt: {between_sz_mean:.6f} ± {between_sz_std:.6f}")
    
    # 3. Between-type distances (Non-Seizure vs Non-Seizure)
    print("\n3. ZWISCHEN SEIZURE-TYPEN (Non-Seizure vs. Non-Seizure):")
    print("-" * 70)
    for i in range(len(SEIZURE_TYPES)):
        for j in range(i+1, len(SEIZURE_TYPES)):
            idx1 = i * 2 + 1
            idx2 = j * 2 + 1
            dist = distance_matrix[idx1, idx2]
            results['between_type_non_seizure'].append(dist)
    
    between_nsz_mean = np.mean(results['between_type_non_seizure'])
    between_nsz_std = np.std(results['between_type_non_seizure'])
    print(f"  Durchschnitt: {between_nsz_mean:.6f} ± {between_nsz_std:.6f}")
    
    # 4. ABSZ vs. alle anderen
    print("\n4. ABSZ SONDERSTELLUNG:")
    print("-" * 70)
    absz_seizure_idx = 0
    for i in range(1, len(labels)):
        dist = distance_matrix[absz_seizure_idx, i]
        results['absz_vs_others'].append(dist)
        print(f"  ABSZ-Seizure vs {labels[i]:15s}: {dist:.6f}")
    
    absz_mean = np.mean(results['absz_vs_others'])
    print(f"\n  Durchschnittliche ABSZ-Distanz: {absz_mean:.6f}")
    
    # 5. Verhältnisse
    print("\n5. VERHÄLTNIS-ANALYSE:")
    print("-" * 70)
    ratio1 = within_mean / between_sz_mean
    print(f"  Within/Between (Seizure): {ratio1:.3f}")
    if ratio1 < 1:
        print(f"    → Seizure-Typen sind sich ÄHNLICHER als Sz/Non-Sz innerhalb Typ")
    else:
        print(f"    → Sz/Non-Sz innerhalb Typ sind sich ÄHNLICHER als verschiedene Typen")
    
    ratio2 = absz_mean / between_sz_mean
    print(f"\n  ABSZ/Andere: {ratio2:.3f}")
    print(f"    → ABSZ ist {ratio2:.1f}× weiter entfernt als andere Typen untereinander")
    
    # 6. Separability Score
    print("\n6. SEPARIERBARKEIT:")
    print("-" * 70)
    for st in SEIZURE_TYPES:
        st_upper = st.upper()
        # Find within-type distance
        within_dist = results['within_type'][st]
        
        # Find average distance to other types
        idx = SEIZURE_TYPES.index(st) * 2  # seizure index
        other_dists = []
        for j in range(len(labels)):
            if j != idx and j != idx + 1:  # exclude self and own non-seizure
                other_dists.append(distance_matrix[idx, j])
        
        avg_other = np.mean(other_dists)
        separability = avg_other / within_dist if within_dist > 0 else float('inf')
        
        print(f"  {st_upper:6s}: {separability:.2f} (höher = besser separierbar)")
        print(f"    Within-type: {within_dist:.6f}")
        print(f"    To others:   {avg_other:.6f}")
    
    return results

def plot_distance_distributions(distance_matrix, labels, output_dir):
    """Plot distribution of distances."""
    print("\n" + "=" * 70)
    print("DISTANZ-VERTEILUNGEN")
    print("=" * 70)
    
    # Extract upper triangle (without diagonal)
    upper_tri_indices = np.triu_indices_from(distance_matrix, k=1)
    all_distances = distance_matrix[upper_tri_indices]
    
    # Categorize distances
    within_type = []
    between_type_sz = []
    between_type_nsz = []
    sz_vs_nsz_same = []
    sz_vs_nsz_diff = []
    
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            dist = distance_matrix[i, j]
            
            # Check if same seizure type
            type_i = labels[i].split('\n')[0]
            type_j = labels[j].split('\n')[0]
            same_type = (type_i == type_j)
            
            # Check if seizure or non-seizure
            is_sz_i = 'Non-Sz' not in labels[i]
            is_sz_j = 'Non-Sz' not in labels[j]
            
            if same_type:
                if is_sz_i != is_sz_j:
                    within_type.append(dist)
            else:
                if is_sz_i and is_sz_j:
                    between_type_sz.append(dist)
                elif not is_sz_i and not is_sz_j:
                    between_type_nsz.append(dist)
                else:
                    sz_vs_nsz_diff.append(dist)
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    data_to_plot = [
        within_type,
        between_type_sz,
        between_type_nsz,
        sz_vs_nsz_diff
    ]
    
    positions = [1, 2, 3, 4]
    labels_plot = [
        'Within Type\n(Sz vs Non-Sz)',
        'Between Types\n(Sz vs Sz)',
        'Between Types\n(Non-Sz vs Non-Sz)',
        'Cross\n(Sz vs Non-Sz\ndiff types)'
    ]
    
    parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, 
                          showmedians=True, widths=0.7)
    
    # Color the violins
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_plot, fontsize=10)
    ax.set_ylabel('Wasserstein Distance', fontsize=12)
    ax.set_title('Verteilung der Wasserstein-Distanzen nach Kategorie', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics
    for i, (data, pos) in enumerate(zip(data_to_plot, positions)):
        mean_val = np.mean(data)
        ax.text(pos, mean_val, f'{mean_val:.5f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'distance_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Verteilungsplot gespeichert: {output_path}")
    plt.close()

def create_confusion_style_matrix(distance_matrix, labels, output_dir):
    """Create a confusion-matrix style visualization highlighting key comparisons."""
    print("\n" + "=" * 70)
    print("ERWEITERTE HEATMAP")
    print("=" * 70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # Left: Original with better annotations
    sns.heatmap(distance_matrix, annot=True, fmt='.6f', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels, ax=ax1,
                cbar_kws={'label': 'Wasserstein Distance'})
    ax1.set_title('Wasserstein Distance Matrix\n(Absolute Werte)', 
                  fontsize=13, fontweight='bold')
    
    # Right: Normalized (relative to row mean)
    row_means = distance_matrix.mean(axis=1, keepdims=True)
    normalized = distance_matrix / row_means
    
    sns.heatmap(normalized, annot=True, fmt='.2f', cmap='RdBu_r', center=1.0,
                xticklabels=labels, yticklabels=labels, ax=ax2,
                cbar_kws={'label': 'Relative Distance (÷ Zeilenmittelwert)'})
    ax2.set_title('Normalisierte Distanz-Matrix\n(> 1 = überdurchschnittlich weit)', 
                  fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'wasserstein_enhanced_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Erweiterte Heatmap gespeichert: {output_path}")
    plt.close()

def main():
    print("=" * 70)
    print("STATISTISCHE ANALYSE DER WASSERSTEIN-DISTANZEN")
    print("=" * 70)
    
    # Load data
    print("\nDATEN LADEN...")
    datasets = {}
    
    for seizure_type in SEIZURE_TYPES:
        data = load_eeg_data(seizure_type)
        datasets[f'{seizure_type}_seizure'] = prepare_samples(data['seizure_data'], MAX_SAMPLES)
        datasets[f'{seizure_type}_non_seizure'] = prepare_samples(data['non_seizure_data'], MAX_SAMPLES)
    
    # Create labels
    labels = []
    dataset_names = []
    for st in SEIZURE_TYPES:
        labels.append(f'{st.upper()}\nSeizure')
        labels.append(f'{st.upper()}\nNon-Sz')
        dataset_names.append(f'{st}_seizure')
        dataset_names.append(f'{st}_non_seizure')
    
    # Calculate distance matrix
    print("\nBERECHNE DISTANZ-MATRIX...")
    n = len(labels)
    distance_matrix = np.zeros((n, n))
    
    total_pairs = n * (n + 1) // 2
    with tqdm(total=total_pairs, desc="Distanzen") as pbar:
        for i in range(n):
            for j in range(i, n):
                dist = calculate_energy_distance_multidim(
                    datasets[dataset_names[i]], 
                    datasets[dataset_names[j]]
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                pbar.update(1)
    
    # Perform analyses
    analyze_distance_statistics(distance_matrix, labels, dataset_names)
    plot_hierarchical_clustering(distance_matrix, labels, OUTPUT_DIR)
    plot_mds(distance_matrix, labels, OUTPUT_DIR)
    plot_distance_distributions(distance_matrix, labels, OUTPUT_DIR)
    create_confusion_style_matrix(distance_matrix, labels, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("✓ ANALYSE ABGESCHLOSSEN!")
    print("=" * 70)
    print(f"\nAlle Plots gespeichert in: {OUTPUT_DIR}")

if __name__ == '__main__':
    np.random.seed(42)
    main()
