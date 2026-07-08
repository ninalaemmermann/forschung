"""
Apply PCA dimensionality reduction to EEG data before histogram comparison.
Reduces 27 channels to a smaller number of principal components.
"""

import pickle
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

# Configuration
N_COMPONENTS = 3  # Reduce to 3 dimensions
MAX_SAMPLES = 100000  # Maximum samples to use for PCA (per seizure type & category)
SEIZURE_TYPES = ['absz', 'cpsz', 'fnsz', 'gnsz']
BASE_PATH = Path('/home/data/ninalaemmermann/forschung')
OUTPUT_DIR = BASE_PATH / 'eeg-channel-histogram-aggregator' / 'data' / 'output'

def load_eeg_data(seizure_type):
    """Load EEG data from pickle file."""
    pkl_path = BASE_PATH / f'eeg_results_{seizure_type}.pkl'
    print(f"Lade {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def prepare_samples(data_array, max_samples=MAX_SAMPLES):
    """
    Convert (27, n_timepoints) to (n_samples, 27) by transposing and sampling.
    
    Args:
        data_array: Array of shape (27, n_timepoints) or list of 27 channel arrays
        max_samples: Maximum number of samples to extract
    
    Returns:
        samples: (n_samples, 27) array
    """
    # Convert to numpy array and ensure shape (27, n_timepoints)
    if isinstance(data_array, list):
        data_array = np.array(data_array)
    
    # Transpose: (27, n_timepoints) → (n_timepoints, 27)
    samples = data_array.T
    
    # Sample if too many datapoints
    if samples.shape[0] > max_samples:
        indices = np.random.choice(samples.shape[0], max_samples, replace=False)
        samples = samples[indices]
    
    return samples

def apply_pca_to_samples(samples, pca=None, fit=True):
    """
    Apply PCA to sample matrix.
    
    Args:
        samples: (n_samples, 27) array
        pca: Existing PCA object (if fit=False)
        fit: Whether to fit PCA or just transform
    
    Returns:
        transformed_samples: (n_samples, n_components) array
        pca: PCA object
    """
    if fit:
        pca = PCA(n_components=N_COMPONENTS)
        transformed = pca.fit_transform(samples)
        print(f"  PCA fitted: {samples.shape[1]} → {N_COMPONENTS} Dimensionen")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        transformed = pca.transform(samples)
    
    return transformed, pca

def process_all_seizure_types():
    """Process all seizure types with PCA."""
    
    # Step 1: Fit PCA on combined data from all seizure types (for consistent transformation)
    print("=" * 70)
    print("SCHRITT 1: PCA auf kombinierten Daten fitten")
    print("=" * 70)
    
    all_samples = []
    for seizure_type in SEIZURE_TYPES:
        data = load_eeg_data(seizure_type)
        
        # Prepare samples: (27, n_timepoints) → (n_samples, 27)
        seizure_samples = prepare_samples(data['seizure_data'])
        non_seizure_samples = prepare_samples(data['non_seizure_data'])
        
        all_samples.append(seizure_samples)
        all_samples.append(non_seizure_samples)
        
        print(f"  {seizure_type.upper()}: {seizure_samples.shape[0]} seizure + {non_seizure_samples.shape[0]} non-seizure samples")
    
    # Combine all samples for PCA fitting
    combined_samples = np.vstack(all_samples)
    print(f"\nKombiniert: {combined_samples.shape[0]} samples, {combined_samples.shape[1]} channels")
    
    # Fit PCA on combined data
    print("\nFitte PCA...")
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(combined_samples)
    
    print(f"\nPCA Results:")
    print(f"  Komponenten: {N_COMPONENTS}")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Kumulative variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Step 2: Transform each seizure type with fitted PCA
    print("\n" + "=" * 70)
    print("SCHRITT 2: Daten mit PCA transformieren")
    print("=" * 70)
    
    pca_results = {
        'pca_model': pca,
        'n_components': N_COMPONENTS,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'seizure_types': {}
    }
    
    for seizure_type in SEIZURE_TYPES:
        print(f"\nTransformiere {seizure_type.upper()}...")
        data = load_eeg_data(seizure_type)
        
        # Prepare and transform seizure samples
        seizure_samples = prepare_samples(data['seizure_data'])
        seizure_pca, _ = apply_pca_to_samples(seizure_samples, pca=pca, fit=False)
        print(f"  Seizure: {seizure_samples.shape} → {seizure_pca.shape}")
        
        # Prepare and transform non-seizure samples
        non_seizure_samples = prepare_samples(data['non_seizure_data'])
        non_seizure_pca, _ = apply_pca_to_samples(non_seizure_samples, pca=pca, fit=False)
        print(f"  Non-seizure: {non_seizure_samples.shape} → {non_seizure_pca.shape}")
        
        # Store transformed data
        pca_results['seizure_types'][seizure_type] = {
            'seizure': seizure_pca,
            'non_seizure': non_seizure_pca,
            'original_shape': seizure_samples.shape,
            'channel_names': data.get('channel_names', None)
        }
    
    # Step 3: Save results
    print("\n" + "=" * 70)
    print("SCHRITT 3: Ergebnisse speichern")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f'pca_data_{N_COMPONENTS}d.pkl'
    
    with open(output_path, 'wb') as f:
        pickle.dump(pca_results, f)
    
    print(f"\n✓ PCA-Daten gespeichert: {output_path}")
    print(f"  Größe: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)
    print(f"Original: 27 Kanäle → Reduziert: {N_COMPONENTS} Komponenten")
    print(f"Erklärte Varianz: {pca.explained_variance_ratio_.sum():.2%}")
    print("\nTransformierte Seizure-Typen:")
    for seizure_type in SEIZURE_TYPES:
        sz_shape = pca_results['seizure_types'][seizure_type]['seizure'].shape
        nsz_shape = pca_results['seizure_types'][seizure_type]['non_seizure'].shape
        print(f"  {seizure_type.upper()}: {sz_shape[0]} seizure, {nsz_shape[0]} non-seizure samples")
    
    print("\n✓ PCA abgeschlossen!")
    print(f"Nächster Schritt: Histogramme oder Wasserstein auf PCA-Daten anwenden")

if __name__ == '__main__':
    process_all_seizure_types()
