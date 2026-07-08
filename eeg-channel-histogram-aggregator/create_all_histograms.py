"""
Erstellt 27D-Histogramme für alle Seizure-Types und speichert sie als PKL
"""
import pickle
import sys
import os
sys.path.append('/home/data/ninalaemmermann/forschung/eeg-channel-histogram-aggregator')

from src.data_loader import load_eeg_data
from src.histogram_aggregator import HistogramAggregator


def create_histogram_for_type(seizure_type):
    """
    Erstellt 27D-Histogram für einen Seizure-Type
    
    Parameters:
    -----------
    seizure_type : str
        'ABSZ', 'CPSZ', 'FNSZ', oder 'GNSZ'
    """
    print(f"\n{'='*60}")
    print(f"ERSTELLE HISTOGRAM FÜR {seizure_type}")
    print(f"{'='*60}\n")
    
    # Pfad zur EEG-Daten-Datei
    base_dir = '/home/data/ninalaemmermann/forschung'
    input_file = os.path.join(base_dir, f'eeg_results_{seizure_type.lower()}.pkl')
    
    # Prüfe ob Datei existiert
    if not os.path.exists(input_file):
        print(f"✗ Datei nicht gefunden: {input_file}")
        print(f"  Überspringe {seizure_type}\n")
        return None
    
    # Lade Daten
    print(f"Lade Daten aus: {input_file}")
    with open(input_file, 'rb') as f:
        eeg_data = pickle.load(f)
    
    n_channels = len(eeg_data['channel_names'])
    print(f"✓ {n_channels} Kanäle geladen\n")
    
    # Bereite Daten vor
    selected_data = {
        'channel_names': eeg_data['channel_names'],
        'seizure_data': eeg_data['seizure_data'],
        'non_seizure_data': eeg_data['non_seizure_data']
    }
    
    # Erstelle Histogram
    histogram_aggregator = HistogramAggregator(selected_data)
    multidim_histogram = histogram_aggregator.create_multidimensional_histogram(
        n_bins_per_dim=50, 
        max_samples=100000
    )
    
    # Füge Metadaten hinzu
    multidim_histogram['seizure_type'] = seizure_type
    
    # Speichere als PKL
    output_file = os.path.join(base_dir, f'histogram_27d_{seizure_type.lower()}.pkl')
    print(f"\nSpeichere Histogram...")
    with open(output_file, 'wb') as f:
        pickle.dump(multidim_histogram, f)
    
    print(f"✓ Gespeichert: {output_file}")
    
    # Statistiken ausgeben
    n_seizure_bins = len(multidim_histogram['seizure_hist'])
    n_non_seizure_bins = len(multidim_histogram['non_seizure_hist'])
    print(f"\nStatistiken:")
    print(f"  Seizure Bins: {n_seizure_bins:,}")
    print(f"  Non-Seizure Bins: {n_non_seizure_bins:,}")
    print(f"  Dimensionen: {multidim_histogram['n_dimensions']}")
    print(f"  Bins pro Dimension: {multidim_histogram['n_bins']}")
    
    return multidim_histogram


def main():
    """
    Erstellt Histogramme für alle Seizure-Types
    """
    print(f"\n{'='*60}")
    print(f"HISTOGRAM GENERATOR FÜR ALLE SEIZURE-TYPES")
    print(f"{'='*60}\n")
    
    seizure_types = ['ABSZ', 'CPSZ', 'FNSZ', 'GNSZ']
    
    results = {}
    
    for seizure_type in seizure_types:
        histogram = create_histogram_for_type(seizure_type)
        if histogram is not None:
            results[seizure_type] = histogram
    
    print(f"\n{'='*60}")
    print(f"✓ FERTIG!")
    print(f"{'='*60}")
    print(f"\nErstellte Histogramme: {len(results)}/{len(seizure_types)}")
    for st in results.keys():
        print(f"  ✓ {st}")
    print()


if __name__ == "__main__":
    main()
