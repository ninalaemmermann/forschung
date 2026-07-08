from data_loader import load_eeg_data
from histogram_aggregator import HistogramAggregator
from plotter_3d import plot_3d_scatter
import sys
sys.path.append('/home/data/ninalaemmermann/forschung/eeg-channel-histogram-aggregator')
from config.settings import INPUT_DATA_PATH


def main():
    """
    Hauptprogramm: Erstellt 27D-Histogram und 3D-Visualisierung
    """
    print(f"\n{'='*60}")
    print(f"EEG CHANNEL HISTOGRAM AGGREGATOR")
    print(f"{'='*60}\n")
    
    # Konfiguration
    SEIZURE_TYPE = 'ABSZ'  # ← Ändern Sie hier: 'CPSZ', 'ABSZ', 'FNSZ', oder 'GNSZ'
    
    # SCHRITT 1: Load EEG data
    eeg_data = load_eeg_data(INPUT_DATA_PATH)
    
    if eeg_data is None:
        print("✗ Abbruch: Daten konnten nicht geladen werden")
        return
    
    # SCHRITT 2: Nutze ALLE Kanäle
    n_channels = len(eeg_data['channel_names'])
    print(f"Nutze alle {n_channels} Kanäle: {eeg_data['channel_names']}\n")
    
    selected_data = {
        'channel_names': eeg_data['channel_names'],
        'seizure_data': eeg_data['seizure_data'],
        'non_seizure_data': eeg_data['non_seizure_data']
    }
    
    # SCHRITT 3: Erstelle multidimensionales Histogram
    histogram_aggregator = HistogramAggregator(selected_data)
    
    multidim_histogram = histogram_aggregator.create_multidimensional_histogram(
        n_bins_per_dim=50, 
        max_samples=100000
    )
    
    # SCHRITT 4: Create 3D scatter plot visualization
    print(f"\n{'='*60}")
    print(f"ERSTELLE 3D-VISUALISIERUNG...")
    print(f"{'='*60}")
    
    sample_matrix = histogram_aggregator.get_sample_matrix(max_samples_per_class=50000)
    
    plot_3d_scatter(sample_matrix, seizure_type=SEIZURE_TYPE, n_channels=n_channels)
    

if __name__ == "__main__":
    main()