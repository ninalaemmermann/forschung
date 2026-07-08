import numpy as np
import sys
sys.path.append('/home/data/ninalaemmermann/forschung/eeg-channel-histogram-aggregator')
from config.settings import N_BINS, PERCENTILE_MIN, PERCENTILE_MAX

class HistogramAggregator:
    """
    Kombiniert Histogramme von mehreren Kanälen
    """
    
    def __init__(self, selected_data):
        """
        Parameters:
        -----------
        selected_data : dict
            Ausgewählte Kanal-Daten aus ChannelSelector
        """
        self.selected_data = selected_data
        self.channel_names = selected_data['channel_names']
        self.seizure_data = selected_data['seizure_data']
        self.non_seizure_data = selected_data['non_seizure_data']
    
    def aggregate_histograms(self):
        """
        OPTION A: Kombiniert ALLE Rohdaten von allen Kanälen
        
        Returns:
        --------
        dict : Kombinierte Daten und Histogramme
        """
        print(f"\n{'='*60}")
        print(f"KOMBINIERE ROHDATEN VON {len(self.channel_names)} KANÄLEN")
        print(f"{'='*60}")
        
        # SCHRITT 1: Alle Seizure-Samples von allen Kanälen sammeln
        print("\nSchritt 1: Sammle alle Seizure-Samples...")
        all_seizure_samples = []
        
        for i, ch_name in enumerate(self.channel_names):
            ch_samples = self.seizure_data[i]
            all_seizure_samples.extend(ch_samples)
            print(f"  {ch_name}: {len(ch_samples):,} Samples hinzugefügt")
        
        all_seizure_samples = np.array(all_seizure_samples)
        print(f"\n✓ Total Seizure-Samples: {len(all_seizure_samples):,}")
        
        # SCHRITT 2: Alle Non-Seizure-Samples von allen Kanälen sammeln
        print("\nSchritt 2: Sammle alle Non-Seizure-Samples...")
        all_non_seizure_samples = []
        
        for i, ch_name in enumerate(self.channel_names):
            ch_samples = self.non_seizure_data[i]
            all_non_seizure_samples.extend(ch_samples)
            print(f"  {ch_name}: {len(ch_samples):,} Samples hinzugefügt")
        
        all_non_seizure_samples = np.array(all_non_seizure_samples)
        print(f"\n✓ Total Non-Seizure-Samples: {len(all_non_seizure_samples):,}")
        
        # SCHRITT 3: Globale Range berechnen (99% Perzentile)
        print("\nSchritt 3: Berechne globale Range...")
        all_data = np.concatenate([all_seizure_samples, all_non_seizure_samples])
        
        global_min = np.percentile(all_data, PERCENTILE_MIN)
        global_max = np.percentile(all_data, PERCENTILE_MAX)
        
        print(f"  Globale Range: {global_min:.6f} bis {global_max:.6f}")
        print(f"  Bereichsgröße: {global_max - global_min:.8f}")
        
        # SCHRITT 4: Histogramme erstellen
        print(f"\nSchritt 4: Erstelle Histogramme mit {N_BINS} Bins...")
        
        hist_range = (global_min, global_max)
        
        # Seizure Histogram
        seizure_hist, seizure_bins = np.histogram(
            all_seizure_samples,
            bins=N_BINS,
            range=hist_range,
            density=True
        )
        
        # Non-Seizure Histogram
        non_seizure_hist, non_seizure_bins = np.histogram(
            all_non_seizure_samples,
            bins=N_BINS,
            range=hist_range,
            density=True
        )
        
        print(f"✓ Histogramme erstellt")
        
        # SCHRITT 5: Statistiken berechnen
        print("\nSchritt 5: Berechne Statistiken...")
        
        stats = {
            'seizure': {
                'n_samples': len(all_seizure_samples),
                'mean': np.mean(all_seizure_samples),
                'std': np.std(all_seizure_samples),
                'min': np.min(all_seizure_samples),
                'max': np.max(all_seizure_samples),
            },
            'non_seizure': {
                'n_samples': len(all_non_seizure_samples),
                'mean': np.mean(all_non_seizure_samples),
                'std': np.std(all_non_seizure_samples),
                'min': np.min(all_non_seizure_samples),
                'max': np.max(all_non_seizure_samples),
            }
        }
        
        print(f"\n  Seizure:")
        print(f"    Samples: {stats['seizure']['n_samples']:,}")
        print(f"    Mean: {stats['seizure']['mean']:.6f}")
        print(f"    Std: {stats['seizure']['std']:.6f}")
        
        print(f"\n  Non-Seizure:")
        print(f"    Samples: {stats['non_seizure']['n_samples']:,}")
        print(f"    Mean: {stats['non_seizure']['mean']:.6f}")
        print(f"    Std: {stats['non_seizure']['std']:.6f}")
        
        print(f"\n{'='*60}")
        print(f"✓ AGGREGATION ABGESCHLOSSEN")
        print(f"{'='*60}\n")
        
        return {
            'channel_names': self.channel_names,
            'seizure_histogram': seizure_hist,
            'non_seizure_histogram': non_seizure_hist,
            'bins': seizure_bins,
            'global_min': global_min,
            'global_max': global_max,
            'statistics': stats,
            'all_seizure_samples': all_seizure_samples,
            'all_non_seizure_samples': all_non_seizure_samples
        }
    
    def create_multidimensional_histogram(self, n_bins_per_dim=50, max_samples=100000):
        """
        Erstellt ein multidimensionales Histogram (15D) mit Sparse-Speicherung
        
        Parameters:
        -----------
        n_bins_per_dim : int
            Anzahl Bins pro Dimension (default: 50)
        max_samples : int
            Maximale Anzahl Samples für Berechnung (für Performance)
        
        Returns:
        --------
        dict : {
            'seizure_hist': dict (bin_tuple -> count),
            'non_seizure_hist': dict (bin_tuple -> count),
            'bin_edges': list of arrays (Bin-Grenzen pro Dimension),
            'n_bins': int,
            'n_dimensions': int,
            'channel_names': list
        }
        """
        print(f"\n{'='*60}")
        print(f"ERSTELLE {len(self.channel_names)}D-HISTOGRAM MIT {n_bins_per_dim} BINS/DIM")
        print(f"{'='*60}")
        
        n_channels = len(self.channel_names)
        
        # Schritt 1: Bin-Grenzen für jede Dimension berechnen
        print(f"\nSchritt 1: Berechne Bin-Grenzen für {n_channels} Dimensionen...")
        bin_edges = []
        
        for i in range(n_channels):
            # Kombiniere Seizure und Non-Seizure für globale Range
            all_samples = np.concatenate([self.seizure_data[i], self.non_seizure_data[i]])
            min_val = np.percentile(all_samples, 0.5)
            max_val = np.percentile(all_samples, 99.5)
            edges = np.linspace(min_val, max_val, n_bins_per_dim + 1)
            bin_edges.append(edges)
            print(f"  {self.channel_names[i]}: Range [{min_val:.6f}, {max_val:.6f}]")
        
        # Schritt 2: Seizure Histogram (Sparse)
        print(f"\nSchritt 2: Berechne Seizure-Histogram...")
        seizure_matrix = self._create_sample_matrix_internal(
            self.seizure_data, n_channels, max_samples
        )
        seizure_hist = self._compute_sparse_histogram(
            seizure_matrix, bin_edges, n_bins_per_dim
        )
        print(f"   Besetzte Bins: {len(seizure_hist):,} von {n_bins_per_dim**n_channels:.2e} möglichen")
        
        # Schritt 3: Non-Seizure Histogram (Sparse)
        print(f"\nSchritt 3: Berechne Non-Seizure-Histogram...")
        non_seizure_matrix = self._create_sample_matrix_internal(
            self.non_seizure_data, n_channels, max_samples
        )
        non_seizure_hist = self._compute_sparse_histogram(
            non_seizure_matrix, bin_edges, n_bins_per_dim
        )
        print(f"  ✓ Besetzte Bins: {len(non_seizure_hist):,} von {n_bins_per_dim**n_channels:.2e} möglichen")
        
        print(f"\n{'='*60}")
        print(f"✓ {n_channels}D-HISTOGRAM ERSTELLT")
        print(f"{'='*60}\n")
        
        return {
            'seizure_hist': seizure_hist,
            'non_seizure_hist': non_seizure_hist,
            'bin_edges': bin_edges,
            'n_bins': n_bins_per_dim,
            'n_dimensions': n_channels,
            'channel_names': self.channel_names
        }
    
    def _create_sample_matrix_internal(self, data_list, n_channels, max_samples):
        """Hilfsfunktion: Erstellt Sample-Matrix aus Daten"""
        lengths = [len(data_list[i]) for i in range(n_channels)]
        min_samples = min(lengths)
        n_samples = min(min_samples, max_samples)
        
        matrix = np.zeros((n_samples, n_channels))
        for i in range(n_channels):
            indices = np.random.choice(len(data_list[i]), n_samples, replace=False)
            matrix[:, i] = data_list[i][indices]
        
        return matrix
    
    def _compute_sparse_histogram(self, sample_matrix, bin_edges, n_bins):
        """
        Hilfsfunktion: Berechnet Sparse-Histogram
        
        Returns dict: {bin_tuple -> count}
        """
        n_samples, n_dims = sample_matrix.shape
        hist_dict = {}
        
        for sample_idx in range(n_samples):
            # Finde Bin-Index für jede Dimension
            bin_indices = []
            for dim in range(n_dims):
                value = sample_matrix[sample_idx, dim]
                # np.digitize gibt 1-based index, wir wollen 0-based
                bin_idx = np.digitize(value, bin_edges[dim]) - 1
                # Clipping für Edge-Cases
                bin_idx = max(0, min(n_bins - 1, bin_idx))
                bin_indices.append(bin_idx)
            
            # Tuple als Dictionary-Key
            bin_tuple = tuple(bin_indices)
            hist_dict[bin_tuple] = hist_dict.get(bin_tuple, 0) + 1
        
        return hist_dict

    
    def get_sample_matrix(self, max_samples_per_class=50000):
        """
        Erstellt eine Matrix mit per-sample Daten für 3D-Visualisierung
        
        Parameters:
        -----------
        max_samples_per_class : int
            Maximale Anzahl von Samples pro Klasse (für Performance)
        
        Returns:
        --------
        dict : {
            'seizure_matrix': array (n_samples, n_channels),
            'non_seizure_matrix': array (n_samples, n_channels),
            'channel_names': list
        }
        """
        print(f"\n{'='*60}")
        print(f"ERSTELLE SAMPLE-MATRIX FÜR 3D-VISUALISIERUNG")
        print(f"{'='*60}")
        
        n_channels = len(self.channel_names)
        
        # Seizure-Matrix erstellen
        print("\nSchritt 1: Erstelle Seizure-Matrix...")
        seizure_lengths = [len(self.seizure_data[i]) for i in range(n_channels)]
        min_seizure_samples = min(seizure_lengths)
        n_seizure_samples = min(min_seizure_samples, max_samples_per_class)
        
        seizure_matrix = np.zeros((n_seizure_samples, n_channels))
        
        for i in range(n_channels):
            # Zufällige Auswahl für Diversität
            indices = np.random.choice(len(self.seizure_data[i]), n_seizure_samples, replace=False)
            seizure_matrix[:, i] = self.seizure_data[i][indices]
            print(f"  {self.channel_names[i]}: {n_seizure_samples:,} Samples ausgewählt")
        
        print(f"\n Seizure-Matrix: {seizure_matrix.shape}")
        
        # Non-Seizure-Matrix erstellen
        print("\nSchritt 2: Erstelle Non-Seizure-Matrix...")
        non_seizure_lengths = [len(self.non_seizure_data[i]) for i in range(n_channels)]
        min_non_seizure_samples = min(non_seizure_lengths)
        n_non_seizure_samples = min(min_non_seizure_samples, max_samples_per_class)
        
        non_seizure_matrix = np.zeros((n_non_seizure_samples, n_channels))
        
        for i in range(n_channels):
            # Zufällige Auswahl für Diversität
            indices = np.random.choice(len(self.non_seizure_data[i]), n_non_seizure_samples, replace=False)
            non_seizure_matrix[:, i] = self.non_seizure_data[i][indices]
            print(f"  {self.channel_names[i]}: {n_non_seizure_samples:,} Samples ausgewählt")
        
        print(f"\n Non-Seizure-Matrix: {non_seizure_matrix.shape}")
        
        return {
            'seizure_matrix': seizure_matrix,
            'non_seizure_matrix': non_seizure_matrix,
            'channel_names': self.channel_names
        }