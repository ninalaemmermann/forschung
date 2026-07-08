import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from config.settings import FIGURE_SIZE, DPI, OUTPUT_DIR
import sys
sys.path.append('/home/data/ninalaemmermann/forschung/eeg-channel-histogram-aggregator')
import os

# Headless Backend für Server
matplotlib.use('Agg')

def plot_combined_histogram(combined_data, save_plot=True, seizure_type='CPSZ', n_channels=15):
    """
    Plottet das kombinierte Histogram aller ausgewählten Kanäle
    
    Parameters:
    -----------
    combined_data : dict
        Ergebnis von HistogramAggregator.aggregate_histograms()
    save_plot : bool
        Ob der Plot gespeichert werden soll
    """
    print("\nErstelle kombiniertes Histogram-Plot...")
    
    # Daten extrahieren
    seizure_hist = combined_data['seizure_histogram']
    non_seizure_hist = combined_data['non_seizure_histogram']
    bins = combined_data['bins']
    channel_names = combined_data['channel_names']
    stats = combined_data['statistics']
    
    # Bin-Zentren berechnen für plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot erstellen
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Seizure Histogram (rot)
    ax.fill_between(bin_centers, seizure_hist, alpha=0.5, color='red', 
                    label=f'Anfall (n={stats["seizure"]["n_samples"]:,})')
    ax.plot(bin_centers, seizure_hist, color='darkred', linewidth=2)
    
    # Non-Seizure Histogram (blau)
    ax.fill_between(bin_centers, non_seizure_hist, alpha=0.5, color='blue',
                    label=f'Kein Anfall (n={stats["non_seizure"]["n_samples"]:,})')
    ax.plot(bin_centers, non_seizure_hist, color='darkblue', linewidth=2)
    
    # Mittelwerte als vertikale Linien
    ax.axvline(stats['seizure']['mean'], color='red', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Anfall Mean: {stats["seizure"]["mean"]:.6f}')
    ax.axvline(stats['non_seizure']['mean'], color='blue', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Normal Mean: {stats["non_seizure"]["mean"]:.6f}')
    
    # Beschriftungen
    ax.set_xlabel('Amplitude', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dichte', fontsize=14, fontweight='bold')
    
    # Titel mit Kanal-Info
    title = f'Kombinierte EEG-Amplitudenverteilung\n'
    title += f'Top-{len(channel_names)} Kanäle: {", ".join(channel_names[:5])}...'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Legende
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Statistik-Textbox
    textstr = f'Seizure:\n'
    textstr += f'  Mean: {stats["seizure"]["mean"]:.6f}\n'
    textstr += f'  Std: {stats["seizure"]["std"]:.6f}\n\n'
    textstr += f'Non-Seizure:\n'
    textstr += f'  Mean: {stats["non_seizure"]["mean"]:.6f}\n'
    textstr += f'  Std: {stats["non_seizure"]["std"]:.6f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    title = f'Kombinierte EEG-Amplitudenverteilung ({seizure_type})\n'
    title += f'Top-{len(channel_names)} Kanäle: {", ".join(channel_names[:5])}...'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # ...existing code...
    
    # Speichern mit aussagekräftigem Dateinamen:
    if save_plot:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = os.path.join(OUTPUT_DIR, 
                               f'combined_histogram_{seizure_type}_top{n_channels}_channels.png')
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        print(f"✓ Plot gespeichert als: {filename}")
    
    plt.close()
    