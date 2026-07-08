import numpy as np
import os
from scipy import stats
import mne
import pickle
import gc
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import pandas as pd


def force_garbage_collection():
    """Force garbage collection to free memory."""
    gc.collect()


def get_sz_start_end(file):
    '''
    Get sz start and sz end per patient from csv file
    file: csv file
    return: dictionary with patient as key and sz start and sz end as value
    '''
    df = pd.read_csv(file, encoding = 'utf-8', sep=";", usecols = ['BName','Sz start', 'Sz stop'])
    sz_start_end = {}
    for i in range(len(df)):
        patient = df['BName'][i]
        sz_start = df['Sz start'][i]
        sz_end = df['Sz stop'][i]
        if patient in sz_start_end:
            sz_start_end[patient].append((sz_start, sz_end))
        else:
            sz_start_end[patient] = [(sz_start, sz_end)]
    return sz_start_end


def analyze_eeg_distributions_from_files(seizure_dict, data_folder, seizure_type,
                                        max_samples_per_channel=500000, preserve_distribution=True):
    """
    Minimal, GNSZ-only version of the distribution extractor.

    Keeps only what's required to collect seizure / non-seizure samples per channel
    and compute per-channel summary statistics (means, std, median, min, max, KS for small sets).
    """
    # Sammle alle Daten über alle Patienten
    all_seizure_data = []  # Liste für jeden Kanal
    all_non_seizure_data = []
    all_channel_names = None
    patient_info = []

    if not seizure_dict:
        print(f"WARNUNG: Keine Seizure-Daten für {seizure_type} gefunden!")
        return {'channel_names': [], 'seizure_data': [], 'non_seizure_data': [], 'patient_info': [], 'statistics': []}

    print(f"Verarbeite {len(seizure_dict)} Patienten für {seizure_type}...")

    for i, (patient, sz_start_end) in enumerate(seizure_dict.items()):
        file_path = os.path.join(data_folder, f"{patient}_res.OWN11101_filtWB_avg.edf")
        try:
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            sampling_rate = raw.info['sfreq']

            # Load ALL data without subsampling
            signal = raw.get_data()
            times = raw.times
            n_channels, n_samples = signal.shape
            
            print(f"  Patient {i+1}: {patient} - {n_channels} Kanäle, {n_samples} Samples")

            if all_channel_names is None:
                all_channel_names = [f"Ch_{i+1}" for i in range(n_channels)]
                all_seizure_data = [[] for _ in range(n_channels)]
                all_non_seizure_data = [[] for _ in range(n_channels)]

            seizure_mask = np.zeros(n_samples, dtype=bool)
            for start_time, end_time in sz_start_end:
                start_idx = int(max(0, start_time * sampling_rate))
                end_idx = int(min(n_samples, end_time * sampling_rate))
                if start_idx < end_idx:
                    seizure_mask[start_idx:end_idx] = True

            non_seizure_mask = ~seizure_mask

            for ch_idx in range(n_channels):
                # KEINE Konvertierung - pure Rohdaten
                channel_data = signal[ch_idx, :]
                
                # Speichere ALLE Daten ohne Subsampling
                if seizure_mask.any():
                    s_data = channel_data[seizure_mask]
                    all_seizure_data[ch_idx].extend(s_data)
                    
                if non_seizure_mask.any():
                    ns_data = channel_data[non_seizure_mask]
                    all_non_seizure_data[ch_idx].extend(ns_data)

            del signal
            del raw
            force_garbage_collection()

            patient_info.append({'patient': patient, 'n_channels': n_channels, 'n_samples': n_samples,
                                 'duration_sec': times[-1] if len(times) > 0 else 0,
                                 'sampling_rate': sampling_rate, 'n_seizures': len(sz_start_end)})

        except Exception as e:
            print(f"Fehler bei Patient {patient}: {e}")
            continue

    results = {'channel_names': all_channel_names,
               'seizure_data': [np.array(d) for d in all_seizure_data],  # Originaler Datentyp
               'non_seizure_data': [np.array(d) for d in all_non_seizure_data],  # Originaler Datentyp
               'patient_info': patient_info,
               'statistics': []}

    # compute basic stats per channel
    if all_channel_names is None:
        return results

    for ch_idx, ch_name in enumerate(all_channel_names):
        s = results['seizure_data'][ch_idx]
        ns = results['non_seizure_data'][ch_idx]
        stats_dict = {'channel': ch_name, 'channel_index': ch_idx,
                      'n_seizure_samples': len(s), 'n_non_seizure_samples': len(ns)}

        if len(s) > 0:
            stats_dict.update({'seizure_mean': float(np.mean(s)), 'seizure_std': float(np.std(s)),
                                'seizure_median': float(np.median(s)), 'seizure_min': float(np.min(s)),
                                'seizure_max': float(np.max(s))})
        else:
            stats_dict.update({'seizure_mean': np.nan, 'seizure_std': np.nan, 'seizure_median': np.nan,
                                'seizure_min': np.nan, 'seizure_max': np.nan})

        if len(ns) > 0:
            stats_dict.update({'non_seizure_mean': float(np.mean(ns)), 'non_seizure_std': float(np.std(ns)),
                                'non_seizure_median': float(np.median(ns)), 'non_seizure_min': float(np.min(ns)),
                                'non_seizure_max': float(np.max(ns))})
        else:
            stats_dict.update({'non_seizure_mean': np.nan, 'non_seizure_std': np.nan, 'non_seizure_median': np.nan,
                                'non_seizure_min': np.nan, 'non_seizure_max': np.nan})

        stats_dict['ks_statistic'] = np.nan
        stats_dict['ks_p_value'] = np.nan
        if len(s) > 0 and len(ns) > 0 and len(s) < 50000 and len(ns) < 50000:
            try:
                ks_stat, ks_p = stats.ks_2samp(s, ns)
                stats_dict['ks_statistic'] = ks_stat
                stats_dict['ks_p_value'] = ks_p
            except Exception:
                pass

        results['statistics'].append(stats_dict)

    return results


# plotting and multi-type comparison utilities removed to keep file focused on GNSZ distribution extraction


def plot_channel_distributions(results, bins=50, figsize=(20, 30), max_samples=50000):
    """
    Plottet Histogramme für alle Kanäle - Verteilung über alle GNSZ-Dateien.
    
    Parameters:
    -----------
    results : dict
        Ergebnisse von analyze_eeg_distributions_from_files()
    bins : int
        Anzahl Bins für Histogramme
    figsize : tuple
        Figur-Größe
    max_samples : int
        Maximale Anzahl Samples für Plotting (für Performance)
    """
    
    if not results or not results.get('channel_names'):
        print("Keine Daten zum Plotten vorhanden!")
        return
    
    channel_names = results['channel_names']
    n_channels = len(channel_names)
    
    # Layout: 3 Spalten
    n_cols = 3
    n_rows = int(np.ceil(n_channels / n_cols))
    
    # Globale x-Achsen-Grenzen bestimmen (über alle Kanäle)
    all_data = []
    for ch_idx in range(n_channels):
        all_data.extend(results['seizure_data'][ch_idx])
        all_data.extend(results['non_seizure_data'][ch_idx])
    
    if len(all_data) == 0:
        print("Keine Daten vorhanden!")
        return
    
    all_data = np.array(all_data)
    # Engere x-Achse: Schneide extreme Outliers ab
    global_min = np.percentile(all_data, 0.5)   # Untere 0.5% abschneiden
    global_max = np.percentile(all_data, 99.5)  # Obere 0.5% abschneiden
    
    print(f"Erstelle Plots für {n_channels} Kanäle...")
    print(f"Globaler Wertebereich (99% der Daten): {global_min:.6f} bis {global_max:.6f}")
    print(f"Bereichsgröße: {global_max - global_min:.8f}")
    print(f"Anzahl Bins: 200")
    print(f"Bin-Breite: {(global_max - global_min) / 200:.8f}")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('EEG-Amplitudenverteilungen aller Kanäle', fontsize=24)
    axes = axes.flatten() if n_channels > 1 else [axes]
     
    for ch_idx in range(n_channels):
        ax = axes[ch_idx]
        ch_name = channel_names[ch_idx]
        
        seizure_data = results['seizure_data'][ch_idx]
        non_seizure_data = results['non_seizure_data'][ch_idx]
        
        # Histogramme mit vielen Bins (schmale Balken) und fester Range
        hist_range = (global_min, global_max)
        n_bins = 200  # Viele Bins = schmale, detaillierte Balken
        
        if len(seizure_data) > 0:
            ax.hist(seizure_data, bins=n_bins, range=hist_range, alpha=0.6, color='red', 
                   label=f'Anfall (n={len(seizure_data):,})', density=True)
        
        if len(non_seizure_data) > 0:
            ax.hist(non_seizure_data, bins=n_bins, range=hist_range, alpha=0.6, color='blue', 
                   label=f'Kein Anfall (n={len(non_seizure_data):,})', density=True)
        
        ax.set_title(f'{ch_name}', fontsize=13)
        ax.set_xlabel('Amplitude', fontsize=13)
        ax.set_ylabel('Dichte', fontsize=13)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Nur 5 statt 10+ Ticks
        ax.set_xlim(global_min, global_max)
        ax.tick_params(labelsize=10)

        # Formatiere mit wissenschaftlicher Notation oder weniger Dezimalstellen
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))  # Wissenschaftliche Notation für kleine Werte
        ax.xaxis.set_major_formatter(formatter)
    
    # Leere Subplots ausblenden
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Speichern
    #output_file = 'gnsz_channel_distributions_all.png'
    #output_file = 'fnsz_channel_distributions_all.png'
    output_file = 'cpsz_channel_distributions_all.png'
    #output_file = 'gnsz_channel_distributions_all.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot gespeichert: {output_file}")
    plt.show()
    
    return fig


def print_summary_report(results):
    """Print a compact summary report (for GNSZ-only results)."""
    if not results or not results.get('statistics'):
        print("Keine Statistikdaten vorhanden.")
        return
    for s in results['statistics'][:10]:
        print(f"{s['channel']}: n_seizure={s['n_seizure_samples']}, n_nonseizure={s['n_non_seizure_samples']}, seizure_mean={s.get('seizure_mean')}")


def process_gnsz_only(base_data_folder, csv_file=None):
    """Wrapper to process only GNSZ seizure type and save results."""
    # if csv_file is None:
    #     csv_file = os.path.join(base_data_folder, 'GNSZ_seizures.csv')
    # data_folder = os.path.join(base_data_folder, 'Data', 'GNSZ_seizure_WB')

    # seizure_dict = get_sz_start_end(csv_file)
    # results = analyze_eeg_distributions_from_files(seizure_dict, data_folder, seizure_type='CPSZ')


    # # FNSZ
    # if csv_file is None:
    #     csv_file = os.path.join(base_data_folder, 'FNSZ_seizures.csv')
    # data_folder = os.path.join(base_d5ata_folder, 'Data', 'FNSZ_seizure_WB')

    # seizure_dict = get_sz_start_end(csv_file)
    # results = analyze_eeg_distributions_from_files(seizure_dict, data_folder, seizure_type='FNSZ')

    # CPSZ
    if csv_file is None:
        csv_file = os.path.join(base_data_folder, 'CPSZ_seizures.csv')
    data_folder = os.path.join(base_data_folder, 'Data', 'CPSZ_seizure_WB')

    seizure_dict = get_sz_start_end(csv_file)
    results = analyze_eeg_distributions_from_files(seizure_dict, data_folder, seizure_type='CPSZ')

    #out_file = 'eeg_results_gnsz.pkl'
    out_file = 'eeg_results_gnsz.pkl'
    #out_file = 'eeg_results_fnsz.pkl'
    #out_file = 'eeg_results_cpsz.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Plot erstellen
    plot_channel_distributions(results)

    # Kurzer Bericht
    print_summary_report(results)
    
    return results

if __name__ == "__main__":
    # Adjust base path as needed
    base = r"/home/data/ninalaemmermann/forschung"
    #process_gnsz_only(base)

    pkl_file = 'eeg_results_cpsz.pkl'
    
    print(f"Lade gespeicherte Ergebnisse aus {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
    
    # print(f"Daten geladen: {len(results['channel_names'])} Kanäle")
    
    # # Nur Plots erstellen
    # print("\nErstelle Verteilungsplots für alle Kanäle...")
    plot_channel_distributions(results)
    