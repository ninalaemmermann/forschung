import matplotlib
import numpy as np
from scipy.spatial.distance import jensenshannon
from Verteilungsfkt import get_sz_start_end
import mne
import pickle
import matplotlib.pyplot as plt
from datetime import datetime



def improved_calculate_kl_divergence(p, q, strategy='adaptive', verbose=False):
    """
    Verbesserte KL-Divergenz Berechnung mit stabilen Bins
    
    Args:
        p, q: Datenarrays zum Vergleichen  
        strategy: 'adaptive', 'fixed_robust', 'percentile_based'
        verbose: Debug-Ausgaben
    
    Returns:
        float: JS-Divergenz
    """
    
    try:
        # Kombiniere Daten für Bin-Berechnung
        all_data = np.concatenate([p, q])
        
        if len(all_data) == 0:
            return 0.0
        
        # Datenbereich analysieren
        data_min, data_max = np.min(all_data), np.max(all_data)
        data_range = data_max - data_min
        
        if data_range == 0:
            return 0.0  # Identische Daten
        
        if verbose:
            print(f"   Data range: [{data_min:.8f}, {data_max:.8f}] V")
            print(f"   Range span: {data_range:.8f} V")
        
        # Bin-Strategie wählen
        if strategy == 'adaptive':
            # Range-basierte adaptive Bins
            if data_range < 1e-6:  # Sehr kleine Range (< 1 µV)
                n_bins = 20
            elif data_range < 1e-4:  # Kleine Range (< 100 µV)
                n_bins = 30
            elif data_range < 1e-3:  # Mittlere Range (< 1 mV)
                n_bins = 50
            else:  # Große Range
                n_bins = 80
                
            if verbose:
                print(f"   Adaptive bins: {n_bins} (range={data_range:.8f})")
                
        elif strategy == 'fixed_robust':
            # Feste robuste Anzahl
            n_bins = 40
            
        elif strategy == 'percentile_based':
            # Robuste Grenzen mit Percentilen (weniger Outlier-sensitiv)
            p1, p99 = np.percentile(all_data, [1, 99])
            
            if p99 - p1 <= 0:
                return 0.0
                
            bins = np.linspace(p1, p99, 40)
            
            if verbose:
                print(f"   Percentile range: [{p1:.8f}, {p99:.8f}] V")
                print(f"   Bins: {len(bins)-1}")
            
            # Histogramme mit Percentile-Grenzen
            hist1, _ = np.histogram(p, bins=bins, density=True)
            hist2, _ = np.histogram(q, bins=bins, density=True)
            
            # Normalisieren für JS
            hist1_norm = hist1 / (np.sum(hist1) + 1e-10)
            hist2_norm = hist2 / (np.sum(hist2) + 1e-10)
            
            # Epsilon gegen Division durch 0
            epsilon = 1e-10
            hist1_norm += epsilon
            hist2_norm += epsilon
            
            js_div = jensenshannon(hist1_norm, hist2_norm)
            
            if verbose:
                print(f"   JS divergence: {js_div:.6f}")
            
            return js_div
            
        elif strategy == 'freedman_diaconis':
            # Freedman-Diaconis Regel für optimale Bin-Breite
            iqr = np.percentile(all_data, 75) - np.percentile(all_data, 25)
            
            if iqr <= 0:
                n_bins = 30  # Fallback
            else:
                bin_width = 2 * iqr / (len(all_data) ** (1/3))
                n_bins = int(data_range / bin_width)
                n_bins = max(10, min(100, n_bins))  # Begrenzen
                
            if verbose:
                print(f"   Freedman-Diaconis bins: {n_bins} (IQR={iqr:.8f})")
        
        else:
            n_bins = 50  # Default
        
        # Standard-Binning für nicht-percentile Strategien
        if strategy != 'percentile_based':
            bins = np.linspace(data_min, data_max, n_bins + 1)
            
            hist1, _ = np.histogram(p, bins=bins, density=True)
            hist2, _ = np.histogram(q, bins=bins, density=True)
            
            # Normalisieren für JS
            hist1_norm = hist1 / (np.sum(hist1) + 1e-10)
            hist2_norm = hist2 / (np.sum(hist2) + 1e-10)
            
            # Epsilon gegen Division durch 0
            epsilon = 1e-10
            hist1_norm += epsilon
            hist2_norm += epsilon
            
            js_div = jensenshannon(hist1_norm, hist2_norm)
            
            if verbose:
                print(f"   Bins: {n_bins}, Bin width: {data_range/n_bins:.8f} V")
                print(f"   JS divergence: {js_div:.6f}")
            
            return js_div
            
    except Exception as e:
        print(f"Error in improved JS calculation: {e}")
        return float('inf')

def analyze_channel_importance(results):
    """
    SETUP-FUNKTION: Berechnet Channel Importance (einmalig ausführen)
    
    Args:
        results: Dictionary mit 'seizure_data' und 'non_seizure_data'
    
    Returns:
        list: Channel Importance sortiert (beste zuerst)
    """
    n_channels = len(results['seizure_data'])
    importance_scores = []
    
    for ch_idx in range(n_channels):
        seizure_data = results['seizure_data'][ch_idx]
        non_seizure_data = results['non_seizure_data'][ch_idx]
        
        if len(seizure_data) == 0 or len(non_seizure_data) == 0:
            continue
        
        # KL-Divergenz zwischen Seizure und Non-Seizure Verteilungen
        kl_divergence = improved_calculate_kl_divergence(seizure_data, non_seizure_data, strategy='adaptive', verbose=False)
        
        # DEBUG: Ausgabe für ALLE Kanäle
        print(f"Ch_{ch_idx+1}: JS-Divergenz = {kl_divergence:.6f} "
              f"(Seizure samples: {len(seizure_data)}, Normal samples: {len(non_seizure_data)})")
        
        if not np.isnan(kl_divergence) and kl_divergence != float('inf'):
            importance_scores.append({
                'channel': f'Ch_{ch_idx+1}',
                'channel_idx': ch_idx,
                'separability_score': kl_divergence
            })
    
    # Nach Separability Score sortieren (höher = besser)
    importance_scores.sort(key=lambda x: x['separability_score'], reverse=True)
    
    print(f"\nTop 10 Kanäle nach JS-Divergenz:")
    for i, ch in enumerate(importance_scores[:10], 1):
        print(f"  {i}. {ch['channel']}: {ch['separability_score']:.6f}")
    
    return importance_scores

def detect_seizure_in_window(window_data, training_results, top_5_channels, bin_strategy='adaptive'):
    """
    Hauptfunktion: Detectiert Seizure in EEG-Amplituden eines 5-Sekunden-Fensters
    
    Parameters:
    -----------
    window_data : array
        EEG-Daten für alle Kanäle [n_channels, n_samples]
    training_results : dict
        Trainingsdaten mit seizure_data und non_seizure_data
    top_5_channels : list
        Liste der wichtigsten Kanäle
    bin_strategy : str
        Bin-Strategie: 'adaptive', 'fixed_robust', 'percentile_based', 'freedman_diaconis'
    """
    
    print(f" Analysiere EEG-Amplituden mit Top 5 Kanälen (Bin-Strategie: {bin_strategy})...")
    
    n_channels, n_samples = window_data.shape
    channel_scores = []
    
    # Für jeden der Top 5 Kanäle
    for i, ch_info in enumerate(top_5_channels):
        ch_idx = ch_info['channel_idx']
        ch_name = ch_info['channel']
        
        if ch_idx >= n_channels:
            print(f"    Kanal {ch_name} (Index {ch_idx}) nicht in EEG-Daten vorhanden")
            continue
        
        print(f"    Analysiere {ch_name} (Index {ch_idx})...")
        
        # Aktuelle EEG-Amplituden für diesen Kanal (alle Samples im 5s-Fenster)
        current_channel_data = window_data[ch_idx, :].copy()
        
        # Trainingsdaten für diesen Kanal
        seizure_ref = training_results['seizure_data'][ch_idx]
        non_seizure_ref = training_results['non_seizure_data'][ch_idx]
        
        if len(seizure_ref) == 0 or len(non_seizure_ref) == 0:
            print(f"       Keine Trainingsdaten für {ch_name}")
            continue
        
        print(f"      - Aktuelle Amplituden: {len(current_channel_data)} Samples")
        print(f"      - Amplitude Range: [{current_channel_data.min():.6f}, {current_channel_data.max():.6f}] Volt")
        print(f"      - Amplitude Mean±Std: {np.mean(current_channel_data):.8f}±{np.std(current_channel_data):.8f} Volt")
        print(f"      - Seizure Training: {len(seizure_ref)} Werte, Range: [{np.min(seizure_ref):.8f}, {np.max(seizure_ref):.8f}] Volt")
        print(f"      - Non-Seizure Training: {len(non_seizure_ref)} Werte, Range: [{np.min(non_seizure_ref):.8f}, {np.max(non_seizure_ref):.8f}] Volt")
        
        # JS-Divergenz berechnen: Beide Datenquellen verwenden jetzt MNE (konsistente Skalierung)
        # Die calculate_kl_divergence Funktion verwendet bereits density=True für robuste Vergleiche
        
        print(f"       DEBUG - Vor JS-Berechnung:")
        print(f"         Current samples: {len(current_channel_data)}")
        print(f"         Seizure ref samples: {len(seizure_ref)}")
        print(f"         Non-Seizure ref samples: {len(non_seizure_ref)}")
        
        # Prüfe ob Seizure und Non-Seizure identisch sind
        if np.array_equal(seizure_ref, non_seizure_ref):
            print(f"          PROBLEM: Seizure und Non-Seizure Trainingsdaten sind IDENTISCH!")
            js_to_seizure = 0.5
            js_to_non_seizure = 0.5
        else:
            print(f"          Trainingsdaten sind unterschiedlich")
            
            # HIER WIRD JS-DIVERGENZ BERECHNET:
            # Verwende verbesserte adaptive Bin-Strategie für robuste Berechnung
            js_to_seizure = improved_calculate_kl_divergence(
                current_channel_data, seizure_ref, 
                strategy=bin_strategy, verbose=True
            )
            js_to_non_seizure = improved_calculate_kl_divergence(
                current_channel_data, non_seizure_ref, 
                strategy=bin_strategy, verbose=True
            )
            
            # Zusätzlicher Test: JS zwischen Seizure und Non-Seizure direkt
            js_seizure_vs_non_seizure = improved_calculate_kl_divergence(
                seizure_ref, non_seizure_ref, 
                strategy=bin_strategy, verbose=False
            )
            print(f"          Baseline JS(Seizure vs Non-Seizure): {js_seizure_vs_non_seizure:.6f}")
            print(f"          Current JS(Current vs Seizure): {js_to_seizure:.6f}")
            print(f"          Current JS(Current vs Non-Seizure): {js_to_non_seizure:.6f}")
            
            if js_seizure_vs_non_seizure < 1e-6:
                print(f"          WARNUNG: Seizure und Non-Seizure sind sehr ähnlich!")
            
            # Interpretation der JS-Werte
            if js_to_seizure < js_to_non_seizure:
                similarity_interpretation = "ähnlicher zu Seizure"
            else:
                similarity_interpretation = "ähnlicher zu Non-Seizure"
            print(f"          Interpretation: Current Window ist {similarity_interpretation}")
        
        print(f"      - JS zu Seizure: {js_to_seizure:.6f}")
        print(f"      - JS zu Non-Seizure: {js_to_non_seizure:.6f}")
        print(f"      - Differenz: {abs(js_to_seizure - js_to_non_seizure):.6f}")
        
        if abs(js_to_seizure - js_to_non_seizure) < 1e-6:
            print(f"       PROBLEM: JS-Divergenzen sind praktisch identisch!")

        # Wahrscheinlichkeit für Seizure berechnen (je kleiner JS, desto ähnlicher)
        similarity_to_non_seizure = 1 - js_to_non_seizure
        similarity_to_seizure = 1 - js_to_seizure

        prob_seizure = similarity_to_non_seizure / (similarity_to_non_seizure + similarity_to_seizure)

        print(f"      - Seizure Wahrscheinlichkeit: {prob_seizure:.3f} ({prob_seizure*100:.1f}%)")

        
        channel_scores.append({
            'channel': ch_name,
            'channel_idx': ch_idx,
            'amplitude_mean': np.mean(current_channel_data),
            'amplitude_std': np.std(current_channel_data),
            'js_to_seizure': js_to_seizure,
            'js_to_non_seizure': js_to_non_seizure,
            'prob_seizure': prob_seizure,
            'separability_score': ch_info['separability_score']
        })
    
    if not channel_scores:
        return {
            'prediction': 'unknown',
            'prob_seizure': 0.5,
            'error': 'Keine gültigen Kanäle für Analyse'
        }
    
    # Finale Vorhersage: Gewichteter Durchschnitt
    print("\n Berechne finale Vorhersage...")
    
    # Gewichtung nach Channel Importance 
    total_weighted_prob = 0
    total_weight = 0
    
    for score in channel_scores:
        # Gewichtung: Separability Score 
        weight = score['separability_score']
        total_weighted_prob += score['prob_seizure'] * weight
        total_weight += weight
        
        print(f"   {score['channel']}: P={score['prob_seizure']:.3f} ")
    
    # Finale Wahrscheinlichkeit
    if total_weight > 0:
        final_prob_seizure = total_weighted_prob / total_weight
    else:
        final_prob_seizure = np.mean([s['prob_seizure'] for s in channel_scores])
    
    # Vorhersage
    prediction = 'seizure' if final_prob_seizure > 0.5 else 'non_seizure'
    
    print(f"\n Finale Vorhersage:")
    print(f"   Seizure Wahrscheinlichkeit: {final_prob_seizure:.3f} ({final_prob_seizure*100:.1f}%)")
    print(f"   Vorhersage: {prediction.upper()}")
    
    return {
        'prediction': prediction,
        'prob_seizure': final_prob_seizure,
        'prob_non_seizure': 1 - final_prob_seizure,
        'channel_details': channel_scores,
        'n_channels_analyzed': len(channel_scores)
    }

def analyze_full_edf_js_divergence(filename, window_duration=5.0, overlap=0.0, 
                                  bin_strategy='adaptive', save_results=True, seizure_type=None, channel_importance=None):
    """
    Analysiert eine komplette EDF-Datei in sliding windows und berechnet JS-Divergenz für Top 5 Kanäle
    
    Parameters:
    -----------
    filename : str
        Name der EDF-Datei (ohne Pfad)
    window_duration : float, default=5.0
        Fensterlänge in Sekunden  
    overlap : float, default=0.0
        Überlappung zwischen Fenstern in Sekunden
    bin_strategy : str, default='adaptive'
        Bin-Strategie für JS-Berechnung
    save_results : bool, default=True
        Ob Ergebnisse gespeichert werden sollen
        
    Returns:
    --------
    dict : Vollständige Analyseergebnisse
    """
    
    # SCHRITT 1: Trainingsdaten und Channel Importance laden
    print("\n SCHRITT 1: Trainingsdaten laden...")
    

    training_files = {
        'ABSZ': 'eeg_results_absz.pkl',
        'GNSZ': 'eeg_results_gnsz.pkl',
        'FNSZ': 'eeg_results_fnsz.pkl',
        'CPSZ': 'eeg_results_cpsz.pkl'
    }
    
    seizure_csv_files = {
        'ABSZ': '/home/data/ninalaemmermann/forschung/ABSZ_seizures.csv',
        'GNSZ': '/home/data/ninalaemmermann/forschung/GNSZ_seizures.csv',
        'FNSZ': '/home/data/ninalaemmermann/forschung/FNSZ_seizures.csv',
        'CPSZ': '/home/data/ninalaemmermann/forschung/CPSZ_seizures.csv'
    }
    
    data_directories = {
        'ABSZ': '/home/data/ninalaemmermann/forschung/Data/ABSZ_seizure_WB',
        'GNSZ': '/home/data/ninalaemmermann/forschung/Data/GNSZ_seizure_WB',
        'FNSZ': '/home/data/ninalaemmermann/forschung/Data/FNSZ_seizure_WB',
        'CPSZ': '/home/data/ninalaemmermann/forschung/Data/CPSZ_seizure_WB'
    }


    try:
        with open(training_files[seizure_type], 'rb') as f:
            training_results = pickle.load(f)
        print(f" Trainingsdaten für {seizure_type} erfolgreich geladen")
    except FileNotFoundError:
        print(f" FEHLER: {training_files[seizure_type]} nicht gefunden!")
        return None
    
    # Channel Importance berechnen
    if channel_importance is None:      
        importance = analyze_channel_importance(training_results)
    top_5_channels = importance[:5]

    print(f"Top 5 Kanäle für {seizure_type} (beste Separability):")
    for i, ch in enumerate(top_5_channels, 1):
        print(f"   {i}. {ch['channel']} (Index {ch['channel_idx']}): JS = {ch['separability_score']:.4f}")
    
    # SCHRITT 2: EDF-Datei laden und in Fenster einteilen
    print(f"\n SCHRITT 2: EDF-Datei laden und in {window_duration}s-Fenster einteilen...")
    
    # EDF-Datei aus dem richtigen Verzeichnis laden
    file_path = f"{data_directories[seizure_type]}/{filename}"
    raw = mne.io.read_raw_edf(file_path, preload=True)
    data = raw.get_data()  # Shape: (n_channels, n_samples)
    sampling_rate = raw.info['sfreq']
    
    n_channels, total_samples = data.shape
    duration_seconds = total_samples / sampling_rate
    
    # SCHRITT 3: Sliding Window Extraktion (adaptiert von Sz_ew_detection.py)
    print(f"\n SCHRITT 3: Sliding Window Extraktion...")
    
    step_size = window_duration - overlap
    num_windows = int((duration_seconds - window_duration) / step_size) + 1
    
    print(f"   - Fenster-Dauer: {window_duration}s")
    print(f"   - Überlappung: {overlap}s") 
    print(f"   - Schritt-Größe: {step_size}s")
    print(f"   - Anzahl Fenster: {num_windows}")
    
    # SCHRITT 4: Seizure-Informationen laden für Ground Truth
    print(f"\n SCHRITT 4: Ground Truth (Seizure-Zeiten) laden...")
    
    patient_key = filename.replace('_res.OWN11101_filtWB_avg.edf', '')
    print(f"   Patient Key: {patient_key}")
    
    seizure_dict = get_sz_start_end(seizure_csv_files[seizure_type])
    seizure_times = seizure_dict.get(patient_key, [])

    if seizure_times:
        print(f"    {len(seizure_times)} Seizure(s) gefunden:")
        for i, (start, end) in enumerate(seizure_times, 1):
            print(f"      {i}. {start:.1f}s - {end:.1f}s ({end-start:.1f}s Dauer)")
    else:
        print(f"   Keine Seizure-Informationen für {patient_key} gefunden")
    
    # SCHRITT 5: Fenster-für-Fenster Analyse (nur reine Fenster)
    print(f"\n SCHRITT 5: JS-Divergenz für reine Fenster berechnen...")
    print(f"    Gemischte Fenster (Seizure + Normal) werden übersprungen für klare Ground Truth")
    
    window_results = []
    seizure_window_count = 0
    normal_window_count = 0
    skipped_window_count = 0  # Für gemischte Fenster
    
    for i in range(num_windows):
        # Fenster-Zeiten berechnen
        start_time = i * step_size
        end_time = start_time + window_duration
        
        # Samples extrahieren
        start_sample = int(start_time * sampling_rate)
        end_sample = int(end_time * sampling_rate)
        
        # Grenzen prüfen
        if end_sample > total_samples:
            end_sample = total_samples
            start_sample = end_sample - int(window_duration * sampling_rate)
            start_time = start_sample / sampling_rate
            end_time = end_sample / sampling_rate
        
        window_data = data[:, start_sample:end_sample]
        
        # Ground Truth bestimmen (reine Fenster-Logik wie Sz_ew_detection.py)
        seizure_overlap = 0
        
        if seizure_times:
            for sz_start, sz_end in seizure_times:
                # Berechne Überlappung zwischen Fenster und Seizure
                overlap_start = max(start_time, sz_start)
                overlap_end = min(end_time, sz_end) 
                overlap_duration = max(0, overlap_end - overlap_start)
                seizure_overlap += overlap_duration
        
        background_overlap = window_duration - seizure_overlap
        
        # Überspringe gemischte Fenster (haben sowohl Seizure als auch Background)
        if seizure_overlap > 0 and background_overlap > 0:
            # Progress-Anzeige alle 50 Fenster (inklusive übersprungene)
            if (i + 1) % 50 == 0 or (i + 1) == num_windows:
                print(f"   Progress: {i+1}/{num_windows} Fenster ({(i+1)/num_windows*100:.1f}%) - Fenster #{i+1} übersprungen (gemischt)")
            skipped_window_count += 1
            continue  # Fenster überspringen
        
        # Klassifiziere nur "reine" Fenster
        if seizure_overlap > 0:
            # 100% Seizure-Fenster
            ground_truth = 'seizure'
            seizure_window_count += 1
        else:
            # 100% Normal-Fenster
            ground_truth = 'non_seizure'
            normal_window_count += 1
        
        # Progress-Anzeige alle 50 Fenster (nur für analysierte Fenster)
        if (len(window_results) + 1) % 50 == 0 or (i + 1) == num_windows:
            analyzed = len(window_results) + 1
            print(f"   Progress: {i+1}/{num_windows} Fenster durchlaufen ({(i+1)/num_windows*100:.1f}%) - {analyzed} analysiert")
        
        # JS-Divergenz für Top 5 Kanäle berechnen
        window_js_scores = []
        
        for ch_info in top_5_channels:
            ch_idx = ch_info['channel_idx']
            ch_name = ch_info['channel']
            
            if ch_idx >= n_channels:
                continue  # Kanal nicht verfügbar
            
            # Aktuelle EEG-Amplituden für diesen Kanal
            current_channel_data = window_data[ch_idx, :].copy()
            
            # Trainingsdaten
            seizure_ref = training_results['seizure_data'][ch_idx]
            non_seizure_ref = training_results['non_seizure_data'][ch_idx]
            
            if len(seizure_ref) == 0 or len(non_seizure_ref) == 0:
                continue
            
            # JS-Divergenzen berechnen
            try:
                js_to_seizure = improved_calculate_kl_divergence(
                    current_channel_data, seizure_ref, 
                    strategy=bin_strategy, verbose=False
                )
                js_to_non_seizure = improved_calculate_kl_divergence(
                    current_channel_data, non_seizure_ref, 
                    strategy=bin_strategy, verbose=False
                )
                
                # Wahrscheinlichkeit berechnen
                similarity_to_seizure = 1 - js_to_seizure
                similarity_to_non_seizure = 1 - js_to_non_seizure
                
                total_similarity = similarity_to_seizure + similarity_to_non_seizure
                
                if total_similarity > 1e-10:
                    prob_seizure = similarity_to_seizure / total_similarity
                else:
                    prob_seizure = 0.5  # Fallback
            
                
                window_js_scores.append({
                    'channel': ch_name,
                    'channel_idx': ch_idx,
                    'js_to_seizure': js_to_seizure,
                    'js_to_non_seizure': js_to_non_seizure,
                    'prob_seizure': prob_seizure,
                    'separability_score': ch_info['separability_score']
                })
                
            except Exception as e:
                continue
        
        # Finale Vorhersage für dieses Fenster (gewichteter Durchschnitt)
        if window_js_scores:
            total_weighted_prob = 0
            total_weight = 0
            
            for score in window_js_scores:
                weight = score['separability_score'] 
                total_weighted_prob += score['prob_seizure'] * weight
                total_weight += weight
            
            if total_weight > 0:
                final_prob_seizure = total_weighted_prob / total_weight
            else:
                final_prob_seizure = np.mean([s['prob_seizure'] for s in window_js_scores])
            
            # ALTERNATIVE: Direkte JS-basierte Entscheidung
            # Berechne durchschnittliche JS-Werte
            avg_js_to_seizure = np.mean([s['js_to_seizure'] for s in window_js_scores])
            avg_js_to_non_seizure = np.mean([s['js_to_non_seizure'] for s in window_js_scores])
            
            # Entscheidung: Zu was ist es ähnlicher? (niedrigere JS = ähnlicher)
            js_based_prediction = 'seizure' if avg_js_to_seizure < avg_js_to_non_seizure else 'non_seizure'
            
            # Verwende JS-basierte Entscheidung
            prediction = js_based_prediction
        else:
            final_prob_seizure = 0.52
            prediction = 'unknown'
        
        # Fenster-Ergebnis speichern
        window_result = {
            'window_id': i + 1,
            'start_time': start_time,
            'end_time': end_time,
            'center_time': (start_time + end_time) / 2,
            'ground_truth': ground_truth,
            'seizure_overlap': seizure_overlap,
            'prediction': prediction,
            'prob_seizure': final_prob_seizure,
            'channel_scores': window_js_scores,
            'n_valid_channels': len(window_js_scores)
        }
        
        # Channel-Results für Plot-Funktionen erstellen
        channel_results = {}
        for score in window_js_scores:
            channel_results[score['channel']] = {
                'js_to_seizure': score['js_to_seizure'],
                'js_to_non_seizure': score['js_to_non_seizure'],
                'prob_seizure': score['prob_seizure']
            }
        window_result['channel_results'] = channel_results
        
        window_results.append(window_result)
    
    # SCHRITT 6: Gesamtergebnisse berechnen
    print(f"\n SCHRITT 6: Gesamtergebnisse auswerten...")
    
    # Accuracy berechnen
    correct_predictions = 0
    total_predictions = 0
    
    seizure_tp = 0  # True Positives
    seizure_fp = 0  # False Positives  
    seizure_tn = 0  # True Negatives
    seizure_fn = 0  # False Negatives
    
    for result in window_results:
        if result['prediction'] != 'unknown':
            total_predictions += 1
            ground_truth = result['ground_truth']
            prediction = result['prediction']
            
            if ground_truth == prediction:
                correct_predictions += 1
            
            # Confusion Matrix
            if ground_truth == 'seizure' and prediction == 'seizure':
                seizure_tp += 1
            elif ground_truth == 'non_seizure' and prediction == 'seizure':
                seizure_fp += 1
            elif ground_truth == 'non_seizure' and prediction == 'non_seizure':
                seizure_tn += 1
            elif ground_truth == 'seizure' and prediction == 'non_seizure':
                seizure_fn += 1
    
    # Metriken berechnen
    precision = seizure_tp / (seizure_tp + seizure_fp) if (seizure_tp + seizure_fp) > 0 else 0
    
    print(f" GESAMTERGEBNISSE:")
    print(f"   - Totale Fenster: {num_windows}")
    print(f"   - Analysierte Fenster: {len(window_results)} ({len(window_results)/num_windows*100:.1f}%)")
    print(f"   - Übersprungene Fenster (gemischt): {skipped_window_count} ({skipped_window_count/num_windows*100:.1f}%)")
    print(f"   - Reine Seizure Fenster: {seizure_window_count} ({seizure_window_count/len(window_results)*100:.1f}% der analysierten)")
    print(f"   - Reine Normal Fenster: {normal_window_count} ({normal_window_count/len(window_results)*100:.1f}% der analysierten)")
    print(f"   - Gültige Vorhersagen: {total_predictions}")
    print(f"   - Precision: {precision:.3f}")

    
    # SCHRITT 7: Ergebnisse zusammenfassen
    analysis_results = {
        'metadata': {
            'filename': filename,
            'patient_key': patient_key,
            'total_duration': duration_seconds,
            'window_duration': window_duration,
            'overlap': overlap,
            'step_size': step_size,
            'num_windows': num_windows,
            'sampling_rate': sampling_rate,
            'n_channels': n_channels,
            'bin_strategy': bin_strategy
        },
        'seizure_info': {
            'seizure_times': seizure_times,
            'seizure_window_count': seizure_window_count,
            'normal_window_count': normal_window_count,
            'skipped_window_count': skipped_window_count,
            'total_windows': num_windows,
            'analyzed_windows': len(window_results)
        },
        'top_channels': top_5_channels,
        'all_channel_separability': importance,  # ALLE Kanäle mit Separability-Scores
        'performance': {
            'precision': precision,
            'confusion_matrix': {
                'tp': seizure_tp, 'fp': seizure_fp,
                'tn': seizure_tn, 'fn': seizure_fn
            }
        },
        'window_results': window_results,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    print(f"\n ANALYSE ABGESCHLOSSEN!")
    
    return analysis_results

def plot_all_channels_js_timeline(analysis_results, save_plot=True):
    """
    Plottet JS-Divergenz für alle verfügbaren Kanäle (die analysiert wurden) einzeln ohne Overall-Plot
    Jeder Kanal bekommt ein eigenes Subplot
    """
    
    window_results = analysis_results['window_results']
    seizure_times = analysis_results['seizure_info']['seizure_times']
    
    # Alle Kanäle sammeln, für die auch wirklich Daten vorliegen
    analyzed_channels = set()
    for result in window_results:
        channel_results = result.get('channel_results', {})
        analyzed_channels.update(channel_results.keys())
    
    analyzed_channels = sorted(list(analyzed_channels))
    
    if not analyzed_channels:
        print("Keine analysierten Kanal-Daten gefunden!")
        return
    
    print(f"Erstelle JS-Timeline für {len(analyzed_channels)} analysierte Kanäle...")
    print(f"Kanäle: {analyzed_channels}")
    
    # Timeline-Daten extrahieren
    time_points = [r['center_time'] for r in window_results]
    ground_truth = [r['ground_truth'] for r in window_results]
    
    # Anzahl Kanäle für Subplot-Layout
    n_channels = len(analyzed_channels)
    cols = 3  # 3 Spalten
    rows = (n_channels + cols - 1) // cols  # Aufrunden für Zeilen
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Für jeden analysierten Kanal einen Plot erstellen
    for i, channel in enumerate(analyzed_channels):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # JS-Divergenz Daten für diesen Kanal extrahieren
        js_values = []
        for result in window_results:
            channel_results = result.get('channel_results', {})
            if channel in channel_results:
                js_to_seizure = channel_results[channel].get('js_to_seizure', 0.5)
                js_values.append(js_to_seizure)
            else:
                js_values.append(None)  # Kein Wert verfügbar
        
        # Nur plotten wenn Daten vorhanden sind
        valid_times = [t for t, v in zip(time_points, js_values) if v is not None]
        valid_values = [v for v in js_values if v is not None]
        
        # Plot erstellen
        if valid_values:
            # Linie plotten
            line = ax.plot(valid_times, valid_values, 'b-', linewidth=1.5, alpha=0.8)[0]
            
            # 5-Sekunden-Datenpunkte markieren
            ax.scatter(valid_times, valid_values, c='blue', s=30, alpha=0.9, zorder=5, 
                      label='5s-Fenster' if i == 0 else "")
        else:
            ax.text(0.5, 0.5, 'Keine Daten', ha='center', va='center', transform=ax.transAxes)
        
        # Threshold-Linien hinzufügen
        ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, linewidth=1, 
                  label='Threshold 0.2 (niedrig)' if i == 0 else "")
        ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.8, linewidth=1.5, 
                  label='Threshold 0.5 (neutral)' if i == 0 else "")
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, linewidth=1, 
                  label='Threshold 0.8 (hoch)' if i == 0 else "")
        
        # Ground Truth als Hintergrund
        for j, (time, gt) in enumerate(zip(time_points, ground_truth)):
            if gt == 1:  # Seizure
                ax.axvspan(time-2.5, time+2.5, alpha=0.2, color='red', 
                          label='Ground Truth Seizure' if i == 0 and j == 0 else "")
        
        # Seizure-Markierungen
        seizure_labeled = False
        for seizure_start, seizure_end in seizure_times:
            ax.axvspan(seizure_start, seizure_end, alpha=0.3, color='orange', 
                      label='Seizure Periode' if i == 0 and not seizure_labeled else "")
            seizure_labeled = True
        
        # Styling
        ax.set_title(f'{channel}', fontsize=14, fontweight='bold')
        ax.set_ylabel('JS to Seizure', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
         # Legende UNTEN RECHTS
    if len(analyzed_channels) > 0:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='orange', alpha=0.3, label='Seizure Periode'),
            plt.Line2D([0], [0], color='green', linestyle='--', label='Threshold 0.2 (niedrig)'),
            plt.Line2D([0], [0], color='gray', linestyle='--', label='Threshold 0.5 (neutral)'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Threshold 0.8 (hoch)')
        ]
        
        # Legende UNTEN RECHTS außerhalb
        fig.legend(handles=legend_elements, 
                  loc='lower right',              # ← Unten rechts
                  bbox_to_anchor=(0.85, 0.18),   # ← Position anpassen
                  ncol=1,                         # ← Vertikal (1 Spalte)
                  fontsize=14,
                  frameon=True,
                  fancybox=True,
                  shadow=True)
        
        # X-Achse nur für untere Reihe
        if row == rows - 1:
            ax.set_xlabel('Zeit (Sekunden)', fontsize=14)
        
        # Y-Achse Beschriftung nur für linke Spalte
        if col > 0:
            ax.set_ylabel('')
    
    # Leere Subplots entfernen
    total_subplots = rows * cols
    for i in range(n_channels, total_subplots):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.suptitle(f'JS-Divergenz zur Seizure-Referenz (Top 5 Kanäle)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    if save_plot:
        patient_id = analysis_results['metadata'].get('patient_key', 'unknown')
        window_dur = analysis_results['metadata'].get('window_duration')
        filename = f"js_{patient_id}_{window_dur}sec.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot gespeichert als: {filename}")

    plt.show(block=True)  
          
    return fig
        


# Hauptfunktion zum Ausführen
if __name__ == "__main__":
    # Verzeichnis mit EDF-Dateien (ABSZ)
    #data_dir = "/home/data/ninalaemmermann/forschung/Data/ABSZ_seizure_WB"

    # Verzeichnis mit EDF-Dateien (ABSZ)
    # specific_files = ['aaaaaccj_s003_t001',
    #                   'aaaaadqe_s001_t001',
    #                   'aaaaacyi_s001_t001',
    #                   'aaaaadyf_s001_t001',
    #                   'aaaaaexe_s001_t001',
    #                   'aaaaaenl_s001_t001']
    #                   'aaaaacyi_s001_t001',
    #                   'aaaaadqe_s001_t001',
    #                   'aaaaadqe_s001_t002',
    #                   'aaaaadyf_s001_t001',
    #                   'aaaaaexe_s001_t001',
    #                   'aaaaaccj_s001_t000', #schlecht
    #                   'aaaaacrb_s001_t001', #schlecht
    #                   'aaaaadqe_s001_t004'] #schlecht
    

    # # Verzeichnis mit EDF-Dateien (GNSZ)
    # specific_files = [
    #                   'aaaaarnq_s003_t006',
    #                   'aaaaaayf_s001_t001',
    #                   'aaaaaayf_s002_t000',
    #                   'aaaaaiup_s001_t000',
    #                   'aaaaamck_s001_t001',
    #                   'aaaaaopm_s002_t001']

    #cpsz
    #specific_files = [ 'aaaaakoe_s002_t003',
                    #   'aaaaardf_s004_t014',
                    #   'aaaaaqtw_s002_t012',
                    #   'aaaaardf_s004_t014',
                    #   'aaaaaqtw_s002_t000',
                    #   'aaaaajqo_s012_t000'  ]
    specific_files = [ 'aaaaardf_s004_t015']
    #                   'aaaaardf_s004_t007',
    #                   'aaaaaizz_s005_t000',
    #                   'aaaaacyf_s011_t001'
                      


    #                   'aaaaagpk_s013_t000',
    #                   'aaaaaooo_s002_t004',
    #                   'aaaaanme_s006_t003',
    #                   'aaaaardk_s002_t010']

    #specific_files = [ 'aaaaaqbt_s001_t000']
    #                   'aaaaaqtq_s006_t004',
    #                   'aaaaajru_s031_t004',
    #                   'aaaaarnq_s003_t006']
    
    seizure_type = 'CPSZ'  # 'ABSZ', 'GNSZ' oder 'FNSZ'
    # Channel Importance EINMAL laden
    importance_file = f'channel_importance_{seizure_type}.pkl'
    try:
        with open(importance_file, 'rb') as f:
            channel_importance = pickle.load(f)
        print(f"✓ Channel Importance für {seizure_type} geladen\n")
    except FileNotFoundError:
        print(f"⚠ {importance_file} nicht gefunden!")
        print("  Führen Sie zuerst setup_channel_importance() aus!")
        channel_importance = None



    # Alle passenden EDF-Dateien auflisten
    for edf_file in specific_files:
        edf_file_full = edf_file + "_res.OWN11101_filtWB_avg.edf"
        print(f"\nStarte Analyse für Datei: {edf_file_full}")
        
        try:
            results = analyze_full_edf_js_divergence(
                filename=edf_file_full,
                window_duration=5.0,
                overlap=0.0,
                seizure_type='CPSZ',
                save_results=True,
                bin_strategy='adaptive',
                channel_importance=channel_importance
            )
            if results:
                plot_all_channels_js_timeline(results, save_plot=True)
            else:
                print(f"Analyse für {edf_file_full} lieferte keine Ergebnisse.")
        except Exception as e:
            print(f"Fehler bei der Analyse von {edf_file_full}: {e}")


    # edf_files = [f for f in os.listdir(data_dir) if f.endswith("_res.OWN11101_filtWB_avg.edf")]
    # if not edf_files:
    #     print(f"Keine EDF-Dateien im Verzeichnis gefunden: {data_dir}")
    # else:
    #     print(f"Gefundene EDF-Dateien: {len(edf_files)}")
    
    # # Für jede Datei: Analyse durchführen und Plot erstellen
    # for edf_fname in sorted(edf_files):
    #     print(f"\n=== Verarbeite Datei: {edf_fname} ===")
    #     try:
    #         results = analyze_full_edf_js_divergence(
    #             filename=edf_fname,
    #             window_duration=3.0,
    #             overlap=0.0,
    #             bin_strategy='adaptive',
    #             save_results=True
    #         )
    #         if results:
    #             try:
    #                 plot_all_channels_js_timeline(results, save_plot=True)
    #                 plt.close('all')  # Freigeben der Figure
    #             except Exception as e:
    #                 print(f"Fehler beim Erstellen des Plots für {edf_fname}: {e}")
    #         else:
    #             print(f"Analyse für {edf_fname} lieferte keine Ergebnisse.")
    #     except Exception as e:
    #         print(f"Fehler bei der Analyse von {edf_fname}: {e}")
    
    # print("\n Alle Dateien verarbeitet.")



