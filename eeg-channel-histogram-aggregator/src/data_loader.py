import pickle
import numpy as np

def load_eeg_data(filepath):
    """
    Lädt EEG-Daten aus PKL-Datei
    
    Parameters:
    -----------
    filepath : str
        Pfad zur PKL-Datei (z.B. eeg_results_cpsz.pkl)
    
    Returns:
    --------
    dict : {'channel_names': list, 'seizure_data': list, 'non_seizure_data': list}
    """
    print(f"Lade EEG-Daten aus: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Erfolgreich geladen")
        print(f"  Anzahl Kanäle: {len(data['channel_names'])}")
        print(f"  Kanal-Namen: {data['channel_names'][:5]}... (erste 5)")
        
        # Statistik ausgeben
        for i, ch_name in enumerate(data['channel_names'][:5]):
            sz_samples = len(data['seizure_data'][i])
            ns_samples = len(data['non_seizure_data'][i])
            print(f"  {ch_name}: Seizure={sz_samples:,}, Non-Seizure={ns_samples:,}")
        
        return data
        
    except FileNotFoundError:
        print(f"✗ FEHLER: Datei nicht gefunden: {filepath}")
        return None
    except Exception as e:
        print(f"✗ FEHLER beim Laden: {e}")
        return None