import pickle
import sys
sys.path.append('/home/data/ninalaemmermann/forschung')
from test_js_divergence import analyze_channel_importance

def setup_channel_importance(results, seizure_type):
    """
    SETUP-FUNKTION: Berechnet Channel Importance einmalig
    """
    print(f"\n{'='*60}")
    print(f"SETUP: Channel Importance für {seizure_type}")
    print(f"{'='*60}\n")
    
    importance = analyze_channel_importance(results)
    
    # Speichern
    filename = f'channel_importance_{seizure_type}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(importance, f)
    
    print(f"\n✓ Channel Importance gespeichert: {filename}")
    return importance

# CPSZ vorbereiten
with open('eeg_results_absz.pkl', 'rb') as f:
    results_absz = pickle.load(f)
setup_channel_importance(results_absz, 'ABSZ')

print("\n✓ Setup abgeschlossen!")