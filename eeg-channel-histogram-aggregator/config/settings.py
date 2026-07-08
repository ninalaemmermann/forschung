import os

# Pfade
BASE_DIR = '/home/data/ninalaemmermann/forschung'
INPUT_DATA_PATH = os.path.join(BASE_DIR, 'eeg_results_absz.pkl')
OUTPUT_DIR = os.path.join(BASE_DIR, 'eeg-channel-histogram-aggregator/data/output')

# Histogram Einstellungen
N_BINS = 200
PERCENTILE_MIN = 0.5
PERCENTILE_MAX = 99.5

# Plot Einstellungen
FIGURE_SIZE = (14, 10)
DPI = 300

# SELECTED_CHANNELS Liste entfernt - wird jetzt automatisch geladen!