"""
GMM-basierte Anfalls-Detektion auf Fenster-Features.

Dieser Prototyp ergaenzt die bestehende Verteilungs-/JS-Divergenz-Analyse um
einen generativen Klassifikator:

  1. Fenster-Feature-Extraktion aus den rohen EDF-Zeitreihen
     (Bandpower delta/theta/alpha/beta/gamma, Line-Length, Varianz, RMS
     pro Fenster und Kanal). Die rohen EDFs sind noetig, weil die
     gespeicherten .pkl-Dateien nur gepoolte Samples pro Kanal enthalten
     und damit die zeitliche Struktur verlieren.
  2. Kanalauswahl per JS-Divergenz (Anfall vs. kein Anfall) - kniupft an
     die vorhandene compare_js_divergence_channels.py an.
  3. GMM-Klassifikator: je ein sklearn.GaussianMixture fuer "Anfall" und
     "kein Anfall", Komponentenzahl per BIC gewaehlt, Entscheidung ueber
     das Log-Likelihood-Verhaeltnis. Zusaetzlich ein reiner Anomalie-Modus
     (nur auf interiktalen Daten gefittet).

Warum GMM und nicht rohe Amplitude: Einzelne Amplituden-Samples sind stark
autokorreliert (nicht iid) und die Verteilungen von Anfall/Nicht-Anfall
ueberlappen stark. Fenster-Features gewinnen die spektrale/zeitliche
Information zurueck und liefern naeherungsweise unabhaengige Beobachtungen.

Nutzung:
    # Selbsttest ohne echte Daten (synthetisch):
    python gmm_seizure_detection.py --selftest

    # Auf echten Daten (in der Forschungs-Umgebung):
    python gmm_seizure_detection.py --seizure-type cpsz \
        --base /home/data/ninalaemmermann/forschung
"""

import argparse
import os

import numpy as np
from scipy import signal as sp_signal
from scipy.spatial.distance import jensenshannon

# np.trapz wurde in numpy 2.0 zu np.trapezoid umbenannt -> versionssicher
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Standard-Frequenzbaender (Hz) fuer die Bandpower-Features
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Namen der Pro-Kanal-Features in fester Reihenfolge
FEATURE_NAMES = list(FREQ_BANDS.keys()) + ["line_length", "variance", "rms"]


# ---------------------------------------------------------------------------
# 1. Fenster-Feature-Extraktion
# ---------------------------------------------------------------------------
def _bandpowers(window, sampling_rate):
    """Relative Bandpower je Frequenzband via Welch-PSD.

    Gibt einen Vektor in der Reihenfolge von FREQ_BANDS zurueck.
    """
    # nperseg an die Fensterlaenge anpassen (mind. 1, hoechstens Fensterlaenge)
    nperseg = int(min(len(window), max(32, sampling_rate)))
    freqs, psd = sp_signal.welch(window, fs=sampling_rate, nperseg=nperseg)
    total = _trapz(psd, freqs)
    if total <= 0:
        return np.zeros(len(FREQ_BANDS))

    powers = []
    for low, high in FREQ_BANDS.values():
        mask = (freqs >= low) & (freqs < high)
        band = _trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
        powers.append(band / total)  # relative Power -> robuster gegen Skalierung
    return np.array(powers)


def _window_features(window, sampling_rate):
    """Feature-Vektor fuer ein einzelnes Fenster eines Kanals."""
    band = _bandpowers(window, sampling_rate)
    line_length = np.sum(np.abs(np.diff(window)))
    variance = np.var(window)
    rms = np.sqrt(np.mean(window ** 2))
    return np.concatenate([band, [line_length, variance, rms]])


def extract_window_features(signal_data, sampling_rate, seizure_mask,
                            window_sec=2.0, overlap=0.5):
    """Zerlegt ein Mehrkanal-Signal in ueberlappende Fenster und extrahiert Features.

    Parameters
    ----------
    signal_data : ndarray (n_channels, n_samples)
        Rohe EEG-Zeitreihen eines Patienten.
    sampling_rate : float
    seizure_mask : ndarray (n_samples,) bool
        True an Positionen innerhalb eines Anfalls.
    window_sec : float
        Fensterlaenge in Sekunden.
    overlap : float
        Ueberlappungsanteil (0..1).

    Returns
    -------
    X : ndarray (n_windows, n_channels * n_features_pro_kanal)
    y : ndarray (n_windows,) int   (1 = Anfall, 0 = kein Anfall)
    """
    n_channels, n_samples = signal_data.shape
    win = int(round(window_sec * sampling_rate))
    if win < 2 or n_samples < win:
        return np.empty((0, 0)), np.empty((0,), dtype=int)

    step = max(1, int(round(win * (1.0 - overlap))))
    starts = range(0, n_samples - win + 1, step)

    X, y = [], []
    for s in starts:
        e = s + win
        feats = [
            _window_features(signal_data[ch, s:e], sampling_rate)
            for ch in range(n_channels)
        ]
        X.append(np.concatenate(feats))
        # Fenster gilt als Anfall, wenn die Mehrheit der Samples iktal ist
        y.append(int(seizure_mask[s:e].mean() >= 0.5))

    return np.asarray(X), np.asarray(y, dtype=int)


def build_feature_dataset(seizure_dict, data_folder, window_sec=2.0,
                          overlap=0.5, verbose=True):
    """Baut ein Fenster-Feature-Dataset ueber alle Patienten eines Anfallstyps.

    Liest die EDFs analog zu Verteilungsfkt.analyze_eeg_distributions_from_files.
    Gibt X, y und die pro-Kanal-Feature-Namen zurueck.
    """
    import mne  # lokaler Import: nur beim echten Datenlauf noetig

    all_X, all_y = [], []
    channel_names = None

    for i, (patient, sz_start_end) in enumerate(seizure_dict.items()):
        file_path = os.path.join(
            data_folder, f"{patient}_res.OWN11101_filtWB_avg.edf"
        )
        try:
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            sampling_rate = raw.info["sfreq"]
            sig = raw.get_data()
            n_channels, n_samples = sig.shape

            if channel_names is None:
                channel_names = [f"Ch_{k+1}" for k in range(n_channels)]

            seizure_mask = np.zeros(n_samples, dtype=bool)
            for start_time, end_time in sz_start_end:
                a = int(max(0, start_time * sampling_rate))
                b = int(min(n_samples, end_time * sampling_rate))
                if a < b:
                    seizure_mask[a:b] = True

            X, y = extract_window_features(
                sig, sampling_rate, seizure_mask, window_sec, overlap
            )
            if X.size:
                all_X.append(X)
                all_y.append(y)

            if verbose:
                print(f"  Patient {i+1}: {patient} -> {len(y)} Fenster "
                      f"({int(y.sum())} Anfall)")

            del sig, raw
        except Exception as e:  # noqa: BLE001 - Robustheit wie im Bestandscode
            print(f"Fehler bei Patient {patient}: {e}")
            continue

    if not all_X:
        return np.empty((0, 0)), np.empty((0,), dtype=int), []

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    feature_names_per_channel = FEATURE_NAMES
    return X, y, (channel_names, feature_names_per_channel)


# ---------------------------------------------------------------------------
# 2. Kanalauswahl per JS-Divergenz
# ---------------------------------------------------------------------------
def js_divergence_1d(a, b, n_bins=50):
    """JS-Divergenz (Basis 2, in bit) zwischen zwei 1D-Stichproben."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 0.0
    rng = (lo, hi)
    ha, _ = np.histogram(a, bins=n_bins, range=rng, density=False)
    hb, _ = np.histogram(b, bins=n_bins, range=rng, density=False)
    ha = ha + 1e-10
    hb = hb + 1e-10
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    return float(jensenshannon(ha, hb, base=2))


def rank_channels_by_js(X, y, n_channels, n_features_per_channel,
                        aggregate="max"):
    """Rangfolge der Kanaele nach Trennschaerfe (JS-Divergenz der Features).

    Fuer jeden Kanal wird die JS-Divergenz zwischen Anfall/Nicht-Anfall ueber
    alle seine Features berechnet und aggregiert (max oder mean).

    Returns Liste von (kanal_index, score), absteigend sortiert.
    """
    sz = y == 1
    ns = y == 0
    scores = []
    for ch in range(n_channels):
        base = ch * n_features_per_channel
        feat_js = [
            js_divergence_1d(X[sz, base + f], X[ns, base + f])
            for f in range(n_features_per_channel)
        ]
        agg = np.max(feat_js) if aggregate == "max" else np.mean(feat_js)
        scores.append((ch, float(agg)))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores


def select_channel_features(X, channel_indices, n_features_per_channel):
    """Waehlt die Feature-Spalten der angegebenen Kanaele aus."""
    cols = []
    for ch in channel_indices:
        base = ch * n_features_per_channel
        cols.extend(range(base, base + n_features_per_channel))
    return X[:, cols]


# ---------------------------------------------------------------------------
# 3. GMM-Klassifikator
# ---------------------------------------------------------------------------
class GMMClassifier:
    """Generativer Klassifikator aus zwei GMMs (Anfall / kein Anfall).

    Entscheidung ueber das Log-Likelihood-Verhaeltnis plus Log-Prior.
    Komponentenzahl je Klasse wird per BIC gewaehlt.
    """

    def __init__(self, n_components_grid=(1, 2, 3, 4, 5),
                 covariance_type="full", standardize=True, random_state=42):
        self.n_components_grid = n_components_grid
        self.covariance_type = covariance_type
        self.standardize = standardize
        self.random_state = random_state
        self.gmm_pos = None
        self.gmm_neg = None
        self.log_prior_ratio = 0.0
        self._mean = None
        self._std = None

    def _scale(self, X, fit=False):
        if not self.standardize:
            return X
        if fit:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0) + 1e-12
        return (X - self._mean) / self._std

    def _fit_best_gmm(self, X):
        from sklearn.mixture import GaussianMixture

        best, best_bic = None, np.inf
        max_k = min(max(self.n_components_grid), len(X))
        for k in self.n_components_grid:
            if k > max_k:
                break
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                reg_covar=1e-5,
                max_iter=200,
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best, best_bic = gmm, bic
        return best

    def fit(self, X, y):
        Xs = self._scale(X, fit=True)
        Xp, Xn = Xs[y == 1], Xs[y == 0]
        if len(Xp) == 0 or len(Xn) == 0:
            raise ValueError("Beide Klassen muessen Beispiele enthalten.")
        self.gmm_pos = self._fit_best_gmm(Xp)
        self.gmm_neg = self._fit_best_gmm(Xn)
        # Klassen-Prior aus den Haeufigkeiten
        self.log_prior_ratio = np.log(len(Xp) / len(Xn))
        return self

    def decision_function(self, X):
        """Log-Likelihood-Verhaeltnis (>0 spricht fuer 'Anfall')."""
        Xs = self._scale(X)
        return (self.gmm_pos.score_samples(Xs)
                - self.gmm_neg.score_samples(Xs)
                + self.log_prior_ratio)

    def predict(self, X, threshold=0.0):
        return (self.decision_function(X) > threshold).astype(int)


class GMMAnomalyDetector:
    """Anomalie-Detektor: nur auf interiktalen (kein Anfall) Daten gefittet.

    Score = negative Log-Likelihood; hoehere Werte = anomaler (eher Anfall).
    Nuetzlich, wenn zu wenige Anfallsfenster fuer ein zweites GMM vorliegen.
    """

    def __init__(self, n_components_grid=(1, 2, 3, 4, 5),
                 covariance_type="full", standardize=True, random_state=42):
        self._clf = GMMClassifier(n_components_grid, covariance_type,
                                  standardize, random_state)

    def fit(self, X, y):
        Xs = self._clf._scale(X, fit=True)
        self._clf.gmm_neg = self._clf._fit_best_gmm(Xs[y == 0])
        return self

    def score_samples(self, X):
        Xs = self._clf._scale(X)
        return -self._clf.gmm_neg.score_samples(Xs)


# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------
def evaluate(scores, y_true):
    """AUC und beste Balanced-Accuracy-Schwelle aus kontinuierlichen Scores."""
    from sklearn.metrics import roc_auc_score, roc_curve

    auc = roc_auc_score(y_true, scores)
    fpr, tpr, thr = roc_curve(y_true, scores)
    youden = tpr - fpr
    best = int(np.argmax(youden))
    return {
        "auc": float(auc),
        "best_threshold": float(thr[best]),
        "tpr_at_best": float(tpr[best]),
        "fpr_at_best": float(fpr[best]),
    }


# ---------------------------------------------------------------------------
# Selbsttest (synthetisch, ohne echte Daten)
# ---------------------------------------------------------------------------
def _make_synthetic_eeg(n_channels=6, sampling_rate=256.0, seconds=200,
                        seizure_fraction=0.2, rng=None):
    """Erzeugt ein Mehrkanal-Signal, in dem Anfallsabschnitte staerkere
    hochfrequente (beta/gamma) Rhythmik zeigen - genug Struktur, um die
    Pipeline zu verifizieren."""
    rng = rng or np.random.default_rng(0)
    n = int(seconds * sampling_rate)
    t = np.arange(n) / sampling_rate
    mask = np.zeros(n, dtype=bool)
    start = int(n * (0.5 - seizure_fraction / 2))
    end = int(n * (0.5 + seizure_fraction / 2))
    mask[start:end] = True

    sig = rng.standard_normal((n_channels, n)) * 0.5
    # Grundrhythmus (alpha) auf allen Kanaelen
    sig += 0.8 * np.sin(2 * np.pi * 10 * t)
    # Anfall: zusaetzliche 25-Hz-Rhythmik + hoehere Amplitude
    burst = np.zeros(n)
    burst[mask] = 1.0
    sig += 1.5 * burst * np.sin(2 * np.pi * 25 * t)
    return sig, mask, sampling_rate


def run_selftest():
    print("=" * 60)
    print("SELBSTTEST: synthetisches EEG -> Fenster-Features -> GMM")
    print("=" * 60)
    rng = np.random.default_rng(42)
    sig, mask, sr = _make_synthetic_eeg(rng=rng)

    X, y = extract_window_features(sig, sr, mask, window_sec=2.0, overlap=0.5)
    n_channels = sig.shape[0]
    n_feat = len(FEATURE_NAMES)
    print(f"Fenster: {X.shape[0]}, Features/Fenster: {X.shape[1]} "
          f"({n_channels} Kanaele x {n_feat})")
    print(f"Anfallsfenster: {int(y.sum())} / {len(y)}")

    ranking = rank_channels_by_js(X, y, n_channels, n_feat)
    print("\nKanal-Ranking nach JS-Divergenz (Top 3):")
    for ch, score in ranking[:3]:
        print(f"  Ch_{ch+1}: {score:.4f} bits")

    top_channels = [ch for ch, _ in ranking[:3]]
    Xsel = select_channel_features(X, top_channels, n_feat)

    # einfacher train/test-Split
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]

    clf = GMMClassifier().fit(Xsel[tr], y[tr])
    scores = clf.decision_function(Xsel[te])
    metrics = evaluate(scores, y[te])
    print("\nGMM-Klassifikator (Likelihood-Ratio):")
    print(f"  AUC={metrics['auc']:.3f}  TPR={metrics['tpr_at_best']:.2f} "
          f"FPR={metrics['fpr_at_best']:.2f}")

    det = GMMAnomalyDetector().fit(Xsel[tr], y[tr])
    an_scores = det.score_samples(Xsel[te])
    an_metrics = evaluate(an_scores, y[te])
    print("\nGMM-Anomalie-Detektor (nur interiktal gefittet):")
    print(f"  AUC={an_metrics['auc']:.3f}")

    print("\n[OK] Pipeline laeuft durch.")


# ---------------------------------------------------------------------------
# Echter Datenlauf
# ---------------------------------------------------------------------------
def run_on_real_data(seizure_type, base_path, window_sec=2.0, overlap=0.5,
                     top_k_channels=10):
    from Verteilungsfkt import get_sz_start_end

    st = seizure_type.upper()
    csv_file = os.path.join(base_path, f"{st}_seizures.csv")
    data_folder = os.path.join(base_path, "Data", f"{st}_seizure_WB")

    print(f"Lade Anfallszeiten aus {csv_file} ...")
    seizure_dict = get_sz_start_end(csv_file)

    print("Extrahiere Fenster-Features aus den EDFs ...")
    X, y, meta = build_feature_dataset(
        seizure_dict, data_folder, window_sec, overlap
    )
    if X.size == 0:
        print("Keine Daten extrahiert - Abbruch.")
        return

    channel_names, feat_names = meta
    n_channels = len(channel_names)
    n_feat = len(feat_names)
    print(f"\nDataset: {X.shape[0]} Fenster, {n_channels} Kanaele, "
          f"{n_feat} Features/Kanal, {int(y.sum())} Anfallsfenster")

    ranking = rank_channels_by_js(X, y, n_channels, n_feat)
    print("\nKanal-Ranking nach JS-Divergenz (Top 10):")
    for ch, score in ranking[:10]:
        print(f"  {channel_names[ch]}: {score:.4f} bits")

    top_channels = [ch for ch, _ in ranking[:top_k_channels]]
    Xsel = select_channel_features(X, top_channels, n_feat)

    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(
        Xsel, y, test_size=0.3, stratify=y, random_state=42
    )
    clf = GMMClassifier().fit(Xtr, ytr)
    metrics = evaluate(clf.decision_function(Xte), yte)
    print("\nGMM-Klassifikator:")
    print(f"  AUC={metrics['auc']:.3f}  "
          f"Schwelle={metrics['best_threshold']:.3f}  "
          f"TPR={metrics['tpr_at_best']:.2f}  FPR={metrics['fpr_at_best']:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true",
                        help="Synthetischer Selbsttest ohne echte Daten.")
    parser.add_argument("--seizure-type", default=None,
                        help="z.B. cpsz, fnsz, gnsz, absz")
    parser.add_argument("--base", default="/home/data/ninalaemmermann/forschung",
                        help="Basis-Pfad der Forschungsdaten.")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.selftest or args.seizure_type is None:
        run_selftest()
    else:
        run_on_real_data(args.seizure_type, args.base,
                         args.window_sec, args.overlap, args.top_k)


if __name__ == "__main__":
    main()
