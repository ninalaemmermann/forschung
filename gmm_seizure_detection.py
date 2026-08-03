"""
GMM-basierte Anfalls-Detektion auf den vorhandenen EEG-Daten.

Dieser Code ergaenzt die bestehende Verteilungs-/JS-Divergenz-Analyse um einen
generativen Klassifikator (Gaussian Mixture Model). Er ist auf DEIN echtes
Datenformat ausgerichtet und laeuft ohne Anpassung auf dem Forschungsserver.

Zwei Betriebsmodi
-----------------
1) PKL-Modus (Standard, empfohlen)  ->  nutzt deine eeg_results_{typ}.pkl
   Format wie von Verteilungsfkt.py / data_loader.py erwartet:
       {
         'channel_names'   : [27 Namen],
         'seizure_data'    : Liste/Array (27, n_iktal),      # Amplituden-Samples
         'non_seizure_data': Liste/Array (27, n_interiktal), # Amplituden-Samples
       }
   Jeder Zeitpunkt liefert einen 27-dim. Amplitudenvektor. Darauf werden
   - die Kanaele per JS-Divergenz gerankt (gleiche Methode wie in
     compare_js_divergence_channels.py) und
   - je ein GMM fuer "Anfall" / "kein Anfall" gefittet (Likelihood-Ratio),
   - zusaetzlich ein reiner Anomalie-Detektor (nur interiktal gefittet).

   Wissenschaftlicher Hinweis: Die pkl enthaelt reine Amplituden-Samples ohne
   Zeitachse. Roh-Amplituden von Anfall/Nicht-Anfall ueberlappen stark (genau
   das zeigt deine JS-Analyse). Das GMM darauf ist die direkt vergleichbare
   generative Variante zu deiner JS-Divergenz. Fuer klar bessere Trennung siehe
   Modus 2.

2) EDF-Modus (optional, staerker)  ->  nutzt die rohen EDFs + *_seizures.csv
   Extrahiert Fenster-Features (Bandpower delta..gamma, Line-Length, Varianz,
   RMS pro 2-s-Fenster und Kanal). Diese gewinnen die spektrale/zeitliche
   Information zurueck, die in der gepoolten pkl verloren geht.

Nutzung
-------
    # Standard: auf deine pkl anwenden (Pfad wird aus --base + --seizure-type gebaut)
    python gmm_seizure_detection.py --seizure-type cpsz

    # oder pkl direkt angeben
    python gmm_seizure_detection.py --pkl /home/data/ninalaemmermann/forschung/eeg_results_cpsz.pkl

    # staerkerer EDF-Fenster-Modus (braucht Data/{TYP}_seizure_WB/ + {TYP}_seizures.csv)
    python gmm_seizure_detection.py --seizure-type cpsz --source edf

    # Selbsttest ohne echte Daten (synthetisch), prueft beide Pfade:
    python gmm_seizure_detection.py --selftest
"""

import argparse
import os
import pickle

import numpy as np
from scipy import signal as sp_signal
from scipy.spatial.distance import jensenshannon

# np.trapz wurde in numpy 2.0 zu np.trapezoid umbenannt -> versionssicher
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Standard-Frequenzbaender (Hz) fuer die Bandpower-Features (EDF-Modus)
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Namen der Pro-Kanal-Features in fester Reihenfolge (EDF-Modus)
FEATURE_NAMES = list(FREQ_BANDS.keys()) + ["line_length", "variance", "rms"]


# ===========================================================================
# Gemeinsame Helfer
# ===========================================================================
def _stack_channels(per_channel):
    """Bringt seizure_data / non_seizure_data auf ein (n_channels, n_samples)-Array.

    Akzeptiert bereits (C, N)-Arrays oder Listen von 1D-Arrays. Falls die Kanaele
    unterschiedlich lang sind, wird auf die kuerzeste Laenge gekuerzt (mit Hinweis).
    """
    if isinstance(per_channel, np.ndarray) and per_channel.ndim == 2:
        return per_channel
    arrs = [np.asarray(a).ravel() for a in per_channel]
    lengths = [len(a) for a in arrs]
    m = min(lengths)
    if max(lengths) != m:
        print(f"  Hinweis: Kanaele unterschiedlich lang ({m}..{max(lengths)}), "
              f"kuerze auf {m} Samples/Kanal.")
    return np.vstack([a[:m] for a in arrs])


def _subsample(X, n_max, rng):
    """Zieht hoechstens n_max Zeilen (ohne Zuruecklegen) aus X."""
    if n_max and len(X) > n_max:
        idx = rng.choice(len(X), size=n_max, replace=False)
        return X[idx]
    return X


# ===========================================================================
# Kanalauswahl per JS-Divergenz (spiegelt compare_js_divergence_channels.py)
# ===========================================================================
def js_divergence_1d(a, b, n_bins=50):
    """JS-Divergenz (Basis 2, in bit) zwischen zwei 1D-Stichproben.

    Gleiche Konvention wie compare_js_divergence_channels.py: gemeinsamer
    Wertebereich, Normierung zur Wahrscheinlichkeit, Epsilon 1e-10 gegen log(0).
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 0.0
    rng = (lo, hi)
    ha, _ = np.histogram(a, bins=n_bins, range=rng)
    hb, _ = np.histogram(b, bins=n_bins, range=rng)
    ha = ha.astype(float)
    hb = hb.astype(float)
    ha = ha / ha.sum() if ha.sum() > 0 else np.ones(n_bins) / n_bins
    hb = hb / hb.sum() if hb.sum() > 0 else np.ones(n_bins) / n_bins
    ha = ha + 1e-10
    hb = hb + 1e-10
    ha /= ha.sum()
    hb /= hb.sum()
    return float(jensenshannon(ha, hb, base=2))


def rank_channels_by_js(X, y, n_channels, n_features_per_channel,
                        aggregate="max", n_bins=50):
    """Rangfolge der Kanaele nach Trennschaerfe (JS-Divergenz der Features).

    Fuer jeden Kanal wird die JS-Divergenz zwischen Anfall/Nicht-Anfall ueber
    seine Feature-Spalten berechnet und aggregiert (max oder mean). Im PKL-Modus
    ist n_features_per_channel=1 (nur die Amplitude).

    Returns Liste von (kanal_index, score), absteigend sortiert.
    """
    sz = y == 1
    ns = y == 0
    scores = []
    for ch in range(n_channels):
        base = ch * n_features_per_channel
        feat_js = [
            js_divergence_1d(X[sz, base + f], X[ns, base + f], n_bins=n_bins)
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


# ===========================================================================
# GMM-Klassifikator
# ===========================================================================
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


# ===========================================================================
# Auswertung
# ===========================================================================
def evaluate(scores, y_true):
    """AUC und beste Youden-Schwelle aus kontinuierlichen Scores."""
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


# ===========================================================================
# PKL-Pipeline (Standard) -- direkt auf deinen eeg_results_{typ}.pkl
# ===========================================================================
def _amplitude_dataset(seizure_data, non_seizure_data, max_samples_per_class,
                       random_state=42):
    """Baut aus den gepoolten pkl-Samples ein (X, y) fuer die GMM-Pipeline.

    Jeder Zeitpunkt = ein Amplitudenvektor ueber alle Kanaele.
    """
    sz = _stack_channels(seizure_data)        # (C, n_iktal)
    ns = _stack_channels(non_seizure_data)    # (C, n_interiktal)
    if sz.shape[0] != ns.shape[0]:
        raise ValueError("Anfall/Nicht-Anfall haben unterschiedliche Kanalzahl.")

    rng = np.random.default_rng(random_state)
    Xpos = _subsample(sz.T, max_samples_per_class, rng)   # (n, C)
    Xneg = _subsample(ns.T, max_samples_per_class, rng)

    X = np.vstack([Xpos, Xneg])
    y = np.concatenate([np.ones(len(Xpos), dtype=int),
                        np.zeros(len(Xneg), dtype=int)])
    return X, y


def run_on_data_dict(data, top_k_channels=10, max_samples_per_class=50000,
                     test_size=0.3, random_state=42):
    """Fuehrt die komplette PKL-Pipeline auf einem bereits geladenen dict aus."""
    from sklearn.model_selection import train_test_split

    channel_names = list(data["channel_names"])
    n_channels = len(channel_names)
    print(f"Kanaele: {n_channels}  (z.B. {channel_names[:5]} ...)")

    X, y = _amplitude_dataset(
        data["seizure_data"], data["non_seizure_data"],
        max_samples_per_class, random_state
    )
    print(f"Samples: {int((y == 1).sum())} Anfall / {int((y == 0).sum())} kein Anfall "
          f"(nach Subsampling auf max. {max_samples_per_class}/Klasse)")

    # 1 Feature pro Kanal (die Amplitude selbst)
    ranking = rank_channels_by_js(X, y, n_channels, 1)
    print("\nKanal-Ranking nach JS-Divergenz (Top 10):")
    for ch, score in ranking[:10]:
        print(f"  {channel_names[ch]:<12s}: {score:.4f} bits")

    k = min(top_k_channels, n_channels)
    top_channels = [ch for ch, _ in ranking[:k]]
    print(f"\nVerwende Top-{k} Kanaele: "
          f"{[channel_names[ch] for ch in top_channels]}")
    Xsel = select_channel_features(X, top_channels, 1)

    Xtr, Xte, ytr, yte = train_test_split(
        Xsel, y, test_size=test_size, stratify=y, random_state=random_state
    )

    clf = GMMClassifier().fit(Xtr, ytr)
    m = evaluate(clf.decision_function(Xte), yte)
    print("\nGMM-Klassifikator (Likelihood-Ratio):")
    print(f"  AUC={m['auc']:.3f}  Schwelle={m['best_threshold']:.3f}  "
          f"TPR={m['tpr_at_best']:.2f}  FPR={m['fpr_at_best']:.2f}")

    det = GMMAnomalyDetector().fit(Xtr, ytr)
    ma = evaluate(det.score_samples(Xte), yte)
    print("\nGMM-Anomalie-Detektor (nur interiktal gefittet):")
    print(f"  AUC={ma['auc']:.3f}")

    return {"ranking": ranking, "channel_names": channel_names,
            "classifier": m, "anomaly": ma}


def run_on_pkl(pkl_path, **kwargs):
    """Laedt eine eeg_results_{typ}.pkl und ruft die PKL-Pipeline auf."""
    print(f"Lade EEG-Daten aus: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for key in ("channel_names", "seizure_data", "non_seizure_data"):
        if key not in data:
            raise KeyError(f"pkl fehlt der Schluessel '{key}'. Vorhanden: "
                           f"{list(data.keys())}")
    print("Erfolgreich geladen.")
    return run_on_data_dict(data, **kwargs)


# ===========================================================================
# EDF-Pipeline (optional, staerker) -- Fenster-Features aus den Roh-EDFs
# ===========================================================================
def _bandpowers(window, sampling_rate):
    """Relative Bandpower je Frequenzband via Welch-PSD."""
    nperseg = int(min(len(window), max(32, sampling_rate)))
    freqs, psd = sp_signal.welch(window, fs=sampling_rate, nperseg=nperseg)
    total = _trapz(psd, freqs)
    if total <= 0:
        return np.zeros(len(FREQ_BANDS))
    powers = []
    for low, high in FREQ_BANDS.values():
        mask = (freqs >= low) & (freqs < high)
        band = _trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
        powers.append(band / total)
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
    """Zerlegt ein Mehrkanal-Signal in ueberlappende Fenster und extrahiert Features."""
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
        y.append(int(seizure_mask[s:e].mean() >= 0.5))

    return np.asarray(X), np.asarray(y, dtype=int)


def build_feature_dataset(seizure_dict, data_folder, window_sec=2.0,
                          overlap=0.5, verbose=True):
    """Baut ein Fenster-Feature-Dataset ueber alle Patienten eines Anfallstyps.

    Liest die EDFs analog zu Verteilungsfkt.analyze_eeg_distributions_from_files
    und uebernimmt die ECHTEN Kanalnamen aus der EDF (raw.ch_names).
    """
    import mne  # lokaler Import: nur im EDF-Modus noetig

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
                channel_names = list(raw.ch_names)  # echte Kanalnamen

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
        return np.empty((0, 0)), np.empty((0,), dtype=int), ([], FEATURE_NAMES)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    return X, y, (channel_names, FEATURE_NAMES)


def run_on_edf(seizure_type, base_path, window_sec=2.0, overlap=0.5,
               top_k_channels=10, test_size=0.3, random_state=42):
    """EDF-Fenster-Feature-Pipeline auf den Roh-EDFs + *_seizures.csv."""
    from sklearn.model_selection import train_test_split

    from Verteilungsfkt import get_sz_start_end

    st = seizure_type.upper()
    csv_file = os.path.join(base_path, f"{st}_seizures.csv")
    data_folder = os.path.join(base_path, "Data", f"{st}_seizure_WB")

    print(f"Lade Anfallszeiten aus {csv_file} ...")
    seizure_dict = get_sz_start_end(csv_file)

    print("Extrahiere Fenster-Features aus den EDFs ...")
    X, y, meta = build_feature_dataset(seizure_dict, data_folder,
                                       window_sec, overlap)
    if X.size == 0:
        print("Keine Daten extrahiert - Abbruch.")
        return None

    channel_names, feat_names = meta
    n_channels = len(channel_names)
    n_feat = len(feat_names)
    print(f"\nDataset: {X.shape[0]} Fenster, {n_channels} Kanaele, "
          f"{n_feat} Features/Kanal, {int(y.sum())} Anfallsfenster")

    ranking = rank_channels_by_js(X, y, n_channels, n_feat)
    print("\nKanal-Ranking nach JS-Divergenz (Top 10):")
    for ch, score in ranking[:10]:
        print(f"  {channel_names[ch]:<12s}: {score:.4f} bits")

    k = min(top_k_channels, n_channels)
    top_channels = [ch for ch, _ in ranking[:k]]
    Xsel = select_channel_features(X, top_channels, n_feat)

    Xtr, Xte, ytr, yte = train_test_split(
        Xsel, y, test_size=test_size, stratify=y, random_state=random_state
    )
    clf = GMMClassifier().fit(Xtr, ytr)
    m = evaluate(clf.decision_function(Xte), yte)
    print("\nGMM-Klassifikator (Fenster-Features):")
    print(f"  AUC={m['auc']:.3f}  Schwelle={m['best_threshold']:.3f}  "
          f"TPR={m['tpr_at_best']:.2f}  FPR={m['fpr_at_best']:.2f}")
    return {"ranking": ranking, "channel_names": channel_names, "classifier": m}


# ===========================================================================
# Selbsttest (synthetisch, ohne echte Daten) -- prueft beide Pfade
# ===========================================================================
def _make_synthetic_pkl(n_channels=27, n_samples=40000, rng=None):
    """Erzeugt ein dict im pkl-Format. Einige Kanaele trennen Anfall/Nicht-Anfall
    (verschobene Amplituden-Verteilung), die uebrigen sind uninformativ."""
    rng = rng or np.random.default_rng(0)
    channel_names = [f"EEG-{k+1:02d}" for k in range(n_channels)]
    informative = set(range(0, n_channels, 5))  # jeder 5. Kanal traegt Signal
    sz, ns = [], []
    for ch in range(n_channels):
        if ch in informative:
            sz.append(rng.normal(0.4, 1.3, n_samples))   # anders verteilt
            ns.append(rng.normal(0.0, 1.0, n_samples))
        else:
            sz.append(rng.normal(0.0, 1.0, n_samples))   # kein Unterschied
            ns.append(rng.normal(0.0, 1.0, n_samples))
    return {"channel_names": channel_names,
            "seizure_data": sz, "non_seizure_data": ns}


def _make_synthetic_eeg(n_channels=6, sampling_rate=256.0, seconds=200,
                        seizure_fraction=0.2, rng=None):
    """Mehrkanal-Signal mit staerkerer 25-Hz-Rhythmik in den Anfallsabschnitten."""
    rng = rng or np.random.default_rng(0)
    n = int(seconds * sampling_rate)
    t = np.arange(n) / sampling_rate
    mask = np.zeros(n, dtype=bool)
    start = int(n * (0.5 - seizure_fraction / 2))
    end = int(n * (0.5 + seizure_fraction / 2))
    mask[start:end] = True
    sig = rng.standard_normal((n_channels, n)) * 0.5
    sig += 0.8 * np.sin(2 * np.pi * 10 * t)
    burst = np.zeros(n)
    burst[mask] = 1.0
    sig += 1.5 * burst * np.sin(2 * np.pi * 25 * t)
    return sig, mask, sampling_rate


def run_selftest():
    print("=" * 64)
    print("SELBSTTEST 1/2: PKL-Modus (synthetisch, dein echtes Datenformat)")
    print("=" * 64)
    data = _make_synthetic_pkl(rng=np.random.default_rng(1))
    run_on_data_dict(data, top_k_channels=6, max_samples_per_class=8000)

    print("\n" + "=" * 64)
    print("SELBSTTEST 2/2: EDF-Fenster-Feature-Modus (synthetisch)")
    print("=" * 64)
    rng = np.random.default_rng(42)
    sig, mask, sr = _make_synthetic_eeg(rng=rng)
    X, y = extract_window_features(sig, sr, mask, window_sec=2.0, overlap=0.5)
    n_channels, n_feat = sig.shape[0], len(FEATURE_NAMES)
    print(f"Fenster: {X.shape[0]}, Features/Fenster: {X.shape[1]}")
    ranking = rank_channels_by_js(X, y, n_channels, n_feat)
    top = [ch for ch, _ in ranking[:3]]
    Xsel = select_channel_features(X, top, n_feat)
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    clf = GMMClassifier().fit(Xsel[tr], y[tr])
    m = evaluate(clf.decision_function(Xsel[te]), y[te])
    print(f"GMM-Klassifikator: AUC={m['auc']:.3f}")

    print("\n[OK] Beide Pipelines laufen durch.")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--selftest", action="store_true",
                        help="Synthetischer Selbsttest ohne echte Daten.")
    parser.add_argument("--source", choices=["auto", "pkl", "edf"],
                        default="auto",
                        help="Datenquelle. auto: pkl bevorzugen, sonst edf.")
    parser.add_argument("--seizure-type", default=None,
                        help="z.B. cpsz, fnsz, gnsz, absz")
    parser.add_argument("--base", default="/home/data/ninalaemmermann/forschung",
                        help="Basis-Pfad der Forschungsdaten.")
    parser.add_argument("--pkl", default=None,
                        help="Direkter Pfad zu einer eeg_results_*.pkl "
                             "(ueberschreibt --base/--seizure-type).")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Anzahl der besten Kanaele fuers GMM.")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Max. Samples pro Klasse (PKL-Modus, Subsampling).")
    parser.add_argument("--window-sec", type=float, default=2.0,
                        help="Fensterlaenge in Sekunden (EDF-Modus).")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Fenster-Ueberlappung 0..1 (EDF-Modus).")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return

    # Quelle bestimmen
    pkl_path = args.pkl
    if pkl_path is None and args.seizure_type is not None:
        pkl_path = os.path.join(
            args.base, f"eeg_results_{args.seizure_type.lower()}.pkl"
        )

    use_edf = args.source == "edf"
    if args.source == "auto":
        use_edf = not (pkl_path and os.path.exists(pkl_path))

    if not use_edf:
        if not pkl_path:
            parser.error("Bitte --seizure-type oder --pkl angeben "
                         "(oder --selftest).")
        if not os.path.exists(pkl_path):
            parser.error(f"pkl nicht gefunden: {pkl_path}\n"
                         f"Alternativ --source edf verwenden.")
        run_on_pkl(pkl_path, top_k_channels=args.top_k,
                   max_samples_per_class=args.max_samples)
    else:
        if args.seizure_type is None:
            parser.error("EDF-Modus braucht --seizure-type.")
        run_on_edf(args.seizure_type, args.base, args.window_sec,
                   args.overlap, args.top_k)


if __name__ == "__main__":
    main()
