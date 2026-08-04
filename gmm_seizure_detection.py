"""
GMM-basierte Anfalls-Detektion auf den vorhandenen EEG-Daten.

Dieser Code ergaenzt die bestehende Verteilungs-/JS-Divergenz-Analyse um einen
generativen Klassifikator (Gaussian Mixture Model) UND validiert ihn methodisch
sauber -- naemlich patientenweise (kein Datenleck).

Empfohlener Weg: EDF-Modus mit patientenweiser Kreuzvalidierung
---------------------------------------------------------------
- Aus den rohen EDFs werden pro 2-s-Fenster Features extrahiert
  (Bandpower delta..gamma, Line-Length, Varianz, RMS je Fenster und Kanal).
- Die Kanaele werden per JS-Divergenz gerankt (gleiche Methode wie in
  compare_js_divergence_channels.py) -- die Auswahl passiert INNERHALB jedes
  Folds nur auf den Trainingspatienten (leckfrei).
- Je ein GMM fuer "Anfall" / "kein Anfall" (Entscheidung ueber Likelihood-Ratio),
  zusaetzlich ein Anomalie-Detektor (nur interiktal gefittet).
- Validierung ueber Patienten-Folds:
    * wenige Patienten  -> Leave-One-Patient-Out (LOPO)
    * viele Patienten   -> gruppiertes k-Fold (z.B. 5-Fold)
  Ergebnis: mittlere AUC +/- Std ueber die Folds.

Warum patientenweise? Fenster desselben Patienten sind sehr aehnlich. Landen
welche im Training und welche im Test, erkennt das Modell den Patienten wieder
-> geschoente, wertlose AUC. Deshalb ist jeder Patient KOMPLETT in genau einem
Fold. Das geht nur ueber die EDFs; die gepoolten eeg_results_*.pkl haben die
Patientenzuordnung verloren -- fuer den Detektor werden die pkl daher NICHT
gebraucht (sie bleiben nur fuer die separate Verteilungs-/JS-Analyse relevant).

Anfall vs. kein Anfall: Aus deiner {TYP}_seizures.csv (Sz start / Sz stop) wird
pro Aufnahme eine seizure_mask gebaut; jedes 2-s-Fenster wird darueber als
Anfall (y=1) oder kein Anfall (y=0) gelabelt. Fuer jede der beiden Klassen wird
ein eigenes GMM gefittet -- die Trennung kommt also direkt aus deinen
Annotationen.

Nutzung
-------
    # ABSZ (wenige Dateien) -> automatisch Leave-One-Patient-Out
    python gmm_seizure_detection.py --seizure-type absz

    # die grossen Typen (~200 Dateien) -> automatisch 5-Fold (oder explizit)
    python gmm_seizure_detection.py --seizure-type cpsz --eval kfold --folds 5

    # laengeres Fenster / mehr Kanaele
    python gmm_seizure_detection.py --seizure-type absz --window-sec 5 --top-k 15

    # Feature-Gruppen vergleichen (welche Merkmale trennen am besten?)
    python gmm_seizure_detection.py --seizure-type absz --features all
    python gmm_seizure_detection.py --seizure-type absz --features bandpower
    python gmm_seizure_detection.py --seizure-type absz --features linelength

    # Selbsttest ohne echte Daten (synthetisch):
    python gmm_seizure_detection.py --selftest
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

# Auswaehlbare Feature-Gruppen (Indizes in FEATURE_NAMES) fuer den Vergleich
# "welche Merkmale trennen bei meinen Anfaellen am besten?"
FEATURE_GROUPS = {
    "all":        [0, 1, 2, 3, 4, 5, 6, 7],  # Bandpower + Line-Length + Var + RMS
    "bandpower":  [0, 1, 2, 3, 4],           # nur die 5 Frequenzbaender
    "linelength": [5],                       # nur Line-Length (starkes Einzelmerkmal)
    "energy":     [6, 7],                    # nur Varianz + RMS
    "timedomain": [5, 6, 7],                 # Zeitbereich ohne Spektrum
}

# Ab wie vielen Patienten von LOPO auf k-Fold umgeschaltet wird (--eval auto)
AUTO_LOPO_MAX_PATIENTS = 25


# ===========================================================================
# 1. Fenster-Feature-Extraktion
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
    """Zerlegt ein Mehrkanal-Signal in ueberlappende Fenster und extrahiert Features.

    Returns X (n_windows, n_channels*n_features) und y (n_windows,), y=1 => Anfall.
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


def build_patient_datasets(seizure_dict, data_folder, window_sec=2.0,
                           overlap=0.5, verbose=True):
    """Liest die EDFs PATIENTENWEISE ein und extrahiert Fenster-Features.

    Anders als beim Poolen (wie in der pkl) bleibt hier die Patientenzuordnung
    erhalten -- Voraussetzung fuer eine leckfreie, patientenweise Validierung.

    Returns
    -------
    patients : Liste von dicts {'patient', 'X', 'y'}
    channel_names : Liste der echten Kanalnamen (raw.ch_names)
    """
    import mne  # lokaler Import: nur im EDF-Modus noetig

    patients = []
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
                patients.append({"patient": patient, "X": X, "y": y})

            if verbose:
                n_sz = int(y.sum()) if X.size else 0
                print(f"  Patient {i+1}: {patient} -> "
                      f"{len(y) if X.size else 0} Fenster ({n_sz} Anfall)")

            del sig, raw
        except Exception as e:  # noqa: BLE001 - Robustheit wie im Bestandscode
            print(f"Fehler bei Patient {patient}: {e}")
            continue

    return patients, (channel_names or [])


# ===========================================================================
# 2. Kanalauswahl per JS-Divergenz (spiegelt compare_js_divergence_channels.py)
# ===========================================================================
def js_divergence_1d(a, b, n_bins=50):
    """JS-Divergenz (Basis 2, in bit) zwischen zwei 1D-Stichproben."""
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
    """Rangfolge der Kanaele nach Trennschaerfe (JS-Divergenz der Features)."""
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
# 3. GMM-Klassifikator
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
# 4. Patientenweise Kreuzvalidierung (LOPO oder gruppiertes k-Fold)
# ===========================================================================
def _make_patient_folds(n_patients, n_splits):
    """Teilt Patienten-Indizes deterministisch in n_splits Gruppen (Round-Robin).

    n_splits == n_patients ergibt LOPO (jede Gruppe genau ein Patient).
    Returns Liste von Arrays mit Test-Patienten-Indizes je Fold.
    """
    n_splits = int(min(max(2, n_splits), n_patients))
    folds = [[] for _ in range(n_splits)]
    for idx in range(n_patients):
        folds[idx % n_splits].append(idx)
    return [np.array(f) for f in folds]


def _subsample_negatives(X, y, max_neg, rng):
    """Behaelt alle Anfalls-Fenster, begrenzt die Ruhe-Fenster auf max_neg."""
    if not max_neg:
        return X, y
    neg_idx = np.where(y == 0)[0]
    pos_idx = np.where(y == 1)[0]
    if len(neg_idx) > max_neg:
        neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
    keep = np.concatenate([pos_idx, neg_idx])
    keep.sort()
    return X[keep], y[keep]


def reduce_to_feature_group(X, n_channels, feature_group,
                            n_feat_full=len(FEATURE_NAMES)):
    """Reduziert die vollen 8 Features/Kanal auf die gewaehlte Gruppe.

    X ist (n_windows, n_channels*n_feat_full) mit interleavtem Layout
    [ch0_f0..ch0_f7, ch1_f0..ch1_f7, ...]. Zurueck kommt X nur mit den Feature-
    Spalten der Gruppe je Kanal, plus die neue Feature-Zahl pro Kanal.
    """
    keep = FEATURE_GROUPS[feature_group]
    cols = []
    for ch in range(n_channels):
        base = ch * n_feat_full
        cols.extend(base + i for i in keep)
    return X[:, cols], len(keep)


def cross_validate_patients(patients, channel_names, top_k_channels=10,
                            eval_mode="auto", n_folds=5, n_feat=len(FEATURE_NAMES),
                            max_neg_per_fold=40000, random_state=42):
    """Patientenweise Kreuzvalidierung des GMM-Detektors.

    Pro Fold: Kanalauswahl (JS) NUR auf Trainingspatienten, GMMs auf Training
    fitten, auf den gehaltenen Testpatienten AUC berechnen. Keine Patienten-
    ueberschneidung zwischen Training und Test.

    n_feat = Anzahl Features pro Kanal (haengt von der gewaehlten Feature-Gruppe
    ab; die patients-X muessen bereits entsprechend reduziert sein).
    """
    from sklearn.metrics import roc_auc_score

    n_patients = len(patients)
    n_channels = len(channel_names)
    rng = np.random.default_rng(random_state)

    if n_patients < 2:
        print("Zu wenige Patienten fuer eine Kreuzvalidierung.")
        return None

    # Fold-Strategie bestimmen
    if eval_mode == "auto":
        eval_mode = "lopo" if n_patients <= AUTO_LOPO_MAX_PATIENTS else "kfold"
    if eval_mode == "lopo":
        n_splits = n_patients
        print(f"Validierung: Leave-One-Patient-Out ({n_patients} Patienten "
              f"=> {n_patients} Folds)")
    else:
        n_splits = min(n_folds, n_patients)
        print(f"Validierung: gruppiertes {n_splits}-Fold ueber "
              f"{n_patients} Patienten")

    folds = _make_patient_folds(n_patients, n_splits)
    k = min(top_k_channels, n_channels)

    fold_rows = []
    channel_selection_count = np.zeros(n_channels, dtype=int)

    for fi, test_idx in enumerate(folds):
        test_set = set(test_idx.tolist())
        train_patients = [p for j, p in enumerate(patients) if j not in test_set]
        test_patients = [patients[j] for j in test_idx]

        Xtr = np.vstack([p["X"] for p in train_patients])
        ytr = np.concatenate([p["y"] for p in train_patients])
        Xte = np.vstack([p["X"] for p in test_patients])
        yte = np.concatenate([p["y"] for p in test_patients])

        n_test_sz = int((yte == 1).sum())
        # AUC braucht beide Klassen im Test und im Training
        if n_test_sz == 0 or (yte == 0).sum() == 0:
            print(f"  Fold {fi+1}: uebersprungen (Test hat {n_test_sz} "
                  f"Anfalls-Fenster) - AUC nicht definiert.")
            continue
        if (ytr == 1).sum() == 0 or (ytr == 0).sum() == 0:
            print(f"  Fold {fi+1}: uebersprungen (Training einklassig).")
            continue

        # Ruhe-Klasse im Training begrenzen (Rechenzeit bei vielen Patienten)
        Xtr, ytr = _subsample_negatives(Xtr, ytr, max_neg_per_fold, rng)

        # Kanalauswahl NUR auf Trainingsdaten (leckfrei)
        ranking = rank_channels_by_js(Xtr, ytr, n_channels, n_feat)
        top = [ch for ch, _ in ranking[:k]]
        for ch in top:
            channel_selection_count[ch] += 1

        Xtr_sel = select_channel_features(Xtr, top, n_feat)
        Xte_sel = select_channel_features(Xte, top, n_feat)

        clf = GMMClassifier().fit(Xtr_sel, ytr)
        auc_clf = roc_auc_score(yte, clf.decision_function(Xte_sel))

        det = GMMAnomalyDetector().fit(Xtr_sel, ytr)
        auc_det = roc_auc_score(yte, det.score_samples(Xte_sel))

        label = (test_patients[0]["patient"] if len(test_patients) == 1
                 else f"{len(test_patients)} Patienten")
        print(f"  Fold {fi+1:>2d} [{label}]: "
              f"AUC(clf)={auc_clf:.3f}  AUC(anom)={auc_det:.3f}  "
              f"(Test: {n_test_sz} Anfalls-Fenster)")
        fold_rows.append({"fold": fi + 1, "auc_clf": auc_clf,
                          "auc_anom": auc_det, "n_test_sz": n_test_sz})

    if not fold_rows:
        print("Keine auswertbaren Folds.")
        return None

    auc_clf = np.array([r["auc_clf"] for r in fold_rows])
    auc_anom = np.array([r["auc_anom"] for r in fold_rows])
    print("\n" + "-" * 60)
    print(f"Ergebnis ueber {len(fold_rows)} Folds:")
    print(f"  GMM-Klassifikator : AUC = {auc_clf.mean():.3f} "
          f"+/- {auc_clf.std():.3f}")
    print(f"  Anomalie-Detektor : AUC = {auc_anom.mean():.3f} "
          f"+/- {auc_anom.std():.3f}")

    # Welche Kanaele wurden ueber die Folds am haeufigsten ausgewaehlt?
    order = np.argsort(channel_selection_count)[::-1]
    print(f"\nAm haeufigsten gewaehlte Kanaele (Top {min(k, 10)}):")
    for ch in order[:min(k, 10)]:
        if channel_selection_count[ch] == 0:
            break
        name = channel_names[ch] if ch < len(channel_names) else f"Ch_{ch}"
        print(f"  {name:<12s}: in {channel_selection_count[ch]}/"
              f"{len(fold_rows)} Folds")

    return {"folds": fold_rows,
            "auc_clf_mean": float(auc_clf.mean()),
            "auc_clf_std": float(auc_clf.std()),
            "auc_anom_mean": float(auc_anom.mean()),
            "auc_anom_std": float(auc_anom.std()),
            "channel_selection_count": channel_selection_count.tolist()}


def run_on_edf(seizure_type, base_path, window_sec=2.0, overlap=0.5,
               top_k_channels=10, eval_mode="auto", n_folds=5,
               feature_group="all", max_neg_per_fold=40000, random_state=42):
    """EDF-Pipeline: patientenweise einlesen + patientenweise Kreuzvalidierung.

    feature_group waehlt die Merkmalsmenge (siehe FEATURE_GROUPS): "all",
    "bandpower", "linelength", "energy" oder "timedomain".
    """
    from Verteilungsfkt import get_sz_start_end

    st = seizure_type.upper()
    csv_file = os.path.join(base_path, f"{st}_seizures.csv")
    data_folder = os.path.join(base_path, "Data", f"{st}_seizure_WB")

    print(f"Lade Anfallszeiten aus {csv_file} ...")
    seizure_dict = get_sz_start_end(csv_file)

    print("Lese EDFs patientenweise und extrahiere Fenster-Features ...")
    patients, channel_names = build_patient_datasets(
        seizure_dict, data_folder, window_sec, overlap
    )
    if not patients:
        print("Keine Daten extrahiert - Abbruch.")
        return None

    n_channels = len(channel_names)

    # Auf die gewaehlte Feature-Gruppe reduzieren (Extraktion bleibt vollstaendig,
    # es werden nur die genutzten Feature-Spalten je Kanal ausgewaehlt).
    used_names = [FEATURE_NAMES[i] for i in FEATURE_GROUPS[feature_group]]
    n_feat = len(FEATURE_GROUPS[feature_group])
    if feature_group != "all":
        for p in patients:
            p["X"], _ = reduce_to_feature_group(p["X"], n_channels, feature_group)

    total_windows = sum(len(p["y"]) for p in patients)
    total_sz = sum(int(p["y"].sum()) for p in patients)
    print(f"\n{len(patients)} Patienten, {total_windows} Fenster gesamt, "
          f"{total_sz} Anfalls-Fenster, {n_channels} Kanaele")
    print(f"Feature-Gruppe '{feature_group}': {n_feat} Feature(s)/Kanal "
          f"({', '.join(used_names)})\n")

    return cross_validate_patients(
        patients, channel_names, top_k_channels=top_k_channels,
        eval_mode=eval_mode, n_folds=n_folds, n_feat=n_feat,
        max_neg_per_fold=max_neg_per_fold, random_state=random_state,
    )


# ===========================================================================
# 5. Selbsttest (synthetisch, ohne echte Daten)
# ===========================================================================
def _make_synthetic_patient(seed, n_channels=6, sampling_rate=256.0,
                            seconds=120, seizure_fraction=0.25):
    """Ein synthetischer Patient: Anfallsabschnitt mit staerkerer 25-Hz-Rhythmik."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sampling_rate)
    t = np.arange(n) / sampling_rate
    mask = np.zeros(n, dtype=bool)
    start = int(n * (0.5 - seizure_fraction / 2))
    end = int(n * (0.5 + seizure_fraction / 2))
    mask[start:end] = True
    sig = rng.standard_normal((n_channels, n)) * 0.5
    sig += 0.8 * np.sin(2 * np.pi * 10 * t)              # alpha-Grundrhythmus
    # patientenabhaengige Anfallsstaerke -> realistische Fold-Streuung
    amp = 1.0 + 0.6 * rng.standard_normal()
    burst = np.zeros(n)
    burst[mask] = 1.0
    sig += abs(amp) * burst * np.sin(2 * np.pi * 25 * t)
    return sig, mask, sampling_rate


def run_selftest():
    print("=" * 64)
    print("SELBSTTEST: EDF-Fenster-Features + patientenweise Kreuzvalidierung")
    print("=" * 64)
    patients = []
    for pid in range(6):  # 6 synthetische Patienten
        sig, mask, sr = _make_synthetic_patient(seed=pid)
        X, y = extract_window_features(sig, sr, mask, window_sec=2.0, overlap=0.5)
        patients.append({"patient": f"synt_{pid:02d}", "X": X, "y": y})
    channel_names = [f"EEG-{k+1}" for k in range(6)]
    print(f"{len(patients)} Patienten je "
          f"{len(patients[0]['y'])} Fenster.\n")

    # bei 6 Patienten waehlt auto -> LOPO
    cross_validate_patients(patients, channel_names, top_k_channels=3,
                            eval_mode="auto")

    print("\n[OK] Patientenweise CV laeuft durch.")


# ===========================================================================
# 6. CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--selftest", action="store_true",
                        help="Synthetischer Selbsttest ohne echte Daten.")
    parser.add_argument("--seizure-type", default=None,
                        help="z.B. absz, cpsz, fnsz, gnsz")
    parser.add_argument("--base", default="/home/data/ninalaemmermann/forschung",
                        help="Basis-Pfad der Forschungsdaten.")
    parser.add_argument("--eval", choices=["auto", "lopo", "kfold"],
                        default="auto",
                        help="Validierungsstrategie. auto: <=25 Patienten "
                             "=> LOPO, sonst k-Fold.")
    parser.add_argument("--folds", type=int, default=5,
                        help="Anzahl Folds fuer --eval kfold.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Anzahl der besten Kanaele fuers GMM.")
    parser.add_argument("--features", choices=list(FEATURE_GROUPS.keys()),
                        default="all",
                        help="Merkmalsmenge: all (Bandpower+LL+Var+RMS), "
                             "bandpower, linelength, energy, timedomain.")
    parser.add_argument("--window-sec", type=float, default=2.0,
                        help="Fensterlaenge in Sekunden (z.B. 5.0).")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Fenster-Ueberlappung 0..1.")
    parser.add_argument("--max-neg-per-fold", type=int, default=40000,
                        help="Max. Ruhe-Fenster im Training pro Fold "
                             "(Rechenzeit bei vielen Patienten).")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return

    if args.seizure_type is None:
        parser.error("Bitte --seizure-type angeben (oder --selftest).")
    run_on_edf(args.seizure_type, args.base, args.window_sec, args.overlap,
               args.top_k, args.eval, args.folds, args.features,
               args.max_neg_per_fold)


if __name__ == "__main__":
    main()
