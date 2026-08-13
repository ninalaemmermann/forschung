"""
Histogramm-Baseline: Anfalls-Detektion allein aus der Amplitudenverteilung.

Wozu diese Baseline?
--------------------
Der GMM-Detektor beschreibt jedes Fenster durch 8 aufwendige Merkmale je Kanal
(Bandpower in fuenf Frequenzbaendern, Line-Length, Varianz, RMS). Diese Baseline
benutzt NUR die rohe Amplitudenverteilung -- kein Spektrum, keine Ableitung,
nichts ueber Rhythmus. Sie beantwortet damit die Frage, die in jeder
Veroeffentlichung kommt:

    Wie viel bringen die Frequenz-Features gegenueber "einfach nur schauen,
    wie gross die Ausschlaege sind"?

Das ist zugleich die konsequente Weiterfuehrung der bestehenden Verteilungs-
und JS-Divergenz-Analyse (Verteilungsfkt.py): dort werden Histogramme von
Anfall und Ruhe verglichen, hier werden sie zum Klassifikator gemacht.

Warum nicht die vorhandenen eeg_results_*.pkl?
----------------------------------------------
Weil deren Werte ueber ALLE Patienten gepoolt sind. Baut man daraus die
Verteilungen und testet auf denselben Patienten, ist es genau das Datenleck,
das die patientenweise Kreuzvalidierung vermeiden soll. Die Histogramme werden
deshalb INNERHALB jedes Folds neu aus den Trainingspatienten gebildet.

Wie ein Histogramm zum Detektor wird
------------------------------------
1. Pro Fenster und Kanal wird gezaehlt, wie viele Messpunkte in welchen
   Amplituden-Bin fallen -- ein kleines Histogramm je Fenster.
2. Im Training werden diese Fenster-Histogramme klassenweise aufsummiert
   (Histogramme sind additiv) -> zwei Verteilungen je Kanal.
3. Ein Testfenster bekommt als Score das mittlere Log-Likelihood-Verhaeltnis
   seiner Messpunkte: sum_bins zaehler[bin] * log(p_anfall[bin]/p_ruhe[bin]).

Schritt 1 passiert einmal beim Einlesen; danach ist alles nur noch Summieren,
weshalb die Kreuzvalidierung Sekunden statt Stunden braucht.

Die fehlende Zeitkomponente -- ein einzelner Messpunkt sagt fast nichts -- wird
durch die Mittelung ueber das Fenster wieder hereingeholt. Genau das ist der
Unterschied zwischen einer globalen Verteilung und einem Detektor.

Nutzung
-------
    python histogram_baseline.py --seizure-type absz --window-sec 2
    python histogram_baseline.py --selftest

Das Ergebnis landet im selben .pkl-Format wie die GMM-Laeufe:
    python plot_detector_results.py --from-pkl <pfad zur pkl>
"""

import argparse
import os
import pickle

import numpy as np

from gmm_seizure_detection import (_make_group_folds, subject_of,
                                   AUTO_LOPO_MAX_PATIENTS)

N_BINS = 50


# ===========================================================================
# 1. Bin-Grenzen bestimmen
# ===========================================================================
def estimate_bin_edges(seizure_dict, data_folder, n_bins=N_BINS,
                       n_sample_files=10, verbose=True):
    """Robuste Amplituden-Grenzen je Kanal aus einer Stichprobe von Aufnahmen.

    Die Grenzen muessen fuer alle Patienten gleich sein, sonst liessen sich die
    Histogramme spaeter nicht addieren. Verwendet werden die Perzentile 0.1 und
    99.9, damit einzelne Artefakt-Ausschlaege nicht die ganze Skala sprengen.

    Anmerkung zur Leckfreiheit: hier fliessen Aufnahmen ein, die spaeter auch
    Testpatienten sein koennen. Es werden aber AUSSCHLIESSLICH Amplituden-
    bereiche bestimmt, ohne Anfalls-Labels -- die Klasseninformation bleibt
    aussen vor. Das ist dieselbe Kategorie wie der Kanalzahl-Abgleich und
    traegt keine Information ueber die Zielgroesse.
    """
    import mne

    names = list(seizure_dict.keys())
    step = max(1, len(names) // n_sample_files)
    sample = names[::step][:n_sample_files]

    lo_all, hi_all = [], []
    for patient in sample:
        path = os.path.join(data_folder,
                            f"{patient}_res.OWN11101_filtWB_avg.edf")
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            sig = raw.get_data()
            lo_all.append(np.percentile(sig, 0.1, axis=1))
            hi_all.append(np.percentile(sig, 99.9, axis=1))
            del sig, raw
        except Exception as e:  # noqa: BLE001
            print(f"  (Bin-Schaetzung: {patient} uebersprungen: {e})")

    if not lo_all:
        raise RuntimeError("Keine Aufnahme fuer die Bin-Schaetzung lesbar.")

    n_ch = min(len(a) for a in lo_all)
    lo = np.min([a[:n_ch] for a in lo_all], axis=0)
    hi = np.max([a[:n_ch] for a in hi_all], axis=0)
    span = np.maximum(hi - lo, 1e-12)
    lo, hi = lo - 0.05 * span, hi + 0.05 * span

    edges = np.stack([np.linspace(lo[c], hi[c], n_bins + 1)
                      for c in range(n_ch)])
    if verbose:
        print(f"  Bin-Grenzen aus {len(lo_all)} Aufnahmen, {n_ch} Kanaele, "
              f"{n_bins} Bins")
        print(f"  Amplitudenbereich Kanal 1: "
              f"[{edges[0,0]:.2e}, {edges[0,-1]:.2e}]")
    return edges


# ===========================================================================
# 2. Fenster-Histogramme
# ===========================================================================
def window_histograms(signal_data, edges, seizure_mask, sampling_rate,
                      window_sec=2.0, overlap=0.5):
    """Zerlegt ein Signal in Fenster und zaehlt je Fenster und Kanal die Bins.

    Returns H (n_windows, n_channels, n_bins) uint16, y, frac.
    """
    n_ch, n_samples = signal_data.shape
    n_bins = edges.shape[1] - 1
    win = int(round(window_sec * sampling_rate))
    if win < 2 or n_samples < win:
        return (np.empty((0, n_ch, n_bins), dtype=np.uint16),
                np.empty((0,), dtype=int), np.empty((0,), dtype=float))

    # Bin-Index fuer das GESAMTE Signal einmal vorab -- danach ist jedes
    # Fenster nur noch ein Abzaehlen. searchsorted ist hier deutlich
    # schneller als np.histogram je Fenster (das waeren Millionen Aufrufe).
    idx = np.empty((n_ch, n_samples), dtype=np.int32)
    for c in range(n_ch):
        idx[c] = np.clip(
            np.searchsorted(edges[c], signal_data[c], side="right") - 1,
            0, n_bins - 1)
    # Kanal in den Index einrechnen -> ein bincount je Fenster statt n_ch
    idx += (np.arange(n_ch, dtype=np.int32) * n_bins)[:, None]

    step = max(1, int(round(win * (1.0 - overlap))))
    starts = range(0, n_samples - win + 1, step)

    H, y, frac = [], [], []
    for s in starts:
        e = s + win
        counts = np.bincount(idx[:, s:e].ravel(), minlength=n_ch * n_bins)
        H.append(counts.reshape(n_ch, n_bins).astype(np.uint16))
        f = float(seizure_mask[s:e].mean())
        frac.append(f)
        y.append(int(f >= 0.5))

    return (np.asarray(H), np.asarray(y, dtype=int),
            np.asarray(frac, dtype=float))


def build_patient_histograms(seizure_dict, data_folder, edges,
                             window_sec=2.0, overlap=0.5, verbose=True):
    """Liest die EDFs patientenweise und legt die Fenster-Histogramme an."""
    import mne
    patients = []
    n_ch_target = edges.shape[0]

    for i, (patient, sz) in enumerate(seizure_dict.items()):
        path = os.path.join(data_folder,
                            f"{patient}_res.OWN11101_filtWB_avg.edf")
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            sr = raw.info["sfreq"]
            sig = raw.get_data()[:n_ch_target]
            n_samples = sig.shape[1]

            mask = np.zeros(n_samples, dtype=bool)
            for a_t, b_t in sz:
                if not (np.isfinite(a_t) and np.isfinite(b_t)):
                    continue
                a = int(max(0, a_t * sr))
                b = int(min(n_samples, b_t * sr))
                if a < b:
                    mask[a:b] = True

            H, y, frac = window_histograms(sig, edges, mask, sr,
                                           window_sec, overlap)
            if H.size:
                patients.append({"patient": patient, "H": H, "y": y,
                                 "frac": frac})
            if verbose:
                print(f"  Patient {i+1}: {patient} -> {len(y)} Fenster "
                      f"({int(y.sum()) if len(y) else 0} Anfall)")
            del sig, raw
        except Exception as e:  # noqa: BLE001
            print(f"Fehler bei Patient {patient}: {e}")
            continue

    return patients


# ===========================================================================
# 3. Der Klassifikator
# ===========================================================================
class HistogramClassifier:
    """Log-Likelihood-Verhaeltnis aus zwei Amplituden-Histogrammen je Kanal.

    Das Gegenstueck zu GMMClassifier, aber nichtparametrisch und eindimensional
    je Kanal: statt einer gemeinsamen Dichte ueber 80 Merkmale werden n_channels
    unabhaengige Verteilungen ueber die Amplitude geschaetzt. In einer Dimension
    ist ein Histogramm dem GMM sogar ueberlegen, weil es keine Form annimmt --
    es skaliert nur nicht in hohe Dimensionen.
    """

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing   # Laplace, verhindert log(0)
        self.log_ratio_ = None       # (n_channels, n_bins)

    def fit(self, H, y, channels=None):
        Hs = H[:, channels] if channels is not None else H
        pos = Hs[y == 1].sum(axis=0, dtype=np.float64)
        neg = Hs[y == 0].sum(axis=0, dtype=np.float64)
        if pos.sum() == 0 or neg.sum() == 0:
            raise ValueError("Beide Klassen muessen Beispiele enthalten.")
        pos += self.smoothing
        neg += self.smoothing
        pos /= pos.sum(axis=1, keepdims=True)
        neg /= neg.sum(axis=1, keepdims=True)
        self.log_ratio_ = np.log(pos) - np.log(neg)
        return self

    def decision_function(self, H, channels=None):
        Hs = H[:, channels] if channels is not None else H
        # mittleres Log-Verhaeltnis je Messpunkt, ueber Kanaele gemittelt
        s = np.einsum("wcb,cb->w", Hs.astype(np.float64), self.log_ratio_)
        n_points = Hs.sum(axis=(1, 2)).astype(np.float64)
        return s / np.maximum(n_points, 1.0)


class HistogramAnomalyDetector:
    """Nur die Ruhe-Verteilung; Score = negative Log-Likelihood.

    Parallel zu GMMAnomalyDetector, damit beide Spalten der Auswertung
    dieselbe Bedeutung haben.
    """

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.log_neg_ = None

    def fit(self, H, y, channels=None):
        Hs = H[:, channels] if channels is not None else H
        neg = Hs[y == 0].sum(axis=0, dtype=np.float64) + self.smoothing
        neg /= neg.sum(axis=1, keepdims=True)
        self.log_neg_ = np.log(neg)
        return self

    def score_samples(self, H, channels=None):
        Hs = H[:, channels] if channels is not None else H
        s = np.einsum("wcb,cb->w", Hs.astype(np.float64), self.log_neg_)
        n_points = Hs.sum(axis=(1, 2)).astype(np.float64)
        return -s / np.maximum(n_points, 1.0)


def rank_channels_by_js_hist(H, y, n_bins_smoothing=1.0):
    """Kanal-Ranking per JS-Divergenz zwischen den Klassen-Histogrammen.

    Dasselbe Kriterium wie im GMM-Pfad, nur direkt auf den Amplituden-
    verteilungen statt auf den Merkmalen -- und damit exakt das Verfahren aus
    der bestehenden JS-Analyse.
    """
    from scipy.spatial.distance import jensenshannon
    pos = H[y == 1].sum(axis=0, dtype=np.float64) + n_bins_smoothing
    neg = H[y == 0].sum(axis=0, dtype=np.float64) + n_bins_smoothing
    pos /= pos.sum(axis=1, keepdims=True)
    neg /= neg.sum(axis=1, keepdims=True)
    scores = [(c, float(jensenshannon(pos[c], neg[c], base=2)))
              for c in range(H.shape[1])]
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores


# ===========================================================================
# 4. Kreuzvalidierung (spiegelt cross_validate_patients)
# ===========================================================================
def cross_validate_histograms(patients, channel_names, top_k_channels=10,
                              eval_mode="auto", n_folds=5, label_purity=0.5,
                              group_by_subject=True):
    """Patientenweise Kreuzvalidierung der Histogramm-Baseline.

    Fold-Bildung, Subjekt-Gruppierung und Reinheitsfilter sind absichtlich
    identisch zum GMM-Pfad -- nur das Modell ist ein anderes. Sonst waere der
    Vergleich wertlos.
    """
    from sklearn.metrics import roc_auc_score

    n_patients = len(patients)
    n_channels = len(channel_names)

    groups = [subject_of(p["patient"]) if group_by_subject else p["patient"]
              for p in patients]
    n_groups = len(set(groups))
    if group_by_subject and n_groups < n_patients:
        print(f"Gruppierung: {n_patients} Aufnahmen von {n_groups} Subjekten "
              f"-- Folds werden nach Subjekt gebildet.")
    if n_groups < 2:
        print("Zu wenige Gruppen fuer eine Kreuzvalidierung.")
        return None

    if eval_mode == "auto":
        eval_mode = "lopo" if n_groups <= AUTO_LOPO_MAX_PATIENTS else "kfold"
    n_splits = n_groups if eval_mode == "lopo" else min(n_folds, n_groups)
    print(f"Validierung: {'Leave-One-Out' if eval_mode=='lopo' else f'{n_splits}-Fold'} "
          f"ueber {n_groups} Subjekte")
    if label_purity > 0.5:
        print(f"Label-Reinheit: Training nur mit Anteil >= {label_purity:.2f} "
              f"bzw. <= {1-label_purity:.2f}; Test bleibt vollstaendig.")

    folds = _make_group_folds(groups, n_splits)
    k = min(top_k_channels, n_channels)

    fold_rows, per_patient = [], []
    channel_selection_count = np.zeros(n_channels, dtype=int)
    total_dropped = 0

    for fi, test_idx in enumerate(folds):
        test_set = set(test_idx.tolist())
        train_p = [p for j, p in enumerate(patients) if j not in test_set]
        test_p = [patients[j] for j in test_idx]

        Htr = np.concatenate([p["H"] for p in train_p])
        ytr = np.concatenate([p["y"] for p in train_p])
        ftr = np.concatenate([p["frac"] for p in train_p])
        Hte = np.concatenate([p["H"] for p in test_p])
        yte = np.concatenate([p["y"] for p in test_p])

        n_test_sz = int((yte == 1).sum())
        if n_test_sz == 0 or (yte == 0).sum() == 0:
            print(f"  Fold {fi+1}: uebersprungen (Test hat {n_test_sz} "
                  f"Anfalls-Fenster).")
            continue

        if label_purity > 0.5:
            keep = (ftr >= label_purity) | (ftr <= 1.0 - label_purity)
            total_dropped += int((~keep).sum())
            ytr = (ftr >= label_purity).astype(int)[keep]
            Htr = Htr[keep]
        if (ytr == 1).sum() == 0 or (ytr == 0).sum() == 0:
            print(f"  Fold {fi+1}: uebersprungen (Training einklassig).")
            continue

        # Kanalauswahl nur auf Trainingsdaten
        ranking = rank_channels_by_js_hist(Htr, ytr)
        top = sorted(ch for ch, _ in ranking[:k])
        for ch in top:
            channel_selection_count[ch] += 1

        clf = HistogramClassifier().fit(Htr, ytr, channels=top)
        s_clf = clf.decision_function(Hte, channels=top)
        auc_clf = roc_auc_score(yte, s_clf)

        det = HistogramAnomalyDetector().fit(Htr, ytr, channels=top)
        s_anom = det.score_samples(Hte, channels=top)
        auc_anom = roc_auc_score(yte, s_anom)

        offset = 0
        for p in test_p:
            n = len(p["y"])
            yp = p["y"]
            both = yp.min() != yp.max()
            per_patient.append({
                "patient": p["patient"], "fold": fi + 1,
                "y": yp, "frac": p["frac"],
                "score_clf": s_clf[offset:offset + n],
                "score_anom": s_anom[offset:offset + n],
                "auc_clf": (float(roc_auc_score(yp, s_clf[offset:offset + n]))
                            if both else None),
                "auc_anom": (float(roc_auc_score(yp, s_anom[offset:offset + n]))
                             if both else None),
            })
            offset += n

        test_groups = sorted({groups[j] for j in test_idx})
        label = (test_p[0]["patient"] if len(test_p) == 1
                 else (f"{test_groups[0]}, {len(test_p)} Aufnahmen"
                       if len(test_groups) == 1
                       else f"{len(test_groups)} Subjekte / {len(test_p)} Aufn."))
        print(f"  Fold {fi+1:>2d} [{label}]: AUC(clf)={auc_clf:.3f}  "
              f"AUC(anom)={auc_anom:.3f}  (Test: {n_test_sz} Anfalls-Fenster)")
        fold_rows.append({"fold": fi + 1, "label": label, "auc_clf": auc_clf,
                          "auc_anom": auc_anom, "n_test_sz": n_test_sz})

    if not fold_rows:
        print("Keine auswertbaren Folds.")
        return None

    a_clf = np.array([r["auc_clf"] for r in fold_rows])
    a_anom = np.array([r["auc_anom"] for r in fold_rows])
    print("\n" + "-" * 60)
    print(f"Ergebnis ueber {len(fold_rows)} Folds:")
    print(f"  Histogramm-Klassifikator : AUC = {a_clf.mean():.3f} "
          f"+/- {a_clf.std():.3f}")
    print(f"  Histogramm-Anomalie      : AUC = {a_anom.mean():.3f} "
          f"+/- {a_anom.std():.3f}")
    if total_dropped:
        print(f"  (verworfene Mischfenster im Training: {total_dropped})")

    order = np.argsort(channel_selection_count)[::-1]
    print(f"\nAm haeufigsten gewaehlte Kanaele (Top {min(k, 10)}):")
    for ch in order[:min(k, 10)]:
        if channel_selection_count[ch] == 0:
            break
        name = channel_names[ch] if ch < len(channel_names) else f"Ch_{ch}"
        print(f"  {name:<12s}: in {channel_selection_count[ch]}/"
              f"{len(fold_rows)} Folds")

    return {"folds": fold_rows, "per_patient": per_patient,
            "channel_names": list(channel_names),
            "n_dropped_train": total_dropped,
            "auc_clf_mean": float(a_clf.mean()),
            "auc_clf_std": float(a_clf.std()),
            "auc_anom_mean": float(a_anom.mean()),
            "auc_anom_std": float(a_anom.std()),
            "channel_selection_count": channel_selection_count.tolist()}


# ===========================================================================
# 5. Selbsttest
# ===========================================================================
def run_selftest():
    from gmm_seizure_detection import _make_synthetic_patient
    print("=" * 64)
    print("SELBSTTEST: Histogramm-Baseline (synthetisch)")
    print("=" * 64)

    # Bin-Grenzen aus dem synthetischen Signal
    sig0, _, _ = _make_synthetic_patient(seed=0)
    n_ch = sig0.shape[0]
    lo, hi = np.percentile(sig0, [0.1, 99.9])
    edges = np.stack([np.linspace(lo, hi, N_BINS + 1) for _ in range(n_ch)])

    patients = []
    for pid in range(6):
        sig, mask, sr = _make_synthetic_patient(seed=pid)
        H, y, frac = window_histograms(sig, edges, mask, sr, 2.0, 0.5)
        patients.append({"patient": f"synt{pid:02d}_s001_t000", "H": H,
                         "y": y, "frac": frac})
    print(f"{len(patients)} Patienten, H-Form {patients[0]['H'].shape} "
          f"(Fenster, Kanaele, Bins)\n")

    r = cross_validate_histograms(patients, [f"EEG-{k+1}" for k in range(n_ch)],
                                  top_k_channels=3, eval_mode="auto",
                                  label_purity=1.0)
    if r is None or not r["folds"]:
        raise SystemExit("[FEHLER] Keine auswertbaren Folds.")

    # Zaehlt jedes Fenster alle seine Messpunkte?
    win = int(round(2.0 * 256))
    tot = patients[0]["H"][0].sum()
    exp = n_ch * win
    print(f"\n  Messpunkte je Fenster: {tot} (erwartet {exp}) -> "
          f"{'OK' if tot == exp else 'FEHLER'}")
    if tot != exp:
        raise SystemExit("[FEHLER] Histogramm verliert Messpunkte.")

    print("\n[OK] Histogramm-Baseline laeuft.")


# ===========================================================================
# 6. CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seizure-type", default="gnsz")
    ap.add_argument("--base", default="/home/data/ninalaemmermann/forschung")
    ap.add_argument("--window-sec", type=float, default=5.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--label-purity", type=float, default=1.0)
    ap.add_argument("--eval", choices=["auto", "lopo", "kfold"], default="auto")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-bins", type=int, default=N_BINS)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    from Verteilungsfkt import get_sz_start_end

    st = args.seizure_type.upper()
    csv_file = os.path.join(args.base, f"{st}_seizures.csv")
    data_folder = os.path.join(args.base, "Data", f"{st}_seizure_WB")

    print(f"Lade Anfallszeiten aus {csv_file} ...")
    seizure_dict = get_sz_start_end(csv_file)

    print("Bestimme Amplituden-Bins ...")
    edges = estimate_bin_edges(seizure_dict, data_folder, n_bins=args.n_bins)

    print("\nLese EDFs und bilde Fenster-Histogramme ...")
    patients = build_patient_histograms(seizure_dict, data_folder, edges,
                                        args.window_sec, args.overlap)
    if not patients:
        print("Keine Daten extrahiert - Abbruch.")
        return

    n_ch = edges.shape[0]
    channel_names = [f"Ch_{k+1}" for k in range(n_ch)]
    total = sum(len(p["y"]) for p in patients)
    total_sz = sum(int(p["y"].sum()) for p in patients)
    mb = sum(p["H"].nbytes for p in patients) / 1e6
    print(f"\n{len(patients)} Patienten, {total} Fenster, {total_sz} "
          f"Anfalls-Fenster, {n_ch} Kanaele, {args.n_bins} Bins ({mb:.0f} MB)\n")

    result = cross_validate_histograms(
        patients, channel_names, top_k_channels=args.top_k,
        eval_mode=args.eval, n_folds=args.folds,
        label_purity=args.label_purity)
    if result is None:
        return

    result["params"] = {
        "seizure_type": args.seizure_type, "window_sec": args.window_sec,
        "overlap": args.overlap, "top_k": args.top_k,
        "feature_group": "amplitude-histogram",
        "label_purity": args.label_purity, "eval_mode": args.eval,
        "group_by_subject": True, "n_bins": args.n_bins,
        "step_sec": args.window_sec * (1.0 - args.overlap),
        "model": "histogram",
    }

    out_dir = args.out_dir or os.path.join(args.base, "Plots", "detector")
    os.makedirs(out_dir, exist_ok=True)
    tag = (f"{args.seizure_type.lower()}_histogram_p{args.label_purity}"
           f"_w{args.window_sec}_bysubj")
    out_pkl = os.path.join(out_dir, tag + ".pkl")
    with open(out_pkl, "wb") as fh:
        pickle.dump(result, fh)
    print(f"\nErgebnis gespeichert: {out_pkl}")
    print(f"Plots zeichnen mit:\n"
          f"  python plot_detector_results.py --from-pkl {out_pkl}")


if __name__ == "__main__":
    main()
