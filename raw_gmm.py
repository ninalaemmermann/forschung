"""
GMM direkt auf den Rohwerten -- der Ansatz der Bachelorarbeit auf EEG uebertragen.

Die Luecke, die diese Datei schliesst
-------------------------------------
Bisher wurden zwei Repraesentationen verglichen:

                        Kanaele unabhaengig   Kanaele gekoppelt
    Amplitude allein    histogram_baseline    -> HIER
    + Frequenz          -                     gmm_seizure_detection

Die Histogramm-Baseline behandelt jeden Kanal getrennt und ignoriert damit die
raeumliche Kopplung ("Fp1 und Fp2 schlagen gleichzeitig aus"). Der volle
GMM-Pfad nutzt zwar die Kopplung, aber auf abgeleiteten Merkmalen. Diese Datei
fuellt die fehlende Zelle: eine GEMEINSAME Verteilung ueber die Rohwerte aller
Kanaele, ohne jedes abgeleitete Merkmal.

Damit laesst sich trennen, woher der AP-Rueckstand der Baseline kommt --
von den fehlenden Frequenzmerkmalen oder von der ignorierten Kanalkopplung.
Das sind zwei voellig verschiedene Schlussfolgerungen.

Genau so arbeitet die Bachelorarbeit (Kapitel 4.3/4.5): ein Datenpunkt ist ein
Zeitpunkt mit allen Sensorwerten, das GMM bewertet ihn unmittelbar. Dort sind es
38 Servermetriken je Zeitpunkt, hier 27 Kanaele je Messpunkt.

Warum die Dimensionalitaet hier unkritisch ist
----------------------------------------------
27 Dimensionen bedeuten 378 Kovarianzparameter je Komponente -- bei Millionen
von Messpunkten bequem schaetzbar. Das Problem, das den full-Lauf auf den
80-dimensionalen Merkmalen ruiniert hat (0.03 Beispiele je Parameter), gibt es
hier nicht. Deshalb ist covariance_type="full" der Default.

Was auch dieser Weg nicht kann
------------------------------
Rhythmus. Ein einzelner Messpunkt weiss nicht, ob er zu einer 3-Hz-Spike-Wave
oder zu einem 10-Hz-Alpha gehoert. Zwei Signale mit gleicher Amplituden-
verteilung, aber verschiedener Frequenz, sind ununterscheidbar.

Vom Messpunkt zum Fenster
-------------------------
Bewertet wird je Messpunkt, ausgewertet je Fenster: der Fenster-Score ist das
mittlere Log-Likelihood-Verhaeltnis seiner Messpunkte -- dieselbe Mittelung wie
in der Histogramm-Baseline. Ueber eine kumulative Summe kostet jedes Fenster
danach nur noch zwei Subtraktionen, unabhaengig von seiner Laenge.

EEG ist gegenueber seinem Informationsgehalt deutlich ueberabgetastet, deshalb
wird nur jeder --stride-te Messpunkt verwendet (Default 4, also 64 statt 256 Hz).

Nutzung
-------
    python raw_gmm.py --seizure-type absz --window-sec 2
    python raw_gmm.py --selftest

Das Ergebnis landet im selben .pkl-Format wie alle anderen Laeufe.
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np

from gmm_seizure_detection import (_make_group_folds, subject_of,
                                   AUTO_LOPO_MAX_PATIENTS)


# ===========================================================================
# 1. Rohdaten einlesen (unterabgetastet)
# ===========================================================================
def build_patient_raw(seizure_dict, data_folder, window_sec=2.0, overlap=0.5,
                      stride=4, n_channels=None, verbose=True):
    """Liest die EDFs und legt je Patient das unterabgetastete Signal ab.

    Gespeichert wird das Signal, nicht die Fenster -- die Fenster ueberlappen,
    eine fensterweise Ablage wuerde jeden Messpunkt doppelt halten. Die
    Fenstergrenzen ergeben sich stattdessen rechnerisch.

    Returns Liste von dicts mit
        S    (n_ch, m) float32  -- Signal, jeder stride-te Messpunkt
        m_sz (m,) bool          -- iktal je unterabgetastetem Messpunkt
        y, frac                 -- Fenster-Label wie in allen anderen Pfaden,
                                   berechnet auf voller Aufloesung
        win_s, step_s           -- Fenstergroesse/-schritt in Stride-Einheiten
    """
    import mne

    patients = []
    for i, (patient, sz) in enumerate(seizure_dict.items()):
        path = os.path.join(data_folder,
                            f"{patient}_res.OWN11101_filtWB_avg.edf")
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            sr = raw.info["sfreq"]
            sig = raw.get_data()
            if n_channels:
                sig = sig[:n_channels]
            n = sig.shape[1]

            mask = np.zeros(n, dtype=bool)
            for a_t, b_t in sz:
                if not (np.isfinite(a_t) and np.isfinite(b_t)):
                    continue
                a, b = int(max(0, a_t * sr)), int(min(n, b_t * sr))
                if a < b:
                    mask[a:b] = True

            win = int(round(window_sec * sr))
            step = max(1, int(round(win * (1.0 - overlap))))
            if win < stride or n < win:
                continue
            # Damit die Fenstergrenzen im unterabgetasteten Signal exakt
            # aufgehen, muessen win und step durch stride teilbar sein.
            win_s, step_s = win // stride, max(1, step // stride)

            # Fenster-Label auf VOLLER Aufloesung -- identisch zu den anderen
            # Pfaden, sonst waeren die Testsaetze nicht dieselben.
            y, frac = [], []
            for s0 in range(0, n - win + 1, step):
                f = float(mask[s0:s0 + win].mean())
                frac.append(f)
                y.append(int(f >= 0.5))
            y = np.asarray(y, dtype=int)
            frac = np.asarray(frac, dtype=float)

            S = np.ascontiguousarray(sig[:, ::stride], dtype=np.float32)
            m_sz = mask[::stride]
            # Auf die Zahl der Fenster zuschneiden
            need = (len(y) - 1) * step_s + win_s
            if S.shape[1] < need:
                keep = (S.shape[1] - win_s) // step_s + 1
                if keep < 1:
                    continue
                y, frac = y[:keep], frac[:keep]
            patients.append({"patient": patient, "S": S, "m_sz": m_sz,
                             "y": y, "frac": frac,
                             "win_s": win_s, "step_s": step_s})
            if verbose:
                print(f"  Patient {i+1}: {patient} -> {len(y)} Fenster "
                      f"({int(y.sum())} Anfall), {S.shape[1]} Messpunkte")
            del sig, raw
        except Exception as e:  # noqa: BLE001
            print(f"Fehler bei Patient {patient}: {e}")
            continue
    return patients


# ===========================================================================
# 2. Kanalauswahl (gleiches Kriterium wie in den anderen Pfaden)
# ===========================================================================
def rank_channels_raw(patients_train, n_bins=50, max_points=400000,
                      random_state=42):
    """JS-Divergenz je Kanal zwischen iktaler und interiktaler Amplitude."""
    from scipy.spatial.distance import jensenshannon
    rng = np.random.default_rng(random_state)

    pos = [p["S"][:, p["m_sz"]] for p in patients_train]
    neg = [p["S"][:, ~p["m_sz"]] for p in patients_train]
    P = np.concatenate(pos, axis=1)
    N = np.concatenate(neg, axis=1)
    if P.shape[1] > max_points:
        P = P[:, rng.choice(P.shape[1], max_points, replace=False)]
    if N.shape[1] > max_points:
        N = N[:, rng.choice(N.shape[1], max_points, replace=False)]

    scores = []
    for c in range(P.shape[0]):
        lo = min(P[c].min(), N[c].min())
        hi = max(P[c].max(), N[c].max())
        if hi <= lo:
            scores.append((c, 0.0)); continue
        hp, _ = np.histogram(P[c], bins=n_bins, range=(lo, hi))
        hn, _ = np.histogram(N[c], bins=n_bins, range=(lo, hi))
        hp = hp + 1e-10; hn = hn + 1e-10
        hp = hp / hp.sum(); hn = hn / hn.sum()
        scores.append((c, float(jensenshannon(hp, hn, base=2))))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores


# ===========================================================================
# 3. Der Klassifikator
# ===========================================================================
class RawSampleGMM:
    """Zwei GMMs ueber die Rohwerte aller Kanaele, Score je Messpunkt.

    Anders als im Merkmals-Pfad gibt es hier KEINE mehrdeutigen Labels: ein
    Messpunkt liegt entweder in einem annotierten Anfall oder nicht. Der
    Reinheitsfilter ist auf dieser Ebene gegenstandslos -- er war ein Artefakt
    der Fensterbildung.
    """

    def __init__(self, n_components_grid=(1, 2, 3, 5, 8),
                 covariance_type="full", reg_covar=1e-6,
                 max_fit_samples=200000, random_state=42):
        self.n_components_grid = n_components_grid
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_fit_samples = max_fit_samples
        self.random_state = random_state
        self.gmm_pos = self.gmm_neg = None
        self._mean = self._std = None
        self.log_prior_ratio = 0.0
        self.k_pos_ = self.k_neg_ = None

    def _fit_best(self, X):
        from sklearn.mixture import GaussianMixture
        best, best_bic, best_k = None, np.inf, None
        for k in self.n_components_grid:
            if k > len(X):
                break
            g = GaussianMixture(n_components=k,
                                covariance_type=self.covariance_type,
                                reg_covar=self.reg_covar, max_iter=200,
                                random_state=self.random_state)
            g.fit(X)
            bic = g.bic(X)
            if bic < best_bic:
                best, best_bic, best_k = g, bic, k
        return best, best_k

    def fit(self, patients_train, channels):
        rng = np.random.default_rng(self.random_state)
        P = np.concatenate([p["S"][channels][:, p["m_sz"]]
                            for p in patients_train], axis=1).T
        N = np.concatenate([p["S"][channels][:, ~p["m_sz"]]
                            for p in patients_train], axis=1).T
        n_pos_all, n_neg_all = len(P), len(N)
        if n_pos_all == 0 or n_neg_all == 0:
            raise ValueError("Beide Klassen muessen Messpunkte enthalten.")

        # Standardisierung auf den Trainingsdaten (beide Klassen zusammen)
        both = np.concatenate([P[rng.choice(n_pos_all, min(n_pos_all, 50000),
                                            replace=False)],
                               N[rng.choice(n_neg_all, min(n_neg_all, 50000),
                                            replace=False)]])
        self._mean = both.mean(axis=0)
        self._std = both.std(axis=0) + 1e-20

        if n_pos_all > self.max_fit_samples:
            P = P[rng.choice(n_pos_all, self.max_fit_samples, replace=False)]
        if n_neg_all > self.max_fit_samples:
            N = N[rng.choice(n_neg_all, self.max_fit_samples, replace=False)]

        self.gmm_pos, self.k_pos_ = self._fit_best((P - self._mean) / self._std)
        self.gmm_neg, self.k_neg_ = self._fit_best((N - self._mean) / self._std)
        self.log_prior_ratio = float(np.log(n_pos_all / n_neg_all))
        return self

    def window_scores(self, patient, channels):
        """Fenster-Score = mittleres Log-Verhaeltnis seiner Messpunkte.

        Ueber eine kumulative Summe kostet jedes Fenster zwei Subtraktionen,
        unabhaengig von seiner Laenge -- bei 50% Ueberlappung waere die naive
        Variante sonst doppelte Arbeit.
        """
        X = ((patient["S"][channels].T - self._mean) / self._std)
        lr = (self.gmm_pos.score_samples(X) - self.gmm_neg.score_samples(X)
              + self.log_prior_ratio)
        cum = np.concatenate([[0.0], np.cumsum(lr)])
        w, st = patient["win_s"], patient["step_s"]
        n_win = len(patient["y"])
        starts = np.arange(n_win) * st
        ends = np.minimum(starts + w, len(lr))
        return (cum[ends] - cum[starts]) / np.maximum(ends - starts, 1)


# ===========================================================================
# 4. Kreuzvalidierung (gleiche Folds wie alle anderen Pfade)
# ===========================================================================
def cross_validate_raw(patients, channel_names, top_k_channels=10,
                       eval_mode="auto", n_folds=5, group_by_subject=True,
                       **gmm_kw):
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
    print(f"Validierung: {'Leave-One-Out' if eval_mode=='lopo' else str(n_splits)+'-Fold'}"
          f" ueber {n_groups} Subjekte")
    print("Hinweis: auf Messpunkt-Ebene gibt es keine Mischlabels, der "
          "Reinheitsfilter entfaellt.")

    folds = _make_group_folds(groups, n_splits)
    k = min(top_k_channels, n_channels) if top_k_channels else n_channels

    fold_rows, per_patient = [], []
    channel_selection_count = np.zeros(n_channels, dtype=int)

    for fi, test_idx in enumerate(folds):
        test_set = set(test_idx.tolist())
        train_p = [p for j, p in enumerate(patients) if j not in test_set]
        test_p = [patients[j] for j in test_idx]

        yte = np.concatenate([p["y"] for p in test_p])
        n_test_sz = int((yte == 1).sum())
        if n_test_sz == 0 or (yte == 0).sum() == 0:
            print(f"  Fold {fi+1}: uebersprungen ({n_test_sz} Anfalls-Fenster).")
            continue

        ranking = rank_channels_raw(train_p)
        top = sorted(c for c, _ in ranking[:k])
        for c in top:
            channel_selection_count[c] += 1

        clf = RawSampleGMM(**gmm_kw).fit(train_p, top)
        s_parts = [clf.window_scores(p, top) for p in test_p]
        s_clf = np.concatenate(s_parts)
        auc_clf = roc_auc_score(yte, s_clf)

        for p, sp in zip(test_p, s_parts):
            yp = p["y"]
            both = yp.min() != yp.max()
            per_patient.append({
                "patient": p["patient"], "fold": fi + 1,
                "y": yp, "frac": p["frac"],
                "score_clf": sp, "score_anom": -sp,
                "auc_clf": float(roc_auc_score(yp, sp)) if both else None,
                "auc_anom": (float(roc_auc_score(yp, -sp)) if both else None),
            })

        test_groups = sorted({groups[j] for j in test_idx})
        label = (test_p[0]["patient"] if len(test_p) == 1
                 else (f"{test_groups[0]}, {len(test_p)} Aufnahmen"
                       if len(test_groups) == 1
                       else f"{len(test_groups)} Subjekte / {len(test_p)} Aufn."))
        print(f"  Fold {fi+1:>2d} [{label}]: AUC={auc_clf:.3f}  "
              f"(K: {clf.k_pos_} Anfall / {clf.k_neg_} Ruhe, "
              f"{n_test_sz} Anfalls-Fenster)")
        fold_rows.append({"fold": fi + 1, "label": label, "auc_clf": auc_clf,
                          "auc_anom": auc_clf, "n_test_sz": n_test_sz})

    if not fold_rows:
        print("Keine auswertbaren Folds.")
        return None

    a = np.array([r["auc_clf"] for r in fold_rows])
    print("\n" + "-" * 60)
    print(f"Ergebnis ueber {len(fold_rows)} Folds:")
    print(f"  Roh-GMM (Messpunkte) : AUC = {a.mean():.3f} +/- {a.std():.3f}")

    order = np.argsort(channel_selection_count)[::-1]
    print(f"\nAm haeufigsten gewaehlte Kanaele (Top {min(k, 10)}):")
    for c in order[:min(k, 10)]:
        if channel_selection_count[c] == 0:
            break
        nm = channel_names[c] if c < len(channel_names) else f"Ch_{c}"
        print(f"  {nm:<12s}: in {channel_selection_count[c]}/{len(fold_rows)}")

    return {"folds": fold_rows, "per_patient": per_patient,
            "channel_names": list(channel_names), "n_dropped_train": 0,
            "auc_clf_mean": float(a.mean()), "auc_clf_std": float(a.std()),
            "auc_anom_mean": float(a.mean()), "auc_anom_std": float(a.std()),
            "channel_selection_count": channel_selection_count.tolist()}


# ===========================================================================
# 5. Selbsttest
# ===========================================================================
def run_selftest():
    from gmm_seizure_detection import _make_synthetic_patient
    print("=" * 64)
    print("SELBSTTEST: Roh-GMM auf Messpunkten (synthetisch)")
    print("=" * 64)
    stride, win_s, step_s = 4, 512 // 4, 256 // 4
    patients = []
    for pid in range(6):
        sig, mask, sr = _make_synthetic_patient(seed=pid)
        n = sig.shape[1]; win, step = 512, 256
        y, frac = [], []
        for s0 in range(0, n - win + 1, step):
            f = float(mask[s0:s0 + win].mean()); frac.append(f); y.append(int(f >= .5))
        patients.append({"patient": f"synt{pid:02d}_s001_t000",
                         "S": np.ascontiguousarray(sig[:, ::stride], dtype=np.float32),
                         "m_sz": mask[::stride],
                         "y": np.asarray(y), "frac": np.asarray(frac),
                         "win_s": win_s, "step_s": step_s})
    print(f"{len(patients)} Patienten, S{patients[0]['S'].shape} "
          f"(Kanaele, Messpunkte)\n")
    r = cross_validate_raw(patients, [f"EEG-{i+1}" for i in range(6)],
                           top_k_channels=3, eval_mode="auto",
                           n_components_grid=(1, 2), max_fit_samples=20000)
    if r is None or not r["folds"]:
        raise SystemExit("[FEHLER] Keine auswertbaren Folds.")
    n_sc = sum(len(p["score_clf"]) for p in r["per_patient"])
    n_y = sum(len(p["y"]) for p in r["per_patient"])
    print(f"\n  Scores: {n_sc}, Fenster: {n_y} -> {'OK' if n_sc == n_y else 'FEHLER'}")
    if n_sc != n_y:
        raise SystemExit("[FEHLER] Score-Zahl passt nicht zur Fensterzahl.")
    print("\n[OK] Roh-GMM laeuft.")


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
    ap.add_argument("--stride", type=int, default=4,
                    help="jeder n-te Messpunkt (4 = 64 statt 256 Hz)")
    ap.add_argument("--top-k", type=int, default=10, help="0 = alle Kanaele")
    ap.add_argument("--eval", choices=["auto", "lopo", "kfold"], default="auto")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--covariance", default="full",
                    choices=["full", "diag", "tied", "spherical"])
    ap.add_argument("--max-fit-samples", type=int, default=200000)
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
    print(f"Lese EDFs (jeder {args.stride}. Messpunkt) ...")
    patients = build_patient_raw(seizure_dict, data_folder, args.window_sec,
                                 args.overlap, args.stride)
    if not patients:
        print("Keine Daten - Abbruch.")
        return

    n_ch = min(p["S"].shape[0] for p in patients)
    for p in patients:
        if p["S"].shape[0] > n_ch:
            p["S"] = p["S"][:n_ch]
    gb = sum(p["S"].nbytes for p in patients) / 1e9
    print(f"\n{len(patients)} Patienten, {n_ch} Kanaele, "
          f"{sum(len(p['y']) for p in patients)} Fenster, {gb:.2f} GB\n")

    result = cross_validate_raw(
        patients, [f"Ch_{i+1}" for i in range(n_ch)],
        top_k_channels=args.top_k, eval_mode=args.eval, n_folds=args.folds,
        covariance_type=args.covariance, max_fit_samples=args.max_fit_samples)
    if result is None:
        return

    result["params"] = {
        "seizure_type": args.seizure_type, "window_sec": args.window_sec,
        "overlap": args.overlap, "top_k": args.top_k,
        "feature_group": "raw-samples", "label_purity": None,
        "eval_mode": args.eval, "group_by_subject": True,
        "stride": args.stride, "covariance": args.covariance,
        "step_sec": args.window_sec * (1.0 - args.overlap), "model": "rawgmm",
    }
    out_dir = args.out_dir or os.path.join(args.base, "Plots", "detector")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.seizure_type.lower()}_rawgmm_w{args.window_sec}_bysubj"
    out_pkl = os.path.join(out_dir, tag + ".pkl")
    with open(out_pkl, "wb") as fh:
        pickle.dump(result, fh)
    print(f"\nErgebnis gespeichert: {out_pkl}")


if __name__ == "__main__":
    main()
