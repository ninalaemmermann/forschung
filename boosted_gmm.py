"""
Boosted Gaussian Mixture Ensemble (BGME) fuer die Anfalls-Detektion.

Uebertraegt den Offline-Teil aus Robin Korns Bachelorarbeit "Entwicklung von
Boosted Gaussian Mixture Ensembles und Einsatz zur Anomaliedetektion in
hochdimensionalen, multivariaten Zeitreihen" (HS Ansbach, 2026) auf die
EEG-Pipeline in gmm_seizure_detection.py.

Warum ueberhaupt ein Ensemble?
------------------------------
Ein einzelnes GMM-Paar beschreibt "Anfall" und "Ruhe" durch je eine Wolke. Bei
ABSZ (sehr stereotype Absencen) reicht das -- AUC 0.94. Bei GNSZ, einer
heterogenen Sammelkategorie, bricht es ein -- AUC 0.68. Genau dafuer ist
Boosting gedacht: jeder weitere Basislerner konzentriert sich auf die Fenster,
die seine Vorgaenger schlecht abgebildet haben, und deckt so Teilbereiche ab,
die eine einzelne Wolke verfehlt.

Was uebernommen wurde (und was nicht)
-------------------------------------
Uebernommen ist der Offline-Teil (Kapitel 3.3 und 3.4 der Arbeit):
  - sequentielles Training mit Umgewichtung der schwer modellierbaren Fenster
  - Score-Aggregation ueber "mean_loglik" oder "logmeanexp"
  - die Zwei-Ensemble-Entscheidung ("separation", Kapitel 3.6.3): der Score ist
    die Differenz der Ensemble-Log-Likelihoods von Anfall und Ruhe
  - covariance_type="diag" als Default (die Arbeit nutzt das fuer 38 Dimensionen;
    wir haben hier 80 bei teils nur 150-300 Anfalls-Fenstern pro Fold -- volle
    80x80-Kovarianzen sind da nicht schaetzbar und trieben die Scores auf 1e7)

NICHT uebernommen ist die gesamte Online-Komponente (Kapitel 3.5-3.7):
Auto-Refit, alpha_decay_rate, Label-Feedback, Ensemble-Management. Die setzt
einen laufenden Datenstrom mit nachtraeglich eintreffenden Annotationen voraus;
hier wird offline ueber feste Aufnahmen kreuzvalidiert. Die Ensemblegewichte
bleiben deshalb konstant gleichverteilt.

Gewichtung ohne sample_weight
-----------------------------
sklearns GaussianMixture.fit(X) kennt kein sample_weight. Die Arbeit loest das
per Resampling mit Zuruecklegen: Fenster mit hohem Gewicht werden haeufiger
gezogen. Dasselbe Verfahren wird hier verwendet -- mit festem Seed, damit die
Laeufe reproduzierbar bleiben.

Nutzung
-------
    # Boosted Ensemble gegen den bisherigen Einzel-GMM
    python boosted_gmm.py --seizure-type absz --window-sec 2

    # nur der Kovarianz-Wechsel, ohne Boosting (isoliert den Effekt)
    python boosted_gmm.py --seizure-type absz --window-sec 2 --model plain-diag

    # Selbsttest ohne echte Daten
    python boosted_gmm.py --selftest

Das Ergebnis landet als .pkl im selben Format wie bei plot_detector_results.py,
die Plots lassen sich also direkt daraus zeichnen:
    python plot_detector_results.py --from-pkl <pfad zur pkl>
"""

import argparse
import os
import pickle

import numpy as np

from gmm_seizure_detection import (FEATURE_GROUPS, GMMClassifier,
                                   cross_validate_patients, run_on_edf)


# ===========================================================================
# 1. Ein Ensemble (eine Klasse)
# ===========================================================================
class BoostedGMM:
    """Ensemble aus n_estimators GMMs, sequentiell mit Boosting gefittet.

    Modelliert die Dichte EINER Klasse. Das erste Modell ist die gewoehnliche
    Maximum-Likelihood-Schaetzung; jedes weitere sieht die Daten umgewichtet,
    sodass die vom Vorgaenger schlecht erklaerten Fenster staerker durchschlagen.
    """

    def __init__(self, n_estimators=6, n_components=5, covariance_type="diag",
                 reg_covar=2e-4, max_iter=200, tol=1e-3, learning_rate=0.7,
                 hard_quantile=0.9, min_weight=1e-6, max_weight=1000.0,
                 combine="logmeanexp", random_state=42):
        self.n_estimators = n_estimators
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.hard_quantile = hard_quantile
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.combine = combine
        self.random_state = random_state
        self.models_ = []

    def fit(self, X):
        from sklearn.mixture import GaussianMixture

        n = len(X)
        rng = np.random.default_rng(self.random_state)
        # Startgewichte gleichverteilt -> erstes Modell ist die klassische
        # Maximum-Likelihood-Schaetzung auf allen Fenstern.
        w = np.full(n, 1.0 / n)
        self.models_ = []

        # Mehr Komponenten als Fenster gehen nicht; bei winzigen Klassen
        # (ABSZ hat pro Fold nur ~150-300 Anfalls-Fenster) runterregeln.
        k = max(1, min(self.n_components, n))

        for m in range(self.n_estimators):
            if m == 0:
                Xb = X
            else:
                # Resampling statt sample_weight -- sklearn kennt keine
                # gewichtete Anpassung. Haeufiger gezogen = staerker gewichtet.
                idx = rng.choice(n, size=n, replace=True, p=w)
                Xb = X[idx]

            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state + m,
            )
            gmm.fit(Xb)
            self.models_.append(gmm)

            if m == self.n_estimators - 1:
                break

            # Gewichte fuer den naechsten Basislerner: die am schlechtesten
            # erklaerten Fenster (unterstes Quantil der Log-Likelihood) hoch.
            ll = gmm.score_samples(X)
            q = np.quantile(ll, 1.0 - self.hard_quantile)
            hard = (ll <= q).astype(float)
            w = w * np.exp(self.learning_rate * hard)
            w = np.clip(w, self.min_weight, self.max_weight)
            s = w.sum()
            w = w / s if s > 0 else np.full(n, 1.0 / n)

        return self

    def score_samples(self, X):
        """Aggregierte Log-Likelihood des Ensembles."""
        ll = np.vstack([g.score_samples(X) for g in self.models_])
        if self.combine == "mean_loglik":
            return ll.mean(axis=0)
        # logmeanexp: Log-Likelihood der Mischdichte, numerisch stabil
        # (Kapitel 3.4.2). Gleichverteilte Ensemblegewichte, da offline.
        a = np.full(len(self.models_), 1.0 / len(self.models_))
        mx = ll.max(axis=0)
        return mx + np.log(np.sum(a[:, None] * np.exp(ll - mx[None, :]), axis=0))


# ===========================================================================
# 2. Zwei Ensembles -> Entscheidung (entspricht scoring_mode "separation")
# ===========================================================================
class BoostedGMMClassifier:
    """Zwei Ensembles (Anfall / Ruhe), Entscheidung ueber die Differenz.

    Schnittstelle absichtlich identisch zu GMMClassifier -- fit(X, y) und
    decision_function(X) -- damit sich das Modell ohne weitere Aenderungen in
    cross_validate_patients einhaengen laesst.
    """

    def __init__(self, standardize=True, **kw):
        self.standardize = standardize
        self.kw = kw
        self.ens_pos = None
        self.ens_neg = None
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

    def fit(self, X, y):
        Xs = self._scale(X, fit=True)
        Xp, Xn = Xs[y == 1], Xs[y == 0]
        if len(Xp) == 0 or len(Xn) == 0:
            raise ValueError("Beide Klassen muessen Beispiele enthalten.")
        self.ens_pos = BoostedGMM(**self.kw).fit(Xp)
        # zweites Ensemble mit verschobenem Seed, sonst zoegen beide Klassen
        # dieselbe Resampling-Folge
        kw_neg = dict(self.kw)
        kw_neg["random_state"] = kw_neg.get("random_state", 42) + 1000
        self.ens_neg = BoostedGMM(**kw_neg).fit(Xn)
        self.log_prior_ratio = np.log(len(Xp) / len(Xn))
        return self

    def decision_function(self, X):
        Xs = self._scale(X)
        return (self.ens_pos.score_samples(Xs)
                - self.ens_neg.score_samples(Xs)
                + self.log_prior_ratio)

    def predict(self, X, threshold=0.0):
        return (self.decision_function(X) > threshold).astype(int)


# ===========================================================================
# 3. Fabriken zum Einhaengen in cross_validate_patients
# ===========================================================================
def make_factory(model, **kw):
    """Liefert eine parameterlose Funktion, die das gewuenschte Modell baut.

    "boosted"    -- das Ensemble aus dieser Datei
    "plain-diag" -- der bestehende GMMClassifier, nur mit diagonaler Kovarianz
                    und groesserem reg_covar. Isoliert den Kovarianz-Effekt vom
                    Boosting-Effekt: sonst weiss man hinterher nicht, welche der
                    beiden Aenderungen gewirkt hat.
    "plain"      -- der Bestand, unveraendert (Referenz)
    """
    if model == "boosted":
        return lambda: BoostedGMMClassifier(**kw)
    if model == "plain-diag":
        return lambda: GMMClassifier(covariance_type="diag")
    if model == "plain":
        return lambda: GMMClassifier()
    raise ValueError(f"unbekanntes Modell: {model}")


# ===========================================================================
# 4. Selbsttest
# ===========================================================================
def run_selftest():
    from gmm_seizure_detection import (_make_synthetic_patient,
                                       extract_window_features)
    print("=" * 64)
    print("SELBSTTEST: Boosted Ensemble gegen Einzel-GMM (synthetisch)")
    print("=" * 64)
    patients = []
    for pid in range(6):
        sig, mask, sr = _make_synthetic_patient(seed=pid)
        X, y, frac = extract_window_features(sig, sr, mask, 2.0, 0.5)
        patients.append({"patient": f"synt{pid:02d}_s001_t000", "X": X, "y": y,
                         "frac": frac})
    ch = [f"EEG-{k+1}" for k in range(6)]

    results = {}
    for name in ("plain", "plain-diag", "boosted"):
        print(f"\n--- {name} ---")
        r = cross_validate_patients(
            patients, ch, top_k_channels=3, eval_mode="auto",
            classifier_factory=make_factory(name, n_estimators=3,
                                            n_components=2))
        if r is None or not r["folds"]:
            raise SystemExit(f"[FEHLER] {name}: kein Ergebnis.")
        results[name] = r["auc_clf_mean"]

    print("\n" + "-" * 60)
    for k, v in results.items():
        print(f"  {k:12s}: AUC = {v:.3f}")

    # Determinismus pruefen: das Resampling darf die Reproduzierbarkeit nicht
    # zerstoeren, sonst sind Vergleichslaeufe wertlos.
    import io, contextlib
    scores = []
    for _ in range(2):
        with contextlib.redirect_stdout(io.StringIO()):
            r = cross_validate_patients(
                patients, ch, top_k_channels=3, eval_mode="auto",
                classifier_factory=make_factory("boosted", n_estimators=3,
                                                n_components=2))
        scores.append(np.concatenate([p["score_clf"] for p in r["per_patient"]]))
    same = np.array_equal(scores[0], scores[1])
    print(f"\n  Zwei Laeufe bitgenau identisch: {same}")
    if not same:
        raise SystemExit("[FEHLER] Boosting ist nicht reproduzierbar.")

    print("\n[OK] Boosted Ensemble laeuft und ist deterministisch.")


# ===========================================================================
# 5. CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", choices=["boosted", "plain-diag", "plain"],
                    default="boosted")
    ap.add_argument("--seizure-type", default="gnsz")
    ap.add_argument("--base", default="/home/data/ninalaemmermann/forschung")
    ap.add_argument("--window-sec", type=float, default=5.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--features", choices=list(FEATURE_GROUPS.keys()),
                    default="all")
    ap.add_argument("--label-purity", type=float, default=1.0)
    ap.add_argument("--eval", choices=["auto", "lopo", "kfold"], default="auto")
    ap.add_argument("--folds", type=int, default=5)
    # Ensemble-Parameter (Defaults nach Tabelle 5.1 der Arbeit, an die
    # kleineren Klassen hier angepasst)
    ap.add_argument("--n-estimators", type=int, default=6)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--covariance", default="diag",
                    choices=["diag", "full", "tied", "spherical"])
    ap.add_argument("--reg-covar", type=float, default=2e-4)
    ap.add_argument("--learning-rate", type=float, default=0.7)
    ap.add_argument("--hard-quantile", type=float, default=0.9)
    ap.add_argument("--combine", default="logmeanexp",
                    choices=["logmeanexp", "mean_loglik"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    kw = dict(n_estimators=args.n_estimators, n_components=args.n_components,
              covariance_type=args.covariance, reg_covar=args.reg_covar,
              learning_rate=args.learning_rate,
              hard_quantile=args.hard_quantile, combine=args.combine)

    print(f"Modell: {args.model}")
    if args.model == "boosted":
        print(f"  {args.n_estimators} Basislerner x {args.n_components} "
              f"Komponenten, {args.covariance}, reg_covar={args.reg_covar}, "
              f"lr={args.learning_rate}, hard_quantile={args.hard_quantile}, "
              f"combine={args.combine}")
    print()

    result = run_on_edf(
        args.seizure_type, args.base, window_sec=args.window_sec,
        overlap=args.overlap, top_k_channels=args.top_k, eval_mode=args.eval,
        n_folds=args.folds, feature_group=args.features,
        label_purity=args.label_purity,
        classifier_factory=make_factory(args.model, **kw),
    )
    if result is None:
        print("Kein Ergebnis.")
        return

    result["params"] = {
        "seizure_type": args.seizure_type, "window_sec": args.window_sec,
        "overlap": args.overlap, "top_k": args.top_k,
        "feature_group": args.features, "label_purity": args.label_purity,
        "eval_mode": args.eval, "group_by_subject": True,
        "step_sec": args.window_sec * (1.0 - args.overlap),
        "model": args.model, **({} if args.model != "boosted" else kw),
    }

    out_dir = args.out_dir or os.path.join(args.base, "Plots", "detector")
    os.makedirs(out_dir, exist_ok=True)
    tag = (f"{args.seizure_type.lower()}_{args.model}_p{args.label_purity}"
           f"_w{args.window_sec}_{args.features}_bysubj")
    out_pkl = os.path.join(out_dir, tag + ".pkl")
    with open(out_pkl, "wb") as fh:
        pickle.dump(result, fh)
    print(f"\nErgebnis gespeichert: {out_pkl}")
    print(f"Plots zeichnen mit:\n"
          f"  python plot_detector_results.py --from-pkl {out_pkl}")


if __name__ == "__main__":
    main()
