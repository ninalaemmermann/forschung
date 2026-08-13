"""
Auswertung und Plots fuer den GMM-Anfallsdetektor.

Trennt Rechnen von Zeichnen: der Lauf wird einmal gerechnet und samt aller
Fenster-Scores in eine .pkl gelegt; die Plots entstehen daraus. Neue Plots
brauchen also keinen neuen Kreuzvalidierungs-Lauf (bei ABSZ ~20 Minuten).

Erzeugt drei Abbildungen:

  1. auc_per_patient  -- AUC je Patient als Punktdiagramm.
     Ein Mittelwert +/- Std verschleiert, dass die Fold-Ergebnisse alles andere
     als normalverteilt sind (bei ABSZ: 16 Patienten gut, einer invertiert).
     Der Plot zeigt die tatsaechliche Streuung.

  2. precision_recall -- PR-Kurve mit Average Precision.
     Wichtiger als die ROC-Kurve, weil die Klassen extrem unbalanciert sind
     (ABSZ: ~3% Anfalls-Fenster). Bei so wenig Positiven ist die ROC-AUC
     systematisch zu optimistisch -- die vielen echten Negative druecken die
     False-Positive-Rate, egal wie brauchbar der Detektor ist. Die PR-Kurve
     beantwortet dagegen die klinisch relevante Frage: wie oft stimmt ein Alarm?
     Die gestrichelte Linie ist die Prävalenz, also das Zufallsniveau.

  3. score_timeline  -- Detektor-Score ueber die Aufnahmezeit je Patient,
     echte Anfaelle grau hinterlegt. Das Diagnosewerkzeug: hier sieht man,
     ob ein Patient durchgehend falsch bewertet wird oder ob einzelne
     Artefakte den Score kippen.

Nutzung
-------
    # rechnen + speichern + alle Plots (Default: gnsz, 5s-Fenster)
    python plot_detector_results.py --label-purity 1.0

    # anderer Anfallstyp (absz braucht wegen kurzer Anfaelle 2s-Fenster)
    python plot_detector_results.py --seizure-type absz --window-sec 2

    # nur neu zeichnen aus einem vorhandenen Lauf (schnell)
    python plot_detector_results.py --from-pkl Plots/detector/gnsz_p1.0_w5.0_all_bysubj.pkl

Die Ergebnisse landen unter Plots/detector/<typ>_p<purity>[...] .
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # keine Display-Verbindung noetig (Server/SSH)
import matplotlib.pyplot as plt

from gmm_seizure_detection import FEATURE_GROUPS, run_on_edf

# Farben aus einer validierten kategorialen Palette (blau/orange trennen auch
# bei Rot-Gruen-Schwaeche und in Graustufen; Grautoene tragen keine Identitaet,
# sondern nur Kontext wie Anfallsbalken und Nulllinien).
C_CLF = "#2a78d6"     # GMM-Klassifikator
C_ANOM = "#eb6834"    # Anomalie-Detektor
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#d8d7d2"
C_SEIZURE = "#c9c8c2"  # Hinterlegung der echten Anfallszeiten

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": C_MUTED,
    "axes.labelcolor": C_TEXT,
    "text.color": C_TEXT,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "font.size": 10,
})


# ===========================================================================
# Lauf rechnen / laden
# ===========================================================================
def compute_and_save(seizure_type, base, out_pkl, window_sec, overlap,
                     top_k, feature_group, label_purity, eval_mode, n_folds,
                     group_by_subject=True):
    """Rechnet die Kreuzvalidierung und legt Ergebnis + Parameter als .pkl ab."""
    result = run_on_edf(
        seizure_type, base, window_sec=window_sec, overlap=overlap,
        top_k_channels=top_k, eval_mode=eval_mode, n_folds=n_folds,
        feature_group=feature_group, label_purity=label_purity,
        group_by_subject=group_by_subject,
    )
    if result is None:
        return None

    # Fensterschritt mitspeichern -> Zeitachse in Plot 3 rekonstruierbar
    result["params"] = {
        "seizure_type": seizure_type, "window_sec": window_sec,
        "overlap": overlap, "top_k": top_k, "feature_group": feature_group,
        "label_purity": label_purity, "eval_mode": eval_mode,
        "group_by_subject": group_by_subject,
        "step_sec": window_sec * (1.0 - overlap),
    }
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, "wb") as fh:
        pickle.dump(result, fh)
    print(f"\nErgebnis gespeichert: {out_pkl}")
    return result


def load_result(pkl_path):
    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)


def _pooled(result):
    """Testfenster aller Folds zusammenfuehren -- fold-weise rangnormalisiert.

    Roh zusammengeworfen waeren die Scores NICHT vergleichbar: jeder Fold hat
    einen eigenen Standardizer, eine eigene Kanalauswahl und ein eigenes GMM.
    Gemessen an ABSZ unterscheiden sich die Wertebereiche der Folds um fuenf
    Groessenordnungen (Fold 5: -2.4e7, Fold 10: -863). Ein globaler Schwellen-
    Durchlauf wuerde dann primaer nach Fold sortieren statt nach Iktalitaet und
    die Average Precision beschoenigen (0.698 roh gegenueber 0.645 korrekt).

    Deshalb wird innerhalb jedes Folds auf Raenge in (0, 1] abgebildet. Das
    laesst die Reihenfolge je Fold unveraendert -- AUC und AP sind rangbasiert,
    verlieren also nichts -- macht die Folds aber erst vergleichbar.
    """
    from scipy.stats import rankdata

    by_fold = defaultdict(list)
    for r in result["per_patient"]:
        by_fold[r["fold"]].append(r)

    y, s_clf, s_anom = [], [], []
    for f in sorted(by_fold):
        recs = by_fold[f]
        y.append(np.concatenate([r["y"] for r in recs]))
        for key, out in (("score_clf", s_clf), ("score_anom", s_anom)):
            s = np.concatenate([r[key] for r in recs])
            out.append(rankdata(s) / len(s))
    return (np.concatenate(y), np.concatenate(s_clf), np.concatenate(s_anom))


def _per_fold_scores(result, key):
    """(y, score) je Fold -- fuer Kennzahlen, die pro Fold gerechnet werden."""
    by_fold = defaultdict(list)
    for r in result["per_patient"]:
        by_fold[r["fold"]].append(r)
    out = []
    for f in sorted(by_fold):
        recs = by_fold[f]
        yy = np.concatenate([r["y"] for r in recs])
        ss = np.concatenate([r[key] for r in recs])
        if yy.min() != yy.max():
            out.append((yy, ss))
    return out


# ===========================================================================
# Plot 1: AUC je Patient
# ===========================================================================
def plot_auc_per_patient(result, out_png):
    recs = [r for r in result["per_patient"] if r["auc_clf"] is not None]
    if not recs:
        print("  (keine Patienten mit beiden Klassen - Plot uebersprungen)")
        return
    recs.sort(key=lambda r: r["auc_clf"])
    names = [r["patient"] for r in recs]
    a_clf = np.array([r["auc_clf"] for r in recs])
    a_anom = np.array([r["auc_anom"] for r in recs])
    ypos = np.arange(len(recs))

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(recs) + 2.2))

    # Verbindungslinie macht die Differenz je Patient lesbar
    for i in range(len(recs)):
        ax.plot([a_clf[i], a_anom[i]], [ypos[i], ypos[i]],
                color=C_GRID, lw=1.4, zorder=1, solid_capstyle="round")

    ax.scatter(a_clf, ypos, s=58, color=C_CLF, zorder=3,
               edgecolor="white", linewidth=1.4, label="GMM-Klassifikator")
    ax.scatter(a_anom, ypos, s=58, color=C_ANOM, zorder=3,
               edgecolor="white", linewidth=1.4, label="Anomalie-Detektor")

    ax.axvline(0.5, color=C_MUTED, lw=1, ls=":", zorder=0)
    ax.text(0.5, len(recs) - 0.3, " Zufall", color=C_MUTED, fontsize=8,
            va="center")
    for vals, col in ((a_clf, C_CLF), (a_anom, C_ANOM)):
        ax.axvline(vals.mean(), color=col, lw=1.2, ls="--", alpha=0.55, zorder=0)

    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.045)
    ax.set_ylim(-0.8, len(recs) - 0.2)
    ax.set_xlabel("AUC (Test = dieser Patient, nie im Training)")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    p = result.get("params", {})
    ax.set_title(
        f"AUC je Patient - {p.get('seizure_type','?').upper()}, "
        f"Fenster {p.get('window_sec','?')}s, Reinheit {p.get('label_purity','?')}\n"
        f"Klassifikator {a_clf.mean():.3f}+/-{a_clf.std():.3f}   "
        f"Anomalie {a_anom.mean():.3f}+/-{a_anom.std():.3f}",
        fontsize=11, loc="left", color=C_TEXT)
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    # Ausreisser direkt beschriften -- der Grund fuer diesen Plot
    worst = int(np.argmin(a_clf))
    if a_clf[worst] < 0.75:
        ax.annotate(f"{a_clf[worst]:.2f}", (a_clf[worst], ypos[worst]),
                    xytext=(6, -12), textcoords="offset points",
                    fontsize=8, color=C_CLF, weight="bold")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_png}")


# ===========================================================================
# Plot 2: Precision-Recall
# ===========================================================================
def plot_precision_recall(result, out_png):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.metrics import roc_auc_score

    y, s_clf, s_anom = _pooled(result)
    prevalence = float(y.mean())

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for score, col, name, key in ((s_clf, C_CLF, "GMM-Klassifikator", "score_clf"),
                                  (s_anom, C_ANOM, "Anomalie-Detektor", "score_anom")):
        prec, rec, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        auc = roc_auc_score(y, score)
        # AP zusaetzlich je Fold -- unabhaengig von der Rangnormalisierung
        fold_ap = [average_precision_score(yy, ss)
                   for yy, ss in _per_fold_scores(result, key)]
        ax.plot(rec, prec, color=col, lw=2, solid_capstyle="round",
                label=f"{name}\nAP={ap:.3f}  (ROC-AUC={auc:.3f})\n"
                      f"je Fold: {np.mean(fold_ap):.3f}+/-{np.std(fold_ap):.3f}")

    ax.axhline(prevalence, color=C_MUTED, lw=1.2, ls="--")
    ax.text(0.015, prevalence + 0.015,
            f"Zufall = Prävalenz {prevalence*100:.1f}%",
            color=C_MUTED, fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("Recall (Anteil erkannter Anfalls-Fenster)")
    ax.set_ylabel("Precision (Anteil korrekter Alarme)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)

    p = result.get("params", {})
    n_pos = int(y.sum())
    ax.set_title(
        f"Precision-Recall - {p.get('seizure_type','?').upper()}\n"
        f"{n_pos} von {len(y)} Fenstern iktal ({prevalence*100:.1f}%);\n"
        f"bei dieser Schieflage ist die ROC-AUC zu optimistisch\n"
        f"Scores fold-weise rangnormalisiert (sonst nicht vergleichbar)",
        fontsize=10, loc="left", color=C_TEXT)
    ax.legend(loc="lower left", frameon=False, fontsize=8,
              bbox_to_anchor=(0.0, 0.06))

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_png}")


# ===========================================================================
# Plot 3: Score-Zeitverlauf je Patient
# ===========================================================================
def plot_score_timelines(result, out_dir, which="clf"):
    """Eine PNG je Patient: Detektor-Score ueber die Zeit, Anfaelle hinterlegt."""
    os.makedirs(out_dir, exist_ok=True)
    step = result.get("params", {}).get("step_sec", 1.0)
    key = "score_clf" if which == "clf" else "score_anom"
    col = C_CLF if which == "clf" else C_ANOM
    title_name = ("GMM-Klassifikator" if which == "clf" else "Anomalie-Detektor")

    for r in result["per_patient"]:
        score = r[key]
        t = np.arange(len(score)) * step / 60.0  # Minuten
        fig, ax = plt.subplots(figsize=(11, 2.9))

        # Echte Anfallszeiten als Band im Hintergrund (aus dem iktalen Anteil)
        ictal = r["frac"] > 0
        if ictal.any():
            edges = np.diff(np.concatenate([[0], ictal.astype(int), [0]]))
            for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                ax.axvspan(t[a], t[min(b, len(t) - 1)], color=C_SEIZURE,
                           zorder=0, lw=0)

        ax.plot(t, score, color=col, lw=0.9, zorder=2)
        if which == "clf":
            # Entscheidungsschwelle des Klassifikators (predict nutzt 0.0)
            ax.axhline(0.0, color=C_MUTED, lw=1, ls=":", zorder=1)

        # Robuste y-Grenzen: einzelne Artefakte (z.B. Einschwingen der
        # Elektroden in den ersten Sekunden) erreichen Scores von mehreren
        # Tausend und wuerden den interessanten Bereich sonst platt druecken.
        lo, hi = np.percentile(score, [0.5, 99.5])
        pad = max((hi - lo) * 0.12, 1e-9)
        n_clip = int(((score < lo - pad) | (score > hi + pad)).sum())
        ax.set_ylim(lo - pad, hi + pad)

        auc = r["auc_clf"] if which == "clf" else r["auc_anom"]
        auc_txt = f"AUC {auc:.3f}" if auc is not None else "AUC n/a"
        clip_txt = f" - {n_clip} Ausreisser ausserhalb" if n_clip else ""
        ax.set_title(f"{r['patient']} - {title_name} - {auc_txt}   "
                     f"(grau = annotierter Anfall){clip_txt}",
                     fontsize=10, loc="left", color=C_TEXT)
        ax.set_xlabel("Zeit [min]")
        ax.set_ylabel("Score")
        ax.set_xlim(t.min(), t.max())
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

        out_png = os.path.join(out_dir, f"score_{which}_{r['patient']}.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"  {out_dir}/score_{which}_*.png  ({len(result['per_patient'])} Patienten)")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seizure-type", default="gnsz")
    ap.add_argument("--base", default="/home/data/ninalaemmermann/forschung")
    ap.add_argument("--out-dir", default=None,
                    help="Zielordner (Default: <base>/Plots/detector).")
    ap.add_argument("--from-pkl", default=None,
                    help="Nur neu zeichnen aus diesem gespeicherten Lauf.")
    ap.add_argument("--label-purity", type=float, default=1.0)
    ap.add_argument("--window-sec", type=float, default=5.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--features", choices=list(FEATURE_GROUPS.keys()),
                    default="all")
    ap.add_argument("--eval", choices=["auto", "lopo", "kfold"], default="auto")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-timelines", action="store_true",
                    help="Plot 3 (eine PNG je Patient) ueberspringen.")
    ap.add_argument("--no-subject-grouping", action="store_true",
                    help="Folds nach Datei statt nach Subjekt (schoent die AUC).")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.base, "Plots", "detector")
    os.makedirs(out_dir, exist_ok=True)

    if args.from_pkl:
        result = load_result(args.from_pkl)
        tag = os.path.splitext(os.path.basename(args.from_pkl))[0]
    else:
        tag = (f"{args.seizure_type.lower()}_p{args.label_purity}"
               f"_w{args.window_sec}_{args.features}"
               f"{'_bydatei' if args.no_subject_grouping else '_bysubj'}")
        result = compute_and_save(
            args.seizure_type, args.base, os.path.join(out_dir, tag + ".pkl"),
            args.window_sec, args.overlap, args.top_k, args.features,
            args.label_purity, args.eval, args.folds,
            group_by_subject=not args.no_subject_grouping)
        if result is None:
            print("Kein Ergebnis - keine Plots.")
            return

    print("\nPlots:")
    plot_auc_per_patient(result, os.path.join(out_dir, f"{tag}_auc_per_patient.png"))
    plot_precision_recall(result, os.path.join(out_dir, f"{tag}_precision_recall.png"))
    if not args.no_timelines:
        plot_score_timelines(result, os.path.join(out_dir, f"{tag}_timelines"))


if __name__ == "__main__":
    main()
