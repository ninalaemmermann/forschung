"""
Modellvergleich in einer Abbildung.

Die Einzelplots aus plot_detector_results.py zeigen jeweils EIN Modell. Fuer den
Bericht braucht es die Gegenueberstellung -- und zwar in genau zwei Bildern, weil
die beiden Kernaussagen unterschiedliche Darstellungen brauchen:

  1. pr_vergleich  -- Precision-Recall-Kurven aller Modelle uebereinander.
     Traegt die zentrale methodische Aussage: AUC und AP widersprechen sich.
     Das Modell mit der besten Rangfolge ist nicht das mit den saubersten
     Alarmen. Ohne diese Abbildung ist der Widerspruch nicht belegbar.

  2. auc_vergleich -- AUC je Aufnahme als Punktwolke, ein Streifen je Modell.
     Traegt die zweite Aussage: nicht der Mittelwert entscheidet, sondern die
     Streuung und die Zahl der Aufnahmen unter Zufallsniveau. Ein Mittelwert
     +/- Std suggeriert eine Normalverteilung, die hier nicht vorliegt.

Beide Abbildungen entstehen aus den gespeicherten .pkl -- kein Neuberechnen.

Nutzung
-------
    python compare_models.py --seizure-type absz
    python compare_models.py --seizure-type gnsz

Die Modelle werden anhand des Dateinamens in Plots/detector/ gefunden.
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# Die Modelle zerfallen in zwei Gruppen, und genau daran haengt die Kernfrage:
# brauchen wir die abgeleiteten Merkmale, oder reicht die rohe Amplitude?
# Deshalb wird nach Gruppe facettiert statt alles in ein Feld zu legen.
#
# Das ist zugleich eine Farbnotwendigkeit: fuenf Serien in einem Feld lassen
# sich nicht mehr so einfaerben, dass alle Paare auch bei Farbfehlsichtigkeit
# unterscheidbar bleiben. Je Feld geprueft (alle Paare, Normalsicht und CVD):
#   Merkmale  #2a78d6 / #eb6834 / #4a3aa7  -- worst Delta-E 13.0 (deutan)
#   Amplitude #1baf7a / #e34948            -- worst Delta-E  6.9 (deutan),
#                                             zulaessig mit Strichmuster + Label
# Die Strichmuster tragen die Unterscheidung zusaetzlich in den SW-Druck.
GROUPS = [
    ("Merkmale (8 je Kanal)", [
        ("full",       "GMM full",    "#2a78d6", "-"),
        ("plain-diag", "GMM diag",    "#eb6834", "--"),
        ("boosted",    "GMM boosted", "#4a3aa7", "-."),
    ]),
    ("Rohe Amplitude", [
        ("histogram",  "Histogramm",  "#1baf7a", (0, (1, 1))),
        ("rawgmm",     "Roh-GMM",     "#e34948", (0, (4, 1.5))),
    ]),
]
MODELS = [m for _, ms in GROUPS for m in ms]

C_TEXT, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": C_MUTED, "axes.labelcolor": C_TEXT, "text.color": C_TEXT,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": C_GRID, "grid.linewidth": 0.6, "font.size": 10,
})


def find_runs(out_dir, seizure_type):
    """Sucht zu einem Anfallstyp die .pkl aller Modelle."""
    found = []
    for key, label, color, ls in MODELS:
        hits = []
        for f in sorted(os.listdir(out_dir)):
            if not f.endswith(".pkl") or not f.startswith(seizure_type.lower()):
                continue
            rest = f[len(seizure_type):]
            if key == "full":
                # der Bestandslauf traegt keinen Modellnamen im Dateinamen
                if any(k in rest for k, _, _, _ in MODELS if k != "full"):
                    continue
                hits.append(f)
            elif key in rest:
                hits.append(f)
        # Nur subjektweise gruppierte Laeufe sind vergleichbar. Aeltere Laeufe
        # ohne "_bysubj" im Namen haben Geschwister-Aufnahmen derselben Person
        # in Training UND Test und sind dadurch zu optimistisch -- die duerfen
        # hier nicht versehentlich gegen die korrigierten antreten.
        hits = [h for h in hits if "bysubj" in h]
        if hits:
            found.append((label, color, ls, os.path.join(out_dir, hits[0])))
        else:
            print(f"  (kein subjektweiser Lauf fuer '{label}' gefunden)")
    return found


def pooled_normalised(result):
    """Testfenster aller Folds, je Fold rangnormalisiert.

    Ohne diese Normalisierung waeren die Scores nicht vergleichbar: jeder Fold
    hat eigenen Standardizer, eigene Kanalauswahl und eigenes Modell, die
    Wertebereiche liegen bis zu fuenf Groessenordnungen auseinander.
    """
    by_fold = defaultdict(list)
    for r in result["per_patient"]:
        by_fold[r["fold"]].append(r)
    Y, S = [], []
    for f in sorted(by_fold):
        recs = by_fold[f]
        s = np.concatenate([r["score_clf"] for r in recs])
        Y.append(np.concatenate([r["y"] for r in recs]))
        S.append(rankdata(s) / len(s))
    return np.concatenate(Y), np.concatenate(S)


def plot_pr(runs, out_png, seizure_type):
    """PR-Kurven, nach Merkmalen und roher Amplitude facettiert.

    Das linke Feld wird im rechten blass wiederholt (und umgekehrt), damit sich
    die Gruppen trotz Trennung direkt vergleichen lassen -- sonst muesste das
    Auge zwischen zwei Achsen hin- und herrechnen.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    lookup = {lab: (col, ls, path) for lab, col, ls, path in runs}

    curves = {}
    prevalence = None
    for lab, (col, ls, path) in lookup.items():
        r = pickle.load(open(path, "rb"))
        y, s = pooled_normalised(r)
        prevalence = float(y.mean())
        prec, rec, _ = precision_recall_curve(y, s)
        curves[lab] = (rec, prec, average_precision_score(y, s), col, ls)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), sharey=True)
    for ax, (gname, members) in zip(axes, GROUPS):
        own = {lab for _, lab, _, _ in members}
        # zuerst die fremden Kurven blass als Kontext
        for lab, (rec, prec, ap, col, ls) in curves.items():
            if lab in own:
                continue
            ax.plot(rec, prec, color=col, lw=1.1, ls=ls, alpha=0.22, zorder=1)
        for lab, (rec, prec, ap, col, ls) in curves.items():
            if lab not in own:
                continue
            ax.plot(rec, prec, color=col, lw=2.2, ls=ls, zorder=3,
                    solid_capstyle="round", label=f"{lab}   AP={ap:.3f}")

        ax.axhline(prevalence, color=C_MUTED, lw=1.1, ls="--", zorder=0)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.grid(zorder=0); ax.set_axisbelow(True)
        ax.set_xlabel("Recall")
        ax.set_title(gname, fontsize=11, loc="left")
        ax.legend(loc="upper right", frameon=False, fontsize=8.8)

    axes[0].set_ylabel("Precision (Anteil korrekter Alarme)")
    axes[0].text(0.015, prevalence + 0.012,
                 f"Zufall = Prävalenz {prevalence*100:.1f}%",
                 color=C_MUTED, fontsize=8.5, va="bottom")
    fig.suptitle(f"Precision-Recall - {seizure_type.upper()}   "
                 f"(Scores fold-weise rangnormalisiert; blass = andere Gruppe)",
                 fontsize=11.5, x=0.012, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_png}")


def plot_auc_spread(runs, out_png, seizure_type):
    fig, ax = plt.subplots(figsize=(7.0, 0.95 * len(runs) + 2.4))
    rng = np.random.default_rng(0)   # feste Jitter-Streuung, reproduzierbar

    for i, (label, color, ls, path) in enumerate(runs):
        r = pickle.load(open(path, "rb"))
        a = np.array([p["auc_clf"] for p in r["per_patient"]
                      if p["auc_clf"] is not None])
        ypos = len(runs) - 1 - i
        jitter = rng.uniform(-0.16, 0.16, size=len(a))
        ax.scatter(a, ypos + jitter, s=22, color=color, alpha=0.5,
                   edgecolor="none", zorder=2)
        # Median als kraeftiger Strich -- robuster als der Mittelwert, weil die
        # Verteilung deutlich linksschief ist
        med = float(np.median(a))
        ax.plot([med, med], [ypos - 0.32, ypos + 0.32], color=color, lw=2.6,
                zorder=4, solid_capstyle="round")
        n_bad = int((a < 0.5).sum())
        ax.text(1.035, ypos, f"med {med:.3f}   <0.5: {n_bad}/{len(a)}",
                fontsize=8.5, va="center", color=C_TEXT,
                fontfamily="monospace")

    ax.axvline(0.5, color=C_MUTED, lw=1, ls=":", zorder=1)
    ax.text(0.5, len(runs) - 0.42, " Zufall", color=C_MUTED, fontsize=8.5)

    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([lab for lab, _, _, _ in runs][::-1], fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.6, len(runs) - 0.3)
    ax.set_xlabel("AUC je Aufnahme (Test = nie im Training gesehen)")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"Streuung ueber die Aufnahmen - {seizure_type.upper()}\n"
                 f"ein Punkt = eine Aufnahme, Strich = Median",
                 fontsize=11.5, loc="left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_png}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seizure-type", default="gnsz")
    ap.add_argument("--base", default="/home/data/ninalaemmermann/forschung")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.base, "Plots", "detector")
    runs = find_runs(out_dir, args.seizure_type)
    if len(runs) < 2:
        print("Zu wenige Laeufe fuer einen Vergleich.")
        return

    print(f"\n{len(runs)} Modelle gefunden:")
    for lab, _, _, p in runs:
        print(f"  {lab:22s} {os.path.basename(p)}")

    print("\nAbbildungen:")
    st = args.seizure_type.lower()
    plot_pr(runs, os.path.join(out_dir, f"{st}_VERGLEICH_pr.png"),
            args.seizure_type)
    plot_auc_spread(runs, os.path.join(out_dir, f"{st}_VERGLEICH_auc.png"),
                    args.seizure_type)


if __name__ == "__main__":
    main()
