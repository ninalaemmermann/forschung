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


def find_runs_by_key(out_dir, seizure_type):
    """Wie find_runs, aber als {Modell-Schluessel: Pfad}."""
    keys = [k for k, _, _, _ in MODELS]
    out = {}
    for key, label, color, ls, path in _search(out_dir, seizure_type, keys):
        out[key] = path
    return out


def _search(out_dir, seizure_type, keys):
    """Gemeinsame Dateisuche fuer beide Zugriffswege."""
    for key, label, color, ls in MODELS:
        if key not in keys:
            continue
        hits = []
        for f in sorted(os.listdir(out_dir)):
            if not f.endswith(".pkl") or not f.startswith(seizure_type.lower()):
                continue
            rest = f[len(seizure_type):]
            if key == "full":
                if any(k in rest for k, _, _, _ in MODELS if k != "full"):
                    continue
                hits.append(f)
            elif key in rest:
                hits.append(f)
        hits = [h for h in hits if "bysubj" in h]
        if hits:
            yield key, label, color, ls, os.path.join(out_dir, hits[0])


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


# Typ-uebergreifende Abbildungen (Anfallstyp x Modell)
# Fuer die Hauptabbildungen werden nur die drei REPRAESENTATIONEN gezeigt --
# diag und boosted sind Varianten innerhalb des Merkmals-Ansatzes und gehoeren
# in eine Nebentabelle, nicht in die Kernaussage.
# Farben je Paar geprueft (alle Paare): Normalsicht worst Delta-E 24.0,
# CVD worst 6.9 (deutan) -- zulaessig, weil die Serien zusaetzlich durch
# Achsenposition bzw. Strichmuster und direkte Beschriftung getrennt sind.
HEADLINE = [
    ("full",      "Feature-GMM", "#2a78d6", "-"),
    ("histogram", "Histogramm",  "#1baf7a", (0, (1, 1))),
    ("rawgmm",    "Roh-GMM",     "#e34948", (0, (4, 1.5))),
]


def _load_headline(out_dir, seizure_type):
    """Laedt die drei Hauptmodelle zu einem Anfallstyp.

    Zugeordnet wird ueber den Modell-SCHLUESSEL (Dateinamensbestandteil), nicht
    ueber die Beschriftung -- die Hauptabbildungen benennen die Modelle nach der
    Repraesentation ("Feature-GMM"), die Detailabbildungen nach der Variante
    ("GMM full").
    """
    paths = find_runs_by_key(out_dir, seizure_type)
    return [(lab, col, ls, paths[key])
            for key, lab, col, ls in HEADLINE if key in paths]


def plot_matrix(per_type, out_png):
    """AUC je Aufnahme als Punktwolke, gruppiert nach Anfallstyp und Modell.

    Bewusst KEINE Balken mit Mittelwert +/- Std: die Verteilungen sind deutlich
    linksschief und bei GNSZ breit gestreut, ein Mittelwert taeuscht Symmetrie
    vor, die nicht existiert.

    Bewusst AUC und nicht AP auf der y-Achse: die AP je Aufnahme korreliert mit
    r = +0.57 mit der Anfallshaeufigkeit dieser Aufnahme (bei GNSZ zwischen
    0.3 % und 99 %). Ein AP-Punktdiagramm zeigte groesstenteils die Praevalenz,
    nicht die Modellguete. Die gepoolte AP steht deshalb als Zahl darunter --
    dort ist sie sinnvoll, weil ueber alle Aufnahmen gemeinsam gerechnet.
    """
    from sklearn.metrics import average_precision_score

    n_groups = len(per_type)
    fig, ax = plt.subplots(figsize=(2.9 * n_groups + 3.2, 5.8))
    rng = np.random.default_rng(0)
    xt, xl, pos = [], [], 0.0

    for gi, (tname, runs) in enumerate(per_type):
        for lab, color, ls, path in runs:
            r = pickle.load(open(path, "rb"))
            a = np.array([p["auc_clf"] for p in r["per_patient"]
                          if p["auc_clf"] is not None])
            y, s = pooled_normalised(r)
            ap = average_precision_score(y, s)
            prev = float(y.mean())

            jit = rng.uniform(-0.19, 0.19, size=len(a))
            ax.scatter(pos + jit, a, s=26, color=color, alpha=0.45,
                       edgecolor="none", zorder=2)
            med = float(np.median(a))
            ax.plot([pos - 0.34, pos + 0.34], [med, med], color=color, lw=3,
                    zorder=4, solid_capstyle="round")
            ax.text(pos, -0.075, f"AP {ap:.3f}", ha="center", fontsize=8.5,
                    color=color, fontfamily="monospace", weight="bold")
            ax.text(pos, -0.125, f"n={len(a)}", ha="center", fontsize=7.5,
                    color=C_MUTED, fontfamily="monospace")
            xt.append(pos); xl.append(lab)
            pos += 1.0
        # Praevalenz-Hinweis ueber der Gruppe
        ax.text(pos - 2.0, 1.045, f"{tname}   (Prävalenz {prev*100:.1f} %)",
                ha="center", fontsize=11.5, weight="bold")
        pos += 0.9
        if gi < n_groups - 1:
            ax.axvline(pos - 1.45, color=C_GRID, lw=1)

    ax.axhline(0.5, color=C_MUTED, lw=1, ls=":", zorder=1)
    ax.text(-0.55, 0.5, "Zufall", fontsize=8.5, color=C_MUTED, va="bottom")
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=9.5)
    ax.set_xlim(-0.7, pos - 1.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC je Aufnahme (Test = nie im Training gesehen)")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=26)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    # Hinweis oberhalb der Gruppenueberschriften, damit nichts kollidiert
    fig.text(0.005, 0.985, "Ein Punkt = eine Aufnahme, Strich = Median",
             fontsize=10, color=C_MUTED, ha="left", va="top")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_png}")


def plot_pr_by_type(per_type, out_png):
    """PR-Kurven, ein Feld je Anfallstyp, dieselben drei Modelle.

    Die gestrichelte Linie ist das Zufallsniveau und liegt je Feld anders --
    3.1 % bei ABSZ, 22.2 % bei GNSZ. Genau daran sieht man, wie weit die
    Modelle bei GNSZ ans Raten heranruecken, was eine gemeinsame y-Achse
    allein nicht zeigen wuerde.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, axes = plt.subplots(1, len(per_type),
                             figsize=(5.6 * len(per_type), 5.2), sharey=True)
    if len(per_type) == 1:
        axes = [axes]
    for ax, (tname, runs) in zip(axes, per_type):
        prev = None
        for lab, color, ls, path in runs:
            r = pickle.load(open(path, "rb"))
            y, s = pooled_normalised(r)
            prev = float(y.mean())
            prec, rec, _ = precision_recall_curve(y, s)
            ax.plot(rec, prec, color=color, lw=2.2, ls=ls, zorder=3,
                    solid_capstyle="round",
                    label=f"{lab}   AP={average_precision_score(y, s):.3f}")
        ax.axhline(prev, color=C_MUTED, lw=1.3, ls="--", zorder=1)
        # Beschriftung ueber die Linie setzen, nicht darauf -- bei vier
        # schmalen Panels laege sie sonst mitten im Strich
        ax.text(0.985, prev + 0.028, f"Zufall = Prävalenz {prev*100:.1f} %",
                ha="right", va="bottom", color=C_MUTED, fontsize=8.5)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.grid(zorder=0); ax.set_axisbelow(True)
        ax.set_xlabel("Recall (Anteil erkannter Anfalls-Fenster)")
        ax.set_title(tname, fontsize=12, loc="left", weight="bold")
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    axes[0].set_ylabel("Precision (Anteil korrekter Alarme)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_png}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seizure-type", default="gnsz")
    ap.add_argument("--matrix", nargs="+", default=None,
                    help="Typ-uebergreifende Hauptabbildungen, z.B. "
                         "--matrix absz gnsz")
    ap.add_argument("--base", default="/home/data/ninalaemmermann/forschung")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.base, "Plots", "detector")

    if args.matrix:
        per_type = []
        for t in args.matrix:
            runs = _load_headline(out_dir, t)
            print(f"{t.upper()}: {len(runs)} Hauptmodelle")
            if runs:
                per_type.append((t.upper(), runs))
        if not per_type:
            print("Keine Laeufe gefunden.")
            return
        print("\nHauptabbildungen:")
        tag = "_".join(t.lower() for t in args.matrix)
        plot_matrix(per_type, os.path.join(out_dir, f"HAUPT_{tag}_matrix.png"))
        plot_pr_by_type(per_type, os.path.join(out_dir, f"HAUPT_{tag}_pr.png"))
        return

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
