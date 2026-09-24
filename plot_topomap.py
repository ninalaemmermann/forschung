"""
Topographische Karte der Kanalauswahl.

Warum diese Abbildung
---------------------
Die Kreuzvalidierung waehlt in jedem Fold die trennschaerfsten Kanaele per
JS-Divergenz -- und zwar nur auf den Trainingspatienten. Wie oft ein Kanal
dabei gewaehlt wird, ist damit ein leckfreies Mass dafuer, wo die Information
sitzt. Als Zahlenliste ist das verschenkt; auf einem Kopf aufgetragen wird
daraus ein neurologisch pruefbares Ergebnis.

Bei ABSZ (Absencen) faellt die Auswahl auf frontale und zentrale Elektroden --
was zur frontal betonten generalisierten Spike-Wave-Aktivitaet passt. Ob das
bei anderen Anfallstypen genauso ist, beantwortet dieselbe Abbildung.

Gezeigt wird der ANTEIL der Folds, nicht die absolute Zahl: ABSZ hat 10 Folds,
GNSZ nur 5, die Rohzahlen waeren nicht vergleichbar.

Nutzung
-------
    python plot_topomap.py --types absz gnsz
    python plot_topomap.py --types absz --model rawgmm
"""

import argparse
import os
import pickle
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_TEXT, C_MUTED = "#0b0b0b", "#52514e"


def clean_names(raw):
    """'EEG F4_a10' -> 'F4' (Schreibweise des Standard-10-20-Montage)."""
    import mne
    montage = mne.channels.make_standard_montage("standard_1020")
    lookup = {n.lower(): n for n in montage.ch_names}
    out = []
    for c in raw:
        n = re.sub(r"^EEG\s*", "", c).replace("_a10", "").strip()
        out.append(lookup.get(n.lower(), n))
    return out


def load_selection(pkl_path):
    """Anteil der Folds je Kanal + bereinigte Namen."""
    r = pickle.load(open(pkl_path, "rb"))
    cnt = np.asarray(r["channel_selection_count"], dtype=float)
    n_folds = len(r["folds"])
    return clean_names(r["channel_names"]), cnt / max(n_folds, 1), n_folds


def find_pkl(out_dir, seizure_type, model):
    """Sucht den passenden subjektweisen Lauf."""
    cands = []
    for f in sorted(os.listdir(out_dir)):
        if not (f.endswith(".pkl") and f.startswith(seizure_type.lower())
                and "bysubj" in f):
            continue
        rest = f[len(seizure_type):]
        if model == "full":
            # der unveraenderte Merkmals-Lauf traegt keinen Modellnamen
            if any(k in rest for k in
                   ("plain-diag", "boosted", "histogram", "rawgmm")):
                continue
        elif model not in rest:
            continue
        cands.append(f)
    return os.path.join(out_dir, cands[0]) if cands else None


def plot_topomaps(entries, out_png, model_label):
    """Ein Kopf je Anfallstyp, Farbe = Anteil der Folds mit diesem Kanal."""
    import mne
    mne.set_log_level("ERROR")

    fig, axes = plt.subplots(1, len(entries),
                             figsize=(4.6 * len(entries) + 2.0, 5.2))
    if len(entries) == 1:
        axes = [axes]

    im = None
    # Elektrodennamen kleiner setzen -- am Vertex draengen sich Fp1/Fpz/Fp2
    with plt.rc_context({"font.size": 7.0}):
        for ax, (tname, names, frac, n_folds) in zip(axes, entries):
            info = mne.create_info(list(names), sfreq=256.0, ch_types="eeg")
            info.set_montage("standard_1020", on_missing="ignore")
            # Einheitliche Skala 0..1, damit die Koepfe vergleichbar sind
            im, _ = mne.viz.plot_topomap(
                frac, info, axes=ax, show=False, cmap="YlOrRd",
                vlim=(0, 1), contours=4, sensors=True,
                names=list(names), image_interp="cubic")
            ax.set_title(f"{tname}\n{n_folds} Folds", fontsize=12,
                         color=C_TEXT, pad=14)

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.06, fraction=0.045)
    cbar.set_label("Anteil der Folds mit diesem Kanal", fontsize=9.5,
                   color=C_TEXT, labelpad=10)
    cbar.ax.tick_params(labelsize=8.5, color=C_MUTED)

    fig.suptitle(f"Kanalauswahl per JS-Divergenz — {model_label}",
                 fontsize=13, x=0.02, ha="left", y=0.99)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white",
                pad_inches=0.35)
    plt.close(fig)
    print(f"  {out_png}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", nargs="+", default=["absz", "gnsz"])
    ap.add_argument("--model", default="full",
                    choices=["full", "plain-diag", "boosted", "rawgmm"],
                    help="histogram entfaellt: dieser Pfad speichert nur "
                         "generische Kanalnamen (Ch_1..Ch_27)")
    ap.add_argument("--base", default="/home/data/ninalaemmermann/forschung")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.base, "Plots", "detector")
    entries = []
    for t in args.types:
        p = find_pkl(out_dir, t, args.model)
        if not p:
            print(f"  (kein '{args.model}'-Lauf fuer {t.upper()})")
            continue
        names, frac, n_folds = load_selection(p)
        entries.append((t.upper(), names, frac, n_folds))
        top = np.argsort(frac)[::-1][:5]
        print(f"{t.upper()}: {os.path.basename(p)}")
        print("   Top 5: " + ", ".join(f"{names[c]} ({frac[c]*100:.0f}%)"
                                       for c in top))
        print(f"   ueberhaupt gewaehlt: {int((frac > 0).sum())} von "
              f"{len(names)} Kanaelen")

    if not entries:
        print("Nichts zu zeichnen.")
        return
    print("\nAbbildung:")
    tag = "_".join(t.lower() for t in args.types)
    plot_topomaps(entries, os.path.join(out_dir, f"TOPO_{tag}_{args.model}.png"),
                  {"full": "Feature-GMM", "plain-diag": "GMM diag",
                   "boosted": "GMM boosted", "rawgmm": "Roh-GMM"}[args.model])


if __name__ == "__main__":
    main()
