"""
Änderung Ruhe → Anfall je Absence: Datenvarianz, Rauschschätzung R und SNR.

Erklärt, warum λ = 1 + SNR bei TUH-ABSZ kaum reagiert, bei Erlangen aber schon:
λ ist ein Verhältnis aus Datenvarianz (C) und Rauschvarianz (R). Steigen beide
gleich stark, bleibt λ gleich.

Liest die Fenster aus den Pickles von eeg_signalraum.py (SNR, TUH) und
erlangen.py (SNR) und lädt nur diese Fenster nach.

    python aenderung_ruhe_anfall.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
from eeg_signalraum import stil_achse, DATEN, SURROGAT, MUTED, INK_SEK
import erlangen as Er
from funktionen import rauschmatrix

BASE = Path("/home/data/ninalaemmermann/forschung")
L = 3.0
N_TOP = 5
GRUPPEN = {"TUH-ABSZ": MUTED, "Erlangen CA": DATEN, "Erlangen ME": SURROGAT}
KENNZAHLEN = {
    "var": "Datenvarianz\nlog₁₀ Spur(C)",
    "R": "Rauschschätzung\nlog₁₀ Spur(R)",
    "snr": "SNR\nMittel log₁₀(λᵢ − 1), Top %d" % N_TOP,
}


def kennzahlen(Y, lam):
    R, _ = rauschmatrix(Y)
    lam = lam[np.isfinite(lam)]
    return {"var": np.log10(np.trace(np.cov(Y, rowvar=False))),
            "R": np.log10(np.trace(R)),
            "snr": np.mean(np.log10(np.clip(lam[:N_TOP] - 1, 1e-3, None)))}


def tuh():
    e = pickle.load(open(HIER / "ergebnisse" / "eigenwerte_eeg_3s.pkl", "rb"))["ergebnisse"]["absz"]
    zeilen = []
    for art in ("ruhe", "anfall"):
        for x in e[art]:
            pfad = BASE / "Data" / "ABSZ_seizure_WB" / f"{x['datei']}_res.OWN11101_filtWB_avg.edf"
            raw = mne.io.read_raw_edf(pfad, verbose=False)
            sr = raw.info["sfreq"]
            i0 = int(round(x["start_s"] * sr))
            Y = raw.get_data(start=i0, stop=i0 + int(round(L * sr))).T * 1e6
            zeilen.append({"gruppe": "TUH-ABSZ", "patient": x["datei"][:8],
                           "anfall": x["datei"] + str(x["anfall"]), "art": art,
                           **kennzahlen(Y, x["lam"])})
    return zeilen


def erlangen():
    e = pickle.load(open(HIER / "ergebnisse" / "erlangen_eigenwerte_3s.pkl", "rb"))["fenster"]
    zeilen = []
    for x in e:
        if x["art"] not in ("ruhe", "anfall"):
            continue
        raw = Er.lies(x["datei"])
        sr = raw.info["sfreq"]
        i0 = int(round(x["start_s"] * sr))
        Y = raw.get_data(picks=Er.eeg_kanaele(raw), start=i0,
                         stop=i0 + int(round(L * sr))).T * 1e6
        zeilen.append({"gruppe": "Erlangen " + x["patient"], "patient": x["patient"],
                       "anfall": x["datei"] + str(x["anfall"]), "art": x["art"],
                       **kennzahlen(Y, x["lam"])})
    return zeilen


def main():
    d = pd.DataFrame(tuh() + erlangen())
    w = d.pivot_table(index=["gruppe", "patient", "anfall"], columns="art",
                      values=list(KENNZAHLEN), aggfunc="first")
    delta = pd.DataFrame({k: w[(k, "anfall")] - w[(k, "ruhe")] for k in KENNZAHLEN}).dropna()

    # TUH: je Patient ein Punkt (Median über seine Anfälle), Erlangen: je Absence ein Punkt
    tuh_pat = delta.loc["TUH-ABSZ"].groupby(level="patient").median()
    punkte = {"TUH-ABSZ": tuh_pat}
    for g in ("Erlangen CA", "Erlangen ME"):
        punkte[g] = delta.loc[g] if g in delta.index.get_level_values(0) else pd.DataFrame()

    print("Median der Änderung Ruhe → Anfall (log10):")
    for g, p in punkte.items():
        print("  %-12s n = %2d  " % (g, len(p)) +
              "  ".join("%s %+.2f" % (k, p[k].median()) for k in KENNZAHLEN))

    fig, achsen = plt.subplots(1, len(KENNZAHLEN), figsize=(14, 4.6), sharey=True)
    fig.suptitle("Änderung Ruhe → Anfall (Fenster %g s, R je Fenster) — "
                 "TUH: je Punkt ein Patient, Erlangen: je Punkt eine Absence" % L)
    rng = np.random.default_rng(0)
    for ax, (k, name) in zip(achsen, KENNZAHLEN.items()):
        stil_achse(ax)
        for x, (g, farbe) in enumerate(GRUPPEN.items()):
            v = punkte[g][k].values if len(punkte[g]) else np.array([])
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, s=22, color=farbe,
                       alpha=0.75, linewidths=0)
            if len(v):
                ax.hlines(np.median(v), x - 0.3, x + 0.3, color=farbe, linewidth=2.5)
        ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(GRUPPEN)))
        ax.set_xticklabels(["%s\n(n=%d)" % (g, len(punkte[g])) for g in GRUPPEN])
        ax.set_title(name, fontsize=10)
    achsen[0].set_ylabel("Δ log₁₀ (Anfall − Ruhe)\n1 = Faktor 10")
    fig.text(0.5, 0.005, "Balken: Median. SNR ≈ Datenvarianz − Rauschschätzung: "
             "wächst R so stark wie C, bleibt das SNR gleich.",
             ha="center", fontsize=8.5, color=INK_SEK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ziel = HIER / "ergebnisse" / "aenderung_ruhe_anfall_3s.png"
    fig.savefig(ziel, dpi=140)
    plt.close(fig)
    delta.to_csv(HIER / "ergebnisse" / "aenderung_ruhe_anfall_3s.csv")
    print("gespeichert:", ziel)


if __name__ == "__main__":
    main()
