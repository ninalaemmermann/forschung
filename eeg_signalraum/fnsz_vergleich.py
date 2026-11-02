"""
FNSZ: WB (gefiltert, 27 Kan.) gegen resampelt ungefiltert (21 Kan.) - nur auf sauberen Paaren,
mit SNR (λ = 1 + SNR, Schritt 1+2) und mit DyCA im Zeitverlauf.

Sauberes Paar = beide Versionen gleich lang, gleiche Annotation (Anfälle < 2 s Abweichung) und
dieselben Daten: Cz-Korrelation je Minute im Median ≥ 0,9 (ungefiltert auf Avg-Ref über die
19 Kopfkanäle, beide 1–30 Hz Bandpass, ``EEG CZ`` gegen ``EEG Cz_a10``). Genommen werden die
4 sauberen Paare mit der höchsten Korrelation, außer ``--dateien`` ist gesetzt.

Ergebnis: genau 4 Abbildungen mit denselben 4 Aufnahmen untereinander
    fnsz_vergleich_3s_snr_wb.png      fnsz_vergleich_3s_snr_ungefiltert.png
    fnsz_vergleich_3s_dyca_wb.png     fnsz_vergleich_3s_dyca_ungefiltert.png

    python fnsz_vergleich.py
    python fnsz_vergleich.py --dateien aaaaaghb_s010_t000 aaaaahsi_s014_t000 ...
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
from eeg_signalraum import stil_achse, MUTED
from erlangen_zeitreihe import detektion
from dyca_zeitreihe import FARBE_LAM as FARBE_DYCA, N_LAM as N_DYCA
from tuh_resampelt import KANAELE, neue_dateien
from tuh_resampelt_zeitreihe import (FARBE_LAM as FARBE_SNR, N_LAM as N_SNR, anfaelle_zeichnen,
                                     gemeinsame, lade, zeitreihe)

N_PLOT = 4
MIN_KORR = 0.9
VERSIONEN = {"wb": ("alt", "WB gefiltert (27 Kan., Avg-Ref)"),
             "ungefiltert": ("neu", "resampelt ungefiltert (21 Kan.)")}
METHODEN = {"snr": "Schritt 1+2, λ = 1 + SNR", "dyca": "DyCA-Eigenwerte"}


def cz_korrelation(X_neu, X_alt, sr):
    """Median der Cz-Korrelation je Minute (ungefiltert auf Avg-Ref, beide 1–30 Hz)."""
    sos = butter(4, [1, 30], btype="band", fs=sr, output="sos")
    kopf = X_neu[:19] - X_neu[:19].mean(axis=0)            # ohne A1, A2
    a = sosfiltfilt(sos, kopf[KANAELE.index("EEG CZ")])
    b = sosfiltfilt(sos, X_alt[13])                        # EEG Cz_a10
    n, m = min(len(a), len(b)), int(60 * sr)
    return np.median([np.corrcoef(a[i:i + m], b[i:i + m])[0, 1] for i in range(0, n - m + 1, m)])


def gleiche_anfaelle(a, b):
    return len(a) == len(b) and all(abs(x[0] - y[0]) < 2 and abs(x[1] - y[1]) < 2 for x, y in zip(a, b))


def paarpruefung(pfade):
    """Je Aufnahme: Längen, Annotation gleich, Cz-Korrelation, sauber ja/nein."""
    aus = {}
    print("Paarprüfung WB – ungefiltert:")
    print("%-20s %14s %10s %8s %7s" % ("Aufnahme", "Länge WB/ung.", "Annot.", "Cz r", "sauber"))
    for name in gemeinsame(pfade, "fnsz"):
        Xa, sr, an_a = lade("alt", name, pfade[name], "fnsz")
        Xn, _, an_n = lade("neu", name, pfade[name], "fnsz")
        la, ln = Xa.shape[1] / sr, Xn.shape[1] / sr
        gleich = gleiche_anfaelle(an_a, an_n)
        r = cz_korrelation(Xn, Xa, sr)
        sauber = abs(la - ln) < 1 and gleich and r >= MIN_KORR
        aus[name] = {"laenge": (la, ln), "annotation_gleich": gleich, "cz_r": r, "sauber": sauber}
        print("%-20s %6.0f/%5.0f s %10s %8.2f %7s" % (name, la, ln, "gleich" if gleich else "anders",
                                                      r, "ja" if sauber else "nein"))
    return aus


def abbildung(methode, version, ergebnis, L, schritt, ziel, ylim=None):
    namen = list(ergebnis)
    fig, achsen = plt.subplots(len(namen), 1, figsize=(18, 3.2 * len(namen)), squeeze=False)
    n_lam, farben = (N_SNR, FARBE_SNR) if methode == "snr" else (N_DYCA, FARBE_DYCA)
    fig.suptitle("FNSZ, %s — %s über die Zeit (Fenster %.3g s, Schritt %.3g s); λ1–λ%d, orange: Anfälle"
                 % (VERSIONEN[version][1], METHODEN[methode], L, schritt, n_lam), fontsize=11)
    for k, (ax, name) in enumerate(zip(achsen[:, 0], namen)):
        e = ergebnis[name][methode][version]
        stil_achse(ax)
        anfaelle_zeichnen(ax, e["anfaelle"], beschriften=(k == 0))
        zeichne = ax.semilogy if methode == "snr" else ax.plot
        for i in range(n_lam):
            zeichne(e["t"], e["lam"][:, i], color=farben[i], linewidth=0.9,
                    label="λ%d" % (i + 1) if k == 0 else None)
        if methode == "snr":
            ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
            ax.set_ylim(*ylim)
            ax.set_ylabel("λ = 1 + SNR")
        else:
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("DyCA-Eigenwert λ")
        ax.set_title("%s — %d Kanäle, %d Anfälle — AUC λ1 = %.2f"
                     % (name, e["n_kanal"], len(e["anfaelle"]), e["auc"]), fontsize=10)
        ax.set_xlim(0, e["t"][-1] + L / 2)
    achsen[0, 0].legend(fontsize=8, frameon=False, ncol=n_lam + 1, loc="upper right")
    achsen[-1, 0].set_xlabel("Zeit (Fenstermitte) [s]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ziel, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dateien", nargs="+", default=None, help="statt der automatischen Auswahl")
    ap.add_argument("--fenster-sek", type=float, default=3.0)
    ap.add_argument("--schritt-sek", type=float, default=0.5)
    args = ap.parse_args()
    L, schritt = args.fenster_sek, args.schritt_sek
    ausgabe = HIER / "ergebnisse"
    ausgabe.mkdir(exist_ok=True)

    pfade = neue_dateien("fnsz")
    pruefung = paarpruefung(pfade)
    if args.dateien:
        dateien = args.dateien
    else:
        sauber = sorted((n for n, p in pruefung.items() if p["sauber"]), key=lambda n: -pruefung[n]["cz_r"])
        dateien = sorted(sauber[:N_PLOT])
    print("\nim Vergleich:", dateien)

    ergebnis = {}
    for name in dateien:
        ergebnis[name] = {m: {} for m in METHODEN}
        for version, (satz, _) in VERSIONEN.items():
            X, sr, anfaelle = lade(satz, name, pfade[name], "fnsz")
            for methode in METHODEN:
                t, lam = zeitreihe(X, sr, L, schritt, X.shape[0], methode=methode)
                e = {"t": t, "lam": lam, "anfaelle": anfaelle, "n_kanal": X.shape[0]}
                e["auc"] = detektion(e, L, schritt)["auc"]
                ergebnis[name][methode][version] = e
        print("fertig:", name, flush=True)

    print("\nAUC λ1 (Anfall gegen anfallsfrei):")
    print("%-20s %10s %13s %10s %13s" % ("Aufnahme", "SNR WB", "SNR ungef.", "DyCA WB", "DyCA ungef."))
    for name in dateien:
        z = ergebnis[name]
        print("%-20s %10.2f %13.2f %10.2f %13.2f" % (name, z["snr"]["wb"]["auc"], z["snr"]["ungefiltert"]["auc"],
                                                     z["dyca"]["wb"]["auc"], z["dyca"]["ungefiltert"]["auc"]))

    # SNR: gemeinsame y-Achse über beide Versionen
    alle_snr = np.concatenate([ergebnis[n]["snr"][v]["lam"][:, 0] for n in dateien for v in VERSIONEN])
    ylim = (0.8, np.nanmax(alle_snr) * 1.5)
    for methode in METHODEN:
        for version in VERSIONEN:
            ziel = ausgabe / ("fnsz_vergleich_%gs_%s_%s.png" % (L, methode, version))
            abbildung(methode, version, ergebnis, L, schritt, ziel, ylim)
            print("gespeichert:", ziel)
    with open(ausgabe / ("fnsz_vergleich_%gs.pkl" % L), "wb") as fh:
        pickle.dump({"fenster_sek": L, "schritt_sek": schritt, "paarpruefung": pruefung,
                     "dateien": dateien, "ergebnis": ergebnis}, fh)


if __name__ == "__main__":
    main()
