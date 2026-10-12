"""
TUH mit der Ruhe als Rauschen (csp.py) - Ruhe, Übergang, Anfall je Anfallstyp.

R PRO DATEI: jedes Fenster wird gegen die Ruhe der Datei getestet, in der
es liegt. Schwelle je TUH-Patient (erste 8 Zeichen des Dateinamens) und
Anfallstyp:

    Ruhepool    3-s-Fenster ≥ 30 s von jedem Anfall der Datei, höchstens
                MAX_RUHE gleichmäßig verteilt, Artefaktfenster (Varianz >
                Median + 5·MAD, relativ zur eigenen Datei) verworfen
    Schwelle    Nullprobe je Datei (jedes zweite Ruhefenster in R, die übrigen
                getestet), 95-%-Quantil von λ1 über alle Dateien des Patienten
                - wie „gleiche Aufnahme“ in nullprobe_erlangen.py
    Test        Fenster Ruhe/Übergang/Anfall wie in eeg_signalraum.py,
                C·v = λ·R·v, K = Zahl der λ über der Schwelle

Ein getestetes Fenster fließt nie in sein eigenes R ein. Rohdaten wie in
eeg_signalraum.py (die EDFs heißen filtWB, sind also von TUH vorverarbeitet).
Dateien mit weniger als MIN_RUHE bereinigten Ruhefenstern fallen heraus,
samt ihren Anfällen (Anzahl wird ausgegeben).

Abbildung: je Anfallstyp λ von Ruhe/Übergang/Anfall (Median je Patient, dann
über Patienten) und als graue Linie die Grenze Ruhe gegen Ruhe (97,5-%-
Quantil der Nullprobe je Richtung). K und Schwelle stehen nur in Tabelle/.pkl.

    python tuh_ruhe.py                 # alle Typen
    python tuh_ruhe.py --typen absz
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import nullprobe_erlangen as nb
from eeg_signalraum import (ARTEN, BESCHRIFTUNG, FARBE, TYPEN, N_KANAL, frei,
                            fenster_je_anfall, stil_achse, MUTED, get_sz_start_end)
from stil import INK_SEK
from csp import ruhe_kovarianz, kontrast
from erlangen_ruhe import ueberlappt

BASE = Path("/home/data/ninalaemmermann/forschung")
L = 3.0
ABSTAND = 30.0
MAX_RUHE = 300
MIN_RUHE = 40


def dateien(typ):
    st = typ.upper()
    liste = get_sz_start_end(BASE / f"{st}_seizures.csv")
    for datei, sz in liste.items():
        pfad = BASE / "Data" / f"{st}_seizure_WB" / f"{datei}_res.OWN11101_filtWB_avg.edf"
        anf = [(float(a), float(b)) for a, b in sz if np.isfinite(a) and np.isfinite(b) and b > a]
        if pfad.exists() and anf:
            yield datei, pfad, anf


def lese(raw, t0, n):
    i0 = int(round(t0 * raw.info["sfreq"]))
    if i0 + n > raw.n_times:
        return None
    return raw.get_data(start=i0, stop=i0 + n).T * 1e6


def pad(lam):
    aus = np.full(N_KANAL, np.nan)
    aus[:min(len(lam), N_KANAL)] = lam[:N_KANAL]
    return aus


def auswerten(typ):
    """R pro Datei, Schwelle pro Patient aus den Nullproben seiner Dateien."""
    null, fenster, weg, n_ruhe = {}, {}, {}, {}
    zaehler = {"dateien": 0, "nutzbar": 0}
    for datei, pfad, anf in dateien(typ):
        zaehler["dateien"] += 1
        raw = mne.io.read_raw_edf(pfad, preload=False, verbose=False)
        sr = raw.info["sfreq"]
        n = int(round(L * sr))
        dauer = raw.n_times / sr
        kand = [t for t in np.arange(0, dauer - L, L) if frei(t, t + L, anf, ABSTAND)]
        if len(kand) < MIN_RUHE:
            continue
        wahl = [kand[i] for i in np.unique(np.linspace(0, len(kand) - 1,
                                                       min(MAX_RUHE, len(kand))).astype(int))]
        pool = [(datei, t, Y) for t in wahl for Y in [lese(raw, t, n)] if Y is not None]
        pool, anteil = nb.bereinige(pool)          # Artefakte relativ zur eigenen Datei
        if len(pool) < MIN_RUHE:
            continue
        zaehler["nutzbar"] += 1
        pat = datei[:8]
        weg.setdefault(pat, []).append(anteil)
        n_ruhe[pat] = n_ruhe.get(pat, 0) + len(pool)

        # Nullprobe in dieser Datei: jedes zweite Fenster in R, die übrigen getestet
        R_null = ruhe_kovarianz([f[2] for f in pool[0::2]])
        null.setdefault(pat, []).extend(pad(kontrast(f[2], R_null)[1]) for f in pool[1::2])

        R_alle = ruhe_kovarianz([f[2] for f in pool])
        gesehen = set()
        for a, b in anf:
            for art, t0 in fenster_je_anfall(a, b, anf, L, ABSTAND, dauer).items():
                if (art, round(t0, 2)) in gesehen:
                    continue
                gesehen.add((art, round(t0, 2)))
                Y = lese(raw, t0, n)
                if Y is None:
                    continue
                rest = [f for f in pool if not ueberlappt(t0, t0 + L, f[1], f[1] + L)]
                R = R_alle if len(rest) == len(pool) else ruhe_kovarianz([f[2] for f in rest])
                fenster.setdefault(pat, []).append(
                    {"patient": pat, "datei": datei, "art": art, "anfall": (a, b),
                     "start_s": t0, "lam": pad(kontrast(Y, R)[1])})

    print("  %d / %d Dateien mit ≥ %d bereinigten Ruhefenstern"
          % (zaehler["nutzbar"], zaehler["dateien"], MIN_RUHE))
    ergebnis = {}
    for pat in sorted(set(null) & set(fenster)):
        lam_null = np.array(null[pat])
        schwelle = np.percentile(lam_null[:, 0], 95)
        for e in fenster[pat]:
            e["K"] = int(np.nansum(e["lam"] > schwelle))
        ergebnis[pat] = {"schwelle": schwelle, "verworfen": float(np.mean(weg[pat])),
                         "n_ruhe": n_ruhe[pat], "lam_null": lam_null, "fenster": fenster[pat]}
    return ergebnis


def tabelle(typ, erg):
    print("\n%s: %d Patienten, Schwelle λ1 Median %.1f (%.1f … %.1f)"
          % (typ.upper(), len(erg), np.median([v["schwelle"] for v in erg.values()]),
             min(v["schwelle"] for v in erg.values()), max(v["schwelle"] for v in erg.values())))
    print("  %-9s %5s %8s %22s %18s" % ("Zustand", "Pat.", "Fenster", "K je Pat. (Median)",
                                         "Anteil Fenster K≥1"))
    for art in ARTEN:
        ks = [np.median([f["K"] for f in v["fenster"] if f["art"] == art])
              for v in erg.values() if any(f["art"] == art for f in v["fenster"])]
        alle = [f["K"] for v in erg.values() for f in v["fenster"] if f["art"] == art]
        if ks:
            print("  %-9s %5d %8d %10.1f (IQR %.1f–%.1f) %17.0f %%"
                  % (art, len(ks), len(alle), np.median(ks), *np.percentile(ks, [25, 75]),
                     100 * np.mean(np.array(alle) >= 1)))


def abbildung(ergebnisse, ziel):
    """Je Anfallstyp ein Panel: λ von Ruhe/Übergang/Anfall und die Grenze Ruhe gegen Ruhe.

    λ ist schon das Verhältnis zur eigenen Ruhe, also zwischen Patienten
    vergleichbar. Je Patient erst Median über seine Fenster, dann Median über
    Patienten. Graue Linie: je Patient das 97,5-%-Quantil der Nullprobe je
    Richtung, dann Median über Patienten.
    """
    typen = list(ergebnisse)
    fig, achsen = plt.subplots(1, len(typen), figsize=(4.6 * len(typen), 4.8), squeeze=False,
                               sharey=True)
    fig.suptitle("TUH — Ruhe als Rauschen, Median über Patienten", fontsize=11)
    x = np.arange(1, N_KANAL + 1)
    for ax, typ in zip(achsen[0], typen):
        erg = ergebnisse[typ]
        stil_achse(ax)
        grenze = np.nanmedian([np.nanpercentile(v["lam_null"], 97.5, axis=0)
                               for v in erg.values()], axis=0)
        ax.semilogy(x, grenze, "-", color=MUTED, linewidth=2.0,
                    label="Ruhe gegen Ruhe (97,5 %): Grenze")
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.0)
        for art in ARTEN:
            P = [np.nanmedian([f["lam"] for f in v["fenster"] if f["art"] == art], axis=0)
                 for v in erg.values() if any(f["art"] == art for f in v["fenster"])]
            if P:
                ax.semilogy(x, np.nanmedian(P, axis=0), "o-", color=FARBE[art], markersize=3,
                            label=BESCHRIFTUNG[art])
        ax.set_title("%s (%d Patienten)" % (typ.upper(), len(erg)), fontsize=10)
        ax.set_xlabel("Richtung")
    achsen[0][0].set_ylabel("λ = Varianz Fenster / Varianz Ruhe")
    achsen[0][0].legend(fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ziel, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--typen", nargs="+", default=list(TYPEN), choices=TYPEN)
    args = ap.parse_args()
    ergebnisse = {}
    for typ in args.typen:
        print("%s wird ausgewertet …" % typ.upper(), flush=True)
        ergebnisse[typ] = auswerten(typ)
        tabelle(typ, ergebnisse[typ])
    stamm = HIER / "ergebnisse" / ("tuh_ruhe_%gs" % L
                                   + ("" if set(args.typen) == set(TYPEN) else "_" + "_".join(args.typen)))
    with open(str(stamm) + ".pkl", "wb") as fh:
        pickle.dump({"fenster_sek": L, "abstand": ABSTAND, "max_ruhe": MAX_RUHE,
                     "min_ruhe": MIN_RUHE, "ergebnisse": ergebnisse}, fh)
    abbildung(ergebnisse, str(stamm) + ".png")
    print("\ngespeichert:", str(stamm) + ".png / .pkl")


if __name__ == "__main__":
    main()
