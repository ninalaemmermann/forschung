"""
Schritt 1 + 2 der SDGL-Rekonstruktion auf echten EEG-Aufnahmen.

    Schritt 1   R aus der zweiten Differenz          (``rauschmatrix``)
    Schritt 2   C·v = λ·R·v,  λ = 1 + SNR je Richtung (``signalraum``)

Je Anfall drei Fenster gleicher Länge: Ruhe, Übergang (Hälfte vor, Hälfte nach
dem Anfallsbeginn) und Anfall (Mitte des Anfalls). Je Anfallstyp ein Panel mit
dem Median der Eigenwertspektren und dem 25–75 %-Band.

    python eeg_signalraum.py --typen absz --fenster-sek 3
"""
import argparse
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.linalg import eigh

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent / "stdgl" / "Signalrekonstruktion"))
sys.path.insert(0, str(HIER.parent / "stdgl" / "werkzeuge"))
sys.path.insert(0, str(HIER.parent))
from funktionen import rauschmatrix, signalraum, signalrichtungen
from stil import stil_achse, DATEN, SURROGAT, MUTED, INK_SEK, RAMPE_STUFEN
from Verteilungsfkt import get_sz_start_end

TYPEN = ("absz", "cpsz", "gnsz", "fnsz")
ARTEN = ("ruhe", "uebergang", "anfall")
BESCHRIFTUNG = {"ruhe": "Ruhe", "uebergang": "Übergang", "anfall": "Anfall"}
FARBE = {"ruhe": DATEN, "uebergang": RAMPE_STUFEN[2], "anfall": SURROGAT}
N_KANAL = 27
# Richtungen von R unterhalb dieses Anteils am größten Eigenwert sind tot
# (Average-Referenz); eigh(C, R) verlangt ein positiv definites R.
RANG_SCHWELLE = 1e-8


# ===========================================================================
# Fensterwahl
# ===========================================================================

def frei(t0, t1, anfaelle, abstand):
    """Liegt [t0, t1] mindestens ``abstand`` Sekunden von jedem Anfall weg?"""
    return all(t1 <= a - abstand or t0 >= b + abstand for a, b in anfaelle)


def ruhefenster(a, b, anfaelle, L, abstand, dauer):
    """Nächstes freies Fenster vor dem Anfall, sonst danach, sonst None."""
    t = a - abstand - L
    while t >= 0:
        if frei(t, t + L, anfaelle, abstand):
            return t
        t -= L
    t = b + abstand
    while t + L <= dauer:
        if frei(t, t + L, anfaelle, abstand):
            return t
        t += L
    return None


def fenster_je_anfall(a, b, anfaelle, L, abstand, dauer):
    """Startzeiten (s) der drei Fensterarten für einen Anfall; None = entfällt."""
    starts = {"anfall": None, "uebergang": None, "ruhe": None}
    if b - a >= L:
        starts["anfall"] = 0.5 * (a + b) - 0.5 * L
    t = a - 0.5 * L
    andere = [(s, e) for s, e in anfaelle if (s, e) != (a, b)]
    if t >= 0 and b - a >= 0.5 * L and frei(t, a, andere, 0.0):
        starts["uebergang"] = t
    starts["ruhe"] = ruhefenster(a, b, anfaelle, L, abstand, dauer)
    return {k: v for k, v in starts.items() if v is not None and v + L <= dauer}


# ===========================================================================
# Schritt 1 + 2 auf einem Fenster
# ===========================================================================

def spektrum(Y):
    """λ absteigend, mit NaN auf N_KANAL aufgefüllt, und der benutzte Rang r."""
    R, _ = rauschmatrix(Y)
    if not np.isfinite(R).all():
        return None, 0
    ev, U = eigh(R)
    if ev[-1] <= 0:
        return None, 0
    P = U[:, ev > RANG_SCHWELLE * ev[-1]]
    Yp = Y @ P
    Rp, _ = rauschmatrix(Yp)
    lam = signalraum(Yp, Rp, k=3)[1]
    aus = np.full(N_KANAL, np.nan)
    aus[:len(lam)] = lam
    return aus, P.shape[1]


def werte_typ_aus(typ, base, L, abstand):
    st = typ.upper()
    anfallsliste = get_sz_start_end(base / f"{st}_seizures.csv")
    ordner = base / "Data" / f"{st}_seizure_WB"
    ergebnis = {art: [] for art in ARTEN}

    for datei, sz in anfallsliste.items():
        pfad = ordner / f"{datei}_res.OWN11101_filtWB_avg.edf"
        if not pfad.exists():
            continue
        anfaelle = [(float(a), float(b)) for a, b in sz
                    if np.isfinite(a) and np.isfinite(b) and b > a]
        if not anfaelle:
            continue
        raw = mne.io.read_raw_edf(pfad, preload=False, verbose=False)
        sr = raw.info["sfreq"]
        n_win = int(round(L * sr))
        dauer = raw.n_times / sr
        gesehen = set()

        for a, b in anfaelle:
            for art, t0 in fenster_je_anfall(a, b, anfaelle, L, abstand, dauer).items():
                i0 = int(round(t0 * sr))
                if (art, i0) in gesehen or i0 + n_win > raw.n_times:
                    continue
                gesehen.add((art, i0))
                Y = raw.get_data(start=i0, stop=i0 + n_win).T * 1e6   # µV
                lam, r = spektrum(Y)
                if lam is None:
                    continue
                ergebnis[art].append({"datei": datei, "anfall": (a, b),
                                      "start_s": i0 / sr, "r": r, "lam": lam})

    for art in ARTEN:
        rs = [e["r"] for e in ergebnis[art]]
        print("  %-9s n = %4d   r: %s" % (art, len(rs),
              "-" if not rs else "%d … %d (Median %d)" % (min(rs), max(rs), np.median(rs))))
    return ergebnis


# ===========================================================================
# Abbildung
# ===========================================================================

def abbildung(ergebnisse, L, ziel):
    typen = list(ergebnisse)
    zeilen = int(np.ceil(len(typen) / 2))
    fig, achsen = plt.subplots(zeilen, 2, figsize=(12, 4.6 * zeilen), squeeze=False)
    fig.suptitle("EEG — Schritt 1+2 je Anfallstyp (Fenster %.3g s, Median und 25–75 %%)" % L)

    for ax, typ in zip(achsen.flat, typen):
        stil_achse(ax)
        x = np.arange(1, N_KANAL + 1)
        for art in ARTEN:
            eintraege = ergebnisse[typ][art]
            if not eintraege:
                continue
            M = np.array([e["lam"] for e in eintraege])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # Spalten nur aus NaN
                med = np.nanmedian(M, axis=0)
                lo, hi = np.nanpercentile(M, [25, 75], axis=0)
            K = signalrichtungen(med[np.isfinite(med)])
            ax.fill_between(x, lo, hi, color=FARBE[art], alpha=0.2, linewidth=0)
            ax.semilogy(x, med, "o-", color=FARBE[art], markersize=4,
                        label="%s (n = %d) → K = %d" % (BESCHRIFTUNG[art], len(eintraege), K))
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
        ax.text(N_KANAL * 0.45, 1.0, " λ = 1: reines Rauschen", fontsize=8,
                color=INK_SEK, va="bottom")
        ax.set_title("%s — Schritt 1+2: λ = 1 + SNR je Richtung" % typ.upper(), fontsize=10)
        ax.set_xlabel("Richtung")
        ax.set_ylabel("λ")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, frameon=False)
        else:
            ax.text(0.5, 0.5, "keine Fenster", transform=ax.transAxes, ha="center")
    for ax in list(achsen.flat)[len(typen):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(ziel, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--typen", nargs="+", default=list(TYPEN), choices=TYPEN)
    ap.add_argument("--fenster-sek", type=float, default=3.0)
    ap.add_argument("--ruhe-abstand", type=float, default=30.0,
                    help="Mindestabstand des Ruhefensters zu jedem Anfall (s)")
    ap.add_argument("--base", type=Path, default=Path("/home/data/ninalaemmermann/forschung"))
    args = ap.parse_args()

    ausgabe = HIER / "ergebnisse"
    ausgabe.mkdir(exist_ok=True)
    L = args.fenster_sek

    ergebnisse = {}
    for typ in args.typen:
        print("%s:" % typ.upper())
        ergebnisse[typ] = werte_typ_aus(typ, args.base, L, args.ruhe_abstand)

    stamm = "eigenwerte_eeg_%gs" % L
    with open(ausgabe / (stamm + ".pkl"), "wb") as fh:
        pickle.dump({"fenster_sek": L, "ruhe_abstand": args.ruhe_abstand,
                     "ergebnisse": ergebnisse}, fh)
    abbildung(ergebnisse, L, ausgabe / (stamm + ".png"))
    print("gespeichert:", ausgabe / (stamm + ".png"))


if __name__ == "__main__":
    main()
