"""
Schritt 1 + 2 der SDGL-Rekonstruktion auf echten EEG-Aufnahmen.

    Schritt 1   R aus der zweiten Differenz          (``rauschmatrix``)
    Schritt 2   C·v = λ·R·v,  λ = 1 + SNR je Richtung (``signalraum``)

Je Anfall drei Fenster gleicher Länge: Ruhe, Übergang (Hälfte vor, Hälfte nach
dem Anfallsbeginn) und Anfall (Mitte des Anfalls). Je Anfallstyp ein Panel mit
dem Median der Eigenwertspektren und dem 25–75 %-Band.

    python eeg_signalraum.py --typen absz --fenster-sek 3
    python eeg_signalraum.py --methode pca     # Vergleich: Eigenwerte von C allein
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
from scipy.linalg import eigh, eigvalsh

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
FARBE_TYP = {"absz": DATEN, "cpsz": SURROGAT, "gnsz": RAMPE_STUFEN[2], "fnsz": INK_SEK}
MARKER_TYP = {"absz": "o", "cpsz": "s", "gnsz": "^", "fnsz": "D"}
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

def spektrum(Y, methode="snr"):
    """λ absteigend, mit NaN auf N_KANAL aufgefüllt, und der benutzte Rang r.

    methode "snr": C·v = λ·R·v, λ = 1 + SNR.
    methode "pca": Eigenwerte von C allein (der Fall R = σ²·I), normiert auf den
    Varianzanteil - sonst entscheidet die absolute Amplitude über den Median.
    """
    R, _ = rauschmatrix(Y)
    if not np.isfinite(R).all():
        return None, 0
    ev, U = eigh(R)
    if ev[-1] <= 0:
        return None, 0
    P = U[:, ev > RANG_SCHWELLE * ev[-1]]
    Yp = Y @ P
    if methode == "pca":
        lam = eigvalsh(np.cov(Yp - Yp.mean(axis=0), rowvar=False))[::-1]
        lam = lam / lam.sum()
    else:
        Rp, _ = rauschmatrix(Yp)
        lam = signalraum(Yp, Rp, k=3)[1]
    aus = np.full(N_KANAL, np.nan)
    aus[:len(lam)] = lam
    return aus, P.shape[1]


def werte_typ_aus(typ, base, L, abstand, methode="snr"):
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
                lam, r = spektrum(Y, methode)
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

def richtungen_fuer_anteil(med, anteil=0.9, sockel=1.0):
    """Wie viele führende Richtungen tragen ``anteil`` von Σ(λ − sockel)?

    sockel = 1 für λ = 1 + SNR, sockel = 0 für PCA-Varianzanteile.
    """
    snr = med[np.isfinite(med)] - sockel
    return int(np.argmax(np.cumsum(snr) / snr.sum() >= anteil)) + 1


def je_patient(eintraege):
    """Median-Spektrum je Patient (n_patienten, 27) und die Zahl der Fenster.

    Nur volle Spektren (r = 27), damit Richtung i überall dasselbe bedeutet.
    Jeder Patient zählt gleich - sonst dominieren Patienten mit vielen Anfällen.
    """
    gruppen = {}
    for e in eintraege:
        if e["r"] == N_KANAL:
            gruppen.setdefault(e["datei"][:8], []).append(e["lam"])
    n_fenster = sum(len(v) for v in gruppen.values())
    return np.array([np.median(v, axis=0) for v in gruppen.values()]), n_fenster


def zeichne(ax, eintraege, farbe, name, log, marker="o", methode="snr"):
    """Median und 25–75 % über Patienten; gibt False zurück, wenn nichts da ist."""
    P, n_fenster = je_patient(eintraege)
    if len(P) == 0:
        return False
    x = np.arange(1, N_KANAL + 1)
    med = np.median(P, axis=0)
    lo, hi = np.percentile(P, [25, 75], axis=0)
    if methode == "pca":
        zusatz = ": 90 %% Varianz in %d Richtungen" % richtungen_fuer_anteil(med, sockel=0.0)
    elif log:
        zusatz = " → K = %d" % signalrichtungen(med)
    else:
        zusatz = ": 90 %% SNR in %d Richtungen" % richtungen_fuer_anteil(med)
    ax.fill_between(x, lo, hi, color=farbe, alpha=0.18, linewidth=0)
    (ax.semilogy if log else ax.plot)(
        x, med, marker + "-", color=farbe, markersize=4,
        label="%s (%d Pat., %d Fenster)%s" % (name, len(P), n_fenster, zusatz))
    return True


def rahmen(ax, titel, log, methode="snr"):
    if methode == "snr":
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
        ax.text(N_KANAL * 0.45, 1.0, " λ = 1: reines Rauschen", fontsize=8,
                color=INK_SEK, va="bottom")
    ax.set_title(titel, fontsize=10)
    ax.set_xlabel("Richtung" if methode == "snr" else "Hauptkomponente")
    ax.set_ylabel("λ" if methode == "snr" else "Varianzanteil λ / Σλ")
    if not log:
        ax.set_ylim(bottom=0)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7.5, frameon=False)
    else:
        ax.text(0.5, 0.5, "keine Fenster", transform=ax.transAxes, ha="center")


def untertitel(L, log):
    return ("Fenster %.3g s, Median und 25–75 %% über Patienten%s"
            % (L, "" if log else ", lineare Achse"))


KOPF = {"snr": "Schritt 1+2", "pca": "PCA-Eigenwerte"}
PANEL = {"snr": "λ = 1 + SNR je Richtung", "pca": "PCA, Varianzanteil"}


def abbildung(ergebnisse, L, ziel, log=True, methode="snr"):
    """Ein Panel je Anfallstyp, darin die drei Zustände; gemeinsame y-Achse."""
    typen = list(ergebnisse)
    zeilen = int(np.ceil(len(typen) / 2))
    fig, achsen = plt.subplots(zeilen, 2, figsize=(12, 4.6 * zeilen), squeeze=False,
                               sharey=True)
    fig.suptitle("EEG — %s je Anfallstyp (%s)" % (KOPF[methode], untertitel(L, log)))
    for ax, typ in zip(achsen.flat, typen):
        stil_achse(ax)
        for art in ARTEN:
            zeichne(ax, ergebnisse[typ][art], FARBE[art], BESCHRIFTUNG[art], log,
                    methode=methode)
        rahmen(ax, "%s — %s" % (typ.upper(), PANEL[methode]), log, methode)
    for ax in list(achsen.flat)[len(typen):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(ziel, dpi=140)
    plt.close(fig)


def abbildung_zustaende(ergebnisse, L, ziel, log=True, methode="snr"):
    """Ein Panel je Zustand, darin die Anfallstypen; gemeinsame y-Achse."""
    fig, achsen = plt.subplots(1, len(ARTEN), figsize=(17, 5.2), sharey=True)
    fig.suptitle("EEG — %s je Zustand (%s)" % (KOPF[methode], untertitel(L, log)))
    for ax, art in zip(achsen, ARTEN):
        stil_achse(ax)
        for typ in ergebnisse:
            zeichne(ax, ergebnisse[typ][art], FARBE_TYP[typ], typ.upper(), log,
                    marker=MARKER_TYP[typ], methode=methode)
        rahmen(ax, "%s — %s" % (BESCHRIFTUNG[art], PANEL[methode]), log, methode)
    fig.tight_layout()
    fig.savefig(ziel, dpi=140)
    plt.close(fig)


def alle_abbildungen(ergebnisse, L, stamm, methode="snr"):
    """Vier Bilder: je Typ und je Zustand, jeweils logarithmisch und linear."""
    stamm = str(stamm)
    m = methode
    abbildung(ergebnisse, L, stamm + ".png", methode=m)
    abbildung(ergebnisse, L, stamm + "_linear.png", log=False, methode=m)
    abbildung_zustaende(ergebnisse, L, stamm + "_zustaende.png", methode=m)
    abbildung_zustaende(ergebnisse, L, stamm + "_zustaende_linear.png", log=False, methode=m)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--typen", nargs="+", default=list(TYPEN), choices=TYPEN)
    ap.add_argument("--fenster-sek", type=float, default=3.0)
    ap.add_argument("--ruhe-abstand", type=float, default=30.0,
                    help="Mindestabstand des Ruhefensters zu jedem Anfall (s)")
    ap.add_argument("--methode", choices=("snr", "pca"), default="snr",
                    help="snr: C·v = λ·R·v (Standard); pca: Eigenwerte von C allein")
    ap.add_argument("--base", type=Path, default=Path("/home/data/ninalaemmermann/forschung"))
    args = ap.parse_args()

    ausgabe = HIER / "ergebnisse"
    ausgabe.mkdir(exist_ok=True)
    L = args.fenster_sek

    ergebnisse = {}
    for typ in args.typen:
        print("%s:" % typ.upper())
        ergebnisse[typ] = werte_typ_aus(typ, args.base, L, args.ruhe_abstand, args.methode)

    stamm = "eigenwerte_eeg_%s%gs" % ("pca_" if args.methode == "pca" else "", L)
    with open(ausgabe / (stamm + ".pkl"), "wb") as fh:
        pickle.dump({"fenster_sek": L, "ruhe_abstand": args.ruhe_abstand,
                     "methode": args.methode,
                     "ergebnisse": ergebnisse}, fh)
    alle_abbildungen(ergebnisse, L, ausgabe / stamm, args.methode)
    print("gespeichert:", ausgabe / (stamm + "*.png"))


if __name__ == "__main__":
    main()
