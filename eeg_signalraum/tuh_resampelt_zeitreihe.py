"""
Zeitverlauf von λ = 1 + SNR über die ganze Aufnahme - TUH-ABSZ alt (ABSZ_seizure_WB,
27 Kanäle) gegen resampelt ungefiltert (21 Kanäle), beide so, wie sie gespeichert sind.

λ sind die generalisierten Eigenwerte von C·v = λ·R·v (Schritt 1 + 2, ``spektrum`` aus
``eeg_signalraum.py``), je Fenster einer pro Richtung, absteigend. Gezeichnet werden
λ1–λ3; gespeichert werden alle.

Gleitendes Fenster (Standard 3 s, Schritt 0,5 s); die Zeitachse ist die Fenstermitte.
Anfälle: alt aus ABSZ_seizures.csv, neu aus der .csv_bi neben der EDF.

    python tuh_resampelt_zeitreihe.py
    python tuh_resampelt_zeitreihe.py --dateien aaaaaccj_s002_t001 aaaaadqe_s001_t001
    python tuh_resampelt_zeitreihe.py --alle --beste 3   # alle gemeinsamen, nur die besten speichern

Bewertung je Datei = Übereinstimmung von Anfall und λ1 über die ganze Aufnahme
(je Datensatz mit seinen eigenen Anfallszeiten):
    r        Korrelation zwischen log10 λ1 und dem Anfallsanteil des Fensters (0 … 1), alle
             Fenster - hoch, wenn λ1 im Anfall hoch und außerhalb niedrig ist (Rangfolge)
    Treffer  Anteil der Anfälle, deren Median-λ1 über dem 90-%-Quantil der anfallsfreien
             Fenster liegt
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
from scipy.ndimage import median_filter, minimum_filter1d

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
from eeg_signalraum import spektrum, stil_achse, get_sz_start_end, MUTED, SURROGAT, DATEN
from stil import RAMPE_STUFEN
from tuh_resampelt import ALIAS, BASE, EINZELN, KANAELE, anfaelle_bi, neue_dateien

N_LAM = 3                                   # gezeichnete Eigenwerte
FARBE_LAM = RAMPE_STUFEN[::-1]              # λ1 dunkel … λ3 hell
SAETZE = {"alt": ("TUH-%s WB (gefiltert, 27 Kan., Avg-Ref)", 27),
          "neu": ("resampelt ungefiltert (21 Kan., wie gespeichert)", len(KANAELE))}


def lade(satz, name, edf_neu, typ="absz"):
    """Daten (n_kanal, n) in µV, Abtastrate, Anfälle [(a, b)]."""
    if satz == "alt":
        st = typ.upper()
        alt_name = ALIAS.get(name, name) if typ == "absz" else name
        pfad = BASE / "Data" / f"{st}_seizure_WB" / f"{alt_name}_res.OWN11101_filtWB_avg.edf"
        raw = mne.io.read_raw_edf(pfad, preload=True, verbose=False)
        X = raw.get_data() * 1e6
        sz = get_sz_start_end(BASE / f"{st}_seizures.csv").get(alt_name, [])
        anfaelle = sorted((float(a), float(b)) for a, b in sz)
    else:
        raw = mne.io.read_raw_edf(edf_neu, preload=True, verbose=False)
        X = raw.get_data(picks=KANAELE) * 1e6
        anfaelle = anfaelle_bi(edf_neu)
    return X, raw.info["sfreq"], anfaelle


def zeitreihe(X, sr, L, schritt, r_voll, methode="snr"):
    """Fenstermitten (s) und λ (n_fenster, r_voll); NaN, wo es keine Lösung gibt."""
    n_win, n_schritt = int(round(L * sr)), int(round(schritt * sr))
    starts = np.arange(0, X.shape[1] - n_win + 1, n_schritt)
    lam = np.full((len(starts), r_voll), np.nan)
    for j, i0 in enumerate(starts):
        l, r = spektrum(X[:, i0:i0 + n_win].T, methode)
        if l is None or r != r_voll:
            continue
        lam[j] = l[:r_voll]
    return (starts + n_win / 2) / sr, lam


def plateau(lam, schritt, spalten=(0, 1, 2), w=5.0, basis=60.0, log=True):
    """Anhaltende Erhöhung von λ über der eigenen Grundlinie.

    x = Mittel von log10 λ über ``spalten`` (DyCA: ``log=False``), NaN -> Minimum.
    g = gleitender Median über ``basis`` s (0 = ohne Grundlinie), d = x - g.
    Plateau = gleitendes Minimum von d über ``w`` s: hoch nur, wenn d die ganzen w s hoch war.
    Beide Filter sind zentriert, nutzen also auch spätere Fenster.
    Rückgabe: Plateau, x, g (je Fenster).
    """
    def spalte(i):
        v = np.log10(lam[:, i]) if log else lam[:, i].astype(float)
        return np.where(np.isfinite(v), v, np.nanmin(v))
    x = np.mean([spalte(i) for i in spalten], axis=0)
    g = median_filter(x, int(basis / schritt), mode="nearest") if basis else np.zeros_like(x)
    n = max(1, int(w / schritt))
    return minimum_filter1d(x - g, n, mode="nearest"), x, g


def anfallsanteil(t, L, intervalle):
    """Anteil jedes Fensters (Mitte t, Länge L), der in den Intervallen liegt (0 … 1)."""
    anteil = np.zeros(len(t))
    for a, b in intervalle:
        anteil += np.clip(np.minimum(t + L / 2, b) - np.maximum(t - L / 2, a), 0, None) / L
    return np.clip(anteil, 0, 1)


def bewertung(e, L, ausschluss=()):
    """Übereinstimmung von Anfall und λ1: Korrelation r und Trefferquote je Anfall.

    Fenster, die ``ausschluss`` berühren (Ereignisse ohne Endzeit), zählen nirgends mit."""
    t, l = e["t"], np.log10(e["lam"][:, 0])
    ok = np.isfinite(l) & (anfallsanteil(t, L, ausschluss) == 0)
    anteil = anfallsanteil(t, L, e["anfaelle"])
    frei = ok & (anteil == 0)
    if not e["anfaelle"] or frei.sum() < 2 or (anteil[ok] > 0).sum() < 2:
        return {"r": np.nan, "treffer": np.nan, "n_anfaelle": len(e["anfaelle"])}
    r = np.corrcoef(l[ok], anteil[ok])[0, 1]
    schwelle = np.quantile(l[frei], 0.9)
    medi = [np.median(l[ok & (t >= a) & (t <= b)]) for a, b in e["anfaelle"]
            if (ok & (t >= a) & (t <= b)).any()]
    return {"r": r, "treffer": np.mean(np.array(medi) > schwelle) if medi else np.nan,
            "n_anfaelle": len(e["anfaelle"])}


def gemeinsame(neu_pfade, typ="absz"):
    ordner = BASE / "Data" / ("%s_seizure_WB" % typ.upper())
    alt = {p.name.split("_res")[0] for p in ordner.glob("*.edf")}
    alias = ALIAS if typ == "absz" else {}
    return sorted(n for n in neu_pfade if alias.get(n, n) in alt)


def anfaelle_zeichnen(ax, anfaelle, beschriften=False):
    for k, (a, b) in enumerate(anfaelle):
        ax.axvspan(a, b, color=SURROGAT, alpha=0.18, linewidth=0,
                   label="Anfall" if beschriften and k == 0 else None)


def abbildung_datei(name, erg, L, schritt, ziel, typ="absz"):
    fig, achsen = plt.subplots(2, 1, figsize=(18, 7.5), sharex=True)
    fig.suptitle("%s — Schritt 1+2, λ = 1 + SNR über die Zeit (Fenster %.3g s, Schritt %.3g s); "
                 "λ1–λ3, orange: Anfälle" % (name, L, schritt), fontsize=11)
    oben = max(np.nanmax(erg[s]["lam"][:, 0]) for s in SAETZE)
    for ax, satz in zip(achsen, SAETZE):
        e = erg[satz]
        stil_achse(ax)
        anfaelle_zeichnen(ax, e["anfaelle"], beschriften=True)
        for i in range(N_LAM):
            ax.semilogy(e["t"], e["lam"][:, i], color=FARBE_LAM[i], linewidth=0.9, label="λ%d" % (i + 1))
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
        ax.set_ylim(0.8, oben * 1.5)
        ax.set_ylabel("λ = 1 + SNR")
        ax.set_title("%s — %d Anfälle" % (SAETZE[satz][0].replace("%s", typ.upper()), len(e["anfaelle"])),
                     fontsize=10)
        ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper right")
    ax = achsen[-1]
    ax.set_xlabel("Zeit (Fenstermitte) [s]")
    ax.set_xlim(0, max(erg[s]["t"][-1] for s in SAETZE) + L / 2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ziel, dpi=130)
    plt.close(fig)


def abbildung_uebersicht(alle, L, schritt, ziel):
    fig, achsen = plt.subplots(len(alle), 1, figsize=(18, 3.2 * len(alle)), squeeze=False)
    fig.suptitle("λ1 über die Zeit, alt (grau) gegen neu (blau) — Fenster %.3g s, Schritt %.3g s; "
                 "orange: Anfälle (neu)" % (L, schritt), fontsize=11)
    for ax, (name, erg) in zip(achsen[:, 0], alle.items()):
        stil_achse(ax)
        anfaelle_zeichnen(ax, erg["neu"]["anfaelle"])
        ax.semilogy(erg["alt"]["t"], erg["alt"]["lam"][:, 0], color=MUTED, linewidth=0.8, label="alt")
        ax.semilogy(erg["neu"]["t"], erg["neu"]["lam"][:, 0], color=DATEN, linewidth=0.8, label="neu")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("λ1")
        ax.set_xlim(0, erg["neu"]["t"][-1] + L / 2)
        ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    achsen[-1, 0].set_xlabel("Zeit (Fenstermitte) [s]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ziel, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dateien", nargs="+", default=list(EINZELN),
                    help="neue Dateinamen (BName); alt wird über ALIAS zugeordnet")
    ap.add_argument("--typ", choices=("absz", "fnsz"), default="absz",
                    help="Anfallstyp: Ordner fnsz in resampelt_ungefiltert und FNSZ_seizure_WB")
    ap.add_argument("--fenster-sek", type=float, default=3.0)
    ap.add_argument("--schritt-sek", type=float, default=0.5)
    ap.add_argument("--alle", action="store_true",
                    help="alle Aufnahmen, die alt und neu vorliegen (ersetzt --dateien)")
    ap.add_argument("--beste", type=int, default=None,
                    help="nur die N Dateien mit der höchsten AUC (neu) speichern")
    args = ap.parse_args()
    L, schritt = args.fenster_sek, args.schritt_sek
    ausgabe = HIER / "ergebnisse"
    ausgabe.mkdir(exist_ok=True)
    neu_pfade = neue_dateien(args.typ)
    praefix = "tuh_resampelt_zeitreihe" if args.typ == "absz" else "tuh_%s_zeitreihe" % args.typ

    dateien = gemeinsame(neu_pfade, args.typ) if args.alle else args.dateien
    alle = {}
    for name in dateien:
        alle[name] = {}
        for satz, (_, r_voll) in SAETZE.items():
            X, sr, anfaelle = lade(satz, name, neu_pfade[name], args.typ)
            t, lam = zeitreihe(X, sr, L, schritt, r_voll)
            alle[name][satz] = {"t": t, "lam": lam, "anfaelle": anfaelle}
            alle[name][satz]["bewertung"] = bewertung(alle[name][satz], L)

    print("\nÜbereinstimmung Anfall – λ1 (sortiert nach r neu):")
    print("%-20s %8s | %6s %6s | %11s %11s"
          % ("Datei", "Anfälle", "r alt", "r neu", "Treffer alt", "Treffer neu"))
    rang = sorted(dateien, key=lambda n: -np.nan_to_num(alle[n]["neu"]["bewertung"]["r"], nan=-9))
    for n in rang:
        a, b = alle[n]["alt"]["bewertung"], alle[n]["neu"]["bewertung"]
        print("%-20s %8d | %6.2f %6.2f | %10.0f%% %10.0f%%"
              % (n, b["n_anfaelle"], a["r"], b["r"], 100 * a["treffer"], 100 * b["treffer"]))
    if args.beste:
        alle = {n: alle[n] for n in rang[:args.beste]}
        print("gespeichert werden nur:", list(alle))
    for name, erg in alle.items():
        abbildung_datei(name, erg, L, schritt,
                        ausgabe / ("%s_%gs_%s.png" % (praefix, L, name)), args.typ)

    stamm = ausgabe / ("%s_%gs" % (praefix, L))
    abbildung_uebersicht(alle, L, schritt, str(stamm) + "_uebersicht.png")
    with open(str(stamm) + ".pkl", "wb") as fh:
        pickle.dump({"fenster_sek": L, "schritt_sek": schritt, "kanaele_neu": KANAELE,
                     "zeitreihen": alle}, fh)
    print("gespeichert:", str(stamm) + "_*.png / .pkl")


if __name__ == "__main__":
    main()
