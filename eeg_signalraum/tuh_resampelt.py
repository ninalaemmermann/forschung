"""
Schritt 1 + 2 auf den resampelten, ungefilterten TUH-Absence-Daten
(``Data/resampelt_ungefiltert``) - Vergleich mit den bisherigen TUH-ABSZ-Daten
(``Data/ABSZ_seizure_WB``: gefiltert, 27 Kanäle, Average-Referenz).

Neue Daten so, wie sie vorliegen: die 21 EEG-Kanäle, die alle Aufnahmen haben, keine
Filterung, keine Umreferenzierung (gespeicherte Referenz: Linked Ear bzw. AR).
Anfallszeiten neu aus der ``.csv_bi`` neben der EDF, alt aus ``ABSZ_seizures.csv``.
Verglichen wird nur über die Aufnahmen, die in beiden Ordnern liegen.

Zusätzlich: Prüfung der neuen Anfallszeiten gegen ``ABSZ_seizures.csv``.

    python tuh_resampelt.py                    # SNR
    python tuh_resampelt.py --methode pca
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
import pandas as pd

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
from eeg_signalraum import (ARTEN, BESCHRIFTUNG, FARBE, KOPF, PANEL, fenster_je_anfall,
                            legende_zusatz, rahmen, spektrum, stamm_zusatz, stil_achse,
                            werte_typ_aus, get_sz_start_end, DATEN, MUTED)

BASE = Path("/home/data/ninalaemmermann/forschung")
NEU = BASE / "Data" / "resampelt_ungefiltert"
KANAELE = ["EEG " + k for k in ("FP1", "FP2", "F3", "F4", "F7", "F8", "FZ", "C3", "C4", "CZ",
                                "T7", "T8", "P3", "P4", "P7", "P8", "PZ", "O1", "O2", "A1", "A2")]
# neuer Dateiname -> Name in ABSZ_seizures.csv / ABSZ_seizure_WB (gleiche Aufnahme:
# Dauer 670 s und alle sechs Anfallszeiten identisch)
ALIAS = {"aaaaadyf_s001_t000": "aaaaadyf_s001_t001"}
TOLERANZ = 0.01        # s, für "gleiche Anfallszeit"
# Einzelvergleich: viele Anfälle (bqv, cyi), abweichende Annotation (crb), AR-Montage (mtc)
EINZELN = ("aaaaabqv_s002_t001", "aaaaacyi_s001_t001", "aaaaacrb_s001_t001", "aaaaamtc_s001_t000")

# Datensatz -> (Beschriftung, Farbe, Marker, voller Rang)
SAETZE = {
    "alt": ("TUHZ-ABSZ WB", MUTED, "o", 27),
    "neu": ("TUHZ-ABSZ resampelt ungefiltert", DATEN, "s", len(KANAELE)),
}


# ===========================================================================
# Dateien und Anfallszeiten
# ===========================================================================

def neue_dateien(typ="absz"):
    """Ungefilterte Aufnahmen: absz = alles außer dem Ordner fnsz, fnsz = nur dieser."""
    return {p.stem: p for p in sorted(NEU.rglob("*.edf"))
            if (p.relative_to(NEU).parts[0] == "fnsz") == (typ == "fnsz")}


def lies_annotation(pfad):
    return pd.read_csv(pfad, comment="#")


def anfaelle_bi(edf):
    """Alle Anfälle der Aufnahme (TERM, seiz) als [(a, b)], sortiert."""
    d = lies_annotation(edf.with_suffix(".csv_bi"))
    d = d[d.label == "seiz"]
    return sorted((float(a), float(b)) for a, b in zip(d.start_time, d.stop_time))


def pruefe_anfallszeiten(dateien):
    """Neue .csv_bi gegen ABSZ_seizures.csv, dazu kanalweise .csv gegen .csv_bi."""
    master = get_sz_start_end(BASE / "ABSZ_seizures.csv")
    zeilen = []
    for name, edf in dateien.items():
        bi = anfaelle_bi(edf)
        m = sorted((float(a), float(b)) for a, b in master.get(ALIAS.get(name, name), []))
        if not m:
            status, abw = "nicht in Master", np.nan
        elif len(m) != len(bi):
            status, abw = "Anzahl verschieden", np.nan
        else:
            abw = float(np.abs(np.array(m) - np.array(bi)).max())
            status = "gleich" if abw <= TOLERANZ else "Abweichung"
        # kanalweise Annotation: liegt jedes Ereignis in einem Anfall der .csv_bi?
        k = lies_annotation(edf.with_suffix(".csv"))
        k = k[k.label != "bckg"]
        drin = all(any(a - TOLERANZ <= s and e <= b + TOLERANZ for a, b in bi)
                   for s, e in zip(k.start_time, k.stop_time))
        zeilen.append({"datei": name, "name_master": ALIAS.get(name, name),
                       "labels": ",".join(sorted(k.label.unique())),
                       "n_bi": len(bi), "n_master": len(m), "status": status,
                       "max_abweichung_s": abw, "kanalweise_in_bi": drin,
                       "bi": bi, "master": m})
    return pd.DataFrame(zeilen)


# ===========================================================================
# Schritt 1 + 2
# ===========================================================================

def werte_neu_aus(dateien, L, abstand, methode, skalieren):
    """Fenster der neuen Daten, wie gespeichert; Einträge wie in ``werte_typ_aus``."""
    ergebnis = {art: [] for art in ARTEN}
    for name, edf in dateien.items():
        anfaelle = anfaelle_bi(edf)
        if not anfaelle:
            continue
        raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
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
                Y = raw.get_data(picks=KANAELE, start=i0, stop=i0 + n_win).T * 1e6   # µV
                sigma = Y.std(axis=0)
                if skalieren:
                    if not (sigma > 0).all():
                        continue
                    Y = (Y - Y.mean(axis=0)) / sigma
                lam, r = spektrum(Y, methode)
                if lam is None:
                    continue
                ergebnis[art].append({"datei": name, "anfall": (a, b), "start_s": i0 / sr,
                                      "r": r, "lam": lam, "sigma": sigma})
    return ergebnis


def rang_bericht(ergebnis, titel):
    print(titel)
    for art in ARTEN:
        rs = [e["r"] for e in ergebnis[art]]
        print("  %-9s n = %4d   r: %s" % (art, len(rs), "-" if not rs else
              "%d … %d (Median %d)" % (min(rs), max(rs), np.median(rs))))


def je_patient_voll(eintraege, r_voll):
    """Median-Spektrum je Patient über Fenster mit vollem Rang, auf r_voll gekürzt."""
    gruppen = {}
    for e in eintraege:
        if e["r"] == r_voll:
            gruppen.setdefault(e["datei"][:8], []).append(e["lam"][:r_voll])
    return np.array([np.median(v, axis=0) for v in gruppen.values()]), \
        sum(len(v) for v in gruppen.values())


# ===========================================================================
# Abbildungen
# ===========================================================================

def linie(ax, P, farbe, name, methode, marker="o", log=True):
    x = np.arange(1, P.shape[1] + 1)
    med = np.median(P, axis=0)
    lo, hi = np.percentile(P, [25, 75], axis=0)
    ax.fill_between(x, lo, hi, color=farbe, alpha=0.18, linewidth=0)
    (ax.semilogy if log else ax.plot)(x, med, marker + "-", color=farbe, markersize=4,
                                      label="%s%s" % (name, legende_zusatz(med, methode, log)))


def titelzeile(L, methode, skalieren, n_auf, log):
    return ("TUHZ-ABSZ WB-Filter vs. resampelt ungefiltert — %s (Fenster %.3g s%s, Median und "
            "25–75%%%s)"
            % (KOPF[methode], L, ", StandardScaler je Fenster" if skalieren else "",
               "" if log else ", lineare Achse"))


def abbildung_datensaetze(erg, L, methode, skalieren, n_auf, ziel, log=True):
    """Je Datensatz ein Panel, darin Ruhe / Übergang / Anfall (wie Erlangen-Plot)."""
    fig, achsen = plt.subplots(1, len(SAETZE), figsize=(6 * len(SAETZE), 4.8), sharex=True,
                               sharey=log)
    fig.suptitle(titelzeile(L, methode, skalieren, n_auf, log))
    for ax, (satz, (name, _, _, r_voll)) in zip(achsen, SAETZE.items()):
        stil_achse(ax)
        for art in ARTEN:
            P, n = je_patient_voll(erg[satz][art], r_voll)
            if len(P):
                linie(ax, P, FARBE[art], "%s (%d Pat., %d Fenster)" % (BESCHRIFTUNG[art], len(P), n),
                      methode, log=log)
        rahmen(ax, name, log, methode)
    fig.tight_layout()
    fig.savefig(ziel, dpi=140)
    plt.close(fig)


def abbildung_zustaende(erg, L, methode, skalieren, n_auf, ziel, log=True):
    """Je Zustand ein Panel, darin alt und neu."""
    fig, achsen = plt.subplots(1, len(ARTEN), figsize=(17, 5.2), sharey=True)
    fig.suptitle(titelzeile(L, methode, skalieren, n_auf, log))
    for ax, art in zip(achsen, ARTEN):
        stil_achse(ax)
        for satz, (name, farbe, marker, r_voll) in SAETZE.items():
            P, n = je_patient_voll(erg[satz][art], r_voll)
            if len(P):
                linie(ax, P, farbe, "%s (%d Pat.)" % (name, len(P)), methode, marker, log)
        rahmen(ax, "%s — %s" % (BESCHRIFTUNG[art], PANEL[methode]), log, methode)
    fig.tight_layout()
    fig.savefig(ziel, dpi=140)
    plt.close(fig)


def je_datei(eintraege, datei, r_voll):
    """Spektren der Fenster einer Aufnahme mit vollem Rang (n_fenster, r_voll)."""
    v = [e["lam"][:r_voll] for e in eintraege if e["datei"] == datei and e["r"] == r_voll]
    return np.array(v) if v else None


def abbildung_einzeln(erg, auswahl, L, methode, skalieren, ziel, log=True):
    """Zeile je Aufnahme, Spalte je Datensatz; darin Ruhe / Übergang / Anfall.

    Median und 25–75 % über die Fenster (= Anfälle) der Aufnahme; y-Achse je Zeile gemeinsam.
    """
    fig, achsen = plt.subplots(len(auswahl), len(SAETZE), figsize=(6 * len(SAETZE), 4 * len(auswahl)),
                               sharex=True, sharey="row" if log else False, squeeze=False)
    fig.suptitle("TUH-ABSZ einzelne Aufnahmen, alt vs. resampelt ungefiltert — %s (Fenster %.3g s%s, "
                 "Median und 25–75%% über die Anfälle der Aufnahme%s)"
                 % (KOPF[methode], L, ", StandardScaler je Fenster" if skalieren else "",
                    "" if log else ", lineare Achse"), fontsize=10)
    for zeile, neu_name in zip(achsen, auswahl):
        for ax, (satz, (name, _, _, r_voll)) in zip(zeile, SAETZE.items()):
            stil_achse(ax)
            datei = ALIAS.get(neu_name, neu_name) if satz == "alt" else neu_name
            for art in ARTEN:
                A = je_datei(erg[satz][art], datei, r_voll)
                if A is not None:
                    linie(ax, A, FARBE[art], "%s (%d Fenster)" % (BESCHRIFTUNG[art], len(A)),
                          methode, log=log)
            rahmen(ax, "%s — %s" % (datei, name), log, methode)
            ax.title.set_fontsize(8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(ziel, dpi=120)
    plt.close(fig)


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methode", choices=("snr", "pca", "pca_roh", "dyca"), default="snr")
    ap.add_argument("--skalieren", action="store_true",
                    help="StandardScaler je Fenster vor Schritt 1 + 2")
    ap.add_argument("--fenster-sek", type=float, default=3.0)
    ap.add_argument("--ruhe-abstand", type=float, default=30.0)
    ap.add_argument("--einzeln", nargs="+", default=list(EINZELN),
                    help="neue Dateinamen (BName) für den Einzelvergleich")
    args = ap.parse_args()
    L, m = args.fenster_sek, args.methode
    ausgabe = HIER / "ergebnisse"
    ausgabe.mkdir(exist_ok=True)

    dateien = neue_dateien()
    pruef = pruefe_anfallszeiten(dateien)
    print("Anfallszeiten .csv_bi (neu) gegen ABSZ_seizures.csv:")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pruef.drop(columns=["bi", "master"]).to_string(index=False))
    for _, z in pruef[pruef.status.isin(["Abweichung", "Anzahl verschieden"])].iterrows():
        print("  %s:" % z.datei)
        for (a, b), (c, d) in zip(z.bi, z.master):
            if max(abs(a - c), abs(b - d)) > TOLERANZ:
                print("    neu %.3f–%.3f s   Master %.3f–%.3f s" % (a, b, c, d))
    pruef.drop(columns=["bi", "master"]).to_csv(ausgabe / "tuh_resampelt_anfallszeiten.csv",
                                                index=False)

    alt_dateien = {p.name.split("_res")[0] for p in (BASE / "Data" / "ABSZ_seizure_WB").glob("*.edf")}
    gemeinsam = sorted(n for n in dateien if ALIAS.get(n, n) in alt_dateien)
    print("\n%d gemeinsame Aufnahmen; nur neu: %s; nur alt: %s"
          % (len(gemeinsam), sorted(set(dateien) - set(gemeinsam)),
             sorted(alt_dateien - {ALIAS.get(n, n) for n in gemeinsam})))

    print("\nneu:")
    neu = werte_neu_aus(dateien, L, args.ruhe_abstand, m, args.skalieren)
    print("alt (ABSZ_seizure_WB):")
    alt = werte_typ_aus("absz", BASE, L, args.ruhe_abstand, m, args.skalieren)

    namen_alt = {ALIAS.get(n, n) for n in gemeinsam}
    erg = {"alt": {art: [e for e in alt[art] if e["datei"] in namen_alt] for art in ARTEN},
           "neu": {art: [e for e in neu[art] if e["datei"] in gemeinsam] for art in ARTEN}}
    print()
    for satz in SAETZE:
        rang_bericht(erg[satz], "%s, gemeinsame Aufnahmen (voller Rang = %d):"
                     % (satz, SAETZE[satz][3]))

    stamm = ausgabe / ("tuh_resampelt_%s%gs" % (stamm_zusatz(m, args.skalieren), L))
    with open(str(stamm) + ".pkl", "wb") as fh:
        pickle.dump({"fenster_sek": L, "ruhe_abstand": args.ruhe_abstand, "methode": m,
                     "skalieren": args.skalieren, "kanaele": KANAELE, "gemeinsam": gemeinsam,
                     "anfallszeiten": pruef, "neu_alle": neu, "alt_alle": alt}, fh)
    n_auf = len(gemeinsam)
    for log, s in ((True, ""), (False, "_linear")):
        abbildung_datensaetze(erg, L, m, args.skalieren, n_auf, "%s_datensaetze%s.png" % (stamm, s), log)
        abbildung_zustaende(erg, L, m, args.skalieren, n_auf, "%s_zustaende%s.png" % (stamm, s), log)
        abbildung_einzeln(erg, args.einzeln, L, m, args.skalieren, "%s_einzeln%s.png" % (stamm, s), log)
    print("gespeichert:", str(stamm) + "_*.png / .pkl, tuh_resampelt_anfallszeiten.csv")


if __name__ == "__main__":
    main()
