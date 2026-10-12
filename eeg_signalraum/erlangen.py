"""
Schritt 1 + 2 auf den Absence-Daten aus Erlangen - genau wie ``eeg_signalraum.py``.

R je 3-s-Fenster, je Absence drei Fenster (Ruhe, Übergang, Anfall), λ über
``spektrum`` aus ``eeg_signalraum.py``. Unterschiede zu TUH:
    - nur zwei Patienten (CA: 25 Kanäle, ME: 21 Kanäle) -> Median und 25-75 %
      über die Absencen je Patient, nicht über Patienten
    - TUH-ABSZ als graue Referenz aus den TUH-Pickles
    - keine Vorverarbeitung, nur die EEG-Kanäle ("E ..."); PHO ist konstant 0

Die Anfallszeiten der Tabelle (von/bis) beziehen sich auf die kurzen Ausschnitte.
Diese sind datengleich Teil der langen Aufnahme; der Versatz kommt aus der
Kreuzkorrelation.

    python erlangen.py                         # SNR
    python erlangen.py --methode pca
    python erlangen.py --methode pca --skalieren
    python erlangen.py --methode dyca
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
from scipy.signal import correlate

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
from eeg_signalraum import (ARTEN, BESCHRIFTUNG, FARBE, KOPF, PANEL, fenster_je_anfall,
                            je_patient, legende_zusatz, rahmen, spektrum, stamm_zusatz,
                            stil_achse, DATEN, SURROGAT, MUTED)

ORDNER = Path("/home/data/ninalaemmermann/forschung/Data/Erlangen/EEG_Daten_erlangen")
SPERRE = 10.0          # ±s um Ereignisse ohne Endzeit: dort kein Ruhefenster
FARBE_PAT = {"CA": DATEN, "ME": SURROGAT}
MARKER_PAT = {"CA": "o", "ME": "s"}

# lange Aufnahme -> (Patient, kurzer Ausschnitt oder None, Sperr-Marker)
AUFNAHMEN = {
    "130711BB_0004-export_Absc_1_Long_Raw": ("CA", "130711BB_0004-export_Absc_1", []),
    "130711BB_0005-export_Absc_2_Long_Raw": ("CA", "130711BB_0005-export_Absc_2", ["zusätzliche Absence"]),
    "130711BB_0012-export_Absc_3_Long_Raw": ("CA", "130711BB_0012-export_Absc_3", []),
    "130711BB_0012-export_Absc_4_Long_Raw": ("CA", "130711BB_0012-export_Absc_4", []),
    "130712BA_0003-export_Absc_5_Long_Raw": ("CA", "130712BA_0003-export_Absc_5", []),
    "130712BA_0014-export_Absc_6_Long_Raw": ("CA", "130712BA_0014-export_Absc_6", []),
    "150714DB_0000-export_Abs1_Long_Raw": ("ME", "150714DB_0000-export-Abs1", []),
    "150714DB_0005-export_Abs2_Long_Raw": ("ME", "150714DB_0005-export-Abs2", ["SW #1", "SW#2"]),
    "150714DB_0009-export_Abs3_Long_Raw": ("ME", None, []),
}
# ohne kurzen Ausschnitt: Beginn aus dem Marker, Dauer aus der Tabelle
OHNE_AUSSCHNITT = {"150714DB_0009-export_Abs3_Long_Raw": ("SW#relevant", "150714DB_0009-export-Abs3.edf")}


def lies(name, preload=False):
    return mne.io.read_raw_edf(ORDNER / "EDF" / (name + ".edf"), preload=preload,
                               verbose=False, encoding="latin1")


def eeg_kanaele(raw):
    return [c for c in raw.ch_names if c.startswith("E ")]


def versatz(kurz, lang):
    """Lage des kurzen Ausschnitts in der langen Aufnahme (s), über Kanal Cz."""
    a, b = lies(kurz, True), lies(lang, True)
    sr = b.info["sfreq"]
    # kurze Ausschnitte heißen "Cz", lange Aufnahmen "E Cz" bzw. "E Cz-Ref"
    cz = lambda raw: [c for c in raw.ch_names if c.split()[-1].split("-")[0] == "Cz"][0]
    xa = a.get_data(picks=[cz(a)])[0]
    xb = b.get_data(picks=[cz(b)])[0]
    grob = (a.info["meas_date"] - b.info["meas_date"]).total_seconds()
    lo = max(0, int((grob - 5) * sr))
    hi = min(len(xb), int((grob + 5) * sr) + len(xa))
    seg = xb[lo:hi]
    c = correlate(seg - seg.mean(), xa - xa.mean(), mode="valid")
    k = int(c.argmax())
    r = np.corrcoef(seg[k:k + len(xa)], xa)[0, 1]
    return (lo + k) / sr, r


def anfallsliste():
    """Je lange Aufnahme: Patient, Absencen [(a, b)], Sperrzonen [(a, b)]."""
    tab = pd.read_csv(ORDNER / "Absc_info_neu.csv").set_index("EDF_Datei")
    zahl = lambda x: float(x) if str(x) not in ("[]", "nan") else np.nan
    aus = {}
    for lang, (pat, kurz, sperr) in AUFNAHMEN.items():
        raw = lies(lang)
        ann = list(zip(raw.annotations.onset, raw.annotations.description))
        if kurz is not None:
            off, r = versatz(kurz, lang)
            z = tab.loc[kurz + ".edf"]
            absencen = [(off + zahl(z.von), off + zahl(z.bis))]
            herkunft = "Versatz %.2f s (Korr. %.3f)" % (off, r)
        else:
            marker, zeile = OHNE_AUSSCHNITT[lang]
            t0 = [o for o, d in ann if marker in d][0]
            dauer = zahl(tab.loc[zeile].bis) - zahl(tab.loc[zeile].von)
            absencen = [(t0, t0 + dauer)]
            herkunft = "Marker '%s' + Dauer %.1f s" % (marker, dauer)
        sperrzonen = [(o - SPERRE, o + SPERRE) for o, d in ann if any(s in d for s in sperr)]
        marker = [(round(o, 1), d.replace("Marker: ", "")) for o, d in ann if d != "Marker: Marker"]
        aus[lang] = {"patient": pat, "absencen": absencen, "sperrzonen": sperrzonen,
                     "dauer": raw.n_times / raw.info["sfreq"], "herkunft": herkunft,
                     "marker": marker}
    return aus


# ===========================================================================

def auswerten(liste, L, abstand, methode, skalieren):
    eintraege = []
    for lang, info in liste.items():
        raw = lies(lang)
        picks = eeg_kanaele(raw)
        sr = raw.info["sfreq"]
        n_win = int(round(L * sr))
        alle = info["absencen"] + info["sperrzonen"]
        for a, b in info["absencen"]:
            for art, t0 in fenster_je_anfall(a, b, alle, L, abstand, info["dauer"]).items():
                i0 = int(round(t0 * sr))
                if i0 + n_win > raw.n_times:
                    continue
                Y = raw.get_data(picks=picks, start=i0, stop=i0 + n_win).T * 1e6   # µV
                sigma = Y.std(axis=0)
                if skalieren:
                    if not (sigma > 0).all():
                        continue
                    Y = (Y - Y.mean(axis=0)) / sigma
                lam, r = spektrum(Y, methode)
                if lam is None:
                    continue
                eintraege.append({"datei": lang, "patient": info["patient"], "art": art,
                                  "anfall": (a, b), "start_s": t0, "r": r,
                                  "n_kanal": len(picks), "lam": lam, "sigma": sigma})
    return eintraege


def kurven(eintraege, patient, art):
    """Spektren mit vollem Rang je Patient (Absencen × n_kanal)."""
    e = [x for x in eintraege if x["patient"] == patient and x["art"] == art and x["r"] == x["n_kanal"]]
    if not e:
        return None
    return np.array([x["lam"][:x["n_kanal"]] for x in e])


def linie(ax, A, farbe, name, methode, marker="o", n_text="", log=True):
    x = np.arange(1, A.shape[1] + 1)
    med = np.median(A, axis=0)
    lo, hi = np.percentile(A, [25, 75], axis=0)
    ax.fill_between(x, lo, hi, color=farbe, alpha=0.18, linewidth=0)
    (ax.semilogy if log else ax.plot)(x, med, marker + "-", color=farbe, markersize=4,
                label="%s%s%s" % (name, n_text, legende_zusatz(med, methode)))


def tuh_referenz(ax, tuh, art, methode, log=True):
    if tuh is None:
        return
    P, n = je_patient(tuh[art])
    if len(P) == 0:
        return
    x = np.arange(1, P.shape[1] + 1)
    med = np.median(P, axis=0)
    lo, hi = np.percentile(P, [25, 75], axis=0)
    ax.fill_between(x, lo, hi, color=MUTED, alpha=0.15, linewidth=0)
    (ax.semilogy if log else ax.plot)(x, med, "-", color=MUTED, linewidth=1.5,
                label="TUH-ABSZ (%d Pat.)%s" % (len(P), legende_zusatz(med, methode)))


def titelzeile(L, methode, skalieren):
    return ("EEG Erlangen, Absencen — %s (Fenster %.3g s%s, Median und 25–75 %% über Absencen)"
            % (KOPF[methode], L, ", StandardScaler je Fenster" if skalieren else ""))


def abbildung_zustaende(e, tuh, L, methode, skalieren, ziel, log=True):
    fig, achsen = plt.subplots(1, len(ARTEN), figsize=(17, 5.2), sharey=True)
    fig.suptitle(titelzeile(L, methode, skalieren) + ("" if log else ", lineare Achse"))
    for ax, art in zip(achsen, ARTEN):
        stil_achse(ax)
        tuh_referenz(ax, tuh, art, methode, log)
        for pat in FARBE_PAT:
            A = kurven(e, pat, art)
            if A is not None:
                linie(ax, A, FARBE_PAT[pat], pat, methode, MARKER_PAT[pat],
                      " (%d Abs., %d Kan.)" % A.shape, log=log)
        rahmen(ax, "%s — %s" % (BESCHRIFTUNG[art], PANEL[methode]), log, methode)
    fig.tight_layout()
    fig.savefig(ziel, dpi=140)
    plt.close(fig)


def abbildung_patienten(e, tuh, L, methode, skalieren, ziel, log=True):
    """Je Patientin ein Panel mit Ruhe/Übergang/Anfall übereinander, dazu TUH-ABSZ."""
    n = len(FARBE_PAT) + (tuh is not None)
    # linear: eigene y-Achse je Panel, sonst drückt TUH (≈10⁴) Erlangen an die Nulllinie
    fig, achsen = plt.subplots(1, n, figsize=(6 * n, 4.8), sharex=True, sharey=log)
    fig.suptitle(titelzeile(L, methode, skalieren) + ("" if log else ", lineare Achse"))
    for ax, pat in zip(achsen, FARBE_PAT):
        stil_achse(ax)
        for art in ARTEN:
            A = kurven(e, pat, art)
            if A is not None:
                linie(ax, A, FARBE[art], BESCHRIFTUNG[art], methode, n_text=" (%d Abs.)" % len(A), log=log)
        rahmen(ax, "Erlangen, Patientin %s — %s" % (pat, PANEL[methode]), log, methode)
    if tuh is not None:
        ax = achsen[-1]
        stil_achse(ax)
        for art in ARTEN:
            P, _ = je_patient(tuh[art])
            if len(P):
                linie(ax, P, FARBE[art], BESCHRIFTUNG[art], methode, n_text=" (%d Pat.)" % len(P), log=log)
        rahmen(ax, "TUH-ABSZ, Median über Patienten — %s" % PANEL[methode], log, methode)
    fig.tight_layout()
    fig.savefig(ziel, dpi=140)
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
    args = ap.parse_args()
    L, m = args.fenster_sek, args.methode

    liste = anfallsliste()
    print("Anfallsliste (Sekunden in der langen Aufnahme):")
    for lang, info in liste.items():
        print("  %-38s %s  Absence %s  Sperren %s  | %s | Marker %s"
              % (lang, info["patient"], ["%.1f–%.1f" % ab for ab in info["absencen"]],
                 ["%.0f–%.0f" % ab for ab in info["sperrzonen"]], info["herkunft"], info["marker"]))

    e = auswerten(liste, L, args.ruhe_abstand, m, args.skalieren)
    print("\nFenster je Patient und Zustand (voller Rang / gesamt):")
    for pat in FARBE_PAT:
        for art in ARTEN:
            x = [z for z in e if z["patient"] == pat and z["art"] == art]
            print("  %s %-9s %d / %d" % (pat, art, sum(z["r"] == z["n_kanal"] for z in x), len(x)))

    zusatz = stamm_zusatz(m, args.skalieren)
    ausgabe = HIER / "ergebnisse"
    ausgabe.mkdir(exist_ok=True)
    tuh_pkl = ausgabe / ("eigenwerte_eeg_%s%gs.pkl" % (zusatz, L))
    tuh = pickle.load(open(tuh_pkl, "rb"))["ergebnisse"].get("absz") if tuh_pkl.exists() else None
    if tuh is None:
        print("Hinweis: keine TUH-Referenz gefunden (%s)" % tuh_pkl.name)

    stamm = ausgabe / ("erlangen_eigenwerte_%s%gs" % (zusatz, L))
    with open(str(stamm) + ".pkl", "wb") as fh:
        pickle.dump({"fenster_sek": L, "ruhe_abstand": args.ruhe_abstand, "methode": m,
                     "skalieren": args.skalieren, "anfallsliste": liste, "fenster": e}, fh)
    abbildung_zustaende(e, tuh, L, m, args.skalieren, str(stamm) + "_zustaende.png")
    abbildung_patienten(e, tuh, L, m, args.skalieren, str(stamm) + "_patienten.png")
    abbildung_zustaende(e, tuh, L, m, args.skalieren, str(stamm) + "_zustaende_linear.png", log=False)
    abbildung_patienten(e, tuh, L, m, args.skalieren, str(stamm) + "_patienten_linear.png", log=False)
    print("gespeichert:", str(stamm) + "_*.png / .pkl")


if __name__ == "__main__":
    main()
