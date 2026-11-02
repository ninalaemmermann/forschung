"""
DyCA-Eigenwerte im Zeitverlauf über die ganzen Aufnahmen - Erlangen (lange Aufnahmen, roh)
und TUH resampelt ungefiltert (21 Kanäle, Satz "neu"). Zum Vergleich auch TUH-ABSZ WB
(gefiltert, 27 Kanäle, Satz "alt"), nur auf Wunsch: --datensaetze tuh_wb.
Andere Anfallstypen (nur als WB vorhanden): --datensaetze gnsz fnsz cpsz; je Patient eine
zufällige Aufnahme mit Anfall, höchstens --max-patienten Patienten (fester Startwert).

λ sind die generalisierten DyCA-Eigenwerte C1·C0⁻¹·C1ᵀ·u = λ·C2·u aus ``spektrum``
(``eeg_signalraum.py``, methode "dyca"), λ in [0, 1], je Fenster absteigend. Gleitendes
Fenster (Standard 3 s, Schritt 0,5 s), Zeitachse = Fenstermitte. Gezeichnet λ1–λ5, linear.

    python dyca_zeitreihe.py                          # beide Datensätze, je die besten 4
    python dyca_zeitreihe.py --datensaetze erlangen --beste 9

Bewertung und Detektion wie in ``tuh_resampelt_zeitreihe.py`` / ``erlangen_zeitreihe.py``:
    r        Korrelation log10 λ1 gegen Anfallsanteil des Fensters (Rangfolge)
    AUC      λ1 bzw. λ2 als Anfallsdetektor, Anfall gegen anfallsfrei; 0,5 = Zufall
    Spitzen  von den 5 höchsten λ1-Spitzen (≥ 5 s Abstand): wie viele im Anfall/Sperrzone
Erlangen: Ereignisse ohne Endzeit sind ±10-s-Sperrzonen (weder Anfall noch anfallsfrei).
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import mne
from eeg_signalraum import stil_achse, get_sz_start_end, MUTED, SURROGAT, INK_SEK, DATEN
from stil import RAMPE_STUFEN
from erlangen import anfallsliste, eeg_kanaele, lies
from erlangen_zeitreihe import detektion, N_SPITZEN
from tuh_resampelt import neue_dateien
from tuh_resampelt_zeitreihe import anfaelle_zeichnen, bewertung, gemeinsame, lade, zeitreihe

N_LAM = 5
_rampe = LinearSegmentedColormap.from_list("rampe", RAMPE_STUFEN[::-1])
FARBE_LAM = [_rampe(i / (N_LAM - 1)) for i in range(N_LAM)]       # λ1 dunkel … λ5 hell
TITEL = {"erlangen": "Erlangen (roh, alle E-Kanäle)", "tuh": "TUH resampelt ungefiltert (21 Kan.)",
         "tuh_wb": "TUH-ABSZ WB (gefiltert, 27 Kan., Avg-Ref)",
         "tuh_fnsz": "TUH-FNSZ resampelt ungefiltert (21 Kan.)",
         "tuh_fnsz_wb": "TUH-FNSZ WB (gefiltert, 27 Kan., Avg-Ref), dieselben Aufnahmen"}
# Datensatz -> (Satz in lade, Anfallstyp)
TUH_SAETZE = {"tuh": ("neu", "absz"), "tuh_wb": ("alt", "absz"),
              "tuh_fnsz": ("neu", "fnsz"), "tuh_fnsz_wb": ("alt", "fnsz")}
# Vergleichspaare (WB, ungefiltert)
PAARE = (("tuh_wb", "tuh"), ("tuh_fnsz_wb", "tuh_fnsz"))
TYP_TITEL = {"gnsz": "GNSZ", "fnsz": "FNSZ", "cpsz": "CPSZ"}
for _t, _n in TYP_TITEL.items():
    TITEL[_t] = "TUH-%s WB (gefiltert, 27 Kan., Avg-Ref)" % _n
STANDARD = ["erlangen", "tuh"]
BASE = HIER.parent
MAX_PATIENTEN = 20


def aufnahmen(datensatz, max_patienten=MAX_PATIENTEN):
    """Je Aufnahme: (Daten µV, Abtastrate, Anfälle, Sperrzonen, Kurzinfo)."""
    if datensatz == "erlangen":
        for name, info in anfallsliste().items():
            raw = lies(name, preload=True)
            picks = eeg_kanaele(raw)
            yield name, raw.get_data(picks=picks) * 1e6, raw.info["sfreq"], info["absencen"], \
                info["sperrzonen"], "Patientin %s, %d Kanäle" % (info["patient"], len(picks))
    elif datensatz in TYP_TITEL:
        st = datensatz.upper()
        liste = get_sz_start_end(BASE / ("%s_seizures.csv" % st))
        ordner = BASE / "Data" / ("%s_seizure_WB" % st)
        je_patient = {}
        for datei, sz in sorted(liste.items()):
            anf = sorted((float(a), float(b)) for a, b in sz if np.isfinite(a) and np.isfinite(b) and b > a)
            pfad = ordner / ("%s_res.OWN11101_filtWB_avg.edf" % datei)
            if anf and pfad.exists():
                je_patient.setdefault(datei.split("_")[0], []).append((datei, pfad, anf))
        rng = np.random.default_rng(0)
        patienten = sorted(je_patient)
        patienten = [patienten[i] for i in sorted(rng.choice(len(patienten),
                     min(max_patienten, len(patienten)), replace=False))]
        for pat in patienten:
            datei, pfad, anf = je_patient[pat][rng.integers(len(je_patient[pat]))]
            raw = mne.io.read_raw_edf(pfad, preload=True, verbose=False)
            X = raw.get_data() * 1e6
            yield datei, X, raw.info["sfreq"], anf, [], "%d Kanäle, %d Anfälle" % (X.shape[0], len(anf))
    else:
        satz, typ = TUH_SAETZE[datensatz]
        pfade = neue_dateien(typ)
        for name in gemeinsame(pfade, typ):
            X, sr, anfaelle = lade(satz, name, pfade[name], typ)
            yield name, X, sr, anfaelle, [], "%d Kanäle, %d Anfälle" % (X.shape[0], len(anfaelle))


def rechne(datensatz, L, schritt, max_patienten=MAX_PATIENTEN):
    alle = {}
    for name, X, sr, anfaelle, sperr, kurz in aufnahmen(datensatz, max_patienten):
        t, lam = zeitreihe(X, sr, L, schritt, X.shape[0], methode="dyca")
        e = {"t": t, "lam": lam, "anfaelle": anfaelle, "sperrzonen": sperr, "kurz": kurz}
        e["bewertung"] = bewertung(e, L, sperr)
        e["det1"] = detektion(e, L, schritt, sperr, spalte=0)
        e["det2"] = detektion(e, L, schritt, sperr, spalte=1)
        e["ohne"] = np.isnan(lam[:, 0]).mean()
        alle[name] = e
        print("fertig: %s %s (%.1f %% ohne Lösung)" % (datensatz, name, 100 * e["ohne"]), flush=True)
    return alle


def tabelle(datensatz, alle, rang, snr=None):
    print("\n%s, DyCA – Übereinstimmung Anfall (sortiert nach r):" % TITEL[datensatz])
    print("%-38s %6s %8s %8s %8s %s" % ("Aufnahme", "r", "AUC λ1", "AUC λ2", "Spitzen",
                                         "| SNR: AUC λ1, Spitzen" if snr else ""))
    for n in rang:
        e = alle[n]
        zus = ""
        if snr and n in snr:
            d = snr[n]["detektion"]
            zus = "|      %.2f   %d/%d" % (d["auc"], d["spitzen_treffer"].sum(), len(d["spitzen_treffer"]))
        print("%-38s %6.2f %8.2f %8.2f %5d/%d %s" % (n, e["bewertung"]["r"], e["det1"]["auc"],
              e["det2"]["auc"], e["det1"]["spitzen_treffer"].sum(), N_SPITZEN, zus))
    a1 = [alle[n]["det1"]["auc"] for n in rang]
    a2 = [alle[n]["det2"]["auc"] for n in rang]
    tr = sum(alle[n]["det1"]["spitzen_treffer"].sum() for n in rang)
    print("gesamt: AUC λ1 Median %.2f (%.2f–%.2f), AUC λ2 Median %.2f (%.2f–%.2f), Spitzen %d/%d"
          % (np.nanmedian(a1), np.nanmin(a1), np.nanmax(a1), np.nanmedian(a2), np.nanmin(a2),
             np.nanmax(a2), tr, N_SPITZEN * len(rang)))


def sperren_zeichnen(ax, sperr, beschriften=False):
    for k, (a, z) in enumerate(sperr):
        ax.axvspan(a, z, color=MUTED, alpha=0.15, linewidth=0,
                   label="Ereignis ohne Ende (±10 s)" if beschriften and k == 0 else None)


def abbildung(datensatz, name, e, L, schritt, ziel):
    fig, ax = plt.subplots(figsize=(18, 5))
    d = e["det1"]
    fig.suptitle("%s — DyCA-Eigenwerte über die Zeit (Fenster %.3g s, Schritt %.3g s); λ1–λ%d, "
                 "orange: Anfall%s" % (name, L, schritt, N_LAM,
                                       ", grau: Ereignis ohne Endzeit" if e["sperrzonen"] else ""),
                 fontsize=11)
    stil_achse(ax)
    anfaelle_zeichnen(ax, e["anfaelle"], beschriften=True)
    sperren_zeichnen(ax, e["sperrzonen"], beschriften=True)
    for i in range(N_LAM):
        ax.plot(e["t"], e["lam"][:, i], color=FARBE_LAM[i], linewidth=0.9, label="λ%d" % (i + 1))
    for tt, tr in zip(d["spitzen_t"], d["spitzen_treffer"]):
        ax.plot(tt, 1.05, "v", color=SURROGAT if tr else INK_SEK, markersize=7)
    ax.plot([], [], "v", color=INK_SEK, label="Top-%d-Spitze λ1 außerhalb" % N_SPITZEN)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("DyCA-Eigenwert λ")
    ax.set_title("%s — %s — r = %.2f, AUC λ1 = %.2f, AUC λ2 = %.2f, Top-%d-Spitzen im Anfall: %d/%d"
                 % (TITEL[datensatz], e["kurz"], e["bewertung"]["r"], d["auc"], e["det2"]["auc"],
                    N_SPITZEN, d["spitzen_treffer"].sum(), len(d["spitzen_treffer"])), fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=9, loc="upper left", bbox_to_anchor=(0, -0.12))
    ax.set_xlabel("Zeit (Fenstermitte) [s]")
    ax.set_xlim(0, e["t"][-1] + L / 2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ziel, dpi=130)
    plt.close(fig)


def abbildung_uebersicht(datensatz, alle, L, schritt, ziel):
    fig, achsen = plt.subplots(len(alle), 1, figsize=(18, 2.8 * len(alle)), squeeze=False)
    fig.suptitle("%s, DyCA λ1 über die Zeit — Fenster %.3g s, Schritt %.3g s; orange: Anfall"
                 % (TITEL[datensatz], L, schritt), fontsize=11)
    for ax, (name, e) in zip(achsen[:, 0], alle.items()):
        stil_achse(ax)
        anfaelle_zeichnen(ax, e["anfaelle"])
        sperren_zeichnen(ax, e["sperrzonen"])
        ax.plot(e["t"], e["lam"][:, 0], color=DATEN, linewidth=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_title("%s (%s) — r = %.2f, AUC λ1 = %.2f" % (name, e["kurz"], e["bewertung"]["r"],
                     e["det1"]["auc"]), fontsize=10)
        ax.set_ylabel("λ1")
        ax.set_xlim(0, e["t"][-1] + L / 2)
    achsen[-1, 0].set_xlabel("Zeit (Fenstermitte) [s]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ziel, dpi=120)
    plt.close(fig)


def abbildung_vergleich(name, ergebnis, L, schritt, ziel):
    """Eine TUH-Aufnahme: DyCA λ1–λ5, WB gefiltert oben, resampelt ungefiltert unten."""
    saetze = next([wb, un] for wb, un in PAARE
                  if name in ergebnis.get(un, {}).get("zeitreihen", {}))
    saetze = [ds for ds in saetze if name in ergebnis.get(ds, {}).get("zeitreihen", {})]
    fig, achsen = plt.subplots(len(saetze), 1, figsize=(18, 4.2 * len(saetze)), sharex=True,
                               squeeze=False)
    fig.suptitle("%s — DyCA-Eigenwerte über die Zeit (Fenster %.3g s, Schritt %.3g s); λ1–λ%d, "
                 "orange: Anfälle" % (name, L, schritt, N_LAM), fontsize=11)
    for ax, ds in zip(achsen[:, 0], saetze):
        e = ergebnis[ds]["zeitreihen"][name]
        stil_achse(ax)
        anfaelle_zeichnen(ax, e["anfaelle"], beschriften=True)
        for i in range(N_LAM):
            ax.plot(e["t"], e["lam"][:, i], color=FARBE_LAM[i], linewidth=0.9, label="λ%d" % (i + 1))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("DyCA-Eigenwert λ")
        ax.set_title("%s — %s — AUC λ1 = %.2f, Top-%d-Spitzen im Anfall: %d/%d"
                     % (TITEL[ds], e["kurz"], e["det1"]["auc"], N_SPITZEN,
                        e["det1"]["spitzen_treffer"].sum(), N_SPITZEN), fontsize=10)
        ax.legend(fontsize=8, frameon=False, ncol=6, loc="lower right")
    achsen[-1, 0].set_xlabel("Zeit (Fenstermitte) [s]")
    achsen[-1, 0].set_xlim(0, max(ergebnis[ds]["zeitreihen"][name]["t"][-1] for ds in saetze) + L / 2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ziel, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datensaetze", nargs="+", choices=tuple(TITEL), default=STANDARD)
    ap.add_argument("--fenster-sek", type=float, default=3.0)
    ap.add_argument("--schritt-sek", type=float, default=0.5)
    ap.add_argument("--beste", type=int, default=4, help="je Datensatz nur die N besten (nach r) speichern")
    ap.add_argument("--max-patienten", type=int, default=MAX_PATIENTEN,
                    help="GNSZ/FNSZ/CPSZ: höchstens so viele Patienten (je eine Aufnahme)")
    ap.add_argument("--vergleich", nargs="*", default=[],
                    help="TUH-Aufnahmen: WB gefiltert über ungefiltert zeichnen (aus dem .pkl)")
    args = ap.parse_args()
    L, schritt = args.fenster_sek, args.schritt_sek
    ausgabe = HIER / "ergebnisse"
    ausgabe.mkdir(exist_ok=True)
    stamm = "dyca_zeitreihe_%gs" % L

    pkl = ausgabe / (stamm + ".pkl")
    ergebnis = pickle.load(open(pkl, "rb"))["datensaetze"] if pkl.exists() else {}
    for ds in args.datensaetze:
        alle = rechne(ds, L, schritt, args.max_patienten)
        rang = sorted(alle, key=lambda n: -np.nan_to_num(alle[n]["bewertung"]["r"], nan=-9))
        snr = None
        snr_pkl = ausgabe / ("erlangen_zeitreihe_%gs.pkl" % L)
        if ds == "erlangen" and snr_pkl.exists():
            snr = pickle.load(open(snr_pkl, "rb"))["zeitreihen"]
        tabelle(ds, alle, rang, snr)
        beste = rang[:args.beste]
        print("gespeichert werden nur:", beste)
        for n in beste:
            abbildung(ds, n, alle[n], L, schritt, ausgabe / ("%s_%s_%s.png" % (stamm, ds, n)))
        if beste:
            abbildung_uebersicht(ds, {n: alle[n] for n in beste}, L, schritt,
                                 ausgabe / ("%s_%s_uebersicht.png" % (stamm, ds)))
        ergebnis[ds] = {"zeitreihen": alle, "rang": rang}

    with open(pkl, "wb") as fh:
        pickle.dump({"fenster_sek": L, "schritt_sek": schritt, "methode": "dyca",
                     "datensaetze": ergebnis}, fh)
    for name in args.vergleich:
        abbildung_vergleich(name, ergebnis, L, schritt,
                            ausgabe / ("%s_vergleich_wb_%s.png" % (stamm, name)))
    print("gespeichert:", ausgabe / (stamm + "_*.png / .pkl"))


if __name__ == "__main__":
    main()
