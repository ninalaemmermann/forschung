"""
Zeitverlauf von λ = 1 + SNR über die ganzen langen Erlangen-Aufnahmen - wie
``tuh_resampelt_zeitreihe.py``, aber ein Datensatz (alle "E ..."-Kanäle, roh, µV).

Je Aufnahme ist nur eine Absence mit Beginn und Ende bekannt (``anfallsliste`` aus
``erlangen.py``). Ereignisse ohne Endzeit ("zusätzliche Absence", "SW #1", "SW#2") sind
±10-s-Sperrzonen: grau gezeichnet, in der Bewertung weder Anfall noch anfallsfrei.

    python erlangen_zeitreihe.py              # alle 9 rechnen, die besten 4 speichern
    python erlangen_zeitreihe.py --beste 9

Bewertung je Aufnahme (Rangfolge nach r):
    r        Korrelation log10 λ1 gegen Anfallsanteil des Fensters (wie TUH)
    Treffer  Median-λ1 der Absence über dem 90-%-Quantil der anfallsfreien Fenster
Detektion (λ1 als Anfallsdetektor), auch für TUH aus tuh_resampelt_zeitreihe_<L>s.pkl:
    AUC      Fenster mit ≥ 50 % Anfall gegen anfallsfreie Fenster; 0,5 = Zufall, 1 = perfekt
    Spitzen  von den N_SPITZEN höchsten λ1-Spitzen (≥ 5 s Abstand): Anteil in Anfall
             oder Sperrzone - wie spezifisch sind hohe Eigenwerte für Anfälle?
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
from eeg_signalraum import stil_achse, MUTED, DATEN, SURROGAT, INK_SEK
from erlangen import anfallsliste, eeg_kanaele, lies
from tuh_resampelt_zeitreihe import (N_LAM, FARBE_LAM, anfaelle_zeichnen, anfallsanteil,
                                     bewertung, zeitreihe)

N_SPITZEN = 5
SPITZEN_ABSTAND = 5.0     # s


def detektion(e, L, schritt, ausschluss=(), spalte=0):
    """AUC von λ_(spalte+1) (Anfall gegen anfallsfrei) und Lage seiner höchsten Spitzen."""
    t, l = e["t"], np.log10(e["lam"][:, spalte])
    ok = np.isfinite(l)
    gesperrt = anfallsanteil(t, L, ausschluss) > 0
    anteil = anfallsanteil(t, L, e["anfaelle"])
    im, frei = ok & ~gesperrt & (anteil >= 0.5), ok & ~gesperrt & (anteil == 0)
    auc = np.nan
    if im.sum() and frei.sum():
        auc = mannwhitneyu(l[im], l[frei]).statistic / (im.sum() * frei.sum())
    # Spitzen auf λ1, NaN durch Minimum ersetzt
    lf = np.where(ok, l, np.nanmin(l))
    idx, _ = find_peaks(lf, distance=max(1, int(round(SPITZEN_ABSTAND / schritt))))
    idx = idx[np.argsort(lf[idx])[::-1][:N_SPITZEN]]
    ereignis = (anteil > 0) | gesperrt
    return {"auc": auc, "spitzen_t": t[idx], "spitzen_lam": e["lam"][idx, spalte],
            "spitzen_treffer": ereignis[idx], "anteil_spitzen": ereignis[idx].mean()}


def abbildung(name, e, info, L, schritt, ziel):
    fig, ax = plt.subplots(figsize=(18, 5))
    b, d = e["bewertung"], e["detektion"]
    fig.suptitle("%s — Schritt 1+2, λ = 1 + SNR über die Zeit (Fenster %.3g s, Schritt %.3g s); "
                 "λ1–λ3, orange: Absence, grau: Ereignis ohne Endzeit" % (name, L, schritt),
                 fontsize=11)
    stil_achse(ax)
    anfaelle_zeichnen(ax, e["anfaelle"], beschriften=True)
    for k, (a, z) in enumerate(info["sperrzonen"]):
        ax.axvspan(a, z, color=MUTED, alpha=0.15, linewidth=0,
                   label="Ereignis ohne Ende (±10 s)" if k == 0 else None)
    for i in range(N_LAM):
        ax.semilogy(e["t"], e["lam"][:, i], color=FARBE_LAM[i], linewidth=0.9, label="λ%d" % (i + 1))
    for tt, ll, tr in zip(d["spitzen_t"], d["spitzen_lam"], d["spitzen_treffer"]):
        ax.plot(tt, ll * 2.2, "v", color=SURROGAT if tr else INK_SEK, markersize=7)
    ax.plot([], [], "v", color=INK_SEK, label="Top-%d-Spitze außerhalb" % N_SPITZEN)
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_ylim(0.8, np.nanmax(e["lam"][:, 0]) * 6)
    ax.set_ylabel("λ = 1 + SNR")
    ax.set_title("Patientin %s, %d Kanäle — r = %.2f, AUC = %.2f, Top-%d-Spitzen im Ereignis: %d/%d"
                 % (info["patient"], e["n_kanal"], b["r"], d["auc"], N_SPITZEN,
                    d["spitzen_treffer"].sum(), len(d["spitzen_treffer"])), fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=6, loc="upper left")
    ax.set_xlabel("Zeit (Fenstermitte) [s]")
    ax.set_xlim(0, e["t"][-1] + L / 2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ziel, dpi=130)
    plt.close(fig)


def abbildung_uebersicht(alle, liste, L, schritt, ziel):
    fig, achsen = plt.subplots(len(alle), 1, figsize=(18, 2.8 * len(alle)), squeeze=False)
    fig.suptitle("Erlangen, λ1 über die Zeit — Fenster %.3g s, Schritt %.3g s; orange: Absence, "
                 "grau: Ereignis ohne Endzeit" % (L, schritt), fontsize=11)
    for ax, (name, e) in zip(achsen[:, 0], alle.items()):
        stil_achse(ax)
        anfaelle_zeichnen(ax, e["anfaelle"])
        for a, z in liste[name]["sperrzonen"]:
            ax.axvspan(a, z, color=MUTED, alpha=0.15, linewidth=0)
        ax.semilogy(e["t"], e["lam"][:, 0], color=DATEN, linewidth=0.8)
        ax.set_title("%s (%s) — r = %.2f, AUC = %.2f" % (name, liste[name]["patient"],
                     e["bewertung"]["r"], e["detektion"]["auc"]), fontsize=10)
        ax.set_ylabel("λ1")
        ax.set_xlim(0, e["t"][-1] + L / 2)
    achsen[-1, 0].set_xlabel("Zeit (Fenstermitte) [s]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ziel, dpi=120)
    plt.close(fig)


def tuh_vergleich(ausgabe, L, schritt):
    """Dieselben Detektionsmaße für die TUH-Zeitreihen (falls vorhanden)."""
    pkl = ausgabe / ("tuh_resampelt_zeitreihe_%gs.pkl" % L)
    if not pkl.exists():
        print("Hinweis: kein TUH-Pickle (%s)" % pkl.name)
        return
    z = pickle.load(open(pkl, "rb"))["zeitreihen"]
    print("\nTUH zum Vergleich (%s):" % pkl.name)
    print("%-20s %-4s %8s %6s %6s %8s" % ("Datei", "Satz", "Anfälle", "r", "AUC", "Spitzen"))
    for name, erg in z.items():
        for satz, e in erg.items():
            d = detektion(e, L, schritt)
            print("%-20s %-4s %8d %6.2f %6.2f %5d/%d" % (name, satz, len(e["anfaelle"]),
                  e["bewertung"]["r"], d["auc"], d["spitzen_treffer"].sum(), len(d["spitzen_treffer"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fenster-sek", type=float, default=3.0)
    ap.add_argument("--schritt-sek", type=float, default=0.5)
    ap.add_argument("--beste", type=int, default=4, help="nur die N besten Aufnahmen (nach r) speichern")
    args = ap.parse_args()
    L, schritt = args.fenster_sek, args.schritt_sek
    ausgabe = HIER / "ergebnisse"
    ausgabe.mkdir(exist_ok=True)

    liste = anfallsliste()
    alle = {}
    for name, info in liste.items():
        raw = lies(name, preload=True)
        picks = eeg_kanaele(raw)
        X = raw.get_data(picks=picks) * 1e6
        t, lam = zeitreihe(X, raw.info["sfreq"], L, schritt, len(picks))
        e = {"t": t, "lam": lam, "anfaelle": info["absencen"], "n_kanal": len(picks)}
        e["bewertung"] = bewertung(e, L, info["sperrzonen"])
        e["detektion"] = detektion(e, L, schritt, info["sperrzonen"])
        alle[name] = e
        print("fertig:", name)

    rang = sorted(alle, key=lambda n: -np.nan_to_num(alle[n]["bewertung"]["r"], nan=-9))
    print("\nErlangen, Übereinstimmung Absence – λ1 (sortiert nach r):")
    print("%-38s %-3s %5s | %6s %7s %6s %8s" % ("Aufnahme", "Pat", "Kan.", "r", "Treffer", "AUC", "Spitzen"))
    for n in rang:
        b, d = alle[n]["bewertung"], alle[n]["detektion"]
        print("%-38s %-3s %5d | %6.2f %6.0f%% %6.2f %5d/%d"
              % (n, liste[n]["patient"], alle[n]["n_kanal"], b["r"], 100 * b["treffer"], d["auc"],
                 d["spitzen_treffer"].sum(), len(d["spitzen_treffer"])))
    tuh_vergleich(ausgabe, L, schritt)

    beste = rang[:args.beste]
    print("\ngespeichert werden nur:", beste)
    for n in beste:
        abbildung(n, alle[n], liste[n], L, schritt, ausgabe / ("erlangen_zeitreihe_%gs_%s.png" % (L, n)))
    stamm = ausgabe / ("erlangen_zeitreihe_%gs" % L)
    abbildung_uebersicht({n: alle[n] for n in beste}, liste, L, schritt, str(stamm) + "_uebersicht.png")
    with open(str(stamm) + ".pkl", "wb") as fh:
        pickle.dump({"fenster_sek": L, "schritt_sek": schritt, "anfallsliste": liste,
                     "zeitreihen": alle, "rang": rang}, fh)
    print("gespeichert:", str(stamm) + "_*.png / .pkl")


if __name__ == "__main__":
    main()
