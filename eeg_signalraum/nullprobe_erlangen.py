"""
Nullprobe für „Ruhe als Rauschen“ (csp.py) auf den Erlangen-Daten.

Ruhefenster gegen eine aus ANDEREN Ruhefenstern gemittelte Ruhe-Kovarianz R.
Weil hier kein Anfall drin ist, zeigt die Streuung von λ, wie groß λ allein
durch Schätzfehler und Schwankungen der Ruhe wird. Das 95-%-Quantil von λ1
ist die Schwelle, ab der eine Richtung im Anfall als „neu“ zählt.

Zwei Aufteilungen je Patient:
    gemischt          Fenster abwechselnd in R und Nullprobe, über alle Aufnahmen
    gleiche Aufnahme  je Aufnahme: R aus jedem zweiten Ruhefenster DIESER
                      Aufnahme, getestet die übrigen - R pro Datei; Artefakte
                      zusätzlich je Aufnahme bereinigt
    andere Aufnahme   R aus allen übrigen Aufnahmen des Patienten - prüft,
                      ob die Ruhe über Minuten bis Tage stabil ist

Ruhefenster: 3 s, nicht überlappend, mindestens ABSTAND s von jeder Absence
und jeder Sperrzone (wie in erlangen.py).

Referenzlinien: Marchenko-Pastur-Grenzen (1 ± √(p/n))² - so weit streut λ
schon bei perfekt stationärem weißem Rauschen mit n = 768 Samples.
Autokorreliertes EEG hat weniger effektive Samples, also mehr Streuung.

Gerechnet wird auf den Rohdaten (ungefiltert, wie erlangen.py). Option:
    --bereinigt   Ruhefenster mit Gesamtvarianz > Median + 5·MAD verwerfen (Artefakte)

    python nullprobe_erlangen.py
    python nullprobe_erlangen.py --bereinigt
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import erlangen as er
from eeg_signalraum import frei, stil_achse, DATEN, SURROGAT, MUTED
from stil import INK_SEK
from csp import ruhe_kovarianz, kontrast, kovarianz

L = 3.0
ABSTAND = 30.0
AUFTEILUNGEN = ("gemischt", "gleiche Aufnahme", "andere Aufnahme")
FARBE_AUFT = {"gemischt": DATEN, "gleiche Aufnahme": INK_SEK, "andere Aufnahme": SURROGAT}
MAD_FAKTOR = 5.0


def ruhefenster(liste):
    """Je Patient: Liste von (Aufnahme, Startzeit s, Y in µV)."""
    aus = {}
    for lang, info in liste.items():
        raw = er.lies(lang, preload=True)
        picks = er.eeg_kanaele(raw)
        sr = raw.info["sfreq"]
        n = int(round(L * sr))
        daten = raw.get_data(picks=picks).T * 1e6
        sperr = info["absencen"] + info["sperrzonen"]
        for t in np.arange(0, info["dauer"] - L, L):
            if frei(t, t + L, sperr, ABSTAND):
                i0 = int(round(t * sr))
                aus.setdefault(info["patient"], []).append((lang, t, daten[i0:i0 + n]))
    return aus


def bereinige(fenster, faktor=MAD_FAKTOR):
    """Verwirft Fenster mit Gesamtvarianz über Median + faktor·MAD; gibt den Anteil zurück."""
    v = np.array([np.trace(kovarianz(f[2])) for f in fenster])
    m = np.median(v)
    gut = v < m + faktor * 1.4826 * np.median(np.abs(v - m))
    return [f for f, g in zip(fenster, gut) if g], 1 - gut.mean()


def nullprobe(fenster, aufteilung):
    """λ-Spektren (m, r) der Nullfenster und ihre Zeitpunkte."""
    lam, wo = [], []
    if aufteilung == "gleiche Aufnahme":
        for aufnahme in dict.fromkeys(f[0] for f in fenster):
            # Artefakte relativ zur EIGENEN Aufnahme bereinigen: in einer ruhigen
            # Aufnahme fallen sie gegenüber dem Patientenpool nicht auf
            eigene, _ = bereinige([f for f in fenster if f[0] == aufnahme])
            R = ruhe_kovarianz([f[2] for f in eigene[0::2]])
            for f in eigene[1::2]:
                lam.append(kontrast(f[2], R)[1])
                wo.append(f[:2])
    elif aufteilung == "gemischt":
        R = ruhe_kovarianz([f[2] for f in fenster[0::2]])
        for f in fenster[1::2]:
            lam.append(kontrast(f[2], R)[1])
            wo.append(f[:2])
    else:
        for aufnahme in dict.fromkeys(f[0] for f in fenster):
            R = ruhe_kovarianz([f[2] for f in fenster if f[0] != aufnahme])
            for f in fenster:
                if f[0] == aufnahme:
                    lam.append(kontrast(f[2], R)[1])
                    wo.append(f[:2])
    return np.array(lam), wo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bereinigt", action="store_true")
    args = ap.parse_args()
    fenster = ruhefenster(er.anfallsliste())
    verworfen = {}
    if args.bereinigt:
        for pat in fenster:
            fenster[pat], verworfen[pat] = bereinige(fenster[pat])
            print("%s: %.0f %% der Ruhefenster als Artefakt verworfen" % (pat, 100 * verworfen[pat]))
    zusatz = "roh, Artefaktfenster verworfen" if args.bereinigt else "roh"
    patienten = sorted(fenster)
    fig, achsen = plt.subplots(len(patienten), 2, figsize=(15, 4.8 * len(patienten)),
                               gridspec_kw={"width_ratios": [1, 1.4]})
    fig.suptitle("Nullprobe Erlangen — Ruhe gegen Ruhe, Fenster %g s, ≥ %g s von jeder Absence, %s"
                 % (L, ABSTAND, zusatz))
    ergebnis = {}
    print("%-4s %-16s %6s %6s %9s %10s %10s %12s"
          % ("Pat.", "Aufteilung", "n", "Kanäle", "λ1 Med.", "λ1 95 %", "λ_r 5 %", "Anteil λ1>10"))

    for zeile, pat in enumerate(patienten):
        F = fenster[pat]
        p, n = F[0][2].shape[1], F[0][2].shape[0]
        ax, ax_t = achsen[zeile]
        stil_achse(ax)
        stil_achse(ax_t)
        for auft in AUFTEILUNGEN:
            lam, wo = nullprobe(F, auft)
            x = np.arange(1, lam.shape[1] + 1)
            lo, med, hi = np.percentile(lam, [2.5, 50, 97.5], axis=0)
            schwelle = np.percentile(lam[:, 0], 95)
            ergebnis[(pat, auft)] = {"lam": lam, "wo": wo, "schwelle": schwelle}
            print("%-4s %-16s %6d %6d %9.2f %10.2f %10.3f %12.2f"
                  % (pat, auft, len(lam), p, med[0], schwelle,
                     np.percentile(lam[:, -1], 5), np.mean(lam[:, 0] > 10)))
            farbe = FARBE_AUFT[auft]
            ax.fill_between(x, lo, hi, color=farbe, alpha=0.18, linewidth=0)
            ax.semilogy(x, med, "o-", color=farbe, markersize=3,
                        label="%s (%d Fenster): Schwelle λ1 = %.1f" % (auft, len(lam), schwelle))
            ax.axhline(schwelle, color=farbe, linestyle=":", linewidth=1.0)

            # λ1 über die Zeit, Aufnahmen hintereinander
            aufnahmen = list(dict.fromkeys(f[0] for f in F))
            versatz = {a: i * 600 for i, a in enumerate(aufnahmen)}
            tt = [versatz[a] + t for a, t in wo]
            ax_t.semilogy(tt, lam[:, 0], ".", color=farbe, markersize=4, alpha=0.7, label=auft)
        for i, a in enumerate(aufnahmen):
            ax_t.axvline(i * 600, color=MUTED, linewidth=0.8)
            ax_t.text(i * 600 + 10, 1.02, a.split("_")[0][-4:] + a.split("export")[1][:6],
                      fontsize=7, color=INK_SEK, transform=ax_t.get_xaxis_transform())

        mp_lo, mp_hi = (1 - np.sqrt(p / n)) ** 2, (1 + np.sqrt(p / n)) ** 2
        for wert in (mp_lo, mp_hi):
            ax.axhline(wert, color="k", linestyle="--", linewidth=0.8)
        ax.axhline(1.0, color=MUTED, linewidth=0.8)
        ax.set_title("%s — λ je Richtung, Median und 95 %% (gestrichelt: Marchenko-Pastur, "
                     "weißes Rauschen)" % pat, fontsize=9.5)
        ax.set_xlabel("Richtung")
        ax.set_ylabel("λ = Varianz Fenster / Ruhe")
        ax.legend(fontsize=8, frameon=False)
        ax_t.set_title("%s — λ1 je Ruhefenster über die Zeit (Aufnahmen aneinandergereiht, "
                       "je 600 s Raster)" % pat, fontsize=9.5)
        ax_t.set_xlabel("Zeit [s], je Aufnahme ab Rasterlinie")
        ax_t.set_ylabel("λ1")
        ax_t.legend(fontsize=8, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ziel = HIER / "ergebnisse" / ("nullprobe_erlangen" + ("_bereinigt" if args.bereinigt else ""))
    fig.savefig(str(ziel) + ".png", dpi=130)
    with open(str(ziel) + ".pkl", "wb") as fh:
        pickle.dump({"fenster_sek": L, "abstand": ABSTAND, "bereinigt": args.bereinigt, "verworfen": verworfen,
                     "ergebnis": {k: {"schwelle": v["schwelle"], "lam": v["lam"], "wo": v["wo"]}
                                  for k, v in ergebnis.items()}}, fh)
    print("\ngespeichert:", str(ziel) + ".png / .pkl")


if __name__ == "__main__":
    main()
