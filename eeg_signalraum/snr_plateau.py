"""
SNR-Plateau als Anfallsmerkmal – TUH-ABSZ, resampelt ungefiltert (16 Aufnahmen).

Merkmal (``plateau`` in ``tuh_resampelt_zeitreihe.py``): Mittel von log10 λ über die gewählten
Richtungen, minus gleitender Median über 60 s (eigene Grundlinie), davon das gleitende Minimum
über w s. Hoch ist es nur, wenn λ die ganzen w s über der Grundlinie lag - kurze
Artefaktspitzen fallen weg.

Varianten: λ1, λ2 oder Mittel λ1–3; w = 3/5/10 s; mit und ohne Grundlinie. Zum Vergleich
SNR roh (λ1) und DyCA (aus ``dyca_zeitreihe_3s.pkl``) mit demselben Detektor.

Detektor: Alarm, wenn das Merkmal mindestens ``DAUER`` s über der Schwelle θ liegt (Lücken
< ``LUECKE`` s werden geschlossen). θ = 99-%-Quantil der anfallsfreien Merkmalswerte der
**anderen** Aufnahmen (leave-one-out), nie aus der Testaufnahme selbst.

Kennzahlen:
    AUC        Anfallsfenster (≥ 50 % Anfall) gegen anfallsfreie Fenster, Median über Aufnahmen
    Top-N      verschiedene Anfälle unter den N höchsten Spitzen (N = Anfälle der Aufnahme)
    erkannt    Anfälle, die ein Alarm überlappt
    FA/h       Alarme ohne Anfall pro Stunde
    Verz.      Median (erster Alarm im Anfall − Anfallsbeginn); negativ möglich, weil Fenster
               und Filter zentriert sind

    python snr_plateau.py              # rechnet SNR für alle 16 einmal und speichert sie
    python snr_plateau.py --neu        # SNR neu rechnen statt aus dem Pickle
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
from eeg_signalraum import stil_achse, MUTED, DATEN, SURROGAT, INK_SEK
from tuh_resampelt import neue_dateien
from tuh_resampelt_zeitreihe import anfaelle_zeichnen, anfallsanteil, gemeinsame, lade, plateau, zeitreihe

L, SCHRITT = 3.0, 0.5
DAUER, LUECKE, QUANTIL = 3.0, 2.0, 0.99
PLOT = ("aaaaabqv_s002_t001", "aaaaadqe_s001_t001", "aaaaacyi_s001_t001", "aaaaamtc_s001_t000")
SPALTEN = {"λ1": (0,), "λ2": (1,), "λ1–3": (0, 1, 2)}


def varianten():
    """Name -> (Methode, Spalten, w, Grundlinie)."""
    v = {"SNR roh λ1": ("snr", (0,), 0.5, 0)}
    for sp, spalten in SPALTEN.items():
        for w in (3, 5, 10):
            for basis in (60, 0):
                v["SNR %s, w %g s%s" % (sp, w, ", rel." if basis else "")] = ("snr", spalten, w, basis)
    v["DyCA roh λ1"] = ("dyca", (0,), 0.5, 0)
    v["DyCA λ1–3, w 3 s, rel."] = ("dyca", (0, 1, 2), 3, 60)
    return v


def merkmal(e, methode, spalten, w, basis):
    return plateau(e["lam"], SCHRITT, spalten, w, basis, log=(methode == "snr"))[0]


def alarme(t, x, theta):
    """Zusammenhängende Abschnitte mit x > θ (Lücken < LUECKE geschlossen, Dauer ≥ DAUER)."""
    ueber = x > theta
    abschnitte, start = [], None
    for i, u in enumerate(ueber):
        if u and start is None:
            start = i
        if not u and start is not None:
            abschnitte.append([t[start], t[i - 1]]); start = None
    if start is not None:
        abschnitte.append([t[start], t[-1]])
    zus = []
    for a, b in abschnitte:
        if zus and a - zus[-1][1] < LUECKE:
            zus[-1][1] = b
        else:
            zus.append([a, b])
    return [(a, b) for a, b in zus if b - a + SCHRITT >= DAUER]


def bewerte(e, x, theta):
    t, anf = e["t"], e["anfaelle"]
    ant = anfallsanteil(t, L, anf)
    im, frei = ant >= 0.5, ant == 0
    auc = mannwhitneyu(x[im], x[frei]).statistic / (im.sum() * frei.sum()) if im.any() and frei.any() else np.nan
    idx, _ = find_peaks(x, distance=int(5 / SCHRITT))
    idx = idx[np.argsort(x[idx])[::-1][:len(anf)]]
    topn = len({k for i in idx for k, (a, b) in enumerate(anf) if a - L / 2 <= t[i] <= b + L / 2})
    al = alarme(t, x, theta)
    trifft = lambda a, b, s, z: a <= z + L / 2 and b >= s - L / 2
    erkannt = [any(trifft(a, b, s, z) for a, b in al) for s, z in anf]
    verz = [min(a for a, b in al if trifft(a, b, s, z)) - s for s, z in anf if any(trifft(a, b, s, z) for a, b in al)]
    fa = sum(not any(trifft(a, b, s, z) for s, z in anf) for a, b in al)
    return {"auc": auc, "topn": topn, "n": len(anf), "erkannt": sum(erkannt), "fa": fa,
            "stunden": (t[-1] + L / 2) / 3600, "verz": verz, "alarme": al, "theta": theta}


def auswerten(daten, methode, spalten, w, basis):
    werte = {n: merkmal(e, methode, spalten, w, basis) for n, e in daten.items()}
    frei = {n: werte[n][anfallsanteil(e["t"], L, e["anfaelle"]) == 0] for n, e in daten.items()}
    aus = {}
    for n, e in daten.items():
        andere = [m for m in daten if m != n]
        assert n not in andere
        theta = np.quantile(np.concatenate([frei[m] for m in andere]), QUANTIL)
        aus[n] = bewerte(e, werte[n], theta)
        aus[n]["x"] = werte[n]
    return aus


def snr_zeitreihen(neu_rechnen):
    pkl = HIER / "ergebnisse" / ("snr_plateau_%gs.pkl" % L)
    if pkl.exists() and not neu_rechnen:
        return pickle.load(open(pkl, "rb"))["zeitreihen"]
    pfade = neue_dateien("absz")
    daten = {}
    for name in gemeinsame(pfade, "absz"):
        X, sr, anfaelle = lade("neu", name, pfade[name], "absz")
        t, lam = zeitreihe(X, sr, L, SCHRITT, X.shape[0])
        daten[name] = {"t": t, "lam": lam, "anfaelle": anfaelle}
        print("fertig:", name, flush=True)
    with open(pkl, "wb") as fh:
        pickle.dump({"fenster_sek": L, "schritt_sek": SCHRITT, "zeitreihen": daten}, fh)
    return daten


def abbildung(daten, aus, name_var, var, ziel):
    methode, spalten, w, basis = var
    fig, achsen = plt.subplots(2 * len(PLOT), 1, figsize=(18, 3.0 * len(PLOT)),
                               gridspec_kw={"height_ratios": [1.3, 1] * len(PLOT)})
    fig.suptitle("TUH-ABSZ ungefiltert — %s (Fenster %g s, Schritt %g s); oben log λ und Grundlinie, "
                 "unten Plateau-Merkmal, Schwelle θ (leave-one-out) und Alarme; orange: Anfälle"
                 % (name_var, L, SCHRITT), fontsize=11)
    for k, name in enumerate(PLOT):
        e, r = daten[name], aus[name]
        _, x, g = plateau(e["lam"], SCHRITT, spalten, w, basis or 60)
        ax, axp = achsen[2 * k], achsen[2 * k + 1]
        for a_ in (ax, axp):
            stil_achse(a_)
            anfaelle_zeichnen(a_, e["anfaelle"], beschriften=(k == 0 and a_ is ax))
            a_.set_xlim(0, e["t"][-1] + L / 2)
        ax.plot(e["t"], x, color=MUTED, linewidth=0.7, label=("log10 %s" if len(spalten) == 1 else "Mittel log10 %s")
                % name_var.split(",")[0][4:] if k == 0 else None)
        ax.plot(e["t"], g, color=INK_SEK, linewidth=1.4, label="Grundlinie (Median 60 s)" if k == 0 else None)
        ax.set_ylabel("log10 λ")
        ax.set_title("%s — %d Anfälle, erkannt %d, Fehlalarme %d, AUC %.2f"
                     % (name, r["n"], r["erkannt"], r["fa"], r["auc"]), fontsize=10)
        ax.tick_params(labelbottom=False)
        axp.plot(e["t"], r["x"], color=DATEN, linewidth=0.9, label="Plateau" if k == 0 else None)
        axp.axhline(r["theta"], color=SURROGAT, linestyle="--", linewidth=1,
                    label="Schwelle θ" if k == 0 else None)
        unten = np.nanmin(r["x"])
        for j, (a, b) in enumerate(r["alarme"]):
            axp.plot([a, b], [unten, unten], color=INK_SEK, linewidth=5, solid_capstyle="butt",
                     label="Alarm" if k == 0 and j == 0 else None)
        axp.set_ylabel("Plateau [Dek.]")
        if k == 0:
            ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
            axp.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
    achsen[-1].set_xlabel("Zeit (Fenstermitte) [s]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ziel, dpi=120)
    plt.close(fig)


def abbildung_tabelle(tab, beste, ziel):
    """Alle Varianten: erkannte Anfälle und Fehlalarme pro Stunde, zwei Felder, gleiche Reihenfolge."""
    t = tab.sort_values(["erkannt", "FA/h"], ascending=[True, False]).reset_index(drop=True)
    farbe = [SURROGAT if v.startswith("DyCA") else DATEN if v == beste else MUTED for v in t.Variante]
    y = np.arange(len(t))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 0.36 * len(t) + 1.6), sharey=True,
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle("TUH-ABSZ ungefiltert, 16 Aufnahmen, 61 Anfälle — Detektor: ≥ %g s über θ "
                 "(%g-%%-Quantil der anfallsfreien Zeit, leave-one-out); blau: beste SNR-Variante, "
                 "orange: DyCA" % (DAUER, 100 * QUANTIL), fontsize=11)
    for ax, spalte, fmt, titel in ((ax1, "erkannt", "{:.0%}", "erkannte Anfälle"),
                                   (ax2, "FA/h", "{:.1f}", "Fehlalarme pro Stunde")):
        stil_achse(ax)
        ax.grid(False, axis="y")
        ax.barh(y, t[spalte], height=0.7, color=farbe, edgecolor="white", linewidth=1)
        for yi, v in zip(y, t[spalte]):
            ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8, color=INK_SEK)
        ax.set_title(titel, fontsize=10)
        ax.set_xlim(0, t[spalte].max() * 1.15)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_yticks(y)
    ax1.set_yticklabels(t.Variante, fontsize=9)
    for lbl, v in zip(ax1.get_yticklabels(), t.Variante):
        lbl.set_fontweight("bold" if v == beste else "normal")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ziel, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neu", action="store_true", help="SNR-Zeitreihen neu rechnen")
    args = ap.parse_args()
    ausgabe = HIER / "ergebnisse"
    snr = snr_zeitreihen(args.neu)
    dy = pickle.load(open(ausgabe / ("dyca_zeitreihe_%gs.pkl" % L), "rb"))["datensaetze"]["tuh"]["zeitreihen"]
    assert set(dy) == set(snr), "DyCA und SNR haben verschiedene Aufnahmen"

    zeilen, ergebnisse = [], {}
    for name_var, var in varianten().items():
        daten = snr if var[0] == "snr" else dy
        aus = auswerten(daten, *var)
        ergebnisse[name_var] = aus
        r = list(aus.values())
        n = sum(x["n"] for x in r)
        verz = [v for x in r for v in x["verz"]]
        zeilen.append({"Variante": name_var, "AUC": np.nanmedian([x["auc"] for x in r]),
                       "Top-N": sum(x["topn"] for x in r) / n, "erkannt": sum(x["erkannt"] for x in r) / n,
                       "FA/h": sum(x["fa"] for x in r) / sum(x["stunden"] for x in r),
                       "Verz. [s]": np.median(verz) if verz else np.nan})
    tab = pd.DataFrame(zeilen)
    n_anf = sum(len(e["anfaelle"]) for e in snr.values())
    print("\nTUH-ABSZ ungefiltert, %d Aufnahmen, %d Anfälle, %.1f h; Detektor: ≥ %g s über θ (%g-%%-Quantil, leave-one-out)"
          % (len(snr), n_anf, sum(e["t"][-1] for e in snr.values()) / 3600, DAUER, 100 * QUANTIL))
    print(tab.to_string(index=False, formatters={"AUC": "{:.2f}".format, "Top-N": "{:.0%}".format,
                                                 "erkannt": "{:.0%}".format, "FA/h": "{:.1f}".format,
                                                 "Verz. [s]": "{:+.1f}".format}))
    tab.to_csv(ausgabe / "snr_plateau.csv", index=False)

    snr_tab = tab[tab.Variante.str.startswith("SNR") & (tab.Variante != "SNR roh λ1")]
    beste = snr_tab.sort_values(["erkannt", "FA/h"], ascending=[False, True]).iloc[0].Variante
    print("\nbeste SNR-Plateau-Variante (meiste erkannt, dann wenigste FA/h):", beste)
    ziel = ausgabe / ("snr_plateau_%gs_absz.png" % L)
    abbildung(snr, ergebnisse[beste], beste, varianten()[beste], ziel)
    abbildung_tabelle(tab, beste, ausgabe / ("snr_plateau_%gs_varianten.png" % L))
    print("gespeichert:", ziel, "und", ausgabe / "snr_plateau.csv")


if __name__ == "__main__":
    main()
