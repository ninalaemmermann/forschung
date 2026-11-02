"""
Wann ist SFA (Schritt 1+2) besser als DyCA, wann gleich, wann schlechter?

Prüfstand wie in rauschleiter_attraktoren.py: Rössler/Lorenz auf EEG-Zeitskala,
3-s-Fenster, 21 Kanäle, „Ruhe“ = Rauschen, „Anfall“ = Rauschen + Attraktor,
SNR global (Varianz Attraktor / Median der Ruhevarianz). Gemessen wird die AUC
von λ1 Ruhe gegen Anfall für beide Verfahren auf DENSELBEN Fenstern.

Zwei Rauscheigenschaften werden stufenlos verstellt - genau die, die in der
Rauschleiter den Unterschied gemacht haben:

    w   Anteil weißen, unabhängigen Kanalrauschens an der Rauschleistung;
        der Rest ist gaußsches Rauschen mit dem TUH-Kreuzspektrum (Stufe S4).
        w = 1: Annahme von Schritt 1 erfüllt; w = 0: EEG-artig
    k   Stärke der Schwankung von Fenster zu Fenster (Lautstärke und
        Spektralneigung), als Vielfaches der in der echten Ruhe gemessenen;
        k = 0: stationär, k = 1: wie EEG (Stufe S5). Wirkt nur auf den
        EEG-artigen Teil, das weiße Messrauschen bleibt stationär.

Drei Schnitte:
    A   w × SNR bei k = 0     reiner Effekt der Rauschfarbe
    B   k × SNR bei w = 0     reiner Effekt der Schwankung
    C   w × SNR bei k = 1     weißes Rauschen auf realistischem EEG

Urteil je Zelle über ein gepaartes Bootstrap von ΔAUC = AUC_SFA − AUC_DyCA
(95-%-Intervall): SFA besser, DyCA besser, gleich; „beide perfekt“, wenn
beide AUC ≥ 0,99.

Die Kalibrierung (Kreuzspektrum, Schwankung) kommt aus rauschleiter.pkl;
vorher einmal ``python rauschleiter_attraktoren.py`` laufen lassen.

    python sfa_dyca_vergleich.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import rauschleiter_attraktoren as rl
from eeg_signalraum import stil_achse, DATEN, SURROGAT, MUTED, INK_SEK
from parameter import SEED, SYSTEME

N = rl.N
N_FENSTER = 150                  # Ruhe- und Anfallsfenster je Zelle
SNR_STUFEN = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
W_STUFEN = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
K_STUFEN = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
N_BOOT = 500
PERFEKT = 0.99
SCHNITTE = {
    "A": dict(achse="w", stufen=W_STUFEN, fest=dict(k=0.0),
              titel="A  Weißanteil w, stationär (k = 0)"),
    "B": dict(achse="k", stufen=K_STUFEN, fest=dict(w=0.0),
              titel="B  Schwankung k, ohne Weißanteil (w = 0)"),
    "C": dict(achse="w", stufen=W_STUFEN, fest=dict(k=1.0),
              titel="C  Weißanteil w, Schwankung wie EEG (k = 1)"),
}
ACHSE = {"w": "Weißanteil w an der Rauschleistung", "k": "Schwankung k (1 = wie echte Ruhe)"}
URTEIL_TEXT = {"sfa": "SFA besser", "dyca": "DyCA besser", "gleich": "gleich",
               "perfekt": "beide perfekt"}


# ===========================================================================
# Rauschen und Fenster
# ===========================================================================

def rauschen(anzahl, w, k, kal, rng):
    """anzahl Fenster: √(1−w)·EEG-artig (Schwankung k) + √w·weiß, globale Normierung."""
    n = anzahl * N
    eeg = rl.kreuz_fenster(n, rng, kal["kreuz"], k * kal["streu_log_var"], k * kal["streu_alpha"])
    weiss = rl.s0(n, rng)
    E = np.sqrt(1 - w) * eeg / eeg.std() + np.sqrt(w) * weiss / weiss.std()
    return rl.fenster(E, anzahl)


def lam1_beide(Y):
    """(λ1 SFA, λ1 DyCA) oder None, wenn eines der Verfahren das Fenster nicht rechnet."""
    a, b = rl.lam(Y, "snr"), rl.lam(Y, "dyca")
    return None if a is None or b is None else (a[0], b[0])


def lam1_liste(fenster_liste):
    werte = [lam1_beide(Y) for Y in fenster_liste]
    return np.array([v for v in werte if v is not None])


# ===========================================================================
# Urteil
# ===========================================================================

def auc_schnell(ruhe, anfall):
    r = rankdata(np.r_[ruhe, anfall])
    return (r[len(ruhe):].sum() - len(anfall) * (len(anfall) + 1) / 2) / (len(ruhe) * len(anfall))


def vergleiche(ruhe, anfall, rng):
    """AUC beider Verfahren, ΔAUC mit gepaartem Bootstrap-Intervall und Urteil.

    ruhe, anfall: (n, 2) mit Spalten λ1 SFA, λ1 DyCA - dieselben Fenster für beide.
    """
    a_sfa, a_dyca = auc_schnell(ruhe[:, 0], anfall[:, 0]), auc_schnell(ruhe[:, 1], anfall[:, 1])
    deltas = []
    for _ in range(N_BOOT):
        r = ruhe[rng.integers(0, len(ruhe), len(ruhe))]
        a = anfall[rng.integers(0, len(anfall), len(anfall))]
        deltas.append(auc_schnell(r[:, 0], a[:, 0]) - auc_schnell(r[:, 1], a[:, 1]))
    lo, hi = np.percentile(deltas, (2.5, 97.5))
    if min(a_sfa, a_dyca) >= PERFEKT:
        urteil = "perfekt"
    elif lo > 0:
        urteil = "sfa"
    elif hi < 0:
        urteil = "dyca"
    else:
        urteil = "gleich"
    return {"auc_sfa": a_sfa, "auc_dyca": a_dyca, "delta": a_sfa - a_dyca,
            "ki_unten": lo, "ki_oben": hi, "urteil": urteil}


# ===========================================================================
# Schnitte
# ===========================================================================

def schnitt(name, spec, kal, attraktoren, zaehler):
    zeilen = []
    for wert in spec["stufen"]:
        p = dict(spec["fest"], **{spec["achse"]: wert})
        zaehler[0] += 1
        rng = np.random.default_rng(SEED + 1000 * zaehler[0])
        ruhe = lam1_liste(rauschen(N_FENSTER, p["w"], p["k"], kal, rng))
        rausch = rauschen(N_FENSTER, p["w"], p["k"], kal, rng)
        bezug = np.median([E.var(axis=0).mean() for E in rausch])
        for system, A in attraktoren.items():
            for snr in SNR_STUFEN:
                Ys = []
                for E, X in zip(rausch, A):
                    q = X @ rl.mischmatrix(E.shape[1])
                    q -= q.mean(axis=0)
                    Ys.append(E + np.sqrt(snr * bezug / q.var(axis=0).mean()) * q)
                erg = vergleiche(ruhe, lam1_liste(Ys), rng)
                zeilen.append(dict(schnitt=name, system=system, w=p["w"], k=p["k"], snr=snr,
                                   n_ruhe=len(ruhe), **erg))
        print("  %s  %s = %-6g fertig" % (name, spec["achse"], wert))
    return zeilen


# ===========================================================================
# Abbildung
# ===========================================================================

KARTE = LinearSegmentedColormap.from_list("dyca_sfa", [SURROGAT, "#f4f3ef", DATEN])
GRENZE = 0.5


def zelle_text(z):
    if z.urteil == "perfekt":
        return "1 | 1"
    zeichen = {"sfa": "S", "dyca": "D", "gleich": "="}[z.urteil]
    return "%+.2f\n%s" % (z.delta, zeichen)


def abbildung(df, ziel):
    systeme = list(SYSTEME)
    fig, ax = plt.subplots(len(systeme), len(SCHNITTE), figsize=(17, 5.2 * len(systeme)))
    fig.suptitle("Wann ist SFA besser als DyCA? ΔAUC = AUC SFA − AUC DyCA (λ1, Ruhe gegen Anfall)",
                 fontsize=13)
    for z, system in enumerate(systeme):
        for sp, (name, spec) in enumerate(SCHNITTE.items()):
            a = ax[z, sp]
            d = df[(df.schnitt == name) & (df.system == system)]
            M = d.pivot(index=spec["achse"], columns="snr", values="delta").loc[list(spec["stufen"])]
            bild = a.imshow(M.values, cmap=KARTE, vmin=-GRENZE, vmax=GRENZE, aspect="auto",
                            origin="lower")
            for i, y in enumerate(spec["stufen"]):
                for j, snr in enumerate(SNR_STUFEN):
                    zelle = d[(d[spec["achse"]] == y) & (d.snr == snr)].iloc[0]
                    stark = zelle.urteil in ("sfa", "dyca")
                    a.text(j, i, zelle_text(zelle), ha="center", va="center", fontsize=8,
                           color="#0b0b0b" if stark else INK_SEK,
                           fontweight="bold" if stark else "normal")
            a.set_xticks(range(len(SNR_STUFEN)))
            a.set_xticklabels(["%g" % s for s in SNR_STUFEN], fontsize=9)
            a.set_yticks(range(len(spec["stufen"])))
            a.set_yticklabels(["%g" % s for s in spec["stufen"]], fontsize=9)
            a.set_xlabel("SNR (Varianz Attraktor / Median der Ruhevarianz)", fontsize=9,
                         color=INK_SEK)
            a.set_ylabel(ACHSE[spec["achse"]], fontsize=9, color=INK_SEK)
            a.set_title("%s — %s" % (system, spec["titel"]), fontsize=10)
            for s in a.spines.values():
                s.set_visible(False)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.08, wspace=0.28, hspace=0.32)
    leiste = fig.colorbar(bild, cax=fig.add_axes([0.925, 0.25, 0.012, 0.5]))
    leiste.set_label("ΔAUC   (blau: SFA besser, orange: DyCA besser)", color=INK_SEK)
    fig.text(0.01, 0.01, "Zellen: ΔAUC und Urteil aus 95-%%-Bootstrap-Intervall — S: SFA besser, "
             "D: DyCA besser, =: kein Unterschied nachweisbar, 1 | 1: beide AUC ≥ %.2f. "
             "%d Ruhe- und %d Anfallsfenster je Zelle." % (PERFEKT, N_FENSTER, N_FENSTER),
             fontsize=9, color=INK_SEK)
    fig.savefig(ziel, dpi=130)
    plt.close(fig)


def main():
    pfad = HIER / "ergebnisse" / "rauschleiter.pkl"
    kal = pickle.load(open(pfad, "rb"))["kalibrierung"]
    attraktoren = {name: rl.attraktor_fenster(name)[:N_FENSTER] for name in SYSTEME}
    zeilen, zaehler = [], [0]
    for name, spec in SCHNITTE.items():
        print("Schnitt", spec["titel"])
        zeilen += schnitt(name, spec, kal, attraktoren, zaehler)
    df = pd.DataFrame(zeilen)

    aus = HIER / "ergebnisse"
    df.to_csv(aus / "sfa_dyca_vergleich.csv", index=False)
    abbildung(df, aus / "sfa_dyca_vergleich.png")
    pd.set_option("display.width", 200)
    for name, spec in SCHNITTE.items():
        print("\n%s" % spec["titel"])
        for system in SYSTEME:
            d = df[(df.schnitt == name) & (df.system == system)]
            print(system)
            print(d.pivot(index=spec["achse"], columns="snr", values="urteil")
                  .loc[list(spec["stufen"])].to_string())
    print("\ngespeichert:", aus / "sfa_dyca_vergleich.png / .csv")


if __name__ == "__main__":
    main()
