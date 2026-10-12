"""
Ruhe als Rauschen (CSP-Ansatz) gegen Schritt 1+2 auf Rössler und Lorenz.

    „Ruhe“    nur Rauschen
    „Anfall“  dasselbe Rauschen + der Attraktor, in DIM Kanäle gemischt

Je Anfallsfenster (N_FENSTER = 768 Samples wie 3 s bei 256 Hz) zwei Wege zu R:
    Schritt 1+2          R aus den zweiten Differenzen des Fensters
    Ruhe als Rauschen    R = Mittel über N_RUHE Ruhefenster

Graue Linie: Nullprobe - weitere Ruhefenster (nicht in R) gegen R, 97,5-%-
Quantil je Richtung. So groß wird λ ohne jedes Signal; nur was darüber liegt,
ist Signal. Bei 1/f² nötig, weil die Ruhe selbst stark schwankt.

Zwei Rauscharten:
    weiß     unabhängig je Kanal, Spreizung wie im Original
    1/f²     1/f²-Rauschen je Quelle, über eine Zufallsmatrix gemischt, dazu
             eine allen Kanälen gemeinsame Komponente - EEG-artig
             (pruefung_signalraum.py)

    python csp_attraktoren.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent / "stdgl" / "Signalrekonstruktion"))
sys.path.insert(0, str(HIER.parent / "stdgl" / "werkzeuge"))
sys.path.insert(0, str(HIER))
from funktionen import rauschmatrix, signalraum
from parameter import bahn, DIM, K, GUT, SPREIZUNG, SEED, SYSTEME
from stil import stil_achse, DATEN, SURROGAT, MUTED, INK_SEK
from csp import ruhe_kovarianz, kontrast

N_FENSTER = 768        # wie 3 s bei 256 Hz
N_RUHE = 30            # Fenster für R
N_NULL = 30            # weitere Ruhefenster für die Nullprobe
N_ANFALL = 20          # Fenster mit Signal
ALPHA, GEMEINSAM = 2.0, 0.3   # 1/f^ALPHA, Anteil der allen Kanälen gemeinsamen Quelle
SNR = GUT[1]           # Signal- zu mittlerer Rauschleistung, wie im guten Fall
RAUSCHARTEN = {"weiß": "weißes Rauschen (wie im Original)",
               "1/f²": "1/f²-Rauschen, Kanäle korreliert (EEG-artig)"}


def rauschen_weiss(n, rng):
    """Unabhängig je Kanal, Lautstärke gespreizt wie in ``add_vector_noise``."""
    sigma = SPREIZUNG ** (rng.random(DIM) - 0.5)
    return rng.standard_normal((n, DIM)) * sigma


def rauschen_1f2(n, rng):
    """DIM Quellen 1/f^ALPHA, über eine feste Zufallsmatrix gemischt,
    plus eine allen Kanälen gemeinsame Quelle."""
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    quellen = np.fft.irfft(np.fft.rfft(rng.standard_normal((n, DIM + 1)), axis=0)
                           * (f ** (-ALPHA / 2))[:, None], n=n, axis=0)
    farbig = quellen[:, :DIM] @ rng.standard_normal((DIM, DIM))
    farbig /= farbig.std()
    gem = quellen[:, DIM:] / quellen[:, DIM:].std()
    return np.sqrt(1 - GEMEINSAM) * farbig + np.sqrt(GEMEINSAM) * gem


def fenster(A, anzahl, start=0):
    return [A[start + i * N_FENSTER:start + (i + 1) * N_FENSTER] for i in range(anzahl)]


def lauf(name, rauschart, W):
    """λ-Spektren (N_ANFALL, DIM) für beide Wege zu R und die der Nullprobe."""
    rng = np.random.default_rng(SEED + 1)
    n = (N_RUHE + N_NULL + N_ANFALL) * N_FENSTER
    E = rauschen_weiss(n, rng) if rauschart == "weiß" else rauschen_1f2(n, rng)

    X = bahn(name, N_ANFALL * N_FENSTER, GUT[0], SEED)
    q = X @ W
    q -= q.mean(axis=0)
    # SNR über die Varianz INNERHALB der Fenster: bei 1/f² sitzt fast die ganze
    # Leistung unterhalb der Fensterfrequenz und fiele sonst beim Mittelwertabzug weg.
    leistung = lambda A: np.mean([w.var(axis=0).mean() for w in fenster(A, len(A) // N_FENSTER)])
    E *= np.sqrt(leistung(q) / (SNR * leistung(E)))

    R_ruhe = ruhe_kovarianz(fenster(E, N_RUHE))
    null = fenster(E, N_NULL, N_RUHE * N_FENSTER)
    anfall = [s + w @ W for s, w in zip(fenster(E, N_ANFALL, (N_RUHE + N_NULL) * N_FENSTER),
                                         fenster(X, N_ANFALL))]
    return {"sdgl": np.array([signalraum(Y, rauschmatrix(Y)[0], K)[1] for Y in anfall]),
            "ruhe": np.array([kontrast(Y, R_ruhe, K)[1] for Y in anfall]),
            "null": np.array([kontrast(Y, R_ruhe, K)[1] for Y in null])}


def main():
    W = np.random.default_rng(SEED).standard_normal((3, DIM))
    fig, achsen = plt.subplots(len(SYSTEME), len(RAUSCHARTEN),
                               figsize=(12, 4.4 * len(SYSTEME)), sharey="row")
    fig.suptitle("Schritt 1+2 gegen Ruhe als Rauschen — Median über %d Fenster à %d Samples"
                 % (N_ANFALL, N_FENSTER), fontsize=11)
    x = np.arange(1, DIM + 1)
    print("%-8s %-5s | %-18s %s" % ("System", "Rausch", "Verfahren", "λ1 … λ5 (Median)"))

    for zeile, name in enumerate(SYSTEME):
        for spalte, art in enumerate(RAUSCHARTEN):
            erg = lauf(name, art, W)
            ax = achsen[zeile][spalte]
            stil_achse(ax)
            for schl, farbe, titel in (("sdgl", DATEN, "Schritt 1+2 (R aus 2. Differenz)"),
                                       ("ruhe", SURROGAT, "Ruhe als Rauschen (R aus Ruhe)")):
                med = np.median(erg[schl], axis=0)
                ax.semilogy(x, med, "o-", color=farbe, markersize=4, label=titel)
                print("%-8s %-5s | %-18s %s" % (name, art, titel[:18],
                                                np.array2string(med[:5], precision=1)))
            ax.semilogy(x, np.percentile(erg["null"], 97.5, axis=0), "-", color=MUTED,
                        linewidth=2.0, label="Ruhe gegen Ruhe (97,5 %): Grenze für Orange")
            ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.0)
            ax.axvline(K + 0.5, color=MUTED, linestyle=":", linewidth=1.0)
            ax.text(K + 0.7, 0.98, "3 Zustandsvariablen", fontsize=8, color=INK_SEK,
                    va="top", transform=ax.get_xaxis_transform())
            ax.set_title("%s — %s" % (name, RAUSCHARTEN[art]), fontsize=10)
            ax.set_xlabel("Richtung")
            ax.set_ylabel("λ")
            if zeile == 0 and spalte == 0:
                ax.legend(fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ziel = HIER / "ergebnisse" / "csp_attraktoren.png"
    fig.savefig(ziel, dpi=140)
    print("\ngespeichert:", ziel)


if __name__ == "__main__":
    main()
