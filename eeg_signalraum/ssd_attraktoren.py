"""
SSD gegen Schritt 1+2 auf Rössler und Lorenz - Originalaufbau, guter Fall.

Daten wie in ``abbildung_schritte`` (stdgl/Signalrekonstruktion): Bahn mit
Prozessrauschen GUT[0]·S_PROZESS, in DIM Kanäle gemischt, weißes
Sensorrauschen mit SNR GUT[1] und Spreizung SPREIZUNG.

f0 ist die Spitze des Spektrums der SAUBEREN Bahn - entspricht „Rhythmus
bekannt“, wie 3 Hz bei Absencen.

    python ssd_attraktoren.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent / "stdgl" / "Signalrekonstruktion"))
sys.path.insert(0, str(HIER.parent / "stdgl" / "werkzeuge"))
sys.path.insert(0, str(HIER))
from funktionen import rauschmatrix, signalraum, signalrichtungen, beobachte_bahn, kosten
from parameter import bahn, DT, DIM, K, GUT, SPREIZUNG, SEED, N_BAHN, SYSTEME
from stil import stil_achse, DATEN, SURROGAT, MUTED, INK_SEK
from ssd import ssd, baender

FS = 1 / DT
# Freiheitsgrade je Kanal im schmalsten Band (den Flanken). Ein Band der
# Breite B über die Dauer T hat etwa 2·B·T unabhängige Werte; für eine
# brauchbare (dim × dim)-Kovarianz braucht es ein Vielfaches von dim.
FG_JE_KANAL = 10


def freiheitsgrade(n, f0, dim=DIM):
    """Effektive Freiheitsgrade der Flanken-Kovarianz, je Kanal."""
    b = baender(f0)
    breite = (b["sperre"][0] - b["flanke"][0]) + (b["flanke"][1] - b["sperre"][1])
    return 2 * breite * n * DT / dim


def laenge(f0, dim=DIM):
    """Samples, damit die Flanken FG_JE_KANAL Freiheitsgrade je Kanal haben."""
    return int(np.ceil(FG_JE_KANAL / freiheitsgrade(1, f0, dim)))


def rhythmus(X):
    """Spitzenfrequenz des Spektrums der sauberen Bahn (Koordinaten standardisiert)."""
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    f, P = periodogram(Xs, fs=FS, axis=0)
    P = np.convolve(P.sum(axis=1), np.ones(5) / 5, mode="same")   # leicht glätten
    ok = f > 0.05
    return f[ok][np.argmax(P[ok])], f, P


def probe_weiss(f0=0.15):
    """Normierung: SSD auf reines weißes Rauschen muss λ ≈ 1 geben."""
    for n in (N_BAHN, laenge(f0)):
        Y = np.random.default_rng(1).standard_normal((n, DIM))
        lam = ssd(Y, FS, f0, K)[1]
        print("Probe weißes Rauschen, n = %6d (%.1f FG je Kanal): SSD-λ Median %.3f, "
              "Spanne %.3f … %.3f" % (n, freiheitsgrade(n, f0), np.median(lam),
                                      lam.min(), lam.max()))


def main():
    probe_weiss()
    W = np.random.default_rng(SEED).standard_normal((3, DIM))
    fig, achsen = plt.subplots(len(SYSTEME), 3, figsize=(15, 4.4 * len(SYSTEME)))
    fig.suptitle("SSD gegen Schritt 1+2 — Originalaufbau, Stufe 1 = %g, Stufe 2 = %g, "
                 "Spreizung %g" % (GUT + (SPREIZUNG,)))

    for zeile, name in enumerate(SYSTEME):
        f0 = rhythmus(bahn(name, N_BAHN, GUT[0], SEED))[0]
        n = max(N_BAHN, laenge(f0))
        X = bahn(name, n, GUT[0], SEED)
        Y = beobachte_bahn(X, GUT[1], DIM, np.random.default_rng(SEED), W, SPREIZUNG)
        f0, f, P = rhythmus(X)
        print("%-8s n = %d Samples, %.1f FG je Kanal in den Flanken (bei N_BAHN: %.1f)"
              % (name, n, freiheitsgrade(n, f0), freiheitsgrade(N_BAHN, f0)))

        if zeile == 0:
            print("\n%-8s %7s  %-12s %3s %8s   λ1 … λ5" % ("System", "f0", "Verfahren", "K", "J"))
        R, _ = rauschmatrix(Y)
        Z_s, lam_s, _ = signalraum(Y, R, K)
        Z_d, lam_d, _ = ssd(Y, FS, f0, K)

        for titel, Z, lam, ax, farbe in (("Schritt 1+2", Z_s, lam_s, achsen[zeile][0], DATEN),
                                         ("SSD", Z_d, lam_d, achsen[zeile][1], SURROGAT)):
            k_est, J = signalrichtungen(lam), kosten(X, Z)
            print("%-8s %7.3f  %-12s %3d %8.3f   %s"
                  % (name, f0, titel, k_est, J, np.array2string(lam[:5], precision=2)))
            stil_achse(ax)
            ax.semilogy(np.arange(1, len(lam) + 1), lam, "o-", color=farbe, markersize=4,
                        label="K = %d, J = 1 − R² = %.3f" % (k_est, J))
            ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
            ax.text(len(lam) * 0.45, 1.0, " λ = 1: reines Rauschen", fontsize=8,
                    color=INK_SEK, va="bottom")
            ax.axvline(K + 0.5, color=MUTED, linestyle=":", linewidth=1.0)
            ax.set_title("%s — %s" % (name, titel + (" (λ = 1 + SNR, R aus 2. Differenz)"
                         if titel != "SSD" else " (λ = Band / Flanken)")), fontsize=10)
            ax.set_xlabel("Richtung")
            ax.set_ylabel("λ")
            ax.legend(fontsize=8, frameon=False)

        # Spektrum der sauberen Bahn mit den Bändern
        ax = achsen[zeile][2]
        stil_achse(ax)
        b = baender(f0)
        ax.axvspan(*b["signal"], color=SURROGAT, alpha=0.25, label="Signalband")
        ax.axvspan(b["flanke"][0], b["sperre"][0], color=DATEN, alpha=0.25, label="Flanken")
        ax.axvspan(b["sperre"][1], b["flanke"][1], color=DATEN, alpha=0.25)
        ax.semilogy(f[1:], P[1:], color=INK_SEK, linewidth=0.8)
        ax.set_xlim(0, 4 * f0)
        ax.set_title("%s — Spektrum der sauberen Bahn, f0 = %.3f, n = %d" % (name, f0, n), fontsize=10)
        ax.set_xlabel("f [1 / Zeiteinheit]")
        ax.set_ylabel("Leistung (Summe x, y, z standardisiert)")
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    ziel = HIER / "ergebnisse" / "ssd_attraktoren.png"
    fig.savefig(ziel, dpi=140)
    print("\ngespeichert:", ziel)


if __name__ == "__main__":
    main()
