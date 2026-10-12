"""
Gegenprobe: rechnet ``spektrum(Y, "dyca")`` auf Rössler und Lorenz dasselbe wie
Versuch h in Rauschanalyse_EEG/Signalrekonstruktion/versuche/h_dyca?

Daten genau wie auf Folie 12 (``abbildungen.beobachtung``, Mischmatrix aus
Seed 0, Fälle GUT und SCHLECHT). Verglichen wird gegen einen direkten Aufruf
der Bibliothek wie in run_h.py und gegen die Zahlen aus dessen README. Bei
Lorenz weicht schon run_h.py selbst auf dieser Maschine von der README ab
(chaotische Bahn), die README-Spalte ist dort nur ein Anhaltspunkt.

    python pruefung_dyca.py
"""
import sys
from pathlib import Path

import numpy as np
from dyca import dyca

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import eeg_signalraum as es                       # setzt die Pfade nach stdgl/
from parameter import SEED, DIM, DT, GUT, SCHLECHT, SYSTEME
from abbildungen import beobachtung

# λ1 / λ2 aus der README von Versuch h (Tabelle 1)
README = {("Rössler", GUT): (0.073, 0.053), ("Rössler", SCHLECHT): (0.011, 0.010),
          ("Lorenz", GUT): (0.151, 0.023), ("Lorenz", SCHLECHT): (0.013, 0.010)}


def bibliothek(Y):
    """DyCA wie in run_h.py: mittelwertfrei, Zeitachse in DT, Ableitung der Bibliothek."""
    return dyca(Y - Y.mean(axis=0), time_index=DT * np.arange(len(Y)))["generalized_eigenvalues"]


def main():
    es.N_KANAL = DIM
    W = np.random.default_rng(SEED).standard_normal((3, DIM))    # wie in main.py
    alles_gleich = True
    for name in SYSTEME:
        for fall in (GUT, SCHLECHT):
            X, Y = beobachtung(name, fall, W=W)
            lam, r = es.spektrum(Y, "dyca")
            ref = bibliothek(Y)
            abw = np.abs(lam[:r] - ref[:r]).max()
            alles_gleich &= abw < 1e-8
            print("%-8s Stufe 1 = %-3g Stufe 2 = %-4g  spektrum λ1..3 %s | Bibliothek %s | "
                  "Abw. %.0e | README %.3g / %.3g | Orakel %s"
                  % ((name, fall[0], fall[1], np.round(lam[:3], 3), np.round(ref[:3], 3), abw)
                     + README[(name, fall)] + (np.round(bibliothek(X), 3),)))
    print("spektrum = Bibliothek in allen Fällen:", "ja" if alles_gleich else "NEIN")


if __name__ == "__main__":
    main()
