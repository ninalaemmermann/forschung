"""
Ruhe als Rauschen: generalisierte Eigenzerlegung wie bei Common Spatial
Patterns (CSP, Koles 1991; Tutorial: Cohen 2022).

Dasselbe Eigenwertproblem wie Schritt 2 der SDGL-Rekonstruktion,

    C·v = λ·R·v,

aber R ist die über viele Ruhefenster gemittelte Kovarianz, C die Kovarianz
eines einzelnen Fensters. λ ist dann das Varianzverhältnis Fenster / Ruhe je
Richtung: λ ≈ 1 heißt „wie in Ruhe“, λ ≫ 1 „hier ist etwas Neues dazugekommen“.

Warum so: im EEG gibt es kein weißes Messrauschen, gegen das man messen
könnte (pruefung_signalraum.py). Die Ruhe ist die naheliegende Referenz,
und weil R über viele Fenster gemittelt wird, reicht für C ein 3-s-Fenster.
"""
import numpy as np
from scipy.linalg import eigh

# dieselbe Schwelle wie in eeg_signalraum.py (Average-Referenz kostet eine Richtung)
RANG_SCHWELLE = 1e-8


def kovarianz(Y):
    """Kovarianz eines Fensters (n, dim), mittelwertfrei."""
    return np.cov(Y - Y.mean(axis=0), rowvar=False)


def ruhe_kovarianz(fenster):
    """Mittel der Fensterkovarianzen über eine Liste von Ruhefenstern.

    Gemittelt wird je Fenster mittelwertfrei - langsame Drifts zwischen den
    Fenstern landen so nicht in R.
    """
    return np.mean([kovarianz(Y) for Y in fenster], axis=0)


def kontrast(Y, R, k=3):
    """Schritt 2 mit der Ruhe als Rauschen.

    Projiziert vorher auf den Rang von R, sonst ist eigh(C, R) bei
    Average-Referenz nicht lösbar.

    Rückgabe wie ``signalraum``: Z (n, k), lam (r,) absteigend, V (dim, k)
    """
    ev, U = eigh(R)
    P = U[:, ev > RANG_SCHWELLE * ev[-1]]
    Y0 = Y - Y.mean(axis=0)
    lam, W = eigh(kovarianz(Y0 @ P), P.T @ R @ P)
    lam, W = lam[::-1], W[:, ::-1]
    V = P @ W[:, :k]
    return Y0 @ V, lam, V
