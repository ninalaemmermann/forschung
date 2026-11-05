"""
HODMD (DMD-d) mit derselben Vorverarbeitung wie hankel_dyca: erste SVD, d Zeitpunkte
stapeln, zweite SVD (rang, stapeln aus hankel_dyca). Nur der letzte Schritt ist anders:
statt DyCA die Koopman-Matrix nach Le Clainche & Vega (2017), in der Darstellung von
Corrochano et al. (2022), arXiv 2203.11574, Gl. 14-16.
Mit d = 1 ist das normales (projiziertes) DMD. Zusammen mit hankel_dyca ergibt das das
2×2-Design {DyCA, DMD} × {d = 1, d > 1}.

    from hankel_dmd import hankel_dmd
    aus = hankel_dmd(Y, d=5, dt=1/256)  # Y: (Zeit, Kanäle)
    aus["frequenzen"]
"""
import numpy as np

from hankel_dyca import rang, stapeln


def hankel_dmd(Y, d, tau=1, eps=1e-3, dt=1.0, zentrieren=False):
    """DMD-Eigenwerte, Frequenzen und Moden der zeitverzögert gestapelten Daten.

    Y, d, tau, eps, dt -- wie bei hankel_dyca
    zentrieren -- False wie in PyDMD (ein Mittelwert ist dann ein eigener Mode mit
                  μ = 1). Für den Methodenvergleich mit DyCA immer True nehmen, sonst
                  unterscheidet sich die Vorverarbeitung wieder; False ist nur für
                  den Abgleich mit PyDMD gedacht.

    Ein Zeitschritt der Koopman-Matrix ist immer 1 Sample, auch bei tau > 1: Die Zeilen
    der gestapelten Matrix sind um 1 Sample gegeneinander verschoben.
    Die Moden sind projizierte Moden (V_N · erster Block von P · Eigenvektor), nicht die
    exakten Moden, die PyDMD mit exact=True liefert. Vergleiche mit PyDMD daher nur über
    Eigenwerte und Frequenzen.
    """
    if zentrieren:
        Y = Y - Y.mean(axis=0)
    _, s, Vt = np.linalg.svd(Y, full_matrices=False)
    N = rang(s, eps)
    q = Y @ Vt[:N].T                                   # reduzierte Snapshots Y·V_N = U_N·Σ_N

    Q = stapeln(q, d, tau)
    _, s2, Vt2 = np.linalg.svd(Q, full_matrices=False)
    N_strich = rang(s2, eps)
    P = Vt2[:N_strich].T
    Z = (Q @ P).T                                      # Snapshots als Spalten (N', Zeit)

    R = Z[:, 1:] @ np.linalg.pinv(Z[:, :-1])           # Koopman-Matrix, Gl. 16
    mu, W = np.linalg.eig(R)
    return {"eigs": mu,
            "frequenzen": np.angle(mu) / (2 * np.pi * dt),
            "wachstum": np.log(np.abs(mu)) / dt,
            "moden": Vt[:N].T @ P[:N] @ W,            # zurück in den Kanalraum
            "V_N": Vt[:N].T, "P": P,
            "s": s, "s2": s2,
            "N": N, "N_strich": N_strich}
