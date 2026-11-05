"""
Hankel-DyCA: DyCA auf zeitverzögert gestapelten Daten, aufgebaut wie HODMD (DMD-d).

Vorlage für die Schritte: Le Clainche & Vega (2017), SIAM J. Appl. Dyn. Syst. 16(2),
in der Darstellung von Corrochano et al. (2022), arXiv 2203.11574, Abschnitt 2:
  1. SVD der Daten, N Richtungen mit σ_{N+1}/σ_1 ≤ ε             (Gl. 7)
  2. d aufeinanderfolgende reduzierte Snapshots stapeln            (Gl. 11)
  3. zweite SVD auf der gestapelten Matrix, N' mit σ̃_{N'+1}/σ̃_1 < ε (Gl. 12-13)
Statt der Koopman-Matrix wird am Ende DyCA gelöst. Dafür kommen zwei Schritte dazu,
die es bei HODMD nicht gibt: Mittelwert abziehen und eine Ableitung bilden. Die
Ableitung wird einmal auf den reduzierten Daten gebildet und dann genauso gestapelt
und projiziert wie das Signal.

    from hankel_dyca import hankel_dyca
    aus = hankel_dyca(Y, d=4)          # Y: (Zeit, Kanäle)
    aus["generalized_eigenvalues"]
"""
import numpy as np
from dyca import dyca


def rang(s, eps):
    """Anzahl der Singulärwerte mit s_i / s_1 > eps (HODMD-Kriterium)."""
    return int(np.sum(s / s[0] > eps))


def ableitung(q, dt=1.0):
    """Zentrale Differenz, an den Rändern einseitig."""
    dq = np.empty_like(q)
    dq[1:-1] = (q[2:] - q[:-2]) / (2 * dt)
    dq[0] = (q[1] - q[0]) / dt
    dq[-1] = (q[-1] - q[-2]) / dt
    return dq


def stapeln(q, d, tau=1):
    """(T, r) -> (T-(d-1)·tau, r·d), Zeile k = [q_k, q_{k+tau}, ..., q_{k+(d-1)tau}]."""
    T = len(q) - (d - 1) * tau
    if T <= 0:
        raise ValueError("Fenster zu kurz für d = %d, tau = %d" % (d, tau))
    return np.hstack([q[i * tau:i * tau + T] for i in range(d)])


def hankel_dyca(Y, d, tau=1, eps=1e-3, dt=1.0):
    """DyCA-Eigenwerte der zeitverzögert gestapelten Daten.

    Y    -- Signal (Zeit, Kanäle)
    d    -- Zahl der gestapelten Zeitpunkte; d = 1 ist normales DyCA
    tau  -- Abstand der Zeitpunkte in Samples; HODMD nimmt 1. Achtung: Bei tau = 1 und
            d ≥ 3 liegt die zentrale Differenz der inneren Blöcke exakt im Raum der
            gestapelten Daten -> (d-2)·N Eigenwerte λ = 1 schon bei reinem Rauschen.
            Ab tau = 2 ist das bei reinem Rauschen weg. Bei glatten Signalen bleiben
            Eigenwerte nahe 1, solange tau·dt klein gegen die Korrelationszeit ist.
    eps  -- Toleranz für beide SVD-Schritte; 0 schneidet nichts ab
    dt   -- Abtastschritt (λ hängt nicht davon ab)

    Bei d = 1 und eps = 0 kommt dasselbe heraus wie bei dyca(Y): Die Projektion ist dann
    nur eine Drehung, und DyCA-λ ändern sich unter invertierbaren linearen Abbildungen
    nicht. Die SVD-Schritte wirken also nur über das Abschneiden.
    """
    Y = Y - Y.mean(axis=0)
    _, s, Vt = np.linalg.svd(Y, full_matrices=False)
    N = rang(s, eps)
    q = Y @ Vt[:N].T                                   # reduzierte Snapshots Y·V_N = U_N·Σ_N

    Q = stapeln(q, d, tau)
    dQ = stapeln(ableitung(q, dt), d, tau)

    _, s2, Vt2 = np.linalg.svd(Q, full_matrices=False)
    N_strich = rang(s2, eps)
    if len(Q) < 5 * N_strich:
        raise ValueError("zu wenige Zeitpunkte: %d für %d Dimensionen nach dem Stapeln"
                         % (len(Q), N_strich))
    P = Vt2[:N_strich].T
    Qp, dQp = Q @ P, dQ @ P

    erg = dyca(Qp, derivative_signal=dQp)
    return {**erg,
            "V_N": Vt[:N].T,          # Kanäle -> reduzierter Raum
            "P": P,                   # gestapelter Raum -> projizierter Raum
            "s": s, "s2": s2,         # Singulärwerte der ersten und zweiten SVD
            "N": N, "N_strich": N_strich,
            "cond_C0": np.linalg.cond(Qp.T @ Qp / len(Qp))}