"""
Tests für hankel_dmd: Abgleich mit PyDMD auf rauschfreien Sinuswellen. Nach jeder
Änderung laufen lassen:

    python test_hankel_dmd.py

Verglichen wird nur rauschfrei: PyDMD schneidet die zweite SVD auf Q[:-1] statt auf
ganz Q ab. Bei exaktem Rang ist das gleich, bei Rauschen kann es leicht abweichen.
"""
import warnings

import numpy as np
from pydmd import DMD, HODMD
from scipy.optimize import linear_sum_assignment

from hankel_dmd import hankel_dmd

# PyDMD warnt bei Rang-6-Daten in 10 Kanälen vor schlechter Kondition; das ist hier gewollt
warnings.filterwarnings("ignore", category=UserWarning, module="pydmd")

SEED = 0
FS = 256
FREQ = np.array([3.0, 7.0, 11.0])
EPS = 1e-8
t = np.arange(768) / FS


def signal_a():
    """Rang 6: je Frequenz Sinus und Kosinus mit eigenem Mischvektor auf 10 Kanälen."""
    rng = np.random.default_rng(SEED)
    A, B = rng.standard_normal((3, 10)), rng.standard_normal((3, 10))
    return sum(np.outer(np.sin(2 * np.pi * f * t), a) + np.outer(np.cos(2 * np.pi * f * t), b)
               for f, a, b in zip(FREQ, A, B))


def signal_b():
    """Ein Kanal mit denselben drei Sinuswellen."""
    return sum(np.sin(2 * np.pi * f * t) for f in FREQ)[:, None]


def abstand(mu, nu):
    """Größte Abweichung, wenn jeder Eigenwert dem nächstgelegenen der anderen Liste
    zugeordnet wird."""
    assert len(mu) == len(nu), (len(mu), len(nu))
    kosten = np.abs(mu[:, None] - nu[None, :])
    i, j = linear_sum_assignment(kosten)
    return kosten[i, j].max()


def frequenzen_richtig(f):
    return np.abs(np.sort(f) - np.sort(np.concatenate([-FREQ, FREQ]))).max()


def test_signal_a_rang6():
    assert hankel_dmd(signal_a(), d=1, eps=EPS)["N"] == 6


def test_dmd_gleich_pydmd():
    Y = signal_a()
    mu = hankel_dmd(Y, d=1, eps=EPS)["eigs"]
    nu = DMD(svd_rank=6, exact=True).fit(Y.T).eigs
    abw = abstand(mu, nu)
    assert abw < 1e-8, abw


def test_hodmd_gleich_pydmd():
    Y = signal_a()
    aus = hankel_dmd(Y, d=5, eps=EPS, dt=1 / FS)
    nu = HODMD(svd_rank=aus["N_strich"], svd_rank_extra=aus["N"], d=5, exact=True).fit(Y.T).eigs
    abw = abstand(aus["eigs"], nu)
    assert abw < 1e-8, abw
    abw = frequenzen_richtig(aus["frequenzen"])
    assert abw < 1e-6, abw


def test_ein_kanal_braucht_lag():
    y = signal_b()
    assert len(hankel_dmd(y, d=1, eps=EPS)["eigs"]) == 1
    aus = hankel_dmd(y, d=8, eps=EPS, dt=1 / FS)
    abw = frequenzen_richtig(aus["frequenzen"])
    assert abw < 1e-6, abw
    nu = HODMD(svd_rank=aus["N_strich"], svd_rank_extra=1, d=8, exact=True).fit(y.T).eigs
    abw = abstand(aus["eigs"], nu)
    assert abw < 1e-8, abw


def test_tau2_gleiche_frequenzen():
    """PyDMD kann kein tau: Selbsttest, tau = 2 liefert dieselben Frequenzen wie tau = 1."""
    y = signal_b()
    f1 = hankel_dmd(y, d=8, tau=1, eps=EPS, dt=1 / FS)["frequenzen"]
    f2 = hankel_dmd(y, d=8, tau=2, eps=EPS, dt=1 / FS)["frequenzen"]
    abw = np.abs(np.sort(f1) - np.sort(f2)).max()
    assert abw < 1e-6, abw


if __name__ == "__main__":
    for test in (test_signal_a_rang6, test_dmd_gleich_pydmd, test_hodmd_gleich_pydmd,
                 test_ein_kanal_braucht_lag, test_tau2_gleiche_frequenzen):
        test()
        print("ok  ", test.__name__)
