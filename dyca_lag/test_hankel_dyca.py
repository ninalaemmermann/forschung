"""
Regressionstests für hankel_dyca. Nach jeder Änderung laufen lassen:

    python test_hankel_dyca.py
"""
import numpy as np
from dyca import dyca

from hankel_dyca import hankel_dyca, ableitung

SEED = 0


def rauschen():
    return np.random.default_rng(SEED).standard_normal((5000, 3))


def test_d1_gleich_dyca():
    """d = 1, eps = 0: dieselben λ wie DyCA direkt (mit derselben Ableitung)."""
    Y = np.cumsum(np.random.default_rng(SEED).standard_normal((768, 20)), axis=0)
    Yc = Y - Y.mean(axis=0)
    ref = dyca(Yc, derivative_signal=ableitung(Yc))["generalized_eigenvalues"]
    lam = hankel_dyca(Y, d=1, eps=0)["generalized_eigenvalues"]
    abw = np.abs(lam - ref).max()
    assert abw < 1e-10, abw


def test_tau1_triviale_einsen():
    """Weißes Rauschen, tau = 1, d = 4: genau (d-2)·N Eigenwerte = 1."""
    d = 4
    aus = hankel_dyca(rauschen(), d=d, tau=1, eps=0)
    anzahl = int(np.sum(aus["generalized_eigenvalues"] > 1 - 1e-8))
    assert anzahl == (d - 2) * aus["N"], anzahl


def test_tau2_keine_einsen():
    """Gleiche Daten, tau = 2: kein Eigenwert über 0,1."""
    lam = hankel_dyca(rauschen(), d=4, tau=2, eps=0)["generalized_eigenvalues"]
    assert lam.max() < 0.1, lam.max()


if __name__ == "__main__":
    for test in (test_d1_gleich_dyca, test_tau1_triviale_einsen, test_tau2_keine_einsen):
        test()
        print("ok  ", test.__name__)
