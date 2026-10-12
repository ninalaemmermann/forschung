"""
Spatio-Spectral Decomposition (SSD) nach Nikulin, Nolte & Curio (2011).

Dasselbe Eigenwertproblem wie Schritt 2 der SDGL-Rekonstruktion,

    C·v = λ·R·v,

nur mit einer anderen Definition von Rauschen:
    C   Kovarianz im Signalband um die Rhythmusfrequenz f0
    R   Kovarianz in den Flanken links und rechts davon

R aus zweiten Differenzen setzt weißes Messrauschen voraus; im EEG misst es
den hochfrequenten Rand des Gehirnsignals (siehe pruefung_signalraum.py).
SSD vergleicht dagegen nur benachbarte Frequenzen - bei einem glatten
Hintergrundspektrum (1/f) liegen Band und Flanken fast gleich hoch.

λ wird mit dem Leistungsgewinn der Filter normiert: weißes Rauschen gibt
λ ≈ 1, wie bei Schritt 1+2. Dann ist λ - 1 das SNR im Band gegen die Flanken.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt, sosfreqz

# Richtungen unterhalb dieses Anteils am größten Eigenwert gelten als tot
# (Average-Referenz); dieselbe Schwelle wie in eeg_signalraum.py.
RANG_SCHWELLE = 1e-8


def baender(f0, signal=(0.8, 1.2), flanke=(0.6, 1.4), sperre=(0.7, 1.3)):
    """Bandgrenzen relativ zu f0: Signalband, äußere Flankengrenzen, Lücke.

    Schema nach Nikulin: das Rauschband ist ein Bandpass über ``flanke``,
    aus dem ``sperre`` herausgeschnitten wird - übrig bleiben zwei Flanken,
    getrennt vom Signalband durch eine kleine Lücke.
    Bei f0 = 3 Hz: Signal 2,4–3,6 Hz, Flanken 1,8–2,1 und 3,9–4,2 Hz.
    """
    return {k: (f0 * a, f0 * b) for k, (a, b) in
            (("signal", signal), ("flanke", flanke), ("sperre", sperre))}


def filter_sos(fs, f0, ordnung=2, **kw):
    """Butterworth-SOS für Signalband, Flanken-Bandpass und Sperre."""
    b = baender(f0, **kw)
    return {"signal": butter(ordnung, b["signal"], "bandpass", fs=fs, output="sos"),
            "flanke": butter(ordnung, b["flanke"], "bandpass", fs=fs, output="sos"),
            "sperre": butter(ordnung, b["sperre"], "bandstop", fs=fs, output="sos")}


def leistungsgewinn(sos_liste, n=2 ** 16):
    """Anteil weißen Rauschens, der eine Kette von sosfiltfilt passiert.

    sosfiltfilt filtert vor- und rückwärts, also wirkt |H|² zweimal: |H|⁴.
    Der Mittelwert über 0 … Nyquist ist der Varianzanteil.
    """
    g = np.ones(n)
    for sos in sos_liste:
        g *= np.abs(sosfreqz(sos, worN=n)[1]) ** 4
    return float(g.mean())


def ssd(Y, fs, f0, k=3, ordnung=2, **kw):
    """SSD auf Y (n, dim) mit Rhythmusfrequenz f0.

    Rückgabe wie ``signalraum``: Z (n, k) Projektion des UNGEFILTERTEN Y,
    lam (r,) absteigend und auf weißes Rauschen = 1 normiert, V (dim, k).
    r ist der Rang der Daten (Average-Referenz kostet eine Richtung).
    """
    Y0 = Y - Y.mean(axis=0)
    ev, U = eigh(np.cov(Y0, rowvar=False))
    P = U[:, ev > RANG_SCHWELLE * ev[-1]]           # (dim, r)
    Yp = Y0 @ P

    sos = filter_sos(fs, f0, ordnung, **kw)
    S = sosfiltfilt(sos["signal"], Yp, axis=0)
    N = sosfiltfilt(sos["sperre"], sosfiltfilt(sos["flanke"], Yp, axis=0), axis=0)
    C = np.cov(S, rowvar=False)
    R = np.cov(N, rowvar=False)

    lam, W = eigh(C, R)                              # aufsteigend
    lam, W = lam[::-1], W[:, ::-1]
    norm = leistungsgewinn([sos["signal"]]) / leistungsgewinn([sos["flanke"], sos["sperre"]])
    V = P @ W[:, :k]
    return Y0 @ V, lam / norm, V
