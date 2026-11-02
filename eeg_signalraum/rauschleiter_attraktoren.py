"""
Rauschleiter: Wie muss das Simulationsrauschen aussehen, damit Rössler/Lorenz
sich wie EEG verhalten - und warum gewinnt DyCA im EEG gegen Schritt 1+2 (SFA)?

Format wie im EEG: fs = 256 Hz, 3-s-Fenster (768 Samples). Der Attraktor wird so
unterabgetastet, dass seine Grundfrequenz bei ≈ 3 Hz liegt (Absence).

    „Ruhe“    nur Rauschen
    „Anfall“  Rauschen + Attraktor, über eine feste Zufallsmatrix in die Kanäle
              gemischt; SNR = Varianz Attraktor / Varianz Rauschen IM FENSTER

Rauschleiter - jede Stufe fügt eine EEG-Eigenschaft hinzu:
    S0  weiß, unabhängig je Kanal (wie add_vector_noise, Spreizung 10)
    S1  S0 + Filter wie TUH-WB (Hochpass 0,5 Hz, Tiefpass 42 Hz, Notch 60 Hz)
    S2  1/f^α je Kanal, unabhängig
    S3  1/f^α aus vielen Quellen, räumlich gemischt + gemeinsame Quelle
    S4  gaußsches Rauschen mit der gemessenen Kreuzspektralmatrix S(f) der
        echten TUHZ-ABSZ-Ruhe (21 Kanäle): je Frequenz X(f) = L(f)·ξ(f),
        L·Lᴴ = S. Trifft Spektrum, räumliche Korrelation und - anders als
        S3 - auch, dass verschiedene Richtungen verschiedene Spektren haben
    S5  S4 + Schwankung von Fenster zu Fenster wie im EEG: Lautstärke und
        Spektralneigung je Fenster zufällig, Streuung aus der echten Ruhe
    S6  echte Ruhefenster als Rauschen (Hybrid): Erlangen roh, TUHZ-ABSZ
        resampelt ungefiltert - Rohdaten, Ruhe ≥ 30 s von jedem Anfall,
        Artefaktfenster verworfen (Median + 5·MAD)

Kalibrierung: α, Anteil der gemeinsamen Quelle, Kreuzspektrum und die
Streuung von Lautstärke und α je Fenster kommen aus der echten Ruhe (S6).

Warum das Spektrum reicht: SFA und DyCA benutzen nur Kovarianzen (von Y, den
Differenzen bzw. der Ableitung), also nur das Kreuzspektrum. Gaußsches
Rauschen mit dem richtigen Kreuzspektrum ist für beide Verfahren im Mittel
nicht von echtem EEG zu unterscheiden; übrig bleibt die Schwankung zwischen
den Fenstern (S5).

SNR global: Varianz des Attraktors / Median der Rauschvarianz aller Fenster
derselben Kanalzahl. Lautere Ruhefenster haben damit ein kleineres SNR - wie
im EEG, wo die Absence nicht mit dem Hintergrund mitskaliert.

Zielpunkte (Stern): AUC von λ1 Ruhe gegen Anfall auf den echten Fenstern
(tuh_resampelt.py, erlangen.py) beim gemessenen SNR = Varianz Anfall /
Varianz Ruhe − 1 (Median je Aufnahme).

Je Stufe: Kennzahlen der Ruhe (Spektrum, Spur R(lag 2)/R(lag 1), räumliche
Korrelation von R, λ-Spektren SFA und DyCA) und AUC von λ1 Ruhe gegen Anfall
über das SNR, für SFA und DyCA. Nullprobe: Ruhe gegen Ruhe (AUC ≈ 0,5).

    python rauschleiter_attraktoren.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, welch
from scipy.stats import rankdata

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import eeg_signalraum as es                     # setzt die Pfade nach stdgl/
from eeg_signalraum import frei, stil_achse, MUTED, SURROGAT, INK_SEK
from funktionen import rauschmatrix
from parameter import bahn, sauber, DT, GUT, SPREIZUNG, SEED, SYSTEME
import erlangen as er
import nullprobe_erlangen as nb
import tuh_resampelt as tr

FS = 256.0
L = 3.0
N = int(L * FS)                  # 768 Samples je Fenster
D = 21                           # Kanäle der synthetischen Stufen
ZIEL_HZ = 3.0                    # Grundfrequenz des Attraktors nach Unterabtastung
N_RUHE = 150                     # Ruhefenster je Stufe
N_ANFALL = 150                   # Anfallsfenster je Stufe, System und SNR
SNR_STUFEN = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
ABSTAND = 30.0                   # s Abstand der Ruhe zu jedem Anfall
TUH_MAX_JE_DATEI = 40            # Ruhefenster je TUH-Datei, gleichmäßig verteilt
N_QUELLEN = 3 * D                # Quellen hinter den Kanälen (S3, S4)
F_MIN = 0.1                      # Hz, unterhalb flach (sonst divergiert 1/f^α)
FIT_BAND, ALPHA_BAND = (2.0, 30.0), (7.0, 14.0)   # Steigungsfit ohne Alpha-Band
HAUPTKOMPONENTE = {"Rössler": 0, "Lorenz": 2}     # Komponente mit klarer Rotationsspitze

ECHT = ("S6 Erlangen", "S6 TUHZ-ABSZ")
STUFEN = ("S0", "S1", "S2", "S3", "S4", "S5") + ECHT
TEXT = {"S0": "S0 weiß",
        "S1": "S1 weiß + WB-Filter",
        "S2": "S2 1/f^α je Kanal",
        "S3": "S3 1/f^α gemischt",
        "S4": "S4 EEG-Kreuzspektrum",
        "S5": "S5 S4 + Schwankung je Fenster",
        "S6 Erlangen": "S6 echte Ruhe Erlangen",
        "S6 TUHZ-ABSZ": "S6 echte Ruhe TUHZ-ABSZ"}
# geordnete Leiter: eine Blau-Rampe hell -> dunkel; echte Ruhe: Orange
FARBE = dict(zip(STUFEN[:6], ("#b7d4f7", "#86b6f0", "#4f94e3", "#2a78d6", "#1a5aa8", "#0d3a73")))
FARBE.update({s: SURROGAT for s in ECHT})
MARKER = dict(zip(STUFEN, ("o", "s", "^", "D", "v", "h", "P", "X")))
LINIE = {s: "-" for s in STUFEN}
LINIE["S6 TUHZ-ABSZ"] = "--"
ZIEL = "#0b0b0b"                 # Zielpunkte echtes EEG


# ===========================================================================
# Echte Ruhe (S5)
# ===========================================================================

def ruhe_erlangen():
    """Bereinigte Ruhefenster beider Patienten (CA 25, ME 21 Kanäle), µV."""
    pool = nb.ruhefenster(er.anfallsliste())
    aus = []
    for pat in sorted(pool):
        gut, weg = nb.bereinige(pool[pat])
        print("Erlangen %s: %d Ruhefenster (%.0f %% verworfen)" % (pat, len(gut), 100 * weg))
        aus.extend(f[2] for f in gut)
    return aus


def ruhe_tuh():
    """Bereinigte Ruhefenster der resampelten, ungefilterten TUHZ-ABSZ-Aufnahmen, µV."""
    aus = []
    for name, edf in tr.neue_dateien("absz").items():
        anf = tr.anfaelle_bi(edf)
        raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
        sr = raw.info["sfreq"]
        if sr != FS:
            continue
        dauer = raw.n_times / sr
        kand = [t for t in np.arange(0, dauer - L, L) if frei(t, t + L, anf, ABSTAND)]
        if len(kand) < 2:
            continue
        wahl = [kand[i] for i in np.unique(np.linspace(0, len(kand) - 1,
                                                       min(TUH_MAX_JE_DATEI, len(kand))).astype(int))]
        daten = raw.get_data(picks=tr.KANAELE).T * 1e6
        pool = [(name, t, daten[int(round(t * sr)):int(round(t * sr)) + N]) for t in wahl]
        pool = [f for f in pool if len(f[2]) == N]
        gut, _ = nb.bereinige(pool)
        aus.extend(f[2] for f in gut)
    print("TUHZ-ABSZ: %d Ruhefenster" % len(aus))
    return aus


# ===========================================================================
# Synthetisches Rauschen (S0–S4), lange Reihe (n, D) - wird in Fenster geschnitten
# ===========================================================================

def potenzrauschen(n, k, alpha, rng):
    """k unabhängige Reihen mit Spektrum 1/f^alpha, Varianz 1."""
    f = np.fft.rfftfreq(n, 1 / FS)
    f = np.maximum(f, F_MIN)
    x = np.fft.irfft(np.fft.rfft(rng.standard_normal((n, k)), axis=0)
                     * (f ** (-alpha / 2))[:, None], n=n, axis=0)
    return x / x.std(axis=0)


def spreizung(rng):
    return SPREIZUNG ** (rng.random(D) - 0.5)


def s0(n, rng, **_):
    return rng.standard_normal((n, D)) * spreizung(rng)


def s1(n, rng, **_):
    sos_hp = butter(2, 0.5, "highpass", fs=FS, output="sos")
    sos_tp = butter(8, 42.0, "lowpass", fs=FS, output="sos")
    b, a = iirnotch(60.0, 30.0, fs=FS)
    x = sosfiltfilt(sos_tp, sosfiltfilt(sos_hp, s0(n, rng), axis=0), axis=0)
    return filtfilt(b, a, x, axis=0)


def s2(n, rng, alpha, **_):
    return potenzrauschen(n, D, alpha, rng) * spreizung(rng)


def kopf(rng):
    """Feste räumliche Abbildung: Mischmatrix der Quellen und Kanallautstärken."""
    return rng.standard_normal((N_QUELLEN, D)), spreizung(rng)


def mischen(q, gemeinsam, M, sigma):
    """N_QUELLEN Quellen über M in die Kanäle, plus die gemeinsame Quelle q[:, -1]."""
    gemischt = q[:, :N_QUELLEN] @ M
    gemischt /= gemischt.std(axis=0)
    return (np.sqrt(1 - gemeinsam) * gemischt + np.sqrt(gemeinsam) * q[:, N_QUELLEN:]) * sigma


def s3(n, rng, alpha, gemeinsam, **_):
    return mischen(potenzrauschen(n, N_QUELLEN + 1, alpha, rng), gemeinsam, *kopf(rng))


def kreuzspektrum(fenster_liste):
    """Mittlere Kreuzspektralmatrix S(f) (F, D, D) der Fenster, Hann-Fenster, Spur 1."""
    w = np.hanning(N)[:, None]
    S = 0
    for Y in fenster_liste:
        X = np.fft.rfft((Y - Y.mean(axis=0)) * w, axis=0)
        S = S + X[:, :, None] * X.conj()[:, None, :]
    S = S / len(fenster_liste)
    return np.fft.rfftfreq(N, 1 / FS), S / np.trace(S, axis1=1, axis2=2).real.sum()


def kreuz_faktor(kreuz, laenge, cache={}):
    """Cholesky-Faktor L(f) von S(f) auf dem FFT-Gitter der Länge ``laenge``.

    S wird zwischen den Stützstellen linear interpoliert (Konvexkombination
    positiv semidefiniter Matrizen bleibt positiv semidefinit); ein kleiner
    Grat macht S bei Average-Referenz-Aufnahmen invertierbar.
    """
    if laenge not in cache:
        f0, S0 = kreuz
        f = np.fft.rfftfreq(laenge, 1 / FS)
        i = np.clip(np.searchsorted(f0, f) - 1, 0, len(f0) - 2)
        t = ((f - f0[i]) / (f0[i + 1] - f0[i]))[:, None, None]
        S = (1 - t) * S0[i] + t * S0[i + 1]
        grat = 1e-6 * np.trace(S, axis1=1, axis2=2).real.mean() / S.shape[1]
        S = S + grat * np.eye(S.shape[1])
        cache[laenge] = (np.linalg.cholesky(S), np.maximum(f, F_MIN))
    return cache[laenge]


def kreuz_fenster(n, rng, kreuz, streu_log_var=0.0, streu_alpha=0.0):
    """Fenster mit Kreuzspektrum S(f), je Fenster aus einem 4-mal längeren Stück.

    Mit streu_* > 0 bekommt jedes Fenster eine eigene Lautstärke und eine
    eigene Neigung f^(−δ/2), δ ~ N(0, streu_alpha) - die Schwankung der Ruhe.
    """
    Lf, f = kreuz_faktor(kreuz, 4 * N)
    aus = []
    for _ in range(n // N):
        xi = (rng.standard_normal(Lf.shape[:2]) + 1j * rng.standard_normal(Lf.shape[:2])) / np.sqrt(2)
        X = np.einsum("fij,fj->fi", Lf, xi) * (f ** (-rng.normal(0, streu_alpha) / 2))[:, None]
        x = np.fft.irfft(X, n=4 * N, axis=0)[N + N // 2:2 * N + N // 2]
        aus.append(x / x.std() * np.exp(0.5 * rng.normal(0, streu_log_var)))
    return np.concatenate(aus)


def s4(n, rng, kreuz, **_):
    return kreuz_fenster(n, rng, kreuz)


def s5(n, rng, kreuz, streu_log_var, streu_alpha, **_):
    return kreuz_fenster(n, rng, kreuz, streu_log_var, streu_alpha)


ERZEUGER = {"S0": s0, "S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}


def fenster(A, anzahl):
    return [A[i * N:(i + 1) * N] for i in range(anzahl)]


# ===========================================================================
# Kennzahlen eines Fensters (wie pruefung_signalraum.kennzahlen, ergänzt)
# ===========================================================================

def kennzahlen(Y):
    Y = Y - Y.mean(axis=0)
    R1, R2 = rauschmatrix(Y, 1)[0], rauschmatrix(Y, 2)[0]
    d = np.sqrt(np.diag(R1))
    C = R1 / np.outer(d, d)
    f, P = welch(Y, fs=FS, nperseg=256, axis=0)
    P = P.mean(axis=1)
    band = (f >= FIT_BAND[0]) & (f <= FIT_BAND[1]) & ~((f >= ALPHA_BAND[0]) & (f <= ALPHA_BAND[1]))
    steigung = np.polyfit(np.log(f[band]), np.log(P[band]), 1)[0]
    return {"spur": np.trace(R2) / np.trace(R1),
            "r_raum": np.abs(C[~np.eye(len(C), dtype=bool)]).mean(),
            "alpha": -steigung, "f": f, "P": P / P.sum(), "var": Y.var(axis=0).mean()}


def auc(ruhe, anfall):
    """P(λ1 Anfall > λ1 Ruhe) über Ränge (Mann-Whitney)."""
    ruhe, anfall = np.asarray(ruhe), np.asarray(anfall)
    r = rankdata(np.r_[ruhe, anfall])
    return (r[len(ruhe):].sum() - len(anfall) * (len(anfall) + 1) / 2) / (len(ruhe) * len(anfall))


def lam(Y, methode):
    """λ-Spektrum (mit NaN aufgefüllt) oder None, wenn das Fenster nicht rechenbar ist."""
    return es.spektrum(Y, methode)[0]


# ===========================================================================
# Attraktor auf EEG-Zeitskala
# ===========================================================================

def unterabtastung(name):
    """Schrittzahl m, damit die Rotationsspitze bei ZIEL_HZ / FS liegt."""
    X = sauber(name, 2 ** 16)
    f, P = welch(X[:, HAUPTKOMPONENTE[name]] - X[:, HAUPTKOMPONENTE[name]].mean(),
                 nperseg=2 ** 13)
    f_spitze = f[1:][P[1:].argmax()]                   # Zyklen je Schritt DT
    m = max(1, int(round(ZIEL_HZ / FS / f_spitze)))
    print("%s: Spitze %.3f Zyklen je Zeiteinheit, Unterabtastung m = %d -> %.2f Hz"
          % (name, f_spitze / DT, m, f_spitze * m * FS))
    return m


def attraktor_fenster(name):
    m = unterabtastung(name)
    X = bahn(name, N_ANFALL * N * m, GUT[0], SEED)[::m]
    return fenster(X, N_ANFALL)


# ===========================================================================
# Kalibrierung gegen die echte Ruhe
# ===========================================================================

def robust_streu(v):
    v = np.asarray(v)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def kalibrieren(ruhe_echt):
    je_satz = {s: [kennzahlen(Y) for Y in satz] for s, satz in ruhe_echt.items()}
    k = [x for v in je_satz.values() for x in v]
    alpha = float(np.median([x["alpha"] for x in k]))
    ziel_r = float(np.median([x["r_raum"] for x in k]))
    # Kreuzspektrum nur aus TUHZ-ABSZ: dort haben alle Aufnahmen dieselben 21 Kanäle
    kreuz = kreuzspektrum(ruhe_echt["S6 TUHZ-ABSZ"])
    # Schwankung je Fenster, je Datensatz gemessen (Kanalzahl und Patienten getrennt
    # würden die Lautstärkestreuung sonst aufblähen), dann gemittelt
    streu_log_var = float(np.mean([robust_streu(np.log([x["var"] for x in v]))
                                   for v in je_satz.values()]))
    streu_alpha = float(np.mean([robust_streu([x["alpha"] for x in v]) for v in je_satz.values()]))
    rng = np.random.default_rng(SEED + 99)
    beste = None
    for g in np.linspace(0, 0.9, 10):
        E = s3(30 * N, rng, alpha=alpha, gemeinsam=g)
        r = np.median([kennzahlen(Y)["r_raum"] for Y in fenster(E, 30)])
        if beste is None or abs(r - ziel_r) < beste[1]:
            beste = (g, abs(r - ziel_r), r)
    print("Kalibrierung: α = %.2f (Median echte Ruhe), räumliche Korrelation Ziel %.2f, "
          "S3 trifft %.2f mit gemeinsamem Anteil %.1f" % (alpha, ziel_r, beste[2], beste[0]))
    print("Schwankung je Fenster: Streuung log-Varianz %.2f, Streuung α %.2f"
          % (streu_log_var, streu_alpha))
    return {"alpha": alpha, "gemeinsam": float(beste[0]), "ziel_r": ziel_r,
            "kreuz": kreuz, "streu_log_var": streu_log_var, "streu_alpha": streu_alpha}


# ===========================================================================
# Auswertung
# ===========================================================================

def rauschfenster(stufe, ruhe_echt, kal):
    """(Ruhefenster, Rauschfenster für die Anfälle) einer Stufe."""
    if stufe in ERZEUGER:
        rng = np.random.default_rng(SEED + 1 + STUFEN.index(stufe))
        E = ERZEUGER[stufe]((N_RUHE + N_ANFALL) * N, rng, **kal)
        alle = fenster(E, N_RUHE + N_ANFALL)
        return alle[:N_RUHE], alle[N_RUHE:]
    pool = ruhe_echt[stufe]
    wahl = np.random.default_rng(SEED).permutation(len(pool))[:N_RUHE + N_ANFALL]
    alle = [pool[i] for i in wahl]
    return alle[0::2], alle[1::2]


def mischmatrix(dim, cache={}):
    if dim not in cache:
        cache[dim] = np.random.default_rng(SEED).standard_normal((3, dim))
    return cache[dim]


def stufe_auswerten(stufe, ruhe_echt, kal, attraktoren):
    ruhe, rausch = rauschfenster(stufe, ruhe_echt, kal)
    aus = {"kennzahlen": [kennzahlen(Y) for Y in ruhe], "n_ruhe": len(ruhe)}
    for meth in ("snr", "dyca"):
        L_ruhe = [l for l in (lam(Y, meth) for Y in ruhe) if l is not None]
        aus[meth] = {"ruhe": np.array(L_ruhe)}
        h = len(L_ruhe) // 2
        aus[meth]["null_auc"] = auc([l[0] for l in L_ruhe[:h]], [l[0] for l in L_ruhe[h:]])
    # Bezugsvarianz je Kanalzahl (Erlangen: CA 25, ME 21 Kanäle)
    bezug = {}
    for E in rausch:
        bezug.setdefault(E.shape[1], []).append(E.var(axis=0).mean())
    bezug = {d: np.median(v) for d, v in bezug.items()}
    for name, A in attraktoren.items():
        for snr in SNR_STUFEN:
            Ys = []
            for E, X in zip(rausch, A):
                q = X @ mischmatrix(E.shape[1])
                q -= q.mean(axis=0)
                a = np.sqrt(snr * bezug[E.shape[1]] / q.var(axis=0).mean())
                Ys.append(E + a * q)
            for meth in ("snr", "dyca"):
                L_anf = np.array([l for l in (lam(Y, meth) for Y in Ys) if l is not None])
                aus[meth][(name, snr)] = {"anfall": L_anf,
                                          "auc": auc(aus[meth]["ruhe"][:, 0], L_anf[:, 0])}
    return aus


def tabelle(erg):
    print("\n=== Ruhe: Kennzahlen (Median) ===")
    print("%-30s %6s %8s %8s %7s %18s %18s" % ("Stufe", "n", "Spur2/1", "|r| R", "α",
                                                "SFA λ1..3", "DyCA λ1..3"))
    for s, e in erg.items():
        k = e["kennzahlen"]
        med = lambda key: np.median([x[key] for x in k])
        print("%-30s %6d %8.2f %8.2f %7.2f %18s %18s"
              % (TEXT[s], e["n_ruhe"], med("spur"), med("r_raum"), med("alpha"),
                 np.array2string(np.nanmedian(e["snr"]["ruhe"], axis=0)[:3], precision=1),
                 np.array2string(np.nanmedian(e["dyca"]["ruhe"], axis=0)[:3], precision=2)))
    print("\n=== AUC λ1 Ruhe gegen Anfall (Nullprobe Ruhe gegen Ruhe in Klammern) ===")
    for name in SYSTEME:
        print("\n%s" % name)
        print("%-30s %-6s %s" % ("Stufe", "Verf.", "  ".join("SNR %-5g" % s for s in SNR_STUFEN)))
        for s, e in erg.items():
            for meth, kurz in (("snr", "SFA"), ("dyca", "DyCA")):
                werte = "  ".join("%9.2f" % e[meth][(name, snr)]["auc"] for snr in SNR_STUFEN)
                print("%-30s %-6s %s   (%.2f)" % (TEXT[s], kurz, werte, e[meth]["null_auc"]))


# ===========================================================================
# Abbildungen
# ===========================================================================

def punkte(ax, erg, key, titel, log=False, linie=None, linie_text=None):
    stil_achse(ax)
    for i, (s, e) in enumerate(erg.items()):
        v = np.array([x[key] for x in e["kennzahlen"]])
        q1, m, q3 = np.percentile(v, (25, 50, 75))
        ax.plot([i, i], [q1, q3], color=FARBE[s], linewidth=2)
        ax.plot(i, m, MARKER[s], color=FARBE[s], markersize=8, markeredgecolor="white")
    if linie is not None:
        ax.axhline(linie, color=MUTED, linestyle="--", linewidth=1)
        ax.text(len(erg) - 0.6, linie, linie_text, fontsize=8, color=INK_SEK, va="bottom", ha="right")
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(erg)))
    ax.set_xticklabels([s.replace(" ", "\n", 1) for s in erg], fontsize=8)
    ax.set_title(titel, fontsize=10)


def abbildung_kennzahlen(erg, kal, ziel):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle("Ruhe je Stufe der Rauschleiter gegen echte EEG-Ruhe (Median, Balken 25–75 %)",
                 fontsize=13)
    a = ax[0, 0]
    stil_achse(a)
    for s, e in erg.items():
        f = e["kennzahlen"][0]["f"]
        P = np.median([x["P"] for x in e["kennzahlen"]], axis=0)
        a.loglog(f[1:], P[1:], LINIE[s], color=FARBE[s], linewidth=1.6)
    a.set_xlabel("Frequenz [Hz]")
    a.set_ylabel("Leistungsanteil je Hz")
    a.set_title("Spektrum (auf Gesamtleistung 1 normiert)", fontsize=10)

    punkte(ax[0, 1], erg, "alpha", "Spektralexponent α (Fit 2–30 Hz ohne 7–14 Hz)",
           linie=kal["alpha"], linie_text="kalibriert")
    punkte(ax[0, 2], erg, "spur", "Spur R(lag 2) / R(lag 1)", log=True,
           linie=1.0, linie_text="weißes Rauschen")
    punkte(ax[1, 0], erg, "r_raum", "räumliche Korrelation von R (mittleres |r|)",
           linie=kal["ziel_r"], linie_text="Ziel echte Ruhe")

    for a, meth, titel, log in ((ax[1, 1], "snr", "SFA: λ = 1 + SNR in Ruhe", True),
                                (ax[1, 2], "dyca", "DyCA-Eigenwerte in Ruhe", False)):
        stil_achse(a)
        for s, e in erg.items():
            med = np.nanmedian(e[meth]["ruhe"], axis=0)
            med = med[np.isfinite(med)]
            x = np.arange(1, len(med) + 1)
            (a.semilogy if log else a.plot)(x, med, LINIE[s], marker=MARKER[s], color=FARBE[s],
                                            markersize=5, linewidth=1.4)
        a.set_xlabel("Richtung")
        a.set_title(titel, fontsize=10)
        if not log:
            a.set_ylim(0, 1.02)
    griffe = [plt.Line2D([], [], color=FARBE[s], linestyle=LINIE[s], marker=MARKER[s],
                         markersize=7, label=TEXT[s]) for s in erg]
    fig.legend(handles=griffe, loc="lower center", ncol=len(erg), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(ziel, dpi=130)
    plt.close(fig)


def ziel_eeg():
    """Echte Fenster Ruhe/Anfall: SNR je Aufnahme, AUC und Median von λ1 je Verfahren.

    Liest die Ergebnisse von tuh_resampelt.py (Satz „neu_alle“) und erlangen.py.
    """
    erg_dir = HIER / "ergebnisse"
    quellen = {"TUHZ-ABSZ": ("tuh_resampelt_3s.pkl", "tuh_resampelt_dyca_3s.pkl"),
               "Erlangen": ("erlangen_eigenwerte_3s.pkl", "erlangen_eigenwerte_dyca_3s.pkl")}
    aus = {}
    for satz, dateien in quellen.items():
        aus[satz] = {}
        for meth, datei in zip(("snr", "dyca"), dateien):
            pfad = erg_dir / datei
            if not pfad.exists():
                return None
            d = pickle.load(open(pfad, "rb"))
            fe = ([dict(e, art=art) for art in ("ruhe", "anfall") for e in d["neu_alle"][art]]
                  if "neu_alle" in d else [e for e in d["fenster"] if e["art"] in ("ruhe", "anfall")])
            ruhe = [e["lam"][0] for e in fe if e["art"] == "ruhe"]
            anfall = [e["lam"][0] for e in fe if e["art"] == "anfall"]
            aus[satz][meth] = {"auc": auc(ruhe, anfall), "ruhe": np.median(ruhe),
                               "anfall": np.median(anfall)}
            if meth == "snr":
                je = {}
                for e in fe:
                    je.setdefault(e["datei"], {}).setdefault(e["art"], []).append(
                        np.mean(e["sigma"] ** 2))
                q = [np.median(v["anfall"]) / np.median(v["ruhe"]) - 1
                     for v in je.values() if "anfall" in v and "ruhe" in v]
                aus[satz]["snr_q"] = np.percentile(q, (25, 50, 75))
    return aus


ZIEL_MARKER = {"TUHZ-ABSZ": dict(marker="*", markersize=15, color=ZIEL),
               "Erlangen": dict(marker="*", markersize=15, markerfacecolor="white",
                                markeredgecolor=ZIEL, color=ZIEL)}


def abbildung_ueber_snr(erg, ziel, wert, eeg=None):
    """wert = "auc": AUC λ1; wert = "lam1": Median λ1 im Anfall, Ruhe gepunktet."""
    fig, ax = plt.subplots(len(SYSTEME), 2, figsize=(14, 4.6 * len(SYSTEME)), sharex=True)
    titel = {"auc": "AUC von λ1, Ruhe gegen Anfall",
             "lam1": "λ1 im Anfall (Linie) und in Ruhe (gepunktet), Median"}[wert]
    fig.suptitle("Rauschleiter: %s" % titel, fontsize=13)
    for z, name in enumerate(SYSTEME):
        for sp, (meth, kurz) in enumerate((("snr", "SFA (Schritt 1+2)"), ("dyca", "DyCA"))):
            a = ax[z, sp]
            stil_achse(a)
            for s, e in erg.items():
                if wert == "auc":
                    y = [e[meth][(name, snr)]["auc"] for snr in SNR_STUFEN]
                else:
                    y = [np.median(e[meth][(name, snr)]["anfall"][:, 0]) for snr in SNR_STUFEN]
                    a.axhline(np.median(e[meth]["ruhe"][:, 0]), color=FARBE[s], linestyle=":",
                              linewidth=1)
                a.plot(SNR_STUFEN, y, LINIE[s], marker=MARKER[s], color=FARBE[s], markersize=6,
                       linewidth=1.6)
            for satz, z_eeg in (eeg or {}).items():
                q1, m, q3 = z_eeg["snr_q"]
                y = z_eeg[meth]["auc" if wert == "auc" else "anfall"]
                a.plot([q1, q3], [y, y], color=ZIEL, linewidth=1)
                a.plot(m, y, linestyle="none", **ZIEL_MARKER[satz])
            a.set_xscale("log")
            if wert == "auc":
                a.set_ylim(0.4, 1.02)
                a.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
            elif meth == "snr":
                a.set_yscale("log")
            else:
                a.set_ylim(0, 1.02)
            a.set_title("%s — %s" % (name, kurz), fontsize=10)
            a.set_ylabel("AUC" if wert == "auc" else "λ1")
            if z == len(SYSTEME) - 1:
                a.set_xlabel("SNR (Varianz Attraktor / Median der Ruhevarianz)")
    griffe = [plt.Line2D([], [], color=FARBE[s], linestyle=LINIE[s], marker=MARKER[s],
                         markersize=7, label=TEXT[s]) for s in erg]
    griffe += [plt.Line2D([], [], linestyle="none", label="echtes EEG %s (SNR: Median, Quartile)"
                          % satz, **ZIEL_MARKER[satz]) for satz in (eeg or {})]
    fig.legend(handles=griffe, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(ziel, dpi=130)
    plt.close(fig)


def main():
    es.N_KANAL = 27
    ruhe_echt = {"S6 Erlangen": ruhe_erlangen(), "S6 TUHZ-ABSZ": ruhe_tuh()}
    kal = kalibrieren(ruhe_echt)
    attraktoren = {name: attraktor_fenster(name) for name in SYSTEME}
    erg = {}
    for stufe in STUFEN:
        print("rechne", TEXT[stufe])
        erg[stufe] = stufe_auswerten(stufe, ruhe_echt, kal, attraktoren)
    tabelle(erg)
    eeg = ziel_eeg()
    if eeg:
        print("\n=== Ziel: echtes EEG, λ1 Ruhe gegen Anfall ===")
        for satz, z in eeg.items():
            print("%-10s SNR Median %.1f (Quartile %.1f–%.1f) | SFA AUC %.2f | DyCA AUC %.2f"
                  % (satz, z["snr_q"][1], z["snr_q"][0], z["snr_q"][2],
                     z["snr"]["auc"], z["dyca"]["auc"]))

    aus = HIER / "ergebnisse"
    abbildung_kennzahlen(erg, kal, aus / "rauschleiter_kennzahlen.png")
    abbildung_ueber_snr(erg, aus / "rauschleiter_auc.png", "auc", eeg)
    abbildung_ueber_snr(erg, aus / "rauschleiter_lambda1.png", "lam1", eeg)
    with open(aus / "rauschleiter.pkl", "wb") as fh:
        pickle.dump({"kalibrierung": kal, "snr_stufen": SNR_STUFEN, "ergebnis": erg,
                     "ziel_eeg": eeg}, fh)
    print("\ngespeichert:", aus / "rauschleiter_*.png / .pkl")


if __name__ == "__main__":
    main()
