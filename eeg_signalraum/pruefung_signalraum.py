"""Prüfung: Code-Identität (B) und Vergleich des 'Rauschens' Lorenz/Rössler vs. EEG (A).

Abbildung: λ-Spektren und Spektrum gegen den Rauschboden aus R.
Rössler/Lorenz nur im guten Fall.
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch

BASE = Path("/home/data/ninalaemmermann/forschung")
sys.path.insert(0, str(BASE / "stdgl" / "Signalrekonstruktion"))
sys.path.insert(0, str(BASE / "stdgl" / "werkzeuge"))
sys.path.insert(0, str(BASE / "eeg_signalraum"))
sys.path.insert(0, str(BASE))

from funktionen import rauschmatrix, signalraum, signalrichtungen, beobachte_bahn
from parameter import bahn, DT, DIM, K, GUT, SCHLECHT, SPREIZUNG, SEED, N_BAHN
import eeg_signalraum as es
import erlangen as er
from Verteilungsfkt import get_sz_start_end

AUS = Path(__file__).resolve().parent / "ergebnisse"
rng_W = np.random.default_rng(SEED)
W = rng_W.standard_normal((3, DIM))


def beobachtung(name, fall):
    X = bahn(name, N_BAHN, fall[0], SEED)
    return beobachte_bahn(X, fall[1], DIM, np.random.default_rng(SEED), W, SPREIZUNG)


# ---------------------------------------------------------------- Schritt B
print("=== Schritt B: Code-Identität ===")
for name in ("Rössler", "Lorenz"):
    for fall in (GUT, SCHLECHT):
        Y = beobachtung(name, fall)
        lam_ref = signalraum(Y, rauschmatrix(Y)[0], K)[1]
        lam_eeg, r = es.spektrum(Y)
        lam_eeg = lam_eeg[:r]
        Yavg = Y - Y.mean(axis=1, keepdims=True)
        lam_avg, r2 = es.spektrum(Yavg)
        lam_avg = lam_avg[:r2]
        print("%-8s %-10s max|Δλ|/λ = %.1e  K=%d | AvgRef: Rang %d, K=%d, λ1..5 = %s"
              % (name, fall, np.max(np.abs(lam_ref - lam_eeg) / lam_ref),
                 signalrichtungen(lam_ref), r2, signalrichtungen(lam_avg),
                 np.round(lam_avg[:5], 2)))
        print("         λ1..5 Referenz = %s" % np.round(lam_ref[:5], 2))


# ---------------------------------------------------------------- Kennzahlen
def kennzahlen(Y, fs):
    """Kennzahlen dessen, was Schritt 1 als Rauschen behandelt."""
    Y = Y - Y.mean(axis=0)
    R1 = rauschmatrix(Y, lag=1)[0]
    ev = np.linalg.eigvalsh(R1)
    keep = ev > 1e-8 * ev[-1]
    # Rangdefizit (Average-Referenz) über Projektion entfernen
    U = np.linalg.eigh(R1)[1][:, keep]
    Yp = Y @ U
    R1p = rauschmatrix(Yp, 1)[0]
    lam = signalraum(Yp, R1p, 3)[1]
    # Spektrum, normiert auf das Niveau, das R als weißes Rauschen annimmt
    f, P = welch(Y, fs=fs, nperseg=min(len(Y), int(2 * fs) if fs > 150 else 512), axis=0)
    boden = np.trace(R1) / len(R1) * 2 / fs
    lueck = lam[:-1] / lam[1:]
    return dict(f=f / fs, P=P.mean(axis=1) / boden,
                lam=lam, lam1=lam[0], gap=lueck.max(), gap_bei=int(lueck.argmax()) + 1)


ergebnis = {}
for name in ("Rössler", "Lorenz"):
    ergebnis[name] = [kennzahlen(beobachtung(name, GUT), 1 / DT)]

# TUH ABSZ: Ruhe und Anfall
def tuh(typ="absz", L=3.0, abstand=30.0, maxn=40):
    st = typ.upper()
    liste = get_sz_start_end(BASE / f"{st}_seizures.csv")
    aus = {"ruhe": [], "anfall": []}
    for datei, sz in liste.items():
        p = BASE / "Data" / f"{st}_seizure_WB" / f"{datei}_res.OWN11101_filtWB_avg.edf"
        if not p.exists():
            continue
        anf = [(float(a), float(b)) for a, b in sz if np.isfinite(a) and np.isfinite(b) and b > a]
        if not anf:
            continue
        raw = mne.io.read_raw_edf(p, preload=False, verbose=False)
        sr = raw.info["sfreq"]
        n = int(round(L * sr))
        for a, b in anf[:2]:
            for art, t0 in es.fenster_je_anfall(a, b, anf, L, abstand, raw.n_times / sr).items():
                if art not in aus or len(aus[art]) >= maxn:
                    continue
                i0 = int(round(t0 * sr))
                Y = raw.get_data(start=i0, stop=i0 + n).T * 1e6
                aus[art].append(kennzahlen(Y, sr))
    return aus, sr

t, sr_tuh = tuh()
ergebnis["TUH-ABSZ Ruhe"], ergebnis["TUH-ABSZ Anfall"] = t["ruhe"], t["anfall"]

# Erlangen
liste = er.anfallsliste()
erl = {"ruhe": [], "anfall": []}
for lang, info in liste.items():
    raw = er.lies(lang)
    picks = er.eeg_kanaele(raw)
    sr_erl = raw.info["sfreq"]
    n = int(round(3.0 * sr_erl))
    alle = info["absencen"] + info["sperrzonen"]
    for a, b in info["absencen"]:
        for art, t0 in es.fenster_je_anfall(a, b, alle, 3.0, 30.0, info["dauer"]).items():
            if art not in erl:
                continue
            i0 = int(round(t0 * sr_erl))
            Y = raw.get_data(picks=picks, start=i0, stop=i0 + n).T * 1e6
            erl[art].append(kennzahlen(Y, sr_erl))
ergebnis["Erlangen Ruhe"], ergebnis["Erlangen Anfall"] = erl["ruhe"], erl["anfall"]
print("fs TUH = %g Hz, fs Erlangen = %g Hz" % (sr_tuh, sr_erl))

# ---------------------------------------------------------------- Tabelle
print("\n=== Schritt A: Was Schritt 1 als Rauschen sieht (Median über Fenster) ===")
print("%-18s %4s %10s %12s" % ("Fall", "n", "λ1", "Lücke (bei)"))
for k, v in ergebnis.items():
    med = lambda s: np.median([x[s] for x in v])
    print("%-18s %4d %10.3g %7.1f (%d)"
          % (k, len(v), med("lam1"), med("gap"),
             int(np.median([x["gap_bei"] for x in v]))))

with open(AUS / "rauschvergleich.pkl", "wb") as fh:
    pickle.dump({k: [{s: x[s] for s in x} for x in v] for k, v in ergebnis.items()}, fh)

# ---------------------------------------------------------------- Abbildung
farben = {"Rössler": "#1f77b4", "Lorenz": "#2ca02c", "TUH-ABSZ Ruhe": "#d62728",
          "TUH-ABSZ Anfall": "#ff9896", "Erlangen Ruhe": "#9467bd", "Erlangen Anfall": "#c5b0d5"}
fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
for k, v in ergebnis.items():
    c = farben[k]
    L = min(len(x["lam"]) for x in v)
    ax[0].semilogy(np.arange(1, L + 1), np.median([x["lam"][:L] for x in v], axis=0), "o-",
                   color=c, ms=3, label=k)
    fmin = min(len(x["f"]) for x in v)
    ax[1].loglog(v[0]["f"][1:fmin], np.median([x["P"][:fmin] for x in v], axis=0)[1:], color=c, label=k)
ax[0].axhline(1, color="k", ls="--")
ax[0].set_title("λ = 1 + SNR (Schritt 1+2)")
ax[0].set_xlabel("Richtung")
ax[0].set_ylabel("λ")
ax[0].legend(fontsize=8)
ax[1].axhline(1, color="k", ls="--")
ax[1].set_title("Spektrum / Rauschboden aus R\n(1 = Niveau, das R als weißes Rauschen annimmt)")
ax[1].set_xlabel("f / fs")
for a in ax:
    a.grid(alpha=0.3)
fig.suptitle("Was Schritt 1 als Rauschen behandelt: Rössler/Lorenz gegen EEG")
fig.tight_layout()
fig.savefig(AUS / "rauschvergleich.png", dpi=110)
print("gespeichert:", AUS / "rauschvergleich.png")
