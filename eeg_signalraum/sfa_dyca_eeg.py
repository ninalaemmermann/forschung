"""
Gegenprobe am echten EEG: kippt der Vorteil von DyCA zu SFA, wenn man dem EEG
weißes Messrauschen hinzufügt?

Vorhersage aus sfa_dyca_vergleich.py (Schnitt C): SFA wird erst besser, wenn
weißes, kanalunabhängiges Rauschen den größten Teil der Rauschleistung stellt.

Versuch: echte Ruhe- und Anfallsfenster (3 s, Ruhe ≥ 30 s von jedem Anfall,
Auswahl wie in tuh_resampelt.py / erlangen.py), dazu weißes Rauschen der Dosis

    w = Anteil an der Rauschleistung,   σ² = w / (1 − w) · Median der
    Ruhevarianz DERSELBEN Aufnahme (Mittel über Kanäle)

Dasselbe Rauschen für SFA und DyCA. Je w: AUC von λ1 Ruhe gegen Anfall,
ΔAUC = AUC_SFA − AUC_DyCA mit gepaartem Bootstrap und Urteil (``vergleiche``
aus sfa_dyca_vergleich.py).

    TUHZ-ABSZ   resampelt ungefiltert, 21 Kanäle - Hauptsatz
    Erlangen    roh - nur ergänzend, wenige Absencen

Vorhersage zum Vergleich: dasselbe Experiment in der Simulation - EEG-artiges
Rauschen (Kreuzspektrum + Schwankung wie echte Ruhe), Attraktor mit SNR wie
die echten TUH-Absencen relativ zu diesem Rauschen, dann dieselbe Dosis
weißes Rauschen obendrauf.

    python sfa_dyca_eeg.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import rauschleiter_attraktoren as rl
import sfa_dyca_vergleich as sv
import tuh_resampelt as tr
import erlangen as er
from erlangen_ruhe import zustandsfenster
from eeg_signalraum import fenster_je_anfall, stil_achse, DATEN, SURROGAT, MUTED, INK_SEK
from parameter import SEED, SYSTEME

L = 3.0
ABSTAND = 30.0
W_STUFEN = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99)
FARBE = {"sfa": DATEN, "dyca": SURROGAT}
NAME = {"sfa": "SFA (Schritt 1+2)", "dyca": "DyCA"}
STIL = {"TUHZ-ABSZ": dict(linestyle="-", marker="o", markersize=7, linewidth=2),
        "Erlangen": dict(linestyle="--", marker="s", markersize=6, linewidth=1.4,
                         markerfacecolor="white"),
        "Rössler": dict(linestyle=":", marker=None, linewidth=1.4),
        "Lorenz": dict(linestyle="-.", marker=None, linewidth=1.4)}
BESCHRIFTUNG = {"TUHZ-ABSZ": "echtes EEG TUHZ-ABSZ", "Erlangen": "echtes EEG Erlangen",
                "Rössler": "Vorhersage Simulation Rössler", "Lorenz": "Vorhersage Simulation Lorenz"}


# ===========================================================================
# Echte Fenster: Liste von {"datei", "art", "Y"} mit art in ruhe / anfall
# ===========================================================================

def fenster_tuh():
    aus = []
    for name, edf in tr.neue_dateien("absz").items():
        anfaelle = tr.anfaelle_bi(edf)
        if not anfaelle:
            continue
        raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
        sr = raw.info["sfreq"]
        n = int(round(L * sr))
        dauer = raw.n_times / sr
        gesehen = set()
        for a, b in anfaelle:
            for art, t0 in fenster_je_anfall(a, b, anfaelle, L, ABSTAND, dauer).items():
                i0 = int(round(t0 * sr))
                if art == "uebergang" or (art, i0) in gesehen or i0 + n > raw.n_times:
                    continue
                gesehen.add((art, i0))
                Y = raw.get_data(picks=tr.KANAELE, start=i0, stop=i0 + n).T * 1e6
                aus.append({"datei": name, "art": art, "Y": Y})
    return aus


def fenster_erlangen():
    return [{"datei": e["datei"], "art": e["art"], "Y": e["Y"]}
            for e in zustandsfenster(er.anfallsliste()) if e["art"] in ("ruhe", "anfall")]


# ===========================================================================
# Dosis weißes Rauschen und Urteil
# ===========================================================================

def bezugsvarianz(fenster):
    """Median der Ruhevarianz je Aufnahme; ohne Ruhefenster: Median über alle Aufnahmen."""
    je = {}
    for f in fenster:
        if f["art"] == "ruhe":
            je.setdefault(f["datei"], []).append(f["Y"].var(axis=0).mean())
    je = {d: np.median(v) for d, v in je.items()}
    rest = np.median(list(je.values()))
    return {f["datei"]: je.get(f["datei"], rest) for f in fenster}


def auswerten(fenster, satz, rng):
    """Je Dosis w: Urteil über beide Verfahren auf denselben verrauschten Fenstern."""
    bezug = bezugsvarianz(fenster)
    zeilen = []
    for w in W_STUFEN:
        werte = {"ruhe": [], "anfall": []}
        for f in fenster:
            Y = f["Y"]
            if w > 0:
                Y = Y + np.sqrt(w / (1 - w) * bezug[f["datei"]]) * rng.standard_normal(Y.shape)
            l = sv.lam1_beide(Y)
            if l is not None:
                werte[f["art"]].append(l)
        erg = sv.vergleiche(np.array(werte["ruhe"]), np.array(werte["anfall"]), rng)
        zeilen.append(dict(satz=satz, w=w, n_ruhe=len(werte["ruhe"]),
                           n_anfall=len(werte["anfall"]), **erg))
        print("  %-10s w = %-5g  SFA %.2f  DyCA %.2f  Δ %+.2f [%+.2f, %+.2f]  %s"
              % (satz, w, erg["auc_sfa"], erg["auc_dyca"], erg["delta"], erg["ki_unten"],
                 erg["ki_oben"], sv.URTEIL_TEXT[erg["urteil"]]))
    return zeilen


def vorhersage(snr, kal, rng):
    """Dasselbe Experiment in der Simulation: Signal fest gegen den EEG-artigen Teil."""
    zeilen = []
    for system in SYSTEME:
        A = rl.attraktor_fenster(system)[:sv.N_FENSTER]
        eeg_ruhe = rl.fenster(rl.kreuz_fenster(sv.N_FENSTER * sv.N, rng, kal["kreuz"],
                                               kal["streu_log_var"], kal["streu_alpha"]),
                              sv.N_FENSTER)
        eeg_anf = rl.fenster(rl.kreuz_fenster(sv.N_FENSTER * sv.N, rng, kal["kreuz"],
                                              kal["streu_log_var"], kal["streu_alpha"]),
                             sv.N_FENSTER)
        bezug = np.median([E.var(axis=0).mean() for E in eeg_ruhe])
        anfall = []
        for E, X in zip(eeg_anf, A):
            q = X @ rl.mischmatrix(E.shape[1])
            q -= q.mean(axis=0)
            anfall.append(E + np.sqrt(snr * bezug / q.var(axis=0).mean()) * q)
        fenster = ([{"datei": "sim", "art": "ruhe", "Y": Y} for Y in eeg_ruhe]
                   + [{"datei": "sim", "art": "anfall", "Y": Y} for Y in anfall])
        zeilen += auswerten(fenster, system, rng)
    return zeilen


# ===========================================================================
# Abbildung
# ===========================================================================

def abbildung(df, snr, ziel):
    x = np.arange(len(W_STUFEN))
    fig, (oben, unten) = plt.subplots(2, 1, figsize=(11, 10), sharex=True,
                                      gridspec_kw={"height_ratios": [1.2, 1]})
    fig.suptitle("Gegenprobe am echten EEG: weißes Rauschen dosiert hinzugefügt", fontsize=13)
    fig.text(0.5, 0.935, "Vorhersage: Simulation mit EEG-artigem Rauschen, Attraktor bei SNR %.0f "
             "wie die echten TUH-Absencen" % snr, ha="center", fontsize=9.5, color=INK_SEK)
    for satz, d in df.groupby("satz", sort=False):
        d = d.set_index("w").loc[list(W_STUFEN)]
        echt = satz in ("TUHZ-ABSZ", "Erlangen")
        for meth in ("sfa", "dyca"):
            oben.plot(x, d["auc_" + meth], color=FARBE[meth], alpha=1 if echt else 0.75,
                      **STIL[satz])
        farbe = "#0b0b0b" if echt else MUTED
        unten.plot(x, d["delta"], color=farbe, **STIL[satz])
        if satz == "TUHZ-ABSZ":
            unten.fill_between(x, d["ki_unten"], d["ki_oben"], color=MUTED, alpha=0.25,
                               linewidth=0)
        if satz == "Erlangen":
            unten.errorbar(x, d["delta"], yerr=[d["delta"] - d["ki_unten"], d["ki_oben"] - d["delta"]],
                           color=farbe, linestyle="none", capsize=3, linewidth=1)
    stil_achse(oben)
    oben.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    oben.set_ylim(0.4, 1.03)
    oben.set_ylabel("AUC von λ1, Ruhe gegen Anfall")
    stil_achse(unten)
    unten.axhline(0, color=MUTED, linestyle="--", linewidth=1)
    unten.text(x[-1] + 0.3, 0.02, "SFA besser", color=INK_SEK, fontsize=9, va="bottom", ha="right")
    unten.text(x[-1] + 0.3, -0.02, "DyCA besser", color=INK_SEK, fontsize=9, va="top", ha="right")
    unten.set_ylabel("ΔAUC = SFA − DyCA\n(TUH: Band = 95-%-Bootstrap)")
    unten.set_xticks(x)
    unten.set_xticklabels(["%g" % w for w in W_STUFEN])
    unten.set_xlabel("Weißanteil w an der Rauschleistung (σ² = w/(1−w) · Ruhevarianz der Aufnahme)")
    griffe = [plt.Line2D([], [], color=FARBE[m], linewidth=3, label=NAME[m]) for m in FARBE]
    griffe += [plt.Line2D([], [], color=INK_SEK, label=BESCHRIFTUNG[s],
                          **{k: v for k, v in STIL[s].items() if k != "linewidth"})
               for s in df.satz.unique()]
    oben.legend(handles=griffe, fontsize=9, frameon=False, loc="lower left", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ziel, dpi=130)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED + 7)
    zeilen = []
    print("TUHZ-ABSZ laden …")
    tuh = fenster_tuh()
    print("  %d Ruhe-, %d Anfallsfenster" % (sum(f["art"] == "ruhe" for f in tuh),
                                            sum(f["art"] == "anfall" for f in tuh)))
    zeilen += auswerten(tuh, "TUHZ-ABSZ", rng)
    print("Erlangen laden …")
    erl = fenster_erlangen()
    zeilen += auswerten(erl, "Erlangen", rng)

    eeg = rl.ziel_eeg()
    snr = float(eeg["TUHZ-ABSZ"]["snr_q"][1]) if eeg else 13.0
    kal = pickle.load(open(HIER / "ergebnisse" / "rauschleiter.pkl", "rb"))["kalibrierung"]
    print("Vorhersage Simulation, SNR %.1f …" % snr)
    zeilen += vorhersage(snr, kal, rng)

    df = pd.DataFrame(zeilen)
    aus = HIER / "ergebnisse"
    df.to_csv(aus / "sfa_dyca_eeg.csv", index=False)
    abbildung(df, snr, aus / "sfa_dyca_eeg.png")
    print("\ngespeichert:", aus / "sfa_dyca_eeg.png / .csv")


if __name__ == "__main__":
    main()
