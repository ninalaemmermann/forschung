"""
Erlangen-Absencen mit der Ruhe als Rauschen (csp.py) - Ruhe, Übergang, Anfall.

    C·v = λ·R·v,   R = mittlere Kovarianz der bereinigten Ruhe des Patienten
                   C = Kovarianz eines 3-s-Fensters

λ = Varianz im Fenster / Varianz in Ruhe, je Richtung. Gezählt werden die
Richtungen über der Schwelle aus der Nullprobe (nullprobe_erlangen.py,
Aufteilung „gemischt“, 95-%-Quantil von λ1): K = Zahl der Richtungen, in
denen sich das Fenster deutlicher von der Ruhe unterscheidet, als es
Ruhefenster untereinander tun.

Rohdaten, ungefiltert. Ruhepool: alle 3-s-Fenster ≥ 30 s von jeder Absence
und Sperrzone, Artefaktfenster (Varianz > Median + 5·MAD) verworfen. Ein
getestetes Fenster fließt nie in sein eigenes R ein.

Fenster je Absence wie in erlangen.py (``fenster_je_anfall``).

    python erlangen_ruhe.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import erlangen as er
import nullprobe_erlangen as nb
from eeg_signalraum import ARTEN, BESCHRIFTUNG, FARBE, fenster_je_anfall, stil_achse, MUTED
from stil import INK_SEK
from csp import ruhe_kovarianz, kontrast

L = nb.L
ABSTAND = nb.ABSTAND


def ueberlappt(t0, t1, s0, s1):
    return t0 < s1 and s0 < t1


def zustandsfenster(liste):
    """Je Absence die Fenster Ruhe/Übergang/Anfall wie in erlangen.py."""
    aus = []
    for lang, info in liste.items():
        raw = er.lies(lang)
        picks = er.eeg_kanaele(raw)
        sr = raw.info["sfreq"]
        n = int(round(L * sr))
        alle = info["absencen"] + info["sperrzonen"]
        for a, b in info["absencen"]:
            for art, t0 in fenster_je_anfall(a, b, alle, L, ABSTAND, info["dauer"]).items():
                i0 = int(round(t0 * sr))
                if i0 + n > raw.n_times:
                    continue
                Y = raw.get_data(picks=picks, start=i0, stop=i0 + n).T * 1e6
                aus.append({"patient": info["patient"], "datei": lang, "art": art,
                            "anfall": (a, b), "start_s": t0, "Y": Y})
    return aus


def main():
    liste = er.anfallsliste()
    pool = nb.ruhefenster(liste)
    fenster = zustandsfenster(liste)
    patienten = sorted(pool)

    referenz = {}
    for pat in patienten:
        gs, weg = nb.bereinige(pool[pat])
        lam_null, _ = nb.nullprobe(gs, "gemischt")
        referenz[pat] = {"pool": gs, "verworfen": weg, "lam_null": lam_null,
                         "schwelle": np.percentile(lam_null[:, 0], 95)}
        print("%s: %d Ruhefenster im Pool (%.0f %% verworfen), Schwelle λ1 = %.1f"
              % (pat, len(gs), 100 * weg, referenz[pat]["schwelle"]))

    R_alle = {p: ruhe_kovarianz([f[2] for f in referenz[p]["pool"]]) for p in patienten}
    for e in fenster:
        p = e["patient"]
        R = R_alle[p]
        t0, t1 = e["start_s"], e["start_s"] + L
        if any(f[0] == e["datei"] and ueberlappt(t0, t1, f[1], f[1] + L) for f in referenz[p]["pool"]):
            R = ruhe_kovarianz([f[2] for f in referenz[p]["pool"]
                                if not (f[0] == e["datei"] and ueberlappt(t0, t1, f[1], f[1] + L))])
        e["lam"] = kontrast(e["Y"], R)[1]
        e["K"] = int((e["lam"] > referenz[p]["schwelle"]).sum())

    print("\n%-3s %-9s %3s %24s %10s" % ("Pat", "Zustand", "n", "λ1 … λ4 (Median)", "K je Fenster"))
    fig, achsen = plt.subplots(1, len(patienten), figsize=(7.5 * len(patienten), 5.4), sharey=True)
    fig.suptitle("Erlangen, Absencen — Ruhe als Rauschen, Median über Absencen", fontsize=11)
    for ax, pat in zip(np.atleast_1d(achsen), patienten):
        stil_achse(ax)
        ref = referenz[pat]
        x = np.arange(1, ref["lam_null"].shape[1] + 1)
        ax.semilogy(x, np.percentile(ref["lam_null"], 97.5, axis=0), "-", color=MUTED,
                    linewidth=2.0, label="Ruhe gegen Ruhe (97,5 %): Grenze")
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.0)
        for art in ARTEN:
            E = [e for e in fenster if e["patient"] == pat and e["art"] == art]
            if not E:
                continue
            A = np.array([e["lam"] for e in E])
            med = np.median(A, axis=0)
            ks = [e["K"] for e in E]
            ax.semilogy(x, med, "o-", color=FARBE[art], markersize=4,
                        label="%s (%d Absencen)" % (BESCHRIFTUNG[art], len(E)))
            print("%-3s %-9s %3d %24s %10s" % (pat, art, len(E),
                  np.array2string(med[:4], precision=1), ks))
        ax.set_title("Patient %s (%d Kanäle)" % (pat, len(x)), fontsize=10)
        ax.set_xlabel("Richtung")
        ax.set_ylabel("λ = Varianz Fenster / Varianz Ruhe")
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stamm = HIER / "ergebnisse" / ("erlangen_ruhe_%gs" % L)
    fig.savefig(str(stamm) + ".png", dpi=140)
    with open(str(stamm) + ".pkl", "wb") as fh:
        pickle.dump({"fenster_sek": L, "abstand": ABSTAND,
                     "schwellen": {p: referenz[p]["schwelle"] for p in patienten},
                     "verworfen": {p: referenz[p]["verworfen"] for p in patienten},
                     "fenster": [{k: v for k, v in e.items() if k != "Y"} for e in fenster]}, fh)
    print("\ngespeichert:", str(stamm) + ".png / .pkl")


if __name__ == "__main__":
    main()
