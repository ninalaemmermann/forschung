"""
Eine TUH-Absence-Aufnahme über die Zeit: Schritt 1+2 (λ = 1 + SNR) und DyCA untereinander,
jeweils WB-Export (gefiltert, 27 Kanäle) und resampelt ungefiltert (21 Kanäle).

Rechnet nichts neu, sondern liest die gespeicherten Zeitreihen:
    SNR   ergebnisse/tuh_resampelt_zeitreihe_<L>s.pkl   (tuh_resampelt_zeitreihe.py)
    DyCA  ergebnisse/dyca_zeitreihe_<L>s.pkl            (dyca_zeitreihe.py, Sätze tuh_wb / tuh)

    python zeitreihe_snr_dyca.py                         # aaaaabqv_s002_t001
    python zeitreihe_snr_dyca.py --dateien aaaaacyi_s001_t001
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
from eeg_signalraum import stil_achse, MUTED, DATEN, INK_SEK
from tuh_resampelt_zeitreihe import anfaelle_zeichnen, FARBE_LAM, N_LAM

# Satz -> (Schlagwort, Beschreibung, Farbe, Schlüssel im SNR-Pickle, Schlüssel im DyCA-Pickle)
SAETZE = (("gefiltert", "WB, 27 Kanäle, Average-Referenz", MUTED, "alt", "tuh_wb"),
          ("ungefiltert", "resampelt, 21 Kanäle", DATEN, "neu", "tuh"))


def panel_titel(ax, schlagwort, beschreibung, farbe):
    """Fettes, farbiges Schlagwort links über dem Panel, dahinter die Beschreibung."""
    t = ax.text(0, 1.03, schlagwort, transform=ax.transAxes, fontsize=12, fontweight="bold",
                color=farbe, va="bottom", ha="left")
    ax.annotate("  " + beschreibung, xy=(1, 0), xycoords=t, va="bottom", ha="left",
                fontsize=10, color=INK_SEK)
    ax.spines["left"].set_color(farbe)
    ax.spines["left"].set_linewidth(3)


def abbildung(name, snr, dyca, L, schritt, ziel):
    """Zwei Blöcke (SFA, DyCA) zu je zwei Zeilen (gefiltert, ungefiltert), gemeinsame Zeitachse."""
    fig, achsen = plt.subplots(5, 1, figsize=(18, 15.5), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1, 0.3, 1, 1]})
    achsen[2].axis("off")
    sfa, dy = achsen[:2], achsen[3:]
    n_anf = len(snr["neu"]["anfaelle"])
    fig.suptitle("SFA vs DyCA Eigenwerte über die Zeit (Fenster %gs, Schritt %gs)" % (L, schritt),
                 fontsize=15, y=0.985)
    fig.text(0.5, 0.955, "TUHZ-ABSZ, Aufnahme %s, %d Anfälle" % (name, n_anf),
             ha="center", fontsize=10.5, color=INK_SEK)

    oben = max(np.nanmax(snr[s[3]]["lam"][:, 0]) for s in SAETZE)
    for ax, (wort, text, farbe, s, _) in zip(sfa, SAETZE):
        e = snr[s]
        stil_achse(ax)
        anfaelle_zeichnen(ax, e["anfaelle"], beschriften=True)
        for i in range(N_LAM):
            ax.semilogy(e["t"], e["lam"][:, i], color=FARBE_LAM[i], linewidth=0.9,
                        label="λ%d" % (i + 1))
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
        ax.set_ylim(0.8, oben * 1.5)
        ax.set_ylabel("λ = 1 + SNR")
        panel_titel(ax, wort, text, farbe)

    for ax, (wort, text, farbe, _, d) in zip(dy, SAETZE):
        e = dyca[d]
        stil_achse(ax)
        anfaelle_zeichnen(ax, e["anfaelle"], beschriften=True)
        for i in range(N_LAM):
            ax.plot(e["t"], e["lam"][:, i], color=FARBE_LAM[i], linewidth=0.9,
                    label="λ%d" % (i + 1))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("DyCA-Eigenwert λ")
        panel_titel(ax, wort, text, farbe)

    # Legende nur im ersten Panel jedes Blocks
    for ax in (sfa[0], dy[0]):
        ax.legend(fontsize=9, frameon=False, ncol=N_LAM + 1, loc="lower right",
                  bbox_to_anchor=(1, 1.0))

    ende = max(r[k]["t"][-1] for r, k in ((snr, "alt"), (snr, "neu"), (dyca, "tuh_wb"), (dyca, "tuh")))
    achsen[-1].set_xlim(0, ende + L / 2)
    achsen[-1].set_xlabel("Zeit (Fenstermitte) [s]")
    fig.tight_layout(rect=(0, 0, 1, 0.925))

    # Blocküberschriften über dem jeweils ersten Panel (nach dem Layout platziert)
    for ax, kopf in ((sfa[0], "SFA (λ = 1 + SNR, λ1–λ%d)" % N_LAM),
                     (dy[0], "DyCA (generalisierte Eigenwerte, λ1–λ%d)" % N_LAM)):
        pos = ax.get_position()
        fig.text(pos.x0, pos.y1 + 0.03, kopf, ha="left", va="bottom", fontsize=13.5,
                 fontweight="bold")
    fig.savefig(ziel, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dateien", nargs="+", default=["aaaaabqv_s002_t001"])
    ap.add_argument("--fenster-sek", type=float, default=3.0)
    args = ap.parse_args()
    L = args.fenster_sek
    ausgabe = HIER / "ergebnisse"

    snr_pkl = pickle.load(open(ausgabe / ("tuh_resampelt_zeitreihe_%gs.pkl" % L), "rb"))
    dyca_pkl = pickle.load(open(ausgabe / ("dyca_zeitreihe_%gs.pkl" % L), "rb"))
    if snr_pkl["schritt_sek"] != dyca_pkl["schritt_sek"]:
        sys.exit("Schrittweite verschieden: SNR %g s, DyCA %g s"
                 % (snr_pkl["schritt_sek"], dyca_pkl["schritt_sek"]))
    schritt = snr_pkl["schritt_sek"]
    snr_alle = snr_pkl["zeitreihen"]
    dyca_alle = {d: dyca_pkl["datensaetze"][d]["zeitreihen"] for *_, d in SAETZE}

    for name in args.dateien:
        if name not in snr_alle or any(name not in dyca_alle[d] for *_, d in SAETZE):
            print("übersprungen (nicht in beiden Pickles): %s" % name)
            continue
        ziel = ausgabe / ("zeitreihe_snr_dyca_%gs_%s.png" % (L, name))
        abbildung(name, snr_alle[name], {d: dyca_alle[d][name] for *_, d in SAETZE},
                  L, schritt, ziel)
        print("gespeichert:", ziel)


if __name__ == "__main__":
    main()
