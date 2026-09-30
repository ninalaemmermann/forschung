"""
Anfallstypen vergleichen über λ = 1 + SNR (Schritt 1 + 2), Ruhe → Anfall.

Warum: In ``eeg_signalraum.py`` bekommt jedes Fenster sein eigenes R. Im Anfall
wächst R (zweite Differenz) aber genauso wie das Signal - Spikes und Muskel-
aktivität sind nicht glatt und landen im "Rauschen". λ = C/R bleibt dann gleich.

Hier zwei Varianten je Fenster:
    eigen   R aus dem Fenster selbst (wie bisher)
    fest    R aus dem Ruhefenster desselben Anfalls, für alle drei Zustände

Je Anfall die Änderung Ruhe → Anfall (und Ruhe → Übergang), Median je Patient,
Vergleich der Typen (Kruskal-Wallis, paarweise Mann-Whitney mit Holm) und,
für Patienten mit mehreren Typen, der Vergleich innerhalb derselben Person.

    python anfallstypen.py
"""
import argparse
import itertools
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.stats import kruskal, mannwhitneyu, wilcoxon

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
from eeg_signalraum import (TYPEN, FARBE_TYP, MARKER_TYP, fenster_je_anfall,
                            get_sz_start_end, stil_achse, MUTED, INK_SEK)
from funktionen import rauschmatrix, signalraum, signalrichtungen

# Strenger als in eeg_signalraum.py (1e-8): dort bleiben die fünf praktisch
# leeren Richtungen nach Richtung 22 knapp drin und täuschen einen Sockel vor.
RANG_SCHWELLE = 1e-6
N_TOP = 5
VARIANTEN = ("eigen", "fest")
KENNZAHLEN = {
    "K": "K (Signalrichtungen)",
    "log_snr1": "log₁₀(λ₁ − 1)",
    "log_snr_top": "Mittel log₁₀(λᵢ − 1), Top %d" % N_TOP,
}
BESCHRIFTUNG_VAR = {"eigen": "R je Fenster (wie bisher)",
                    "fest": "R fest aus der Ruhe"}


# ===========================================================================
# Kennzahlen
# ===========================================================================

def kennzahlen(lam):
    snr = np.clip(lam - 1.0, 1e-3, None)       # λ < 1 ist Schätzfehler, kein negatives SNR
    return {"K": signalrichtungen(lam),
            "log_snr1": np.log10(snr[0]),
            "log_snr_top": np.mean(np.log10(snr[:N_TOP]))}


def anfall_auswerten(fenster):
    """fenster: {art: Y}, muss "ruhe" enthalten. Liste von Zeilen oder []."""
    R_ruhe, _ = rauschmatrix(fenster["ruhe"])
    if not np.isfinite(R_ruhe).all():
        return []
    ev, U = eigh(R_ruhe)
    if ev[-1] <= 0:
        return []
    P = U[:, ev > RANG_SCHWELLE * ev[-1]]
    if P.shape[1] <= N_TOP:
        return []
    R_fest, _ = rauschmatrix(fenster["ruhe"] @ P)

    zeilen = []
    for art, Y in fenster.items():
        Yp = Y @ P
        R_eigen, _ = rauschmatrix(Yp)
        z = {"art": art, "r": P.shape[1],
             "log_spur_R": np.log10(np.trace(R_eigen))}
        try:
            for var, R in (("eigen", R_eigen), ("fest", R_fest)):
                lam = signalraum(Yp, R, k=1)[1]
                z.update({"%s_%s" % (var, k): v for k, v in kennzahlen(lam).items()})
        except (np.linalg.LinAlgError, ValueError):
            continue                               # R nicht positiv definit
        zeilen.append(z)
    return zeilen


def datei_auswerten(typ, datei, sz, base, L, abstand):
    pfad = base / "Data" / f"{typ.upper()}_seizure_WB" / f"{datei}_res.OWN11101_filtWB_avg.edf"
    if not pfad.exists():
        return []
    anfaelle = [(float(a), float(b)) for a, b in sz
                if np.isfinite(a) and np.isfinite(b) and b > a]
    if not anfaelle:
        return []
    raw = mne.io.read_raw_edf(pfad, preload=False, verbose=False)
    sr = raw.info["sfreq"]
    n_win = int(round(L * sr))
    dauer = raw.n_times / sr
    aus = []
    for a, b in anfaelle:
        fenster = {}
        for art, t0 in fenster_je_anfall(a, b, anfaelle, L, abstand, dauer).items():
            i0 = int(round(t0 * sr))
            if i0 + n_win <= raw.n_times:
                fenster[art] = raw.get_data(start=i0, stop=i0 + n_win).T * 1e6   # µV
        if "ruhe" not in fenster:
            continue
        for z in anfall_auswerten(fenster):
            z.update(typ=typ, datei=datei, patient=datei[:8], anfall=(a, b))
            aus.append(z)
    return aus


def alle_auswerten(base, L, abstand, n_jobs):
    jobs = []
    for typ in TYPEN:
        for datei, sz in get_sz_start_end(base / f"{typ.upper()}_seizures.csv").items():
            jobs.append((typ, datei, sz))
    res = Parallel(n_jobs=n_jobs)(
        delayed(datei_auswerten)(typ, d, sz, base, L, abstand) for typ, d, sz in jobs)
    df = pd.DataFrame([z for r in res for z in r])
    df["anfall_id"] = df.datei + df.anfall.astype(str)
    return df


# ===========================================================================
# Statistik
# ===========================================================================

def aenderungen(df, ziel):
    """Änderung Ruhe → ziel je Anfall, dann Median je (Typ, Patient)."""
    spalten = ["log_spur_R"] + ["%s_%s" % (v, k) for v in VARIANTEN for k in KENNZAHLEN]
    w = df.pivot_table(index=["typ", "patient", "anfall_id"], columns="art",
                       values=spalten, aggfunc="first")
    d = pd.DataFrame({s: w[(s, ziel)] - w[(s, "ruhe")] for s in spalten})
    return d.dropna(how="all").groupby(level=["typ", "patient"]).median()


def holm(p):
    p = np.asarray(p, float)
    ordnung = np.argsort(p)
    aus = np.empty_like(p)
    laufend = 0.0
    for rang, i in enumerate(ordnung):
        laufend = max(laufend, (len(p) - rang) * p[i])
        aus[i] = min(laufend, 1.0)
    return aus


def statistik(pat, ziel):
    zeilen = []
    for spalte in pat.columns:
        gruppen = {t: pat.loc[t, spalte].dropna().values for t in TYPEN if t in pat.index.levels[0]}
        basis = {"vergleich": "ruhe→" + ziel, "kennzahl": spalte}
        zeilen.append({**basis, "test": "kruskal", "gruppe": "alle",
                       "p": kruskal(*gruppen.values()).pvalue,
                       **{"median_" + t: np.median(v) for t, v in gruppen.items()},
                       **{"n_" + t: len(v) for t, v in gruppen.items()}})
        paare = list(itertools.combinations(gruppen, 2))
        ps = [mannwhitneyu(gruppen[a], gruppen[b]).pvalue for a, b in paare]
        for (a, b), p, ph in zip(paare, ps, holm(ps)):
            zeilen.append({**basis, "test": "mannwhitney", "gruppe": "%s-%s" % (a, b),
                           "p": p, "p_holm": ph})
    return zeilen


def innerhalb_patienten(pat, ziel):
    """Patienten mit mehreren Typen: Differenz der Typen in derselben Person."""
    zeilen = []
    mehrfach = pat.groupby(level="patient").size()
    mehrfach = mehrfach[mehrfach > 1].index
    sub = pat[pat.index.get_level_values("patient").isin(mehrfach)]
    for spalte in pat.columns:
        breit = sub[spalte].unstack("typ")
        for a, b in itertools.combinations(TYPEN, 2):
            if a not in breit or b not in breit:
                continue
            diff = (breit[a] - breit[b]).dropna()
            if len(diff) == 0:
                continue
            p = wilcoxon(diff).pvalue if len(diff) >= 5 and (diff != 0).any() else np.nan
            zeilen.append({"vergleich": "ruhe→" + ziel, "kennzahl": spalte,
                           "test": "innerhalb_patient", "gruppe": "%s-%s" % (a, b),
                           "n_patienten": len(diff), "median_diff": diff.median(), "p": p})
    return zeilen, len(mehrfach)


# ===========================================================================
# Abbildung
# ===========================================================================

def abbildung(pat, stat, ziel, pfad, L):
    fig, achsen = plt.subplots(len(VARIANTEN), len(KENNZAHLEN),
                               figsize=(4.6 * len(KENNZAHLEN), 4.0 * len(VARIANTEN)),
                               sharey="col", squeeze=False)
    fig.suptitle("EEG — Änderung Ruhe → %s je Anfallstyp (Fenster %.3g s, je Punkt ein Patient)"
                 % ("Anfall" if ziel == "anfall" else "Übergang", L))
    rng = np.random.default_rng(0)
    kw = stat[stat.test == "kruskal"].set_index("kennzahl").p
    for i, var in enumerate(VARIANTEN):
        for j, (k, name) in enumerate(KENNZAHLEN.items()):
            ax = achsen[i, j]
            stil_achse(ax)
            spalte = "%s_%s" % (var, k)
            for x, typ in enumerate(TYPEN):
                v = pat.loc[typ, spalte].dropna().values
                ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=12, alpha=0.55,
                           color=FARBE_TYP[typ], marker=MARKER_TYP[typ], linewidths=0)
                ax.hlines(np.median(v), x - 0.3, x + 0.3, color=FARBE_TYP[typ], linewidth=2.5)
            ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
            ax.set_xticks(range(len(TYPEN)))
            ax.set_xticklabels(["%s\n(n=%d)" % (t.upper(), pat.loc[t, spalte].notna().sum())
                                for t in TYPEN])
            ax.set_title("%s — Kruskal-Wallis p = %.2g" % (BESCHRIFTUNG_VAR[var], kw[spalte]),
                         fontsize=9)
            if i == len(VARIANTEN) - 1:
                ax.set_xlabel(name)
            if j == 0:
                ax.set_ylabel("Δ (Anfall − Ruhe)" if ziel == "anfall" else "Δ (Übergang − Ruhe)")
    fig.text(0.5, 0.005, "Balken: Median über Patienten. Δ > 0: mehr Signal über dem Rauschen "
             "als in Ruhe.", ha="center", fontsize=8, color=INK_SEK)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(pfad, dpi=140)
    plt.close(fig)


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fenster-sek", type=float, default=3.0)
    ap.add_argument("--ruhe-abstand", type=float, default=30.0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--base", type=Path, default=Path("/home/data/ninalaemmermann/forschung"))
    args = ap.parse_args()

    ausgabe = HIER / "ergebnisse"
    ausgabe.mkdir(exist_ok=True)
    L = args.fenster_sek
    stamm = ausgabe / ("anfallstypen_snr_%gs" % L)

    df = alle_auswerten(args.base, L, args.ruhe_abstand, args.jobs)
    with open(str(stamm) + ".pkl", "wb") as fh:
        pickle.dump({"fenster_sek": L, "ruhe_abstand": args.ruhe_abstand,
                     "rang_schwelle": RANG_SCHWELLE, "fenster": df}, fh)
    print("%d Fenster, %d Anfälle, %d Patienten"
          % (len(df), df.anfall_id.nunique(), df.patient.nunique()))

    alle = []
    for ziel in ("anfall", "uebergang"):
        pat = aenderungen(df, ziel)
        stat = pd.DataFrame(statistik(pat, ziel))
        innen, n_mehrfach = innerhalb_patienten(pat, ziel)
        alle += stat.to_dict("records") + innen
        abbildung(pat, stat, ziel, "%s_%s.png" % (stamm, ziel), L)

        print("\nRuhe → %s, Kruskal-Wallis über die Typen (Median der Änderung je Typ):" % ziel)
        k = stat[stat.test == "kruskal"]
        print(k[["kennzahl", "p"] + ["median_" + t for t in TYPEN]]
              .round(4).to_string(index=False))
        print("Patienten mit mehreren Typen: %d" % n_mehrfach)

    pd.DataFrame(alle).to_csv(str(stamm) + ".csv", index=False)
    print("\ngespeichert:", str(stamm) + "_*.png / .csv / .pkl")


if __name__ == "__main__":
    main()
