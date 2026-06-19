"""TP5 1b — figuras E9-E11 + ganador reforzado, desde results/. Estilo central."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
RES = HERE / "results"; FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
FONT = ROOT / "font.h"

from tp5lib.fonts import load_font, LABELS, H, W, bitflip            # noqa: E402
from tp5lib.plotstyle import apply_style, PRIMARY, SECONDARY, ACCENT  # noqa: E402
from mlp.network import MLP                                           # noqa: E402

apply_style()
X = load_font(FONT); N = len(X)


def save(fig, name):
    fig.savefig(FIGS / name); plt.close(fig); print("  ", name)


def e9():
    df = pd.read_csv(RES / "e9_bottleneck.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for col, lab, c in [("px@0.1", "test 10%", "#2563EB"), ("px@0.2", "test 20%", PRIMARY), ("px@0.3", "test 30%", "#7C3AED")]:
        a1.plot(df["cuello"], df[col], "o-", label=lab, color=c, lw=2, ms=8)
    a1.set_xlabel("ancho del cuello (dim latente)"); a1.set_ylabel("px incorrectos tras limpiar (prom)")
    a1.set_title("denoising mejora al ensanchar el cuello"); a1.legend(); a1.set_xticks(df["cuello"])
    a2.bar(df["cuello"].astype(str), df["%<=1px@0.2"], color=SECONDARY)
    a2.set_xlabel("ancho del cuello"); a2.set_ylabel("% letras recuperadas (≤1px) @ test 20%")
    a2.set_title("% recuperadas vs cuello")
    fig.suptitle("E9 · El cuello 2D (de 1a) denoisa mal; ensanchar a 10 ayuda (20 ya no mejora)")
    save(fig, "fig_e9_bottleneck.png")


def e10():
    df = pd.read_csv(RES / "e10_train_noise.csv")
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for ptr, c in [(0.05, "#2563EB"), (0.15, ACCENT), (0.30, PRIMARY)]:
        sub = df[df["p_train"] == ptr]
        ax.plot(100 * sub["test_p"], sub["px_mean"], "o-", color=c, lw=2, ms=7, label=f"entrenado a {int(ptr * 100)}%")
        ax.axvline(100 * ptr, color=c, ls=":", alpha=0.5)
    ax.set_xlabel("nivel de ruido en test (% píxeles flipeados)")
    ax.set_ylabel("px incorrectos tras limpiar (prom)")
    ax.set_title("E10 · Trade-off del ruido de train: poco=frágil ante mucho ruido, mucho=peor en limpio")
    ax.legend(title="ruido de entrenamiento")
    save(fig, "fig_e10_train_noise.png")


def e_champion():
    df = pd.read_csv(RES / "e_champion.csv")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    x = np.arange(len(df)); w = 0.38
    ax.bar(x - w / 2, df["pct_leq1"], w, label="% ≤1px", color=SECONDARY)
    ax.bar(x + w / 2, df["pct_perfect"], w, label="% perfectas", color=PRIMARY)
    for i, (a, b) in enumerate(zip(df["pct_leq1"], df["pct_perfect"])):
        ax.text(i - w / 2, a + 1.5, f"{a:.0f}", ha="center", fontsize=12)
        ax.text(i + w / 2, b + 1.5, f"{b:.0f}", ha="center", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels([f"test {int(p * 100)}%" for p in df["test_p"]])
    ax.set_ylabel("% de letras"); ax.set_ylim(0, 105)
    ax.set_title("Ganador denoising (cuello 10, 15000 ep): % de letras recuperadas")
    ax.legend()
    save(fig, "fig_e_champion.png")


def e11():
    dae = MLP.load(RES / "dae_champion.npz")
    letters = [LABELS.index(c) for c in ["a", "e", "g", "r", "s"]]
    levels = [0.10, 0.20, 0.30]
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(len(levels) * 3, len(letters), figsize=(len(letters) * 1.6, len(levels) * 3 * 1.0))
    for li, p in enumerate(levels):
        Xn = bitflip(X, p, rng)
        rec = (dae.predict_proba(Xn) > 0.5).astype(float)
        for ci, idx in enumerate(letters):
            r0 = li * 3
            for r, img, lab in [(r0, X[idx], "limpio"), (r0 + 1, Xn[idx], f"ruido {int(p*100)}%"),
                                (r0 + 2, rec[idx], "recuperado")]:
                ax = axes[r, ci]; ax.imshow(img.reshape(H, W)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                if ci == 0:
                    ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=11)
            if li == 0:
                axes[r0, ci].set_title(LABELS[idx], fontsize=13)
    fig.suptitle("E11 · DAE ganador (cuello=10, p_train=15%): limpio / ruidoso / recuperado")
    save(fig, "fig_e11_triplets.png")


if __name__ == "__main__":
    print("Generando figuras 1b:")
    e9(); e10(); e_champion(); e11()
    print("OK figuras 1b en", FIGS)
