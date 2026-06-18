"""TP5 1a — genera las figuras desde los artefactos de results/. Estilo central de entrega.

E0 (exploración) se computa acá; E1..E8 salen de CSVs / curvas / campeón. No reentrena.
"""
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
RES = HERE / "results"
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
FONT = ROOT / "font.h"

from tp5lib.fonts import load_font, LABELS, H, W, hamming_matrix, pixel_density  # noqa: E402
from tp5lib.autoencoder import decode                                            # noqa: E402
from tp5lib.plotstyle import apply_style, PRIMARY, SECONDARY, ACCENT              # noqa: E402
from mlp.network import MLP                                                       # noqa: E402

apply_style()
X = load_font(FONT)
N = len(X)
curves = np.load(RES / "curves_1a.npz")


def save(fig, name):
    fig.savefig(FIGS / name); plt.close(fig); print("  ", name)


def e0a():
    fig, ax = plt.subplots(4, 8, figsize=(11, 6.5))
    for j in range(N):
        a = ax[j // 8, j % 8]
        a.imshow(X[j].reshape(H, W)); a.set_title(LABELS[j], fontsize=13); a.axis("off")
    fig.suptitle("E0 · Dataset font.h: 32 caracteres de 7×5 (binarios)")
    save(fig, "fig_e0a_dataset_letters.png")


def e0b():
    hm = hamming_matrix(X); dens = pixel_density(X)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.8))
    im = a1.imshow(hm, cmap="magma"); a1.grid(False)
    a1.set_xticks(range(N)); a1.set_xticklabels(LABELS, fontsize=7)
    a1.set_yticks(range(N)); a1.set_yticklabels(LABELS, fontsize=7)
    a1.set_title("Distancia Hamming entre letras (# píxeles distintos)")
    fig.colorbar(im, ax=a1, fraction=0.046)
    hmoff = hm + np.eye(N, dtype=int) * 999
    i, j = np.unravel_index(np.argmin(hmoff), hm.shape)
    a1.set_xlabel(f"par más parecido: '{LABELS[i]}' ~ '{LABELS[j]}' a sólo {hm[i, j]} px", color=PRIMARY)
    a2.bar(range(N), dens, color=SECONDARY)
    a2.set_xticks(range(N)); a2.set_xticklabels(LABELS, fontsize=7)
    a2.set_ylabel("# píxeles encendidos (de 35)"); a2.set_title("Densidad de píxeles por letra")
    fig.suptitle("E0 · Dificultad: comprimir 35D→2D con letras casi idénticas")
    save(fig, "fig_e0b_dataset_stats.png")


def e1():
    df = pd.read_csv(RES / "e1_linear_vs_pca.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(range(len(df)), df["px_mean"], color=[PRIMARY, SECONDARY, "#9CA3AF"])
    for i, (m, mx) in enumerate(zip(df["px_mean"], df["px_max"])):
        ax.text(i, m + 0.15, f"prom {m:.2f}\nmax {mx}", ha="center", fontsize=12)
    ax.set_xticks(range(len(df))); ax.set_xticklabels(df["modelo"], fontsize=11)
    ax.set_ylabel("px incorrectos por letra (promedio)")
    ax.set_title("E1 · AE lineal ≡ PCA (~7 px); el no-lineal llega a 0")
    save(fig, "fig_e1_linear_vs_pca.png")


def e2():
    df = pd.read_csv(RES / "e2_latent_sweep.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df["latent"], df["px_max"], "o-", color=PRIMARY, label="px máximo", lw=2.5, ms=9)
    ax.plot(df["latent"], df["px_mean"], "s--", color=SECONDARY, label="px promedio", lw=2, ms=8)
    ax.axhline(1, color="gray", ls=":", label="objetivo ≤1px")
    ax.axvline(2, color=ACCENT, ls=":", alpha=0.7, label="latente 2D (enunciado)")
    ax.set_xlabel("dimensión del espacio latente"); ax.set_ylabel("px incorrectos")
    ax.set_title("E2 · El codo: con 1D no alcanza; desde 2D el error cae a 0")
    ax.legend()
    save(fig, "fig_e2_latent_elbow.png")


def e3():
    df = pd.read_csv(RES / "e3_architecture.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    a1.bar(range(len(df)), df["px_max"], color=PRIMARY)
    a1.set_xticks(range(len(df))); a1.set_xticklabels(df["hidden"], fontsize=11)
    a1.set_ylabel("px máximo"); a1.axhline(1, color="gray", ls=":")
    a1.set_title("px máximo vs arquitectura del encoder")
    for key, lab in [("e3_none", "()"), ("e3_10", "(10,)"), ("e3_20", "(20,)"),
                     ("e3_30", "(30,)"), ("e3_20-20", "(20,20)")]:
        a2.plot(curves[key], label=lab, lw=1.8)
    a2.set_yscale("log"); a2.set_xlabel("época"); a2.set_ylabel("train loss (BCE, log)")
    a2.set_title("convergencia"); a2.legend(title="hidden")
    fig.suptitle("E3 · Sin capa oculta el encoder es lineal (≈PCA); con ≥20 unidades llega a 0 px")
    save(fig, "fig_e3_architecture.png")


def e4():
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for key, lab, c in [("e4_SGD(0.5)", "SGD(0.5)", "#9CA3AF"),
                        ("e4_Momentum(0.1)", "Momentum(0.1)", SECONDARY), ("e4_Adam(0.01)", "Adam(0.01)", PRIMARY)]:
        ax.plot(curves[key], label=lab, color=c, lw=2)
    ax.set_yscale("log"); ax.set_xlabel("época"); ax.set_ylabel("train loss (BCE, log)")
    ax.set_title("E4 · Optimizadores: Adam converge ~100× más bajo que SGD")
    ax.legend()
    save(fig, "fig_e4_optimizers.png")


def e5():
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for key, lab, c in [("e5_lr_0.0003", "lr=0.0003 (lento)", "#2563EB"),
                        ("e5_lr_0.01", "lr=0.01 (justo)", ACCENT),
                        ("e5_lr_0.3", "lr=0.3 (no aprende)", PRIMARY)]:
        ax.plot(curves[key], label=lab, color=c, lw=2)
    ax.set_yscale("log"); ax.set_xlabel("época"); ax.set_ylabel("train loss (BCE, log)")
    ax.set_title("E5 · Learning rate: muy bajo es lento; muy alto (0.3) queda atascado arriba")
    ax.legend()
    save(fig, "fig_e5_lr.png")


def e6():
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for key, lab in [("e6_tanh", "tanh"), ("e6_relu", "relu"), ("e6_sigmoid", "sigmoid")]:
        ax.plot(curves[key], label=lab, lw=2)
    ax.set_yscale("log"); ax.set_xlabel("época"); ax.set_ylabel("train loss (BCE, log)")
    ax.set_title("E6 · Activación oculta: las tres llegan a 0 px, pero difieren en velocidad")
    ax.legend()
    save(fig, "fig_e6_activation.png")


def e7():
    df = pd.read_csv(RES / "e7_loss.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    a1.bar(df["loss"].str.upper(), df["px_max"], color=[PRIMARY, SECONDARY])
    a1.set_ylabel("px máximo"); a1.axhline(1, color="gray", ls=":")
    a1.set_title("error de reconstrucción")
    for key, lab in [("e7_bce", "BCE"), ("e7_mse", "MSE")]:
        a2.plot(curves[key], label=lab, lw=2)
    a2.set_yscale("log"); a2.set_xlabel("época"); a2.set_ylabel("train loss (log)")
    a2.set_title("convergencia"); a2.legend()
    fig.suptitle("E7 · Para imágenes binarias, BCE reconstruye perfecto; MSE deja 2 px")
    save(fig, "fig_e7_loss.png")


def e8():
    ae = MLP.load(RES / "champion_1a.npz")
    Z = np.load(RES / "champion_latent.npz")["Z"]
    rec = (ae.predict_proba(X) > 0.5).astype(float)
    # E8a — reconstrucciones (orig fila par / recon fila impar)
    fig, ax = plt.subplots(8, 8, figsize=(11, 11))
    for j in range(N):
        blk, col = j // 8, j % 8
        ax[blk * 2, col].imshow(X[j].reshape(H, W)); ax[blk * 2, col].set_title(LABELS[j], fontsize=11); ax[blk * 2, col].axis("off")
        ax[blk * 2 + 1, col].imshow(rec[j].reshape(H, W)); ax[blk * 2 + 1, col].axis("off")
    fig.suptitle("E8 · Campeón 35-20-2-20-35: Original (sup.) vs Reconstrucción (inf.) — 0 px error")
    save(fig, "fig_e8a_reconstructions.png")
    # E8b — scatter latente con inset de zoom al cluster central
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.scatter(Z[:, 0], Z[:, 1], s=20, c=PRIMARY, zorder=3)
    for j in range(N):
        ax.annotate(LABELS[j], (Z[j, 0], Z[j, 1]), fontsize=12, ha="center", va="center", color=SECONDARY, weight="bold")
    ax.set_title("E8 · Espacio latente 2D: las 32 letras (req. 1a-3)"); ax.set_xlabel("z1"); ax.set_ylabel("z2")
    c = Z.mean(0); d = np.linalg.norm(Z - c, axis=1); keep = d <= np.percentile(d, 68)
    zx0, zx1 = Z[keep, 0].min(), Z[keep, 0].max(); zy0, zy1 = Z[keep, 1].min(), Z[keep, 1].max()
    mx, my = (zx1 - zx0) * 0.12, (zy1 - zy0) * 0.12
    axins = ax.inset_axes([1.04, 0.0, 0.62, 1.0])
    axins.scatter(Z[:, 0], Z[:, 1], s=22, c=PRIMARY, zorder=3)
    for j in range(N):
        axins.annotate(LABELS[j], (Z[j, 0], Z[j, 1]), fontsize=14, ha="center", va="center", color=SECONDARY, weight="bold")
    axins.set_xlim(zx0 - mx, zx1 + mx); axins.set_ylim(zy0 - my, zy1 + my)
    axins.set_title("zoom del centro (legible)", fontsize=12)
    ax.indicate_inset_zoom(axins, edgecolor="gray")
    save(fig, "fig_e8b_latent2d.png")
    # E8c — generación por barrido de la grilla latente
    x0, x1, y0, y1 = Z[:, 0].min(), Z[:, 0].max(), Z[:, 1].min(), Z[:, 1].max()
    dx, dy = (x1 - x0) * .15, (y1 - y0) * .15
    gx = np.linspace(x0 - dx, x1 + dx, 14); gy = np.linspace(y1 + dy, y0 - dy, 14)
    canvas = np.ones((len(gy) * H, len(gx) * W))
    for r, yy in enumerate(gy):
        for c2, xx in enumerate(gx):
            canvas[r * H:(r + 1) * H, c2 * W:(c2 + 1) * W] = decode(ae, [[xx, yy]])[0].reshape(H, W)
    fig, ax = plt.subplots(figsize=(9, 9)); ax.imshow(canvas); ax.axis("off")
    ax.set_title("E8 · Generación: barrido del espacio latente (cada celda = letra decodificada, req. 1a-4)")
    save(fig, "fig_e8c_generation_grid.png")
    # E8d — interpolación a -> o
    za, zo = Z[LABELS.index("a")], Z[LABELS.index("o")]
    S = 9; fig, ax = plt.subplots(1, S, figsize=(S * 1.2, 2.0))
    for k in range(S):
        t = k / (S - 1); g = decode(ae, ((1 - t) * za + t * zo)[None, :])[0].reshape(H, W)
        ax[k].imshow(g); ax[k].axis("off")
        ax[k].set_title("a" if k == 0 else "o" if k == S - 1 else f"{t:.2f}", fontsize=12)
    fig.suptitle("E8 · Interpolación en el latente: de 'a' a 'o' (los intermedios son letras nuevas)")
    save(fig, "fig_e8d_interpolation.png")


if __name__ == "__main__":
    print("Generando figuras 1a:")
    e0a(); e0b(); e1(); e2(); e3(); e4(); e5(); e6(); e7(); e8()
    print("OK figuras 1a en", FIGS)
