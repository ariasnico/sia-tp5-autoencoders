"""TP5 1a — genera las figuras desde los artefactos de results/. Estilo central de entrega.

E0 (exploración) se computa acá; E1..E8 salen de CSVs / curvas / ganador. No reentrena.
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
    fig, ax = plt.subplots(4, 8, figsize=(12, 7))
    for j in range(N):
        a = ax[j // 8, j % 8]
        a.imshow(X[j].reshape(H, W)); a.set_title(LABELS[j], fontsize=13); a.axis("off")
    fig.suptitle("E0 · Dataset font.h: 32 caracteres de 7×5 (binarios)")
    fig.subplots_adjust(wspace=0.4, hspace=0.5, left=0.03, right=0.97)
    fig.savefig(FIGS / "fig_e0a_dataset_letters.png", pad_inches=0.3)
    plt.close(fig); print("   fig_e0a_dataset_letters.png")


def e0b():
    hm = hamming_matrix(X); dens = pixel_density(X)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.8))
    im = a1.imshow(hm, cmap="magma"); a1.grid(False)
    a1.set_xticks(range(N)); a1.set_xticklabels(LABELS, fontsize=7)
    a1.set_yticks(range(N)); a1.set_yticklabels(LABELS, fontsize=7)
    a1.set_title("Distancia Hamming entre caracteres (# píxeles distintos)")
    fig.colorbar(im, ax=a1, fraction=0.046)
    hmoff = hm + np.eye(N, dtype=int) * 999
    i, j = np.unravel_index(np.argmin(hmoff), hm.shape)
    a1.set_xlabel(f"par más parecido: '{LABELS[i]}' ~ '{LABELS[j]}' a sólo {hm[i, j]} px", color=PRIMARY)
    a2.bar(range(N), dens, color=SECONDARY)
    a2.set_xticks(range(N)); a2.set_xticklabels(LABELS, fontsize=7)
    a2.set_ylabel("# píxeles encendidos (de 35)"); a2.set_title("Densidad de píxeles por letra")
    fig.suptitle("E0 · Dificultad: comprimir 35D→2D con caracteres casi idénticos")
    save(fig, "fig_e0b_dataset_stats.png")


def e1():
    df = pd.read_csv(RES / "e1_linear_vs_pca.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(range(len(df)), df["px_mean"], color=[PRIMARY, SECONDARY, "#9CA3AF"])
    ax.set_ylim(0, max(df["px_mean"]) * 1.4)  # aire para que las etiquetas no toquen el título
    for i, (m, mx) in enumerate(zip(df["px_mean"], df["px_max"])):
        ax.text(i, m + 0.2, f"prom {m:.2f}\nmax {mx}", ha="center", fontsize=11)
    ax.set_xticks(range(len(df))); ax.set_xticklabels(df["modelo"], fontsize=11)
    ax.set_ylabel("px incorrectos por letra (promedio)")
    ax.set_title("E1 · AE lineal ≡ PCA (~7 px); el no-lineal llega a 0", pad=14)
    save(fig, "fig_e1_linear_vs_pca.png")


def e2():
    df = pd.read_csv(RES / "e2_latent_sweep.csv")
    n = int(df["n_seeds"].iloc[0]) if "n_seeds" in df else 1
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # banda min-max + barras de error (desvío) sobre las 3 semillas
    ax.fill_between(df["latent"], df["px_max_min"], df["px_max_max"],
                    color=PRIMARY, alpha=0.15, label="px máx. rango min–max")
    ax.errorbar(df["latent"], df["px_max_mean"], yerr=df["px_max_std"], fmt="o-",
                color=PRIMARY, label="px máximo (media ± desvío)", lw=2.5, ms=9, capsize=4)
    ax.errorbar(df["latent"], df["px_mean_mean"], yerr=df["px_mean_std"], fmt="s--",
                color=SECONDARY, label="px promedio (media ± desvío)", lw=2, ms=8, capsize=4)
    ax.axhline(1, color="gray", ls=":", label="objetivo ≤1px")
    ax.axvline(2, color=ACCENT, ls=":", alpha=0.7, label="latente 2D (enunciado)")
    ax.set_xlabel("dimensión del espacio latente"); ax.set_ylabel("px incorrectos")
    ax.set_title(f"E2 · El codo: con 1D no alcanza; desde 2D el error cae a 0 (n={n} semillas)")
    ax.legend()
    save(fig, "fig_e2_latent_elbow.png")


def e3():
    df = pd.read_csv(RES / "e3_architecture.csv")
    n = int(df["n_seeds"].iloc[0]) if "n_seeds" in df else 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    a1.bar(range(len(df)), df["px_max_mean"], yerr=df["px_max_std"], color=PRIMARY,
           capsize=5, error_kw=dict(ecolor="#374151", lw=1.5))
    a1.set_xticks(range(len(df))); a1.set_xticklabels(df["hidden"], fontsize=11)
    a1.set_ylabel("px máximo (media ± desvío)"); a1.axhline(1, color="gray", ls=":")
    a1.set_title(f"px máximo vs arquitectura del encoder (n={n} semillas)")
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
    df = pd.read_csv(RES / "e5_lr.csv")
    n = int(df["n_seeds"].iloc[0]) if "n_seeds" in df else 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    # curva de convergencia (semilla 0)
    for key, lab, c in [("e5_lr_0.0003", "lr=0.0003 (lento)", "#2563EB"),
                        ("e5_lr_0.01", "lr=0.01 (justo)", ACCENT),
                        ("e5_lr_0.3", "lr=0.3 (no aprende)", PRIMARY)]:
        a1.plot(curves[key], label=lab, color=c, lw=2)
    a1.set_yscale("log"); a1.set_xlabel("época"); a1.set_ylabel("train loss (BCE, log)")
    a1.set_title("convergencia (semilla 0)"); a1.legend()
    # px_max medio ± desvío sobre las semillas
    a2.bar(range(len(df)), df["px_max_mean"], yerr=df["px_max_std"], color=PRIMARY,
           capsize=5, error_kw=dict(ecolor="#374151", lw=1.5))
    a2.set_xticks(range(len(df))); a2.set_xticklabels([f"lr={lr}" for lr in df["lr"]], fontsize=11)
    a2.set_ylabel("px máximo (media ± desvío)"); a2.axhline(1, color="gray", ls=":", label="objetivo ≤1px")
    a2.set_title(f"px máximo vs learning rate (n={n} semillas)"); a2.legend()
    fig.suptitle("E5 · Learning rate: muy bajo es lento; muy alto (0.3) queda atascado arriba; 0.01 llega a 0 px")
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
    fig.suptitle("E8 · Ganador 35-20-2-20-35: Original (sup.) vs Reconstrucción (inf.) — 0 px error")
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
    NG = 14
    gx = np.linspace(x0 - dx, x1 + dx, NG); gy = np.linspace(y1 + dy, y0 - dy, NG)
    canvas = np.ones((len(gy) * H, len(gx) * W))
    cell_bits = np.zeros((NG, NG, H * W))     # decodificación BINARIZADA de cada celda
    for r, yy in enumerate(gy):
        for c2, xx in enumerate(gx):
            dec = decode(ae, [[xx, yy]])[0]
            canvas[r * H:(r + 1) * H, c2 * W:(c2 + 1) * W] = dec.reshape(H, W)
            cell_bits[r, c2] = (dec > 0.5).astype(float)
    fig, ax = plt.subplots(figsize=(9.5, 9.8)); ax.imshow(canvas); ax.axis("off")
    # Anotación HONESTA: una celda es "copia" si su imagen decodificada REPRODUCE en PÍXELES
    # (Hamming <= THR_PX) alguna de las 32 letras reales. Conteo real (no un percentil).
    Xbits = (X > 0.5).astype(float)           # las 32 letras del dataset, binarizadas
    THR_PX = 1
    n_known = 0
    for r in range(NG):
        for c2 in range(NG):
            ham = np.abs(cell_bits[r, c2][None, :] - Xbits).sum(1).min()  # px vs la letra más parecida
            if ham <= THR_PX:
                rect = plt.Rectangle((c2 * W - 0.5, r * H - 0.5), W, H,
                                     fill=False, edgecolor="#d62728", lw=2.0, zorder=5)
                ax.add_patch(rect)
                n_known += 1
    ax.set_title("E8 · Generación: barrido del espacio latente (cada celda = letra decodificada, req. 1a-4)")
    ax.text(0.5, -0.045,
            f"Recuadro rojo = la celda reproduce (≤{THR_PX} px) una de las 32 letras reales "
            f"({n_known}/{NG*NG}). El resto son combinaciones continuas del decoder:\n"
            f"cerca de las letras aparecen glifos nuevos; en zonas alejadas el decoder satura.",
            transform=ax.transAxes, ha="center", va="top", fontsize=11, color="#333333")
    fig.subplots_adjust(bottom=0.10)
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


def e0c_hamming():
    """Heatmap de distancia Hamming solo (para la slide del dataset): motiva que comprimir a 2D
    es difícil porque hay letras casi idénticas que igual hay que separar."""
    hm = hamming_matrix(X)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    im = ax.imshow(hm, cmap="magma"); ax.grid(False)
    ax.set_xticks(range(N)); ax.set_xticklabels(LABELS, fontsize=7)
    ax.set_yticks(range(N)); ax.set_yticklabels(LABELS, fontsize=7)
    hmoff = hm + np.eye(N, dtype=int) * 999
    i, j = np.unravel_index(np.argmin(hmoff), hm.shape)
    ax.set_title("Qué tan distintos son los caracteres entre sí\n(# de píxeles en que difieren)")
    ax.set_xlabel(f"hay pares casi idénticos: '{LABELS[i]}' ~ '{LABELS[j]}' a sólo {hm[i, j]} px",
                  color=PRIMARY)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "fig_e0c_hamming.png")


def e8e_latent_vs_hamming():
    """Estructura de similitud en píxeles (Hamming) vs en el mapa latente 2D del campeón.
    La correlación de Spearman entre las distancias de a pares dice si el AE preservó el
    parecido entre letras. Determinístico (sin azar): mismo orden de LABELS en ambas matrices."""
    from scipy.stats import spearmanr
    Z = np.load(RES / "champion_latent.npz")["Z"]
    hm = hamming_matrix(X).astype(float)
    diff = Z[:, None, :] - Z[None, :, :]
    lm = np.sqrt((diff ** 2).sum(-1))                 # distancia euclídea por par en el latente
    iu = np.triu_indices(N, k=1)                      # 496 pares (triángulo superior)
    rho, _ = spearmanr(hm[iu], lm[iu])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, M, t in [(a1, hm, "Distancia en píxeles (Hamming)"),
                     (a2, lm, "Distancia en el mapa latente 2D")]:
        im = ax.imshow(M, cmap="magma"); ax.grid(False)
        ax.set_xticks(range(N)); ax.set_xticklabels(LABELS, fontsize=6)
        ax.set_yticks(range(N)); ax.set_yticklabels(LABELS, fontsize=6)
        ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"E8 · ¿El latente preservó el parecido entre letras?   Spearman ρ = {rho:.2f}")
    save(fig, "fig_e8e_latent_vs_hamming.png")
    print(f"   [Spearman Hamming vs latente = {rho:.4f}]")
    return rho


def e1_pca_pixel_error():
    """Dónde falla el modelo lineal/PCA(2): fracción de letras con ese píxel mal, promediada
    sobre las 32 letras y reshape a 7×5. Determinístico (PCA por SVD analítico)."""
    mu = X.mean(0); Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xr = (Xc @ Vt[:2].T) @ Vt[:2] + mu                # reconstrucción PCA(2)
    err = np.abs((Xr > 0.5).astype(float) - X).mean(0).reshape(H, W)
    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    im = ax.imshow(err, cmap="magma", vmin=0); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("E1 · Dónde falla PCA(2)\n(error por píxel, prom. 32 letras)")
    fig.colorbar(im, ax=ax, fraction=0.05, label="frac. de letras con ese píxel mal")
    save(fig, "fig_e1_pca_pixel_error.png")


if __name__ == "__main__":
    print("Generando figuras 1a:")
    e0a(); e0b(); e0c_hamming(); e1(); e1_pca_pixel_error(); e2(); e3(); e4(); e5(); e6(); e7()
    e8(); e8e_latent_vs_hamming()
    print("OK figuras 1a en", FIGS)
