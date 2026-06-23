"""Figuras para las nuevas diapos de denoising (presentación).

Genera 4 figuras en figs/:
  fig_noise_examples.png      — a,e,g,r,s limpias + bit-flip a 10/15/30% (cómo se ve el ruido)
  fig_ae1a_on_noise.png       — forward pass del AE de 1a (NO denoiser) con esas letras ruidosas
  fig_ae1a_pxerror.png        — px error promedio del AE de 1a con inputs a 0/10/15/30%
  fig_a_table_denoisers.png   — letra 'a': limpia / con ruido 0-40% / salida de 3 denoisers (5/15/30%)

Uso:  python3 figs_presentacion.py   (TP5_EPOCHS=300 para prueba rápida)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
FONT = ROOT / "font.h"
AE1A = ROOT / "ej1a_secuencial" / "results" / "champion_1a.npz"

from tp5lib.fonts import load_font, LABELS, H, W, D, bitflip       # noqa: E402
from tp5lib.autoencoder import build_ae, train_denoising, reconstruct_binary, px_err_clean  # noqa: E402
from tp5lib.plotstyle import apply_style, PRIMARY, SECONDARY, ACCENT, WARN  # noqa: E402
from mlp.network import MLP                                         # noqa: E402
from mlp.optimizers import Adam                                     # noqa: E402

apply_style()
X = load_font(FONT)
SEED = 0
EPOCHS = int(os.environ.get("TP5_EPOCHS", 6000))
LETTERS = ["a", "e", "g", "r", "s"]
LIDX = [LABELS.index(c) for c in LETTERS]
NOISE3 = [0.10, 0.15, 0.30]


def show(ax, vec):
    ax.imshow(vec.reshape(H, W), cmap="binary", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def noisy(vec, p, seed):
    rng = np.random.default_rng(seed)
    return bitflip(vec[None, :], p, rng)[0]


# ----------------------------- F1: ejemplos de ruido -----------------------------
def fig_noise_examples():
    rows = [("limpio", 0.0)] + [(f"{int(p*100)}%", p) for p in NOISE3]
    nR, nC = len(rows), len(LETTERS)
    fig, axes = plt.subplots(nR, nC, figsize=(nC * 1.25, nR * 1.35))
    for r, (lab, p) in enumerate(rows):
        for c, li in enumerate(LIDX):
            ax = axes[r][c]
            vec = X[li] if p == 0 else noisy(X[li], p, SEED * 100 + r * 10 + c)
            show(ax, vec)
            if r == 0:
                ax.set_title(LETTERS[c], fontsize=14)
        axes[r][0].set_ylabel(lab, fontsize=12, rotation=0, ha="right", va="center", labelpad=28)
    fig.suptitle("El ruido es bit-flip: cada píxel se invierte con probabilidad p (fresco cada época)", fontsize=12)
    fig.subplots_adjust(left=0.13, top=0.9, hspace=0.08, wspace=0.08)
    fig.savefig(FIGS / "fig_noise_examples.png"); plt.close(fig)
    print("  -> fig_noise_examples.png")


# ----------------------------- F2: AE de 1a sobre ruido -----------------------------
def fig_ae1a_on_noise():
    ae = MLP.load(AE1A)
    # filas: limpio, y por cada nivel (entrada ruidosa, salida del AE)
    row_specs = [("limpio", None, "clean")]
    for p in NOISE3:
        row_specs.append((f"ruido {int(p*100)}%", p, "in"))
        row_specs.append((f"salida {int(p*100)}%", p, "out"))
    nR, nC = len(row_specs), len(LETTERS)
    fig, axes = plt.subplots(nR, nC, figsize=(nC * 1.25, nR * 1.15))
    for r, (lab, p, kind) in enumerate(row_specs):
        for c, li in enumerate(LIDX):
            ax = axes[r][c]
            if kind == "clean":
                vec = X[li]
            else:
                nz = noisy(X[li], p, SEED * 100 + int(p * 100) * 10 + c)
                vec = nz if kind == "in" else reconstruct_binary(ae, nz[None, :])[0]
            show(ax, vec)
            if r == 0:
                ax.set_title(LETTERS[c], fontsize=14)
        color = SECONDARY if kind == "out" else ("#111" if kind == "clean" else WARN)
        axes[r][0].set_ylabel(lab, fontsize=11, rotation=0, ha="right", va="center", labelpad=30, color=color)
    fig.suptitle("AE de 1a (35-20-2-20-35, NO entrenado para limpiar) ante entradas ruidosas", fontsize=12)
    fig.subplots_adjust(left=0.16, top=0.93, hspace=0.06, wspace=0.06)
    fig.savefig(FIGS / "fig_ae1a_on_noise.png"); plt.close(fig)
    print("  -> fig_ae1a_on_noise.png")


# ----------------------------- F3: px error del AE de 1a -----------------------------
def fig_ae1a_pxerror():
    ae = MLP.load(AE1A)
    levels = [0.0, 0.10, 0.15, 0.30]
    trials = 30
    means = []
    for p in levels:
        rng = np.random.default_rng(123)
        errs = np.concatenate([px_err_clean(ae, bitflip(X, p, rng), X) for _ in range(trials)])
        means.append(errs.mean())
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = [f"{int(p*100)}%" for p in levels]
    cols = [SECONDARY] + [WARN] * (len(levels) - 1)
    bars = ax.bar(labels, means, color=cols, width=0.6)
    for b, m in zip(bars, means):
        ax.annotate(f"{m:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=11, xytext=(0, 2), textcoords="offset points")
    ax.set_ylim(0, max(means) * 1.25)
    ax.set_xlabel("ruido en la entrada"); ax.set_ylabel("px incorrectos (salida vs original limpio)")
    ax.set_title("El AE de 1a NO limpia: el error crece con el ruido de entrada")
    fig.savefig(FIGS / "fig_ae1a_pxerror.png"); plt.close(fig)
    print("  -> fig_ae1a_pxerror.png")


# ----------------------------- F6: tabla letra 'a' × denoisers -----------------------------
def _train_denoiser(p_train):
    rng = np.random.default_rng(SEED)
    dae = build_ae(D, hidden=(30,), latent=10, act_hidden="tanh", act_latent="identity",
                   act_out="sigmoid", loss="bce", optimizer=Adam(0.01), seed=SEED)
    train_denoising(dae, X, p_train, EPOCHS, rng)
    return dae


def _pick_letter(nets, p_trains, levels, hi=(0.30, 0.40), reps=12):
    """Elige la letra donde MÁS se nota la diferencia entre redes a ruido alto:
    el denoiser de 30% debería ganarle al de 5% (y a la inversa en limpio empatan).
    score = (err_net5 - err_net30) promediado en niveles altos, sobre `reps` realizaciones."""
    net_lo, net_hi = nets[p_trains[0]], nets[p_trains[-1]]
    best = []
    for li in range(len(X)):
        clean = X[li][None, :]
        gap = 0.0
        for p in hi:
            for r in range(reps):
                rng = np.random.default_rng(1000 + li * 50 + int(p * 100) + r)
                nz = bitflip(clean, p, rng)
                e_lo = px_err_clean(net_lo, nz, clean)[0]
                e_hi = px_err_clean(net_hi, nz, clean)[0]
                gap += (e_lo - e_hi)
        best.append((gap / (len(hi) * reps), li))
    best.sort(reverse=True)
    top = [(LABELS[li], round(g, 2)) for g, li in best[:6]]
    print(f"  letras con mayor diferencia entre redes (ruido alto): {top}")
    return best[0][1]


def fig_a_table_denoisers():
    levels = [0.0, 0.05, 0.15, 0.30, 0.40]
    p_trains = [0.05, 0.15, 0.30]
    print("  entrenando 3 denoisers (5/15/30%) ...")
    nets = {p: _train_denoiser(p) for p in p_trains}

    forced = os.environ.get("TP5_LETTER")
    ai = LABELS.index(forced) if forced else _pick_letter(nets, p_trains, levels)
    letter = LABELS[ai]
    print(f"  letra elegida para la tabla: '{letter}'")

    noisy_a = [X[ai] if p == 0 else noisy(X[ai], p, 777 + int(p * 100)) for p in levels]

    nC = len(levels)
    # filas: 0 = ref centrada, 1 = separador, 2 = entrada, 3..5 = denoisers
    nrows = 3 + len(p_trains)
    fig = plt.figure(figsize=(nC * 1.5, nrows * 1.2))
    gs = GridSpec(nrows, nC, figure=fig, hspace=0.18, wspace=0.10,
                  height_ratios=[1.0, 0.25] + [1.0] * (1 + len(p_trains)))

    # fila 0: letra limpia centrada
    axc = fig.add_subplot(gs[0, nC // 2])
    show(axc, X[ai]); axc.set_title(f"'{letter}' original (sin ruido)", fontsize=12)

    # fila 2: entradas con ruido (con los headers de nivel)
    for c, p in enumerate(levels):
        ax = fig.add_subplot(gs[2, c])
        show(ax, noisy_a[c])
        ax.set_title(f"{int(p*100)}%", fontsize=13)
        if c == 0:
            ax.set_ylabel("entrada", fontsize=11, rotation=0, ha="right", va="center", labelpad=34)

    # filas 3..: salida de cada denoiser
    for r, pt in enumerate(p_trains, start=3):
        for c in range(nC):
            ax = fig.add_subplot(gs[r, c])
            show(ax, reconstruct_binary(nets[pt], noisy_a[c][None, :])[0])
            if c == 0:
                ax.set_ylabel(f"denoiser {int(pt*100)}%", fontsize=11, rotation=0,
                              ha="right", va="center", labelpad=34, color=SECONDARY)
    fig.suptitle(f"Letra '{letter}' a distintos niveles de ruido, limpiada por cada denoiser (entrenado a 5/15/30%)", fontsize=12.5)
    fig.subplots_adjust(left=0.16, top=0.93)
    fig.savefig(FIGS / "fig_a_table_denoisers.png"); plt.close(fig)
    print("  -> fig_a_table_denoisers.png")


# ----------------------------- F7: las 32 letras por el denoiser 30% -----------------------------
def fig_denoise30_all_letters():
    """Cualitativo: las 32 letras → con ruido 30% → recuperadas por el denoiser entrenado a 30%.
    4 bloques de 8 columnas; cada bloque tiene 3 filas: limpio / ruido 30% / recuperado."""
    print("  entrenando denoiser 30% para las 32 letras ...")
    net = _train_denoiser(0.30)
    rng = np.random.default_rng(2024)
    noisy_all = bitflip(X, 0.30, rng)
    rec_all = reconstruct_binary(net, noisy_all)

    N = len(X)
    per = 8                      # letras por bloque
    blocks = (N + per - 1) // per
    rows_per = 3                 # limpio / ruido / recuperado
    nrows = blocks * rows_per + (blocks - 1)   # +1 fila separadora entre bloques
    fig = plt.figure(figsize=(per * 1.35, nrows * 1.05))
    hr = []
    for b in range(blocks):
        hr += [1.0, 1.0, 1.0]
        if b < blocks - 1:
            hr.append(0.35)
    gs = GridSpec(nrows, per, figure=fig, hspace=0.12, wspace=0.10, height_ratios=hr)

    rowlabs = ["limpio", "ruido 30%", "recuperado"]
    rowcols = ["#111", WARN, SECONDARY]
    src = [X, noisy_all, rec_all]
    for b in range(blocks):
        base = b * per
        gr0 = b * (rows_per + 1)            # fila de grid donde empieza el bloque
        for rr in range(rows_per):
            grow = gr0 + rr
            for c in range(per):
                li = base + c
                if li >= N:
                    continue
                ax = fig.add_subplot(gs[grow, c])
                show(ax, src[rr][li])
                if rr == 0:
                    ax.set_title(LABELS[li], fontsize=12)
                if c == 0:
                    ax.set_ylabel(rowlabs[rr], fontsize=10, rotation=0, ha="right",
                                  va="center", labelpad=30, color=rowcols[rr])
    fig.suptitle("Las 32 letras con ruido 30% recuperadas por el denoiser entrenado a 30%", fontsize=13)
    fig.subplots_adjust(left=0.12, top=0.95)
    fig.savefig(FIGS / "fig_denoise30_all.png"); plt.close(fig)
    print("  -> fig_denoise30_all.png")


if __name__ == "__main__":
    import sys as _sys
    print("Generando figuras de presentación denoising ...")
    if "--all30" in _sys.argv:
        fig_denoise30_all_letters(); print("OK"); _sys.exit(0)
    fig_noise_examples()
    fig_ae1a_on_noise()
    fig_ae1a_pxerror()
    fig_a_table_denoisers()
    fig_denoise30_all_letters()
    print("OK")
