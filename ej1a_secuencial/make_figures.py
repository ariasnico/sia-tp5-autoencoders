"""Figuras del camino SECUENCIAL (coordinate descent → ganador).

Lee los CSV de results/ y escribe PNGs en figs/ (carpeta propia, no pisa ../ej1a_autoencoder/figs).
Una figura por paso (exp1..exp7) + una figura resumen del camino.

Uso:  python3 make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
RES = HERE / "results"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

from tp5lib.plotstyle import apply_style, PRIMARY, SECONDARY, ACCENT, WARN, PALETTE  # noqa: E402
apply_style()

GREY = "#9AA5B1"


def _csv(name):
    return pd.read_csv(RES / name)


def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    print("  ->", name)


def bar_colors(values, win_idx, fail_pred=None):
    """Verde el ganador, rojo los que fallan (px_max>1), gris el resto."""
    cols = []
    for i, v in enumerate(values):
        if i == win_idx:
            cols.append(ACCENT)
        elif fail_pred is not None and fail_pred(v):
            cols.append(WARN)
        else:
            cols.append(GREY)
    return cols


def annotate_bars(ax, bars, labels):
    for b, lab in zip(bars, labels):
        ax.annotate(lab, (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 2),
                    textcoords="offset points")


def finish_bars(ax, heights, ymin_top=2.0):
    """Da aire arriba (para que las anotaciones no toquen el título). Devuelve el tope."""
    top = max(max(heights), ymin_top) * 1.28
    ax.set_ylim(0, top)
    return top


def mark_winner(ax, xi, top, text="★ ganador (0px · 32/32)"):
    """Marca al ganador con una estrella visible (útil cuando su barra es de altura ~0)."""
    y = top * 0.07
    ax.scatter([xi], [y], marker="*", s=360, color=ACCENT, zorder=6,
               edgecolor="white", linewidth=0.6)
    ax.annotate(text, (xi, y), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=9.5, fontweight="bold", color=ACCENT, zorder=6)


def best_so_far(g, col="train_loss"):
    """Curva del mejor valor acumulado: coincide con los pesos que fit() conserva."""
    return g[col].cummin()


# ----------------------------- exp1 · linealidad -----------------------------
def fig_exp1():
    df = _csv("exp1_summary.csv")
    order = ["no_lineal", "lineal", "pca_svd"]
    df = df.set_index("variante").loc[order].reset_index()
    labels = ["AE no-lineal", "AE lineal", "PCA(2)"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = range(len(df))
    cols = bar_colors(df["px_max"], 0, fail_pred=lambda v: v > 1)
    bars = ax.bar(x, df["px_max"], color=cols, width=0.6)
    annotate_bars(ax, bars, [f"{int(p)}/32 ok" for p in df["perfectas"]])
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("error máximo (px)")
    ax.set_title("Paso 1 · La no-linealidad rompe la barrera (lineal ≈ PCA)")
    finish_bars(ax, df["px_max"])
    ax.axhline(1, color=SECONDARY, ls="--", lw=1, alpha=.7)
    ax.annotate("objetivo ≤1px", (len(df) - 1, 1.3), color=SECONDARY, fontsize=9, ha="right")
    save(fig, "fig_exp1_linealidad.png")


# ----------------------------- exp2 · optimizador -----------------------------
def fig_exp2():
    cur = _csv("exp2_curves.csv")
    summ = _csv("exp2_summary.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, (v, g) in enumerate(cur.groupby("variante")):
        a1.plot(g["epoch"], best_so_far(g), label=v, color=PALETTE[i], lw=2)
    a1.set_yscale("log"); a1.set_xlabel("época"); a1.set_ylabel("mejor train loss acumulada (BCE, log)")
    a1.set_title("Convergencia por optimizador (best-so-far)"); a1.legend()

    labels = [f"{o}({lr})" for o, lr in zip(summ["optimizer"], summ["lr"])]
    win = int(summ["px_max"].values.argmin())
    cols = bar_colors(summ["px_max"], win, fail_pred=lambda v: v > 1)
    bars = a2.bar(labels, summ["px_max"], color=cols, width=0.6)
    annotate_bars(a2, bars, [f"{int(p)}/32" for p in summ["perfectas"]])
    a2.set_ylabel("error máximo (px)"); a2.set_title("Resultado final — gana Adam")
    top = finish_bars(a2, summ["px_max"]); mark_winner(a2, win, top)
    a2.axhline(1, color=SECONDARY, ls="--", lw=1, alpha=.7)
    fig.suptitle("Paso 2 · Optimizador: Adam llega a 0 px donde SGD se estanca")
    save(fig, "fig_exp2_optimizador.png")


# ----------------------------- exp3 · learning rate -----------------------------
def fig_exp3():
    cur = _csv("exp3_curves.csv")
    summ = _csv("exp3_summary.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    palette = {"lr=0.0003": SECONDARY, "lr=0.01": ACCENT, "lr=0.3": WARN}
    for v, g in cur.groupby("variante"):
        a1.plot(g["epoch"], best_so_far(g), label=v, color=palette.get(v, GREY), lw=2)
    a1.set_yscale("log"); a1.set_xlabel("época"); a1.set_ylabel("mejor train loss acumulada (BCE, log)")
    a1.set_title("Convergencia por learning rate (best-so-far)"); a1.legend()

    labels = [f"{lr}" for lr in summ["lr"]]
    win = int(summ["px_max"].values.argmin())
    cols = bar_colors(summ["px_max"], win, fail_pred=lambda v: v > 1)
    bars = a2.bar(labels, summ["px_max"], color=cols, width=0.6)
    annotate_bars(a2, bars, [f"{int(p)}/32" for p in summ["perfectas"]])
    a2.set_xlabel("learning rate"); a2.set_ylabel("error máximo (px)")
    a2.set_title("Resultado final — 0.01 es el punto justo")
    top = finish_bars(a2, summ["px_max"]); mark_winner(a2, win, top)
    a2.axhline(1, color=SECONDARY, ls="--", lw=1, alpha=.7)
    fig.suptitle("Paso 3 · Learning rate: chico = lento, grande = no aprende")
    save(fig, "fig_exp3_learning_rate.png")


# ----------------------------- exp4 · arquitectura -----------------------------
def fig_exp4():
    summ = _csv("exp4_summary.csv")
    summ["hidden"] = summ["hidden"].astype(str)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    # ganador = mínimos params con px_max<=1
    ok = summ[summ["px_max"] <= 1]
    win_label = ok.sort_values("n_params").iloc[0]["hidden"]
    win = int(summ.index[summ["hidden"] == win_label][0])
    cols = bar_colors(summ["px_max"], win, fail_pred=lambda v: v > 1)
    bars = ax.bar(summ["hidden"], summ["px_max"], color=cols, width=0.6)
    annotate_bars(ax, bars, [f"{int(p)}/32\n{int(n)}w" for p, n in zip(summ["perfectas"], summ["n_params"])])
    ax.set_xlabel("capa oculta (por lado)"); ax.set_ylabel("error máximo (px)")
    ax.set_title("Paso 4 · Capacidad: 20 es el mínimo que da 0 px (30 / 20+20 empatan → parsimonia)")
    top = finish_bars(ax, summ["px_max"]); mark_winner(ax, win, top)
    ax.axhline(1, color=SECONDARY, ls="--", lw=1, alpha=.7)
    save(fig, "fig_exp4_arquitectura.png")


# ----------------------------- exp5 · latente -----------------------------
def fig_exp5():
    summ = _csv("exp5_summary.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = [str(L) for L in summ["latente"]]
    win = int(summ.index[summ["latente"] == 2][0])
    # px_max
    cols = bar_colors(summ["px_max"], win, fail_pred=lambda v: v > 1)
    b1 = a1.bar(labels, summ["px_max"], color=cols, width=0.6)
    top = finish_bars(a1, summ["px_max"]); mark_winner(a1, win, top)
    a1.axhline(1, color=SECONDARY, ls="--", lw=1, alpha=.7)
    a1.set_xlabel("dimensión latente"); a1.set_ylabel("error máximo (px)")
    a1.set_title("Error máximo")
    # perfectas / 32 (el "subconjunto")
    colsp = bar_colors(summ["perfectas"], win)
    b2 = a2.bar(labels, summ["perfectas"], color=colsp, width=0.6)
    annotate_bars(a2, b2, [f"{int(p)}/32" for p in summ["perfectas"]])
    a2.axhline(32, color=ACCENT, ls=":", lw=1, alpha=.6)
    a2.set_xlabel("dimensión latente"); a2.set_ylabel("letras perfectas")
    a2.set_title("Subconjunto aprendido (1D no alcanza)")
    fig.suptitle("Paso 5 · Latente: 2D es el mínimo que aprende las 32 (la consigna pide 2)")
    save(fig, "fig_exp5_latente.png")


# ----------------------------- exp6 · activación -----------------------------
def fig_exp6():
    cur = _csv("exp6_curves.csv")
    summ = _csv("exp6_summary.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    palette = {"tanh": PRIMARY, "relu": ACCENT, "sigmoid": SECONDARY}
    for v, g in cur.groupby("variante"):
        ax.plot(g["epoch"], best_so_far(g), label=v, color=palette.get(v, GREY), lw=2)
    ax.set_yscale("log"); ax.set_xlabel("época"); ax.set_ylabel("mejor train loss acumulada (BCE, log)")
    convs = {r["act_hidden"]: int(r["epoca_conv"]) for _, r in summ.iterrows()}
    txt = " · ".join(f"{k}: conv@{v}" for k, v in convs.items())
    ax.set_title("Paso 6 · Activación: las tres llegan a 0 px, cambia la velocidad (best-so-far)")
    ax.annotate(txt, (0.5, -0.22), xycoords="axes fraction", ha="center", fontsize=10, color=SECONDARY)
    ax.legend()
    save(fig, "fig_exp6_activacion.png")


# ----------------------------- exp7 · loss -----------------------------
def fig_exp7():
    summ = _csv("exp7_summary.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    labels = [l.upper() for l in summ["loss"]]
    win = int(summ["px_max"].values.argmin())
    cols = bar_colors(summ["px_max"], win, fail_pred=lambda v: v > 1)
    bars = ax.bar(labels, summ["px_max"], color=cols, width=0.55)
    annotate_bars(ax, bars, [f"{int(p)}/32 ok" for p in summ["perfectas"]])
    top = finish_bars(ax, summ["px_max"]); mark_winner(ax, win, top)
    ax.axhline(1, color=SECONDARY, ls="--", lw=1, alpha=.7)
    ax.set_ylabel("error máximo (px)")
    ax.set_title("Paso 7 · Loss: BCE llega a 0 px, MSE deja píxeles")
    save(fig, "fig_exp7_loss.png")


# ----------------------------- camino (resumen) -----------------------------
def fig_camino():
    # px_max de la config ACUMULADA tras cada paso (winner de cada experimento)
    lk = [
        ("1·linealidad", _csv("exp1_summary.csv"), "variante", "no_lineal"),
        ("2·optimizador", _csv("exp2_summary.csv"), "optimizer", "adam"),
        ("3·lr", _csv("exp3_summary.csv"), "lr", 0.01),
        ("4·arquitectura", _csv("exp4_summary.csv"), "hidden", "20"),
        ("5·latente", _csv("exp5_summary.csv"), "latente", 2),
        ("6·activación", _csv("exp6_summary.csv"), "act_hidden", "tanh"),
        ("7·loss", _csv("exp7_summary.csv"), "loss", "bce"),
    ]
    steps, pxmax = [], []
    for name, df, col, val in lk:
        row = df[df[col].astype(str) == str(val)].iloc[0]
        steps.append(name); pxmax.append(int(row["px_max"]))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = range(len(steps))
    ax.plot(x, pxmax, "-o", color=PRIMARY, lw=2.5, ms=9, zorder=3)
    for xi, y in zip(x, pxmax):
        ax.annotate(f"{y}px", (xi, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=11, fontweight="bold",
                    color=(ACCENT if y <= 1 else WARN))
    ax.axhline(1, color=SECONDARY, ls="--", lw=1, alpha=.7)
    ax.annotate("objetivo ≤1px", (len(steps) - 1, 1.6), color=SECONDARY, fontsize=9, ha="right")
    ax.set_xticks(list(x)); ax.set_xticklabels(steps, rotation=20, ha="right")
    ax.set_ylabel("error máximo de la config acumulada (px)")
    ax.set_title("El camino: error máximo tras fijar cada parámetro (cae a 0 al adoptar Adam)")
    save(fig, "fig_camino.png")


# ----------------------------- exp2 · letras reconstruidas -----------------------------
def fig_exp2_letras():
    """Grilla 4 × 32: original + reconstrucción de cada optimizador (reentrenamiento)."""
    import numpy as np
    from tp5lib.fonts import load_font, D
    from tp5lib.autoencoder import build_ae, reconstruct_binary
    from mlp.optimizers import SGD, Momentum, Adam

    FONT = HERE.parent / "font.h"
    X = load_font(FONT)
    SEED = 0
    BASE = dict(hidden=[20], latent=2, act_hidden="tanh", act_latent="identity",
                act_out="sigmoid", loss="bce", epochs=6000)
    ADAM_EPS = 1e-4

    variants = [
        ("Original", None, None),
        ("SGD (0.5)",      SGD(0.5),           "5px"),
        ("Momentum (0.1)", Momentum(0.1),       "1px"),
        ("Adam (0.01)",    Adam(0.01, eps=ADAM_EPS), "0px"),
    ]

    recons = []
    for label, opt, px in variants:
        if opt is None:
            recons.append(X)
        else:
            np.random.seed(SEED)
            ae = build_ae(D, hidden=tuple(BASE["hidden"]), latent=BASE["latent"],
                          act_hidden=BASE["act_hidden"], act_latent=BASE["act_latent"],
                          act_out=BASE["act_out"], loss=BASE["loss"],
                          optimizer=opt, seed=SEED)
            print(f"  entrenando {label} …")
            ae.fit(X, X, X, X, epochs=BASE["epochs"], batch_size=32)
            recons.append(reconstruct_binary(ae, X))

    N = X.shape[0]  # 32
    H, W = 7, 5
    ROWS = len(variants)
    GAP = 1  # pixels de separación entre letras

    fig, axes = plt.subplots(ROWS, N, figsize=(N * (W + GAP) * 0.13, ROWS * (H + GAP) * 0.25))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    row_labels = [v[0] + (f"  [{v[2]}]" if v[2] else "") for v in variants]

    for r, (label, rec) in enumerate(zip(row_labels, recons)):
        for c in range(N):
            ax = axes[r][c]
            ax.imshow(rec[c].reshape(H, W), cmap="binary", vmin=0, vmax=1,
                      interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[r][0].set_ylabel(label, fontsize=8, rotation=0, labelpad=60,
                              ha="right", va="center")

    fig.suptitle("Exp 2 · Optimizador — reconstrucción de las 32 letras", fontsize=11, y=1.01)
    save(fig, "fig_exp2_letras.png")


# ============================================================
# Figuras heredadas de ej1a_autoencoder (dataset + ganador)
# Para regenerar champion_1a.npz / champion_latent.npz:
#   python3 train_champion.py
# ============================================================

def _legacy_imports():
    import numpy as np
    from tp5lib.fonts import load_font, LABELS, H, W, hamming_matrix, pixel_density
    from tp5lib.autoencoder import decode
    from mlp.network import MLP
    FONT = HERE.parent / "font.h"
    X = load_font(FONT)
    N = len(X)
    curves = np.load(RES / "curves_1a.npz")
    return np, X, N, LABELS, H, W, hamming_matrix, pixel_density, decode, MLP, curves


def fig_e0a():
    np, X, N, LABELS, H, W, *_ = _legacy_imports()
    fig, ax = plt.subplots(4, 8, figsize=(12, 7))
    for j in range(N):
        a = ax[j // 8, j % 8]
        a.imshow(X[j].reshape(H, W)); a.set_title(LABELS[j], fontsize=13); a.axis("off")
    fig.suptitle("E0 · Dataset font.h: 32 caracteres de 7×5 (binarios)")
    fig.subplots_adjust(wspace=0.4, hspace=0.5, left=0.03, right=0.97)
    fig.savefig(FIGS / "fig_e0a_dataset_letters.png", pad_inches=0.3)
    plt.close(fig); print("  -> fig_e0a_dataset_letters.png")


def fig_e0b():
    np, X, N, LABELS, H, W, hamming_matrix, pixel_density, *_ = _legacy_imports()
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


def fig_e0c():
    np, X, N, LABELS, H, W, hamming_matrix, *_ = _legacy_imports()
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


def fig_e1():
    np, X, N, LABELS, H, W, *_ = _legacy_imports()
    df = _csv("e1_linear_vs_pca.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(range(len(df)), df["px_mean"], color=[PRIMARY, SECONDARY, "#9CA3AF"])
    ax.set_ylim(0, max(df["px_mean"]) * 1.4)
    for i, (m, mx) in enumerate(zip(df["px_mean"], df["px_max"])):
        ax.text(i, m + 0.2, f"prom {m:.2f}\nmax {mx}", ha="center", fontsize=11)
    ax.set_xticks(range(len(df))); ax.set_xticklabels(df["modelo"], fontsize=11)
    ax.set_ylabel("px incorrectos por letra (promedio)")
    ax.set_title("E1 · AE lineal ≡ PCA (~7 px); el no-lineal llega a 0", pad=14)
    save(fig, "fig_e1_linear_vs_pca.png")


def fig_e1_pca_pixel_error():
    np, X, N, LABELS, H, W, *_ = _legacy_imports()
    mu = X.mean(0); Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xr = (Xc @ Vt[:2].T) @ Vt[:2] + mu
    err = np.abs((Xr > 0.5).astype(float) - X).mean(0).reshape(H, W)
    fig, ax = plt.subplots(figsize=(5.2, 6.7))
    im = ax.imshow(err, cmap="magma", vmin=0); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("E1 · En qué parte del glifo (7×5) se equivoca PCA\n"
                 "cada celda = un píxel (no una letra), promediado sobre las 32", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.05, label="frac. de las 32 letras con ese píxel mal")
    save(fig, "fig_e1_pca_pixel_error.png")


def fig_e2():
    df = _csv("e2_latent_sweep.csv")
    n = int(df["n_seeds"].iloc[0]) if "n_seeds" in df else 1
    fig, ax = plt.subplots(figsize=(9, 5.5))
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


def fig_e3():
    np, X, N, LABELS, H, W, *rest = _legacy_imports()
    curves = rest[-1]
    df = _csv("e3_architecture.csv")
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


def fig_e4():
    *_, curves = _legacy_imports()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for key, lab, c in [("e4_SGD(0.5)", "SGD(0.5)", "#9CA3AF"),
                        ("e4_Momentum(0.1)", "Momentum(0.1)", SECONDARY),
                        ("e4_Adam(0.01)", "Adam(0.01)", PRIMARY)]:
        ax.plot(curves[key], label=lab, color=c, lw=2)
    ax.set_yscale("log"); ax.set_xlabel("época"); ax.set_ylabel("train loss (BCE, log)")
    ax.set_title("E4 · Optimizadores: Adam converge ~100× más bajo que SGD")
    ax.legend()
    save(fig, "fig_e4_optimizers.png")


def fig_e5():
    *_, curves = _legacy_imports()
    df = _csv("e5_lr.csv")
    n = int(df["n_seeds"].iloc[0]) if "n_seeds" in df else 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for key, lab, c in [("e5_lr_0.0003", "lr=0.0003 (lento)", "#2563EB"),
                        ("e5_lr_0.01", "lr=0.01 (justo)", ACCENT),
                        ("e5_lr_0.3", "lr=0.3 (no aprende)", PRIMARY)]:
        a1.plot(curves[key], label=lab, color=c, lw=2)
    a1.set_yscale("log"); a1.set_xlabel("época"); a1.set_ylabel("train loss (BCE, log)")
    a1.set_title("convergencia (semilla 0)"); a1.legend()
    a2.bar(range(len(df)), df["px_max_mean"], yerr=df["px_max_std"], color=PRIMARY,
           capsize=5, error_kw=dict(ecolor="#374151", lw=1.5))
    a2.set_xticks(range(len(df))); a2.set_xticklabels([f"lr={lr}" for lr in df["lr"]], fontsize=11)
    a2.set_ylabel("px máximo (media ± desvío)"); a2.axhline(1, color="gray", ls=":", label="objetivo ≤1px")
    a2.set_title(f"px máximo vs learning rate (n={n} semillas)"); a2.legend()
    fig.suptitle("E5 · Learning rate: muy bajo es lento; muy alto (0.3) queda atascado arriba; 0.01 llega a 0 px")
    save(fig, "fig_e5_lr.png")


def fig_e6():
    *_, curves = _legacy_imports()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for key, lab in [("e6_tanh", "tanh"), ("e6_relu", "relu"), ("e6_sigmoid", "sigmoid")]:
        ax.plot(curves[key], label=lab, lw=2)
    ax.set_yscale("log"); ax.set_xlabel("época"); ax.set_ylabel("train loss (BCE, log)")
    ax.set_title("E6 · Activación oculta: las tres llegan a 0 px, pero difieren en velocidad")
    ax.legend()
    save(fig, "fig_e6_activation.png")


def fig_e7():
    df = _csv("e7_loss.csv")
    *_, curves = _legacy_imports()
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


def fig_e8():
    import numpy as np
    from tp5lib.fonts import load_font, LABELS, H, W, hamming_matrix
    from tp5lib.autoencoder import decode
    from mlp.network import MLP
    FONT = HERE.parent / "font.h"
    X = load_font(FONT); N = len(X)

    ae = MLP.load(RES / "champion_1a.npz")
    Z = np.load(RES / "champion_latent.npz")["Z"]
    rec = (ae.predict_proba(X) > 0.5).astype(float)

    # E8a — reconstrucciones (orig fila par / recon fila impar)
    fig, ax = plt.subplots(8, 8, figsize=(11, 11))
    for j in range(N):
        blk, col = j // 8, j % 8
        ax[blk*2, col].imshow(X[j].reshape(H, W)); ax[blk*2, col].set_title(LABELS[j], fontsize=11); ax[blk*2, col].axis("off")
        ax[blk*2+1, col].imshow(rec[j].reshape(H, W)); ax[blk*2+1, col].axis("off")
    fig.suptitle("E8 · Ganador 35-20-2-20-35: Original (sup.) vs Reconstrucción (inf.) — 0 px error")
    save(fig, "fig_e8a_reconstructions.png")

    # E8b — scatter latente con inset de zoom
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.scatter(Z[:, 0], Z[:, 1], s=20, c=PRIMARY, zorder=3)
    for j in range(N):
        ax.annotate(LABELS[j], (Z[j, 0], Z[j, 1]), fontsize=12, ha="center", va="center", color=SECONDARY, weight="bold")
    ax.set_title("E8 · Espacio latente 2D: las 32 letras"); ax.set_xlabel("z1"); ax.set_ylabel("z2")
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
    cell_bits = np.zeros((NG, NG, H * W))
    for r, yy in enumerate(gy):
        for c2, xx in enumerate(gx):
            dec = decode(ae, [[xx, yy]])[0]
            canvas[r * H:(r+1)*H, c2*W:(c2+1)*W] = dec.reshape(H, W)
            cell_bits[r, c2] = (dec > 0.5).astype(float)
    fig, ax = plt.subplots(figsize=(9.5, 9.8)); ax.imshow(canvas); ax.axis("off")
    Xbits = (X > 0.5).astype(float); THR_PX = 1; n_known = 0
    for r in range(NG):
        for c2 in range(NG):
            ham = np.abs(cell_bits[r, c2][None, :] - Xbits).sum(1).min()
            if ham <= THR_PX:
                rect = plt.Rectangle((c2*W - 0.5, r*H - 0.5), W, H,
                                     fill=False, edgecolor="#d62728", lw=2.0, zorder=5)
                ax.add_patch(rect); n_known += 1
    ax.set_title("E8 · Generación: barrido del espacio latente (cada celda = letra decodificada)")
    ax.text(0.5, -0.045,
            f"Recuadro rojo = reproduce (≤{THR_PX} px) una de las 32 letras reales ({n_known}/{NG*NG}). "
            f"El resto son combinaciones continuas del decoder.",
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


def fig_e8e():
    from scipy.stats import spearmanr
    import numpy as np
    from tp5lib.fonts import load_font, LABELS, hamming_matrix
    FONT = HERE.parent / "font.h"
    X = load_font(FONT); N = len(X)
    Z = np.load(RES / "champion_latent.npz")["Z"]
    hm = hamming_matrix(X).astype(float)
    diff = Z[:, None, :] - Z[None, :, :]
    lm = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(N, k=1)
    rho, _ = spearmanr(hm[iu], lm[iu])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, M, t in [(a1, hm, "Distancia en píxeles (Hamming)"),
                     (a2, lm, "Distancia en el mapa latente 2D")]:
        im = ax.imshow(M, cmap="magma"); ax.grid(False)
        ax.set_xticks(range(N)); ax.set_xticklabels(LABELS, fontsize=6)
        ax.set_yticks(range(N)); ax.set_yticklabels(LABELS, fontsize=6)
        ax.set_title(t); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"E8 · ¿El latente preservó el parecido entre letras?   Spearman ρ = {rho:.2f}")
    save(fig, "fig_e8e_latent_vs_hamming.png")


if __name__ == "__main__":
    import sys as _sys
    only_letras = "--letras" in _sys.argv
    only_legacy = "--legacy" in _sys.argv
    print("Generando figuras en figs/ ...")
    if only_letras:
        fig_exp2_letras()
    elif only_legacy:
        fig_e0a(); fig_e0b(); fig_e0c()
        fig_e1(); fig_e1_pca_pixel_error(); fig_e2(); fig_e3(); fig_e4(); fig_e5(); fig_e6(); fig_e7()
        fig_e8(); fig_e8e()
    else:
        fig_exp1(); fig_exp2(); fig_exp3(); fig_exp4(); fig_exp5(); fig_exp6(); fig_exp7()
        fig_camino()
        fig_exp2_letras()
        fig_e0a(); fig_e0b(); fig_e0c()
        fig_e1(); fig_e1_pca_pixel_error(); fig_e2(); fig_e3(); fig_e4(); fig_e5(); fig_e6(); fig_e7()
        fig_e8(); fig_e8e()
    print("OK")
