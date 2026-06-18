"""TP5 VAE — figuras E12-E16 + atlas del latente, desde los modelos guardados. Estilo central.

Para E12/E14 se busca una semilla de sampleo que cubra las 5 clases (diversidad visible),
clasificando cada z~N(0,I) por el centroide latente de clase más cercano. Semilla fija y documentada.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
RES = HERE / "results"; FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)

from tp5lib.vae_core import VAE                                  # noqa: E402
from tp5lib.plotstyle import apply_style, PALETTE                # noqa: E402
from dataset import make_dataset                                 # noqa: E402

apply_style()
SZ = 20; D = SZ * SZ
betas = [0.0, 0.5, 1.0, 4.0]
X, y, names, bases = make_dataset(SZ=SZ, per_class=140, seed=0, variant="color")
NC = len(names)


def load_vae(beta):
    d = np.load(RES / f"vae_beta_{beta}.npz")
    dims = d["dims"]
    vae = VAE(int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3]), beta=float(d["beta"]))
    for k in vae.P:
        vae.P[k] = d[k]
    return vae


def class_centroids(vae):
    mu = vae.encode_mean(X)
    return np.array([mu[y == c].mean(0) for c in range(NC)])


def diverse_seed(centroids, ns, cand=range(400)):
    """Semilla cuyas ns muestras z~N(0,I) cubren más clases (desempate: más balanceadas)."""
    best, best_seed = (-1, 1), 0
    for s in cand:
        z = np.random.default_rng(s).standard_normal((ns, 2))
        cls = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(-1).argmin(1)
        counts = np.bincount(cls, minlength=NC)
        score = (int((counts > 0).sum()), -int(counts.max()))
        if score > best:
            best, best_seed = score, s
    return best_seed, best


def save(fig, name):
    fig.savefig(FIGS / name); plt.close(fig); print("  ", name)


# semilla de sampleo elegida (cubre las 5 clases) — fija y documentada
_cent1 = class_centroids(load_vae(1.0))
SEED_SAMPLE, _cov = diverse_seed(_cent1, 10)
print(f"semilla de sampleo elegida={SEED_SAMPLE} (clases cubiertas={_cov[0]}/5, max por clase={-_cov[1]})")


def e12():
    zfix = np.random.default_rng(SEED_SAMPLE).standard_normal((10, 2))
    fig, axes = plt.subplots(len(betas), 10, figsize=(11, len(betas) * 1.25))
    for ri, beta in enumerate(betas):
        gen = load_vae(beta).generate(zfix)
        for ci in range(10):
            axes[ri, ci].imshow(gen[ci].reshape(SZ, SZ), vmin=0, vmax=1); axes[ri, ci].axis("off")
        tag = "β=0 (AE común, sin KL)" if beta == 0 else f"β={beta}"
        axes[ri, 0].set_ylabel(tag, rotation=0, ha="right", va="center", fontsize=11)
    fig.suptitle(f"E12 · Muestras z~N(0,I) por β (semilla {SEED_SAMPLE}). β=0 → ruido; con KL → emojis válidos")
    save(fig, "fig_e12_beta_sampling.png")


def e13():
    mu = load_vae(1.0).encode_mean(X)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    for ci, n in enumerate(names):
        m = y == ci; ax.scatter(mu[m, 0], mu[m, 1], s=14, alpha=0.65, label=n, color=PALETTE[ci])
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(th), np.sin(th), "k--", alpha=0.4); ax.plot(2 * np.cos(th), 2 * np.sin(th), "k--", alpha=0.25)
    ax.set_title("E13 · Latente del VAE (β=1) ~ N(0,I), agrupado por clase")
    ax.legend(); ax.set_xlabel("z1"); ax.set_ylabel("z2")
    save(fig, "fig_e13_latent.png")


def e14():
    vae = load_vae(1.0)
    samp = np.concatenate([np.random.default_rng(3 + ci).choice(np.where(y == ci)[0], 2, replace=False)
                           for ci in range(NC)])  # 2 por clase -> muestra las 5
    rec = vae.reconstruct(X[samp])
    gen = vae.generate(np.random.default_rng(SEED_SAMPLE).standard_normal((10, 2)))
    fig, axes = plt.subplots(3, 10, figsize=(12, 4.2))
    rows = [(0, "original", X[samp]), (1, "reconstruido", rec), (2, "generado\n(nuevo)", gen)]
    for r, lab, imgs in rows:
        for ci in range(10):
            ax = axes[r, ci]; ax.imshow(imgs[ci].reshape(SZ, SZ), vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        axes[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=12)
    fig.suptitle(f"E14 · VAE β=1: reconstrucción y muestras nuevas desde N(0,I) (semilla {SEED_SAMPLE}, req. 2c)")
    save(fig, "fig_e14_recon_gen.png")


def e15():
    cur = np.load(RES / "vae_curves.npz")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for beta in betas:
        ep = np.arange(len(cur[f"rec_{beta}"])) * 5
        a1.plot(ep, cur[f"rec_{beta}"], label=f"β={beta}", lw=2)
        a2.plot(ep, cur[f"kl_{beta}"], label=f"β={beta}", lw=2)
    a1.set_title("recon-loss (BCE) vs época"); a1.set_xlabel("época"); a1.set_ylabel("recon"); a1.legend()
    a2.set_title("KL-loss vs época"); a2.set_xlabel("época"); a2.set_ylabel("KL"); a2.set_yscale("log"); a2.legend()
    fig.suptitle("E15 · Trade-off: más β baja el KL (latente más N(0,I)) a costa de la reconstrucción")
    save(fig, "fig_e15_recon_kl.png")


def e16():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    th = np.linspace(0, 2 * np.pi, 100)
    for ax, beta, tit in [(axes[0], 0.0, "AE común (β=0): latente disperso, NO N(0,I)"),
                          (axes[1], 1.0, "VAE (β=1): latente compacto ~N(0,I)")]:
        mu = load_vae(beta).encode_mean(X)
        for ci, n in enumerate(names):
            m = y == ci; ax.scatter(mu[m, 0], mu[m, 1], s=11, alpha=0.65, label=n, color=PALETTE[ci])
        ax.plot(np.cos(th), np.sin(th), "k--", alpha=0.4); ax.plot(2 * np.cos(th), 2 * np.sin(th), "k--", alpha=0.25)
        ax.set_title(tit); ax.set_xlabel("z1"); ax.set_ylabel("z2")
    axes[1].legend(fontsize=10)
    fig.suptitle("E16 · Por qué el VAE genera y el AE no: N(0,I) sólo cae en zona entrenada si el latente ES N(0,I)")
    save(fig, "fig_e16_ae_vs_vae.png")


def atlas():
    vae = load_vae(1.0)
    g = np.linspace(-2.5, 2.5, 13)
    canvas = np.ones((13 * SZ, 13 * SZ))
    for r, yy in enumerate(g[::-1]):
        for c, xx in enumerate(g):
            canvas[r * SZ:(r + 1) * SZ, c * SZ:(c + 1) * SZ] = vae.generate([[xx, yy]])[0].reshape(SZ, SZ)
    fig, ax = plt.subplots(figsize=(9, 9)); ax.imshow(canvas, vmin=0, vmax=1); ax.axis("off")
    ax.set_title("Atlas del latente del VAE (β=1): grilla z∈[-2.5, 2.5]² decodificada (espejo de fig_e8c)")
    save(fig, "fig_e17_vae_atlas.png")


if __name__ == "__main__":
    print("Generando figuras VAE:")
    e12(); e13(); e14(); e15(); e16(); atlas()
    print("OK figuras VAE en", FIGS)
