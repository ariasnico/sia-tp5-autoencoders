"""TP5 VAE — figuras E12-E18 + atlas del latente, desde los modelos guardados. Estilo central.

Para E12/E14 se ELIGE (y se documenta) una semilla de sampleo que cubra las 5 clases, sólo para
ilustrar la diversidad; clasificando cada z~N(0,I) por el centroide latente de clase más cercano.
E18 es el control honesto: usa una semilla NO elegida (la 0, fija) y reporta la fracción PROMEDIO
de clases que aparece al samplear N(0,I) sobre 200 semillas, sin cherry-picking.
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
    fig.suptitle(f"E12 · Muestras z~N(0,I) por β (semilla {SEED_SAMPLE}, elegida para cubrir las 5 clases).\n"
                 f"β=0 → ruido; con KL → muestras reconocibles (variaciones de los 5 prototipos)")
    save(fig, "fig_e12_beta_sampling.png")


def e13():
    mu = load_vae(1.0).encode_mean(X)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    for ci, n in enumerate(names):
        m = y == ci; ax.scatter(mu[m, 0], mu[m, 1], s=14, alpha=0.65, label=n, color=PALETTE[ci])
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(th), np.sin(th), "k--", alpha=0.4); ax.plot(2 * np.cos(th), 2 * np.sin(th), "k--", alpha=0.25)
    ax.set_title("E13 · Latente del VAE (β=1): el KL empuja cada q(z|x) a N(0,I),\n"
                 "pero el agregado conserva 5 cúmulos de clase (no es una sola nube)")
    ax.legend(); ax.set_xlabel("z1"); ax.set_ylabel("z2")
    save(fig, "fig_e13_latent.png")


def e14():
    vae = load_vae(1.0)
    samp = np.concatenate([np.random.default_rng(3 + ci).choice(np.where(y == ci)[0], 2, replace=False)
                           for ci in range(NC)])  # 2 por clase -> muestra las 5
    rec = vae.reconstruct(X[samp])
    gen = vae.generate(np.random.default_rng(SEED_SAMPLE).standard_normal((10, 2)))
    fig, axes = plt.subplots(3, 10, figsize=(12, 4.2))
    rows = [(0, "original", X[samp]), (1, "reconstruido", rec), (2, "generado\n(nuevo, no copia)", gen)]
    for r, lab, imgs in rows:
        for ci in range(10):
            ax = axes[r, ci]; ax.imshow(imgs[ci].reshape(SZ, SZ), vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        axes[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=12)
    fig.suptitle(f"E14 · VAE β=1: reconstrucción y muestras nuevas (no copias) desde N(0,I) "
                 f"(semilla {SEED_SAMPLE}, elegida para mostrar las 5 clases; req. 2c)")
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
    # Un panel por modelo, grande y legible a tamaño proyectado (fuentes >=14).
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    th = np.linspace(0, 2 * np.pi, 200)
    for ax, beta, tit in [(axes[0], 0.0, "AE común (β=0): cada q(z|x) NO está empujada a N(0,I)\nlatente disperso (escala ±decenas)"),
                          (axes[1], 1.0, "VAE (β=1): KL empuja cada q(z|x) a N(0,I)\nlatente compacto, pero conserva 5 cúmulos")]:
        mu = load_vae(beta).encode_mean(X)
        for ci, n in enumerate(names):
            m = y == ci
            ax.scatter(mu[m, 0], mu[m, 1], s=55, alpha=0.7, label=n, color=PALETTE[ci],
                       edgecolors="white", linewidths=0.4)
        ax.plot(np.cos(th), np.sin(th), "k--", alpha=0.45, lw=2)
        ax.plot(2 * np.cos(th), 2 * np.sin(th), "k--", alpha=0.3, lw=2)
        ax.set_title(tit, fontsize=18)
        ax.set_xlabel("z1", fontsize=16); ax.set_ylabel("z2", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=15, loc="best", framealpha=0.9, markerscale=1.4)
    fig.suptitle("E16 · Por qué el VAE genera y el AE no: samplear N(0,I) sólo cae en zona "
                 "entrenada cuando el KL ha empujado el latente hacia N(0,I)", fontsize=19)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig_e16_ae_vs_vae.png")


def class_fraction_over_seeds(centroids, ns=10, n_seeds=200):
    """Sobre n_seeds semillas, samplea ns~N(0,I), clasifica cada z por centroide más
    cercano y promedia la fracción de las 5 clases que aparece. Sin elegir semilla."""
    covered = np.zeros(n_seeds)
    per_class_total = np.zeros(NC)
    for s in range(n_seeds):
        z = np.random.default_rng(s).standard_normal((ns, 2))
        cls = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(-1).argmin(1)
        counts = np.bincount(cls, minlength=NC)
        covered[s] = (counts > 0).sum() / NC
        per_class_total += counts
    frac_mean = covered.mean()
    per_class_share = per_class_total / per_class_total.sum()
    return frac_mean, covered, per_class_share


def e18():
    """Sampleo HONESTO: grilla z~N(0,I) con semilla 0 fija (no elegida), + métrica de
    cobertura promedio de clases sobre 200 semillas."""
    vae = load_vae(1.0)
    cent = class_centroids(vae)
    frac_mean, covered, share = class_fraction_over_seeds(cent, ns=10, n_seeds=200)
    print(f"  [e18] cobertura media de clases (200 semillas, 10 muestras c/u): "
          f"{frac_mean*100:.1f}% de las {NC} clases por sampleo "
          f"(= {frac_mean*NC:.2f}/{NC} clases en promedio)")
    print(f"  [e18] reparto de clases al samplear N(0,I): "
          f"{ {names[i]: round(float(share[i]),3) for i in range(NC)} }")

    # grilla generada con semilla 0 fija, sin elegir
    z = np.random.default_rng(0).standard_normal((20, 2))
    gen = vae.generate(z)
    fig, axes = plt.subplots(2, 10, figsize=(13, 3.4))
    for k in range(20):
        ax = axes[k // 10, k % 10]
        ax.imshow(gen[k].reshape(SZ, SZ), vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(
        f"E18 · Sampleo honesto: 20 muestras z~N(0,I) con semilla 0 FIJA (no elegida).\n"
        f"En promedio aparece {frac_mean*100:.0f}% de las {NC} clases por tirada "
        f"({frac_mean*NC:.1f}/{NC}, sobre 200 semillas) — el sampleo no cubre las 5 siempre.",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig_e18_sampleo_honesto.png")
    return frac_mean, share


def atlas():
    vae = load_vae(1.0)
    g = np.linspace(-2.5, 2.5, 13)
    canvas = np.ones((13 * SZ, 13 * SZ))
    for r, yy in enumerate(g[::-1]):
        for c, xx in enumerate(g):
            canvas[r * SZ:(r + 1) * SZ, c * SZ:(c + 1) * SZ] = vae.generate([[xx, yy]])[0].reshape(SZ, SZ)
    fig, ax = plt.subplots(figsize=(9, 9)); ax.imshow(canvas, vmin=0, vmax=1); ax.axis("off")
    ax.set_title("Atlas del latente del VAE (β=1): grilla z∈[-2.5, 2.5]² decodificada")
    save(fig, "fig_e17_vae_atlas.png")


if __name__ == "__main__":
    print("Generando figuras VAE:")
    e12(); e13(); e14(); e15(); e16(); e18(); atlas()
    print("OK figuras VAE en", FIGS)
