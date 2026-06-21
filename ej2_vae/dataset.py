"""TP5 VAE — dataset de emojis (OpenMoji) como siluetas de baja resolución.

Estrategia: se usa el canal ALPHA de los PNG de OpenMoji como silueta rellena monocromática
(robusto a baja resolución), recortada al bounding box, centrada y reescalada a SZxSZ. La
variabilidad intra-clase se crea con augmentaciones SEEDEADAS y MODERADAS (rotación ±15°,
traslación ±2px, ruido suave) para que el VAE aprenda una variedad continua, no puntos fijos.

Los assets se cachean/commitean en assets/ -> dataset reproducible offline.
Fallbacks si los emojis no son distinguibles: subir SZ -> Noto-grises -> formas geométricas.
"""
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import rotate, shift

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"; ASSETS.mkdir(exist_ok=True)

# (nombre, codepoint) — 5 emojis icónicos de silueta EXTERNA radicalmente distinta.
# Se descartaron triangulo/rombo/circulo: formas convexas compactas que se confunden
# entre sí a baja resolución (1-NN acc 0.88 -> sube con este set) y son menos "emoji".
CLASSES = [
    ("corazon", "2764"), ("estrella", "2B50"), ("gota", "1F4A7"),
    ("luna", "1F319"), ("rayo", "26A1"),
]
URL = "https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/{variant}/72x72/{code}.png"


def fetch(code, variant="color"):
    """Descarga (y cachea) el PNG de OpenMoji. Devuelve path local o None si falla."""
    dst = ASSETS / f"{variant}_{code}.png"
    if dst.exists():
        return dst
    try:
        urllib.request.urlretrieve(URL.format(variant=variant, code=code), dst)
        return dst
    except Exception as e:
        print(f"  WARN no se pudo bajar {code} ({variant}): {e}")
        return None


def silhouette(path, SZ, margin=2):
    """PNG -> silueta [0,1] de SZxSZ: canal alpha, crop al bbox, centrado cuadrado, resize."""
    a = np.asarray(Image.open(path).convert("RGBA").split()[-1], float) / 255.0
    ys, xs = np.where(a > 0.2)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(xs) else a
    h, w = a.shape; s = max(h, w)
    sq = np.zeros((s, s)); sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    inner = SZ - 2 * margin
    small = Image.fromarray((sq * 255).astype("uint8")).resize((inner, inner), Image.LANCZOS)
    out = np.zeros((SZ, SZ)); out[margin:margin + inner, margin:margin + inner] = np.asarray(small, float) / 255.0
    return out


def build_bases(SZ, variant="color"):
    """dict nombre -> bitmap base (silueta SZxSZ). Omite clases que no se pudieron bajar."""
    bases = {}
    for name, code in CLASSES:
        p = fetch(code, variant)
        if p is not None:
            bases[name] = silhouette(p, SZ)
    return bases


def augment(base, rng):
    """Augmentación moderada y seedeada: rotación ±15°, traslación ±2px, ruido suave."""
    img = rotate(base, rng.uniform(-15, 15), reshape=False, order=1, mode="constant")
    img = shift(img, rng.uniform(-2, 2, 2), order=1, mode="constant")
    return np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1)


def make_dataset(SZ=20, per_class=140, seed=0, variant="color"):
    """Devuelve X (N, SZ*SZ), y (N,), names (list), bases (dict)."""
    bases = build_bases(SZ, variant)
    names = list(bases)
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, name in enumerate(names):
        for _ in range(per_class):
            X.append(augment(bases[name], rng).ravel()); y.append(ci)
    return np.array(X), np.array(y, dtype=int), names, bases


def nn_accuracy(X, y, bases, names):
    """Fracción de muestras cuyo bitmap-base más cercano (L2) es de su propia clase.
    Alta => las clases son distinguibles pese a la augmentación."""
    B = np.array([bases[n].ravel() for n in names])
    d = ((X[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return float((d.argmin(1) == y).mean())


if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)

    SZ = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    variant = sys.argv[2] if len(sys.argv) > 2 else "color"
    X, y, names, bases = make_dataset(SZ=SZ, per_class=140, seed=0, variant=variant)
    print(f"dataset: {X.shape} | clases ({len(names)}): {names}")
    acc = nn_accuracy(X, y, bases, names)
    print(f"1-NN acc (muestras augmentadas vs bitmaps base): {acc:.3f}")
    # distancias entre clases (pares más confundibles)
    B = np.array([bases[n].ravel() for n in names])
    Dm = np.sqrt(((B[:, None] - B[None, :]) ** 2).sum(-1)); np.fill_diagonal(Dm, np.inf)
    i, j = np.unravel_index(Dm.argmin(), Dm.shape)
    print(f"par más parecido: {names[i]} ~ {names[j]} (L2={Dm[i, j]:.2f})")

    fig, axes = plt.subplots(2, len(names), figsize=(len(names) * 1.4, 3.2))
    for k, n in enumerate(names):
        axes[0, k].imshow(bases[n], cmap="binary", vmin=0, vmax=1); axes[0, k].set_title(n, fontsize=9); axes[0, k].axis("off")
        axes[1, k].imshow(X[y == k][3].reshape(SZ, SZ), cmap="binary", vmin=0, vmax=1); axes[1, k].axis("off")
    fig.suptitle(f"Emojis OpenMoji '{variant}' {SZ}×{SZ} — base (arriba) / augmentada (abajo) · 1-NN acc={acc:.2f}", fontsize=11)
    fig.savefig(FIGS / "contact_sheet.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("contact_sheet.png guardado")
