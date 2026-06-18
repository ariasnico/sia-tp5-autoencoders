"""Diagnóstico para DEFENSA — ¿por qué E1 (AE lineal) da ~7.19px pero E3 '()' da ~14px?

Ambos son [35,2,35] SIN capa oculta; difieren en (act_out, loss). Se aíslan las causas y se
prueban varias semillas para distinguir 'limitación estructural' de 'mínimo local'.
Imprime una tabla; el resultado se cita en DEFENSA.md.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tp5lib.fonts import load_font, D            # noqa: E402
from tp5lib.autoencoder import build_ae, px_err  # noqa: E402
from mlp.optimizers import Adam                   # noqa: E402

X = load_font(ROOT / "font.h")


def run(act_latent, act_out, loss, seed, epochs=6000):
    np.random.seed(seed)
    ae = build_ae(D, hidden=(), latent=2, act_latent=act_latent, act_out=act_out,
                  loss=loss, optimizer=Adam(0.01), seed=seed)
    ae.fit(X, X, X, X, epochs=epochs, batch_size=32)
    return px_err(ae, X)


# PCA(2) analítica como referencia del óptimo lineal
mu = X.mean(0); Xc = X - mu
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
e_pca = ((((Xc @ Vt[:2].T) @ Vt[:2] + mu) > 0.5).astype(float) != X).sum(1)
print(f"PCA(2) analítica: px_max={int(e_pca.max())} px_mean={e_pca.mean():.2f}\n")

configs = [
    ("A identity/identity + MSE (=E1 lineal=PCA)", "identity", "identity", "mse"),
    ("B identity/sigmoid  + BCE (=E3 '()')",       "identity", "sigmoid", "bce"),
    ("C identity/sigmoid  + MSE",                  "identity", "sigmoid", "mse"),
]
print(f"{'config':42s} {'px_max(seeds 0,1,2)':22s} {'mean px_max':>12s}")
for name, al, ao, ls in configs:
    maxes = []
    for s in (0, 1, 2):
        e = run(al, ao, ls, s)
        maxes.append(int(e.max()))
    print(f"{name:42s} {str(maxes):22s} {np.mean(maxes):12.1f}")
