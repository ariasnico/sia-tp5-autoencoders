"""TP5 VAE — experimentos E12-E16 sobre emojis (OpenMoji color-alpha, 5 clases, 20×20).

E12  barrido de beta {0, 0.5, 1, 4}  -> error de reconstrucción + estructura del latente
E15  curvas recon-loss vs KL-loss por época (por beta)
E13/E14/E16 usan los modelos guardados (scatter del latente, recon+generación, AE vs VAE)

Reusa tp5lib.vae_core.VAE (backprop verificado por gradient check, NO se modifica). Semillas fijas.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
RES = HERE / "results"; RES.mkdir(exist_ok=True)

from tp5lib.vae_core import VAE     # noqa: E402
from dataset import make_dataset    # noqa: E402

SEED = 0
SZ = 20; D = SZ * SZ
He, Z, Hd = 128, 2, 128
EPOCHS, BS = 3500, 128

X, y, names, bases = make_dataset(SZ=SZ, per_class=140, seed=SEED, variant="color")
print("dataset:", X.shape, "| clases:", names)
np.save(RES / "emoji_y.npy", y)
np.save(RES / "emoji_names.npy", np.array(names))
np.save(RES / "emoji_bases.npy", np.array([bases[n] for n in names]))


def train_vae(beta, epochs=EPOCHS, bs=BS, seed=SEED, log_every=5):
    """Entrena un VAE; registra recon y KL (sobre todo X, eps=0) cada log_every épocas."""
    vae = VAE(D, He, Z, Hd, beta=beta, lr=1e-3, seed=seed)
    idx = np.arange(len(X)); r = np.random.default_rng(seed)
    eps0 = np.zeros((len(X), Z)); rec_c, kl_c = [], []
    for ep in range(epochs):
        r.shuffle(idx)
        for b in range(0, len(X), bs):
            vae.step(X[idx[b:b + bs]])
        if ep % log_every == 0:
            _, _, parts = vae.loss_and_grads(X, eps=eps0)
            rec_c.append(parts["recon"]); kl_c.append(parts["kl"])
    return vae, np.array(rec_c), np.array(kl_c)


betas = [0.0, 0.5, 1.0, 4.0]
t0 = time.time(); rows = []; curves = {}
for beta in betas:
    vae, rec_c, kl_c = train_vae(beta)
    rec = vae.reconstruct(X); mu = vae.encode_mean(X)
    recon_px = float(np.mean((rec > 0.5) != (X > 0.5)))
    rows.append({"beta": beta, "recon_px_frac": round(recon_px, 4),
                 "lat_absmean": round(float(np.abs(mu.mean(0)).mean()), 3),
                 "lat_std": round(float(mu.std(0).mean()), 3),
                 "recon_final": round(float(rec_c[-1]), 3), "kl_final": round(float(kl_c[-1]), 3)})
    curves[f"rec_{beta}"] = rec_c; curves[f"kl_{beta}"] = kl_c
    np.savez(RES / f"vae_beta_{beta}.npz", beta=beta, dims=np.array([D, He, Z, Hd]),
             **{k: vae.P[k] for k in vae.P})
    print(f"  beta={beta}: recon_px={recon_px:.4f} | latente |mean|={np.abs(mu.mean(0)).mean():.2f} "
          f"std={mu.std(0).mean():.2f}")

pd.DataFrame(rows).to_csv(RES / "e12_beta_sweep.csv", index=False)
np.savez(RES / "vae_curves.npz", **curves)
(RES / "config_used.json").write_text(json.dumps(
    {"seed": SEED, "SZ": SZ, "dims": dict(D=D, He=He, Z=Z, Hd=Hd), "epochs": EPOCHS, "batch": BS,
     "dataset": "OpenMoji color-alpha, 5 clases (corazon/estrella/gota/luna/rayo), augment seedeado",
     "betas": betas}, indent=2))
print(f"\nOK VAE en {time.time() - t0:.1f}s")
print(pd.DataFrame(rows).to_string(index=False))
