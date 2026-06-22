"""Entrena el modelo ganador de 1a y guarda pesos + coordenadas latentes.

Genera:
  results/champion_1a.npz     — pesos del AE ganador (MLP.save)
  results/champion_latent.npz — coordenadas Z de las 32 letras (encode)

Config ganadora: 35-20-2-20-35 · tanh/identity/sigmoid · BCE · Adam(0.01, eps=1e-4) · 6000 ep.

Uso:
  python3 train_champion.py
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (                         # noqa: E402
    build_ae, D, X, RESULTS, ADAM_EPS, SEED, INITIAL_CONFIG
)
from tp5lib.autoencoder import encode        # noqa: E402
from mlp.optimizers import Adam              # noqa: E402

WINNER = {**INITIAL_CONFIG, "optimizer": "adam", "lr": 0.01,
          "hidden": [20], "latent": 2, "epochs": 6000}


def main():
    print("Entrenando ganador:", WINNER)
    np.random.seed(SEED)
    opt = Adam(WINNER["lr"], eps=ADAM_EPS)
    ae = build_ae(D, hidden=tuple(WINNER["hidden"]), latent=WINNER["latent"],
                  act_hidden=WINNER["act_hidden"], act_latent=WINNER["act_latent"],
                  act_out=WINNER["act_out"], loss=WINNER["loss"],
                  optimizer=opt, seed=SEED)
    ae.fit(X, X, X, X, epochs=WINNER["epochs"], batch_size=32)

    ae.save(RESULTS / "champion_1a.npz")
    print("  -> champion_1a.npz")

    Z = encode(ae, X)
    np.savez(RESULTS / "champion_latent.npz", Z=Z)
    print("  -> champion_latent.npz")

    from tp5lib.autoencoder import px_err
    e = px_err(ae, X)
    print(f"  px_max={e.max()}  perfectas={int((e==0).sum())}/32")


if __name__ == "__main__":
    main()
