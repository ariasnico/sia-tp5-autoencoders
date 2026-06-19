"""TP5 1b — Denoising Autoencoder sobre font.h. Experimentos E9-E11.

E9   ancho de cuello {2,5,10,20}                 -> capacidad de denoising vs compresión
E10  ruido de entrenamiento {.05,.15,.30} × test  -> trade-off robustez/precisión
E11  (guarda el DAE ganador para los tripletes cualitativos en make_figures)

Entrenamiento: entrada con bit-flip FRESCO cada época, target = patrón limpio. Semillas fijas.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
RES = HERE / "results"; RES.mkdir(exist_ok=True)
FONT = ROOT / "font.h"

from tp5lib.fonts import load_font, D, bitflip            # noqa: E402
from tp5lib.autoencoder import build_ae, train_denoising, px_err_clean  # noqa: E402
from mlp.optimizers import Adam                            # noqa: E402

SEED = 0
EPOCHS = 6000
X = load_font(FONT)
N = len(X)


def make_dae(latent=10, hidden=(25,), seed=SEED):
    return build_ae(D, hidden=hidden, latent=latent, act_hidden="tanh",
                    act_latent="identity", act_out="sigmoid", loss="bce",
                    optimizer=Adam(0.01), seed=seed)


def eval_denoise(net, levels, trials=30, seed=123):
    """Por nivel de ruido de test: px error medio y fracción ≤1px / perfecta
    sobre trials realizaciones × 32 letras."""
    rng = np.random.default_rng(seed)
    out = {}
    for p in levels:
        errs = np.concatenate([px_err_clean(net, bitflip(X, p, rng), X) for _ in range(trials)])
        out[p] = dict(px_mean=float(errs.mean()),
                      frac_leq1=float((errs <= 1).mean()), frac_perfect=float((errs == 0).mean()))
    return out


TEST_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
t0 = time.time()

print("== E9: barrido de ancho de cuello (p_train=0.15) ==")
rows = []
for latent in [2, 5, 10, 20]:
    rng = np.random.default_rng(SEED)
    dae = make_dae(latent=latent)
    train_denoising(dae, X, 0.15, EPOCHS, rng)
    ev = eval_denoise(dae, [0.1, 0.2, 0.3])
    rows.append({"cuello": latent,
                 "px@0.1": round(ev[0.1]["px_mean"], 2), "px@0.2": round(ev[0.2]["px_mean"], 2),
                 "px@0.3": round(ev[0.3]["px_mean"], 2), "%<=1px@0.2": round(100 * ev[0.2]["frac_leq1"], 1)})
pd.DataFrame(rows).to_csv(RES / "e9_bottleneck.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== E10: ruido de entrenamiento × nivel de test (cuello=10) ==")
allrows = []
saved = None
for ptr in [0.05, 0.15, 0.30]:
    rng = np.random.default_rng(SEED)
    dae = make_dae(latent=10)
    train_denoising(dae, X, ptr, EPOCHS, rng)
    if abs(ptr - 0.15) < 1e-9:
        saved = dae
    ev = eval_denoise(dae, TEST_LEVELS)
    for p, m in ev.items():
        allrows.append({"p_train": ptr, "test_p": p, "px_mean": round(m["px_mean"], 3),
                        "frac_leq1": round(m["frac_leq1"], 3)})
df10 = pd.DataFrame(allrows)
df10.to_csv(RES / "e10_train_noise.csv", index=False)
print(df10.pivot(index="test_p", columns="p_train", values="px_mean").to_string())

# --- Ganador reforzado: cuello=10, p_train=0.15 reentrenado a FULL épocas (número contundente) ---
print("\n== Ganador denoising (cuello=10, p_train=0.15) a full épocas ==")
EPOCHS_CHAMP = 15000
rng = np.random.default_rng(SEED)
champ = make_dae(latent=10)
train_denoising(champ, X, 0.15, EPOCHS_CHAMP, rng)
ev = eval_denoise(champ, [0.10, 0.15, 0.20], trials=50)
champ_rows = [{"test_p": p, "px_mean": round(ev[p]["px_mean"], 3),
               "pct_leq1": round(100 * ev[p]["frac_leq1"], 1),
               "pct_perfect": round(100 * ev[p]["frac_perfect"], 1)} for p in [0.10, 0.15, 0.20]]
pd.DataFrame(champ_rows).to_csv(RES / "e_champion.csv", index=False)
print(pd.DataFrame(champ_rows).to_string(index=False))
champ.save(RES / "dae_champion.npz")  # -> tripletes E11 (ganador reforzado)

(RES / "config_used.json").write_text(json.dumps(
    {"seed": SEED, "epochs_sweep": EPOCHS, "epochs_champion": EPOCHS_CHAMP,
     "dae": "35-25-{cuello}-25-35 | tanh/identity/sigmoid | BCE | Adam(0.01)",
     "champion": "cuello=10, p_train=0.15", "test_levels": TEST_LEVELS}, indent=2))
print(f"\nOK 1b en {time.time() - t0:.1f}s. DAE ganador reforzado (15000 ep) guardado.")
