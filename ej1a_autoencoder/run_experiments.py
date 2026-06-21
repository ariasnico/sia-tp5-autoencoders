"""TP5 1a — corre los experimentos E1..E8 del autoencoder básico (latente 2D, objetivo ≤1px).

Entrena, mide error en píxeles y persiste métricas (CSV) + curvas de loss (npz) + el
ganador (npz). Las figuras las produce make_figures.py desde estos artefactos. Semillas fijas.

Experimentos:
  E1  AE lineal vs AE no-lineal vs PCA(2)      -> la no-linealidad hace falta (lineal ≈ PCA)
  E2  barrido de dimensión latente {1,2,3,5,8} -> el "codo": 2D es el mínimo viable
  E3  barrido de arquitectura (hidden)         -> capacidad necesaria para meter 32 patrones en 2D
  E4  optimizadores SGD/Momentum/Adam          -> Adam converge donde SGD se estanca
  E5  learning rate {bajo, justo, alto}        -> sensibilidad; el alto no aprende (atascado arriba)
  E6  activación hidden tanh/relu/sigmoid      -> efecto de la no-linealidad
  E7  loss BCE vs MSE                           -> para binario, BCE es mejor
  E8  ganador -> guarda modelo + latente        -> ≤1px, scatter 2D y generación (en figuras)
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
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
FONT = ROOT / "font.h"

from tp5lib.fonts import load_font, D                      # noqa: E402
from tp5lib.autoencoder import build_ae, px_err, encode    # noqa: E402
from mlp.optimizers import SGD, Momentum, Adam             # noqa: E402

np.seterr(over="ignore", invalid="ignore")  # lr alto (E5) puede desbordar: es el punto

SEED = 0
EPOCHS = 6000
X = load_font(FONT)
N = len(X)


def train(hidden=(20,), latent=2, act_hidden="tanh", act_latent="identity",
          act_out="sigmoid", loss="bce", opt=None, epochs=EPOCHS, seed=SEED):
    """Entrena un AE y devuelve (modelo, error_px_por_letra, curva_loss)."""
    np.random.seed(seed)
    ae = build_ae(D, hidden=hidden, latent=latent, act_hidden=act_hidden,
                  act_latent=act_latent, act_out=act_out, loss=loss,
                  optimizer=opt if opt is not None else Adam(0.01, eps=1e-4), seed=seed)
    curve = []
    ae.fit(X, X, X, X, epochs=epochs, batch_size=32,
           callback=lambda ep, m: curve.append(m["train_loss"]))
    return ae, px_err(ae, X), np.array(curve)


def stats(e):
    return dict(px_max=int(e.max()), px_mean=round(float(e.mean()), 3),
                perfectas=int((e == 0).sum()), leq1=int((e <= 1).sum()))


SEEDS = [0, 1, 2]  # E2/E3/E4 se promedian sobre estas semillas (robustez n=3)


def stats_multiseed(train_kwargs, seeds=SEEDS):
    """Corre train() con varias semillas y agrega media/desvío de las métricas clave.
    Devuelve (row_dict_con_mean_std, curva_de_la_primera_semilla)."""
    px_max_l, perf_l, leq1_l, pxmean_l, curves_l = [], [], [], [], []
    for s in seeds:
        _, e, c = train(seed=s, **train_kwargs)
        px_max_l.append(int(e.max())); perf_l.append(int((e == 0).sum()))
        leq1_l.append(int((e <= 1).sum())); pxmean_l.append(float(e.mean()))
        curves_l.append(c)
    row = dict(
        px_max_mean=round(float(np.mean(px_max_l)), 3), px_max_std=round(float(np.std(px_max_l)), 3),
        px_max_min=int(np.min(px_max_l)), px_max_max=int(np.max(px_max_l)),
        perfectas_mean=round(float(np.mean(perf_l)), 3), perfectas_std=round(float(np.std(perf_l)), 3),
        perfectas_min=int(np.min(perf_l)), perfectas_max=int(np.max(perf_l)),
        leq1_mean=round(float(np.mean(leq1_l)), 3), leq1_std=round(float(np.std(leq1_l)), 3),
        px_mean_mean=round(float(np.mean(pxmean_l)), 3), px_mean_std=round(float(np.std(pxmean_l)), 3),
        n_seeds=len(seeds))
    return row, curves_l[0]


curves = {}
t0 = time.time()

print("== E1: lineal vs no-lineal vs PCA ==")
ae_champ, e_nl, c_nl = train()                                   # no-lineal ganador
_, e_lin, _ = train(hidden=(), act_hidden="identity",
                    act_latent="identity", act_out="identity", loss="mse")  # AE lineal
mu = X.mean(0); Xc = X - mu                                      # PCA(2) por SVD
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
Xr = (Xc @ Vt[:2].T) @ Vt[:2] + mu
e_pca = ((Xr > 0.5).astype(float) != X).sum(1).astype(int)
df_e1 = pd.DataFrame([
    {"modelo": "AE no-lineal (tanh,BCE)", **stats(e_nl)},
    {"modelo": "AE lineal (identity,MSE)", **stats(e_lin)},
    {"modelo": "PCA (2 comp)", **stats(e_pca)},
])
df_e1.to_csv(RESULTS / "e1_linear_vs_pca.csv", index=False)
print(df_e1.to_string(index=False))

print("\n== E2: barrido de dimensión latente (3 semillas) ==")
rows = []
for L in [1, 2, 3, 5, 8]:
    row, _ = stats_multiseed({"latent": L})
    rows.append({"latent": L, **row})
pd.DataFrame(rows).to_csv(RESULTS / "e2_latent_sweep.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== E3: barrido de arquitectura (hidden) (3 semillas) ==")
rows = []
for hid in [(), (10,), (20,), (30,), (20, 20)]:
    row, c = stats_multiseed({"hidden": hid})
    curves[f"e3_{'-'.join(map(str, hid)) or 'none'}"] = c  # curva semilla 0 (para convergencia)
    rows.append({"hidden": str(hid) if hid else "()", **row})
pd.DataFrame(rows).to_csv(RESULTS / "e3_architecture.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== E4: optimizadores (3 semillas) ==")
rows = []
# fábricas: cada semilla recibe un optimizador FRESCO (los optimizadores son stateful).
# Adam usa eps=1e-4 unificado con el resto de 1a (SGD/Momentum no tienen eps).
opt_factories = [("SGD(0.5)", lambda: SGD(0.5)),
                 ("Momentum(0.1)", lambda: Momentum(0.1)),
                 ("Adam(0.01)", lambda: Adam(0.01, eps=1e-4))]
for name, make_opt in opt_factories:
    px_max_l, perf_l, leq1_l, pxmean_l, lossf_l, curves_l = [], [], [], [], [], []
    for s in SEEDS:
        _, e, c = train(opt=make_opt(), seed=s)
        px_max_l.append(int(e.max())); perf_l.append(int((e == 0).sum()))
        leq1_l.append(int((e <= 1).sum())); pxmean_l.append(float(e.mean()))
        lossf_l.append(float(c[-1])); curves_l.append(c)
    curves[f"e4_{name}"] = curves_l[0]  # curva semilla 0 (para convergencia)
    rows.append({"optimizer": name,
                 "px_max_mean": round(float(np.mean(px_max_l)), 3), "px_max_std": round(float(np.std(px_max_l)), 3),
                 "perfectas_mean": round(float(np.mean(perf_l)), 3), "perfectas_std": round(float(np.std(perf_l)), 3),
                 "leq1_mean": round(float(np.mean(leq1_l)), 3), "leq1_std": round(float(np.std(leq1_l)), 3),
                 "px_mean_mean": round(float(np.mean(pxmean_l)), 3), "px_mean_std": round(float(np.std(pxmean_l)), 3),
                 "loss_final_mean": round(float(np.mean(lossf_l)), 5), "loss_final_std": round(float(np.std(lossf_l)), 5),
                 "n_seeds": len(SEEDS)})
pd.DataFrame(rows).to_csv(RESULTS / "e4_optimizers.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== E5: learning rate (Adam) (3 semillas) ==")
rows = []
for lr in [0.0003, 0.01, 0.3]:
    # multi-semilla igual que E2/E3/E4: el optimizador es stateful, así que cada
    # semilla recibe un Adam FRESCO vía la fábrica que train() instancia internamente.
    px_max_l, perf_l, leq1_l, pxmean_l, lossf_l, curves_l = [], [], [], [], [], []
    for s in SEEDS:
        _, e, c = train(opt=Adam(lr, eps=1e-4), seed=s)
        px_max_l.append(int(e.max())); perf_l.append(int((e == 0).sum()))
        leq1_l.append(int((e <= 1).sum())); pxmean_l.append(float(e.mean()))
        lossf_l.append(float(c[-1]) if np.isfinite(c[-1]) else np.nan); curves_l.append(c)
    curves[f"e5_lr_{lr}"] = curves_l[0]  # curva semilla 0 (para convergencia)
    px_max_mean = round(float(np.mean(px_max_l)), 3)
    # "no_aprende": criterio directo sobre el resultado (px_max_mean>1 => no llega al objetivo ≤1px).
    # Reemplaza la heurística vieja (c[-1]>c[0]), que marcaba mal lr=0.3 como diverge=False.
    rows.append({"lr": lr,
                 "px_max_mean": px_max_mean, "px_max_std": round(float(np.std(px_max_l)), 3),
                 "px_max_min": int(np.min(px_max_l)), "px_max_max": int(np.max(px_max_l)),
                 "px_mean_mean": round(float(np.mean(pxmean_l)), 3), "px_mean_std": round(float(np.std(pxmean_l)), 3),
                 "perfectas_mean": round(float(np.mean(perf_l)), 3), "perfectas_std": round(float(np.std(perf_l)), 3),
                 "perfectas_min": int(np.min(perf_l)), "perfectas_max": int(np.max(perf_l)),
                 "leq1_mean": round(float(np.mean(leq1_l)), 3), "leq1_std": round(float(np.std(leq1_l)), 3),
                 "loss_final_mean": round(float(np.nanmean(lossf_l)), 5), "loss_final_std": round(float(np.nanstd(lossf_l)), 5),
                 "no_aprende": bool(px_max_mean > 1), "n_seeds": len(SEEDS)})
pd.DataFrame(rows).to_csv(RESULTS / "e5_lr.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== E6: activación hidden ==")
rows = []
for act in ["tanh", "relu", "sigmoid"]:
    # eps=1e-4 unificado con el resto de 1a (mismo optimizador en todos los experimentos).
    _, e, c = train(act_hidden=act, opt=Adam(0.01, eps=1e-4))
    curves[f"e6_{act}"] = c
    rows.append({"act_hidden": act, **stats(e)})
pd.DataFrame(rows).to_csv(RESULTS / "e6_activation.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== E7: BCE vs MSE ==")
rows = []
for loss in ["bce", "mse"]:
    _, e, c = train(loss=loss)
    curves[f"e7_{loss}"] = c
    rows.append({"loss": loss, **stats(e)})
pd.DataFrame(rows).to_csv(RESULTS / "e7_loss.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== E8: ganador -> guardar modelo + latente ==")
ae_champ.save(RESULTS / "champion_1a.npz")
np.savez(RESULTS / "champion_latent.npz", Z=encode(ae_champ, X), px=e_nl)
curves["e8_champion"] = c_nl
np.savez(RESULTS / "curves_1a.npz", **curves)
(RESULTS / "config_used.json").write_text(json.dumps(
    {"seed": SEED, "epochs": EPOCHS, "batch_size": 32, "adam_eps": 1e-4,
     "seeds_multirun": SEEDS,
     "champion": "35-20-2-20-35 | tanh/identity/sigmoid | BCE | Adam(0.01, eps=1e-4)"}, indent=2))

print(f"\nOK 1a en {time.time() - t0:.1f}s. "
      f"Ganador: px_max={int(e_nl.max())} perfectas={int((e_nl == 0).sum())}/32 "
      f"<=1px={int((e_nl <= 1).sum())}/32")
