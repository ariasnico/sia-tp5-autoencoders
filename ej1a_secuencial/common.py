"""Común a los experimentos SECUENCIALES de 1a (coordinate descent → config ganadora).

Idea: hay una CONFIG INICIAL (base/naive con SGD) y cada experimento (exp1..exp7) varía UN
solo eje, mide, elige el ganador y **actualiza** `results/state.json`. El siguiente experimento
parte de esa config acumulada. Así se "gana el orden": el camino desde la base hasta el ganador.

Cada experimento escribe:
  results/<exp>_curves.csv   -> curvas por época (cada LOG_EVERY épocas) por variante
  results/<exp>_summary.csv  -> métricas finales por variante (1 fila c/u)
y actualiza results/state.json (la config acumulada).

Selección del ganador (regla explícita, Occam): objetivo px_max <= 1; entre las que lo logran,
la más parsimoniosa / la fijada por la consigna. Empates en px → se mantiene el incumbente.

Variables de entorno para pruebas rápidas:
  TP5_EPOCHS (default 6000), TP5_LOG_EVERY (default 50)
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
FONT = ROOT / "font.h"

from tp5lib.fonts import load_font, D                       # noqa: E402
from tp5lib.autoencoder import build_ae, px_err, encode     # noqa: E402
from mlp.optimizers import SGD, Momentum, Adam              # noqa: E402

np.seterr(over="ignore", invalid="ignore")  # lr alto (exp3) puede desbordar: es el punto

SEED = 0
EPOCHS = int(os.environ.get("TP5_EPOCHS", 6000))
LOG_EVERY = int(os.environ.get("TP5_LOG_EVERY", 50))
ADAM_EPS = 1e-4  # unificado en todo 1a (estabiliza E6, evita "dying relu")
X = load_font(FONT)

# --- Config INICIAL (base/naive): el punto de partida del camino ---
INITIAL_CONFIG = {
    "hidden": [20],          # 1 capa oculta de 20 por lado (provisional; se confirma en exp4)
    "latent": 2,             # lo exige la consigna (se confirma en exp5)
    "act_hidden": "tanh",
    "act_latent": "identity",  # teoría: el cuello es lineal
    "act_out": "sigmoid",      # acoplado a BCE (salida = probabilidad)
    "loss": "bce",             # heredado del TP3 + teoría binaria (se confirma en exp7)
    "optimizer": "sgd",        # arranque naive
    "lr": 0.5,
    "epochs": EPOCHS,
}

STATE_PATH = RESULTS / "state.json"


# ----------------------------- estado acumulado -----------------------------
def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return dict(INITIAL_CONFIG)


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def reset_state() -> dict:
    s = dict(INITIAL_CONFIG)
    save_state(s)
    return s


# ----------------------------- helpers de arquitectura -----------------------------
def layer_sizes(cfg: dict) -> list[int]:
    h = list(cfg["hidden"])
    return [D] + h + [cfg["latent"]] + h[::-1] + [D]


def arch_str(cfg: dict) -> str:
    return "-".join(map(str, layer_sizes(cfg)))


def count_params(cfg: dict) -> int:
    s = layer_sizes(cfg)
    return int(sum(s[i + 1] * (s[i] + 1) for i in range(len(s) - 1)))  # +1 por bias


def hidden_label(h: list[int]) -> str:
    return "()" if len(h) == 0 else "+".join(map(str, h))


def parse_hidden(label: str) -> list[int]:
    return [] if label == "()" else [int(x) for x in label.split("+")]


def make_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "sgd":
        return SGD(lr)
    if name == "momentum":
        return Momentum(lr)          # beta=0.9 por default
    if name == "adam":
        return Adam(lr, eps=ADAM_EPS)
    raise ValueError(f"optimizador desconocido: {name!r}")


# ----------------------------- entrenamiento con logging -----------------------------
def train_logged(cfg: dict, seed: int = SEED):
    """Entrena un AE según `cfg` y registra métricas cada LOG_EVERY épocas.

    Devuelve (ae, curves_df, final_metrics_dict). `curves_df` tiene una fila cada LOG_EVERY
    épocas con: epoch, train_loss, px_max, px_mean, perfectas, leq1.
    """
    np.random.seed(seed)
    opt = make_optimizer(cfg["optimizer"], cfg["lr"])
    ae = build_ae(D, hidden=tuple(cfg["hidden"]), latent=cfg["latent"],
                  act_hidden=cfg["act_hidden"], act_latent=cfg["act_latent"],
                  act_out=cfg["act_out"], loss=cfg["loss"], optimizer=opt, seed=seed)
    epochs = cfg["epochs"]
    rows = []

    def cb(ep, m):
        if ep % LOG_EVERY == 0 or ep == epochs - 1:
            e = px_err(ae, X)
            rows.append(dict(epoch=int(ep), train_loss=float(m["train_loss"]),
                             px_max=int(e.max()), px_mean=round(float(e.mean()), 4),
                             perfectas=int((e == 0).sum()), leq1=int((e <= 1).sum())))

    ae.fit(X, X, X, X, epochs=epochs, batch_size=32, callback=cb)
    e = px_err(ae, X)  # métricas finales (pesos del mejor val_loss, restaurados por fit)
    final = dict(px_max=int(e.max()), px_mean=round(float(e.mean()), 4),
                 perfectas=int((e == 0).sum()), leq1=int((e <= 1).sum()))
    return ae, pd.DataFrame(rows), final


# ----------------------------- escritura de CSVs -----------------------------
def write_curves(frames, exp: str) -> None:
    df = pd.concat(frames, ignore_index=True) if isinstance(frames, list) else frames
    df.to_csv(RESULTS / f"{exp}_curves.csv", index=False)


def write_summary(df: pd.DataFrame, exp: str) -> None:
    df.to_csv(RESULTS / f"{exp}_summary.csv", index=False)


def banner(txt: str) -> None:
    print("\n" + "=" * 64 + f"\n{txt}\n" + "=" * 64)
