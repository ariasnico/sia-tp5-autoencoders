"""E9b — barrido del ancho de la CAPA OCULTA para el denoiser (latente fijo en 10).

Narrativa (coordinate descent, igual que 1a):
  1. Partimos de la arquitectura de 1a (oculta=20) y barrimos el CUELLO -> elegimos latente=10 (E9).
  2. Con el cuello ya fijado en 10, ahora barremos el ancho de la capa OCULTA {20,30,35,40}.

Así el ancho de la capa oculta queda "ganado" experimentalmente (antes el 25 estaba sin justificar).
Entrena cada DAE con bit-flip fresco (p_train=0.15), evalúa en varios niveles de test (30 trials).

Salidas:
  results/e_hidden_sweep.csv  — métricas por ancho
  figs/fig_e9b_hidden.png     — px error y %≤1px vs ancho de capa oculta

Uso:  python3 exp_hidden_sweep.py    (TP5_EPOCHS=300 para prueba rápida)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
RES = HERE / "results"; RES.mkdir(exist_ok=True)
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
FONT = ROOT / "font.h"

from tp5lib.fonts import load_font, D, bitflip                       # noqa: E402
from tp5lib.autoencoder import build_ae, train_denoising, px_err_clean  # noqa: E402
from tp5lib.plotstyle import apply_style, PRIMARY, SECONDARY, ACCENT, WARN  # noqa: E402
from mlp.optimizers import Adam                                       # noqa: E402

apply_style()
GREY = "#9AA5B1"

SEED = 0
EPOCHS = int(os.environ.get("TP5_EPOCHS", 6000))
LATENT = 10
P_TRAIN = 0.15
HIDDENS = [20, 30, 35, 40]
X = load_font(FONT)


def make_dae(hidden, latent=LATENT, seed=SEED):
    return build_ae(D, hidden=(hidden,), latent=latent, act_hidden="tanh",
                    act_latent="identity", act_out="sigmoid", loss="bce",
                    optimizer=Adam(0.01), seed=seed)


def n_params(net):
    return int(sum(w.size for w in net.weights))


def eval_denoise(net, levels, trials=30, seed=123):
    rng = np.random.default_rng(seed)
    out = {}
    for p in levels:
        errs = np.concatenate([px_err_clean(net, bitflip(X, p, rng), X) for _ in range(trials)])
        out[p] = dict(px_mean=float(errs.mean()),
                      frac_leq1=float((errs <= 1).mean()), frac_perfect=float((errs == 0).mean()))
    return out


def main():
    print(f"E9b · barrido capa oculta {HIDDENS} (latente={LATENT}, p_train={P_TRAIN}, epochs={EPOCHS})")
    rows = []
    for h in HIDDENS:
        rng = np.random.default_rng(SEED)
        dae = make_dae(h)
        train_denoising(dae, X, P_TRAIN, EPOCHS, rng)
        ev = eval_denoise(dae, [0.1, 0.2, 0.3])
        rows.append({
            "hidden": h, "arquitectura": f"35-{h}-{LATENT}-{h}-35", "n_params": n_params(dae),
            "px@0.1": round(ev[0.1]["px_mean"], 3), "px@0.2": round(ev[0.2]["px_mean"], 3),
            "px@0.3": round(ev[0.3]["px_mean"], 3),
            "pct_leq1@0.2": round(100 * ev[0.2]["frac_leq1"], 1),
        })
        print(f"  oculta={h:>2}: {rows[-1]}")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "e_hidden_sweep.csv", index=False)
    print("\n", df.to_string(index=False))

    # ---- figura ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    x = range(len(df))
    for col, lab, c in [("px@0.1", "test 10%", "#2563EB"), ("px@0.2", "test 20%", PRIMARY),
                        ("px@0.3", "test 30%", "#7C3AED")]:
        a1.plot(x, df[col], "-o", label=lab, color=c, lw=2, ms=8)
    a1.set_xticks(list(x)); a1.set_xticklabels(df["hidden"])
    a1.set_xlabel("ancho de capa oculta (latente=10)"); a1.set_ylabel("px incorrectos tras limpiar (prom)")
    a1.set_title("Error de denoising vs ancho de capa oculta"); a1.legend()

    bars = a2.bar(x, df["pct_leq1@0.2"], color=GREY, width=0.6)
    a2.set_xticks(list(x)); a2.set_xticklabels(df["hidden"])
    a2.set_xlabel("ancho de capa oculta"); a2.set_ylabel("% letras ≤1px @ 20% ruido")
    for b, v, n in zip(bars, df["pct_leq1@0.2"], df["n_params"]):
        a2.annotate(f"{v:.0f}%\n{n}w", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")
    a2.set_ylim(0, max(df["pct_leq1@0.2"]) * 1.25)
    a2.set_title("% recuperadas (≤1px) @ 20% ruido")
    fig.subplots_adjust(top=0.92)
    fig.savefig(FIGS / "fig_e9b_hidden.png"); plt.close(fig)
    print("\n  -> figs/fig_e9b_hidden.png")
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    t0 = time.time()
    main()
