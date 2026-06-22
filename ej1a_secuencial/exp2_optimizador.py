"""PASO 2 — optimizador.

Con la arquitectura ya capaz (no-lineal, hidden=20), compara SGD(0.5) / Momentum(0.1) /
Adam(0.01). Comparación limpia porque la capacidad no es el cuello de botella.

Conclusión esperada: Adam llega a 0 px donde SGD se estanca. ACTUALIZA la config:
optimizer -> adam, lr -> 0.01.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403


def main():
    banner("PASO 2 · optimizador (SGD vs Momentum vs Adam)")
    state = load_state()
    print("Config de partida:", state)

    opts = [("sgd", 0.5), ("momentum", 0.1), ("adam", 0.01)]
    curves, summary = [], []
    for name, lr in opts:
        cfg = {**state, "optimizer": name, "lr": lr}
        _, c, final = train_logged(cfg)
        c.insert(0, "variante", f"{name}({lr})")
        curves.append(c)
        summary.append(dict(optimizer=name, lr=lr, **final,
                            final_loss=round(float(c["train_loss"].iloc[-1]), 6)))
        print(f"  {name}({lr}): {final}")

    write_curves(curves, "exp2")
    df = pd.DataFrame(summary)
    write_summary(df, "exp2")

    # Selección: menor px_max, desempate por menor loss final.
    win = df.sort_values(["px_max", "final_loss"]).iloc[0]
    state["optimizer"] = str(win["optimizer"]); state["lr"] = float(win["lr"])
    save_state(state)
    print(f"GANADOR: {win['optimizer']}({win['lr']}). Config actualizada:", state)
    return dict(exp="exp2", eje="optimizador", ganador=f"{win['optimizer']}({win['lr']})",
                cambio=True, config=dict(state))


if __name__ == "__main__":
    main()
