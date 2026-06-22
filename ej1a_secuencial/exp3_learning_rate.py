"""PASO 3 — learning rate (con Adam ya fijado).

El lr es específico del optimizador, por eso se barre DESPUÉS de elegir Adam. Valores:
0.0003 (chico) / 0.01 (justo) / 0.3 (grande). El grande puede desbordar (np.seterr ignora).

Conclusión esperada: 0.01 es el punto justo (chico = lento, grande = no aprende). Si Adam ya
venía con lr=0.01 del paso 2, este paso lo CONFIRMA.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403


def main():
    banner("PASO 3 · learning rate (Adam)")
    state = load_state()
    print("Config de partida:", state)
    # exp3 es, por definición, "tunear el lr de Adam" (el optimizador elegido en exp2).
    if state.get("optimizer") != "adam":
        print(f"  [aviso] optimizer={state.get('optimizer')!r} != adam; lo fijo a adam para este barrido.")
        state["optimizer"] = "adam"

    curves, summary = [], []
    for lr in [0.0003, 0.01, 0.3]:
        cfg = {**state, "optimizer": "adam", "lr": lr}
        _, c, final = train_logged(cfg)
        c.insert(0, "variante", f"lr={lr}")
        curves.append(c)
        fl = c["train_loss"].iloc[-1]
        summary.append(dict(lr=lr, **final, final_loss=round(float(fl), 6),
                            no_aprende=bool(final["px_max"] > 1)))
        print(f"  lr={lr}: {final}")

    write_curves(curves, "exp3")
    df = pd.DataFrame(summary)
    write_summary(df, "exp3")

    win = df.sort_values(["px_max", "final_loss"]).iloc[0]
    cambio = float(win["lr"]) != float(state["lr"])
    state["lr"] = float(win["lr"]); save_state(state)
    print(f"GANADOR: lr={win['lr']}. Config:", state)
    return dict(exp="exp3", eje="learning_rate", ganador=f"lr={win['lr']}",
                cambio=cambio, config=dict(state))


if __name__ == "__main__":
    main()
