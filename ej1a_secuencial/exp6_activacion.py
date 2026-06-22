"""PASO 6 — activación de la capa oculta (confirmación).

Barre tanh / relu / sigmoid en las dos capas ocultas (enc+dec). Las tres llegan a 0 px: lo
único que cambia es la VELOCIDAD de convergencia, que registramos como la época en que la loss
baja de 1e-2 (columna `epoca_conv`).

Conclusión esperada: empate en px → se mantiene el incumbente (tanh). relu suele ser la más
rápida, pero la velocidad no es la métrica que pide la consigna (px_max). NO cambia la config.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403


def main():
    banner("PASO 6 · activación oculta (confirmación)")
    state = load_state()
    print("Config de partida:", state)

    curves, summary = [], []
    for act in ["tanh", "relu", "sigmoid"]:
        cfg = {**state, "act_hidden": act}
        _, c, final = train_logged(cfg)
        c.insert(0, "variante", act)
        curves.append(c)
        below = c[c["train_loss"] < 1e-2]
        ep_conv = int(below["epoch"].iloc[0]) if len(below) else -1
        summary.append(dict(act_hidden=act, **final, epoca_conv=ep_conv,
                            final_loss=round(float(c["train_loss"].iloc[-1]), 6)))
        print(f"  {act:8s}: {final}  conv(loss<1e-2)@{ep_conv}")

    write_curves(curves, "exp6")
    write_summary(pd.DataFrame(summary), "exp6")

    # Empate en px → se mantiene el incumbente (tanh). No se cambia la config.
    save_state(state)
    print(f"GANADOR: empate en px → se mantiene '{state['act_hidden']}' (incumbente). Config:", state)
    return dict(exp="exp6", eje="activacion", ganador=f"{state['act_hidden']} (empate)",
                cambio=False, config=dict(state))


if __name__ == "__main__":
    main()
