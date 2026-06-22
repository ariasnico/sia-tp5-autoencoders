"""PASO 7 — función de pérdida (confirmación de una elección HEREDADA).

BCE no se "descubre" acá: viene del TP3 (donde construimos y validamos `mlp/` y elegimos BCE
para targets binarios) y de la teoría (píxeles Bernoulli). Este paso solo CONFIRMA esa elección:
BCE vs MSE con todo lo demás fijo. MSE mantiene la salida sigmoid (como en E7 original).

Conclusión esperada: BCE 0 px vs MSE ~2 px → se mantiene BCE. NO cambia la config.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403


def main():
    banner("PASO 7 · loss BCE vs MSE (confirmación)")
    state = load_state()
    print("Config de partida:", state)

    curves, summary = [], []
    for loss in ["bce", "mse"]:
        cfg = {**state, "loss": loss}  # act_out queda sigmoid en ambos (igual que E7 original)
        _, c, final = train_logged(cfg)
        c.insert(0, "variante", loss)
        curves.append(c)
        summary.append(dict(loss=loss, **final,
                            final_loss=round(float(c["train_loss"].iloc[-1]), 6)))
        print(f"  {loss}: {final}")

    write_curves(curves, "exp7")
    df = pd.DataFrame(summary)
    write_summary(df, "exp7")

    win = df.sort_values(["px_max", "px_mean"]).iloc[0]["loss"]
    cambio = (win != state["loss"])
    if cambio:  # salvaguarda; no debería pasar
        state["loss"] = str(win)
    save_state(state)
    print(f"GANADOR: {win} → se mantiene BCE (heredado del TP3, confirmado). Config:", state)
    return dict(exp="exp7", eje="loss", ganador=win, cambio=cambio, config=dict(state))


if __name__ == "__main__":
    main()
