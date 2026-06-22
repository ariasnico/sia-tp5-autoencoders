"""PASO 4 — tamaño y profundidad de la capa oculta (con Adam 0.01 ya fijado).

Barre, sobre el encoder (el decoder se espeja solo):
  ()      -> 35-2-35            (sin capa oculta: encoder lineal)
  (10,)   -> 35-10-2-10-35      (1 capa de 10 por lado)
  (20,)   -> 35-20-2-20-35      (1 capa de 20 por lado)
  (30,)   -> 35-30-2-30-35      (1 capa de 30 por lado)
  (20,20) -> 35-20-20-2-20-20-35 (2 capas de 20 por lado: más PROFUNDA)

Mezcla tres preguntas: ¿hace falta capa oculta? (), ¿qué tan ancha? (10/20/30), ¿qué tan
profunda? (20 vs 20+20). Conclusión esperada: (20,) es la más simple que da px_max<=1;
30 y 20+20 empatan en 0 px → por PARSIMONIA gana (20,).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403


def main():
    banner("PASO 4 · arquitectura (capa oculta)")
    state = load_state()
    print("Config de partida:", state)

    curves, summary = [], []
    for h in [[], [10], [20], [30], [20, 20]]:
        # caso (): encoder lineal -> activación oculta irrelevante (no hay capa oculta)
        cfg = {**state, "hidden": h,
               "act_hidden": ("identity" if len(h) == 0 else state["act_hidden"])}
        _, c, final = train_logged(cfg)
        label = hidden_label(h)
        c.insert(0, "variante", label)
        curves.append(c)
        summary.append(dict(hidden=label, n_capas_por_lado=len(h), arquitectura=arch_str(cfg),
                            n_params=count_params(cfg), **final))
        print(f"  hidden={label:6s} [{arch_str(cfg)}] ({count_params(cfg)} params): {final}")

    write_curves(curves, "exp4")
    df = pd.DataFrame(summary)
    write_summary(df, "exp4")

    # Selección por parsimonia: entre las que logran px_max<=1, la de MENOS parámetros.
    ok = df[df["px_max"] <= 1]
    win = (ok.sort_values("n_params").iloc[0] if len(ok)
           else df.sort_values(["px_max", "n_params"]).iloc[0])
    state["hidden"] = parse_hidden(win["hidden"]); save_state(state)
    print(f"GANADOR: hidden={win['hidden']} (mín. params con px_max<=1). Config:", state)
    return dict(exp="exp4", eje="arquitectura", ganador=f"hidden={win['hidden']}",
                cambio=(parse_hidden(win["hidden"]) != [20]), config=dict(state))


if __name__ == "__main__":
    main()
