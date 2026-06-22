"""PASO 5 — dimensión latente (confirmación).

La consigna FIJA el latente en 2D. Barremos 1, 2, 3 para mostrar el "codo": con 1 no alcanza
(solo un subconjunto de letras perfectas), con 2 ya se cumple, y 3 no agrega. La columna
`subconjunto` registra cuántas letras se aprenden perfecto en cada caso (la cláusula del
enunciado: "en caso de aprender un subconjunto, mostrar por qué no fue posible el completo").

Conclusión esperada: 2 es el mínimo que cumple px_max<=1 → coincide con lo que pide la consigna.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403


def main():
    banner("PASO 5 · dimensión latente (confirmación del 2D)")
    state = load_state()
    print("Config de partida:", state)

    curves, summary = [], []
    for L in [1, 2, 3]:
        cfg = {**state, "latent": L}
        _, c, final = train_logged(cfg)
        c.insert(0, "variante", f"latente={L}")
        curves.append(c)
        summary.append(dict(latente=L, **final,
                            subconjunto=f"{final['perfectas']}/32",
                            aprende_completo=bool(final["px_max"] <= 1)))
        print(f"  latente={L}: {final}  (perfectas {final['perfectas']}/32)")

    write_curves(curves, "exp5")
    df = pd.DataFrame(summary)
    write_summary(df, "exp5")

    # Selección: menor latente que logra px_max<=1 (parsimonia + consigna pide 2D).
    ok = df[df["px_max"] <= 1]
    win = ok.sort_values("latente").iloc[0]
    state["latent"] = int(win["latente"]); save_state(state)
    print(f"GANADOR: latente={win['latente']} (mínimo que cumple ≤1px). Config:", state)
    return dict(exp="exp5", eje="latente", ganador=f"latente={int(win['latente'])}",
                cambio=(int(win["latente"]) != 2), config=dict(state))


if __name__ == "__main__":
    main()
