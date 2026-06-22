"""PASO 1 — ¿hace falta no-linealidad?

Compara, partiendo de la config INICIAL (SGD, BCE, latente 2):
  - no_lineal : la base (35-20-2-20-35, tanh)            <- arm con capa oculta no-lineal
  - lineal    : 35-2-35 (hidden=(), identity)            <- encoder lineal, mismo loss BCE
  - pca_svd   : PCA(2) analítico por SVD (referencia, no entrena)

Conclusión esperada: no_lineal ≫ lineal ≈ PCA. La dirección es robusta al optimizador
(acá todavía SGD), así que justifica adoptar la no-linealidad. La config NO cambia (ya es
no-lineal); el experimento la confirma.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403


def main():
    banner("PASO 1 · linealidad (lineal vs no-lineal vs PCA)")
    state = load_state()
    print("Config de partida:", state)

    variants = {
        "no_lineal": {**state},
        "lineal": {**state, "hidden": [], "act_hidden": "identity"},
    }

    curves, summary = [], []
    for vname, cfg in variants.items():
        _, c, final = train_logged(cfg)
        c.insert(0, "variante", vname)
        curves.append(c)
        summary.append(dict(variante=vname, arquitectura=arch_str(cfg),
                            n_params=count_params(cfg), **final))
        print(f"  {vname:10s} [{arch_str(cfg)}]: {final}")

    # PCA(2) analítico por SVD (referencia)
    mu = X.mean(0); Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xr = (Xc @ Vt[:2].T) @ Vt[:2] + mu
    e = ((Xr > 0.5).astype(float) != X).sum(1).astype(int)
    summary.append(dict(variante="pca_svd", arquitectura="PCA(2) analítico", n_params=0,
                        px_max=int(e.max()), px_mean=round(float(e.mean()), 4),
                        perfectas=int((e == 0).sum()), leq1=int((e <= 1).sum())))
    print(f"  {'pca_svd':10s} [PCA(2)]: px_max={int(e.max())} perfectas={int((e == 0).sum())}/32")

    write_curves(curves, "exp1")
    write_summary(pd.DataFrame(summary), "exp1")

    # Selección: menor px_max entre los entrenados.
    df = pd.DataFrame(summary)
    trained = df[df.variante.isin(["no_lineal", "lineal"])]
    win = trained.sort_values(["px_max", "px_mean"]).iloc[0]["variante"]
    cambio = (win != "no_lineal")
    if cambio:  # no debería pasar; salvaguarda
        state["hidden"] = []; state["act_hidden"] = "identity"
    save_state(state)
    print(f"GANADOR: {win} -> se mantiene capa oculta no-lineal. Config:", state)
    return dict(exp="exp1", eje="linealidad", ganador=win, cambio=cambio, config=dict(state))


if __name__ == "__main__":
    main()
