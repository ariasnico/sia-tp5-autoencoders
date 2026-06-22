"""Corre los 7 experimentos SECUENCIALES en orden, partiendo de la config INICIAL.

Resetea el estado, ejecuta exp1..exp7 (cada uno actualiza results/state.json), y escribe
results/camino.csv con la traza del coordinate descent: qué eje se tocó, qué ganó, si cambió
la config y cómo quedó la config acumulada tras cada paso.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import reset_state, save_state, RESULTS, arch_str, banner  # noqa: E402
import pandas as pd  # noqa: E402

import exp1_linealidad, exp2_optimizador, exp3_learning_rate  # noqa: E402
import exp4_arquitectura, exp5_latente, exp6_activacion, exp7_loss  # noqa: E402

PASOS = [exp1_linealidad, exp2_optimizador, exp3_learning_rate,
         exp4_arquitectura, exp5_latente, exp6_activacion, exp7_loss]


def main():
    banner("CAMINO SECUENCIAL → config ganadora (coordinate descent)")
    init = reset_state()
    print("Config INICIAL:", init)

    traza = []
    for i, mod in enumerate(PASOS, 1):
        r = mod.main()
        cfg = r["config"]
        traza.append(dict(
            paso=i, exp=r["exp"], eje=r["eje"], ganador=r["ganador"],
            cambio_config=r["cambio"], arquitectura=arch_str(cfg),
            optimizer=cfg["optimizer"], lr=cfg["lr"], loss=cfg["loss"],
            act_hidden=cfg["act_hidden"], latente=cfg["latent"],
        ))

    df = pd.DataFrame(traza)
    df.to_csv(RESULTS / "camino.csv", index=False)

    banner("CONFIG FINAL ACUMULADA")
    final = json.loads((RESULTS / "state.json").read_text())
    print(json.dumps(final, indent=2))
    print("\nTraza del camino (results/camino.csv):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
