# Configuraciones de los experimentos (TP5)

Referencia con la **configuración exacta y completa de cada ejecución**, extraída de los
`run_experiments.py`, de las librerías (`tp5lib/`, `mlp/`) y de los `results/config_used.json`.

- **Reproducibilidad:** todo corre con `seed=0`. 1a y 1b reproducen **al dígito**; el VAE no es
  bit-a-bit reproducible (el muestreo del reparam usa el RNG global de numpy), pero las
  conclusiones se mantienen.
- **Implementación:** numpy puro, desde cero. La red de 1a/1b es la del TP3 (`mlp/`); el VAE es
  `tp5lib/vae_core.py` (backprop derivado a mano, verificado por gradient check).
- Salvo aclaración, cada experimento **cambia una sola cosa** respecto de la receta base de su
  parte.

Fuentes:
[`ej1a_autoencoder/run_experiments.py`](ej1a_autoencoder/run_experiments.py) ·
[`ej1b_denoising/run_experiments.py`](ej1b_denoising/run_experiments.py) ·
[`ej2_vae/run_experiments.py`](ej2_vae/run_experiments.py) ·
[`tp5lib/autoencoder.py`](tp5lib/autoencoder.py) ·
[`mlp/network.py`](mlp/network.py) ·
[`mlp/optimizers.py`](mlp/optimizers.py) ·
[`mlp/initializers.py`](mlp/initializers.py) ·
[`tp5lib/vae_core.py`](tp5lib/vae_core.py).

---

## Parte 1a — Autoencoder (32 letras → 2D)

### Configuración base (receta común a E1–E8)

| Ítem                          | Valor                                                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset                       | `font.h`: 32 caracteres, 7×5 = **35 px** binarios `{0,1}` (fila-mayor)                                                                      |
| Arquitectura                  | `35-20-2-20-35` (capas `[35, 20, 2, 20, 35]`)                                                                                               |
| Activaciones (por transición) | `tanh` (oculta enc) · `identity` (cuello) · `tanh` (oculta dec) · `sigmoid` (salida)                                                        |
| Init de pesos                 | `auto` por capa: **Xavier** `N(0, 1/fan_in)` para tanh/sigmoid, **He** `N(0, 2/fan_in)` para relu, **uniforme** `[−0.1, 0.1]` para identity |
| Bias                          | incluido vía *bias trick* (columna de 1s); se inicializa con el mismo esquema que la capa                                                   |
| Loss                          | `BCE` (binary cross-entropy)                                                                                                                |
| Optimizador                   | `Adam(lr=0.01, β1=0.9, β2=0.999, eps=1e-4)`                                                                                                 |
| Épocas                        | 6000                                                                                                                                        |
| `batch_size`                  | 32 (= dataset completo ⇒ **full-batch**; el shuffle interno es irrelevante)                                                                 |
| Train / Val                   | ambos = `X` (las 32 letras son el universo completo; memorizarlas **es** el objetivo)                                                       |
| Selección de pesos            | `fit()` trackea el mejor `val_loss` y **restaura esos pesos** al terminar                                                                   |
| Semilla                       | `seed=0`                                                                                                                                    |

**Arquitectura**
Capas de entrada y salida 35 si o si por los datos
Capa latente 2 por la consigna
queda inconcluso el "20" pero parece un buen valor que podria elegirse arbitrariamente

**Loss**
BCE

**Activación**
latente identity si o si
salida sigmoid  -> para BCE , sigmoid queda en [0, 1]
las 2 ocultas tanh porque son no lineal?

> **`eps=1e-4` en vez del default `1e-8`:** en 1a se unifica este valor en todos los
> experimentos para estabilizar las comparaciones y evitar el "dying relu" en E6.

### E0 — Exploración del dataset
| Ítem | Valor |
|---|---|
| Tipo | solo análisis, **no entrena** |
| Datos | 32 letras de `font.h`, 7×5 = 35 px binarios |
| Métricas | matriz de distancias de **Hamming** entre pares + nº de píxeles encendidos por letra |

### E1 — No-linealidad: AE no-lineal vs AE lineal vs PCA
| modelo | arquitectura | activaciones | init | loss | optimizador |
|---|---|---|---|---|---|
| **AE no-lineal** (ganador) | `35-20-2-20-35` | tanh / identity / tanh / sigmoid | auto (Xavier/uniform) | BCE | Adam(0.01, β1=.9, β2=.999, eps=1e-4) |
| **AE lineal** | `35-2-35` (`hidden=()`) | identity / identity | uniforme `[−0.1, 0.1]` | **MSE** | Adam(0.01, β1=.9, β2=.999, eps=1e-4) |
| **PCA(2)** | — | — | — | — | analítico por **SVD** sobre `X − mean(X)`, 2 primeras componentes |

Comunes: 6000 ep, full-batch (32), `seed=0`. El AE lineal+MSE converge al mismo error que el
SVD analítico (≈7.19 px), confirmando el teorema "AE lineal ≡ PCA".

### E2 — Barrido de dimensión latente
| Ítem | Valor |
|---|---|
| **Variable** | `latent ∈ {1, 2, 3, 5, 8}` |
| Hereda | toda la receta base (arq. `35-20-·-20-35`, tanh/id/tanh/sigmoid, init auto, BCE, Adam(0.01, eps=1e-4), 6000 ep, full-batch) |
| Robustez | promedio sobre **3 semillas** (`seeds = [0, 1, 2]`); se reporta media ± desvío |

### E3 — Barrido de arquitectura (capa oculta)
| Ítem | Valor |
|---|---|
| **Variable** | `hidden ∈ {(), (10,), (20,), (30,), (20, 20)}` |
| Hereda | receta base, `latent=2`, BCE, Adam(0.01, eps=1e-4), 6000 ep, full-batch |
| Caso `()` | `35-2-35`, activaciones `identity / sigmoid` ⇒ **encoder lineal** (mismo techo que PCA) |
| Robustez | promedio sobre **3 semillas** (`[0, 1, 2]`) |

Complemento: [`diagnostics.py`](ej1a_autoencoder/diagnostics.py) corre 3 semillas (px_máx
`[14, 15, 16]`) para confirmar que la barrera lineal es **estructural, no un mínimo local**.

### E4 — Optimizadores
| optimizador | hiperparámetros |
|---|---|
| **SGD** | `lr = 0.5` |
| **Momentum** | `lr = 0.1`, `β = 0.9` |
| **Adam** | `lr = 0.01`, `β1 = 0.9`, `β2 = 0.999`, `eps = 1e-4` |

| Ítem | Valor |
|---|---|
| Hereda | receta base (arq, init, BCE, 6000 ep, full-batch) |
| Robustez | **3 semillas** (`[0, 1, 2]`); cada una recibe un optimizador **fresco** (son stateful) |

### E5 — Learning rate (con Adam)
| Ítem | Valor |
|---|---|
| **Variable** | `lr ∈ {0.0003, 0.01, 0.3}` con `Adam(β1=.9, β2=.999, eps=1e-4)` |
| Hereda | receta base (arq, init, BCE, 6000 ep, full-batch) |
| Robustez | promedio sobre **3 semillas** (`[0, 1, 2]`) |
| Criterio "no aprende" | `px_max_mean > 1` (no alcanza el objetivo ≤1px) |
| Numérico | `np.seterr(over="ignore", invalid="ignore")`: `lr=0.3` puede desbordar (es el punto) |

### E6 — Activación de la capa oculta
| Ítem | Valor |
|---|---|
| **Variable** | `act_hidden ∈ {tanh, relu, sigmoid}` (afecta las dos capas ocultas enc+dec) |
| Init asociado | cambia con la activación: tanh/sigmoid → Xavier, relu → He |
| Hereda | receta base, `Adam(0.01, eps=1e-4)`, BCE, 6000 ep, full-batch |
| Semillas | 1 (`seed=0`) |
| Métrica extra | época en que `loss < 1e-2` (velocidad de convergencia) |

### E7 — Función de pérdida
| Ítem | Valor |
|---|---|
| **Variable** | `loss ∈ {bce, mse}` |
| Hereda | receta base completa (arq, init, Adam(0.01, eps=1e-4), 6000 ep, full-batch) |
| Semillas | 1 (`seed=0`) |

### E8 — El ganador en acción
| Ítem | Valor |
|---|---|
| Modelo | `35-20-2-20-35`, tanh/identity/tanh/sigmoid, init auto, BCE, `Adam(0.01, eps=1e-4)` |
| Entrenamiento | 6000 ep, full-batch (32), `seed=0`, pesos del mejor `val_loss` |
| Reentrena | **no** — reusa `champion_1a.npz` |
| Generación (grilla) | grilla **14×14** de puntos del latente, cada uno decodificado |
| Interpolación | **9 pasos** lineales `(1−t)·z_a + t·z_o`, de `a` a `o` |
| Métrica de mapa | Hamming (px) vs distancia latente → Spearman **ρ = 0.57** |

---

## Parte 1b — Denoising Autoencoder

### Configuración base (común a E9–ganador)

| Ítem | Valor |
|---|---|
| Dataset | mismas 32 letras de `font.h` (35 px binarios) |
| Arquitectura | `35-25-{cuello}-25-35` (capas `[35, 25, cuello, 25, 35]`) |
| Activaciones | `tanh` / `identity` (cuello) / `tanh` / `sigmoid` (salida) |
| Init de pesos | `auto` por capa (Xavier para tanh/sigmoid, uniforme para el cuello identity) |
| Loss | `BCE` |
| Optimizador | `Adam(lr=0.01, β1=0.9, β2=0.999, eps=1e-8)` ← **eps default** (a diferencia de 1a) |
| Entrenamiento | `train_denoising`: **full-batch**, entrada con **bit-flip fresco cada época** (cada píxel se invierte con prob. `p`), target = patrón **limpio**. No usa `fit()` (orquesta forward/backward/step a mano); sin tracking de best-weights |
| Semilla pesos | `seed=0`; RNG de ruido propio por corrida |
| Evaluación | `eval_denoise`: por nivel de test, promedio de px-error sobre `trials` realizaciones × 32 letras; RNG `seed=123`. Reporta px medio, frac ≤1px y frac perfectas |

### E9 — Ancho de cuello
| Ítem | Valor |
|---|---|
| **Variable** | `cuello (latent) ∈ {2, 5, 10, 20}` |
| Ruido de entrenamiento | `p_train = 0.15` |
| Épocas | 6000 |
| Evaluación | **30** realizaciones × 32 letras, test `p ∈ {0.1, 0.2, 0.3}` |

### E10 — Ruido de entrenamiento × nivel de test
| Ítem | Valor |
|---|---|
| **Variable** | `p_train ∈ {0.05, 0.15, 0.30}` |
| Cuello | 10 |
| Épocas | 6000 |
| Evaluación | **30** realizaciones × 32 letras, test `p ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40}` |

El modelo con `p_train=0.15` se guarda para los tripletes de E11.

### E11 — Tripletes (limpio / sucio / recuperado)
| Ítem | Valor |
|---|---|
| Modelo | DAE **ganador reforzado** (`dae_champion.npz`) |
| Letras | `a e g r s` |
| Niveles | 3 (cualitativo, visual) |

### Ganador 1b — número final
| Ítem | Valor |
|---|---|
| Configuración | cuello = **10**, `p_train = 0.15` |
| Arquitectura | `35-25-10-25-35`, tanh/id/tanh/sigmoid, BCE, Adam(0.01, eps=1e-8) |
| Épocas | **15000** (vs 6000 de los barridos), `seed=0` |
| Evaluación | **50** realizaciones × 32 letras, test `p ∈ {0.10, 0.15, 0.20}` |

---

## Parte 2 — VAE (5 emojis)

### Configuración base (común a E12–E17)

| Ítem | Valor |
|---|---|
| Dataset | OpenMoji `color` (canal **alpha** = silueta rellena), 5 clases, 20×20 px, 140 variantes/clase = **700** imágenes |
| Augment (seedeado) | rotación ±15°, traslación ±2 px, ruido gaussiano `σ=0.03`, recorte `[0,1]` |
| Arquitectura | `400 → 128 → (μ, logσ²) en Z=2 → 128 → 400` (`D=400, He=128, Z=2, Hd=128`) |
| Encoder | `tanh(X·Wₑ + bₑ)` → cabezas lineales `μ` y `logσ²` |
| Decoder | `tanh(z·Wd + bd)` → logits → `sigmoid` |
| Reparam | `z = μ + σ·ε`, `ε ~ N(0,I)`, `σ = exp(½·logσ²)` |
| Init de pesos | **He** `N(0, 2/fan_in)`; bias en 0; **`W_lv` inicializado en 0** (⇒ `logσ²=0`, `σ=1` al inicio) |
| Loss | `recon(BCE-con-logits, sum/B) + β·KL`, con `KL = −0.5·Σ(1 + logσ² − μ² − σ²)/B` |
| Optimizador | `Adam(lr=1e-3, β1=0.9, β2=0.999, eps=1e-8)` |
| Épocas / batch | **3500** épocas, `batch=128`, shuffle por época |
| Semilla | `seed=0` (pesos y barajado; el `ε` usa el RNG global → no bit-a-bit reproducible) |

> Todo el backprop (KL + reparam trick incluidos) está derivado a mano en numpy y verificado con
> un **gradient check numérico (error 5e-08)** contra diferencias finitas. Reusa
> [`tp5lib/vae_core.py`](tp5lib/vae_core.py) sin modificar.

### Dataset — 5 emojis (OpenMoji)
| Ítem | Valor |
|---|---|
| Clases | corazón (`2764`), estrella (`2B50`), gota (`1F4A7`), luna (`1F319`), rayo (`26A1`) |
| Variante | `color` → se usa el canal **alpha** como silueta monocromática |
| Preproceso | crop al bounding box, centrado cuadrado, resize a 20×20 (margen 2 px) |
| Tamaño | 140 variantes/clase = **700** imágenes |
| Descartado | line-art (contornos se confunden a 20×20, 1-NN ~0.5) y formas geométricas |

### E12 — Barrido de β (el central)
| Ítem | Valor |
|---|---|
| **Variable** | `β ∈ {0, 0.5, 1, 4}` (4 VAEs entrenados) |
| Hereda | arquitectura y entrenamiento base (Adam 1e-3, 3500 ep, batch 128) |
| Sampleo | 10 muestras de `N(0,I)`; **semilla elegida** (búsqueda en `range(400)`) para máxima diversidad de clases |

### E13 — Mapa latente por clase
| Ítem | Valor |
|---|---|
| Modelo | VAE `β=1` guardado (`vae_beta_1.0.npz`) |
| Grafica | `μ = encode_mean(X)` de cada emoji, coloreado por clase, con círculos del prior N(0,I) (radio 1 y 2) |
| E18 (cobertura) | sobre **200 semillas** × 10 muestras: cobertura media **85%** (4.3/5 clases); reparto estrella ≈33%, luna ≈23%, gota ≈19%, rayo ≈14%, corazón ≈12% |

### E14 — Reconstrucción + generación
| Ítem | Valor |
|---|---|
| Modelo | VAE `β=1` |
| Reconstrucción | usa `μ` (sin ruido): fila real → fila reconstruida |
| Generación | 10 muestras nuevas desde `N(0,I)` (semilla diversa, misma que E12) |

### E15 — Curvas recon vs KL
| Ítem | Valor |
|---|---|
| **Variable** | `β ∈ {0, 1, 4}` |
| Registra | términos `recon` y `KL` por época |
| Frecuencia | log cada **5** épocas, evaluado con `eps=0` (determinista) sobre todo `X` |
| Entrenamiento | Adam(1e-3), 3500 ep, batch 128 |

### E16 — AE (β=0) vs VAE (β=1)
| Ítem | Valor |
|---|---|
| Modelos | VAE `β=0` (≈AE) y VAE `β=1`, ambos guardados (no reentrena) |
| Compara | dispersión del mapa (±49 vs ±3.3), parecido a N(0,I) (std 1.17), capacidad de samplear |

### E17 — Atlas del latente
| Ítem | Valor |
|---|---|
| Modelo | VAE `β=1` |
| Grilla | **13×13** puntos en `[−2.5, 2.5]²` |
| Qué hace | decodifica cada punto de la grilla (`generate`) → atlas continuo |
