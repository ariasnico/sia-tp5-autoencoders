# TP5 — Setup experimental (cómo se corrió cada experimento)

> Qué configuración e hiperparámetros usamos, con el **porqué** de cada elección. Esta info se guarda
> automáticamente en `<ejercicio>/results/config_used.json` por cada corrida. **Semilla fija (seed=0)** en todo;
> evaluaciones promediadas sobre varias realizaciones de ruido donde aplica.

## 1a — Autoencoder

### Configuración base (el ganador)
| Parámetro | Valor | Por qué |
|---|---|---|
| Arquitectura | `35-20-2-20-35` | MLP simétrico; cuello 2D = objetivo del enunciado |
| Activaciones | `tanh` ocultas · `identity` latente · `sigmoid` salida | identity deja el latente libre (scatter); sigmoid → pixeles en [0,1] |
| Loss | `BCE` | pixeles binarios (Bernoulli); MSE deja 2 px (E7) |
| Optimizador | `Adam (lr=0.01)` | converge donde SGD se estanca (E4) |
| Épocas | `6000` (sin early-stop) | de sobra: ya a ~1500 da 0 px |
| Batch | `32` (full-batch) | son solo 32 muestras |
| Inicialización | `auto` (Xavier/tanh, He/relu) | según la activación de cada capa |
| Dataset | 32 letras 7×5 (font.h), **sin split** | el objetivo es *memorizar* las 32 (no generalizar) |
| Métrica | px incorrectos/letra (umbral 0.5) | objetivo: máx ≤ 1 |

### Barridos y resultados
| Exp | Qué variamos | Valores | Resultado / lectura |
|---|---|---|---|
| E1 | tipo de modelo | lineal · PCA(2) · no-lineal | lineal ≡ PCA = 7.19 px → no-lineal **0 px** |
| E2 | dim. latente | 1, 2, 3, 5, 8 | 1→18/32; **≥2→0 px** (codo) |
| E3 | capa oculta | (), (10,), (20,), (30,), (20,20) | ()→14 px (=PCA); **≥20→0 px** |
| E4 | optimizador | SGD(0.5) · Momentum(0.1) · Adam(0.01) | Adam 0 · Mom 1 · SGD 5 px |
| E5 | learning rate | 0.0003 · 0.01 · 0.3 | 0.3 **no aprende** (28 px); 0.01 justo (0 px) |
| E6 | activación oculta | tanh · relu · sigmoid | las 3 → 0 px (difieren en velocidad) |
| E7 | loss | BCE · MSE | BCE 0 · MSE 2 px |

## 1b — Denoising Autoencoder

### Configuración base
| Parámetro | Valor | Por qué |
|---|---|---|
| Arquitectura | `35-25-{cuello}-25-35` | cuello variable; ganador usa 10 (E9) |
| Entrenamiento | entrada `bit-flip(p)` fresco/época → target limpio | el ruido binario natural; target ≠ entrada |
| `p_train` (ganador) | `0.15` | compromiso del trade-off (E10) |
| Loss / Opt | `BCE` / `Adam(0.01)` | igual que 1a |
| Épocas | `6000` (barridos) · `15000` (ganador) | el ganador se refuerza para un número contundente |
| Evaluación | 30–50 realizaciones de ruido × 32 letras por nivel | robustez estadística |

### Barridos y resultados
| Exp | Qué variamos | Valores | Resultado / lectura |
|---|---|---|---|
| E9 | ancho de cuello | 2, 5, 10, 20 | 2→48 % · **10→59 %** · 20 satura (≤1px @20% ruido) |
| E10 | ruido de entrenamiento | 0.05, 0.15, 0.30 | curvas se **cruzan**: trade-off poco/mucho ruido |
| ganador | (15000 ep, cuello 10) | test 10/15/20 % | **81 % ≤1px @15 %** (92 % @10 %, 64 % @20 %) |

## 2 — VAE

### Configuración base
| Parámetro | Valor | Por qué |
|---|---|---|
| Arquitectura | VAE `D=400 · He=128 · Z=2 · Hd=128` | encoder→(μ,logσ²), `z=μ+σε`, decoder→sigmoid |
| Loss | `recon(BCE) + β·KL` | reparam trick + KL derivados a mano (gradient check **5e-08**) |
| Optimizador | `Adam (lr=1e-3)` | |
| Épocas / batch | `3500` / `128` | |
| Dataset | 5 emojis OpenMoji, 140/clase = **700**, 20×20 grises | canal alpha (siluetas rellenas; acc 1-NN 0.95) |
| Augmentación | rot ±15° · trasl ±2 px · ruido σ=0.03 (seedeadas) | variedad intra-clase (que el latente sea continuo) |
| Sampleo (E12/E14) | semilla **26**, elegida y documentada para ilustrar las 5 clases | densidad no uniforme; control honesto en E18 (semilla 0) |
| Sampleo honesto (E18) | semilla **0** fija, sin elegir | cobertura media 85 % de clases (4.3/5) sobre 200 semillas |

### Barrido y resultados
| Exp | Qué variamos | Valores | Resultado / lectura |
|---|---|---|---|
| E12 | β (peso del KL) | 0, 0.5, 1, 4 | β=0 **ruido** (std 11.5); β=1 equilibrio; β=4 sobre-regulariza |
| — | recon (% px) por β | 0→2.6 % · 1→3.2 % · 4→3.7 % | más β = peor reconstrucción |
| — | latente std por β | 0→11.5 · 1→1.17 · 4→1.06 | más β = latente más ~N(0,I) |
