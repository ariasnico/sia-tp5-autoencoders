# Cómo leer `tabla_1a_resultados.png` — Experimentos de 1a (los barridos)

**Qué es esta tabla:** acá SÍ están los experimentos. Cada fila es **un experimento independiente** donde
**variamos UNA sola cosa** respecto a la receta base (`tabla_1a_config.png`), dejando todo lo demás igual,
para ver qué pasa. Esta es la tabla de "qué valores probamos y qué dio cada uno".

**Cómo se lee cada fila:**
- **Exp** — el código del experimento (E1, E2…).
- **Qué variamos** — el único hiperparámetro que cambiamos.
- **Valores probados** — la lista de valores que comparamos (sí, **probamos todos esos**, uno por vez).
- **Resultado / lectura** — qué dio + la conclusión.

## Fila por fila

- **E1 · tipo de modelo · `lineal · PCA · no-lineal`** — entrenamos 3 versiones del cuello 2D. El AE **lineal**
  y **PCA** dan el mismo error (**7.19 px**); el **no-lineal** baja a **0 px**. → la no-linealidad es necesaria
  (un AE lineal *es* PCA).
- **E2 · dim. latente · `1, 2, 3, 5, 8`** — probamos 5 tamaños de cuello. Con **1** solo aprende 18/32; desde
  **2** ya da **0 px**; más no mejora. → 2D es el mínimo viable (el "codo").
- **E3 · capa oculta · `(), (10), (20), (30), (20,20)`** — probamos 5 arquitecturas del encoder. **Sin** capa
  oculta queda en 15 px máx (encoder lineal≈PCA); con **≥20** unidades → 0 px. → hace falta la capa oculta no-lineal.
- **E4 · optimizador · `SGD · Momentum · Adam`** — misma red, 3 optimizadores. **Adam 0** · Momentum 2 · SGD 5 px (máx).
  → Adam gana.
- **E5 · learning rate · `0.0003, 0.01, 0.3`** — 3 tamaños de paso con Adam. Muy bajo = lento; **0.01** justo (0 px);
  **0.3 no aprende** (28 px). → sensibilidad al lr; mostramos uno que falla.
- **E6 · activación oculta · `tanh, relu, sigmoid`** — 3 activaciones. Las 3 llegan a **0 px**, difieren en
  *velocidad* de convergencia (por eso la figura es de curvas).
- **E7 · loss · `BCE, MSE`** — 2 pérdidas. **BCE 0 px** · MSE 2 px. → para binario, BCE.

## La gran conclusión
De juntar lo mejor de cada barrido sale la **receta ganadora** (`tabla_1a_config`): no-lineal + 2D + ≥20 unidades
+ Adam + tanh + BCE → **0 px**. Cada fila de esta tabla justifica una decisión de esa receta.
