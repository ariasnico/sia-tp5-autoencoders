# Cómo leer `tabla_1a_config.png` — Configuración del Autoencoder (1a)

**Qué es esta tabla:** la **receta del modelo ganador** de 1a — un solo modelo, el ganador. Cada fila es
**un valor fijo** y **todos conviven al mismo tiempo** en esa única red. NO son valores que comparamos
(las comparaciones están en `tabla_1a_resultados.png`).

> Analogía: esta tabla es la **receta final**; la de resultados son las **pruebas** que hicimos para llegar a ella.

## Fila por fila

- **Arquitectura `35-20-2-20-35`** — la **forma** de la red, leída como el viaje de los datos: entran **35**
  píxeles (7×5) → capa oculta **20** → **cuello de 2** (el latente, el corazón del AE) → capa oculta **20**
  → salida **35** (la reconstrucción). Es **una sola** red con esas 5 capas, todas juntas; no son alternativas.
  (El "2" del cuello lo elegimos en E2; el "20" en E3.)
- **Activaciones `tanh / identity / sigmoid`** — tres funciones que la red usa **a la vez**, una por zona:
  `tanh` en las ocultas, `identity` en el cuello (deja el latente libre), `sigmoid` en la salida (valores 0–1
  como los píxeles). Las 3 al mismo tiempo. (Compararlas como alternativas fue E6.)
- **Loss `BCE`** — la función que mide el error al entrenar. Un valor. La elegimos porque los píxeles son 0/1.
  (BCE vs MSE = E7.)
- **Optimizador `Adam (lr=0.01)`** — el algoritmo que ajusta los pesos; `lr` = tamaño del paso. Un valor.
  (SGD/Momentum/Adam se comparó en E4.)
- **Épocas `6000`** — cuántas veces la red recorre todo el dataset entrenando. Un valor.
- **Batch `32 (full-batch)`** — de a cuántas muestras actualiza. Como hay 32 letras y usamos 32, entran
  **todas juntas** en cada paso.
- **Inicialización `auto`** — con qué valores **arrancan** los pesos antes de entrenar (Xavier/tanh, He/relu,
  elegido solo). Un valor.
- **Dataset `32 letras 7×5 · sin split`** — los datos. "Sin split" = **no** separamos train/test, porque el
  objetivo es *memorizar* esas 32 exactas (no generalizar a otras).
- **Métrica `px incorrectos (umbral 0.5)`** — binarizamos la salida en 0.5 y contamos píxeles distintos del
  original. Objetivo: máximo ≤ 1.
- **Seed `0`** — la semilla del azar (pesos iniciales, orden de datos). Fija = siempre da lo mismo (reproducible).

## Qué nos dio
Con TODA esta receta junta, la red logra **0 px de error en las 32 letras (32/32 perfectas)** → cumple el
objetivo del enunciado (≤1 px). Esa es la conclusión de esta tabla.

## Cómo conecta con los experimentos
En cada experimento (E1–E7) tomamos esta receta y **cambiamos UNA sola fila**, dejando el resto igual, para
ver qué pasa. Eso es lo que muestra `tabla_1a_resultados.png`.
