# Cómo leer `tabla_1b_config.png` — Configuración del Denoising AE (1b)

**Qué es esta tabla:** la **receta del denoiser ganador** — un solo modelo. Cada fila es **un valor fijo**, todos
juntos. Las comparaciones (barridos) están en `tabla_1b_resultados.png`.

## Fila por fila

- **Arquitectura `35-30-{cuello}-30-35`** — misma idea que 1a pero con capa oculta de **30** y un cuello más
  ancho. El `{cuello}` es el parámetro que barrimos en E9 (**ganador 10**, no 2, porque limpiar ruido necesita
  más capacidad); el ancho de la capa oculta **30** se gana en E9b (parsimonia sobre la meseta 30/35/40). Es una sola red.
- **Entrenamiento `bit-flip(p) → limpio`** — la **clave** del denoising: la entrada va con ruido (bits volteados
  con probabilidad `p`, fresco cada época) y el target es el patrón **limpio**. La red aprende "sucio → limpio",
  no "x → x".
- **p_train (ganador) `0.15`** — el nivel de ruido con el que entrenamos el ganador. Un valor (elegido por el
  trade-off de E10).
- **Loss / Opt `BCE / Adam(0.01)`** — iguales que 1a. Un valor cada uno.
- **Épocas `6000 · ganador 15000`** — los barridos corren 6000; el ganador se reentrena a 15000 para un número
  más contundente.
- **Evaluación `30-50 realiz. x 32 letras`** — no medimos con un solo ruido: promediamos sobre 30–50
  realizaciones aleatorias × 32 letras por nivel, para que el número sea robusto.
- **Seed `0`** — semilla fija, reproducible.

## Qué nos dio
El ganador (35-30-10-30-35, ruido de train 15 %, 15000 épocas) recupera **80 % de las letras con ≤1 px a 15 % de
ruido** (92 % a 10 %).

## Cómo conecta con los experimentos
E9 varía el `{cuello}` (base 35-20); E9b varía el ancho de la capa oculta (cuello fijo en 10); E10 varía el
`p_train`. El resto de la receta queda fijo. Ver `tabla_1b_resultados.png`.
