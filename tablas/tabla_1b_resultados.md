# Cómo leer `tabla_1b_resultados.png` — Experimentos de 1b (los barridos)

**Qué es esta tabla:** los experimentos del denoising. Cada fila varía **una sola cosa** respecto a la receta
base (`tabla_1b_config.png`). Mismas columnas que en 1a: Exp · Qué variamos · Valores probados · Resultado.

## Fila por fila

- **E9 · ancho de cuello · `2, 5, 10, 20`** — probamos 4 cuellos (entrenando cada uno como denoiser y midiendo
  cuántas letras recupera a 20 % de ruido). **2→48 %** · **10→59 %** · 20 satura. → el cuello 2 de 1a es malo para
  limpiar; ensanchar a 10 ayuda; más de 10 no agrega.
- **E10 · ruido de train · `0.05, 0.15, 0.30`** — entrenamos 3 denoisers con distinto nivel de ruido y los
  probamos contra muchos niveles de test. Las curvas **se cruzan**: poco ruido = mejor en limpio pero frágil;
  mucho ruido = robusto pero peor en limpio. → es un **trade-off**, no hay un único "mejor".
- **ganador · `cuello 10 · 15000 ep`** — no es un barrido: es el modelo final reforzado, evaluado a 10/15/20 %.
  Da **81 % ≤1px @15 %** (92 % @10 %, 64 % @20 %). → el número contundente del denoiser.

## La gran conclusión
Limpiar ruido **necesita más capacidad latente** que comprimir (E9), y el **ruido de entrenamiento** es un
hiperparámetro con trade-off (E10). El ganador cuello 10 / 15 % es el equilibrio.
