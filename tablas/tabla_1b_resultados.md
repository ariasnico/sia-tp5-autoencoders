# Cómo leer `tabla_1b_resultados.png` — Experimentos de 1b (los barridos)

**Qué es esta tabla:** los experimentos del denoising. Cada fila varía **una sola cosa** respecto a la receta
base (`tabla_1b_config.png`). Mismas columnas que en 1a: Exp · Qué variamos · Valores probados · Resultado.

## Fila por fila

- **E9 · ancho de cuello · `2, 5, 10, 20`** (base 35-20) — probamos 4 cuellos (entrenando cada uno como denoiser
  y midiendo cuántas letras recupera a 20 % de ruido). **2→44 %** · **10→54 %** · 20 satura. → el cuello 2 de 1a
  es malo para limpiar; ensanchar a 10 ayuda; más de 10 no agrega.
- **E9b · ancho de capa oculta · `20, 30, 35, 40`** (cuello fijo en 10) — con el cuello ya elegido, barremos el
  ancho de la capa oculta. **20→54 %** · **30→63 %** · 35/40 forman una meseta. → por **parsimonia** elegimos
  **30** (la más chica de la meseta); así el ancho queda justificado (antes era 25, sin barrer).
- **E10 · ruido de train · `0.05, 0.15, 0.30`** — entrenamos 3 denoisers con distinto nivel de ruido y los
  probamos contra muchos niveles de test. Las curvas **se cruzan**: poco ruido = mejor en limpio pero frágil;
  mucho ruido = robusto pero peor en limpio. → es un **trade-off**, no hay un único "mejor".
- *(El modelo ganador —35-30-10-30-35, 15000 ep— ya no figura en esta tabla-resumen para no adelantar el número:
  su resultado contundente, **80 % ≤1px @15 %** / 92 % @10 % / 64 % @20 %, se muestra en su propia slide.)*

## La gran conclusión
Limpiar ruido **necesita más capacidad latente** que comprimir (E9), el ancho de la capa oculta se gana por
parsimonia (E9b), y el **ruido de entrenamiento** es un hiperparámetro con trade-off (E10). El ganador
35-30-10-30-35 / 15 % es el equilibrio.
