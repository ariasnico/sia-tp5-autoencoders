# 1a — Experimentos secuenciales (coordinate descent → config ganadora)

Estos experimentos **se ganan el orden**: muestran un *camino* desde una configuración inicial
naive hasta la receta ganadora de la presentación, moviendo **un eje a la vez** y acumulando el
ganador de cada paso. Responden la pregunta "¿cómo llegamos a esta config?" sin inventar una
cronología falsa: es un **análisis de coordinate descent** que converge a un **punto fijo**.

> No reemplaza a [`../ej1a_autoencoder/`](../ej1a_autoencoder/) (que presenta las ablaciones
> ancladas en el ganador). Esto es el complemento: el camino hasta ese ganador.

## Config inicial (punto de partida)

`35-20-2-20-35` · tanh/identity/tanh/sigmoid · **BCE** (heredado del TP3 + teoría binaria) ·
**SGD(0.5)** (arranque naive) · 6000 épocas · full-batch (32) · `seed=0`.

## El camino (cada paso actualiza la config acumulada)

| paso | eje | valores | gana | ¿cambia config? |
|---|---|---|---|---|
| exp1 | linealidad | no-lineal vs lineal vs PCA(2) | no-lineal | no (confirma) |
| exp2 | optimizador | SGD(0.5) / Momentum(0.1) / Adam(0.01) | **Adam(0.01)** | **sí** |
| exp3 | learning rate | 0.0003 / 0.01 / 0.3 | 0.01 | no (confirma) |
| exp4 | capa oculta | (), 10, 20, 30, 20+20 | 20 (parsimonia) | no (confirma) |
| exp5 | latente | 1, 2, 3 | 2 (consigna) | no (confirma) |
| exp6 | activación | tanh / relu / sigmoid | tanh (empate→incumbente) | no |
| exp7 | loss | BCE / MSE | BCE | no (confirma) |

Regla de selección (Occam): objetivo `px_max ≤ 1`; entre las que lo logran, la más parsimoniosa.
Empates en píxeles → se mantiene el incumbente. Con esta regla el camino aterriza **exactamente**
en `35-20-2-20-35 · tanh · BCE · Adam(0.01)`, que es el ganador del HTML.

## Cómo correr

```bash
# requiere numpy, pandas, scipy, matplotlib (ver ../README.md)
cd ej1a_secuencial
python3 run_all.py                 # 1) corre exp1..exp7 en orden -> results/*.csv
python3 make_figures.py            # 2) genera figs/*.png desde esos CSV
# o un paso suelto (lee el state.json acumulado):
python3 exp2_optimizador.py

# prueba rápida (no converge, solo valida que corre):
TP5_EPOCHS=60 TP5_LOG_EVERY=20 python3 run_all.py
```

## Salidas (en `results/`)

- `state.json` — config acumulada (se actualiza en cada paso).
- `camino.csv` — traza del coordinate descent: paso, eje, ganador, si cambió la config y cómo quedó.
- `expN_curves.csv` — curvas por época (cada `LOG_EVERY`) por variante: `epoch, train_loss,
  px_max, px_mean, perfectas, leq1`. Para graficar convergencia.
- `expN_summary.csv` — métricas finales por variante (1 fila c/u). Para barras/tablas.

Variables de entorno: `TP5_EPOCHS` (default 6000), `TP5_LOG_EVERY` (default 50).

## Figuras (`figs/` — carpeta propia, no pisa `../ej1a_autoencoder/figs`)

`make_figures.py` genera una PNG por paso + el resumen del camino:

- `fig_exp1_linealidad.png` — barras px_max: no-lineal ≪ lineal ≈ PCA.
- `fig_exp2_optimizador.png` — curvas de loss + barras finales (gana Adam).
- `fig_exp3_learning_rate.png` — curvas de loss + barras (0.01 justo, 0.3 no aprende).
- `fig_exp4_arquitectura.png` — px_max por capa oculta, con nº de params (★ gana 20).
- `fig_exp5_latente.png` — px_max + letras perfectas (1D solo aprende un subconjunto).
- `fig_exp6_activacion.png` — curvas de loss (las tres → 0 px, distinta velocidad).
- `fig_exp7_loss.png` — px_max BCE vs MSE.
- `fig_camino.png` — error máximo de la config acumulada tras cada paso (cae a 0 al adoptar Adam).
