# SIA TP5 — Autoencoders (numpy from scratch)

Trabajo Práctico 5 de **Sistemas de Inteligencia Artificial** (ITBA). Autoencoder básico, Denoising
Autoencoder y VAE generativo, **todo en numpy puro** sobre la librería MLP de TP3 (sin PyTorch/TensorFlow).

## Resultados

- **1a — Autoencoder:** la red `35-20-2-20-35` reconstruye las 32 letras de `font.h` con **0 px de error**
  en un espacio latente **2D** (objetivo del enunciado: ≤1 px).
- **1b — Denoising:** el denoiser (cuello 10) recupera el **81 % de las letras con ≤1 px** a 15 % de ruido
  (92 % a 10 %).
- **2 — VAE:** genera **emojis nuevos** desde `N(0,I)`; el término KL organiza el latente (sin él, el
  muestreo cae en zona muerta → ruido).

## Estructura

```
mlp/                  librería MLP de TP3 (red feedforward + backprop, numpy puro) + tests
tp5lib/               lógica reutilizable: fonts, autoencoder, vae_core (VAE con gradient check), plotstyle
ej1a_autoencoder/     ejercicio 1a — run_experiments.py · make_figures.py · diagnostics.py · results/ · figs/ · README
ej1b_denoising/       ejercicio 1b — denoising AE
ej2_vae/              ejercicio 2  — VAE con dataset de emojis (assets self-contained)
PRESENTACION.md       slides (figura + mensaje + conclusión)
DEFENSA.md            preguntas y respuestas de defensa
font.h                dataset (32 caracteres de 7×5)
```

Cada ejercicio separa **lógica** (`run_experiments.py` → CSV/modelos) de **análisis**
(`make_figures.py` → figuras), y tiene su propio README con el detalle de los experimentos.

## Cómo correr

```bash
python3 -m pytest mlp/tests/ -q                 # tests de la librería (61 passed)
python3 tp5lib/vae_core.py                       # gradient check del VAE (~5e-08)
for e in ej1a_autoencoder ej1b_denoising ej2_vae; do
  python3 $e/run_experiments.py && python3 $e/make_figures.py
done
```

Semillas fijas en todo. El dataset de emojis está cacheado en `ej2_vae/assets/` (reproducible offline);
PIL/scipy se usan sólo para preparar imágenes, no para los modelos.
