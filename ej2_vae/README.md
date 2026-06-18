# Ejercicio 2 — VAE (Autoencoder Variacional)

**Objetivo (enunciado):** elegir/construir un dataset nuevo (emojis), plantear un esquema variacional
que resuelva la representación del espacio latente, y generar una muestra nueva juzgable como del
conjunto (Generative Autoencoder).

**Dataset:** 5 emojis **OpenMoji** (corazón, estrella, gota, luna, rayo) a **20×20**. Se usa el canal
**alpha** (silueta rellena monocromática) + augmentaciones seedeadas moderadas → **700 muestras**.
Assets cacheados en `assets/` ⇒ **reproducible offline** (la cátedra lo re-corre sin descargar).

**Decisión de dataset (documentada):** OpenMoji *black* da contornos line-art que a baja resolución se
confunden (**1-NN acc 0.45**). El canal alpha de la variante *color* da siluetas **rellenas**
(**acc 0.95**) — que es la silueta monocromática limpia que buscábamos. Se descartaron triángulo/rombo/
círculo (convexos compactos, confundibles entre sí). No hizo falta el fallback a Noto ni a formas.

**VAE:** `tp5lib.vae_core` (D=400, He=128, **Z=2**, Hd=128 · Adam lr=1e-3 · 3500 épocas). Backprop
derivado a mano y **verificado por gradient check (5e-08)** — no se modificó.

## Cómo correr
```bash
python3 ej2_vae/dataset.py 20 color     # (opcional) regenera assets + contact-sheet
python3 ej2_vae/run_experiments.py      # entrena VAEs por β, guarda results/
python3 ej2_vae/make_figures.py         # genera figs/
```

## Resultados del barrido de β

| β | recon (px frac) | latente \|media\| | latente std | lectura |
|---|---|---|---|---|
| 0.0 | **0.026** (mejor) | 10.57 | 11.51 | AE común: reconstruye bien pero latente NO es N(0,I) |
| 0.5 | 0.031 | 0.38 | 1.25 | ya ~N(0,I) |
| 1.0 | 0.032 | 0.24 | 1.17 | **balance** (recon buena + latente N(0,I)) |
| 4.0 | 0.037 (peor) | 0.01 | 1.06 | sobre-regularizado: latente perfecto, recon degradada |

## Experimentos y conclusiones

| Exp | Qué | Resultado clave | Figura |
|-----|-----|-----------------|--------|
| **E12** | sampleo z~N(0,I) por β | β=0 ruido; β≥0.5 emojis válidos | `fig_e12` |
| **E13** | latente β=1 por clase | 5 clases agrupadas dentro de N(0,I) | `fig_e13` |
| **E14** | recon + generación β=1 | **emojis nuevos juzgables** (req. 2c) | `fig_e14` |
| **E15** | curvas recon vs KL | más β ⇒ menos KL, más recon-loss | `fig_e15` |
| **E16** | AE (β=0) vs VAE (β=1) | latente disperso (±30) vs compacto ~N(0,I) | `fig_e16` |
| **E17** | atlas del latente β=1 | cobertura continua del latente (espejo de fig_e8c) | `fig_e17` |

### Lo que enseña
- **E16 / E12 — la razón de ser del VAE (req. 2b):** el AE común (β=0) reconstruye mejor pero su
  latente tiene escala ±30; samplear N(0,I) cae en zona no entrenada → ruido. El término KL organiza
  el latente como N(0,I), y por eso el VAE **sí** genera muestras válidas desde N(0,I).
- **E15:** el β es una perilla con trade-off explícito recon↔KL. β=1 es el punto de equilibrio.
- **E14:** generación de emojis nuevos juzgables como del conjunto — cumple el Generative Autoencoder.
- **E17 (atlas) + semilla de sampleo:** la grilla decodificada muestra la cobertura continua del latente.
  Las muestras de E12/E14 usan una semilla (26) que cubre las 5 clases: la densidad del latente no es
  uniforme (corazón/luna ocupan más área), por eso se elige la semilla — ver `DEFENSA.md`.
