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

**Referencia (cátedra):** implementa en **numpy desde cero** el VAE de `KerasAutoencoders.ipynb` / Langr
*"GANs in action"* (Keras sobre MNIST), porque el enunciado prohíbe Keras/TF. **Mismas fórmulas**:
reparam `z = μ + exp(½·logvar)·ε`, KL `−½·Σ(1 + logvar − μ² − exp(logvar))`, recon **BCE sumada** sobre
los píxeles, loss = recon + KL (con β=1 es idéntica), latente 2D. **Extendido** con barrido de β (β-VAE)
y dataset propio de emojis; backprop verificado con gradient check 5e-08. Diferencias justificadas: tanh
vs relu, emojis 20×20 vs MNIST 28×28. Ver `DEFENSA.md`.

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
| **E12** | sampleo z~N(0,I) por β | β=0 ruido; β≥0.5 muestras reconocibles | `fig_e12` |
| **E13** | latente β=1 por clase | KL empuja cada q(z\|x) a N(0,I); el agregado conserva 5 cúmulos | `fig_e13` |
| **E14** | recon + generación β=1 | **muestras nuevas (no copias) reconocibles** (req. 2c) | `fig_e14` |
| **E15** | curvas recon vs KL | más β ⇒ menos KL, más recon-loss | `fig_e15` |
| **E16** | AE (β=0) vs VAE (β=1) | latente disperso vs compacto (cada q(z\|x)→N(0,I)) | `fig_e16` |
| **E17** | atlas del latente β=1 | cobertura continua del latente (espejo de fig_e8c) | `fig_e17` |
| **E18** | sampleo honesto (semilla 0) | cobertura media 85 % de clases (4.3/5) sobre 200 semillas | `fig_e18` |

### Lo que enseña
- **E16 / E12 — la razón de ser del VAE (req. 2b):** el AE común (β=0) reconstruye mejor pero su
  latente queda disperso; samplear N(0,I) cae en zona no entrenada → ruido. El término KL empuja cada
  q(z|x) **individual** hacia N(0,I) (el agregado conserva estructura de clases, no es una sola nube),
  y por eso el VAE **sí** genera muestras válidas desde N(0,I).
- **E15:** el β es una perilla con trade-off explícito recon↔KL. β=1 es el punto de equilibrio.
- **E14:** genera muestras **nuevas (no copias) reconocibles** — variaciones continuas de los 5
  prototipos — cumple el Generative Autoencoder.
- **E17 (atlas) + semilla de sampleo:** la grilla decodificada muestra la cobertura continua del latente.
  Las muestras de E12/E14 usan una semilla (26) elegida y documentada que cubre las 5 clases. La densidad
  del latente no es uniforme: medido sobre 200 semillas (E18, `fig_e18`), una tirada cubre en promedio el
  **85 % de las clases (4.3/5)** y el reparto es desigual (estrella ≈33 %, luna ≈23 %, gota ≈19 %,
  rayo ≈14 %, corazón ≈12 %). Por eso se elige la semilla para ilustrar — ver `DEFENSA.md` y `fig_e18`.
