# Cómo leer `tabla_vae_config.png` — Configuración del VAE (2)

**Qué es esta tabla:** la **receta del VAE** que usamos para generar emojis. Cada fila es **un valor fijo**, todos
juntos. El único barrido está en `tabla_vae_resultados.png` (el de β).

## Fila por fila

- **Arquitectura `D=400 · He=128 · Z=2 · Hd=128`** — la forma del VAE: entrada **D=400** (imágenes 20×20) →
  capa del encoder **He=128** → **cuello Z=2** (latente 2D, para poder graficarlo) → capa del decoder **Hd=128**
  → salida 400. A diferencia de 1a, el encoder no da un punto sino una **distribución** (μ, logσ²). Una sola red.
- **Loss `recon(BCE) + beta·KL`** — la pérdida tiene **dos términos**: la reconstrucción (BCE) y el **KL** (que
  empuja el latente hacia N(0,I)). `beta` pesa el KL; **el campeón usa β=1** (el barrido de β es E12).
- **Optimizador `Adam (lr=1e-3)`** — un valor.
- **Épocas / batch `3500 / 128`** — un valor cada uno.
- **Dataset `5 emojis · 700 · 20×20`** — 5 clases (corazón, estrella, gota, luna, rayo), 140 imágenes cada una =
  700, en 20×20 grises. Usamos el **canal alpha** del emoji (la silueta rellena); quedaron bien distinguibles
  (1-NN acc 0.95).
- **Augmentación `rot ±15° · trasl ±2 · ruido 0.03`** — a cada emoji base le aplicamos rotaciones/traslaciones/
  ruido **chicos y seedeados** para tener variedad (si no, el VAE vería 5 imágenes fijas). Moderadas a propósito,
  para que las muestras sigan siendo "juzgables".
- **Sampleo `semilla 26`** — para las figuras de generación elegimos (entre varias) la semilla que cubre las 5
  clases, porque el latente no es uniforme (corazón/luna ocupan más área).
- **Seed `0`** — semilla fija del entrenamiento.

## Qué nos dio
Con β=1, el VAE reconstruye bien (3.2 % px) y su latente queda ~N(0,I), así que **genera emojis nuevos juzgables
desde N(0,I)** (el objetivo del enunciado, 2c).

## Cómo conecta con los experimentos
El único hiperparámetro que barrimos es **β** (E12); el resto de la receta queda fijo. Ver `tabla_vae_resultados.png`.
