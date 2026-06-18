# TP5 — Autoencoders · Presentación

> Cada slide es **auto-contenida**: ruta de figura + mensaje + conclusión. La narrativa sigue **el camino**
> (qué probamos, qué funcionó y qué NO). Marcadas con 🔹**(opcional)** las 4 slides recortables si falta tiempo.
> Figuras en `ej1a_autoencoder/figs/`, `ej1b_denoising/figs/`, `ej2_vae/figs/`. Detalle: READMEs por ejercicio.

---

## 1 · Portada + objetivo
- **Mensaje:** TP5 = Autoencoders sobre `font.h` (1a/1b) + VAE generativo con emojis (2). Todo **numpy from
  scratch** sobre la librería MLP de TP3; semillas fijas; lógica (`tp5lib/`) separada del análisis (`ejXX/`).
- **Conclusión:** un autoencoder es un MLP con `target=input` y un cuello angosto.

## 2 · El dataset (E0)
- **Figura:** `fig_e0a_dataset_letters.png`.
- **Mensaje:** 32 caracteres de 7×5 (35D). Comprimir a 2D es 17:1; hay pares casi idénticos (Hamming mín 2 px).
- **Conclusión:** problema no trivial → motiva estudiar arquitectura y optimización.

## · 1a — Setup experimental (cómo lo corrimos)
- **Tablas:** `tablas/tabla_1a_config.png` (arquitectura, optimizador, épocas, seeds…) · `tablas/tabla_1a_resultados.png` (qué barrimos en cada experimento + resultado).
- **Mensaje:** dejamos explícitos los hiperparámetros y el porqué de cada elección, no solo los gráficos.

## 3 · ¿Hace falta no-linealidad? AE lineal vs PCA (E1)
- **Figura:** `fig_e1_linear_vs_pca.png`.
- **Mensaje:** el AE lineal y PCA(2) dan el **mismo** error (prom 7.19 px); el AE no-lineal llega a 0.
- **Conclusión:** un AE lineal **es** PCA; la no-linealidad es lo que habilita el ≤1 px. *(conexión con TP4)*

## 4 · El codo de la dimensión latente (E2)
- **Figura:** `fig_e2_latent_elbow.png`.
- **Mensaje:** latente=1 sólo aprende 18/32 (caso "subconjunto" del enunciado); desde 2D → 0 px.
- **Conclusión:** 2D es el mínimo viable → justifica el requisito del enunciado.

## 5 · Arquitectura: capacidad del encoder (E3)
- **Figura:** `fig_e3_architecture.png`.
- **Mensaje:** sin capa oculta el encoder es **lineal** (≈PCA: max 14 px); con ≥20 unidades no-lineales → 0 px.
- **Conclusión:** lo que rompe la barrera de PCA es la capa oculta no-lineal, no el tamaño del cuello.
  *(Ojo defensa: "14 px" de E3 es el máximo; 7.19 de E1 es el promedio — ver DEFENSA.md.)*

## 6 · Optimización (E4) 🔹**+ (opcional) lr (E5) y activación (E6)**
- **Figuras:** `fig_e4_optimizers.png` · `fig_e5_lr.png` 🔹 · `fig_e6_activation.png` 🔹.
- **Mensaje:** Adam converge ~100× más bajo que SGD (0 vs 5 px). 🔹 lr=0.3 **no aprende** (33 px); las 3
  activaciones llegan a 0 px pero a distinta velocidad.
- **Conclusión:** la optimización importa tanto como la arquitectura; mostramos también lo que **falla** (lr alto).

## 7 · Función de pérdida: BCE vs MSE (E7)
- **Figura:** `fig_e7_loss.png`.
- **Mensaje:** para píxeles binarios, BCE reconstruye perfecto (0 px); MSE deja 2 px.
- **Conclusión:** la pérdida correcta (log-verosimilitud Bernoulli) es parte del diseño.

## 8 · Campeón: cumple el objetivo (E8)
- **Figuras:** `fig_e8a_reconstructions.png` (0 px) · `fig_e8b_latent2d.png` (con zoom legible, req. 1a-3).
- **Mensaje:** 35-20-2-20-35 → **32/32 letras perfectas en 2D**; el latente ordena las 32 letras.
- **Conclusión:** ✅ objetivo del enunciado (≤1 px) superado.

## 9 · Campeón: generación de letras nuevas (E8)
- **Figuras:** `fig_e8c_generation_grid.png` · `fig_e8d_interpolation.png`.
- **Mensaje:** barriendo el latente y decodificando salen letras que **no** están en el set; interpolación a→o.
- **Conclusión:** ✅ generación (req. 1a-4) — el latente es continuo y navegable.

## · 1b — Setup experimental
- **Tablas:** `tablas/tabla_1b_config.png` · `tablas/tabla_1b_resultados.png`.
- **Mensaje:** el entrenamiento denoising (entrada ruidosa → target limpio), las épocas, y qué se barrió (cuello, ruido de train).

## 10 · Denoising: ¿por qué ensanchar el cuello? (E9)
- **Figura:** `fig_e9_bottleneck.png`.
- **Mensaje:** cuello 2 recupera 48 % de letras; 10 → 59 %; 20 ya no mejora.
- **Conclusión:** el cuello 2D de 1a es malo para limpiar ruido → se elige cuello 10.

## 11 · Denoising: trade-off del ruido + campeón (E10, campeón)
- **Figuras:** `fig_e10_train_noise.png` (curvas que se cruzan) · `fig_e_champion.png`.
- **Mensaje:** poco ruido = frágil, mucho ruido = peor en limpio (se cruzan). Campeón (cuello 10, 15 000 ep):
  **81 % ≤1 px a 15 % de ruido** (92 % a 10 %).
- **Conclusión:** el ruido de train es un hiperparámetro con trade-off; el campeón limpia con contundencia.

## 12 · Denoising: evidencia visual (E11) 🔹**(opcional)**
- **Figura:** `fig_e11_triplets.png`.
- **Mensaje:** limpio / ruidoso / recuperado para varias letras y niveles.
- **Conclusión:** el denoiser reconstruye el patrón limpio desde la versión corrupta.

## · VAE — Setup experimental
- **Tablas:** `tablas/tabla_vae_config.png` · `tablas/tabla_vae_resultados.png`.
- **Mensaje:** arquitectura del VAE, dataset de emojis y augmentaciones, y el barrido de β.

## 13 · VAE: dataset de emojis + la decisión
- **Figura:** `ej2_vae/figs/contact_sheet.png`.
- **Mensaje / lo que NO funcionó:** OpenMoji *black* (contornos) se confunde a baja res (1-NN acc 0.45);
  el **alpha de color** (siluetas rellenas) sube a 0.95. 5 emojis distintos, dataset reproducible offline.
- **Conclusión:** elegimos el dato con criterio y lo dejamos self-contained.

## 14 · VAE: barrido de β y por qué el KL importa (E12, E16)
- **Figuras:** `fig_e12_beta_sampling.png` · `fig_e16_ae_vs_vae.png`.
- **Mensaje:** β=0 (sin KL) samplea ruido y deja el latente disperso (±30); el KL lo organiza como N(0,I).
- **Conclusión:** el término KL es la razón de ser del esquema variacional (req. 2b).

## 15 · VAE: latente, generación y atlas (E13, E14, E17)
- **Figuras:** `fig_e13_latent.png` · `fig_e14_recon_gen.png` · `fig_e17_vae_atlas.png` (nuevo).
- **Mensaje:** las 5 clases se agrupan dentro de N(0,I); se generan **emojis nuevos juzgables** desde N(0,I)
  (semilla 26, que cubre las 5 clases); el atlas muestra la cobertura continua del latente.
- **Conclusión:** ✅ Generative Autoencoder (req. 2c).

## 16 · Curvas recon vs KL (E15) 🔹**(opcional)**
- **Figura:** `fig_e15_recon_kl.png`.
- **Mensaje:** a más β, menor KL (latente más N(0,I)) y mayor recon-loss.
- **Conclusión:** β es una perilla con trade-off explícito; β=1 es el equilibrio.

## 17 · Conclusiones
- AE no-lineal ≤1 px en 2D (PCA no llega); denoising necesita más cuello y el ruido de train es un trade-off;
  el VAE resuelve la generación organizando el latente como N(0,I). Todo numpy from scratch, reproducible.
- Preguntas de defensa preparadas en `DEFENSA.md`.

---
### Estado de las 3 decisiones (cerradas)
- **Pulido de figuras:** ✅ hecho (estilo central `tp5lib/plotstyle.py`; `fig_e8b` con zoom; `fig_e14` con
  etiquetas de fila; `fig_e12`/`fig_e14` con semilla 26 que muestra las 5 clases).
- **Grilla del latente VAE:** ✅ agregada (`fig_e17_vae_atlas.png`, slide 15).
- **Orden:** 13 slides núcleo + 4 opcionales marcadas (E5, E6, E11, E15) recortables por tiempo.
