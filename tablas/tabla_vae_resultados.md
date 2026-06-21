# Cómo leer `tabla_vae_resultados.png` — Experimento del VAE (barrido de β)

**Qué es esta tabla:** el experimento central del VAE. A diferencia de 1a/1b (que tienen muchos barridos), acá
el barrido principal es **uno solo: β** (el peso del término KL). Las otras dos filas **no son experimentos
nuevos** — son **dos formas de mirar el mismo barrido**.

**Cómo se lee cada fila:**
- **E12 · beta · `0, 0.5, 1, 4`** — entrenamos 4 VAEs, uno por β, y muestreamos de N(0,I). **β=0** (sin KL) sale
  **ruido**; **β=1** equilibrio (emojis válidos); **β=4** sobre-regularizado. → el KL es lo que hace generable el latente.
- **`—` recon (% px) por β · `0 / 1 / 4`** — la misma corrida, mirada por el **error de reconstrucción**:
  **2.6 % · 3.2 % · 3.7 %**. → a más β, peor reconstrucción.
- **`—` latente std por β · `0 / 1 / 4`** — la misma corrida, mirada por **cuán N(0,I) es el latente** (su desvío):
  **11.5 · 1.17 · 1.06**. → a más β, el **desvío marginal** del latente tiende a 1 (ojo: std→1 no es lo mismo
  que el agregado *sea* N(0,I) — conserva 5 cúmulos de clase; ver E13/`fig_e13`).

El `—` en la columna Exp significa "misma corrida que E12, otra métrica" (no un experimento aparte).

## La gran conclusión
β es una **perilla con trade-off**: subirlo hace que el KL empuje cada `q(z|x)` individual hacia N(0,I) (el
agregado conserva 5 cúmulos de clase, no es una sola gaussiana), lo que permite generar, pero empeora la
reconstrucción. **β=1** es el punto de equilibrio: reconstruye bien y genera. β=0 reconstruye mejor pero **no
puede generar** (sin KL, cada `q(z|x)` queda dispersa lejos de N(0,I)) — esa es la razón de ser del VAE.
