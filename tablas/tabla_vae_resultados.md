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
  **11.5 · 1.17 · 1.06**. → a más β, el latente se parece más a N(0,I) (std → 1).

El `—` en la columna Exp significa "misma corrida que E12, otra métrica" (no un experimento aparte).

## La gran conclusión
β es una **perilla con trade-off**: subirlo organiza el latente como N(0,I) (permite generar) pero empeora la
reconstrucción. **β=1** es el punto de equilibrio: reconstruye bien y genera. β=0 reconstruye mejor pero **no
puede generar** (su latente no es N(0,I)) — esa es la razón de ser del VAE.
