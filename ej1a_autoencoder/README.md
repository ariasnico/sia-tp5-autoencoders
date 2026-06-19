# Ejercicio 1a — Autoencoder básico (espacio latente 2D)

**Objetivo (enunciado):** aprender los 32 patrones de 7×5 de `font.h` en un espacio latente de
**2 dimensiones** con **error máximo de 1 píxel** por letra; graficar el latente 2D y generar
una letra nueva fuera del conjunto.

**Resultado:** ✅ ganador **35-20-2-20-35** (tanh / identity / sigmoid · BCE · Adam(0.01) · 6000 épocas)
→ **0 px de error en las 32 letras** (32/32 perfectas, supera holgado el ≤1px).

## Cómo correr
```bash
python3 ej1a_autoencoder/run_experiments.py   # entrena, guarda results/*.csv + curvas/ganador .npz
python3 ej1a_autoencoder/make_figures.py      # genera figs/*.png desde results/
python3 ej1a_autoencoder/diagnostics.py       # (defensa) E1 vs E3: confirma que sin capa oculta = PCA
```
Semilla fija (`SEED=0`). Lógica reutilizable en `tp5lib/`, métricas en `results/`, figuras en `figs/`
(separación lógica/análisis).

## Experimentos y conclusiones

| Exp | Qué varía | Resultado clave | Figura(s) |
|-----|-----------|-----------------|-----------|
| **E0** | exploración del dataset | par más parecido a 2px Hamming; densidad 3–35 px/letra | `fig_e0a`, `fig_e0b` |
| **E1** | lineal vs no-lineal vs PCA | **AE lineal ≡ PCA = 7.19 px**; no-lineal = **0 px** | `fig_e1` |
| **E2** | latente {1,2,3,5,8} | 1D falla (5 px, 18/32); **≥2D → 0 px** (codo) | `fig_e2` |
| **E3** | arquitectura `hidden` | sin capa oculta 14 px; **≥20 u → 0 px** | `fig_e3` |
| **E4** | SGD / Momentum / Adam | **Adam 0 px**, Momentum 1 px, SGD 5 px | `fig_e4` |
| **E5** | learning rate | 0.0003 lento (3 px), 0.01 justo (0 px), **0.3 no aprende (33 px)** | `fig_e5` |
| **E6** | activación oculta | tanh/relu/sigmoid → todas 0 px, distinta **velocidad** | `fig_e6` |
| **E7** | BCE vs MSE | **BCE 0 px**, MSE 2 px | `fig_e7` |
| **E8** | ganador | reconstrucción · latente 2D · generación · interpolación | `fig_e8a`–`fig_e8d` |

### Lo que enseña cada cosa (incluido lo que NO funcionó)
- **E1 — conexión con TP4 (PCA):** el AE lineal y PCA(2) dan *exactamente* el mismo error (7.188 px).
  Un autoencoder lineal **es** PCA; sólo la no-linealidad del encoder/decoder permite bajar a 0 px.
  PCA / AE lineal **no cumplen** el objetivo → motivan el modelo no-lineal.
- **E2 — el codo / caso "subconjunto":** con latente=1 sólo 18/32 perfectas (1D no puede meter las 32);
  es el caso que el enunciado permite documentar como "por qué no se aprende el set completo". 2D es el mínimo viable.
- **E3 — sin capa oculta = PCA:** `()` tiene encoder lineal → reconstrucción ≈ PCA (max 14 px / prom 6.84).
  *No* es "peor que E1": es la misma barrera lineal (E1 prom 7.19 / max 15) vista con otra métrica. El
  diagnóstico multi-semilla (`diagnostics.py`, px_max `[14,15,16]`) confirma que es **estructural, no mínimo
  local**; lo que rompe la barrera es la capa oculta no-lineal (≥20 unidades → 0 px).
- **E5 — config que diverge:** `lr=0.3` queda atascado arriba (33 px) → se muestra explícitamente un caso que no funciona.
- **E8 — generación (req. 1a-4):** letras nuevas por barrido del latente e interpolación `a → o`.
