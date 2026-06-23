# Ejercicio 1b — Denoising Autoencoder

**Objetivo (enunciado):** plantear una arquitectura conveniente para denoising y explicarla;
distorsionar las entradas en distintos niveles y estudiar la capacidad de eliminar el ruido.

**Arquitectura elegida:** **35-30-10-30-35** (tanh / identity / sigmoid · BCE · Adam(0.01)).
Coordinate descent (igual que 1a): partiendo de la base de 1a (capa oculta 20), primero barremos el
**cuello** → 10 (no 2: el denoising necesita más capacidad latente que la compresión pura, ver E9);
con el cuello fijo en 10 barremos el **ancho de la capa oculta** {20,30,35,40} → **30** por parsimonia
(ver E9b). Entrenamiento: entrada con **bit-flip fresco cada época**, target = patrón limpio
(el bit-flip es el ruido natural para imágenes binarias).

## Cómo correr
```bash
python3 ej1b_denoising/run_experiments.py   # entrena DAEs, guarda results/*.csv + dae_champion.npz
python3 ej1b_denoising/make_figures.py      # genera figs/*.png
```
Semilla fija (`SEED=0`). Evaluación robusta: 30 realizaciones de ruido × 32 letras por nivel.

## Experimentos y conclusiones

| Exp | Qué varía | Resultado clave | Figura |
|-----|-----------|-----------------|--------|
| **E9**  | ancho de cuello {2,5,10,20} (base 35-20) | cuello 2 → **44% ≤1px**; 10 → **54%**; 20 satura | `fig_e9` |
| **E9b** | ancho de capa oculta {20,30,35,40} (cuello 10) | 20 → **54%**; 30 → **63%**; 35/40 plateau → **30** | `fig_e9b` |
| **E10** | ruido train {5,15,30%} × test | **trade-off**: cada nivel óptimo en su rango de test | `fig_e10` |
| **E11** | tripletes cualitativos | recupera bien a 10–20%, se degrada a 30% | `fig_e11` |
| **ganador** | 35-30-10-30-35 a 15 000 ep | **80% ≤1px a 15% ruido** (92% a 10%, 64% a 20%) | `fig_e_champion` |

### Lo que enseña cada cosa
- **E9 — por qué NO el cuello 2D de 1a:** con cuello 2 sólo se recupera el 44% de las letras a 20% de
  ruido; ensanchar a 10 sube a 54%. Más allá de 10 no hay ganancia (20 satura, el cuello deja de ser
  el factor limitante). Justifica subir el cuello a 10.
- **E9b — por qué la capa oculta 30:** con el cuello ya fijo en 10, barremos el ancho de la capa oculta.
  20 recupera 54%, 30 sube a 63%; 35 y 40 forman una meseta (no mejoran de forma significativa). Por
  **parsimonia** elegimos 30 (la más chica de la meseta). Esto "gana" el ancho que antes estaba sin
  justificar (era 25, nunca barrido). Corré `python3 exp_hidden_sweep.py`.
- **E10 — el ruido de entrenamiento es un hiperparámetro con trade-off:** entrenar con poco ruido
  (5%) da el mejor resultado en test limpio pero se rompe ante mucho ruido; entrenar con mucho (30%)
  es robusto a ruido alto pero peor en limpio. Las curvas se **cruzan** → 15% es buen compromiso.
- **E11:** evidencia visual de que el denoiser reconstruye el patrón limpio desde la versión corrupta.
- **Ganador reforzado:** el DAE 35-30-10-30-35 reentrenado a 15 000 épocas recupera el **80 % de las letras
  con ≤1 px a su nivel de ruido (15 %)** y 92 % a 10 % — el número contundente del denoiser (`fig_e_champion`).
