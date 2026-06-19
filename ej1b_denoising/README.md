# Ejercicio 1b — Denoising Autoencoder

**Objetivo (enunciado):** plantear una arquitectura conveniente para denoising y explicarla;
distorsionar las entradas en distintos niveles y estudiar la capacidad de eliminar el ruido.

**Arquitectura elegida:** **35-25-10-25-35** (tanh / identity / sigmoid · BCE · Adam(0.01)).
Cuello = **10** (no 2): el denoising necesita más capacidad latente que la compresión pura de 1a
(ver E9). Entrenamiento: entrada con **bit-flip fresco cada época**, target = patrón limpio
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
| **E9**  | ancho de cuello {2,5,10,20} | cuello 2 → **48% ≤1px**; 10 → **59%**; 20 satura | `fig_e9` |
| **E10** | ruido train {5,15,30%} × test | **trade-off**: cada nivel óptimo en su rango de test | `fig_e10` |
| **E11** | tripletes cualitativos | recupera bien a 10–20%, se degrada a 30% | `fig_e11` |
| **ganador** | cuello 10 a 15 000 ep | **81% ≤1px a 15% ruido** (92% a 10%, 64% a 20%) | `fig_e_champion` |

### Lo que enseña cada cosa
- **E9 — por qué NO el cuello 2D de 1a:** con cuello 2 sólo se recupera el 48% de las letras a 20% de
  ruido; ensanchar a 10 sube a 59% y baja el error de 4.2→2.6 px. Más allá de 10 no hay ganancia
  (el cuello deja de ser el factor limitante). Justifica la arquitectura.
- **E10 — el ruido de entrenamiento es un hiperparámetro con trade-off:** entrenar con poco ruido
  (5%) da el mejor resultado en test limpio pero se rompe ante mucho ruido; entrenar con mucho (30%)
  es robusto a ruido alto pero peor en limpio. Las curvas se **cruzan** → 15% es buen compromiso.
- **E11:** evidencia visual de que el denoiser reconstruye el patrón limpio desde la versión corrupta.
- **Ganador reforzado:** el DAE cuello 10 reentrenado a 15 000 épocas recupera el **81 % de las letras con
  ≤1 px a su nivel de ruido (15 %)** y 92 % a 10 % — el número contundente del denoiser (`fig_e_champion`).
