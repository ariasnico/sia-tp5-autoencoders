# SIA · TP5 — Autoencoders (numpy desde cero)

Trabajo Práctico 5 de **Sistemas de Inteligencia Artificial** (ITBA). Tres modelos, **todo en numpy puro**
sobre la librería MLP del TP3 — **sin PyTorch, sin TensorFlow, sin Keras**:

1. **Autoencoder básico** (`ej1a`) — comprime las 32 letras de `font.h` a un mapa de **2 números** y las reconstruye con **0 píxeles de error**.
2. **Denoising Autoencoder** (`ej1b`) — recibe letras con ruido y las limpia: **81 % de las letras a ≤1 px de error con 15 % de ruido**.
3. **VAE generativo** (`ej2_vae`) — aprende un mapa ordenado de 5 emojis y **genera muestras nuevas** (no copias) pidiéndole puntos al azar.

> **Este README es un manual de reproducción.** Cualquier persona (o agente) puede, **sin leer el código**,
> instalar las dependencias, correr todos los experimentos y obtener exactamente los números que se reportan.
> Todos los resultados ya están commiteados en `*/results/` y `*/figs/`, así que también se pueden inspeccionar sin correr nada.

---

## Índice
1. [Instalación (1 comando)](#1-instalación)
2. [Reproducir TODO](#2-reproducir-todo)
3. [Qué debería dar (chequeos rápidos)](#3-qué-debería-dar-chequeos-rápidos)
4. [Comando por comando (qué hace, qué genera, cuánto tarda)](#4-comando-por-comando)
5. [Mapa de experimentos → figura → conclusión](#5-mapa-de-experimentos)
6. [Ver la presentación](#6-ver-la-presentación)
7. [Estructura del repo](#7-estructura-del-repo)
8. [Notas importantes](#8-notas-importantes)

---

## 1. Instalación

Python **3.8+** y seis paquetes (no hay deep-learning de por medio):

```bash
pip install numpy scipy matplotlib pillow pandas pytest
```

| Paquete | Para qué se usa |
|---------|-----------------|
| **numpy** | TODO el cálculo (redes, backprop, VAE). Es el único imprescindible para los modelos. |
| **pandas** | Guardar los resultados en CSV. |
| **matplotlib** | Generar todas las figuras `.png`. |
| **scipy** + **pillow** | Sólo el dataset del VAE (rotar/trasladar emojis y leer los PNG). |
| **pytest** | Correr los 61 tests de la librería. |

> **No hace falta internet.** Los emojis del VAE ya están cacheados en `ej2_vae/assets/`. (Sólo se
> descargarían de OpenMoji si esa carpeta estuviera vacía.)

---

## 2. Reproducir TODO

Parado en la raíz del repo:

```bash
# (0) Validar la librería y el VAE
python3 -m pytest mlp/tests/ -q          # 61 tests, ~4 s
python3 tp5lib/vae_core.py               # chequeo de las cuentas del VAE, <1 s

# (1) Correr los tres ejercicios: entrenar + generar figuras
for e in ej1a_autoencoder ej1b_denoising ej2_vae; do
  python3 $e/run_experiments.py          # entrena  -> $e/results/
  python3 $e/make_figures.py             # figuras  -> $e/figs/
done

# (2) (opcional) Regenerar las tablas-imagen
python3 tools/tablas_setup.py            # -> tablas/*.png
```

⏱️ **Tiempo total ≈ 55 min**, y casi todo es **`ej1b`** (≈ 40 min, ver abajo). El resto junto son ~12 min.
Si querés una corrida rápida, salteá `ej1b/run_experiments.py` (sus resultados ya están commiteados) y corré
sólo `ej1a` y `ej2_vae` (~10 min — el VAE solo ya son ~4–6 min según la máquina).

> **No hace falta correr nada para ver los resultados:** `results/` (CSV + modelos) y `figs/` (PNG) ya están
> en el repo. Correr los scripts los regenera: **idénticos** en 1a y 1b; el **VAE** puede variar levemente
> entre corridas (su muestreo es estocástico, ver §8 — las conclusiones se mantienen).

---

## 3. Qué debería dar (chequeos rápidos)

Si reprodujiste todo, estos son los números que tienen que salir (están fijados con semillas):

| Chequeo | Comando | Resultado esperado |
|---------|---------|--------------------|
| Tests de la librería | `pytest mlp/tests/ -q` | **61 passed** |
| Cuentas del VAE | `python3 tp5lib/vae_core.py` | `MAX REL ERR GLOBAL = 5.02e-08  (OK)` |
| **Campeón 1a** | (al final de `ej1a/run_experiments.py`) | `px_max=0  perfectas=32/32` |
| **Campeón denoiser** | (al final de `ej1b/run_experiments.py`) | `81 % ≤1px @ 15 % ruido` (92 % @ 10 %) |
| **VAE** | recon: `run_experiments.py` · cobertura (E18): `make_figures.py` | recon β=1 ≈ 3.2 % · cobertura ≈ 85 % *(varían levemente, muestreo estocástico)* |

Si alguno no coincide, algo del entorno cambió (versión de numpy, semilla, etc.).

---

## 4. Comando por comando

Cada `run_experiments.py` **entrena y escribe `results/`**; cada `make_figures.py` **lee `results/` y dibuja `figs/`**.
Siempre se corre `run_experiments` **antes** que `make_figures`.

### Validación
| Comando | Genera | Tarda | Imprime |
|---------|--------|-------|---------|
| `python3 -m pytest mlp/tests/ -q` | — | ~4 s | `61 passed` |
| `python3 tp5lib/vae_core.py` | — | <1 s | el error de las cuentas del VAE, `5.02e-08` |

### Ejercicio 1a — Autoencoder
| Comando | Genera | Tarda | Imprime |
|---------|--------|-------|---------|
| `python3 ej1a_autoencoder/run_experiments.py` | `results/` (7 CSV + 3 modelos `.npz` + `config_used.json`) | ~180 s | resultados E1–E8 + `Ganador: px_max=0 perfectas=32/32` |
| `python3 ej1a_autoencoder/make_figures.py` | `figs/` (16 PNG `fig_e0a…e8e`) | ~15 s | la lista de figuras |
| `python3 ej1a_autoencoder/diagnostics.py` | — (sólo imprime) | ~30 s | tabla que confirma *AE lineal = PCA* con 3 semillas |

### Ejercicio 1b — Denoising  ⚠️ el lento
| Comando | Genera | Tarda | Imprime |
|---------|--------|-------|---------|
| `python3 ej1b_denoising/run_experiments.py` | `results/` (3 CSV + 1 modelo `.npz` + `config_used.json`) | **~40 min** | E9/E10 + campeón `81 % ≤1px @15 %` |
| `python3 ej1b_denoising/make_figures.py` | `figs/` (4 PNG) | ~20 s | la lista de figuras |

> Tarda porque mide cada configuración promediando **30–50 realizaciones de ruido** (para que el número no
> dependa del azar). Sus `results/` ya están commiteados: si sólo querés las figuras, corré `make_figures.py` directo.

### Ejercicio 2 — VAE (emojis)
| Comando | Genera | Tarda | Imprime |
|---------|--------|-------|---------|
| `python3 ej2_vae/dataset.py 20 color` | `figs/contact_sheet.png` (y cachea emojis en `assets/`) | ~2 s | `dataset: (700, 400)` · `1-NN acc 0.95` |
| `python3 ej2_vae/run_experiments.py` | `results/` (1 CSV + 4 modelos `.npz` + curvas + `config_used.json`) | ~4–6 min | el barrido de β + `OK VAE` |
| `python3 ej2_vae/make_figures.py` | `figs/` (8 PNG `fig_e12…e18`, incl. el atlas `e17`) | ~30 s | métricas de E18 (cobertura ~85 %) |

> `dataset.py` es opcional: los emojis ya están cacheados. El primer argumento es el tamaño (20×20) y el
> segundo la variante (`color` — usa la silueta del canal alpha, que se distingue mejor que `black`).

### Tablas y presentación (opcional)
| Comando | Genera |
|---------|--------|
| `python3 tools/tablas_setup.py` | `tablas/` — las 6 tablas-imagen (config + resultados de cada ejercicio) |
| `cd presentacion && python3 build_offline.py` | `presentacion_offline.html` (slides self-contained, sin internet) |
| `cd presentacion && python3 build_experimentos.py` | `experimentos_offline.html` (galería self-contained) |

---

## 5. Mapa de experimentos

Para entender qué prueba cada experimento sin leer el código. Cada uno deja un CSV en `results/` y una figura en `figs/`.

### 1a — Autoencoder (red `35-20-2-20-35`)
| Exp | Qué prueba | Conclusión |
|-----|-----------|-----------|
| **E1** | ¿una red "recta" alcanza? | AE lineal = PCA = 7,2 px de error; **con no-linealidad → 0 px**. |
| **E2** | ¿de cuántos números necesita el mapa? | 1 número no alcanza (18/32); **con 2 → perfecto**. |
| **E3** | ¿cuánta "capacidad" oculta? | sin capa intermedia = PCA; **con ≥20 neuronas → 0 px**. |
| **E4** | ¿qué optimizador? | Adam **0 px** · Momentum 1 · SGD 5. |
| **E5** | ¿qué tan grande el paso de aprendizaje? | chico = lento (5 px) · justo = 0 px · **grande (0.3) no aprende (28 px)**. *Mostramos a propósito uno que falla.* |
| **E6** | ¿qué tipo de neurona? | las tres (tanh/relu/sigmoid) llegan a 0 px, cambia la velocidad. |
| **E7** | ¿cómo medir el error? | BCE **0 px** vs MSE 2 px. |
| **E8** | el ganador, junto | reconstrucción 0 px, mapa 2D separado por letra, genera mezclas suaves. |

### 1b — Denoising (red `35-25-{cuello}-25-35`)
| Exp | Qué prueba | Conclusión |
|-----|-----------|-----------|
| **E9** | ¿qué tan angosto el cuello? | 2 = 48 % · **10 = 59 %** · 20 satura → cuello 10. |
| **E10** | ¿cuánto ruido en el entrenamiento? | es un **equilibrio**: cada nivel rinde mejor en su rango. |
| **E11** | ejemplos visuales | limpia bien a 10–15 %, se degrada a 30 %. |
| **Campeón** | cuello 10, 15 000 épocas | **81 % de las letras a ≤1 px con 15 % de ruido** (92 % con 10 %). |

### 2 — VAE (5 emojis, mapa de 2 números)
| Exp | Qué prueba | Conclusión |
|-----|-----------|-----------|
| **E12** | samplear el mapa según β | β=0 → ruido; **β≥0.5 → emojis reconocibles**. |
| **E13** | el mapa ordenado (β=1) | cada emoji ocupa su zona, todas juntas en el centro. |
| **E14** | reconstruir + generar | **genera muestras nuevas (no copias)** reconocibles. |
| **E15** | el equilibrio del β | más orden = mapa prolijo pero imágenes un poco peores. |
| **E16** | AE vs VAE en el mapa | sin orden el mapa queda disperso; con orden, compacto. |
| **E17** | el mapa completo decodificado | los emojis se transforman de a poco, uno en otro. |
| **E18** | sampleo honesto (control) | una tirada al azar saca **~85 % de las clases** (4,3 de 5) — medido sobre 200 semillas. |

---

## 6. Ver la presentación

Carpeta `presentacion/`:

| Archivo | Qué es | Necesita internet |
|---------|--------|-------------------|
| **`presentacion_offline.html`** | Las slides, **todo embebido** (la mejor para mostrar). | **No** |
| `index.html` | Las slides (usa reveal.js desde internet). | Sí |
| `experimentos_offline.html` | Galería de todas las figuras E0–E18, embebida. | No |
| `experimentos.html` | La misma galería (desde internet). | Sí |

```bash
# La más simple: abrir el HTML offline en el navegador
xdg-open presentacion/presentacion_offline.html      # Linux
# open presentacion/presentacion_offline.html        # macOS
```

Dentro de las slides: las flechas avanzan; la tecla **`S`** abre las notas del orador.

---

## 7. Estructura del repo

```
mlp/                  librería de redes del TP3 (numpy puro). NO se modifica. 61 tests.
tp5lib/               código compartido del TP5:
  ├─ fonts.py           cargar font.h, agregar ruido (bit-flip), distancias entre letras
  ├─ autoencoder.py     armar el AE/denoiser, medir error en píxeles, entrenar
  ├─ vae_core.py        el VAE (con chequeo de cuentas). NO se modifica.
  └─ plotstyle.py       estilo común de las figuras
ej1a_autoencoder/     Autoencoder      · run_experiments.py · make_figures.py · diagnostics.py · results/ · figs/
ej1b_denoising/       Denoising AE     · run_experiments.py · make_figures.py · results/ · figs/
ej2_vae/              VAE (emojis)     · dataset.py · run_experiments.py · make_figures.py · assets/ · results/ · figs/
tablas/               6 tablas-imagen (config + resultados) + su explicación en .md
tools/tablas_setup.py genera las tablas-imagen
presentacion/         slides (reveal.js) + versiones offline
font.h                el dataset: 32 letras de 7×5 píxeles
```

Cada ejercicio separa **lógica** (`run_experiments.py` → CSV/modelos) de **dibujo** (`make_figures.py` → PNG),
y tiene su propio `README.md` con el detalle.

---

## 8. Notas importantes

- **Todo es numpy puro.** `scipy`/`pillow` se usan sólo para preparar las imágenes de los emojis, nunca para los modelos.
- **Reproducibilidad.** 1a y 1b tienen **semillas fijas → reproducibles al dígito** en la misma máquina. El
  **VAE** usa muestreo estocástico (el ruido ε del reparam no fija la semilla global de numpy): sus números
  varían **levemente** entre corridas (p. ej. recon β=1 ≈ 3.2–3.5 %), pero las conclusiones (recon sube con β,
  el modelo genera muestras nuevas) se mantienen — y el control honesto es **E18**. Entre versiones de
  numpy/BLAS también pueden moverse los últimos dígitos.
- **`tp5lib/vae_core.py` no se toca.** Su backprop (el reparam y la KL) está derivado a mano y verificado con un
  chequeo numérico de gradientes (`5.02e-08`, muy por debajo del umbral `1e-5`).
- **El VAE sigue la referencia de la cátedra.** Es el VAE de `KerasAutoencoders.ipynb` (libro de Langr,
  *"GANs in action"*, en Keras sobre MNIST) **reimplementado en numpy desde cero** (el enunciado prohíbe Keras/TF).
  Coinciden las fórmulas clave; lo extendimos con el barrido de β, un dataset propio de emojis y el chequeo de
  gradientes. Ver `ej2_vae/README.md` y `DEFENSA.md`.
- **Reproducibilidad de los lentos:** `ej1b` tarda ~40 min porque promedia muchas realizaciones de ruido; sus
  resultados ya están en `results/`.
