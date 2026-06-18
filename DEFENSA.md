# TP5 — Borrador de preguntas de defensa

> Borrador para refinar con el alumno. Respuestas de 1–2 líneas ancladas en los **configs y números
> reales** (no a mano alzada). Las "finas" están marcadas con ⚠️.

## Ejercicio 1a — Autoencoder

**¿Qué es exactamente el autoencoder acá?** Un MLP (la librería de TP3) con `target = input` y un cuello
de 2 neuronas: encoder 35→20→2, decoder 2→20→35. No se agregó nada a `mlp/`; sólo se orquestan sus
primitivas (`tp5lib/autoencoder.py`).

⚠️ **¿Por qué E1 (AE lineal) "da 7.19 px" pero E3 "sin capa oculta" "da 14 px"? ¿El segundo es peor?**
No: es una comparación de **métricas distintas**. 7.19 es el px **promedio** de E1; 14 es el px **máximo**
de E3. Con la misma métrica son equivalentes: E1 lineal = mean 7.19 / max 15; E3 `()` = mean 6.84 / max 14.
La causa común: **ambos tienen encoder lineal** (sin capa oculta), así que su capacidad de comprimir a 2D
es la de **PCA** — de hecho el AE lineal+MSE converge al óptimo PCA (verificado: 7.19 = el del SVD analítico).
La sigmoid de salida de E3 no agrega capacidad al *encoder*. El diagnóstico con 3 semillas (px_max de E3 =
`[14, 15, 16]`) confirma que es **estructural, no un mínimo local**. Lo que rompe la barrera de PCA es la
**capa oculta no-lineal**: E3 con `(20,)` → 0 px. (Reproducible: `python3 ej1a_autoencoder/diagnostics.py`.)

⚠️ **¿Por qué con latente=1 sólo aprende 18/32?** Con una sola dimensión el cuello debe ordenar las 32
letras sobre una **recta**; letras distintas colapsan a valores cercanos y el decoder no las separa
(px_max=5). Es el caso "subconjunto" que el enunciado permite documentar: 1D no alcanza, 2D sí (0 px).

**¿Por qué Adam y no SGD?** Misma arquitectura: SGD(0.5) queda en 5 px (loss 0.14), Adam(0.01) llega a
0 px (loss 6e-4). Adam adapta el paso por parámetro y escapa de la meseta donde SGD se estanca (E4).

⚠️ **¿Por qué BCE mejor que MSE en imágenes binarias?** Los píxeles son Bernoulli (0/1); BCE es su
log-verosimilitud natural y penaliza fuerte el error confiado (predecir 0.99 cuando es 0). MSE trata
todo como regresión continua y su gradiente se satura cerca de 0/1 con la sigmoid, dejando píxeles
ambiguos → MSE deja 2 px, BCE 0 (E7).

**¿La generación de "letras nuevas" es real?** Sí: se decodifican puntos del latente que **no** son
ninguna de las 32 (barrido de la grilla e interpolación a→o, fig_e8c/e8d). Son combinaciones continuas
aprendidas, no copias.

## Ejercicio 1b — Denoising

**¿Por qué el cuello 10 y no el 2 de 1a?** El denoising necesita más capacidad latente que la compresión
pura: con cuello 2 sólo se recupera el 48 % de las letras (test 20 %), con 10 sube a 59 %; más de 10 no
mejora (E9). El cuello 2 estaba optimizado para *comprimir*, no para *limpiar*.

**¿Cómo se entrena un denoiser?** Entrada con **bit-flip fresco cada época** (ruido natural para binario),
target = patrón limpio. La red aprende a mapear "ruidoso → limpio", no "x → x".

**¿Qué tan bueno es el campeón?** Reentrenado a 15 000 épocas (cuello 10, ruido de train 15 %): a su nivel
de ruido recupera **81 % de las letras con ≤1 px (72 % perfectas)**; a 10 % de ruido, 92 % (fig_e_champion).

**¿Por qué el ruido de entrenamiento es un trade-off?** Entrenar con poco ruido (5 %) da lo mejor en test
limpio pero se rompe ante mucho ruido; con mucho (30 %) es robusto pero peor en limpio. Las curvas se
**cruzan** (E10): no hay un único "mejor", depende del ruido esperado.

## Ejercicio 2 — VAE

**¿Qué cambia respecto del autoencoder de 1a?** El encoder produce una **distribución** por muestra
(μ, logσ²) en lugar de un punto; se muestrea z y se agrega el término **KL** a la pérdida. Eso convierte
el latente en algo del que se puede *generar*.

⚠️ **¿Qué hacen el reparametrization trick y el KL, y por qué sin KL no se puede samplear?**
El trick escribe `z = μ + σ·ε` con `ε~N(0,I)`: mantiene el azar **fuera** del grafo y permite backprop a
través del muestreo. El KL empuja cada `q(z|x)` hacia `N(0,I)`. Sin KL (β=0) el latente queda disperso
(escala ±30, fig_e16): muestrear `N(0,I)` cae en zonas **nunca entrenadas** → ruido. El KL es lo que hace
que `N(0,I)` sea un buen lugar para generar (β=1: latente std 1.17, fig_e13).

⚠️ **¿Por qué las generaciones se cargan a corazón/luna?** No es un bug: se muestrea `z~N(0,I)` uniforme,
pero las clases **no ocupan áreas iguales** del latente (fig_e13) — corazón y luna ocupan regiones más
amplias, así que un z aleatorio cae más seguido ahí. Por eso para las figuras se eligió, entre varias
semillas, una (la 26) que cubriera las 5 clases; la densidad desigual es real y esperada.

**¿Por qué β=1 y no 0 ó 4?** Es el equilibrio: β=0 reconstruye mejor (2.6 % px) pero no genera; β=4
genera un latente perfecto N(0,I) pero la reconstrucción se degrada (3.7 %); β=1 da buen latente
(std 1.17) con recon 3.2 % (E12/E15).

**¿El backprop del VAE es confiable si es a mano?** Sí: `tp5lib/vae_core.py` tiene un gradient check
numérico con error **5.0e-08** (`python3 tp5lib/vae_core.py`). No se modificó en todo el TP.

**¿Por qué emojis y no las letras?** El enunciado pide un dataset nuevo. Se usaron 5 siluetas de emojis
OpenMoji (canal alpha) con augmentaciones; el dataset es self-contained (assets commiteados).

## Conceptuales (las que más probablemente pregunten)

⚠️ **Si en 1a-4 el AE ya genera letras nuevas, ¿para qué necesitás el VAE?** Son dos cosas distintas. El
AE puede decodificar cualquier punto del latente, y como las 32 letras quedan dispersas y el decoder es
suave, interpolar ENTRE puntos conocidos da glifos razonables (la grilla y la interpolación a→o). Pero el
AE no te dice DÓNDE están los puntos válidos: su latente no es una distribución conocida, tiene huecos, y
un punto al azar puede caer en zona muerta → basura. El VAE organiza el latente como N(0,I), así que podés
samplear AL AZAR de una distribución conocida y caer siempre en zona válida. En una línea: **AE = navegar a
mano entre puntos conocidos; VAE = samplear al azar de un prior.** El VAE resuelve el "de dónde sampleo".

⚠️ **¿El 0 px no es overfitting? ¿No te preocupa la generalización?** Acá el overfitting es el OBJETIVO, no
un problema. La tarea es reconstruir esas 32 letras exactas — son el universo completo, no una muestra de
una población. No hay test set ni letras nuevas que generalizar; querés que la red memorice los 32 patrones
lo más fiel posible (por eso 1a entrena a fondo y sin regularización). Distinto del denoising y el VAE,
donde sí importa generalizar a ruido / muestras nuevas.

**¿Por qué 20 unidades en la capa oculta?** El barrido E3 mostró que `(20,)` ya llega a 0 px; menos no
alcanza y más no agrega. Es la capacidad mínima que mete las 32 letras en 2D con reconstrucción perfecta.

**¿El VAE genera emojis nuevos o copia los 5 base?** Nuevos: las muestras de N(0,I) son blends/variaciones
continuas — se ve en el atlas (`fig_e17`), con transiciones entre clases — no copias exactas de los 5 base.

## Generales

**¿Todo es numpy from scratch?** Sí. La red es la de TP3 (numpy); el VAE es numpy puro. PIL/scipy se usan
sólo para **preparar imágenes** (rasterizar/augmentar emojis), no para el modelo.

**¿Reproducible?** Semillas fijas en todo; `run_experiments.py` persiste CSV/modelos y `make_figures.py`
regenera las figuras sin reentrenar. Tests de la librería: 61 passed.
