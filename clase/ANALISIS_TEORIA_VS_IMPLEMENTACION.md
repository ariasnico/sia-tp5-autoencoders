# TP5 — Teoría de la cátedra vs. implementación

> Documento de trazabilidad: mapea **cada decisión de implementación y experimento** del repo
> contra la teoría y los *tips* de las clases 18.1 (matemática) y 18.2 (autoencoders), y marca
> explícitamente **dónde el proyecto se aparta de lo dado en clase** y **cuánto** (cosmético / extensión
> fundamentada / matiz conceptual). Fuentes: `clase/informe-clase18.1-matematica.md`,
> `clase/informe-clase18.2-autoencoders.md`, y el código del repo.
>
> **Veredicto en una línea:** la implementación es **fiel a la cátedra en todo lo conceptual y en
> todas las fórmulas del VAE**; lo que se aparta son elecciones de hiperparámetros menores y un puñado
> de **extensiones que van más allá del notebook** pero que están **fundamentadas en el propio marco
> teórico de la cátedra**. No hay ningún punto que *contradiga* lo dado en clase.

---

## 0. Marco general

La cátedra ancla todo el TP5 en tres ideas que el repo respeta al pie de la letra:

| Idea de la cátedra | Dónde aparece en clase | Cómo se respeta en el repo |
|---|---|---|
| Un AE **es una MLP** con cuello y `target = input` (no hace falta verlo como dos redes) | 18.2 §2–§3, slides 3–4 | `tp5lib/autoencoder.py::build_ae` arma una sola `MLP` simétrica reusando `mlp/` del TP3; `fit(X, X, ...)` |
| La **inteligencia surge de la restricción** (cuello angosto fuerza buena representación) | 18.2 §1, §3, slide 6 | Cuello 2D en 1a (objetivo del enunciado), cuello variable en 1b, `Z=2` en el VAE |
| Todo se reduce a **minimizar una función de costo** `J = L + λR` (Tikhonov) | 18.1 §3; 18.2 §6, slide 23 | BCE en 1a/1b; `J = recon + β·KL` en el VAE, con **β = el λ de Tikhonov** |

El enunciado prohíbe Keras/TF; la cátedra **enseña con un notebook en Keras** (18.2 §9) pero dice *"esto es
lo que ustedes van a tener que codear"* (slide 77). El repo cumple: **todo en numpy puro** sobre la
librería del TP3. Esa es la primera y mayor forma de "ir más allá del material de clase" — pero es
exactamente lo que el TP pide.

---

## 1. Ejercicio 1a — Autoencoder básico

### 1.1 Lo que la cátedra pide y el repo implementa

| Concepto de clase | Cita | Implementación | Estado |
|---|---|---|---|
| Encoder–decoder simétrico, salida = dim. de entrada | 18.2 §2 (slide 3) | `35-20-2-20-35` (`build_ae`) | ✅ fiel |
| `target = input` (función identidad por el cuello) | 18.2 §3 (slide 4) | `ae.fit(X, X, X, X, ...)` | ✅ fiel |
| **AE lineal ≡ PCA**; con no-linealidad → extensión no lineal de PCA | 18.2 §4 completo (demostración slides 7–16) | **E1** compara AE lineal vs PCA(2) por SVD vs AE no-lineal | ✅ fiel — *operacionaliza el teorema central de la clase* |
| La capa latente es un **perceptrón lineal con activación identidad** | 18.2 §8.7 (slide 83) | `act_latent="identity"` en el cuello | ✅ fiel |
| Reducción de dimensionalidad / compresión como uso original | 18.2 §5.1 (slide 18) | 35 px → 2 números → 35 px | ✅ fiel |
| AE **generativo**: descartar encoder, moverse en el latente, decodificar | 18.2 §7.2 (slides 31–34) | **E8c** grilla decodificada + **E8d** interpolación `a→o` (`decode`) | ✅ fiel (cumple enunciado 1a.4) |
| Métrica de reconstrucción + estudio de arquitecturas/optimización | enunciado 1a.2 | **E1–E7** barridos (latente, hidden, optim, lr, activación, loss) | ✅ fiel |

### 1.2 *Tips* de la cátedra que se siguieron

- **BCE para píxeles binarios.** La cátedra deriva BCE como costo (18.1 §6.5) y la usa en el notebook
  porque MNIST es [0,1]. El repo usa BCE y lo **prueba empíricamente** en E7 (BCE 0 px vs MSE 2 px).
- **Latente chico y experimental.** *"La dimensión latente debe ser chica… 100% experimental"* (18.2 §7.2).
  E2 barre {1,2,3,5,8} y encuentra el "codo" en 2 — exactamente el enfoque experimental sugerido.
- **El AE es difícil de entrenar.** La cátedra lo advierte (18.2 §7.2); E3/E4/E5 muestran justo eso
  (sin capa oculta = PCA; SGD se estanca; lr alto no aprende).

### 1.3 Dónde difiere / va más allá

| Punto | Qué hace el repo | Qué decía la cátedra | Magnitud |
|---|---|---|---|
| **Activación oculta `tanh`** | tanh en ocultas | El notebook (VAE) usa ReLU; la cápsula del reparam usa Mish | **Cosmético** — E6 muestra que tanh/relu/sigmoid llegan todas a 0 px |
| **Optimizador Adam** | Adam(0.01) | El notebook VAE usa RMSprop (default Keras); la cápsula del reparam **sí usa Adam** | **Cosmético** — consistente con la cápsula 18.1 |
| **0 píxeles de error** (32/32 perfectas) | Reconstrucción perfecta | La cátedra **advierte** que reconstrucción *idéntica* ajusta al ruido y que lo deseable es *"lo más parecido posible, no idéntico"* (18.2 §3) | **Matiz conceptual** — ver nota abajo |
| **PCA calculada por SVD a mano** | `np.linalg.svd` para la línea base | La cátedra prueba AE-lineal≡PCA pero no pide calcular PCA | **Extensión** — refuerza el teorema de clase con evidencia numérica |
| **Multi-semilla (n=3)** en E2/E3/E4/E5 | media ± desvío sobre 3 semillas | No discutido en clase | **Extensión metodológica** (rigor estadístico) |

> **Nota sobre el "0 px" vs la advertencia de overfitting.** La cátedra advierte contra la reconstrucción
> idéntica *cuando el objetivo es generalizar*. Acá el **enunciado pide explícitamente memorizar los 32
> patrones con ≤1 px de error** — no hay conjunto de test ni meta de generalización. Por eso 0 px **no es
> un defecto**: es el objetivo. El repo es consciente de esto (no hace split: "el objetivo es *memorizar*
> las 32"). La estructura del latente sí importa para la parte generativa, y ahí entra el VAE (§3).

---

## 2. Ejercicio 1b — Denoising Autoencoder

> La cátedra remarca que el DAE *"normalmente lo hacen mal y se equivocan"* (18.2 §5.3). El error típico:
> entrenar `(dato → dato)`. El repo lo hace **bien**.

### 2.1 Procedimiento correcto (18.2 §5.3, slide 22) vs implementación

| Paso de la cátedra | Implementación (`train_denoising`) | Estado |
|---|---|---|
| Modelar el ruido numéricamente | `bitflip(X, p, rng)` voltea cada bit con prob. `p` | ✅ |
| **Entrada = ruidosa, target = limpio** (NO dato→dato) | `forward(Xn)` con `Xn` ruidoso, `backward(Xn, X, ...)` con `X` limpio | ✅ — *justo el punto que "se hace mal"* |
| Reemplazar la muestra ruidosa por la salida del decoder | `reconstruct_binary` a umbral 0.5 | ✅ |
| Experimentar **cuánto ruido tolera** | **E10** test en {0, .05, .10, .15, .20, .30, .40}; campeón 81 % ≤1px @15 % | ✅ (cumple enunciado 1b.2) |
| "Se puede modelar el ruido como con Hopfield" | bit-flip = exactamente el ruido de Hopfield binario | ✅ |

### 2.2 Dónde difiere / va más allá

| Punto | Qué hace el repo | Qué decía la cátedra | Magnitud |
|---|---|---|---|
| **Tipo de ruido: bit-flip** | flip de bits | Nombra *salt-and-pepper*, *Gaussiano*, *Rayleigh* | **Cosmético** — bit-flip es el análogo binario de salt-and-pepper; la cátedra dice "o en general una PDF" y el repo lo justifica como "ruido natural para imágenes binarias" |
| **Ruido fresco cada época** | `bitflip` nuevo en cada época (no usa `fit`) | La cátedra describe el set ruidoso de forma estática | **Extensión** — augmentación on-the-fly; es buena práctica estándar, refuerza la robustez |
| **Evaluación sobre 30–50 realizaciones de ruido** | promedio para que el número no dependa del azar | No discutido | **Extensión metodológica** (rigor) |
| **Cuello como hiperparámetro (E9)** | barre {2,5,10,20}, gana 10 | La cátedra no fija el cuello del DAE | **Extensión** — fundamentada (trade-off compresión/denoising) |

**Conclusión 1b:** el núcleo (qué es entrada y qué es target) es **exactamente** el de la cátedra; las
diferencias son de *qué ruido* y *cuánta estadística*, ambas justificadas.

---

## 3. Ejercicio 2 — VAE

Es donde la cátedra pone toda la maquinaria matemática (18.2 §8, slides 54–89). El repo
(`tp5lib/vae_core.py`) la sigue **fórmula por fórmula**.

### 3.1 Correspondencia exacta de fórmulas

| Elemento | Fórmula de la cátedra | Código (`vae_core.py`) | Estado |
|---|---|---|---|
| Prior y posterior | `p(z)=N(0,I)`, `q(z|x)=N(μ,Σ)` diagonal (18.2 §8.4) | encoder → `(mu, lv)`, `lv` = log-varianza | ✅ |
| **Reparam trick** | `z = ε⊙exp(Σ/2)+μ`, `ε~N(0,1)` (18.2 §8.7, slide 45/83) | `std=exp(0.5*lv); z = mu + std*eps` | ✅ idéntico |
| Reconstrucción (binario → BCE) | `log p(x|z) ∝ (x−x̂)²`; en el notebook **BCE** porque es binario (18.2 §8.4, §9.7) | `recon = Σ BCE(logits, X)/B` (forma estable softplus) | ✅ |
| **KL con log-varianza** (ajuste numérico, slide 77) | `KL = −½Σ(1+logσ²−μ²−σ²)` | `kl = -0.5*Σ(1+lv-mu²-exp(lv))/B` | ✅ **idéntico a la slide 78 y al notebook** |
| Costo final | `min −L = recon + KL` (`J=L+λR`, slide 78) | `loss = recon + beta*kl` | ✅ (con β = λ) |
| Balance recon/KL | recon = **suma** sobre píxeles, KL = suma sobre latente, `K.mean` sobre batch (notebook §9.7) | `recon` y `kl` ambos suma-por-muestra `/B` | ✅ mismo balance (no promedia sobre píxeles) |

### 3.2 Backpropagation a través de la capa estocástica (18.2 §8.8, slides 86–89)

La cátedra da las reglas exactas; el repo las implementa **literalmente** y además **las verifica**:

| Regla de la cátedra | Código | Estado |
|---|---|---|
| **Decoder**: se actualiza **solo** con el gradiente de reconstrucción (= MLP normal) | `g["W_out"], g["W_dec"]` salen solo de `dlogits` | ✅ |
| **Encoder**: se actualiza con **ambos** (recon vía reparam + KL analítico), **sumados** | `dmu = dz + beta*mu/B`; `dlv = dz*eps*0.5*std + beta*0.5*(exp(lv)-1)/B` | ✅ |
| `∂z/∂μ = 1`, `∂z/∂σ = ε`; el KL **no pasa por el decoder** (es analítico) | `dz` es `∂L/∂z` (recon); los términos `beta*…` son `λ·∂R/∂μ`, `λ·∂R/∂σ` | ✅ — calca la Eq. de slide 86–87 |
| Capa latente lineal (identidad) → su δ pasa directo | el cuello no tiene activación no lineal | ✅ |

> **Extensión clave (más allá de la cátedra, pero en su espíritu):** el repo **deriva el backprop a mano**
> (la cátedra lo hace en Keras, que diferencia solo) y lo **valida con un gradient check numérico**
> (`gradient_check()`, error relativo `5.02e-08 << 1e-5`). La cátedra nunca pidió esto; es la prueba de
> que la reimplementación en numpy es correcta. **Es el aporte técnico más fuerte del TP.**

### 3.3 Dataset, sampleo y *tips* seguidos

| Concepto de clase | Implementación | Estado |
|---|---|---|
| Elegir/construir dataset nuevo (enunciado 2a; el notebook usa MNIST) | 5 emojis OpenMoji (silueta del canal alpha), 700 muestras 20×20 | ✅ fiel al enunciado |
| Augmentación para que el latente sea **continuo** ("campanas gaussianas" alrededor de cada punto, 18.2 §7.3) | rot ±15°, trasl ±2px, ruido σ=0.03 seedeados → variedad intra-clase | ✅ fiel a la intuición |
| Generar muestra nueva = samplear el latente y decodificar (enunciado 2c; 18.2 §7.2) | E12/E14/E18 samplean `z~N(0,I)` (`standard_normal`) y decodifican | ✅ fiel — *samplea del prior, lo más honesto* |
| Latente clusteriza por clase (manifold con sentido, 18.2 §8.9) | **E13/E16** scatter del latente; **E17** atlas decodificado | ✅ fiel |
| Reconstruir con la **media** (sin ruido) | `reconstruct` usa `mu`, no `z` | ✅ (igual que el notebook: grafica `z_mean`) |

### 3.4 Dónde difiere / va más allá

| Punto | Qué hace el repo | Qué decía la cátedra | Magnitud |
|---|---|---|---|
| **Barrido de β {0, 0.5, 1, 4}** (β-VAE) | estudia el peso del KL | El notebook fija **λ=1**; pero la cátedra **sí** plantea el marco `J=L+λR` con λ ajustable (Tikhonov, 18.1 §3) | **Extensión fundamentada** — es β-VAE (Higgins 2017), no nombrado en clase, pero es *exactamente* el λ que la cátedra enseñó a variar |
| **Atlas E17 con grilla lineal** `linspace(-2.5,2.5)` | recorre el latente uniforme | El notebook recorre con `norm.ppf` (Inverse Sampling Theorem, 18.1 §4.2) porque el prior es gaussiano | **Cosmético** — solo afecta la *visualización* del manifold; para *generar* (E12/14/18) el repo samplea bien del prior `N(0,I)` |
| **Activación oculta `tanh`** | tanh en encoder/decoder | El notebook usa ReLU | **Cosmético** |
| **Optimizador Adam(1e-3)** | Adam | El notebook usa RMSprop; la cápsula del reparam usa Adam | **Cosmético** — consistente con 18.1 §5.5 |
| **Dims 128** (vs 256), `D=400` (vs 784) | red más chica | Notebook: intermedia 256, MNIST 784 | **Cosmético** (dataset más chico) |
| **ε ~ N(0,1)** | normal estándar | La cátedra oral dudó "uniforme/normal"; el notebook usa `random_normal` | ✅ usa la versión correcta/estándar |
| **Numpy puro + gradient check** | reimplementa el VAE Keras a mano y lo valida | El notebook es Keras | **Extensión mayor** — lo pide el enunciado; el gradient check es extra |

---

## 4. Síntesis: ¿cuánto nos alejamos de la cátedra?

```
            CONTRADICE la cátedra        ▒ (nada)
            ────────────────────────────────────────────────
  FIEL  ███████████████████████████████████████░░░░░░░░░  EXTIENDE
        │                                        │
        └ todo lo conceptual y todas             └ β-VAE, gradient check,
          las fórmulas del VAE                     multi-seed, PCA-por-SVD,
                                                   ruido fresco/época, numpy puro
```

**Tres niveles de divergencia, ninguno conflictivo:**

1. **Cosmético (no cambia conclusiones):** tanh vs ReLU, Adam vs RMSprop, dims 128 vs 256, atlas con
   grilla lineal vs `norm.ppf`, bit-flip vs salt-and-pepper nominal. Son elecciones de implementación
   intercambiables; el propio repo muestra (E6) que la activación no cambia el resultado.

2. **Extensiones fundamentadas en la teoría de la cátedra:** el **barrido de β** *es* el `λ` de Tikhonov
   que enseñó la clase 18.1 §3; el **gradient check** valida el backprop de las slides 86–89; la
   **PCA-por-SVD** operacionaliza la demostración AE-lineal≡PCA de 18.2 §4; el **multi-semilla / muchas
   realizaciones de ruido** son rigor estadístico. Todo esto **va más allá del notebook**, pero **dentro
   del marco conceptual** que la cátedra dio.

3. **Matices conceptuales bien resueltos:** el "0 px" en 1a parece chocar con la advertencia de
   overfitting (18.2 §3), pero se reconcilia porque el enunciado pide *memorizar*, no generalizar; y la
   falta de estructura del latente del AE simple (18.2 §7.3) se aborda **exactamente como propone la
   cátedra**: pasando al VAE en el ej2.

**Lo que NO se hizo distinto de la cátedra (fidelidad total):** la arquitectura encoder-decoder, el
`target=input`, la equivalencia AE-lineal≡PCA, el procedimiento del DAE (entrada ruidosa → target limpio,
*el punto que "se hace mal"*), **todas** las fórmulas del VAE (reparam, KL con log-varianza, costo
`recon+β·KL`), y el reparto de gradientes encoder/decoder de las slides 86–89.

**Conclusión:** el proyecto no se "aleja" de la cátedra en lo esencial — la **implementa fielmente** y
luego la **extiende con experimentación y verificación** que el material de clase habilita pero no exige.
La distancia con el notebook de clase es la que el propio enunciado impone (numpy en vez de Keras, dataset
propio en vez de MNIST) más una capa de rigor experimental opcional.
