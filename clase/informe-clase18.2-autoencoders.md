# Informe Completo — Clase 18.2: Autoencoders

**Materia:** Sistemas de Inteligencia Artificial (SIA) — Centro de Inteligencia Computacional, 2026
**Fuentes integradas:** Transcript de la clase oral (960 líneas), slides PDF (95 páginas) y notebook de implementación (`clase18.2-autoencoder.ipynb`).
**Propósito:** Base de conocimiento exhaustiva para implementar el Trabajo Práctico (TP5: autoencoder, denoising autoencoder y autoencoder variacional, extensión del TP3).

---

## Tabla de Contenidos

1. [Motivación y definición](#1-motivación-y-definición)
2. [Arquitectura encoder-decoder](#2-arquitectura-encoder-decoder)
3. [Aprendizaje: la función identidad y su utilidad](#3-aprendizaje-la-función-identidad-y-su-utilidad)
4. [Autoencoder lineal y su conexión con PCA/SVD](#4-autoencoder-lineal-y-su-conexión-con-pcasvd)
5. [El autoencoder como herramienta](#5-el-autoencoder-como-herramienta)
   - 5.1 Compresión
   - 5.2 Detección de outliers / anomalías
   - 5.3 Denoising Autoencoder (DAE)
   - 5.4 Contractive Autoencoder (CAE)
   - 5.5 Sparse Autoencoder (SAE)
6. [Regularización (marco general)](#6-regularización-marco-general)
7. [Modelos generativos profundos](#7-modelos-generativos-profundos)
   - 7.1 Generativo vs. discriminativo
   - 7.2 Autoencoder generativo y concept vector
   - 7.3 El problema de la falta de estructura del espacio latente
8. [Autoencoder Variacional (VAE)](#8-autoencoder-variacional-vae)
   - 8.1 Inferencia variacional
   - 8.2 Repasos previos (PDF, info theory, red estocástica, reparam)
   - 8.3 Derivación del ELBO
   - 8.4 Término de reconstrucción
   - 8.5 Término KL (regularizador)
   - 8.6 Función de costo final
   - 8.7 El truco de la reparametrización
   - 8.8 Backpropagation a través de la capa estocástica
   - 8.9 Algoritmo completo y forward pass
9. [Implementación en el notebook (Keras/TensorFlow)](#9-implementación-en-el-notebook-kerastensorflow)
10. [Notas para el TP5](#10-notas-para-el-tp5)
11. [Referencias bibliográficas](#11-referencias-bibliográficas)

---

## 1. Motivación y definición

Un **Autoencoder** es una **arquitectura de redes neuronales no supervisada** (slide 2). Sus puntos clave introductorios:

- Es una idea **vieja** (de los años 80–90 y anteriores), pero potenciada por hardware moderno. El profesor enfatiza que es "el ABC para entender modelos generativos": si se entiende bien el autoencoder, "a todos los modelos generativos les cae la ficha de por qué están operando".
- Uso original: **reducción de la dimensionalidad** (era el objetivo cuando se propusieron).
- Son **la base de algunos modelos de redes neuronales generativas** y encapsulan muy bien la idea de las redes generativas.

**Pregunta germinal (slide 2):** *"¿Qué pasa si la salida final de una MLP es una entrada para otra MLP inversa?"*

La idea: tomar un perceptrón multicapa que reduce neuronas hacia el centro y luego "invertirlo" (otra red que las vuelve a expandir), conectando ambas con algo en el medio. De ahí surge la arquitectura.

**Intuición central del profesor:** la inteligencia aparece cuando hay **restricciones / condicionamientos / "apretes"**. Al forzar que el código interno tenga pocas dimensiones, se obliga a aprender una representación buena. "Las lecciones se aprenden a los golpes."

---

## 2. Arquitectura encoder-decoder

**Definición (slide 3):**
- Dos redes neuronales artificiales de perceptrones multicapa, donde la salida de la primera red se conecta con la entrada de la segunda.
- La segunda red tiene la **distribución invertida** de neuronas en las capas y como salida tiene la **misma dimensión** que la entrada de la primera red.

**Componentes (slide 5 — "Función identidad"):**
- **Encoder (codificador):** primera red. Genera el código interno.
- **Espacio latente / Código latente `Z`:** la capa central (cuello de botella). Se llama "latente" porque "late ahí adentro" y no se observa directamente.
- **Decoder (decodificador):** segunda red. Toma `Z` y regenera la entrada.

**Formulación general (slides 3–4):**

$$Z = f(X) = h(XW + b)$$
$$X' = g(Z) = h(ZV + b)$$

Esto debe satisfacerse minimizando la pérdida de reconstrucción:

$$L(X, X') = \|X - X'\|^2$$

donde `h(·)` es una función de activación (posiblemente no lineal), `W` los pesos del encoder, `V` los pesos del decoder, `b` los bias.

> **Nota de notación:** en el desarrollo lineal posterior el decoder se escribe con la matriz `V` (slide 9), mientras que en el transcript oral el profesor usa `B`. Son equivalentes: matriz de pesos del decoder.

El profesor también aclara que el autoencoder **no tiene que verse necesariamente como dos redes enfrentadas**: se puede ver como **una única MLP con una capa intermedia que tiene menos neuronas** (un "cuello").

---

## 3. Aprendizaje: la función identidad y su utilidad

**Aprendizaje (slide 4):**
- La red se entrena por cualquier método válido para una MLP, pero para cada patrón de entrada `X` se pone como **salida esperada el mismo patrón `X`**.
- Por lo tanto, la red aprende los pesos sinápticos que generan en la salida `X'` el mismo valor presentado a la entrada (aprende la "identidad", pero pasando por `Z`).

**Pregunta clave (slide 6):** *"¿Qué utilidad puede representar en definitiva una función identidad?"*

**Respuesta — Restricciones (slide 6):** la utilidad está en `Z`. Al agregar restricciones se fuerza al autoencoder a aprender algo útil de la distribución del dataset. La restricción natural por arquitectura: **al usar menos neuronas en la capa central, se fuerza una codificación más eficiente del conjunto de datos** (compresión).

**Sobre overfitting (transcript, diálogo con Rodri):** si la capa central siempre es más chica que la entrada, en general no hay problema de overfitting. PERO sí puede haberlo: si la reconstrucción es **perfecta/idéntica** se hace una reducción "sin pérdida" que ajusta incluso al ruido intrínseco de los datos. Lo deseable es que `X'` sea **lo más parecido posible, no idéntico**.

---

## 4. Autoencoder lineal y su conexión con PCA/SVD

Un autoencoder **lineal** = sin funciones de activación. Esta sección demuestra que **el autoencoder lineal es equivalente a PCA** (resultado que el profesor señala como pregunta típica en entrevistas de DeepMind, y que "no está bien escrito en los libros").

### 4.1 Descomposición espectral (repaso, slide 7)

Si una matriz `X` es invertible y sus autovalores son reales, puede diagonalizarse:

$$X = E L E^{T} \tag{1}$$

donde `L` es la matriz diagonal de autovalores y `E` la matriz de filas formada por sus autovectores.

### 4.2 Descomposición en valores singulares — SVD (slide 8)

Para matrices no cuadradas, generalización más general. Dada `X` de `n × d`:

$$SVD(X) = \tilde{Z}\,\Sigma\,V^{T} \tag{2}$$

donde `Z̃` son los vectores columna singulares de izquierda, `Σ` una matriz diagonal positiva, y `V` los vectores singulares derechos. El cálculo de SVD se realiza minimizando:

$$\|X - \tilde{Z}\Sigma V^{T}\| \tag{3}$$

### 4.3 Lo que minimiza el autoencoder lineal (slide 9)

Una vez entrenado, el autoencoder lineal minimiza:

$$J = \|X - ZV^{T}\| \tag{4}$$
$$X \approx ZV^{T} \tag{5}$$

Definiciones de dimensiones:
- `X`: matriz de datos de `n × d`. Hay `n` datos de dimensión `d`.
- `Z`: salida de la capa interna (el código), de `n × k`. `k` = dimensión reducida (neuronas del cuello).
- `V`: matriz de pesos sinápticos del decoder, de `k × d`.

### 4.4 Lema PCA (slides 10, 15)

> **Lema (PCA):** La salida del código interno `Z` del autoencoder lineal son las proyecciones de los datos en los componentes principales.

$$T_{PCA}(X) = X\,E = Z \tag{6}$$

### 4.5 Demostración (slides 11–14)

**Paso 1 — Conexión con SVD (slide 11):** partiendo de `J = \|X - ZV^{T}\|`, si `V^{T}` (pesos del decoder) es **ortonormal**, entonces `X ≈ ZV^{T}` puede verse como la descomposición en valores singulares:

$$X \approx SVD(X) = \tilde{Z}\,\Sigma\,V^{T}, \quad \text{con } \tilde{Z}\Sigma = Z$$

porque la descomposición SVD surge de resolver la misma minimización de la Ecuación (4). Es decir, en el autoencoder lo que la SVD llama `Z̃Σ` queda fusionado en una sola matriz `Z`.

**Paso 2 — Conexión con PCA (slide 12):** por la descomposición espectral, con los autovectores de la matriz de covarianza `XᵀX/(n−1)`:

$$\frac{X^{T}X}{n-1} = E\,L\,E^{T}$$

donde `E` es la matriz de transformación de PCA (autovectores).

**Paso 3 — Sustituir X por su factorización SVD (slide 13):** reemplazando `X ≈ Z̃ Σ Vᵀ`:

$$\frac{X^{T}X}{n-1} = (\tilde{Z}\Sigma V^{T})^{T}(\tilde{Z}\Sigma V^{T})\frac{1}{n-1}$$
$$= (V^{T})^{T}\Sigma^{T}\tilde{Z}^{T}\tilde{Z}\,\Sigma\,V^{T}\frac{1}{n-1}$$
$$= V\Sigma^{T}\Sigma V^{T}\frac{1}{n-1} = V\Sigma^{2}V^{T}\frac{1}{n-1}$$

Aquí `Z̃ᵀZ̃ = I` (ortonormalidad) y `Σ` es diagonal, así que `ΣᵀΣ = Σ²`.

**Paso 4 — Igualar con la descomposición espectral (slide 14):**

$$\frac{X^{T}X}{n-1} = V\Sigma^{2}V^{T}\frac{1}{n-1} = V\left(\frac{\Sigma^{2}}{n-1}\right)V^{T}$$

Por la descomposición espectral esto también es igual a `E\,L\,E^{T}`, por lo que (si `X` tiene media cero):

$$\lambda_i = \frac{S_i^2}{n-1}, \quad \text{con } S_i \text{ los valores singulares de } \Sigma$$
$$V = E$$

Es decir: **los autovalores de PCA son los cuadrados de los valores singulares sobre `n−1`, y los autovectores coinciden** con los vectores singulares.

**Conclusión (slide 14):** al transformar los datos con PCA:

$$T_{PCA}(X) = X\,E = (\tilde{Z}\Sigma\underbrace{V^{T})\,E}_{I} = \tilde{Z}\Sigma = Z$$

porque `VᵀE = I` (ya que `V = E`). Esto corresponde exactamente a la salida del espacio latente. **QED.**

### 4.6 Generalización no lineal (slide 16)

> **Reducción no lineal de la dimensionalidad:** No es una manera muy eficiente de obtener las proyecciones en componentes principales, pero permite pensar al autoencoder como una **extensión no lineal de PCA** cuando se usan funciones de activación no lineales. "Es una asumción fuerte."

**Intuición del profesor:** PCA es una operación lineal; al haber probado la equivalencia con una red neuronal lineal, agregar activaciones no lineales "extiende" PCA, capturando estructuras interesantes en espacios más complejos donde hay no linealidad. La calidad de la representación latente depende de la **correlación de los datos de entrada** (igual lógica que PCA).

---

## 5. El autoencoder como herramienta

(Slide 17: sección "Autoencoder como herramienta". El profesor lo describe como "una navajita suiza".)

### 5.1 Compresión (slide 18)

> **Mecanismo de compresión:** fue el **uso original** que se le dio a los autoencoders, conocidos entonces como **"autoasociadores"**, como mecanismos de compresión de información (hace ~30–35 años).

Ventaja sobre PCA: es una red neuronal, puede tener muchas capas y ser no lineal → más potente que PCA.

### 5.2 Detección de outliers / anomalías (slide 19)

**Procedimiento (slide 19):**
1. Se entrena el autoencoder con el conjunto de entrenamiento.
2. Luego se lo somete a nuevas muestras que no necesariamente están en el conjunto.
3. Se mide con alguna métrica la diferencia entre cada entrada y la salida obtenida: `\|X_i − X_i'\|`.

**Intuición (transcript):** para cada muestra `i` se calcula la distancia entre entrada y salida y se arma una curva (eje X = ID de cada muestra, eje Y = distancia de reconstrucción). Las muestras cercanas a cero son **"compliant"** con el dataset (más regularidad, fácil de reconstruir). Las que dan lejos de cero son **outliers / anomalías**: al autoencoder le cuesta reconstruirlas.

**Aplicación:** detector de anomalías (p. ej. en medicina, donde lo normal es ser sano). También se llaman **"unary classifiers"** (clasificadores unarios): se **umbraliza** la curva → si supera el umbral es anomalía. Es un ejemplo de hacer primero algo no supervisado y construir luego un modelo supervisado encima.

### 5.3 Denoising Autoencoder (DAE) — slides 20–22

> **Parte del TP5.** El profesor recalca que "este normalmente lo hacen mal y se equivocan".

**Concepto (slides 20–21):** "Denoising" = eliminación del ruido. Como el autoencoder genera una aproximación de `X` mediante `X' ≈ ZVᵀ`, puede usarse para eliminar ruido sobre la entrada `X` y recuperar el original. Ruido = perturbaciones sobre los datos que contaminan el contenido de información que se quiere recuperar; es inherente a cualquier proceso de obtención de información, "el mal de todos", imposible de eliminar por completo. La estructura interna encoder/decoder intenta **preservar el contenido de información más relevante**.

**Procedimiento correcto (slide 22 y transcript, recalcado):**
1. Una vez aprendido el conjunto de datos `X`, cuando aparecen nuevas muestras contaminadas con ruido, se reemplaza la muestra ruidosa por la muestra obtenida a la salida del decoder.
2. **NO** se entrena con (dato original → dato original). En cambio:
   - Se **modela el ruido** numéricamente: por ejemplo *salt-and-pepper*, o en general una función de distribución de probabilidad (**Gaussiano, Rayleigh**).
   - Se agrega ese ruido sintético a los datos `X` generando una instancia `X̃` (ruidosa).
   - **Se pone `X̃` (ruidoso) como ENTRADA y se demanda como SALIDA el dato `X` original sin ruido.**
3. Así el autoencoder **aprende a quitar el ruido** (se convierte en un "dispositivo que saca ruido").

**Para el TP5:** experimentar cuánto ruido "se banca" el autoencoder, qué nivel de ruido tolera y cuán bueno es eliminándolo. Se puede modelar el ruido como se hizo con Hopfield.

### 5.4 Contractive Autoencoder (CAE) — slides 23–25

Los CAE agregan un término regularizador a la función de costo que **resta sensibilidad a la entrada**, intentando aprender representaciones más simples y "crudas" de los datos. El término es el cuadrado de la **norma de Frobenius del Jacobiano** de la representación oculta respecto a la entrada:

$$\|J_h(X)\|_F^2 = \sum_{ij}\left(\frac{\partial h_j(X)}{\partial X_i}\right)^2$$

Es decir, penaliza la "variación de la variación" (cómo cambia la salida del encoder ante cambios en la entrada), forzando que sea lo más baja posible.

**Intuición (slide 25 + cálculo 1):** datos con mayor variabilidad ven reducida su variabilidad, de modo que el problema puede **reparametrizarse con menos variables**. Analogía: una curva en 2D que requiere `(x, y)` puede describirse como una **trayectoria** con un único parámetro `t` (reparametrización). El CAE acomoda los datos sobre la trayectoria natural dada por la curva de optimización y reduce un poco el ruido natural.

### 5.5 Sparse Autoencoder (SAE) — slide 26

Los SAE usan una arquitectura donde el espacio latente tiene **MÁS dimensiones que la entrada** (al revés del undercomplete). Para imponer restricciones útiles, agregan un término regularizador que **favorece la esparcidad**: que muchas neuronas del espacio latente estén en 0 (anuladas).

- Sirven para **Feature Learning** (aprendizaje de características): identificar caracterizaciones o transformaciones de los datos útiles para discriminarlos.
- La esparcidad se impone mediante una **restricción umbralizada**: de no superarse el umbral, desactiva todas las neuronas involucradas.
- Ejemplo de representación esparsa: **one-hot encoding** (todos ceros y un único uno).

**Ventajas (transcript):** representaciones con muchos ceros tienen ventajas numéricas (cálculos rápidos, pocos problemas numéricos). El profesor conecta esto con **embeddings**: "buenas maneras de aprender buenas representaciones internas". (En la época en que se planteó el autoencoder, no se usaba el término "embeddings".)

---

## 6. Regularización (marco general) — slide 23

Regularizar = agregar una condición extra `R` sobre la función objetivo `L`, que limita el espacio de búsqueda de las soluciones:

$$J = L + \lambda R = \min_{\phi}\frac{1}{N}\|Y - X\phi\| \underbrace{(+\lambda\|f(\phi)\|)}_{\text{Término Regularizador}} \tag{7}$$

Desalienta soluciones complejas o extremas. El parámetro `λ` ajusta la importancia del término regularizador (**Regularización de Tikhonov**). Este marco se reutiliza directamente en el VAE (`J = L + λR`).

---

## 7. Modelos generativos profundos

(Slide 27: "Deep Generative Model".)

### 7.1 Generativo vs. discriminativo (slides 28–30)

- **Modelo Discriminativo:** "crudo en los datos". Una ANN busca la hiper-sábana que separa los datos en clases **sin importar cómo se generan**. Origen: inferencia bayesiana, aprendizaje automático, estadística.
- **Modelo Generativo:** establece cierta **relación causal**; hipotetiza **cómo se generan los datos en sí**, capturando la **distribución** de los datos. Es como tener "la maquinola de generación de datos".

**Pregunta (slide 29):** *"¿Es posible, utilizando una red neuronal, implementar un mecanismo generativo causal que permita obtener muestras nuevas que compartan características representativas de un conjunto de datos?"*

**Idea (slide 30):** dado `X = (x_1, ..., x_n)` (features), se quiere generar `X̂ = (x̂_1, ..., x̂_n)` que sean aproximaciones/representantes de los datos.

### 7.2 Autoencoder generativo y Concept Vector (slides 31–34)

**Respuesta (sugerida por un alumno en el transcript):** generar datos nuevos **tomándolos del espacio latente** / de la distribución que vive ahí.

**Mecanismo (slides 31–32, ejemplo con caras de perros):**
- Se entrena el autoencoder normalmente. Cada foto de entrada "enciende" un punto en el espacio latente 2D (`z_1, z_2`).
- Dos fotos distintas → dos puntos distintos en `Z`.
- Uno puede **moverse entre esas representaciones** y construir **representaciones intermedias** que no corresponden a ningún dato original del dataset, con la esperanza de que sean una "mezcla" con sentido de los dos datos.

**Concept Vector (slide 33):** el vector entre dos representaciones latentes. Una vez entrenado, el espacio latente se transforma en una estructura — el famoso **data manifold** — donde los datos viven y capturan los conceptos inherentes. Datos parecidos viven cerca; moverse a lo largo de una dirección cambia un concepto (p. ej. "hacer una cara más vieja").

**Algoritmo del Autoencoder Generativo (slide 34):**
1. Utilizar un autoencoder para codificar en el espacio latente todos los patrones.
2. **Descartar el encoder** ("lo tiro a la basura").
3. Moverse dentro del espacio latente, especificando **directamente** los valores `z_i` (p. ej. `z_1 = ...`, `z_2 = ...`).
4. Para cada tupla de valores `z_i`, el **decoder** produce una nueva muestra generada.

**Dato poderoso (transcript):** si se toman los píxeles en un punto particular de las imágenes generadas y se compara su distribución con la del dataset original, **las distribuciones coinciden** → el autoencoder captura la distribución de los datos.

**Sobre la dimensión del espacio latente (transcript, pregunta de Lucila):** es la que uno quiera (2, 3, 4, 5...). Debe ser **relativamente chica** para forzar un aprendizaje que valga la pena. 100% experimental. Ejemplo: StyleGAN usa ~256 dimensiones para imágenes gigantescas. El autoencoder es **difícil de entrenar** (más que una MLP normal) justamente porque se fuerza una representación más chica.

### 7.3 El problema: falta de estructura del espacio latente (slides 35–38)

**Problema (slides 35–36):** el autoencoder generativo simple **no impone ninguna estructura sobre el espacio latente**. Por eso no hay garantía de que las representaciones intermedias produzcan cosas con sentido. Si uno se para en un punto intermedio no explorado durante el entrenamiento, "se pierde" — los datos quedan desasociados.

> **La conquista del espacio latente (slide 36):** "Necesitamos algo más que nos permita **capturar** lo que pasa en el espacio latente, para poder generar nuevas muestras **válidas** a medida que nos movemos."

**Solución (slide 37):** darle una **estructura estadística** al espacio latente. En vez de ver la salida para un único punto `z`, ver las salidas para un **conjunto de puntos** (campanas/"sábanas" gaussianas alrededor de cada punto, con el punto en la media).

**Intuición (transcript):** agregar la posibilidad de que al poner un dato de entrada no se encienda siempre el mismo valor latente, sino un **conjunto de valores con estructura probabilística** (exploración estocástica). Se ponen "campanas gaussianas" alrededor de cada punto (media), y se demanda que aún seleccionando un punto de alrededor, la salida sea la misma entrada. Esto da estructura al manifold.

**Modelo deseado (slide 38):** una función estocástica donde `x → p(z/x) → z → p(x/z) → x̂`.

---

## 8. Autoencoder Variacional (VAE)

(Slides 54–89. El concepto más complejo de la clase y central para el TP5.)

> **VAE (slide 55):** convergencia de dos ideas:
> - Una **estimación variacional** de un modelo generativo.
> - Una **solución basada en dos redes feedforward acopladas** (Encoder + Decoder) que conforman un Autoencoder.
>
> Propuesto en el famoso paper de **Kingma & Welling (2014)** ("Auto-Encoding Variational Bayes"). El profesor lo llama "Quimba/Kimba" en el audio.

### 8.1 Inferencia variacional (slides 56, 58)

> **Inferencia Variacional (slide 56):** los métodos variacionales son estrategias de la **física estadística** que se basan en establecer una función de costo que es mínima en la solución verdadera hipotética, pero que al plantear soluciones alternativas (paramétricas) permite minimizar la función y encontrar una solución cada vez mejor.
> Referencias: Peterson & Anderson 1987 (Mean Field Theory), Jordan 1999, Hinton & Van Camp 1993.

**Idea (slide 58, transcript):** estimar una **función de densidad de probabilidad (pdf) desconocida** aproximándola con funciones **conocidas** y optimizando sus parámetros mediante un problema de optimización. Análogo a la **estimación de parámetros por máxima verosimilitud** en estadística: encontrar el parámetro de una normal que mejor matchee los datos. Conecta con **Expectation-Maximization (EM)**, inferencia bayesiana y redes bayesianas (uno de sus creadores: Michael Jordan).

### 8.2 Repasos previos necesarios

**Aproximación de una pdf — Inverse Sampling Theorem (slides 39–40):** si se tiene un generador uniforme `Y ~ U[0,1]` y una pdf con `F = CDF(pdf())` invertible, entonces:

$$Y \sim U[0,1],\; F \text{ invertible},\; F = CDF(pdf()),\quad X = F^{-1}(Y)\text{ tiene }CDF(X) = F$$

Caso general (no invertible):
$$X = \inf_{x}\{x : F(x) \geq Y\}$$

(El notebook usa exactamente esto: `norm.ppf` = inversa de la CDF gaussiana, para muestrear la grilla latente.)

**TCL para Gaussianas (slide 42):** para generar `N(n/2, n/12)` se pueden sumar muchos `y ∈ U[0,1]`.

**Stochastic Feedforward Neural Networks (slides 43–44, 47–48):** una red de una capa oculta es un **aproximador universal** (Teorema de Cybenko) y puede representar cualquier función computable, incluyendo una **función estocástica** regida por una pdf. Una función estocástica:

$$f(x) = L(x) + \epsilon(x), \qquad y = f(x) \text{ tal que, dado } x,\; y = f(x) \text{ según } p(y/x)$$

La red estocástica produce parámetros `μ, Σ` de una pdf `p(y/x)` y luego muestrea con `r* ∈ U[0,1]` de `N(μ, Σ)` (slide 48).

**Repaso de Teoría de la Información (slides 49–51):**
- Información: `I = −log p(x_i)`.
- Entropía: `H = E_{p(x_i)} I(x_i) = −Σ p(x_i) log p(x_i)`. Mínima (0) en certeza total; máxima cuando todos los eventos son equiprobables. `0 ≤ H(x) ≤ log(2k+1)`.
- Información mutua: `I(x,y) = H(y) − H(y/x) = Σ_x Σ_y p(x,y) log[p(x,y)/(p(x)p(y))]`.
- Entropía condicional: `H(x/y) = H(x,y) − H(y)`. Entropía conjunta: `H(x,y) = Σ Σ p(x_i,y_j) log[1/p(x_i,y_j)]`.
- **Cross Entropy:** `H_p(q) = −Σ_k q(y_k) log p(y_k)`.
- **Binary Cross Entropy (slide 51):**
$$H_p(q) = -\frac{1}{N}\sum_{i=1}^{N}\{(y_i)\log p(y_i) + (1 - y_i)\log(1 - p(y_i))\}$$
donde `q(y_k) = y_k ∈ {0,1}` es el label y `p(y_i)` la probabilidad asignada. Un clasificador perfecto da BCE = 0. **(Esta es la pérdida de reconstrucción usada en el notebook.)**

**Divergencia Kullback-Leibler (slides 52–53):** una "norma" (no en rigor) que mide la distancia/similaridad entre distribuciones de probabilidad; si tiende a cero, son parecidas.

$$KL(q\|p) = -\sum_{x_i} q(x)\log\frac{q(x)}{p(x)}$$

donde `p(x)` es la referencia a la que se compara `q(x)`. **Propiedades:**
$$KL(q\|p) \neq KL(p\|q), \quad KL(q\|q) = 0, \quad KL(q\|p) \geq 0, \quad KL(q\|p) = H_p(q) - H(q) \geq 0$$

### 8.3 Derivación del ELBO (slides 59–73)

**Objetivo (slides 59–61):** aproximar `p(z/x)` con una `q(z/x)`, siendo `q(z)` una función que **controlamos** (p. ej. Gaussiana).
- `p(z/x)`: pdf de los datos, desconocida → la aproximamos con `q(z/x)`.
- `q(z/x)`: pdf que **planteamos a gusto** (p. ej. normal alrededor de cada `z`).
- `p(z)`: se asume que cada `x` mapea **multidimensionalmente normal** a `z` (restricción; podría ser otra).

**Minimización (slide 62):**
$$\min KL(q\|p) = -\sum q(x)\log\frac{p(x)}{q(x)}$$

**Desarrollo (slides 63–65):**
$$KL(q(z)\|p(z/x)) = -\sum q(z)\log\frac{p(z/x)}{q(z)}$$
$$= -\sum q(z)\log\frac{\frac{p(z,x)}{p(x)}}{q(z)} = -\sum q(z)\log\frac{p(z,x)}{q(z)}\frac{1}{p(x)}$$
$$= -\sum q(z)\left\{\log\frac{p(z,x)}{q(z)} - \log p(x)\right\}$$

Distribuyendo la suma (slide 64):
$$= -\sum_z q(z)\log\frac{p(x,z)}{q(z)} + \log p(x)\underbrace{\sum_z q(z)}_{1}$$

Como `Σ_z q(z) = 1` (es una pdf y `log p(x)` no depende de `z`, sale como constante):

$$KL(q(z)\|p(z/x)) = -\sum_z q(z)\log\frac{p(x,z)}{q(z)} + \log p(x) \tag{9}$$

Reordenando (slide 65) — **el truco de optimización clave**:

$$\log p(x) = KL(q(z)\|p(z/x)) + \sum_z q(z)\log\frac{p(x,z)}{q(z)} \tag{10}$$

**Identificación de términos (slide 66):**

$$\underbrace{\log p(x)}_{\text{fijo para } x} = \underbrace{KL(q(z)\|p(z/x))}_{\text{lo más pequeño posible}} + \underbrace{\sum_z q(z)\log\frac{p(x,z)}{q(z)}}_{\mathcal{L},\text{ lo más grande posible}} \tag{11}$$

**Razonamiento (transcript + slide 67):** `log p(x)` es **fijo / constante** (depende sólo de los datos, sin control). Por lo tanto, **minimizar el KL es equivalente a maximizar `L`**. `L` se llama **Variational Lower Bound (ELBO)**:

$$\mathcal{L} \leq \log p(x) \tag{12}$$

Cuando `KL → 0`, `L` es exactamente `log p(x)`. Maximizar `L` garantiza que la `q` planteada se parezca a la `p` de los datos.

**Desarrollo de L (slides 68–69):**
$$\mathcal{L} = \sum q(z)\log\frac{p(x,z)}{q(z)} = \sum q(z)\log\frac{p(x/z)p(z)}{q(z)}$$
$$= \sum q(z)\{\log p(x/z) + \log\frac{p(z)}{q(z)}\} = \sum q(z)\log p(x/z) + \sum q(z)\log\frac{p(z)}{q(z)}$$

Resultado final (slide 69) — **los dos términos del VAE**:

$$\boxed{\mathcal{L} = \mathbb{E}_{q(z)}\log p(x/z) - KL(q(z)\|p(z))}$$

- `E_{q(z)} log p(x/z)`: término de **reconstrucción**.
- `−KL(q(z)\|p(z))`: término **regularizador**.

**Ingredientes y objetivo de optimización (slides 70–71):**
$$\theta^*, \phi^* = \arg\max_{\theta,\phi}\mathcal{L}(\theta, \phi, x)$$
- `p_θ(z)`: distribución simple asumida (prior).
- `q_φ(z)`: distribución elegida que **determina la estructura del espacio latente**.
- Hay que optimizar **al mismo tiempo**: `p_θ(x/z)` para que dado un `z` muestreado genere un `x` válido, **y** `q_φ(z/x)` para que dado un `x` real genere parámetros para muestrear `z`. **Todo a la vez.**

**Esquema del VAE (slide 72):** `x → q_φ(z/x)` produce `(μ, Σ)` → `Sample z` → `p_θ(x/z) → x̂`. Los parámetros `θ*, φ*` son los **pesos de las dos redes**.

**Función de costo del VAE (slide 73):**

$$-\mathcal{L} = -\underbrace{\mathbb{E}_{q(z)}\log p(x/z)}_{\text{Error de reconstrucción}} + \underbrace{KL(q(z)\|p(z))}_{\text{Término regularizador}}$$

(maximizar `L` ≡ minimizar `−L`).

### 8.4 Término de reconstrucción (slides 74–75)

**Asunciones (slide 74):**
$$p_\theta(z) = \mathcal{N}(0, \mathcal{I}) \qquad q_\phi(z) = \mathcal{N}(\mu(x), \Sigma(x)), \text{ con } \Sigma(x) \text{ diagonal}$$

**Error de reconstrucción (slide 75):** asumiendo `p_θ(x/z)` normal multidimensional:

$$p_\theta(x/z) = \frac{1}{\sqrt{(2\pi)^k}\sqrt{\Sigma(z)}}\exp\{(x - \mu(z))^T\Sigma(z)^{-1}(x - \mu(z))\}$$

Tomando log, queda proporcional a (la media `μ(z)` es el `x̂` reconstruido):

$$\log p_\theta(x/z) \propto (x - \underbrace{\mu(z)}_{\hat{x}})^T\Sigma(z)^{-1}(x - \underbrace{\mu(z)}_{\hat{x}})$$

Esto es esencialmente `(x − x̂)²` → **equivale al error cuadrático medio (MSE)** entre entrada y reconstrucción. En el notebook se usa **binary cross-entropy** en su lugar (válido porque MNIST es binario/[0,1]); el profesor aclara que se puede usar MSE.

### 8.5 Término KL / regularizador (slides 76–77)

KL entre dos gaussianas, `q_φ(z) = N(μ(x), Σ(x))` vs `p_θ(z) = N(0, I)`:

$$KL = \frac{1}{2}\{trace(\Sigma(x)) + \mu^T\mu(x) - k - \log(|\Sigma(x)|)\}$$
$$= \frac{1}{2}\{\sum_k \Sigma(x) + \sum_k(\mu(x))^2 - \sum_k 1 - \log(\Pi_k\Sigma(x))\}$$
$$= \frac{1}{2}\{\sum_k \Sigma(x) + \sum_k(\mu(x))^2 - \sum_k 1 - \sum_k(\log\Sigma(x))\}$$

Forma compacta (slide 76):

$$KL = \frac{1}{2}\sum_k(\Sigma(x) + (\mu(x))^2 - 1 - \log\Sigma(x))$$

**Ajuste numérico (slide 77):** por estabilidad numérica se reemplaza `Σ(x) = exp(Σ(x))` (la red predice el **log-varianza**), lo que elimina el logaritmo:

$$KL = \frac{1}{2}\sum_k(\exp\Sigma(x) + (\mu(x))^2 - 1 - \Sigma(x))$$

> El profesor: "esto es más estable numéricamente, la mejor. Esto es lo que ustedes van a tener que codear."

### 8.6 Función de costo final (slide 78)

Marco de regularización `J = L + λR` (Eq. 13). Desarrollo completo:

$$\max \mathcal{L} = \mathbb{E}_{q(z)}\log p(x/z) - KL(q(z)\|p(z))$$
$$\min \mathcal{L} = -\mathbb{E}_{q(z)}\log p(x/z) + KL(q(z)\|p(z))$$
$$\min \mathcal{L} = \|\bar{X} - \bar{X}'\| + \frac{1}{2}\sum_k(\exp\Sigma(x) + (\mu(x))^2 - 1 - \Sigma(x))$$

**Forma final a codear:**

$$\boxed{\min \mathcal{L} = \|\bar{X} - \bar{X}'\| - \frac{1}{2}\sum_k(1 + \Sigma(x) - (\mu(x))^2 - \exp\Sigma(x))}$$

- Primer término: error de reconstrucción (MSE o cross-entropy).
- Segundo término: regularizador KL con el ajuste numérico (log-varianza).

> Nótese la coincidencia exacta con el código del notebook: `kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)`.

### 8.7 El truco de la reparametrización (slides 45, 80, 83)

**Problema:** muestrear `z` es estocástico y **no diferenciable** → no se puede retropropagar a través de un nodo aleatorio.

**Solución — Reparametrization Trick (slide 45, 83):** hacer que `z` se calcule en función de `μ̄` y `Σ̄`, sacando la aleatoriedad a una variable externa `ε`:

$$z = h(\bar{X}) = \epsilon \odot \bar{\Sigma}(\bar{X}) + \bar{\mu}(\bar{X}) \tag{8/14}$$

donde `ε ~ N(0,1)` (se muestrea de una normal estándar), `⊙` es producto elemento a elemento. Así el nodo estocástico se mueve "afuera" y el camino `μ, Σ → z` se vuelve **determinístico y diferenciable**.

**Forma original vs. reparametrizada (slide 83):** la forma original tiene un nodo estocástico `z ~ p_φ(z/x)` que bloquea el gradiente; la forma reparametrizada `z = g(φ, x, ε)` con `ε ~ N(0,1)` permite que el backprop fluya `∂f/∂z` y `∂f/∂φ`. Las dos salidas del encoder (`μ, Σ`) actúan como entradas a la capa "z", que se comporta como un **perceptrón lineal con función de activación identidad**.

### 8.8 Backpropagation a través de la capa estocástica (slides 82, 84–89)

**Dos gradientes a retropropagar (slides 88–89):**
1. **Gradiente de reconstrucción** (MSE/cross-entropy).
2. **Gradiente de regularización** (KL).

Reglas (slides 88–89):
- **Pesos del DECODER:** se actualizan **exclusivamente** con la contribución del gradiente de **reconstrucción**. Esto es igual a un perceptrón multicapa (idéntico al TP3).
- **Pesos del ENCODER:** se actualizan con la contribución de **ambos** gradientes:
  - Gradientes de reconstrucción del encoder = se obtienen multiplicando el gradiente que viene del decoder por la derivada de la función del truco de reparametrización respecto de media y varianza.
  - Gradientes del término de regularización del encoder = derivada de la divergencia KL respecto de media y varianza.
  - **Se suman** ambos valores en cada paso del encoder.

**Detalle clave (slides 82, 86–87):** el **término regularizador NO depende de la salida del decoder** (`X̄`). Depende de variables internas (`Σ` y `μ`) que llegan a la capa latente → **se calcula analíticamente** y va **directo al encoder** (no pasa por el decoder).

**Fórmula del gradiente (slides 86–87):** para actualizar un peso `ω_e` del encoder con `J = L + λR`:

$$\omega_e^{t+1} = \omega_e^{t} - \eta\frac{\partial J}{\partial \omega_e}$$

$$\frac{\partial J}{\partial \omega_e} = \left(\frac{\partial L}{\partial z}\frac{\partial z}{\partial \mu}\frac{\partial \mu}{\partial \omega_e} + \lambda\frac{\partial R}{\partial \mu}\frac{\partial \mu}{\partial \omega_e}\right) + \left(\frac{\partial L}{\partial z}\frac{\partial z}{\partial \sigma}\frac{\partial \sigma}{\partial \omega_e} + \lambda\frac{\partial R}{\partial \sigma}\frac{\partial \sigma}{\partial \omega_e}\right)$$

donde `∂L/∂z` es la retropropagación del error en el decoder, `∂z/∂μ = 1`, `∂z/∂σ = ε`, `∂R/∂μ` y `∂R/∂σ` son las derivadas de la Eq. 13 (KL), y `∂μ/∂ω_e`, `∂σ/∂ω_e` son la retropropagación hacia el encoder.

**Detalle del transcript sobre el paso por la capa latente:** al retropropagar el MSE por el decoder y llegar al espacio latente, esta capa es **lineal sin función de activación** → su derivada es **1** (`δ_i` pasa directo). Luego se llega al encoder. El término regularizador, en cambio, se deriva de forma directa y entra al encoder sin pasar por el decoder. Finalmente se suman todos los `δW` y se ajustan los pesos.

### 8.9 Algoritmo completo y forward pass (slides 79–81)

**Arquitectura (slide 79):** Encoder = **Recognition Model**; Decoder = **Generative Model**; en el medio el **Latent Space**.

**Recapitulación del algoritmo (slides 80–81):**
1. **(1/2)** Se toma un `x` y se pone a la entrada de la red. Con ese `x` el Encoder produce a su salida `Σ` y `μ`. Luego se muestrea un `z` de `p(z) = N(0, I)` (equivalente a obtener un `ε`, multiplicarlo por la varianza p.a.p. y sumarle la media): `z = h(x) = ε ⊙ Σ(x) + μ(x)`.
2. **(2/2)** Con ese `z` se pasa por el Decoder y se obtiene `x̂`. Se entrena el autoencoder para que minimice `L`, actualizando primero los pesos del Decoder, luego pasando por la función `h(x)` y de ahí retropropagando el error obtenido en `μ` y `Σ` hacia los pesos del Encoder.

**Forward pass detallado (transcript):** el encoder da `μ` y `Σ` (cada uno de dimensión del espacio latente — si latente=3, la salida son 6 valores: 3 de media + 3 de varianza diagonal). Se muestrea `ε`, se escala por `Σ` y se suma `μ` → se obtiene `z` (capa lineal sin activación). Ese `z` entra al decoder, que mapea una pdf y produce `x̂`, que se busca parecido a `x`.

**Resultado cualitativo (transcript):** al entrenar, en el espacio latente se generan **regiones con significado** asociado a los datos (p. ej. "gente vieja", "gente con barba", "hombres/mujeres"). Diferentes clases se **clusterizan** (como en la red de Hopfield), generando un manifold con sentido.

---

## 9. Implementación en el notebook (Keras/TensorFlow)

**Archivo:** `clase18.2-autoencoder.ipynb`. Implementa un **VAE** (no un autoencoder simple). Tomado del libro **"GANs in Action"** de Langr. Genera dígitos manuscritos interpolando en el espacio latente de MNIST.

### 9.1 Framework y setup

- **Framework:** **Keras** (con backend TensorFlow). Versiones: `numpy==1.19.5`, `tensorflow==2.2.0`.
- Se llama `disable_eager_execution()` (modo grafo de TF1, necesario para la `vae_loss` custom con tensores globales `z_mean`, `z_log_var`).

```python
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
```

### 9.2 Hiperparámetros

```python
batch_size = 100
original_dim = 28*28      # 784, dimensión de entrada (imágenes MNIST aplanadas)
latent_dim = 2            # espacio latente de dimensión 2 (784 -> 2)
intermediate_dim = 256    # capa oculta (mismo tamaño en encoder y decoder)
epochs = 50
epsilon_std = 1.0         # desvío estándar del ε muestreado
```

| Hiperparámetro | Valor |
|---|---|
| Dataset | MNIST (60.000 train, 10.000 test) |
| Dimensión de entrada | 784 (28×28) |
| Dimensión latente | 2 |
| Capa intermedia | 256 (ReLU) |
| Batch size | 100 |
| Épocas | 50 |
| `epsilon_std` | 1.0 |
| Optimizer | por defecto de Keras (RMSprop, ya que `vae.compile(loss=vae_loss)` no especifica) |
| Activación de salida | sigmoid |
| Pérdida reconstrucción | binary crossentropy × 784 |

### 9.3 Función de sampling (truco de reparametrización)

Toma la media y la log-varianza estimadas por el encoder y genera una muestra `z` (implementa `z = μ + exp(log_var/2)·ε`):

```python
def sampling(args: tuple):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon  # h(z)
```

> Nota: la red estima `z_log_var` (log-varianza), no la varianza directa — coincide con el ajuste numérico `Σ = exp(Σ)` de la slide 77. `exp(z_log_var/2)` = desvío estándar.

### 9.4 Encoder

```python
x = Input(shape=(original_dim,), name="input")            # 784
h = Dense(intermediate_dim, activation='relu', name="encoding")(x)  # 256, ReLU
z_mean = Dense(latent_dim, name="mean")(h)                # 2 (μ)
z_log_var = Dense(latent_dim, name="log-variance")(h)     # 2 (log σ²)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])  # 2 (muestreo)
encoder = Model(x, [z_mean, z_log_var, z], name="encoder")
```

El encoder devuelve **tres** salidas: `[z_mean, z_log_var, z]`. Total de parámetros: **201.988** (input→256: 200.960; 256→2 para media: 514; 256→2 para log-var: 514).

### 9.5 Decoder

```python
input_decoder = Input(shape=(latent_dim,), name="decoder_input")   # 2
decoder_h = Dense(intermediate_dim, activation='relu', name="decoder_h")(input_decoder)  # 256, ReLU
x_decoded = Dense(original_dim, activation='sigmoid', name="flat_decoded")(decoder_h)     # 784, sigmoid
decoder = Model(input_decoder, x_decoded, name="decoder")
```

Salida con **sigmoid** porque MNIST está normalizado a [0,1] (apropiado para binary cross-entropy). Total de parámetros: **202.256** (2→256: 768; 256→784: 201.488).

### 9.6 Modelo VAE completo (composición)

```python
output_combined = decoder(encoder(x)[2])  # se usa la 3ra salida del encoder: z (el muestreado)
vae = Model(x, output_combined)
```

Total de parámetros entrenables: **404.244**. El profesor lo describe como "matrioskas rusas" (modelos anidados).

### 9.7 Función de pérdida (exactamente la fórmula final del PDF)

```python
def vae_loss(x: tf.Tensor, x_decoded_mean: tf.Tensor):
    # cross-entropy entre los píxeles x (0/1) y la salida del Decoder
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)   # término reconstrucción
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  # término KL
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss

vae.compile(loss=vae_loss)
```

- **`xent_loss`**: reconstrucción, binary cross-entropy escalada por `original_dim` (784). El profesor aclara: se puede usar MSE (`x − x̂`) en lugar de cross-entropy.
- **`kl_loss`**: corresponde **exactamente** a `−½ Σ(1 + log σ² − μ² − σ²)`, idéntico a la fórmula final de la slide 78.
- Se promedia (`K.mean`) en lugar de sumar — "detalles numéricos que dan mejor".

### 9.8 Carga de datos y entrenamiento

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.   # normaliza a [0,1]
x_test  = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  # aplana a 784
x_test  = x_test.reshape((len(x_test),  np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train,        # entrada y target son ambos x_train (autoencoder)
        shuffle=True, epochs=epochs, batch_size=batch_size)
```

- Normalización a [0,1] dividiendo por 255.
- Aplanado de 28×28 a vector de 784.
- `fit(x_train, x_train, ...)`: entrada = target (no supervisado, reconstrucción).
- **Loss observada:** baja de **189.09** (epoch 1) a **149.40** (epoch 50). ~9 s/época en 60.000 muestras.

### 9.9 Visualizaciones / resultados

**(a) Espacio latente coloreado por clase:**
```python
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)[0]   # [0] = z_mean (la media)
plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=y_test, cmap='viridis')
```
Scatter de las medias `z_mean` de cada dígito de test, coloreado por etiqueta. **Resultado:** los dígitos se **clusterizan** por clase en el espacio latente 2D (ceros juntos, unos juntos, etc.) → el manifold adquiere sentido asociado al dataset. El profesor aclara que se grafica **la media** (primera salida del encoder).

**(b) Grilla generativa (manifold de dígitos):**
```python
n = 15                  # grilla 15x15 de dígitos
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# coordenadas espaciadas linealmente transformadas por la inversa de la CDF gaussiana (ppf),
# porque el prior del espacio latente es Gaussiano
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)        # genera SOLO con el decoder
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i*digit_size:(i+1)*digit_size, j*digit_size:(j+1)*digit_size] = digit

plt.imshow(figure, cmap='Greys_r')
```
Se barre la grilla del espacio latente usando `norm.ppf` (inversa de la CDF gaussiana — **Inverse Sampling Theorem**, slides 39–40) porque el prior es `N(0,I)`. Valores de `grid_x`: de −1.645 a +1.645. **Resultado:** una grilla de dígitos **generados** (que no existen en MNIST), donde los ceros se transforman en 6, los 6 en 1, los 1 en 9, etc., con transiciones suaves (números inclinados agrupados, etc.). Esta es "la clave de todos los modelos generativos". Se generan **solo con el decoder**, especificando directamente `z`.

**Enlaces Colab citados en las slides:**
- Reparametrization Trick: `https://colab.research.google.com/drive/1N8HMJTNgCXOflDEwbx9Um6hD1D1vS1Ha`
- Notebook VAE: `https://colab.research.google.com/drive/1Hobi6plfrrUQ9DCiFPZ6vaztOu5Q05ND`

---

## 10. Notas para el TP5

El TP5 pide implementar (extensión del TP3, una MLP con cuello):
1. **Autoencoder básico** (reconstrucción, función identidad con cuello).
2. **Denoising Autoencoder** (DAE): entrenar con entrada ruidosa → salida limpia. Modelar el ruido (salt-and-pepper, Gaussiano, Rayleigh). Experimentar niveles de ruido tolerados. *No confundir: la entrada lleva ruido, el target NO.*
3. **Autoencoder Variacional** (VAE): codear la función de costo `J = MSE(x, x̂) + KL`, con el KL en su forma numérica estable usando log-varianza:
   $$J = \|X - X'\| - \frac{1}{2}\sum_k(1 + \Sigma(x) - (\mu(x))^2 - \exp\Sigma(x))$$
   Implementar el **truco de reparametrización** `z = ε ⊙ exp(Σ/2) + μ` y el backprop manual a través de la capa estocástica (dos gradientes: reconstrucción al decoder+encoder, KL solo al encoder).

**Advertencias del profesor:** el autoencoder es **más difícil de entrenar** que una MLP; la dimensión latente debe ser **chica** (experimental); el DAE "lo hacen mal" frecuentemente (cuidado con qué es entrada y qué target).

---

## 11. Referencias bibliográficas

Citadas en las slides (pp. 91–95):

1. Charu C. Aggarwal. *Neural Networks and Deep Learning*. Springer, 2018.
2. James R. Anderson & Carsten Peterson. *A mean field theory learning algorithm for neural networks*. Complex Systems, 1987.
3. David M. Blei, Alp Kucukelbir, Jon D. McAuliffe. *Variational inference: A review for statisticians*. JASA, 2017.
4. Carl Doersch. *Tutorial on variational autoencoders*, 2021.
5. ErmonGroup. *Variational Autoencoder*, 2020 (`ermongroup.github.io/cs228-notes/extras/vae/`).
6. Charles W. Fox & Stephen J. Roberts. *A tutorial on variational bayesian inference*. AI Review, 2012.
7. Gregory Gundersen. *Reparametrization Trick*, 2021 (`gregorygundersen.com/blog/2018/04/29/reparameterization/`).
8. Geoffrey E. Hinton & Drew Van Camp. *Keeping the neural networks simple by minimizing the description length of the weights*. COLT, 1993.
9. jhu.edu. *Variational Tutorial*, 2011 (`cs.jhu.edu/~jason/tutorials/variational.html`).
10. Jeremy Jordan. *Autoencoders*, 2022 (`jeremyjordan.me/autoencoders/`).
11. Mohammad Emtiyaz Khan & Guillaume Bouchard. *Variational EM algorithms for correlated topic models*. UBC, 2009.
12. **Diederik P. Kingma & Max Welling. *Auto-Encoding Variational Bayes*, 2014.** (El paper fundacional del VAE — "Kimba/Quimba" en el audio.)
13. Diederik P. Kingma & Max Welling. *An introduction to variational autoencoders*. Foundations and Trends in ML, 2019.
14. **Kevin P. Murphy. *Machine learning: a probabilistic perspective*. MIT Press, 2012.** (Recomendado por el profesor: "libro madre de aprendizaje automático muy bayesiano", explica la parte física y la inferencia variacional.)
15. Stephen Odaibo. *Tutorial: Deriving the standard variational autoencoder (VAE) loss function*, 2019.
16. Baptise Rocca. *Stack Exchange, Machine Learning*, 2021 (backprop en VAE).
17. Charlie Tang & Russ R. Salakhutdinov. *Learning stochastic feedforward neural networks*. NeurIPS, 2013.
18. Martin J. Wainwright & Michael I. Jordan. *Graphical models, exponential families, and variational inference*. Now Publishers, 2008.
19. Eric W. Weisstein. *Stochastic Function*, 2023 (MathWorld).

**Libros adicionales mencionados oralmente:** "GANs in Action" (Langr — fuente del notebook); *Understanding Deep Learning* (recomendado por el profesor como buena explicación, "se los voy a subir").
