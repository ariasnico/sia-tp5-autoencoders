# Informe Clase 18.1 — Repaso de Matemática (preparatorio para Autoencoders)

> **Materia:** Sistemas de Inteligencia Artificial — Repaso
> **Centro de Inteligencia Computacional, 2025**
> **Fuentes integradas:** transcript oral completo (`clase18.1-matematica-transcript.txt`, 534 líneas) + slides PDF (`clase18.1-matematica.pdf`, 21 páginas).
> **Propósito:** Estas son "cápsulas" de matemática que el profesor da por separado para luego "independizarse" y poder dar de corrido la clase de **autoencoders** (en particular **autoencoders variacionales / VAE**). Todos los temas reaparecen en esa clase, así que conviene entenderlos a fondo.

---

## Tabla de Contenidos

1. [Descomposición Espectral (autovalores / autovectores) y PCA](#1-descomposición-espectral-autovalores--autovectores-y-pca)
2. [Descomposición en Valores Singulares (SVD) y pseudoinversas](#2-descomposición-en-valores-singulares-svd-y-pseudoinversas)
3. [Regularización (Tikhonov) y penalizaciones en la función de costo](#3-regularización-tikhonov-y-penalizaciones-en-la-función-de-costo)
4. [Aproximación de una PDF: Inverse Sampling Theorem](#4-aproximación-de-una-pdf-inverse-sampling-theorem)
   - 4.1 [Generación de números pseudoaleatorios](#41-generación-de-números-pseudoaleatorios)
   - 4.2 [El teorema del sampleo inverso](#42-el-teorema-del-sampleo-inverso)
   - 4.3 [Teorema Central del Límite para gaussianas](#43-teorema-central-del-límite-para-gaussianas)
5. [Redes Feedforward Estocásticas y el Reparametrization Trick](#5-redes-feedforward-estocásticas-y-el-reparametrization-trick)
   - 5.1 [Idea de red estocástica](#51-idea-de-red-estocástica)
   - 5.2 [Maneras de implementarla](#52-maneras-de-implementarla)
   - 5.3 [El truco de la reparametrización](#53-el-truco-de-la-reparametrización)
   - 5.4 [Funciones estocásticas](#54-funciones-estocásticas)
   - 5.5 [Ejemplo en Colab / PyTorch](#55-ejemplo-en-colab--pytorch)
6. [Repaso de Teoría de la Información](#6-repaso-de-teoría-de-la-información)
   - 6.1 [Información](#61-información)
   - 6.2 [Entropía](#62-entropía)
   - 6.3 [Información mutua](#63-información-mutua)
   - 6.4 [Entropía condicional y conjunta](#64-entropía-condicional-y-conjunta)
   - 6.5 [Entropía cruzada (Cross Entropy) y Binary Cross Entropy](#65-entropía-cruzada-cross-entropy-y-binary-cross-entropy)
   - 6.6 [Divergencia KL / Entropía relativa](#66-divergencia-kl--entropía-relativa)
7. [Referencias bibliográficas](#7-referencias-bibliográficas)
8. [Resumen de fórmulas clave](#8-resumen-de-fórmulas-clave)

---

## 1. Descomposición Espectral (autovalores / autovectores) y PCA

*(Slide 2 — "Autoencoder lineal"; transcript líneas 1–27)*

**Contexto:** Esto se vio en la parte de **aprendizaje no supervisado** y **PCA** (descomposición de valores/componentes principales). Son conceptos de álgebra general; en informática a veces se sacan de las primeras materias (en el ITBA se dejaron en MNA / Métodos Numéricos), pero hacen falta para esta materia.

### Autovalores y autovectores — intuición

Dada una matriz, los **autovectores** son las direcciones tales que, al multiplicar cualquier vector por esa matriz, se **mantiene la misma dirección** del vector original. Son "direcciones importantes" que la matriz, en algún punto, preserva. Cada autovector tiene asociado un **autovalor** (cuánto se escala en esa dirección).

### Descomposición espectral

Si una matriz $X$ es **cuadrada**, **invertible** (no singular) y sus **autovalores son reales**, entonces es **diagonalizable** y se puede **factorizar** como producto de tres matrices:

$$X = E L E^{T} \tag{1}$$

Donde:
- $E$ es la matriz de autovectores (en el slide: "matriz de **filas** formada por sus autovectores"; en el oral aclara que la convención columna/fila depende de cómo se tome).
- $L$ es la matriz **diagonal** formada por los **autovalores**.
- $E^{T}$ es la traspuesta de la matriz de autovectores.

Esto se llama **descomposición espectral** ("un nombre rimbombante"). La gracia es que permite **descomponer una matriz que es una mezcla de muchos números en tres cosas separadas**, donde la del medio ($L$) es **diagonal**.

### Por qué aplica a PCA

PCA produce **siempre una matriz cuadrada**, porque surge de tomar la matriz de datos $X$ (muestras × dimensiones) y multiplicarla por sí misma ($X^{T}X$ o $X X^{T}$). El resultado es cuadrado (de tamaño "cantidad de muestras" o "dimensiones", según cómo se tome), por lo tanto se le puede aplicar la descomposición espectral. Este teorema **solo vale para matrices cuadradas** (invertibles, no singulares). *"Esto después nos va a servir un montonazo."*

---

## 2. Descomposición en Valores Singulares (SVD) y pseudoinversas

*(Slide 3 — "Autoencoder lineal"; transcript líneas 28–48)*

**Motivación:** ¿Cómo generalizar la descomposición espectral a matrices de **cualquier forma** (no cuadradas)? Con la **descomposición en valores singulares (SVD)**. Es la **misma idea** que la espectral, pero **sin la restricción** de que la matriz sea cuadrada.

### Definición

Dada una matriz $X$ de $n \times d$ (no cuadrada):

$$\mathrm{SVD}(X) = \tilde{Z}\, \Sigma\, V^{T} \tag{2}$$

Donde:
- $\tilde{Z}$ son los **vectores columna singulares de izquierda**.
- $\Sigma$ es una matriz **diagonal positiva** con los **valores singulares**.
- $V$ son los **vectores singulares derechos** (aparecen traspuestos como $V^{T}$).

Es la **generalización de la descomposición espectral** para matrices de cualquier forma: dos matrices de "autovectores singulares" (una por filas, una por columnas) y una diagonal de valores singulares.

### Pseudoinversas

La SVD es muy importante para encontrar **pseudoinversas**: cuando una matriz **no tiene inversa** (o es difícil/costoso calcularla), conviene por **performance** calcular primero la SVD y a partir de ella la pseudoinversa. Intuición de pseudoinversa: una matriz que, al multiplicarla por la original, *"más o menos da la diagonal, más o menos"*.

### Cálculo de la SVD vía minimización

Los valores singulares se calculan **minimizando** la diferencia entre la matriz original y la reconstrucción:

$$\min \; \lVert X - \tilde{Z}\, \Sigma\, V^{T} \rVert \tag{3}$$

Es decir, se toma $X$ y se le resta el producto de las tres matrices, buscando que esa diferencia (norma) sea **lo más chica posible**. Es una **optimización matemática** estándar. *"Esta es una de las maneras en que se calcula la SVD."*

---

## 3. Regularización (Tikhonov) y penalizaciones en la función de costo

*(Slide 4 — "Regularización"; transcript líneas 56–104)*

**Contexto:** En todos estos métodos (Hopfield, Oja/Sanger, perceptrón multicapa con gradiente descendente, etc.) lo que se hace es **encontrar el mínimo de una función de costo** que depende de los **parámetros**. Los parámetros se modifican siguiendo un algoritmo de optimización (gradiente descendente). Gran parte de la IA consiste en encontrar **formas de esa función** (a veces llamada *funcional*, por análisis funcional / teoría de optimización), agregándole o quitándole términos.

### Idea de regularización

A cualquier función $L$ que uno quiera minimizar se le puede **agregar un término adicional** (o varios) que **penaliza** situaciones que no se desean. La función que finalmente se minimiza es $J$:

$$J = L + \lambda R = \min_{\phi} \frac{1}{N}\lVert Y - X\phi \rVert \; \underbrace{(+\,\lambda \lVert f(\phi)\rVert)}_{\text{Término Regularizador}} \tag{4}$$

Donde:
- $L$ es la función de costo básica (en el slide, el error de reconstrucción $\frac{1}{N}\lVert Y - X\phi\rVert$).
- $R = \lVert f(\phi)\rVert$ es el **término regularizador**: una condición extra sobre $\phi$ que **limita el espacio de búsqueda** de soluciones.
- $\lambda$ es el **parámetro de regularización**: ajusta **cuánta importancia** se le da al término regularizador.

**Ejemplo de intuición (bordes):** si quiero una solución que **no esté en los bordes**, agrego un término que mide la distancia a los bordes; si esa distancia es grande, da un valor grande y la minimización lo evita, concentrando los valores en el medio.

### Regularización de Tikhonov

El nombre que usa el profe viene de **teoría de optimización**: la **regularización de Tikhonov**. Consiste en agregar un término que mantiene el **módulo de los parámetros** chico y, sobre todo, que todos sean **medianamente iguales** entre sí.

**Intuición física:** se busca la solución más **viable / regular**, no las **extravagantes**. Una solución donde *"un solo peso vale un millón y todos los otros valen 0,2"* no es regular y es obvio que no va a ser buena (es una solución "muy oblicua"/extrema). El término regularizador penaliza a los pesos de módulo muy grande, empujándolos hacia valores chicos y parejos.

**Otros nombres:** aparece muchísimo en aprendizaje automático con distintos nombres, p. ej. *empirical risk minimization*. *"Si ven siempre regularización, tiene que ver con esto."*

---

## 4. Aproximación de una PDF: Inverse Sampling Theorem

*(Slides 5–9 — "Repaso/Aproximación de una pdf"; transcript líneas 109–212)*

> *"Les prometo que alguna vez en su vida de profesionales de ingeniería van a tener que usar esto. Es hiperútil."*

### 4.1 Generación de números pseudoaleatorios

**Problema base:** necesito generar números al azar en un programa. ¿Cómo?

- **Mala idea criptográfica:** usar el **PID** (ID de proceso) — es **muy predecible**, no es seguro.
- **Forma estándar (PRNG):** un **generador de números pseudoaleatorios**, frecuentemente provisto por el SO (`/dev/random` o similar). La idea: un **grupo cíclico** de números discretos **enorme** (como un anillo gigantesco que se maximiza). Uno va pidiendo "el siguiente, el siguiente…"; como el ciclo es enorme, la probabilidad de acertar el próximo es bajísima y el comportamiento es el de una **distribución uniforme**. Por eso se llaman **pseudo**aleatorios.
- **Semilla (seed):** establece **dónde empieza la rueda** (el punto inicial del ciclo). El método de generación es el mismo; la semilla solo fija el arranque. Conviene inicializarla capturando **entropía** real: la **hora**, el **movimiento del usuario**, etc.
- **Aleatorios "puros" (físicos):** se pueden obtener de fenómenos físicos, p. ej. **random.org**, o la **desintegración radiactiva** de un átomo (detectar el disparo de un núcleo según la vida media). **Anécdota Cloudflare:** usan una pared de **lámparas de lava** filmadas como fuente de aleatoriedad física. *Nota del profe:* estos números físicos pueden ser **malos para criptografía** porque no se sabe cómo se van a comportar (un aleatorio "verdadero" podría darte, p. ej., todos unos); con un pseudoaleatorio uno **sabe** cómo se comporta.

**Conclusión:** por software, lo que uno obtiene de fábrica es una **distribución uniforme**. La pregunta clave: **¿cómo genero números con OTRA distribución (no uniforme) a partir de uniformes?**

### 4.2 El teorema del sampleo inverso

*(Slides 5, 6, 8)*

**Idea / receta:**

1. Tengo la PDF que quiero (puede ser **cualquier curva rarísima**, o una **empírica** armada con un histograma).
2. Calculo su **CDF** (función de distribución acumulada) $F$ = la **integral** de la PDF; va acumulando hasta llegar a 1.
3. Calculo la **función inversa** $F^{-1}$.
4. Tomo un número $Y$ del "pozo" de **uniformes** $U[0,1]$.
5. Evalúo $X = F^{-1}(Y)$: el uniforme entra como eje $y$ (porque es la inversa) y la salida es el $x$ que quiero.
6. Repito muchas veces: la **distribución de los $X$ obtenidos coincide con la PDF original**.

**Enunciado formal (slides 5–6):**

$$Y \sim U[0,1], \quad F \text{ invertible}, \quad F = \mathrm{CDF}(\mathrm{pdf}()) \;\Rightarrow\; X = F^{-1}(Y) \text{ tiene } \mathrm{CDF}(X) = F$$

Y la versión general (sin requerir $F$ estrictamente invertible), usando el ínfimo:

$$Y \sim U[0,1], \quad F = \mathrm{CDF}(\mathrm{pdf}()) \;\Rightarrow\; X = \inf_{x}\{\, x : F(x) \ge Y \,\}$$

**Diagrama (slide 8):** a la izquierda la curva $x \in \mathrm{pdf}$ (curva irregular); en el centro la CDF en forma de S; abajo el eje $y \in U[0,1]$. Se entra por $y$, se sube a la curva CDF y se proyecta a $x$. A la derecha: $F = \mathrm{CDF}(\mathrm{pdf}) \Rightarrow F^{-1}$.

**Ilustración con Betas (slide 7, transcript 187–190):** se muestran dos PDF Beta con distintos parámetros; la curva de arriba es la **directa** (CDF, sube hacia 1) y la de abajo la **recíproca/PDF**. Curvas para $\lambda = 0.5$, $\lambda = 1$, $\lambda = 1.5$.

**Detalle clave (Q&A, transcript 196–212):** Pregunta de Rodri: *"si tenés muchos uniformes, al hacer la acumulada e invertir, ¿no quedan todas parecidas? ¿Dónde está la variabilidad?"* — Respuesta: **no importa cuántos números agarres**; cada punto del uniforme, pasado por $F^{-1}$, produce **una muestra** de la distribución deseada. La distribución construida **es exactamente la PDF** que pusiste. Con un solo valor ves una muestra; con muchos ves que el acumulado de los $X$ se aproxima a la PDF objetivo. *"Lo pasás por este pipeline, entró el uniforme y te sale una Beta/Gamma con el parámetro que quisiste, porque lo pasaste justo por la inversa de la acumulada de esa distribución."*

**Aplicación:** así funcionan los generadores de las **librerías** (Beta, Gamma, etc., e incluso el propio uniforme): toman el uniforme del SO y lo pasan por este proceso. Sirve también para quedarse solo con **un pedacito** de una distribución (p. ej. solo la parte central de una normal).

### 4.3 Teorema Central del Límite para gaussianas

*(Slide 9; transcript 191–194)*

Para generar **gaussianas** hay otra manera: el **Teorema Central del Límite (TCL)**. Si tomo un número **grande** de valores $y \in U[0,1]$ (uniformes) y los **sumo/promedio** muchas veces, la estructura resultante tiende a una **gaussiana**:

$$\text{TCL: muchos } y \in U[0,1] \;\Rightarrow\; \mathcal{N}\!\left(\frac{n}{2}, \frac{n}{12}\right)$$

(Los parámetros $\mu = n/2$ y $\sigma^2 = n/12$ se derivan de sumar $n$ uniformes, cada uno con media $1/2$ y varianza $1/12$.)

---

## 5. Redes Feedforward Estocásticas y el Reparametrization Trick

*(Slides 10–15; transcript 213–359)*

### Punto de partida — teorema de aproximación universal

*(Slide 10)* Una red neuronal con **una sola capa oculta** y **funciones de activación no lineales** es un **aproximador universal**: puede representar **cualquier función computable** (*Arbitrary Continuous Functions*) — este es el **teorema de Cybenko** (el profe lo nombra como "Stepsie/Cybenko"). Entre otras cosas, puede representar una **función estocástica** regida por una **PDF**.

### 5.1 Idea de red estocástica

*(Slides 11, 15)* ¿Cómo usar una red para que se comporte como una **función de probabilidad** (p. ej. para generar números aleatorios condicionados a una entrada)?

Las redes que todos conocen son **totalmente determinísticas**: misma entrada → misma salida (la semilla solo afecta el orden de patrones del SGD, etc.). Una **red feedforward estocástica** rompe eso: para una **misma entrada** $x$ puede dar **distintas salidas** $y$, pero esas salidas **siguen una distribución** (una PDF condicional $p(y/x)$).

**Diagrama (slide 11):** entradas $x$ → una neurona que produce los parámetros $\mu$ y $\Sigma$ → de ahí sale $y$ según $p(y/x)$.

Este concepto está **poco documentado** en los libros, pero es **fundamental para explicar autoencoders**. El paper de referencia ([1], ver §7) "se usa contrabando" pero rara vez se cita.

### 5.2 Maneras de implementarla

1. **Paramétrica (recomendada):** la red recibe $x$ y produce los **parámetros de una PDF** (p. ej. la **media** $\mu$ y la **covarianza** $\Sigma$ de una normal). Con esos parámetros se hace un **sampling** interno para obtener $y$. Así se está modelando $p(y \mid x)$ — la probabilidad de $y$ **condicionada** a $x$ —, que tendrá una **forma** dada por los parámetros y la PDF elegida (puede ser cualquier distribución paramétrica, no solo la normal).
   - **Diagrama (slide 15):** $x$ → bloque $p(\theta/x)$ que emite $\Sigma$ y $\mu$ → se inyecta $r^{*} \in U[0,1]$ → nodo $\mathcal{N}(\mu, \Sigma)$ → salidas $y$, todo bajo $p(y/x)$.
2. **Ruido en los pesos (alternativa, menos control):** una vez entrenada la red, en cada pasada se le **agrega ruido al azar a los pesos**, lo que produce salidas distintas. Funciona, pero con **menos control** que la versión paramétrica.

### 5.3 El truco de la reparametrización

*(Slide 12; transcript 270–299, 328–338)*

**Problema:** queremos hacer el sampling **con control** y que la red siga siendo **entrenable**. La clave de las redes neuronales es que **todas las capas sean derivables** para poder aplicar **gradiente descendente + backpropagation**. Un paso de "samplear" directamente **no es derivable**.

**Solución — reparametrization trick.** Se construye la **capa estocástica** haciendo que $z$ se calcule **en función de** $\bar{\mu}$ y $\bar{\Sigma}$ que produce la red, más una fuente de ruido externa:

$$z = h(\bar{X}) = \epsilon \odot \bar{\Sigma}(\bar{X}) + \bar{\mu}(\bar{X}) \tag{5}$$

Donde:
- $\epsilon$ es un valor sampleado de una distribución (un **uniforme**, según el oral; comúnmente una normal estándar) **sacado de otro lugar** (fuente de ruido externa).
- $\bar{\Sigma}(\bar{X})$ es la varianza/covarianza que produce la red (cuánto puede variar).
- $\bar{\mu}(\bar{X})$ es la media que produce la red.
- $\odot$ es el producto elemento a elemento.

**Intuición (paralelo con el sampleo inverso):** tomo un valor random $\epsilon$, lo **multiplico por la varianza** (tiene sentido: en una normal eso controla cuánto puede variar) y lo **desplazo sumándole la media**. El efecto es que los $z$ obtenidos **corresponden a una normal** centrada en la media y dispersos según la covarianza.

**Por qué es bueno:** **todo es derivable**. Puedo derivar respecto de los parámetros ($\mu$, $\Sigma$) que la red produce, porque el ruido $\epsilon$ queda **fuera** del camino de derivación (entra como una constante muestreada). Así backpropagation puede **entrenar** los pesos que generan $\mu$ y $\Sigma$.

**Resumen del profe (transcript 354–357):** *redes estocásticas feedforward donde hay una capa intermedia en la que se hace un sampling de una distribución uniforme, que se convierte —vía el truco de la reparametrización— en una distribución normal, y la ventaja es que esa capa queda derivable.*

### 5.4 Funciones estocásticas

*(Slide 14; transcript 301–322)*

Una **función estocástica** es una función donde la asignación de $x$ a $y$ es **estocástica**, regida por una **PDF condicional**:

$$f(x) = L(x) + \epsilon(x)$$
$$y = f(x) \text{ tal que, dado } x,\; y = f(x) \text{ según } p(y/x)$$

Es decir, para un **mismo valor** de entrada puede haber **muchos valores** de salida.

**Ejemplo del collage (slide 14, líneas 307–315):** la función "media" es lo que sale en promedio de la red neuronal — una curva común (p. ej. *"una especie de seno más algo más"*). Pero al elegir un valor concreto (en el ejemplo, $x = 3.5$) y pasarlo por la red, se obtiene una **nube de puntos rojos**: distintas salidas para esa misma entrada. El **promedio** de esa nube cae sobre la curva. Extendiendo a izquierda y derecha, los promedios reconstruyen la curva.

**Usos:** situaciones donde se quiere comportamiento **no totalmente determinístico** pero con cierta **previsibilidad** (que "en el medio dé más o menos lo mismo"): típicamente **videojuegos**, generación de contenido, sensación, etc.

### 5.5 Ejemplo en Colab / PyTorch

*(Slide 13; transcript 324–353)*

- **Colab:** `https://colab.research.google.com/drive/1N8HMJTNgCXOflDEwbx9Um6hD1D1vS1Ha?usp=sharing`
- El ejemplo construye, en **PyTorch**, una red multicapa que aprende a reproducir un **seno** de forma **estocástica**.
- La red modela una **normal** y expone un parámetro **`lambda`** que permite **insertar una capa con la función que uno quiera** — concretamente, la capa del **reparametrization trick**: genera un valor al azar, lo **multiplica por la varianza** y le **suma la media**. Eso está **entrenado en los pesos** (es derivable, ese es el truco).
- Optimizador: **Adam**.
- Función de activación: **Mish**, descrita como *"una ReLU continua"*, definida como $x$ por la tangente hiperbólica (forma usual: $\mathrm{Mish}(x) = x \cdot \tanh(\mathrm{softplus}(x))$; el profe la describe informalmente como "x por la tangente hiperbólica"). Sirve en algunos casos pero "a veces termina dando cualquier cosa".
- Resultado: la **curva de salida** de la red entrenada copia el seno. Lo que se grafica como curva son las **medias**; si se prueba con un único valor de $x$ repetidamente y se marcan las salidas, se obtiene la nube centrada en la media (igual que el ejemplo del slide 14).
- **Conexión con el TP3:** el TP3 (perceptrón multicapa) ya hacía esto — aprender la salida de una función (coseno × seno con algunos outliers). La diferencia aquí es el componente estocástico.

---

## 6. Repaso de Teoría de la Información

*(Slides 16–20; transcript 360–519)*

**Contexto:** ideas de **Shannon**; usa **probabilidad/estadística para cuantificar información**, medida en **bits**. Conecta física/termodinámica con informática. Se necesita en cripto y en esta materia (autoencoders). El profe la da como una "cápsula rápida de 20 minutos".

Sea $X$ una **variable aleatoria** que toma valores $x_i$.

### 6.1 Información

$$I = -\log p(x_i) \tag{Información}$$

**Intuición — telégrafo e indios (líneas 377–393, 524–528):** estás en 1800 en la pampa húmeda con un telégrafo. La posta puede avisar "vienen indios" o "no vienen". ¿Qué aporta **más información**? El evento de **menor probabilidad de ocurrencia** — la **sorpresa**. En un canal de dos estados (un **bit**, 0 o 1), el valor con menos probabilidad es el que más información lleva. Lo importante en el puesto es **saber que los indios vienen**, no que no vienen.

**Intuición — la llamada a las 5 AM (líneas 388–391):** si te llaman de madrugada te asustás, porque es raro (baja probabilidad) → alto contenido de información. Si fuera normal, no informaría nada.

- El **logaritmo es base 2**, por eso la información se mide en **bits**.
- $\log p$ con $p \to 0^{+}$ tiende a $-\infty$; con el signo menos, $I \to +\infty$ (sorpresa máxima). El límite está tomado para que sea positivo.
- $I = 0 \iff p(x_i) = 1$ → **certeza absoluta**, **nada** de información (si sé que siempre pasa, no me sorprende).
- $I > 0$ → hay algo de información.
- $I(x_k) > I(x_i) \Rightarrow P_k < P_i$: lo más raro (menos probable) aporta **más** información.

### 6.2 Entropía

La información es de **un evento** $x_i$; la **entropía** es sobre **toda la variable aleatoria** $X$. Es el **promedio ponderado** de la información de todos los eventos posibles, ponderado por su probabilidad:

$$H = \mathbb{E}_{p(x_i)}\, I(x_i) = -\sum p(x_i) \log p(x_i) \tag{Entropía}$$

**Propiedades / intuiciones:**
- Refleja cuánta información tienen **todos los estados posibles** de $X$; mide qué tan **difícil de predecir** es la variable.
- **Entropía mínima = 0** en situación de **total certeza** (teniendo en cuenta $\lim_{p \to 0^{+}} p \log p = 0$). Si todo es perfectamente predecible → entropía 0.
- **Entropía máxima** cuando todos los eventos son **equiprobables** (distribución **uniforme**): es lo más difícil de predecir. Ahí se está usando el canal **al máximo de su capacidad** (conexión con teoría de comunicaciones: dos eventos equiprobables → un bit usado al 100%).
- Cota: $0 \le H(X) \le \log(2k+1)$ (valor máximo cuando la distribución es uniforme; $k$ = nº de estados/parametrización del slide).

### 6.3 Información mutua

Cuánto sirve **conocer una variable para predecir la otra**:

$$I(x,y) = H(y) - H(y/x) = \sum_{x}\sum_{y} p(x,y) \log \frac{p(x,y)}{p(x)\,p(y)} \tag{Información mutua}$$

- Se define como la entropía de una variable **menos** la entropía **condicional**: cuánto aporta (o no) $X$ sobre $Y$.
- **Eventos independientes:** no se puede predecir uno del otro → la información mutua es **0**. Esto surge directo de la fórmula: si $p(x,y) = p(x)p(y)$, el log es $\log 1 = 0$.
- La forma con $\log\frac{p(x,y)}{p(x)p(y)}$ se obtiene **abriendo algebraicamente** la diferencia de entropías.
- **Auto-información:** la entropía de $X$ es la información conjunta de la variable consigo misma: $H(x) = I(x,x)$.

### 6.4 Entropía condicional y conjunta

**Entropía condicional** (relacionada con la información mutua):

$$H(x/y) = H(x,y) - H(y) \tag{Entropía condicional}$$

**Entropía conjunta** (de dos variables a la vez — caso multivariado; extensible a más variables):

$$H(x,y) = \sum_{i=1}^{m}\sum_{j=1}^{m} p(x_i, y_i) \log \frac{1}{p(x_i, y_i)} \tag{Entropía conjunta}$$

Describe **cómo interactúan** las dos variables simultáneamente (una PDF de dos variables).

### 6.5 Entropía cruzada (Cross Entropy) y Binary Cross Entropy

*(Slide 18)* Sirve como **función de costo** (reaparece en GANs y en autoencoders).

**Cross Entropy:**

$$H_p(q) = -\sum_{k=1} q(y_k) \log p(y_k) \tag{Cross Entropy}$$

**Idea (líneas 463–469):** es como calcular la entropía pero **ponderando con una función de probabilidad distinta** ($q$) de la que se usa para calcular la información ($p$) — **por eso es "cruzada"**. Dado un evento, se usan **dos** distribuciones $P$ y $Q$: una ($p$) para la información de la variable, la otra ($q$) para ponderarla. De ahí la notación $H_p(q)$ ("H chiquita de p, de q").

**Binary Cross Entropy** (cuando los $q$ valen 0 o 1 — el **label**):

$$H_p(q) = -\frac{1}{N}\sum_{i=1}^{N}\Big\{\, (y_i)\log p(y_i) + (1 - y_i)\log\big(1 - p(y_i)\big) \,\Big\} \tag{Binary Cross Entropy}$$

Donde:
- $q(y_k) = y_i \in \{0, 1\}$: $y$ representa el **label** (la salida que quiero).
- $p(y_i)$: la **probabilidad asignada** a ese label para la muestra $i$ (lo que predice el clasificador), $i = 1..N$.

**Interpretación (líneas 470–489):**
- El primer término ($y_i = 1$) aporta cuando el label real es 1; el segundo ($1 - y_i$) es el **complemento**, aporta cuando el label real es 0.
- BCE **mide la efectividad de un clasificador**: da un número que **tiende a 0** cuando el clasificador asigna **alta probabilidad a la clase correcta** (alta prob. de 1 cuando es 1, alta prob. de 0 cuando es 0).
- Un **clasificador perfecto** (estima $y_i$ con probabilidad 1) → **BCE = 0**.
- Por eso es útil como función a **minimizar**: tiene su **mínimo en 0**, y ese mínimo se alcanza justo cuando la clasificación es **perfecta**.

### 6.6 Divergencia KL / Entropía relativa

*(Slides 19–20)* **Kullback-Leibler Divergence** (o **entropía relativa**).

**Para qué sirve:** medir la **"distancia" / similaridad entre dos distribuciones de probabilidad** — si tienden a ser parecidas (misma forma). Es un problema difícil en general: con distribuciones **paramétricas** se comparan por parámetros, pero con **no paramétricas** es más complicado; KL es una manera de hacerlo.

**Definición:**

$$KL(q \| p) = -\sum_{x_i} q(x) \log \frac{q(x)}{p(x)} \tag{KL}$$

Donde $p(x)$ es la **referencia** contra la que se compara $q(x)$. (Equivalente: una suma ponderada del log de la relación entre las dos distribuciones, ponderada por la distribución $q$ que se compara contra $p$.)

**Es una "norma" (en rigor, no lo es):** se llama **divergencia** porque se comporta *parecido* a una norma (como la norma de un vector / espacio normado / métrica abstracta), pero **no cumple todas** las propiedades de una norma.

**Propiedades (slide 20):**
- **No conmutativa:** $KL(q \| p) \neq KL(p \| q)$. El **orden importa**; depende de cuál es la referencia.
- **Cero contra sí misma:** $KL(q \| q) = 0$ (como un vector de norma cero → las distribuciones son iguales).
- **No negativa:** $KL(q \| p) \ge 0$ (positiva, como una norma).
- **Relación con entropía cruzada:** $KL(q \| p) = H_p(q) - H(q) \ge 0$.
  - $H_p(q)$ es la **entropía cruzada** (similar al Binary Cross Entropy) entre las dos.
  - $H(q)$ es la **entropía natural** de $q$ (la distribución que se está comparando, **no** la referencia).

*"Esto lo vamos a usar ahora en Autoencoders Variacionales."*

---

## 7. Referencias bibliográficas

*(Slide 21 — "Referencias I")*

1. **Charlie Tang and Russ R. Salakhutdinov.** *Learning stochastic feedforward neural networks.* Advances in Neural Information Processing Systems, **26, 2013.** — (citado como **[1]**, base de las redes feedforward estocásticas).
2. **Eric W. Weisstein.** *Stochastic Function*, 2023 (accessed November 17, 2023). `https://mathworld.wolfram.com/StochasticFunction.html` — (citado como **[2]**, base de las funciones estocásticas).

> El profe menciona además que dejará un **"librito de repaso de matemáticas"** para quienes quieran profundizar (no nombrado explícitamente en las slides).

---

## 8. Resumen de fórmulas clave

| Tema | Fórmula |
|------|---------|
| Descomposición espectral | $X = E L E^{T}$ |
| SVD | $\mathrm{SVD}(X) = \tilde{Z}\,\Sigma\,V^{T}$ |
| SVD vía minimización | $\min \lVert X - \tilde{Z}\,\Sigma\,V^{T}\rVert$ |
| Regularización (Tikhonov) | $J = L + \lambda R = \min_{\phi}\frac{1}{N}\lVert Y - X\phi\rVert + \lambda\lVert f(\phi)\rVert$ |
| Inverse Sampling | $Y\sim U[0,1],\; X = F^{-1}(Y) \Rightarrow \mathrm{CDF}(X)=F$ |
| Inverse Sampling (general) | $X = \inf_x\{x : F(x) \ge Y\}$ |
| TCL para gaussianas | muchos $y\in U[0,1] \Rightarrow \mathcal{N}(n/2,\, n/12)$ |
| Reparametrization trick | $z = h(\bar X) = \epsilon \odot \bar\Sigma(\bar X) + \bar\mu(\bar X)$ |
| Función estocástica | $f(x) = L(x) + \epsilon(x),\; y = f(x)\sim p(y/x)$ |
| Información | $I = -\log p(x_i)$ |
| Entropía | $H = \mathbb{E}_{p(x_i)}I(x_i) = -\sum p(x_i)\log p(x_i)$ |
| Información mutua | $I(x,y) = H(y)-H(y/x) = \sum_x\sum_y p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$ |
| Entropía condicional | $H(x/y) = H(x,y)-H(y)$ |
| Entropía conjunta | $H(x,y) = \sum_{i}\sum_{j} p(x_i,y_i)\log\frac{1}{p(x_i,y_i)}$ |
| Cross Entropy | $H_p(q) = -\sum_k q(y_k)\log p(y_k)$ |
| Binary Cross Entropy | $H_p(q) = -\frac{1}{N}\sum_{i=1}^{N}\{y_i\log p(y_i) + (1-y_i)\log(1-p(y_i))\}$ |
| KL / entropía relativa | $KL(q\|p) = -\sum_{x_i} q(x)\log\frac{q(x)}{p(x)} = H_p(q)-H(q) \ge 0$ |
| Cotas de entropía | $0 \le H(x) \le \log(2k+1)$ |

---

*Fin del informe. Cubre la totalidad de las 21 páginas del PDF y las 534 líneas del transcript.*
