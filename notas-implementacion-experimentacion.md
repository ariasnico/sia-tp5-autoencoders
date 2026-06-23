# Notas de implementación y experimentación

## BCE (Binary Cross-Entropy)

### Fórmula

```
BCE = -mean( y * log(p) + (1-y) * log(1-p) )
```

- `y` = píxel real (0 o 1)
- `p` = salida de la red (sigmoid, en (0,1))
- El `mean` corre sobre **todos los píxeles de todas las muestras** (array de forma `(32, 35)`)
- El clip `p = clip(p, 1e-12, 1-1e-12)` evita `log(0)`

### Implementación

`mlp/losses.py:19`:
```python
def bce(y_true, y_pred):
    p = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
```

### Cómo llega al CSV

`network.py` calcula el loss en cada paso del forward. Lo devuelve al callback definido en `ej1a_secuencial/common.py:133`. El callback se llama cada `LOG_EVERY` épocas y escribe una fila al CSV `expN_curves.csv`:

```
epoch, train_loss, px_max, px_mean, perfectas, leq1
```

---

## Error máximo en píxeles (px_max)

### Qué son los "outputs"

La red tiene capa de salida **sigmoid** → produce floats en (0, 1), uno por píxel. Estos floats son probabilidades de que el píxel sea 1. Para medir el error, se **binariza** con umbral 0.5.

### Flujo completo

```
X (32×35, binario)
  → forward pass
  → sigmoid output (32×35, floats en (0,1))
  → > 0.5
  → recon (32×35, binario {0,1})
  → recon != X
  → error por pixel (32×35, bool)
  → .sum(axis=1)
  → px por letra (array de 32 enteros)
  → .max()
  → px_max (un número: la peor letra del batch)
```

### Implementación

`tp5lib/autoencoder.py:58-65`:
```python
def reconstruct_binary(net, X):
    return (net.predict_proba(X) > 0.5).astype(float)

def px_err(net, X):
    return (reconstruct_binary(net, X) != X).sum(axis=1).astype(int)
```

En `common.py`, `px_err` se llama dentro del callback (cada `LOG_EVERY` épocas) y también al final del entrenamiento sobre los **pesos restaurados del mejor val_loss** (línea 138).

### Interpretación de las métricas del CSV

| Columna | Qué mide |
|---|---|
| `px_max` | Peor letra: máx píxeles incorrectos en el batch |
| `px_mean` | Promedio de píxeles incorrectos por letra |
| `perfectas` | Letras con 0 píxeles incorrectos |
| `leq1` | Letras con ≤ 1 píxel incorrecto |

`px_max == 0` → todas las 32 letras se reconstruyeron perfectamente bit a bit.

# Notas teoricas

## 1. El decoder es un "modelo generativo"

La clase distingue (slide 29) un modelo **discriminativo** de uno **generativo**: el generativo _"hipotetiza cómo se generan los datos en sí, capturando la distribución"_ — es **"la maquinola de generación de datos"**.

En el autoencoder, esa maquinola es **el decoder**: una vez entrenado, es una función `z → x̂` que toma un punto del espacio latente y produce una muestra. Generar = **alimentar el decoder con un `z` y ver qué sale**.

## 2. La idea de generación (slides 30–33)

El procedimiento que da la clase es literalmente:

1. Entrenás el autoencoder normalmente. Cada dato de entrada _"enciende"_ un punto en el espacio latente (`z₁, z₂`).
2. El espacio latente entrenado se vuelve una **estructura — el _data manifold_** — donde _"los datos viven y capturan los conceptos inherentes. Datos parecidos viven cerca; moverse a lo largo de una dirección cambia un concepto"_ (slide 33, el **concept vector**: el vector entre dos representaciones latentes).
3. Te movés dentro del espacio latente **especificando directamente los valores `zᵢ`** (no codificando un dato real, sino eligiendo coordenadas a mano).
4. Para cada tupla `(z₁, z₂)`, el **decoder produce una muestra nueva**.

Esto es **exactamente** lo que hacen tus figuras del ej1a:

- **Grilla de generación** (diapo 16): barrés una grilla de coordenadas `(z₁, z₂)` y decodificás cada una → `decode(ae, [z1, z2])`. Las celdas que caen cerca de una letra real dan esa letra; las del medio dan glifos nuevos.
- **Interpolación a→o** (diapo 17): tomás `z_a` y `z_o` (los puntos de esas dos letras) y caminás la recta entre ambos → los intermedios son letras inventadas. Eso es recorrer el _concept vector_.

## Grilla recorrida para la interpolación / generación de nuevas letras

**La grilla de generación (diapo 16)** es una grilla **uniforme** (`np.linspace`), de **14×14 = 196 puntos**, que cubre el **bounding box de las letras reales** en el latente, extendido un **15% para cada lado**:

```python
x0, x1, y0, y1 = Z[:,0].min(), Z[:,0].max(), Z[:,1].min(), Z[:,1].max()  # caja de las 32 letras
dx, dy = (x1-x0)*.15, (y1-y0)*.15                                        # margen 15%
gx = np.linspace(x0-dx, x1+dx, 14)   # 14 columnas, izquierda → derecha
gy = np.linspace(y1+dy, y0-dy, 14)   # 14 filas, arriba → abajo (orientación imagen)
# para cada (z1, z2) de la grilla: decode(ae, [z1, z2]) y se pega en el canvas
```

Es decir: **espaciado lineal y regular** sobre el rectángulo donde caen las letras. Cada celda se decodifica; las que caen cerca de una letra real reproducen esa letra, las del medio dan glifos nuevos.

**Importante: es lineal, NO gaussiana.** El espaciado gaussiano (`norm.ppf`, la inversa de la CDF normal) se usa recién en el **atlas del VAE** (`fig_e17_vae_atlas`), porque ahí el latente sí tiene densidad `N(0,I)` que respetar. El AE simple no tiene esa estructura, así que se barre la caja a secas.

**La interpolación (diapo 17)** es un caso particular de recorrido: en vez de una grilla 2D, se camina **una recta** entre dos puntos conocidos. Tomás `z_a` y `z_o` (los códigos latentes de 'a' y 'o') y evaluás `z(t) = (1-t)·z_a + t·z_o` para `t ∈ [0,1]` (9 pasos), decodificando cada uno. Eso es recorrer el **concept-vector** `a→o`: los pasos intermedios son letras inventadas que mezclan ambas. La figura `fig_concept_vector.png` muestra ese recorrido geométricamente sobre el mapa latente (la flecha + los 9 puntos de muestreo).

## 3. El problema del AE simple (slides 35–38) — y por qué existe el VAE

Acá está la parte fina de la teoría. El autoencoder generativo simple **no impone ninguna estructura sobre el espacio latente**:

> _"No hay garantía de que las representaciones intermedias produzcan cosas con sentido. Si uno se para en un punto intermedio no explorado durante el entrenamiento, **se pierde** — los datos quedan desasociados."_ (slide 36)

O sea: el decoder solo aprendió a mapear bien los **puntos donde cayeron las letras reales**. En los huecos del mapa no hay garantía de nada → puede salir basura. Por eso en tu grilla de generación las zonas alejadas de las letras saturan o dan ruido.

La solución (slide 37) es darle **estructura estadística** al latente: en vez de que cada entrada encienda un único punto, enciende una **campana gaussiana** (media = el punto, con varianza alrededor), y se exige que aún muestreando un punto _de alrededor_ la salida sea la misma entrada. Eso "rellena" el manifold y lo hace continuo/válido para muestrear → **eso es el VAE** (tu ej2).

Sí, exactamente acá. Leo la sección del truco de reparametrización y la capa estocástica para darte el detalle preciso:
## Dónde aparece la estocasticidad

En el AE simple, el encoder es determinístico: una entrada `x` produce **un** punto `z`. Siempre el mismo.

En el VAE, el encoder ya **no produce un punto**, produce los **parámetros de una distribución**: una media `μ(x)` y una varianza `Σ(x)` (ambas de dimensión del latente — si latente=2, el encoder escupe 4 números: `μ₁, μ₂, Σ₁, Σ₂`). Y el `z` se **muestrea** de esa gaussiana:

$$z \sim \mathcal{N}(\mu(x), \Sigma(x))$$

Ese **muestreo es el nodo estocástico**. La misma `x` ahora puede encender puntos distintos del latente (una "campana gaussiana" alrededor de `μ`). Es lo que la clase llama **Stochastic Feedforward Neural Network** (slides 43–48): por Cybenko, una red puede aproximar incluso una _función estocástica_ `f(x) = L(x) + ε(x)` — la red da `μ, Σ` y después se samplea.

## El problema que crea (y por qué el truco)

La estocasticidad rompe el backprop:

> **muestrear `z` es estocástico y NO diferenciable → no se puede retropropagar a través de un nodo aleatorio** (slide 80).

No podés calcular `∂z/∂μ` si `z` salió de un `random()`. El gradiente se corta ahí.

## El truco de reparametrización (slides 45, 83) — el corazón

La solución es **sacar la aleatoriedad afuera de la red**, a una variable externa `ε`:

$$z = \mu(x) + \epsilon \odot \sigma(x), \qquad \epsilon \sim \mathcal{N}(0, 1)$$

Ahora el azar vive en `ε` (que se samplea aparte, antes), y dado ese `ε`, el camino `μ, σ → z` es **determinístico y diferenciable**. El nodo aleatorio se movió "a un costado" del grafo, fuera del camino de gradientes:

- `∂z/∂μ = 1`
- `∂z/∂σ = ε`

La clase lo dice textual: _"las dos salidas del encoder (`μ, Σ`) actúan como entradas a la capa 'z', que se comporta como un **perceptrón lineal con función de activación identidad**"_. O sea: la capa latente del VAE es lineal, derivada 1, el `δ` pasa directo — igual que cualquier capa del TP3, salvo que tiene esta inyección de `ε`.

## La función de costo: dos términos

$$\min \mathcal{L} = \underbrace{|X - \hat{X}|}_{\text{reconstrucción}} + \underbrace{\tfrac{1}{2}\sum_k\big(\exp\Sigma + \mu^2 - 1 - \Sigma\big)}_{\text{KL regularizador}}$$

- **Reconstrucción**: que `x̂` se parezca a `x` (MSE o BCE), igual que el AE simple.
- **KL**: empuja cada gaussiana `N(μ(x), Σ(x))` hacia `N(0, I)`. **Este término es el que le da estructura al latente** — rellena los huecos, junta todo cerca del origen, y por eso cualquier punto que samplees decodifica a algo válido (la solución al problema de generación de la respuesta anterior).

Detalle de implementación (slide 77): la red predice **log-varianza** (`Σ = exp(...)`), lo que mantiene `σ > 0` y simplifica el KL eliminando el `log`. Es el `kl_loss = -0.5 * sum(1 + z_log_var - z_mean² - exp(z_log_var))` del notebook.

## Cómo se reparten los gradientes (slides 88–89)

Acá está lo elegante:

||se actualiza con…|
|---|---|
|**Decoder**|**solo** el gradiente de reconstrucción (idéntico a un MLP del TP3)|
|**Encoder**|**ambos**: reconstrucción (que llega vía el truco, `∂z/∂μ=1`, `∂z/∂σ=ε`) **+** KL|

Y un detalle fino: el **término KL no pasa por el decoder**. Depende solo de `μ` y `Σ` (variables internas del encoder), así que se deriva **analíticamente** y entra directo al encoder. En cada peso del encoder se **suman** las dos contribuciones.

**Respondiendo tu pregunta directa:** sí, el VAE convierte el MLP en una red estocástica al reemplazar el punto latente por _"muestrear de una gaussiana cuyos parámetros predice el encoder"_. Pero como eso no es entrenable, el truco de reparametrización (`z = μ + ε·σ`) devuelve la estocasticidad a un `ε` externo y deja la red **determinística y diferenciable de nuevo** — entrenás un MLP normal que, además del error de reconstrucción, paga un KL que ordena el latente. Eso es tu ej2.

## ¿KL en el AE simple? No

El autoencoder común tiene **un solo término**: el error de reconstrucción (`‖X − X̂‖` o BCE). No hay regularizador sobre el latente → el latente queda **sin estructura** (el problema de los slides 35–38). El KL aparece **recién en el VAE**, como segundo término, justamente para darle la estructura que al AE le falta.

El KL solo tiene sentido cuando hay una **distribución** que comparar contra el prior. En el AE simple el latente es un **punto** (no una distribución), así que no hay nada contra qué calcular un KL. Recién cuando el encoder produce `(μ, Σ)` aparece la gaussiana `q(z|x)` y, con ella, el término KL.

> Nota: el BCE de reconstrucción y el KL son **ambos** objetos de teoría de la información (entropía cruzada, divergencia KL — slides 50–51), pero cumplen roles distintos: BCE compara `x̂` vs `x` (reconstrucción); KL compara `q(z|x)` vs el prior (estructura del latente).

## Qué fuerza exactamente el KL

No fuerza "una gaussiana cualquiera": fuerza la **normal estándar `N(0, I)`** (media 0, varianza 1, sin correlación). Mirando la fórmula:

$$KL = \tfrac{1}{2}\sum_k\big(\exp\Sigma + \mu^2 - 1 - \Sigma\big)$$

- el término `μ²` empuja las medias **hacia 0** (todo se centra en el origen);
- el término `exp Σ − Σ` empuja la varianza **hacia 1** (ni colapsada a un punto, ni explotada).

Y lo que se vuelve gaussiano es el **conjunto agregado**, no cada código aislado: cada entrada produce su campanita `N(μ(x), Σ(x))`, el KL tira de cada una hacia `N(0,I)`, y la **nube de todos los códigos juntos** termina llenando una sola gaussiana estándar, continua y sin huecos.

## La tensión reconstrucción ↔ KL (β-VAE)

Si el KL ganara del todo, **todas** las entradas mapearían a `N(0,I)` exacto → todos los códigos encimados en el origen → el decoder no distingue nada → reconstrucción pésima (*posterior collapse*). Hay un forcejeo:

| término | qué quiere |
|---|---|
| **reconstrucción** | **separar** los códigos para distinguir cada entrada |
| **KL** | **juntar** todos los códigos en `N(0,I)` |

El equilibrio se controla con el **β** del β-VAE:

$$\mathcal{L} = \|X - \hat{X}\| + \beta \cdot KL$$

- `β` chico → gana reconstrucción, latente más desordenado (más parecido a un AE común);
- `β` grande → gana el KL, latente más gaussiano/ordenado pero reconstrucción más pobre.

Es lo que muestran las figuras del ej2 (`fig_e12_beta_sampling`, `fig_e15_recon_kl`, `fig_e16_ae_vs_vae`): al subir β el latente se ordena hacia la gaussiana pero la reconstrucción se degrada.

## La imagen de las gaussianas anidadas

- **Cada entrada** `x` → una **mini-gaussiana** `N(μ(x), Σ(x))` ubicada en el plano latente.
- El KL tira de todas hacia `N(0,I)` → el **agregado** forma **una gran gaussiana continua** centrada en el origen.
- Entradas parecidas → mini-gaussianas cercanas/superpuestas → forman **regiones** ("acá viven las tipo A, allá las tipo B").

Detalle clave: como el KL empuja la varianza **hacia 1** (no hacia 0), las campanitas **no son puntitos**, son borrones que **se solapan**. Ese solapamiento es el rasgo deseado: no hay huecos ni fronteras duras → el espacio es continuo → te movés de una región a otra pasando siempre por zonas con sentido.

## Cómo se genera (AE simple vs VAE)

En el **AE simple**, generar es **a ojo y arriesgado**: mirás el mapa, ves dónde cayeron las letras, elegís coordenadas a mano cerca o entre ellas. Si caés en un hueco → basura. No sabés "dónde es seguro pararse".

En el **VAE** ya **no hace falta mirar el mapa**, porque sabés que todos los datos viven dentro de `N(0,I)`:

1. **Sampleás un punto al azar del prior:** `z ~ N(0, I)` (dos números de una normal estándar, sin elegir nada).
2. **Pasás ese `z` por el decoder** → `x̂`.
3. Como el KL garantizó que la nube llena `N(0,I)` sin huecos, ese punto random **cae en zona poblada** → sale algo **válido**.

Esa es la magia: la estructura del KL convierte "samplear" en algo confiable. En el AE simple samplear al azar no servía (huecos); en el VAE, samplear de `N(0,I)` **es** el método de generación.

Las dos figuras del ej2 son las dos formas de samplear:

- **`fig_e18_sampleo_honesto`** — sampleo honesto: `z ~ N(0,I)` al azar, sin cherry-picking, y decodificás. Demuestra que aún a ciegas salen muestras válidas (lo que el AE simple **no** podía).
- **`fig_e17_vae_atlas`** — el atlas: recorrés una **grilla** del latente espaciada según la gaussiana (con `norm.ppf`, la inversa de la CDF normal, para que la grilla respete la densidad) y decodificás cada celda → el mapa completo de qué genera cada región.

---

# Cómo se genera el dataset de emojis (ej2)

Las "muestras" del ej2 son el **dataset de entrenamiento** del VAE (700 imágenes), y **no** se generan con la red: se construyen a partir de 5 emojis base con augmentaciones propias. Todo el pipeline está en `ej2_vae/dataset.py`.

## 1. De PNG de OpenMoji a silueta (5 bitmaps base)

Se parte de 5 emojis de **OpenMoji** (corazón, estrella, gota, luna, rayo), variante *color* a 72×72. Para cada uno (`silhouette()`):

- Se toma el **canal alpha** del PNG → da una **silueta rellena monocromática** (en vez del dibujo a color). Es lo clave: a baja resolución la silueta rellena es mucho más distinguible que los contornos line-art.
- Se **recorta al bounding box** (lo no transparente), se **centra** en un cuadrado y se **reescala a 20×20** (con 2px de margen).

Eso da **5 bitmaps "base"**, uno por clase (la fila de arriba del contact sheet).

**Decisión documentada:** la variante *black* de OpenMoji da contornos line-art que a 20×20 se confunden (1-NN acc **0.45**); el canal alpha de la variante *color* da siluetas rellenas (acc **0.95**). También se descartaron triángulo/rombo/círculo por ser formas convexas compactas confundibles entre sí.

## 2. Augmentación: de 5 bases a 700 muestras

5 imágenes no alcanzan para entrenar, así que se genera **variabilidad intra-clase** aplicando transformaciones aleatorias suaves a cada base, 140 veces (`augment()`):

```python
img = rotate(base, rng.uniform(-15, 15), ...)        # rotación ±15°
img = shift(img, rng.uniform(-2, 2, 2), ...)          # traslación ±2px en x e y
return np.clip(img + rng.normal(0, 0.03, ...), 0, 1)  # ruido gaussiano suave (σ=0.03)
```

**5 clases × 140 = 700 muestras** (`make_dataset()`).

## 3. Por qué augmentar así (y no más)

Las augmentaciones son **moderadas y seedeadas a propósito**, para que *"el VAE aprenda una variedad continua, no puntos fijos"*. Si solo le dieras las 5 bases, el latente memorizaría 5 puntos y no habría nada continuo entre ellos que generar. Al darle versiones ligeramente rotadas/desplazadas/ruidosas, cada clase ocupa una **pequeña región** del espacio de imágenes → el VAE aprende a interpolar dentro y entre clases (lo que después le permite generar muestras nuevas).

Dos detalles de robustez:

- **Seedeado** (`rng = np.random.default_rng(seed)`): el dataset es **reproducible** bit a bit, y los PNG quedan cacheados en `assets/` → corre offline sin descargar.
- Se mide **1-NN accuracy** (cada muestra augmentada vs los 5 bitmaps base, `nn_accuracy()`): da **0.95**, confirmando que las clases siguen distinguibles pese al ruido.
