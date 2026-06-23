# Guion — TP5 Autoencoders

---

## Slide 1 — Portada

**🎤 Orador: Pina**

Bueno, empezamos con el TP5 de Autoencoders. El trabajo tiene tres partes: primero entrenamos un autoencoder básico para comprimir y reconstruir 32 letras ASCII, después entrenamos uno específicamente para limpiar ruido, y por último un VAE para generar imágenes nuevas de emojis.

---

## Slide 2 — Dataset: 32 caracteres ASCII

**🎤 Orador: Pina**

El dataset que vamos a usar es un conjunto de 32 letras ASCII. Son grillas de 7 por 5 píxeles que pueden ser blancos o negros, así que cada letra son 35 valores.

También medimos qué tan distintas son las letras entre sí, en cantidad de píxeles. En general se diferencian entre 10 y 20 píxeles. Hay algunos casos extremos: por ejemplo la K y la Q están bastante lejos, y en el otro extremo la L y el pipe son prácticamente iguales.

---

## Slide 3 — La configuración base

**🎤 Orador: Pina**

Para empezar, la configuración base del autoencoder aprovecha mucho de lo que ya teníamos del TP3: un perceptrón multicapa con 35 entradas y 35 salidas, una capa oculta de 20 neuronas, y el espacio latente en el medio con 2 neuronas, que es lo que pide la consigna.

Las activaciones son tangente hiperbólica para las capas ocultas, identidad para la capa latente, y sigmoide para la salida, para que cada píxel salga entre 0 y 1 y después lo umbralizamos en 0.5 para decidir si es blanco o negro.

Usamos la función de pérdida de entropía cruzada binaria, también heredada del TP3. Arrancamos con SGD, 6000 épocas, y full batch sobre las 32 letras.

---

## Slide 4 — Elección de optimizador (config)

**🎤 Orador: Narias**

El primer paso fue elegir el optimizador. Probamos tres: SGD con learning rate 0.5, Momentum con learning rate 0.1, y Adam con learning rate 0.01. El resto de la configuración se mantuvo fija.

---

## Slide 5 — Adam llega a 0 px (resultado)

**🎤 Orador: Narias**

Podemos ver algo muy bueno: Adam, de entrada, ya logra lo que pide la consigna, que es llegar a cero píxeles de error.

En el gráfico de barras vemos la cantidad de letras recuperadas perfectas y el error máximo que tuvo. SGD recuperó 10 de 32, con hasta 5 píxeles de error. Momentum recuperó 30 de 32, bastante bien, pero quedaron 2 letras con un pixel de error. Adam las recuperó todas perfectas.

Como Adam ya logró la perfección, nos quedamos con Adam.

---

## Slide 6 — Letras reconstruidas por optimizador

**🎤 Orador: Katia**

Visualmente, con SGD se ven letras bastante distintas a las originales. Momentum pifió en solo dos letras, de hecho la letra se lee bien, pero el pipe se convirtió en una L. Adam perfecta.

---

## Slide 7 — ¿Qué tan grande el paso? (config)

**🎤 Orador: Katia**

Ahora probamos distintos valores de learning rate: 0.0003, 0.01 y 0.3. El resto de la configuración se mantuvo fija con Adam.

---

## Slide 8 — 0.01 es el punto justo (resultado)

**🎤 Orador: Katia**

Podemos ver que 0.01 es el punto justo. El learning rate grande, 0.3, directamente no aprende: no converge y tiene un error máximo gigante, llegando a errar casi todos los píxeles. El learning rate chico, 0.0003, va en la dirección correcta pero no llega en 6000 épocas. Con más épocas probablemente llegaría al mismo resultado, pero no hace falta. 0.01 llega a cero píxeles de error, así que nos quedamos con ese.

---

## Slide 9 — Arquitectura: tamaño de la capa oculta (config)

**🎤 Orador: Mate**

Después investigamos la arquitectura. Probamos cambiar la capa oculta: sin capa oculta (lo que sería equivalente a una compresión lineal tipo PCA), una capa de 10, de 20, de 30, o dos capas apiladas de 20 de cada lado.

---

## Slide 10 — 20 neuronas es el mínimo justo (resultado)

**🎤 Orador: Mate**

Sin capa oculta solo recuperó perfecta una letra y tuvo hasta 14 píxeles de error. Con 10 neuronas reconstruye 18 letras pero igual quedan algunas con hasta 4 píxeles de error. A partir de 20 neuronas ya logra el resultado esperado: cero píxeles de error.

30 neuronas y la doble capa de 20 dieron el mismo resultado que 20. Como con 20 ya alcanza, nos quedamos con eso, no tiene sentido usar más.

---

## Slide 11 — Tamaño del espacio latente (config)

**🎤 Orador: Mate**

También probamos el tamaño del espacio latente, más allá de lo que pide la consigna. ¿Qué pasa si lo reducimos a una sola neurona? ¿Alcanza con eso para guardar 32 letras distintas?

---

## Slide 12 — Con 1 solo se aprende un subconjunto (resultado)

**🎤 Orador: Mate**

No. Con una sola neurona la red solo aprende un subconjunto de las letras: recupera 14 de 32 a la perfección, y las otras tienen hasta 9 píxeles de error. No alcanza con una sola dimensión para separar 32 letras diferentes.

Con 2 y 3 neuronas llega a cero error. La consigna pide 2, y confirmaos que alcanza, así que nos quedamos con eso.

---

## Slide 13 — Reconstrucciones (E8a)

**🎤 Orador: Pina**

Acá podemos ver visualmente que la red está reconstruyendo a la perfección las 32 letras: la fila de arriba son las originales y la de abajo las reconstruidas por el autoencoder.

---

## Slide 14 — Mapa del espacio latente

**🎤 Orador: Katia**

El espacio latente de 2 dimensiones se puede graficar. Cada letra queda ubicada en un punto del plano. Podemos ver que letras que visualmente se parecen tienden a quedar cerca: la O, la E y la B quedan juntas porque todas tienen una forma circular. La H, la N y la M también quedan relativamente cerca, aunque no tanto.

---

## Slide 15 — Inventar letras nuevas

**🎤 Orador: Katia**

Si tomamos una grilla de valores en el espacio latente y los pasamos por el decoder, podemos ver qué "inventa" la red. Aparecen las letras reales en sus posiciones, y entre ellas aparecen símbolos inventados, algunos que mezclan rasgos de las letras cercanas.

---

## Slide 16 — Interpolación A → O

**🎤 Orador: Mate**

Si en el espacio latente trazamos el vector que va desde la posición de la A hasta la posición de la O, y recorremos ese camino paso a paso, vemos cómo la letra va cambiando gradualmente. La A se deforma, pasa por una especie de U extraña, y termina siendo una O.

---

## Slide 17 — Denoising: Limpiar ruido

**🎤 Orador: Narias**

Ahora pasamos al denoising: entrenar una red específicamente para limpiar letras que tienen ruido.

---

## Slide 18 — ¿Cómo ensuciamos las letras?

**🎤 Orador: Narias**

El ruido que usamos es bit-flip: cada píxel se invierte con probabilidad P. Es el mismo mecanismo que en el TP4. Importante: en cada época de entrenamiento se generan inputs nuevos con ruido, la red nunca ve dos veces la misma mancha.

Visualmente, con 10% de ruido las letras todavía se parecen bastante a la original. Con 15% ya se empieza a perder la forma. Con 30% las letras ya no se parecen en nada a la original.

---

## Slide 19 — ¿Y si le damos ruido al AE de 1a?

**🎤 Orador: Narias**

Primero probamos qué pasa si le damos entradas con ruido al autoencoder que ya teníamos, sin reentrenarlo. La respuesta es que no funciona: con 10% de ruido ya empieza a errar en varias letras, y a partir de 15% no logra reconstruir ninguna correctamente. No fue entrenado para limpiar, así que la salida se deforma.

---

## Slide 20 — El AE de 1a no limpia

**🎤 Orador: Mate**

El error crece a medida que sube el ruido de entrada. Hay que entrenar la red específicamente para limpiar.

---

## Slide 21 — Entrenemos un denoiser

**🎤 Orador: Mate**

La idea es simple: generamos entradas ensuciando cada letra, y le exigimos a la red que devuelva la letra original limpia. En vez de pedirle que reconstruya la entrada (limpio → limpio), le pedimos que limpie (sucio → limpio).

La arquitectura y la función de pérdida son las mismas que antes. Solo cambia qué le pedimos como salida.

Para las pruebas que vienen, evaluamos 30 realizaciones con ruido por nivel × 32 letras, lo que da 960 entradas por cada nivel de ruido.

---

## Slide 22 — Nueva arquitectura: espacio latente

**🎤 Orador: Pina**

Para el denoiser probamos agrandar el espacio latente: 2, 5, 10 y 20 neuronas. Entrenamos cada modelo con 15% de ruido de entrada y después los evaluamos con entradas al 10%, 20% y 30% de ruido.

El gráfico de la izquierda muestra el promedio de píxeles incorrectos. El de la derecha muestra cuántas letras se recuperan casi perfectas (0 o 1 píxel de error) cuando la entrada tiene 20% de ruido.

Vemos un salto de calidad entre 5 y 10 neuronas similar al salto entre 2 y 5. Nos quedamos con 10 neuronas en el espacio latente.

---

## Slide 23 — Nueva arquitectura: capas ocultas (primera vista)

**🎤 Orador: Pina**

Con el espacio latente fijo en 10 neuronas, probamos distintos tamaños de capa oculta: 20, 30, 35 y 40 neuronas. Todos entrenados con 15% de ruido.

> [!warning] ACÁ LEER LA DIAPO TAMBIEN

---

## Slide 24 — Nueva arquitectura: capas ocultas (conclusión)

**🎤 Orador: Katia**

Hay un salto de calidad claro al pasar de 20 a 30 neuronas. Después de 30, aumentar la capacidad no mejora prácticamente nada.

Nos quedamos con 30 neuronas en la capa oculta.

---

## Slide 25 — ¿Cuánto ruido al entrenar? (descripción)

**🎤 Orador: Katia**

Con la arquitectura final 35-30-10-30-35, es preguntamos cómo se comporta al entrenar con distintos niveles de ruido de entrada.. Después a cada uno lo evaluamos pasándole entradas con niveles de ruido que van de 0% (limpio) hasta 40%, en saltos de 5%.

---

## Slide 26 — ¿Cuánto ruido al entrenar? (lectura)

**🎤 Orador: Katia**

Lo que vemos es que cada modelo se desenvuelve mejor en el rango de ruido con el que fue entrenado. La red entrenada al 5% funciona un poquito mejor con entrada completamente limpia, pero desde 10% de ruido en adelante, la supera la red entrenada al 15%. Y desde 15% para arriba, la entrenada al 5% es la que peor resultado da.

La red entrenada al 30% devuelve letras más ruidosas para entradas con poco ruido. Recién supera a las demás cuando el ruido de entrada llega a 30% o más.

En resumen, la red entrenada al 15% parece la mejor opción para un uso general: se porta bien desde entradas limpias hasta 25% de ruido.

---

## Slide 27 — Cualitativamente: efecto del 30% de ruido

**🎤 Orador: Mate**

Cuantitativamente la red entrenada al 30% parece comportarse mejor para entradas con mucho ruido, con 6 píxeles de error en promedio. Pero si miramos visualmente lo que devuelve, en la mayoría de los casos no logra reconstruir algo que se parezca a la letra original. Hay algunos casos donde sí: la P, la H, la N, la X y la Z las recupera bastante bien, probablemente porque son letras bastante distintas a las demás. Pero en general no es funcional.
Con nuestra arquitectura propuesta, no logramos un modelo que funcione para niveles altos de ruido. La resolución de los datos originales (7x5, 35 pixeles) implica que un 30% de ruido ya es bastante destructivo, aprox. 10 pixeles en promedio se modifican.

---

## Slide 28 — Resultado del denoiser (ganador)

**🎤 Orador: Mate**

El denoiser ganador tiene capa oculta de 30 neuronas, espacio latente de 10 y fue entrenado con 15% de ruido.

Lo probamos con 50 generaciones ruidosas × 32 letras por nivel, o sea 1600 entradas por nivel. Con 10% de ruido en la entrada, el 92% de las salidas tiene 0 o 1 píxel de error. Con 15% de ruido, sigue portándose bien: 80% de las letras quedan con 0 o 1 píxel de error.

---

## Slide 29 — Limpio / sucio / recuperado

**🎤 Orador: Narias**

Acá podemos ver tríos de imágenes: la letra original limpia, la entrada con ruido que le dimos a la red, y lo que la red devolvió.

Con 10% de ruido reconstruye 4 de 5 letras perfectas. Con 20% todavía se porta bien: la A y la R las recupera, la S perfecta. Con 30% de ruido, que visualmente las letras ya no se parecen a nada, la red igual logra recuperar la G y la S a la perfección. La A, la E y la R no las logra.

---

## Slide 30 — Ahora queremos generar, no sólo reconstruir

**🎤 Orador: Narias**

Bueno, pasamos al ejercicio 2. Hasta acá el autoencoder reconstruía, pero ahora lo que queremos es generar imágenes nuevas, no sólo copiar las que ya teníamos.

El problema es que el autoencoder de 1a guarda un punto por imagen. Si nosotros elegimos un punto nuevo cualquiera del espacio latente, casi siempre cae en una zona vacía que la red nunca vio durante el entrenamiento, así que lo que sale es ruido. De hecho ya lo vimos en 1a cuando inventamos letras: entre las letras reales aparecían símbolos raros.

Entonces, para poder generar muestras nuevas, necesitamos que el espacio latente tenga una forma conocida, donde cualquier punto que elijamos sea válido. Esa es justamente la idea del Autoencoder Variacional, el VAE.

---

## Slide 31 — Datos nuevos: 5 emojis

**🎤 Orador: Narias**

Para esta parte cambiamos de dataset. Elegimos 5 emojis: corazón, estrella, gota, luna y rayo. Los tomamos de OpenMoji y los usamos como siluetas rellenas en 20×20 píxeles, o sea 400 valores por imagen. 

A partir de esos 5 símbolos generamos 700 muestras con aumentaciones: a cada simbolo base le aplicamos rotaciones, desplazamientos y ruido suave, 140 veces cada uno. Son 5 clases por 140 igual a 700 muestras. La idea de aumentar así es que cada clase ocupe una pequeña región y no un único punto, para que el VAE aprenda una variedad continua y pueda interpolar entre los símbolos, no que memorice 5 puntos fijos.

---

## Slide 32 — Arquitectura del VAE

**🎤 Orador: Pina**

Es un autoencoder simétrico: la entrada son los 400 píxeles del emoji, pasa por una capa oculta de 128 neuronas con tangente hiperbólica, y llega al cuello, que es de 2 dimensiones (4 neuronas).

Acá está la diferencia con el autoencoder común: el encoder no devuelve un punto, devuelve una media μ y un logvar (el logaritmo de la varianza). 
Con eso armamos el z, con el truco de reparametrización, z = μ + σ·ε, y ese z lo pasamos por el decoder: otra capa de 128 con tanh y la salida de 400 con sigmoide. 

Lo entrenamos con Adam y learning rate 1e-, durante 3500 épocas con mini-batches de 128, sobre las 700 muestras de emojis.

---

## Slide 33 — Espacio latente: AE vs VAE

**🎤 Orador: Pina**

Entrenamos al AE del ej1 y al nuevo VAE con las 700 muestras, 
Antes de leer el gráfico conviene explicar rápido cómo funciona el VAE. La diferencia clave es que el encoder ya no produce un punto, produce los parámetros de una distribución: una media μ y un desvío σ. Cada imagen, en vez de caer en un punto, enciende una campanita gaussiana en el espacio latente. Después se samplea un z de esa campana y se lo pasa al decoder. Lo importante es que a la pérdida le sumamos el término KL, que empuja cada una de esas campanitas hacia la normal estándar, la N(0,I). Eso es lo que junta todo cerca del origen y rellena los huecos.

Graficamos la media latente de cada emoji: A la izquierda tenemos un autoencoder común, sin el término KL, y a la derecha el VAE. En el eje X e Y están las dos dimensiones del espacio latente.

Vemos que sin el KL, el latente queda disperso, en una escala de decenas. Entonces si sampleamos un punto de la N(0,I), prácticamente siempre caemos en una zona vacía y el decoder genera ruido. 
En cambio con el KL, el espacio latente queda compacto y centrado en 0, en una escala de más menos 2, conservando los 5 cúmulos separados. Por eso, samplear de la N(0,I) en el VAE sí nos da simbolos que hacen sentido.

---

## Slide 34 — El mapa de los emojis

**🎤 Orador: Pina**

Este es el mapa del espacio latente del VAE ya entrenado. Cada punto es la media de una muestra, coloreado por clase. Podemos ver que cada clase ocupa su propia zona dentro de la Normal 0 I: no es una sola nube uniforme, todavía se distinguen los 5 cúmulos. Las circunferencias marcan varianza 1σ y 2σ de la normal estándar, vemos que la mayor cantidad de muestras caen dentro de la gaussiana y se distribuyen alrededor de esta.

---

## Slide 35 — Reconstruir el original

**🎤 Orador: Katia**

Vemos que el VAE sigue reconstruyendo bien. Usamos de entrada algunas de las muestras que creamos nosotros, emojis aumentados. Arriba está el original y abajo la reconstrucción. Vemos que reconstruye bien muestras provenientes de las 5 clases.

---

## Slide 36 — Generar símbolos nuevos

**🎤 Orador: Katia**

Y acá viene lo interesante, que es generar nuevas muestras. Ahora no usamos ningún emoji de entrada, ni el encoder. 
Sampleamos puntos sintéticos z de la N(0,I), esos son los (z1, z2) que están anotados sobre cada símbolo, y los pasamos sólo por el decoder.

Vemos que salen muestras nuevas, no son copias de las del dataset: son variaciones continuas de los 5 prototipos. 

---

## Slide 37 — El mapa completo decodificado

**🎤 Orador: Katia**

Acá hacemos algo parecido a lo que hicimos en con letras inventadas del autoencoder, pero ahora sobre el latente del VAE. Tomamos una grilla de puntos z en el cuadrado de menos 2.5 a 2.5 en cada eje y decodificamos cada celda. 
Vemos que los emojis se transforman de a poco, uno se va convirtiendo en otro a medida que nos movemos por el espacio. O sea que hay una cobertura continua, no son copias sueltas: vemos que entre cada par de clases hay una transición gradual.

---

## Slide 38 — ¡Gracias!

**🎤 Orador: (a definir)**

Bueno, eso fue todo. ¡Muchas gracias!

---

<!-- FIN DEL GUION -->

