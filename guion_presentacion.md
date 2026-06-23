# Guion — TP5 Autoencoders

---

## Slide 1 — Portada

Bueno, empezamos con el TP5 de Autoencoders. El trabajo tiene tres partes: primero entrenamos un autoencoder básico para comprimir y reconstruir 32 letras ASCII, después entrenamos uno específicamente para limpiar ruido, y por último un VAE para generar imágenes nuevas de emojis.

---

## Slide 2 — Dataset: 32 caracteres ASCII

El dataset que vamos a usar es un conjunto de 32 letras ASCII. Son grillas de 7 por 5 píxeles que pueden ser blancos o negros, así que cada letra son 35 valores.

También medimos qué tan distintas son las letras entre sí, en cantidad de píxeles. En general se diferencian entre 10 y 20 píxeles. Hay algunos casos extremos: por ejemplo la K y la Q están bastante lejos, y en el otro extremo la L y el pipe son prácticamente iguales.

---

## Slide 3 — La configuración base

Para empezar, la configuración base del autoencoder aprovecha mucho de lo que ya teníamos del TP3: un perceptrón multicapa con 35 entradas y 35 salidas, una capa oculta de 20 neuronas, y el espacio latente en el medio con 2 neuronas, que es lo que pide la consigna.

Las activaciones son tangente hiperbólica para las capas ocultas, identidad para la capa latente, y sigmoide para la salida, para que cada píxel salga entre 0 y 1 y después lo umbralizamos en 0.5 para decidir si es blanco o negro.

Usamos la función de pérdida de entropía cruzada binaria, también heredada del TP3. Arrancamos con SGD, 6000 épocas, y full batch sobre las 32 letras.

---

## Slide 4 — Elección de optimizador (config)

El primer paso fue elegir el optimizador. Probamos tres: SGD con learning rate 0.5, Momentum con learning rate 0.1, y Adam con learning rate 0.01. El resto de la configuración se mantuvo fija.

---

## Slide 5 — Adam llega a 0 px (resultado)

Podemos ver algo muy bueno: Adam, de entrada, ya logra lo que pide la consigna, que es llegar a cero píxeles de error.

En el gráfico de barras vemos la cantidad de letras recuperadas perfectas y el error máximo que tuvo. SGD recuperó 10 de 32, con hasta 5 píxeles de error. Momentum recuperó 30 de 32, bastante bien, pero quedaron 2 letras con un pixel de error. Adam las recuperó todas perfectas.

Como Adam ya logró la perfección, nos quedamos con Adam.

---

## Slide 6 — Letras reconstruidas por optimizador

Visualmente, con SGD se ven letras bastante distintas a las originales. Momentum pifió en solo dos letras, de hecho la letra se lee bien, pero el pipe se convirtió en una L. Adam perfecta.

---

## Slide 7 — ¿Qué tan grande el paso? (config)

Ahora probamos distintos valores de learning rate: 0.0003, 0.01 y 0.3. El resto de la configuración se mantuvo fija con Adam.

---

## Slide 8 — 0.01 es el punto justo (resultado)

Podemos ver que 0.01 es el punto justo. El learning rate grande, 0.3, directamente no aprende: no converge y tiene un error máximo gigante, llegando a errar casi todos los píxeles. El learning rate chico, 0.0003, va en la dirección correcta pero no llega en 6000 épocas. Con más épocas probablemente llegaría al mismo resultado, pero no hace falta. 0.01 llega a cero píxeles de error, así que nos quedamos con ese.

---

## Slide 9 — Arquitectura: tamaño de la capa oculta (config)

Después investigamos la arquitectura. Probamos cambiar la capa oculta: sin capa oculta (lo que sería equivalente a una compresión lineal tipo PCA), una capa de 10, de 20, de 30, o dos capas apiladas de 20 de cada lado.

---

## Slide 10 — 20 neuronas es el mínimo justo (resultado)

Sin capa oculta solo recuperó perfecta una letra y tuvo hasta 14 píxeles de error. Con 10 neuronas reconstruye 18 letras pero igual quedan algunas con hasta 4 píxeles de error. A partir de 20 neuronas ya logra el resultado esperado: cero píxeles de error.

30 neuronas y la doble capa de 20 dieron el mismo resultado que 20. Como con 20 ya alcanza, nos quedamos con eso, no tiene sentido usar más.

---

## Slide 11 — Tamaño del espacio latente (config)

También probamos el tamaño del espacio latente, más allá de lo que pide la consigna. ¿Qué pasa si lo reducimos a una sola neurona? ¿Alcanza con eso para guardar 32 letras distintas?

---

## Slide 12 — Con 1 solo se aprende un subconjunto (resultado)

No. Con una sola neurona la red solo aprende un subconjunto de las letras: recupera 14 de 32 a la perfección, y las otras tienen hasta 9 píxeles de error. No alcanza con una sola dimensión para separar 32 letras diferentes.

Con 2 y 3 neuronas llega a cero error. La consigna pide 2, y confirmaos que alcanza, así que nos quedamos con eso.

---

## Slide 13 — Reconstrucciones (E8a)

Acá podemos ver visualmente que la red está reconstruyendo a la perfección las 32 letras: la fila de arriba son las originales y la de abajo las reconstruidas por el autoencoder.

---

## Slide 14 — Mapa del espacio latente

El espacio latente de 2 dimensiones se puede graficar. Cada letra queda ubicada en un punto del plano. Podemos ver que letras que visualmente se parecen tienden a quedar cerca: la O, la E y la B quedan juntas porque todas tienen una forma circular. La H, la N y la M también quedan relativamente cerca, aunque no tanto.

---

## Slide 15 — Inventar letras nuevas

Si tomamos una grilla de valores en el espacio latente y los pasamos por el decoder, podemos ver qué "inventa" la red. Aparecen las letras reales en sus posiciones, y entre ellas aparecen símbolos inventados, algunos que mezclan rasgos de las letras cercanas.

---

## Slide 16 — Interpolación A → O

Si en el espacio latente trazamos el vector que va desde la posición de la A hasta la posición de la O, y recorremos ese camino paso a paso, vemos cómo la letra va cambiando gradualmente. La A se deforma, pasa por una especie de U extraña, y termina siendo una O.

---

## Slide 17 — Denoising: Limpiar ruido

Ahora pasamos al denoising: entrenar una red específicamente para limpiar letras que tienen ruido.

---

## Slide 18 — ¿Cómo ensuciamos las letras?

El ruido que usamos es bit-flip: cada píxel se invierte con probabilidad P. Es el mismo mecanismo que en el TP4. Importante: en cada época de entrenamiento se generan inputs nuevos con ruido, la red nunca ve dos veces la misma mancha.

Visualmente, con 10% de ruido las letras todavía se parecen bastante a la original. Con 15% ya se empieza a perder la forma. Con 30% las letras ya no se parecen en nada a la original.

---

## Slide 19 — ¿Y si le damos ruido al AE de 1a?

Primero probamos qué pasa si le damos entradas con ruido al autoencoder que ya teníamos, sin reentrenarlo. La respuesta es que no funciona: con 10% de ruido ya empieza a errar en varias letras, y a partir de 15% no logra reconstruir ninguna correctamente. No fue entrenado para limpiar, así que la salida se deforma.

---

## Slide 20 — El AE de 1a no limpia

El error crece a medida que sube el ruido de entrada. Hay que entrenar la red específicamente para limpiar.

---

## Slide 21 — Entrenemos un denoiser

La idea es simple: generamos entradas ensuciando cada letra, y le exigimos a la red que devuelva la letra original limpia. En vez de pedirle que reconstruya la entrada (limpio → limpio), le pedimos que limpie (sucio → limpio).

La arquitectura y la función de pérdida son las mismas que antes. Solo cambia qué le pedimos como salida.

Para las pruebas que vienen, evaluamos 30 realizaciones con ruido por nivel × 32 letras, lo que da 960 entradas por cada nivel de ruido.

---

## Slide 22 — Nueva arquitectura: espacio latente

Para el denoiser probamos agrandar el espacio latente: 2, 5, 10 y 20 neuronas. Entrenamos cada modelo con 15% de ruido de entrada y después los evaluamos con entradas al 10%, 20% y 30% de ruido.

El gráfico de la izquierda muestra el promedio de píxeles incorrectos. El de la derecha muestra cuántas letras se recuperan casi perfectas (0 o 1 píxel de error) cuando la entrada tiene 20% de ruido.

Vemos un salto de calidad entre 5 y 10 neuronas similar al salto entre 2 y 5. Nos quedamos con 10 neuronas en el espacio latente.

---

## Slide 23a — Nueva arquitectura: capas ocultas (primera vista)

Con el espacio latente fijo en 10 neuronas, probamos distintos tamaños de capa oculta: 20, 30, 35 y 40 neuronas. Todos entrenados con 15% de ruido.

> [!warning] ACÁ LEER LA DIAPO TAMBIEN

---

## Slide 23b — Nueva arquitectura: capas ocultas (conclusión)

Hay un salto de calidad claro al pasar de 20 a 30 neuronas. Después de 30, aumentar la capacidad no mejora prácticamente nada.

Nos quedamos con 30 neuronas en la capa oculta.

---

## Slide 24 — ¿Cuánto ruido al entrenar? (descripción)

Con la arquitectura final 35-30-10-30-35, es preguntamos cómo se comporta al entrenar con distintos niveles de ruido de entrada.. Después a cada uno lo evaluamos pasándole entradas con niveles de ruido que van de 0% (limpio) hasta 40%, en saltos de 5%.

---

## Slide 25 — ¿Cuánto ruido al entrenar? (lectura)

Lo que vemos es que cada modelo se desenvuelve mejor en el rango de ruido con el que fue entrenado. La red entrenada al 5% funciona un poquito mejor con entrada completamente limpia, pero desde 10% de ruido en adelante, la supera la red entrenada al 15%. Y desde 15% para arriba, la entrenada al 5% es la que peor resultado da.

La red entrenada al 30% devuelve letras más ruidosas para entradas con poco ruido. Recién supera a las demás cuando el ruido de entrada llega a 30% o más.

En resumen, la red entrenada al 15% parece la mejor opción para un uso general: se porta bien desde entradas limpias hasta 25% de ruido.

---

## Slide 26 — Cualitativamente: efecto del 30% de ruido

Cuantitativamente la red entrenada al 30% parece comportarse mejor para entradas con mucho ruido, con 6 píxeles de error en promedio. Pero si miramos visualmente lo que devuelve, en la mayoría de los casos no logra reconstruir algo que se parezca a la letra original. Hay algunos casos donde sí: la P, la H, la N, la X y la Z las recupera bastante bien, probablemente porque son letras bastante distintas a las demás. Pero en general no es funcional.
Con nuestra arquitectura propuesta, no logramos un modelo que funcione para niveles altos de ruido. La resolución de los datos originales (7x5, 35 pixeles) implica que un 30% de ruido ya es bastante destructivo, aprox. 10 pixeles en promedio se modifican.

---

## Slide 27 — Resultado del denoiser (ganador)

El denoiser ganador tiene capa oculta de 30 neuronas, espacio latente de 10 y fue entrenado con 15% de ruido.

Lo probamos con 50 generaciones ruidosas × 32 letras por nivel, o sea 1600 entradas por nivel. Con 10% de ruido en la entrada, el 92% de las salidas tiene 0 o 1 píxel de error. Con 15% de ruido, sigue portándose bien: 80% de las letras quedan con 0 o 1 píxel de error.

---

## Slide 28 — Limpio / sucio / recuperado

Acá podemos ver tríos de imágenes: la letra original limpia, la entrada con ruido que le dimos a la red, y lo que la red devolvió.

Con 10% de ruido reconstruye 4 de 5 letras perfectas. Con 20% todavía se porta bien: la A y la R las recupera, la S perfecta. Con 30% de ruido, que visualmente las letras ya no se parecen a nada, la red igual logra recuperar la G y la S a la perfección. La A, la E y la R no las logra.

---

<!-- FIN DEL GUION -->

