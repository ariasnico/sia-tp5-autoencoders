# Guía para entender el TP5 — diapo por diapo

> Pensado para leer y entender TODO, como si no supieras nada. Está en criollo a propósito.
> Cada diapo tiene: **qué se ve**, **la idea de fondo**, y cuando hace falta, **si te preguntan** (la
> trampa típica y cómo zafar). El guion hablado está en `presentacion/index_guion.html` (tecla S).

---

## Primero: las 4 ideas base (con esto entendés todo lo demás)

1. **Red neuronal.** Una máquina con miles de "perillas" (se llaman *pesos*) que transforma una entrada
   (una imagen) en una salida. Al principio las perillas están al azar, así que devuelve cualquier cosa.

2. **Entrenar.** Mostrarle ejemplos miles de veces y, cada vez que se equivoca, mover un poquito las
   perillas para errar menos. Después de miles de vueltas (las llamamos *épocas*), "aprendió".

3. **Autoencoder.** Una red con forma de **reloj de arena**: le entra una imagen, la hace pasar por un
   **cuello angosto** en el medio, y del otro lado tiene que devolver **la misma imagen**. Como por el
   cuello no entra toda la info, la red está obligada a **resumir**: a quedarse con lo esencial.

4. **El "mapa" (espacio latente).** Son los poquitos números que quedan en el cuello: el resumen de la
   imagen. Acá el cuello son **2 números**, así que cada imagen es **un punto en un mapa de 2 ejes**
   (como una posición con coordenada X y coordenada Y).

> Todo el TP es jugar con ese cuello: **1a** ¿se puede resumir muchísimo?, **1b** ¿sirve para limpiar?,
> **2 (VAE)** ¿se puede inventar?

---

# PARTE 1a — Comprimir letras

## Diapo 1 — Portada
**Qué se ve:** título, materia, integrantes. **Qué decir:** somos el grupo, esto es el TP5, lo hicimos
todo a mano con numpy (sin librerías de redes ya hechas). Son tres partes.

## Diapo 2 — ¿Qué es un autoencoder?
**La idea de fondo:** el reloj de arena de la idea base 3. Le entra una imagen, pasa por el cuello chico,
y tiene que salir igual. Al no poder copiar todo, **resume**. Las tres preguntas del TP, una por parte.

## Diapo 3 — El dataset: 32 letras
**Qué se ve:** a la izquierda, las 32 letras de `font.h` (7×5 puntos, blanco y negro); a la derecha, un
**mapa de calor** de cuán distintas son entre sí (cuántos puntos cambian de una letra a otra).
**La idea:** guardar cada letra con **solo 2 números** y reconstruirla. Es difícil porque varias letras se
parecen muchísimo: el mapa de calor muestra que el par más parecido (`l` y `|`) difiere en **solo 2 puntitos**
— y al modelo le va a tocar separarlas igual.
**Si te preguntan "¿por qué 2 números?":** porque la consigna pide un espacio latente de 2 dimensiones
(para poder dibujarlo en un gráfico plano), y además queremos comprimir lo más posible.

## Diapo 4 — E1: ¿Hacen falta las curvas?
**Qué se ve:** a la izquierda, barras del error de 3 modelos; a la derecha, un mapa de calor de **dónde**
falla el PCA (error por puntito, promedio de las 32 letras) — se concentra en el **centro del glifo**, los
puntos que distinguen una letra de otra. **La idea:** una red puede ser "recta" (lineal) o "con
curvas" (no-lineal). La recta resultó ser **idéntica a un método clásico de estadística, el PCA**, y se
equivoca en ~7 puntos por letra. La con curvas: **0 errores**.
**Concepto nuevo — PCA:** es una técnica vieja que también comprime datos a pocos números, pero **solo
con líneas rectas** (proyecta los datos sobre los 2 ejes que mejor los explican). Un autoencoder sin
curvas hace exactamente eso → da el mismo resultado.
**Si te preguntan "¿por qué el lineal es igual a PCA?":** porque matemáticamente, un autoencoder sin
funciones no-lineales que minimiza el error cuadrático **es** PCA. No es casualidad, es el mismo cálculo.
**Conclusión:** las curvas (la no-linealidad) son lo que permite bajar de 7 a 0.

## Diapo 5 — E2: ¿Cuántos números por letra?
**Qué se ve:** una curva de error según cuántos números usamos en el cuello (1, 2, 3, 5, 8).
**La idea:** con **1** no alcanza (solo aprende 18 de las 32 — las letras se pisan sobre una sola línea).
Con **2** ya salen todas perfectas, y agregar más no mejora. Por eso usamos 2. A esa forma de curva
(baja de golpe y después se aplana) se la llama "el codo".
**Si te preguntan por el caso de 1:** es el ejemplo que la consigna pide de "qué pasa cuando no se puede
aprender todo el conjunto": con 1 número no entran las 32, con 2 sí.

## Diapo 6 — E3: ¿Qué tan grande la red?
**Qué se ve:** barras (error según el tamaño) + curvas de entrenamiento. **La idea:** probamos redes de
distinto tamaño. Sin capa intermedia, la red es "recta" otra vez (≈PCA, falla). Con **20 neuronas** ya
sale perfecto; más grande no aporta.
**Si ves picos en las curvas:** son saltos normales del optimizador (Adam) cuando hay pocos datos; la
red se recupera sola en pocas vueltas. No es un error.

## Diapo 7 — E4: ¿Cómo entrenar?
**Qué se ve:** curvas de error a lo largo del entrenamiento, una por optimizador. **Concepto nuevo —
optimizador:** es la "estrategia" para mover las perillas. Probamos 3: SGD (la básica), Momentum (con
impulso) y **Adam** (la más lista, ajusta el paso sola). **Adam llega más abajo** (aprende mejor); las
otras se quedan a mitad de camino. → Usamos Adam.

## Diapo 8 — E5: ¿Qué tan grande el paso? (opcional)
**La idea:** el "paso" (*learning rate*) es cuánto se corrige en cada vuelta. Muy chico = tarda una
eternidad; muy grande = no aprende (se queda arriba); el del medio = justo. Sirve para mostrar también
**lo que NO funciona**, que la cátedra valora.

## Diapo 9 — E6: ¿Qué tipo de neurona? (opcional)
**Concepto — activación:** cada neurona aplica una "función" (tanh, relu, sigmoid). Probamos las 3.
Las **3 terminan perfectas**; lo único que cambia es cuál llega más rápido. O sea, cualquiera sirve.

## Diapo 10 — E7: ¿Cómo medir el error?
**La idea:** para entrenar hay que medirle el error a la red. Probamos 2 formas de medir: **BCE** y MSE.
Para imágenes en blanco y negro, **BCE** es la que mejor les queda → 0 errores. MSE deja 2.
**Por qué (en simple):** los puntos son blanco o negro (no grises), y BCE es la forma de medir pensada
para cosas que son "sí o no". MSE las trata como números cualquiera y queda peor.

## Diapo 11 — Resumen de 1a
**Qué se ve:** una tabla con TODOS los experimentos: qué cambiamos, **para qué** (la pregunta) y qué dio.
Es la foto completa del laburo, antes de mostrar el ganador.

## Diapo 12 — La receta ganadora
**Qué se ve:** la configuración del mejor modelo (todos los ingredientes juntos: 2 números, curvas,
20 neuronas, Adam, BCE). **Importante:** esta tabla es **un solo modelo, la receta final** — no es lo
que probamos (eso fue la tabla anterior). Es "juntando lo mejor de cada experimento, queda esto".

## Diapo 13 — Reconstrucción: sin errores (el ganador en acción)
**Qué se ve:** pares de letras: arriba la original, abajo la que devuelve la red. **Son idénticas: 0
errores en las 32.** Cumplimos el objetivo de la consigna con margen.

## Diapo 14 — El mapa de 2 números
**Qué se ve:** las 32 letras ubicadas como puntos en el mapa de 2 ejes. Cada letra cayó en un lugar.
El recuadro de la derecha es un **zoom** del centro (donde quedan amontonadas) para poder leerlas.
**La idea:** este mapa ES el "resumen" que aprendió la red: cada letra = una posición (2 números).

## Diapo 15 — ¿El mapa respeta las similitudes? (nueva)
**Qué se ve:** dos mapas de calor lado a lado: a la izquierda, cuán distintas son las letras en **puntos**
(píxeles); a la derecha, cuán lejos quedaron en el **mapa de 2 números** del modelo. Arriba, un número: la
correlación entre ambas (ρ = 0.57).
**La idea — el contraste (esto es lo bueno):** el modelo **distingue las 32 letras a la perfección** (tiene
que, si no, no podría reconstruirlas sin error). Pero el **parecido** entre letras lo conserva **solo a
medias** (ρ = 0.57): las parecidas *tienden* a quedar cerca, pero los pares casi idénticos (como `l` y `|`)
los **separa a propósito** — porque para reconstruir las dos sin error necesita poder distinguirlas.
**En una frase:** preserva la **distinguibilidad** entera; el **parecido**, a medias. Optimiza
reconstrucción, no la métrica de distancias.
**Si te preguntan "¿0.57 no es poco?":** es lo honesto — la correlación es claramente positiva (no es azar),
pero no perfecta, y el motivo es justo que el AE separa los casi-idénticos para poder reconstruirlos.

## Diapo 16 — Inventar letras nuevas
**Qué se ve:** una grilla donde recorrimos el mapa y dibujamos qué hay en cada punto. **La idea:** en los
puntos donde NO hay ninguna letra real, igual aparece algo dibujado — letras nuevas. O sea, la red ya
puede inventar un poco (decodificar puntos que nunca le mostramos).

## Diapo 17 — De la 'a' a la 'o'
**Qué se ve:** una fila que va morfeando de la 'a' a la 'o'. **La idea:** caminamos en línea recta en el
mapa, de donde está la 'a' a donde está la 'o', y dibujamos los puntos del medio. Salen **mezclas suaves**.
Muestra que el mapa es **continuo** (no tiene huecos): entre dos letras hay un camino de formas válidas.

---

# PARTE 1b — Limpiar ruido (denoising)

## Diapo 18 — Intro: Limpiar ruido
**La idea:** misma red, pero entrenada distinto. En vez de darle la imagen y pedirle la misma, le damos
la imagen **sucia** (con puntos cambiados al azar, como manchas) y le pedimos la **limpia**. Así aprende
a **sacar el ruido**, no solo a copiar.

## Diapo 19 — E9: ¿Por qué un cuello más ancho?
**Qué se ve:** cuánto recupera según el ancho del cuello (2, 5, 10, 20). **La idea:** el cuello de 2 (el
de la parte anterior) para limpiar funciona **mal** (recupera menos de la mitad). Con **cuello 10** mejora
mucho; más de 10 ya no cambia.
**Por qué:** limpiar es más difícil que solo resumir — la red necesita **más lugar para trabajar**. Por
eso acá usamos 10 y no 2.

## Diapo 20 — E10: ¿Cuánto ruido al entrenar?
**Qué se ve:** varias curvas que **se cruzan**. **La idea:** si entrenás con **poco** ruido, después
limpia bien lo poco sucio pero se rompe con lo muy sucio. Si entrenás con **mucho**, al revés (aguanta
mucha mugre pero pierde precisión con lo limpio). Como las curvas se cruzan, **no hay un único mejor**:
depende de cuánta suciedad esperás. Elegimos un punto intermedio (15%) como equilibrio.

## Diapo 21 — Resumen de 1b
**Qué se ve:** la tabla con los dos experimentos (cuello y ruido), para qué y qué dieron.

## Diapo 22 — La receta ganadora (1b)
**Qué se ve:** la receta del mejor limpiador: cuello 10, ruido de entrenamiento en el punto justo, bien
entrenado a fondo.

## Diapo 23 — Resultado del limpiador
**Qué se ve:** barras con cuánto recupera a distintos niveles de ruido. **La idea:** recupera **la gran
mayoría** de las letras: casi todas con poco ruido, todavía muchas con bastante ruido.

## Diapo 24 — Limpio / sucio / recuperado
**Qué se ve:** tríos de imágenes: arriba la letra limpia, en el medio la ensuciada, abajo la que recupera
la red. Es la prueba visual de que funciona. Anda bien hasta niveles de ruido altos.

---

# PARTE 2 — Generar (VAE)

> **La gran idea de esta parte:** el autoencoder normal **resume**, pero no sabe **inventar de cero**. Su
> mapa tiene huecos: si pedís un punto al azar, probablemente caés en un vacío y sale basura. El **VAE**
> arregla eso **ordenando el mapa** para que no tenga huecos.

## Diapo 25 — Intro: Generar
**La idea:** el modelo nuevo (VAE) guarda **una zona** (una nubecita) por imagen, en vez de un solo punto.
Eso, que parece un detalle, es justo lo que después le permite **inventar**: como cada imagen ocupa una
zona, el mapa se llena y no quedan huecos.

## Diapo 26 — Datos nuevos: emojis
**Qué se ve:** los 5 emojis que usamos (corazón, estrella, gota, luna, rayo) y una versión augmentada.
**La idea:** cambiamos de letras a emojis (la consigna pide un dataset nuevo). La primera versión que
probamos (de contorno) se confundía a tamaño chico, así que la cambiamos por **siluetas rellenas**, que
se ven bien claras. Lo verificamos con un test antes de seguir.

## Diapo 27 — E12: El ingrediente que ordena el mapa
**Qué se ve:** filas de emojis generados, una por cada valor del "ingrediente". **La idea:** la fila de
arriba es el modelo **sin** el ingrediente que ordena el mapa → al pedirle algo nuevo salen **manchas,
basura**. Las otras filas, **con** el ingrediente → salen **emojis de verdad**. Esto muestra para qué
sirve ordenar el mapa.
**Concepto — el "ingrediente" es el término KL:** en criollo, es una regla extra durante el entrenamiento
que **empuja todas las zonas hacia el centro del mapa**, juntitas y sin huecos. Así, después, cualquier
punto del centro que pidas cae en algo válido.

## Diapo 28 — E16: Por qué uno genera y el otro no
**Qué se ve:** dos mapas lado a lado. **La idea:** a la izquierda, **sin** ordenar: los puntos quedan
**re desparramados** (escala enorme). Si pedís uno al azar, caés en un lugar vacío → basura. A la derecha,
**ordenado**: todo queda juntito en el centro. Por eso este **sí** genera y el otro no. **Esta es la
diapo clave** que explica de fondo todo el VAE.

## Diapo 29 — El experimento y la receta del VAE
**Qué se ve:** la tabla con el único experimento que variamos: un número llamado **beta**, que es **cuánto
ordenamos el mapa**. Con beta 0 sale ruido; con beta alto se ordena perfecto pero las imágenes pierden
detalle; elegimos el **punto del medio** (beta 1) como equilibrio.

## Diapo 30 — El mapa de los emojis
**Qué se ve:** el mapa del VAE, con cada tipo de emoji pintado de un color. **La idea:** cada clase quedó
en **su propia zona**, y todas agrupadas cerca del centro (que es lo que el "ingrediente" logró). Justo
lo que buscábamos.

## Diapo 31 — Generar muestras nuevas  ⭐ (la que te confundía)
**Qué se ve:** 3 filas. **Ojo, no son lo mismo:**
- **Fila 1 (original)** y **fila 2 (reconstruido)** van **emparejadas**: a la red le entró *esa* imagen
  (fila 1) y devolvió *esto* (fila 2). Muestra que reconstruye bien.
- **Fila 3 (generado nuevo)** **NO** está emparejada con las de arriba: son 10 muestras **nuevas (no
  copias)**, pidiéndole al modelo **10 puntos al azar del mapa**. No salieron de ninguna imagen de entrada;
  son variaciones continuas de los 5 prototipos que vio (no una sexta clase inventada).
- **Honesto:** una tirada al azar no siempre saca las 5 clases — en promedio cubre el 85 % (≈4 de 5,
  medido sobre 200 semillas, ver `fig_e18`). Por eso para esta figura se eligió una semilla que muestra
  las cinco; el control sin elegir es `fig_e18`.

**Qué quiere decir "nuevo":** NO una clase que nunca vio (sí vio lunas). Quiere decir **una luna que el
modelo dibujó él, que no es ninguna de las lunas del dataset**. Analogía: le mostrás a un nene 140 lunas
distintas, aprende a dibujar lunas, y después dibuja una nueva — es nueva aunque sea una luna.
**Reconstruir vs generar:** reconstruir (fila 1→2) es "te doy esta luna, devolvémela" (copia lo que le
das). Generar (fila 3) es "inventá una desde números al azar" (no hay nada que copiar).
**Si te preguntan "¿cómo sé que no copia?":** lo mandás a la diapo 32 (el atlas), donde hay emojis a
medio camino entre dos (mitad luna, mitad rayo) — mezclas que **no existen en el dataset**.

## Diapo 32 — El mapa completo decodificado (atlas)
**Qué se ve:** una grilla gigante donde recorrimos TODO el mapa y dibujamos qué hay en cada punto.
**La idea:** los emojis se **transforman de a poco** uno en otro (corazón → estrella → gota → rayo → luna),
suavemente. Esto confirma que el modelo **genera mezclas continuas nuevas** de los 5 prototipos, no que
copia los que ya tenía. Es la prueba más fuerte de generación de verdad.

## Diapo 33 — El equilibrio (opcional)
**Qué se ve:** dos curvas según beta. **La idea:** cuanto más ordenás el mapa (beta más alto), más prolijo
queda pero las imágenes salen un poco **peores**. Hay un punto de equilibrio, y elegimos el del medio.

## Diapo 34 — Conclusiones
- **1a:** con curvas y 2 números guardamos las 32 letras perfecto; un método clásico (PCA) no llega.
- **1b:** limpiar ruido necesita un cuello más ancho, y el ruido de entrenamiento es un equilibrio.
- **VAE:** ordenando el mapa, el modelo pasa de **resumir** a **generar** cosas nuevas.
- Todo programado por nosotros, a mano.

---

## Las 4 preguntas capciosas más probables (tené las respuestas listas)
1. **¿Por qué el AE lineal da igual que PCA?** Porque un autoencoder sin curvas que minimiza error
   cuadrático **es** PCA, mismo cálculo. (Diapo 4)
2. **¿El 0 error no es "hacer trampa" / overfitting?** No: el objetivo acá es memorizar esas 32 letras
   exactas (son todo el universo, no una muestra). Querés que las clave. Distinto del denoising y el VAE,
   donde sí importa generalizar.
3. **¿Cómo sabés que el VAE genera y no copia?** Por el atlas (diapo 32): hay mezclas entre clases que no
   existen en el dataset. (Diapos 31 y 32)
4. **¿Qué es el "ingrediente" / KL en criollo?** Una regla que durante el entrenamiento empuja todas las
   zonas del mapa hacia el centro, juntas y sin huecos, para que después cualquier punto al azar caiga en
   algo válido. (Diapos 27 y 28)
