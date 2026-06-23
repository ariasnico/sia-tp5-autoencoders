  
### eliminar diapo 2 "Que es un autoencoder"
### diapo 3 "## El dataset: 32 caracteres"
Cambiar titulo a "Dataset: 32 caracteres ASCII"
Eliminar "- 32 caracteres de 7×5 (casi todos letras), a guardar con solo 2 números. Difícil: hay pares casi idénticos (l y | a 2 px) que igual hay que separar."

### diapo 4
Cambair subtitulo "- Partimos de acá y, cambiando un parámetro por vez, llegamos a la receta ganadora. Para cada paso: primero la config, después el resultado." Reemplazar por Aprovechamos lo implementado en el mlp del TP3"

### diapo 5
Cambiar titulo a "Elección de optimizador"
Eliminar subtitulo "- Con la arquitectura ya capaz, comparamos optimizadores en igualdad de condiciones."

### diapo 6 
Agregar subtitulo:
- Vemos que usar Adam ya logra superar la limitación de espacio latente de dim 2

### diapo 7
Eliminar subtitulo "- El learning rate es propio del optimizador → se barre después de fijar Adam."

### diapo 10
Modificar titulo "Arquitectura -> Tamaño de la capa oculta"
Eliminar subtitulo "- ¿Hace falta capa oculta? ¿qué tan ancha? ¿qué tan profunda?"

### diapo 12
Cambiar titulo "Tamaño del espacio latente"
Eliminar subtitulo "- ¿2 dimensiones alcanzan para separar las 32 letras?"

## diapo 14
Eliminar titulo ## Reconstrucción: sin errores
Eliminar subtitulo
Dejar solo el grafico ocupando la mayor cantidad posible de espacio de la slide

# denoising

### diapo 18
eliminar todo el texto y que quede solo
"Denoising -- Limpiar ruido"
### La diapo 19 de "## Todo lo que probamos", eliminarla.
### Nueva diapo 19
Estaria bueno explicar y mostrar como se genera el ruido y ejemplos de 10/15/30% de ruido:
En la actual diapo 24 hay un grafico mostrando distintos niveles de ruido y su recuperacion para las letras a e g r s ; Hacer un grafico que muestre las letras y su version con cada nivel de ruido e incluirlo justo despues de la diapo 18 del titulo
### Nueva diapo 20 + 21
Estaria bueno hacer un forward pass de letras con ruido de 10% , 15% y 30% con la red original a ver como se comporta. 
diapo 20 Mostrar comparativa para las letras a e g r s con estas cantidades de ruido vs su output generado.
diapo 21 Mostrar graficos de px promedio errados con inputs de 10/15/30%
### actual diapo 20 - ahora es la 22
Cambiar el titulo a "Nueva arquitectura - Espacio latente"
Subtitulo "Probamos de agrandar la dimensión del espacio latente : 2, 5, 10 y 20 neuronas"
Regenerar el grafico para que no se llame "ancho del cuello" si no "tamaño del E latente"
Claude, fijate si hay otros lugares donde se llame cuello, hay que corregir.
Texto debajo de los graficos
- Medimos el promedio de px incorrectos en los resultados generados vs el input original sin ruido
- Medimos la cantidad de letras recuperadas "perfecto" con 20% de ruido, i.e. con error de px e {0, 1}.
### nueva diapo 23 -- Nueva arquitectura -- Capas ocultas
Explicar que se probo y graficos que muestran los resultados y la decision de capa oculta de 30 neuronas
### actual diapo 21 -- ahora es la 24
Agregar una diapo DESPUES de esta , mostrando graficos de como se comporta cada una de las 3 redes en estos casos: letra a, nivel de ruido de input 0% 5%, 15%, 30%, 40%.
Quiero que quede graficado en una tabla:
Primer fila (centrada)la letra sin ruido, segunda fila la letra con cada nivel de ruido, tercer, cuarta y quinta filas la respuesta generada de cada red entrenada con cada nivel de ruido.
### eliminar: actual diapo 22 "Juntando lo mejor"
