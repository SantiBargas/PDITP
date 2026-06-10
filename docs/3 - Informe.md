#### DOCUMENTACION PARA GENERAR EL INFORME #####

## 01_entrada.py :
en este codigo lo que hacemos es leer annotations.json que tiene relacionada a cada imagen del dataset su informacion: por ej: 

Imagen: group_01/00001.png
Dimensiones: 2464x2056
Cantidad de peces: 8
  - fish_id= 316 especie=horse_mackerel largo=35.5cm side_up=R
  - fish_id= 419 especie=whiting        largo=36.0cm side_up=L
  - fish_id= 139 especie=haddock        largo=32.5cm side_up=L
  - fish_id= 159 especie=haddock        largo=32.0cm side_up=L
  - fish_id= 318 especie=horse_mackerel largo=37.0cm side_up=R
  - fish_id= 420 especie=whiting        largo=39.5cm side_up=L
  - fish_id= 140 especie=haddock        largo=33.0cm side_up=L
  - fish_id= 434 especie=whiting        largo=41.0cm side_up=R

  esto nos va a servir para la etapa de entrenamiento y procesamientoy procesamiento

## 02_preprocesamiento.py :
a partir de la imagen cruda (toda la cinta con varios peces), separa el fondo y detecta dónde está cada pez individual — sin recortar todavía, solo ubicándolos.
Sin esto, el clasificador (etapa 04) recibiría la foto completa con 8 peces mezclados — no podría asignar una especie por pez ni medir cada uno por separado. Esta etapa es el primer paso para aislar cada individuo, que la etapa 03 va a recortar y enderezar usando estos rects.

* 1 Conversión a escala de grises (cv2.cvtColor)
* 2 Suavizado (cv2.GaussianBlur, kernel 7x7)
* 3 Umbralización con Otsu (cv2.threshold + THRESH_OTSU)
* 4 Limpieza morfológica (MORPH_OPEN + MORPH_CLOSE, kernel cruz 35x35)
* 5 Detección de contornos (cv2.findContours, RETR_EXTERNAL)
* 6 Filtrado por área (MIN_CONTOUR_AREA = 25.000 px) descartamos contornos muy chicos por ruido
* 7 Rectángulo rotado por individuo (cv2.minAreaRect)

## 03_rectificacion.py
annotations.json nos da para cada pez del dataset dos datos que en la etapa 03 todavía no podemos calcular nosotros mismos:

* especie → la etapa 04 (clasificación YOLO) es la que debería darnos esto, pero todavía no está implementada. Por ahora la "tomamos prestada" del ground truth para poder nombrar los archivos de salida (pez_X_especie.png) y poder probar/visualizar el pipeline 01→03 de punta a punta.

* side_up → indica qué costado del pez está hacia arriba. Esto no se puede derivar del rect de la etapa 02 (que solo da posición/ángulo), así que sin esta anotación no sabríamos si hace falta espejar o no.

* 1 Calculamos la matriz de perspectiva para enderezar un rect rotado a horizontal
* 2 Recortamos y endereza un individuo (y su máscara) según su rect rotado