#### DOCUMENTACION PARA GENERAR EL INFORME #####

## LIBRERIAS USADAS
OpenCV (cv2) — lectura/escritura de imágenes, conversión a escala de grises, blur, umbralización (Otsu), morfología, contornos, minAreaRect, warpPerspective, flip, manejo de ventanas
NumPy (np) — manejo de arrays, máscaras, operaciones vectorizadas (sumas, promedios, nonzero, etc.)
Pathlib (Path) — manejo de rutas de archivos/carpetas (de la librería estándar de Python, no hace falta instalarla)
json — para leer annotations.json (también estándar de Python)
importlib — para importar los módulos 01_entrada, 02_preprocesamiento, 03_rectificacion desde nombres con números (también estándar)
Ultralytics (ultralytics) — entrenamiento e inferencia con YOLOv8n-cls
PyTorch (torch, torchvision) — backend de Ultralytics, con soporte CUDA para usar tu GPU

pip install opencv-python numpy ultralytics
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

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

  esto nos va a servir para la etapa de entrenamiento y procesamiento

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
annotations.json nos da para cada pez del dataset datos que en la etapa 03 todavía no podemos calcular nosotros mismos:

* especie → la etapa 04 (clasificación YOLO) es la que debería darnos esto, pero todavía no está implementada. Por ahora la "tomamos prestada" del ground truth para poder nombrar los archivos de salida (pez_X_especie.png) y poder probar/visualizar el pipeline 01→03 de punta a punta.


### como sabemos que informacion de annotations.json corresponde a cada pez?
Cada anotación en annotations.json trae un bbox ([x, y, w, h]) que indica dónde está ese pez en la imagen original (en píxeles). Por ejemplo, el pez fish_id=316 tiene su bbox en cierta posición de la foto de 2464×2056.
Por otro lado, la etapa 02 (detectar_individuos) detecta los blobs por umbralización/contornos y devuelve un rect (de cv2.minAreaRect) por cada uno, que también tiene un centro (cx, cy) en las mismas coordenadas de la imagen original

Nota: este matching por bbox/centro es algo flojo del desarrollo, igual que el uso de "especie". En el pipeline final (fotos de campo, sin annotations.json) este paso no existe — la etapa 04 (YOLO) va a clasificar cada recorte directamente, sin necesidad de "saber de antemano" qué pez es cuál.

### Pasos del algoritmo (rectificación y unificación de orientación):

* 1 Calculamos la matriz de perspectiva para enderezar un rect rotado a horizontal
* 2 Recortamos y enderezamos un individuo (y su máscara) según su rect rotado
* 3 Estimamos en qué lado está la cabeza: comparamos el "grosor" (alto de la máscara) entre el primer y el último tercio del recorte — el extremo más grueso (ojo/opérculo) es la cabeza
* 4 Aplicamos espejo horizontal si la cabeza no quedó del lado canónico (izquierda)
* 5 Estimamos hacia dónde apunta el hocico: comparamos la posición vertical de la punta del hocico contra el centro vertical de la zona de la cabeza
* 6 Aplicamos espejo vertical si el hocico no apunta hacia el lado canónico (abajo)

Estos criterios (pasos 3-6) NO dependen de annotations.json — funcionan a partir de la máscara propia del pez, por lo que son aplicables al pipeline final con fotos reales.

## 04_clasificacion.py

* ENTRENAMOS CON YOLOV8N-CLS (LIVIANO)
* CADA CARPETA TIENE 60 FOTOS, DE 00001.PNG A 00040.PNG USABLES, LAS OTRAS TIENEN SUPERPUESTOS DE 41 A 60 .PNG
* ENTRENAMIENTO: group_01 a group_20 (20 grupos) → fotos 00001-00040 de cada uno
* VALIDACIÓN/PRUEBA: group_21 a group_25 (5 grupos) → fotos 00001-00040 de cada uno

ARMAMOS LA CARPETA yolo_dataset DENTRO DE data
yolo_dataset contendra las carpetas: 
train: para entrenar con cada clase separada por especie
val: para testear con los ultimos group_21 a group_25

COMO SE ARMA LA CARPETA YOLO_DATASET?
* El archivo 04_clasificacion.py corre la funcion generar_dataset() se corren los archivos 01 02 y 03 sobre cada foto y va guardando cada pez recortado según su especie.

* Por cada (grupo,foto) la funcion rectificacion.run() devuelve la lista de peces recortados de esa foto y cada uno se guarda como:
carpeta = DATASET_DIR / split / pez["especie"]
nombre = f"g{group_id:02d}_i{image_id:05d}_pez{i}.png"

Ej: data/yolo_dataset/train/whiting/g01_i00003_pez4.png = el pez nº4 detectado en group_01/00003.png, que según annotations.json es un whiting.

USAMOS ANNOTATIONS.JSON PARA PODER CLASIFICAR
--------------------------------------------------------------------------------------------------------------------------
ENTRENAR:
cd c:\Users\santi\Documents\PDI_tp\codigo
python -c "import importlib; m = importlib.import_module('04_clasificacion'); m.entrenar()"
--------------------------------------------------------------------------------------------------------------------------

Al archivo 04_clasificacion.py le pasamos el parametro patience = 10 a modelo.train()
Esto quiere decir que si durante 10 epocas seguidas no nota mejoras, corta el entrenamiento
Por el momento esta en 25 epocas