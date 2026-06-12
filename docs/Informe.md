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

### Estructura de annotations.json

El archivo tiene 3 claves principales:

```json
{
  "images": [...],       // 1500 entradas, una por imagen
  "annotations": [...],  // 18158 entradas, una por pez
  "categories": [...]    // 7 especies
}
```

**images** — info de cada foto:
```json
{
  "height": 2056,
  "width": 2464,
  "id": 1,
  "file_name": "group_01/00001.png",
  "group": 1
}
```
- `id`: identificador interno de la imagen (las anotaciones apuntan a este id)
- `file_name`: ruta relativa dentro de data/
- `width`/`height`: tamaño de la foto en píxeles

**annotations** — info de cada pez:
```json
{
  "iscrowd": 0,
  "image_id": 1,
  "bbox": [381.0, 1123.0, 822.0, 378.0],
  "segmentation": [...],
  "category_id": 0,
  "length": 35.5,
  "fish_id": 316,
  "side_up": "R",
  "id": 1,
  "area": 92164
}
```
- `image_id`: a qué imagen pertenece este pez (matchea con images[i]["id"])
- `bbox = [x, y, w, h]`: caja delimitadora del pez en píxeles de la imagen original (x,y = esquina superior izquierda; w,h = ancho/alto)
- `category_id`: número de especie (se traduce con categories)
- `length`: largo real del pez en cm (ground truth)
- `fish_id`: id único del pez (se repite en distintas fotos del mismo individuo)
- `side_up`: qué lado del pez está hacia arriba en la foto (L/R)
- `segmentation`: máscara de segmentación del pez (polígono)

**categories** — diccionario de especies:
```json
[
  {"id": 0, "name": "horse_mackerel"},
  {"id": 1, "name": "whiting"},
  {"id": 2, "name": "haddock"},
  {"id": 3, "name": "cod"},
  {"id": 4, "name": "hake"},
  {"id": 5, "name": "saithe"},
  {"id": 6, "name": "other"}
]
```

### Dónde se usa esto en el código

**01_entrada.py → cargar_anotaciones()**: recorre images y crea un diccionario indice indexado por image_id, con file_name, width, height, group y una lista vacía peces. Recorre annotations y, para cada pez, lo agrega a indice[ann["image_id"]]["peces"] con: fish_id, bbox (el [x,y,w,h]), segmentation, length, category_id, especie (traducido con CATEGORIAS), side_up.

**01_entrada.py → cargar_imagen()**: busca en indice la entrada cuyo file_name sea group_XX/000NN.png y devuelve (imagen, info), donde info["peces"] es la lista de peces de esa foto, cada uno con su bbox.

**03_rectificacion.py → emparejar_con_anotaciones()**: acá es donde se usa el bbox = [x,y,w,h] de cada pez:
```python
x, y, w, h = pez["bbox"]
pcx, pcy = x + w / 2, y + h / 2
```
Calcula el centro del bbox (pcx, pcy) y lo compara con el centro (cx, cy) del rect que detectó la etapa 02 (cv2.minAreaRect), para encontrar qué blob detectado corresponde a qué pez anotado. Si la distancia entre ambos centros es menor a un umbral (max(w,h)/2 al cuadrado), los asocia. Si no, los descarta (segmentación imperfecta).

Para chequear un caso puntual (por ejemplo el pez fish_id=316, bbox=[381, 1123, 822, 378] de group_01/00001.png), se puede correr 01_entrada.run(group_id=1, image_id=1) y comparar esos valores contra lo que imprime, o correr 03_rectificacion.run(group_id=1, image_id=1, verbose=True) y ver con qué rect detectado quedó emparejado ese pez.

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

## Reunión 12/06/2026 — cambios a partir del feedback del profe

El profe validó el enfoque general (segmentación + rectificación propias, no depender de los bboxes precalculados de annotations.json). Pidió agregar 3 cosas: data augmentation justificada, un split train/val/test correcto (test 100% ciego), y documentar limitaciones/mejoras.

### Split del dataset (v2)

Antes: train=group_01-20 (80%), val=group_21-25 (20%) — pero ese "val" se usaba para el early stopping, o sea que influía en el entrenamiento. No era un test ciego.

Ahora, dentro de 04_clasificacion.py, el split queda así (sigue siendo 80/20 en total):

| Conjunto | Grupos | % | Para qué |
|---|---|---|---|
| train | group_01 a group_16 | 64% | ajusta los pesos del modelo |
| val | group_17 a group_20 | 16% | monitoreo época a época / early stopping (sigue siendo parte del 80%) |
| test | group_21 a group_25 | 20% | ciego — no se usa para nada durante el entrenamiento, se evalúa una sola vez al final |

`generar_dataset()` ahora borra `data/yolo_dataset` y lo regenera con estas 3 carpetas (train/val/test).

### Data augmentation (entrenar())

Se agregaron parámetros nativos de Ultralytics a `entrenar()`, cada uno justificado para este proyecto:

| Augmentación | Parámetro | Valor | Justificación |
|---|---|---|---|
| Brillo/contraste | `hsv_v` | 0.4 | luz variable según el momento del día en el desembarco |
| Variación de color | `hsv_h`, `hsv_s` | 0.015 / 0.7 | variaciones naturales de coloración entre ejemplares de la misma especie |
| Recorte parcial (occlusion) | `erasing` | 0.4 | un pez puede estar parcialmente fuera de cuadro o tapado |
| Flip vertical | `flipud` | 0.5 | la etapa 03 deja ~1 de cada 10 peces "boca arriba" por error de orientación — entrenar con flip vertical hace al modelo robusto a ese error |
| Flip horizontal | `fliplr` | 0 (desactivado) | la etapa 03 ya deja todos los peces con la cabeza a la izquierda (orientación canónica). Activar fliplr generaría peces con cabeza a la derecha, algo que nunca va a pasar en producción — contradiría la rectificación |

### Evaluación ciega (evaluar())

Nueva función `evaluar()` en 04_clasificacion.py: carga `best.pt` y corre `modelo.val(data=..., split="test")` sobre el 20% de test (group_21-25), que no participó ni del entrenamiento ni del early stopping. Resultados en `resultados/evaluacion_test/`.

### Pendiente para el informe

- Medir la tasa de error de orientación de la etapa 03 (comparar `side_up` esperado vs. lado/orientación detectado)
- Sección de limitaciones: confusión pez/fondo con panza plateada sobre fondo claro de la cinta
- Mejora propuesta: usar fondo de color verde (ningún pez tiene ese color) para el dataset propio de INALI, mejora el contraste para la segmentación