# PDI — Trabajo Práctico Integrador

## Descripción del proyecto

Sistema de relevamiento digital para fiscalizadores en puntos de desembarco del río Paraná. Procesa fotos tomadas desde móvil para: segmentar individuos, estimar talla via objeto de referencia calibrado, y clasificar especies según el catálogo local.

Materia: Procesamiento Digital de Imágenes | Entregable: artículo estilo congreso (4 páginas)

## Pipeline (6 etapas)

```
01 Imagen entrada        → foto móvil + objeto de referencia
02 Preprocesamiento      → separar fondo, aislar individuos (sin solapamiento primero)
03 Rectificación         → orientar horizontal, cabeza al mismo lado (mirror si hace falta)
04 Clasificación         → YOLO, 6-7 especies del río Paraná
05 Estimación de talla   → largo, ancho, correlación peso/edad
06 Salida                → especie + medidas por individuo
```

Prioridad actual: etapas 01-03. Las etapas 04-06 son próxima etapa / trabajo futuro.

## Estructura de carpetas

```
PDI_tp/
├── codigo/
│   ├── 01_entrada.py           # Etapa 01: carga de imagen y objeto de referencia
│   ├── 02_preprocesamiento.py  # Etapa 02: separación de fondo e individuos
│   ├── 03_rectificacion.py     # Etapa 03: orientación y normalización
│   ├── 04_clasificacion.py     # Etapa 04: entrenamiento y uso de YOLO
│   ├── 05_estimacion_talla.py  # Etapa 05: cálculo de medidas morfológicas
│   └── 06_salida.py            # Etapa 06: salida estructurada por individuo
├── data/                   # Dataset (NO commitear imágenes — ver .gitignore)
│   ├── raw/                # Imágenes originales del dataset de cinta transportadora
│   └── processed/          # Subimágenes rectificadas listas para clasificar
├── docs/                   # PDFs de descripción del proyecto y pasos a seguir
├── notebooks/              # Jupyter notebooks de exploración y análisis
├── pruebas/                # Scripts experimentales y pruebas rápidas
├── resultados/             # Outputs, métricas y visualizaciones generadas
└── README.md
```

## Dataset

- **Nombre:** AutoFish — https://huggingface.co/datasets/vapaau/autofish
- **Origen:** cinta transportadora, Visual Analysis and Perception Lab (Aalborg University)
- **Tamaño:** ~1.500 imágenes, 18.160 máscaras de segmentación de instancia, 15.9 GB total
- **Clases (6):** Cod, Haddock, Whiting, Hake, Horse mackerel, Other
- **Estructura:** 25 grupos (`group_01` a `group_25`), cada uno con:
  - `Set1` y `Set2` — peces sin solapamiento (mitad cada uno)
  - `All` — todos los peces con solapamiento intencional
  - 20 imágenes por subconjunto con variación de posición, orientación y flip
- **Anotaciones (`annotations.json`, 12.6 MB):** máscaras de segmentación, bounding boxes, largo de cada pez, IDs únicos, y `side_up` (qué lado del pez es visible — útil para etapa 03)
- **Para etapas 01-03:** usar `Set1` y `Set2` (sin solapamiento). `All` queda para trabajo futuro.
- **Dataset INALI propio:** disponible desde julio/agosto 2026 (salidas de campo río Paraná)
- Las imágenes van en `data/raw/` y **no se suben al repositorio** (están en `.gitignore`)

## Stack técnico

- Python 3.x
- OpenCV (`cv2`) — preprocesamiento, morfología, contornos
- NumPy
- Ultralytics YOLO — clasificación y (futuro) segmentación con solapamiento
- Jupyter notebooks para exploración

## Convenciones

- Los scripts en `pruebas/` son experimentales; el código limpio y funcional va en `codigo/`
- Cada etapa tiene su propio archivo en `codigo/` con prefijo numérico: `01_entrada.py`, `02_preprocesamiento.py`, etc.
- Cada etapa debe tener una función principal con entrada y salida claramente definidas (facilita integrar el pipeline completo)
- No commitear el dataset ni imágenes generadas (usar `.gitignore`)
- Commits en español, descriptivos: `"agrega umbralización con Otsu para separación de fondo"`

## Estado actual

- [x] Descripción del problema y pipeline definido
- [x] Script experimental de preprocesamiento (`pruebas/prueba4.py`): umbralización Otsu + morfología + bounding boxes rotados
- [x] Dataset identificado: AutoFish, 6 clases, 1.500 imgs, anotaciones completas en `annotations.json`
- [ ] Etapa 02 limpia en `codigo/02_preprocesamiento.py`
- [ ] Etapa 03: rectificación de individuos
- [ ] Entrenamiento YOLO (etapa 04)
- [ ] Estimación de talla (etapa 05)

## Condiciones de captura de imagen

- Peces sobre superficie plana con fondo homogéneo y contrastante
- Objeto de referencia incluido en la toma para calibrar escala de píxeles
- Sin sostener con la mano
- Todas las condiciones sujetas a los resultados de las pruebas
