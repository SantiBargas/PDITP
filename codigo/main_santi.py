import importlib

etapas = [
    "01_entrada",
    "02_preprocesamiento",
    "03_rectificacion",
    "04_clasificacion",
    "05_estimacion_talla",
    "06_salida",
]

for etapa in etapas:
    modulo = importlib.import_module(etapa)
    modulo.run()
