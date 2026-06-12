# Pipeline completo: 01 (entrada) -> 06 (salida estructurada)
# Corre todas las etapas de punta a punta, regenerando dataset, modelo y resultados.

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pruebas"))

entrada = importlib.import_module("01_entrada")
rectificacion = importlib.import_module("03_rectificacion")
clasificacion = importlib.import_module("04_clasificacion")
talla = importlib.import_module("05_estimacion_talla")
salida = importlib.import_module("06_salida")
figuras = importlib.import_module("figuras_informe")

# Imagen de demo para las etapas 01-03 y 05-06 (group_25 = parte del test ciego,
# no vista durante el entrenamiento de la etapa 04)
GROUP_ID = 25
IMAGE_ID = 1


def main() -> None:
    print("=== Etapa 01: carga de imagen y anotaciones ===")
    anotaciones = entrada.cargar_anotaciones()
    entrada.run(group_id=GROUP_ID, image_id=IMAGE_ID, mostrar=False)

    print("\n=== Etapas 02-03: segmentación y rectificación de individuos ===")
    resultados_rect = rectificacion.run(
        group_id=GROUP_ID,
        image_id=IMAGE_ID,
        mostrar=False,
        guardar=True,
        verbose=True,
        anotaciones=anotaciones,
    )

    print("\n=== Etapa 04: clasificación (regenerar dataset + entrenar + evaluar) ===")
    clasificacion.generar_dataset()
    clasificacion.entrenar()
    _, carpeta_test = clasificacion.evaluar()

    print("\n=== Etapa 04: inferencia sobre los individuos de la imagen de demo ===")
    crops = [crop for crop, _ in resultados_rect]
    predicciones = clasificacion.predecir(crops)

    print("\n=== Etapa 05: estimación de talla (en píxeles) ===")
    medidas = [talla.estimar(crop) for crop in crops]

    print("\n=== Etapa 06: salida estructurada ===")
    salida.generar(resultados_rect, predicciones, medidas, group_id=GROUP_ID, image_id=IMAGE_ID)

    print("\n=== Figuras para el informe ===")
    figuras.figura_pipeline(group_id=GROUP_ID, image_id=IMAGE_ID)
    figuras.figura_curvas_entrenamiento()
    figuras.figura_matriz_confusion(out_dir=carpeta_test)
    figuras.figura_predicciones(out_dir=carpeta_test)
    print(f"Figuras de esta evaluación en {carpeta_test}")


if __name__ == "__main__":
    main()
