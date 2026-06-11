# Etapa 04 — Clasificación de especies (YOLO, 6-7 clases)

import importlib
from pathlib import Path

import cv2

entrada = importlib.import_module("01_entrada")
rectificacion = importlib.import_module("03_rectificacion")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = DATA_DIR / "yolo_dataset"

GRUPOS_TRAIN = range(1, 21)   # group_01 .. group_20
GRUPOS_VAL = range(21, 26)    # group_21 .. group_25
IMAGENES = range(1, 41)       # 00001 .. 00040 (sin solapamiento: Set1 + Set2)


def generar_dataset() -> None:
    """Corre 01->03 sobre group_01..25 (imgs 1-40) y guarda los recortes
    organizados por especie en data/yolo_dataset/{train,val}/<especie>/."""
    anotaciones = entrada.cargar_anotaciones()

    for grupos, split in [(GRUPOS_TRAIN, "train"), (GRUPOS_VAL, "val")]:
        for group_id in grupos:
            for image_id in IMAGENES:
                resultados = rectificacion.run(
                    group_id=group_id,
                    image_id=image_id,
                    mostrar=False,
                    guardar=False,
                    verbose=False,
                    anotaciones=anotaciones,
                )

                for i, (crop, pez) in enumerate(resultados):
                    carpeta = DATASET_DIR / split / pez["especie"]
                    carpeta.mkdir(parents=True, exist_ok=True)
                    nombre = f"g{group_id:02d}_i{image_id:05d}_pez{i}.png"
                    cv2.imwrite(str(carpeta / nombre), crop)

            print(f"  {split}: group_{group_id:02d} listo")


def entrenar(
    epochs: int = 50,    # cantidad máxima de pasadas completas sobre el dataset de train
    patience: int = 10,  # si pasan estas épocas sin mejorar val, corta antes (early stopping)
    imgsz: int = 224,    # tamaño (px) al que se redimensiona cada imagen antes de entrar a la red
    batch: int = 64,     # cantidad de imágenes que se procesan juntas en cada paso
):
    """Entrena YOLOv8n-cls sobre data/yolo_dataset/{train,val}."""
    from ultralytics import YOLO

    modelo = YOLO("yolov8n-cls.pt")
    resultados = modelo.train(
        data=str(DATASET_DIR),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        project=str(DATA_DIR.parent / "resultados"),
        name="clasificacion_yolov8n",
    )
    return resultados


def run() -> None:
    generar_dataset()
    entrenar()


if __name__ == "__main__":
    run()
