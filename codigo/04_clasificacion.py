# Etapa 04 — Clasificación de especies (YOLO, 6-7 clases)

import importlib
import shutil
from pathlib import Path

import cv2

entrada = importlib.import_module("01_entrada")
rectificacion = importlib.import_module("03_rectificacion")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = DATA_DIR / "yolo_dataset"

# Split 80/20 (train+val = 80%, test = 20%, ciego hasta la evaluación final)
GRUPOS_TRAIN = range(1, 17)   # group_01 .. group_16 (64%) -> ajusta los pesos
GRUPOS_VAL = range(17, 21)    # group_17 .. group_20 (16%) -> monitoreo/early stopping
GRUPOS_TEST = range(21, 26)   # group_21 .. group_25 (20%) -> evaluación ciega final
IMAGENES = range(1, 41)       # 00001 .. 00040 (sin solapamiento: Set1 + Set2)


def generar_dataset() -> None:
    """Corre 01->03 sobre group_01..25 (imgs 1-40) y guarda los recortes
    organizados por especie en data/yolo_dataset/{train,val,test}/<especie>/."""
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    anotaciones = entrada.cargar_anotaciones()

    for grupos, split in [(GRUPOS_TRAIN, "train"), (GRUPOS_VAL, "val"), (GRUPOS_TEST, "test")]:
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
    epochs: int = 50,     # cantidad máxima de pasadas completas sobre el dataset de train
    patience: int = 10,   # si pasan estas épocas sin mejorar val, corta antes (early stopping)
    imgsz: int = 224,     # tamaño (px) al que se redimensiona cada imagen antes de entrar a la red
    batch: int = 64,      # cantidad de imágenes que se procesan juntas en cada paso
    hsv_h: float = 0.015,  # variación de tono de color (ejemplares de la misma especie no son idénticos)
    hsv_s: float = 0.7,    # variación de saturación
    hsv_v: float = 0.4,    # variación de brillo/contraste (luz variable en el desembarco)
    erasing: float = 0.4,  # tapa zonas aleatorias (peces parcialmente ocluidos/recortados)
    flipud: float = 0.5,   # flip vertical: robustez a los ~10% de peces que la etapa 03 deja boca arriba
    fliplr: float = 0.0,   # flip horizontal desactivado: la etapa 03 ya deja todos los peces con cabeza a la izquierda
):
    """Entrena YOLOv8n-cls sobre data/yolo_dataset/{train,val} con data augmentation."""
    from ultralytics import YOLO

    modelo = YOLO("yolov8n-cls.pt")
    resultados = modelo.train(
        data=str(DATASET_DIR),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        erasing=erasing,
        flipud=flipud,
        fliplr=fliplr,
        project=str(DATA_DIR.parent / "resultados"),
        name="clasificacion_yolov8n",
        exist_ok=True,
    )
    return resultados


def evaluar(pesos: str | Path | None = None) -> tuple:
    """Evalúa el modelo sobre el conjunto de test (20% ciego, nunca visto durante el entrenamiento).

    Cada corrida queda en su propia carpeta numerada
    resultados/clasificacion_yolov8n/evaluaciones/testN/ (test1, test2, ...) para poder
    comparar varias corridas (la confianza puede variar un poco si se reentrena el modelo
    entre medio). Devuelve (resultados, carpeta_de_esta_corrida)."""
    from ultralytics import YOLO

    if pesos is None:
        pesos = DATA_DIR.parent / "resultados" / "clasificacion_yolov8n" / "weights" / "best.pt"

    eval_dir = DATA_DIR.parent / "resultados" / "clasificacion_yolov8n" / "evaluaciones"
    eval_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while (eval_dir / f"test{n}").exists():
        n += 1
    nombre = f"test{n}"

    modelo = YOLO(str(pesos))
    resultados = modelo.val(
        data=str(DATASET_DIR),
        split="test",
        project=str(eval_dir),
        name=nombre,
    )
    print(f"\nAccuracy (top1) en test ciego: {resultados.top1:.3f}")
    return resultados, eval_dir / nombre


def predecir(imagenes: list, pesos: str | Path | None = None, batch: int = 32) -> list:
    """Clasifica cada recorte (imagen ya rectificada por la etapa 03) con el modelo
    entrenado. Devuelve una lista de {"especie": str, "confianza": float}."""
    from ultralytics import YOLO

    if pesos is None:
        pesos = DATA_DIR.parent / "resultados" / "clasificacion_yolov8n" / "weights" / "best.pt"

    modelo = YOLO(str(pesos))
    resultados = modelo(imagenes, verbose=False, batch=batch)

    predicciones = []
    for r in resultados:
        top1 = r.probs.top1
        predicciones.append({
            "especie": r.names[top1],
            "confianza": float(r.probs.top1conf),
        })
    return predicciones


def run() -> None:
    generar_dataset()
    entrenar()
    evaluar()


if __name__ == "__main__":
    run()
