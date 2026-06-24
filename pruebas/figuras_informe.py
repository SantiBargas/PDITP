# Genera figuras prolijas para el informe (no es parte del pipeline 01-06).

import csv
import importlib
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "codigo"))

entrada = importlib.import_module("01_entrada")
preprocesamiento = importlib.import_module("02_preprocesamiento")
rectificacion = importlib.import_module("03_rectificacion")
clasificacion = importlib.import_module("04_clasificacion")

RESULTADOS_DIR = Path(__file__).resolve().parent.parent / "resultados"
FIGURAS_DIR = RESULTADOS_DIR / "figuras"
DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "yolo_dataset"


def _bgr_a_rgb(imagen: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)


def figura_pipeline(group_id: int = 25, image_id: int = 1, n_crops: int = 6) -> None:
    """Figura paso a paso: imagen original -> máscara -> individuos detectados -> recortes rectificados."""
    anotaciones = entrada.cargar_anotaciones()
    imagen, info = entrada.cargar_imagen(group_id, image_id, anotaciones)

    mascara = preprocesamiento.segmentar(imagen)
    rects = preprocesamiento.detectar_individuos(mascara)
    bboxes = preprocesamiento.dibujar_rects(imagen, rects)

    resultados = rectificacion.run(
        group_id=group_id, image_id=image_id,
        mostrar=False, guardar=False, verbose=False, anotaciones=anotaciones,
    )

    cols = 3
    n_crops = min(n_crops, len(resultados))
    filas_crops = max(1, (n_crops + cols - 1) // cols)
    alto_fila_crop = 2.2  # los recortes son tiras finas, no necesitan tanto alto como la fila 1
    fig, axes = plt.subplots(
        1 + filas_crops, cols,
        figsize=(4 * cols, 4 + alto_fila_crop * filas_crops),
        gridspec_kw={"height_ratios": [4] + [alto_fila_crop] * filas_crops},
    )

    axes[0, 0].imshow(_bgr_a_rgb(imagen))
    axes[0, 0].set_title("1. Imagen original")

    axes[0, 1].imshow(mascara, cmap="gray")
    axes[0, 1].set_title("2. Máscara (Otsu + morfología)")

    axes[0, 2].imshow(_bgr_a_rgb(bboxes))
    axes[0, 2].set_title(f"3. Individuos detectados ({len(rects)})")

    for i in range(n_crops):
        crop, pez = resultados[i]
        fila, col = divmod(i, cols)
        axes[1 + fila, col].imshow(_bgr_a_rgb(crop))
        axes[1 + fila, col].set_title(f"4. Rectificado: {pez['especie']}")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(f"Pipeline 01→03 — group_{group_id:02d}/{image_id:05d}.png", fontsize=14)
    fig.tight_layout()
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURAS_DIR / "pipeline.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"-> {out}")


def figura_predicciones(n: int = 6, group_id: int = 25, imagenes: range = range(1, 6), out_dir: Path | None = None) -> None:
    """Ejemplos de clasificación: recorte + especie real vs predicha + confianza."""
    anotaciones = entrada.cargar_anotaciones()

    crops, peces = [], []
    for image_id in imagenes:
        resultados = rectificacion.run(
            group_id=group_id, image_id=image_id,
            mostrar=False, guardar=False, verbose=False, anotaciones=anotaciones,
        )
        for crop, pez in resultados:
            crops.append(crop)
            peces.append(pez)
        if len(crops) >= n:
            break

    crops, peces = crops[:n], peces[:n]
    predicciones = clasificacion.predecir(crops)

    cols = 3
    filas = (n + cols - 1) // cols
    fig, axes = plt.subplots(filas, cols, figsize=(4 * cols, 4 * filas))
    axes = np.atleast_2d(axes)

    for idx in range(filas * cols):
        f, c = divmod(idx, cols)
        ax = axes[f, c]
        ax.axis("off")
        if idx >= len(crops):
            continue
        crop, pez, pred = crops[idx], peces[idx], predicciones[idx]
        ax.imshow(_bgr_a_rgb(crop))
        correcto = pred["especie"] == pez["especie"]
        color = "green" if correcto else "red"
        ax.set_title(
            f"Real: {pez['especie']}\nPredicho: {pred['especie']} ({pred['confianza']:.2f})",
            color=color, fontsize=11,
        )

    fig.suptitle("Ejemplos de clasificación (etapa 04)", fontsize=14)
    fig.tight_layout()
    out_dir = out_dir or FIGURAS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "predicciones.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"-> {out}")


def figura_matriz_confusion(out_dir: Path | None = None) -> None:
    """Matriz de confusión sobre el test ciego (group_21-25), normalizada por especie real."""
    especies = sorted(d.name for d in (DATASET_DIR / "test").iterdir() if d.is_dir())
    indice = {especie: i for i, especie in enumerate(especies)}

    imagenes, reales = [], []
    for especie in especies:
        for path in (DATASET_DIR / "test" / especie).glob("*.png"):
            imagenes.append(cv2.imread(str(path)))
            reales.append(especie)

    print(f"Clasificando {len(imagenes)} imágenes del test...")
    predicciones = clasificacion.predecir(imagenes)

    n = len(especies)
    matriz = np.zeros((n, n), dtype=int)
    for real, pred in zip(reales, predicciones):
        matriz[indice[real], indice[pred["especie"]]] += 1

    matriz_norm = matriz / matriz.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matriz_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(especies, rotation=45, ha="right")
    ax.set_yticklabels(especies)
    ax.set_xlabel("Especie predicha")
    ax.set_ylabel("Especie real")
    ax.set_title(f"Matriz de confusión — test ciego (n={len(imagenes)})")

    for i in range(n):
        for j in range(n):
            valor = matriz_norm[i, j]
            color = "white" if valor > 0.5 else "black"
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Proporción")
    fig.tight_layout()
    out_dir = out_dir or FIGURAS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "matriz_confusion.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"-> {out}")


def _leer_resultados_entrenamiento() -> tuple:
    csv_path = RESULTADOS_DIR / "clasificacion_yolov8n" / "results.csv"
    epocas, train_loss, val_loss, val_top1 = [], [], [], []
    with open(csv_path) as f:
        for fila in csv.DictReader(f):
            epocas.append(int(fila["epoch"]))
            train_loss.append(float(fila["train/loss"]))
            val_loss.append(float(fila["val/loss"]))
            val_top1.append(float(fila["metrics/accuracy_top1"]))
    return epocas, train_loss, val_loss, val_top1


def figura_perdida_entrenamiento() -> None:
    """Curva de loss (train/val) por época."""
    epocas, train_loss, val_loss, _ = _leer_resultados_entrenamiento()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(epocas, train_loss, label="train loss")
    ax.plot(epocas, val_loss, label="val loss")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Pérdida durante el entrenamiento")
    ax.legend()

    fig.tight_layout()
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURAS_DIR / "perdida_entrenamiento.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"-> {out}")


def figura_accuracy_entrenamiento() -> None:
    """Curva de accuracy (top1) de validación por época."""
    epocas, _, _, val_top1 = _leer_resultados_entrenamiento()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(epocas, val_top1, color="tab:green")
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy (top1)")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy de validación por época")

    fig.tight_layout()
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURAS_DIR / "accuracy_entrenamiento.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"-> {out}")


def figura_curvas_entrenamiento() -> None:
    """Curvas de loss y accuracy (top1) de entrenamiento/validación por época."""
    epocas, train_loss, val_loss, val_top1 = _leer_resultados_entrenamiento()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epocas, train_loss, label="train loss")
    ax1.plot(epocas, val_loss, label="val loss")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title("Pérdida durante el entrenamiento")
    ax1.legend()

    ax2.plot(epocas, val_top1, color="tab:green")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy (top1)")
    ax2.set_ylim(0, 1)
    ax2.set_title("Accuracy de validación por época")

    fig.suptitle("Entrenamiento YOLOv8n-cls", fontsize=14)
    fig.tight_layout()
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURAS_DIR / "curvas_entrenamiento.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"-> {out}")


def run() -> None:
    figura_pipeline()
    figura_predicciones()
    figura_matriz_confusion()
    figura_curvas_entrenamiento()


if __name__ == "__main__":
    run()
