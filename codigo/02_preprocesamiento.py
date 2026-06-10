# Etapa 02 — Separación de fondo e individuos

from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MIN_CONTOUR_AREA = 25_000
MORPHOLOGY_KERNEL_SIZE = (35, 35)


def segmentar(imagen: np.ndarray) -> np.ndarray:
    """Separa fondo de individuos: escala de grises + Otsu + morfología."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(gris, (7, 7), 0)

    _, mascara = cv2.threshold(
        suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, MORPHOLOGY_KERNEL_SIZE)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    return mascara


def detectar_individuos(mascara: np.ndarray) -> list:
    """Devuelve un rectángulo rotado (cv2.minAreaRect) por cada individuo detectado."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < MIN_CONTOUR_AREA:
            continue
        rects.append(cv2.minAreaRect(contorno))

    return rects


def dibujar_rects(imagen: np.ndarray, rects: list) -> np.ndarray:
    """Dibuja los rectángulos rotados sobre una copia de la imagen (visualización)."""
    resultado = imagen.copy()
    for rect in rects:
        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(resultado, [box], 0, (0, 0, 255), 3)
    return resultado


def run(group_id: int = 1, image_id: int = 1, mostrar: bool = True) -> tuple:
    path = DATA_DIR / f"group_{group_id:02d}/{image_id:05d}.png"
    imagen = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if imagen is None:
        raise FileNotFoundError(f"No se encontró la imagen: {path}")

    mascara = segmentar(imagen)
    rects = detectar_individuos(mascara)

    print(f"Imagen: {path.relative_to(DATA_DIR)}")
    print(f"Individuos detectados: {len(rects)}")

    if mostrar:
        bboxes = dibujar_rects(imagen, rects)

        cv2.namedWindow("Mascara", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Individuos detectados", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mascara", 800, 600)
        cv2.resizeWindow("Individuos detectados", 800, 600)

        cv2.imshow("Mascara", mascara)
        cv2.imshow("Individuos detectados", bboxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return imagen, rects


if __name__ == "__main__":
    run()
