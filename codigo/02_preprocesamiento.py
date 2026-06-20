# Etapa 02 — Separación de fondo e individuos

from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MIN_CONTOUR_AREA = 25_000
MAX_CONTOUR_AREA = 150_000
MORPHOLOGY_KERNEL_SIZE = (35, 35)
BORDE_MUESTRA_PX = 30


def _color_fondo(imagen: np.ndarray) -> np.ndarray:
    """Estima el color de fondo muestreando el borde de la imagen (BGR)."""
    borde = BORDE_MUESTRA_PX
    franjas = np.concatenate([
        imagen[:borde, :].reshape(-1, 3),
        imagen[-borde:, :].reshape(-1, 3),
        imagen[:, :borde].reshape(-1, 3),
        imagen[:, -borde:].reshape(-1, 3),
    ])
    return np.median(franjas, axis=0)


def segmentar(imagen: np.ndarray) -> np.ndarray:
    """Separa fondo de individuos por distancia de color al fondo + Otsu + morfología.

    El color de fondo se estima muestreando el borde de la imagen, por lo que
    funciona tanto con el fondo claro de AutoFish como con fondos de otro
    color (p. ej. verde), siempre que sea homogéneo y visible en los bordes.
    """
    suavizada = cv2.GaussianBlur(imagen, (7, 7), 0)
    color_fondo = _color_fondo(suavizada)
    distancia = np.linalg.norm(suavizada.astype(np.float32) - color_fondo, axis=2)
    distancia_8u = cv2.normalize(distancia, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mascara = cv2.threshold(distancia_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue
        rects.append(cv2.minAreaRect(contorno))

    return rects


def dibujar_rects(imagen: np.ndarray, rects: list) -> np.ndarray:
    """Dibuja los rectángulos rotados sobre una copia de la imagen (visualización)."""
    resultado = imagen.copy()
    for rect in rects:
        box = np.intp(cv2.boxPoints(rect))
        largo = max(np.linalg.norm(box[0]-box[1], 2), np.linalg.norm(box[0]-box[2], 2)) * (100/2048) #cm/pixel        
        cv2.putText(resultado, 
            f"{largo:2.2f}cm", 
            (int(rect[0][0]), int(rect[0][1])),
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4
        )

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
    for i, rect in enumerate(rects):
        (cx, cy), (w, h), angulo = rect
        print(
            f"  - individuo {i}: centro=(cx={cx:.0f}, cy={cy:.0f}) "
            f"tamaño=({w:.0f}x{h:.0f}) angulo={angulo:.1f}"
        )

    if mostrar:
        bboxes = dibujar_rects(imagen, rects)

        coords = {"x": 0, "y": 0}

        def actualizar_mouse(event, x, y, flags, param):
            coords["x"], coords["y"] = x, y

        cv2.namedWindow("Mascara", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Individuos detectados", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mascara", 800, 600)
        cv2.resizeWindow("Individuos detectados", 800, 600)
        cv2.setMouseCallback("Individuos detectados", actualizar_mouse)

        while True:
            frame = bboxes.copy()
            texto = f"x={coords['x']}, y={coords['y']}"
            cv2.putText(
                frame, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3
            )

            cv2.imshow("Mascara", mascara)
            cv2.imshow("Individuos detectados", frame)

            if cv2.waitKey(30) != -1:
                break
            if cv2.getWindowProperty("Individuos detectados", cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    return imagen, rects


if __name__ == "__main__":
    for group_id in range(1, 6):
        for image_id in range(1, 30):
            run(group_id=group_id, image_id=image_id, mostrar=True)
