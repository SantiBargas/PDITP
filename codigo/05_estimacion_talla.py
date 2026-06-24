# Etapa 05 — Estimación de talla: largo, ancho (en píxeles)

import numpy as np

# AutoFish fotografía una sección de 100x100 cm de la cinta con una imagen de
# 2464x2056 px. El recorte rectificado ya perdió la orientación
# original (dx, dy) respecto de la imagen, así que se usa el promedio de la
# escala cm/píxel de ambos ejes como aproximación.
ESCALA_CM_PROMEDIO = (100 / 2464 + 100 / 2056) / 2


def estimar(crop: np.ndarray) -> dict:
    """Mide el largo y ancho (en píxeles) de un individuo ya rectificado por la
    etapa 03. Como la etapa 02 ajusta un rectángulo rotado al contorno del pez y
    la etapa 03 lo recorta y endereza con ese rectángulo, las dimensiones del
    propio recorte son ya el largo y el ancho del individuo.    """
    alto, ancho = crop.shape[:2]
    largo = max(alto, ancho) * ESCALA_CM_PROMEDIO
    return {"largo": largo}


def run() -> None:
    print("Etapa 05: usar estimar(crop) sobre los recortes rectificados de la etapa 03.")


if __name__ == "__main__":
    run()
