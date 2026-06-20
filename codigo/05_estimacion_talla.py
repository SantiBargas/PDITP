# Etapa 05 — Estimación de talla: largo, ancho (en píxeles)

import numpy as np


def estimar(crop: np.ndarray) -> dict:
    """Mide el largo y ancho (en píxeles) de un individuo ya rectificado por la
    etapa 03. Como la etapa 02 ajusta un rectángulo rotado al contorno del pez y
    la etapa 03 lo recorta y endereza con ese rectángulo, las dimensiones del
    propio recorte son ya el largo y el ancho del individuo.    """
    alto, ancho = crop.shape[:2]
    largo = max(alto, ancho)* (100/2048) #cm/pixel
    return {"largo": largo}


def run() -> None:
    print("Etapa 05: usar estimar(crop) sobre los recortes rectificados de la etapa 03.")


if __name__ == "__main__":
    run()
