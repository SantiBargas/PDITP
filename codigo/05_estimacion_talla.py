# Etapa 05 — Estimación de talla: largo, ancho (en píxeles)

import numpy as np


def estimar(crop: np.ndarray) -> dict:
    """Mide el largo y ancho (en píxeles) de un individuo ya rectificado por la
    etapa 03. Como la etapa 02 ajusta un rectángulo rotado al contorno del pez y
    la etapa 03 lo recorta y endereza con ese rectángulo, las dimensiones del
    propio recorte son ya el largo y el ancho del individuo.

    La conversión a centímetros requiere un objeto de referencia calibrado en la
    foto (ver condiciones de captura en CLAUDE.md) — no disponible en AutoFish,
    queda pendiente para el dataset propio de INALI."""
    alto, ancho = crop.shape[:2]
    return {"largo_px": ancho, "ancho_px": alto}


def run() -> None:
    print("Etapa 05: usar estimar(crop) sobre los recortes rectificados de la etapa 03.")


if __name__ == "__main__":
    run()
