# Etapa 03 — Rectificación: orientación horizontal y unificación de lado visible

import importlib
from pathlib import Path

import cv2
import numpy as np

entrada = importlib.import_module("01_entrada")
preprocesamiento = importlib.import_module("02_preprocesamiento")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LADO_CABEZA_CANONICO = "izquierda"  # todos los recortes finales quedan con la cabeza de este lado
ORIENTACION_CANONICA = "abajo"  # todos los recortes finales quedan con la boca hacia abajo


def calcular_transformacion(rect) -> tuple:
    """Calcula la matriz de perspectiva para enderezar un rect rotado a horizontal."""
    box = np.intp(cv2.boxPoints(rect)).astype(np.float32)

    lado_a = np.linalg.norm(box[1] - box[0])
    lado_b = np.linalg.norm(box[2] - box[1])

    if lado_a >= lado_b:
        ancho, alto = int(round(lado_a)), int(round(lado_b))
        dst = np.array(
            [[0, alto - 1], [ancho - 1, alto - 1], [ancho - 1, 0], [0, 0]],
            dtype=np.float32,
        )
    else:
        ancho, alto = int(round(lado_b)), int(round(lado_a))
        dst = np.array(
            [[0, 0], [0, alto - 1], [ancho - 1, alto - 1], [ancho - 1, 0]],
            dtype=np.float32,
        )

    transformacion = cv2.getPerspectiveTransform(box, dst)
    return transformacion, (ancho, alto)


def rectificar(imagen: np.ndarray, mascara: np.ndarray, rect) -> tuple:
    """Recorta y endereza un individuo (y su máscara) según su rect rotado."""
    transformacion, (ancho, alto) = calcular_transformacion(rect)
    crop = cv2.warpPerspective(imagen, transformacion, (ancho, alto))
    mascara_crop = cv2.warpPerspective(
        mascara, transformacion, (ancho, alto), flags=cv2.INTER_NEAREST
    )
    return crop, mascara_crop


def ajustar_a_ventana(crop: np.ndarray, ancho_ventana: int, alto_ventana: int) -> np.ndarray:
    """Reescala el crop para que entre en la ventana sin deformarse (letterbox)."""
    alto, ancho = crop.shape[:2]
    escala = min(ancho_ventana / ancho, alto_ventana / alto)
    nuevo_ancho = max(1, int(round(ancho * escala)))
    nuevo_alto = max(1, int(round(alto * escala)))

    redimensionado = cv2.resize(crop, (nuevo_ancho, nuevo_alto))

    canvas = np.zeros((alto_ventana, ancho_ventana, 3), dtype=np.uint8)
    x = (ancho_ventana - nuevo_ancho) // 2
    y = (alto_ventana - nuevo_alto) // 2
    canvas[y:y + nuevo_alto, x:x + nuevo_ancho] = redimensionado
    return canvas


def detectar_lado_cabeza(mascara: np.ndarray) -> str:
    """Estima en qué lado está la cabeza: el extremo más grueso del pez (ojo/opérculo)."""
    grosor = (mascara > 0).sum(axis=0)

    tercio = max(1, mascara.shape[1] // 3)
    grosor_izq = grosor[:tercio].mean()
    grosor_der = grosor[-tercio:].mean()

    return "izquierda" if grosor_izq >= grosor_der else "derecha"


def unificar_cabeza(crop: np.ndarray, mascara: np.ndarray, lado_cabeza: str) -> tuple:
    """Aplica espejo horizontal si la cabeza no está del lado canónico."""
    if lado_cabeza != LADO_CABEZA_CANONICO:
        return cv2.flip(crop, 1), cv2.flip(mascara, 1)
    return crop, mascara


def detectar_orientacion_vertical(mascara: np.ndarray) -> str:
    """Estima si la punta del hocico apunta hacia arriba o abajo respecto al centro de la cabeza."""
    _, ancho = mascara.shape[:2]
    region_cabeza = mascara[:, :max(1, int(ancho * 0.2))]

    ys, xs = np.nonzero(region_cabeza)
    centro_y_cabeza = ys.mean()

    x_punta = xs.min()
    y_punta = ys[xs <= x_punta + 1].mean()

    return "abajo" if y_punta > centro_y_cabeza else "arriba"


def unificar_orientacion_vertical(crop: np.ndarray, mascara: np.ndarray, orientacion: str) -> tuple:
    """Aplica espejo vertical si la boca no está del lado canónico."""
    if orientacion != ORIENTACION_CANONICA:
        return cv2.flip(crop, 0), cv2.flip(mascara, 0)
    return crop, mascara


def emparejar_con_anotaciones(rects: list, peces: list) -> list:
    """Asocia cada rect detectado con el pez anotado más cercano (por centro), descartando
    rects/peces sin una correspondencia razonable (segmentación imperfecta: blobs fragmentados
    o peces no separados)."""
    candidatos = []
    for ri, rect in enumerate(rects):
        cx, cy = rect[0]
        for pi, pez in enumerate(peces):
            x, y, w, h = pez["bbox"]
            pcx, pcy = x + w / 2, y + h / 2
            dist2 = (pcx - cx) ** 2 + (pcy - cy) ** 2
            umbral2 = (max(w, h) / 2) ** 2
            if dist2 <= umbral2:
                candidatos.append((dist2, ri, pi))

    candidatos.sort(key=lambda c: c[0])

    rects_usados, peces_usados = set(), set()
    asignaciones = []
    for _, ri, pi in candidatos:
        if ri in rects_usados or pi in peces_usados:
            continue
        rects_usados.add(ri)
        peces_usados.add(pi)
        asignaciones.append((rects[ri], peces[pi]))

    return asignaciones


def run(
    group_id: int = 1,
    image_id: int = 1,
    mostrar: bool = True,
    guardar: bool = True,
    verbose: bool = True,
    anotaciones: dict | None = None,
) -> list:
    if anotaciones is None:
        anotaciones = entrada.cargar_anotaciones()
    imagen, info = entrada.cargar_imagen(group_id, image_id, anotaciones)

    mascara = preprocesamiento.segmentar(imagen)
    rects = preprocesamiento.detectar_individuos(mascara)
    pares = emparejar_con_anotaciones(rects, info["peces"])

    output_dir = DATA_DIR / "processed" / f"group_{group_id:02d}_{image_id:05d}"
    if guardar:
        output_dir.mkdir(parents=True, exist_ok=True)

    resultados = []
    for i, (rect, pez) in enumerate(pares):
        crop, mascara_crop = rectificar(imagen, mascara, rect)
        lado_cabeza = detectar_lado_cabeza(mascara_crop)
        crop, mascara_crop = unificar_cabeza(crop, mascara_crop, lado_cabeza)
        orientacion = detectar_orientacion_vertical(mascara_crop)
        crop, mascara_crop = unificar_orientacion_vertical(crop, mascara_crop, orientacion)
        resultados.append((crop, pez))

        if verbose:
            print(
                f"  pez {i}: especie={pez['especie']:<14} "
                f"side_up={pez['side_up']} cabeza_detectada={lado_cabeza:<9} "
                f"orientacion_detectada={orientacion:<6} -> "
                f"tamaño={crop.shape[1]}x{crop.shape[0]}"
            )

        if guardar:
            path = output_dir / f"pez_{i}_{pez['especie']}.png"
            cv2.imwrite(str(path), crop)

    if mostrar:
        ventanas = []
        for i, (crop, pez) in enumerate(resultados):
            ventana = f"Pez {i} ({pez['especie']})"
            alto, ancho = crop.shape[:2]
            cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(ventana, ancho, alto)
            ventanas.append(ventana)

        while True:
            for crop, ventana in zip((c for c, _ in resultados), ventanas):
                if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                    continue
                _, _, w, h = cv2.getWindowImageRect(ventana)
                if w <= 0 or h <= 0:
                    continue
                cv2.imshow(ventana, ajustar_a_ventana(crop, w, h))

            if cv2.waitKey(30) != -1:
                break
            if all(cv2.getWindowProperty(v, cv2.WND_PROP_VISIBLE) < 1 for v in ventanas):
                break

        cv2.destroyAllWindows()

    return resultados


if __name__ == "__main__":
    run()
