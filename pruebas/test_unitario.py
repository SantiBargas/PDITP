import os
import sys
import importlib.util

import cv2
import numpy as np

PALETA_COLORES = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
]

# Carga del módulo de preprocesamiento
ruta_remota = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codigo"))
sys.path.append(ruta_remota)

especificacion = importlib.util.find_spec("02_preprocesamiento")
if especificacion is not None:
    preprocesamiento = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(preprocesamiento)
else:
    raise ImportError("No se pudo encontrar el módulo '02_preprocesamiento'")


def cargar_modelo_yolo():
    """Carga el modelo YOLO entrenado del proyecto o el modelo base si no existe."""
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_codigo = os.path.abspath(os.path.join(ruta_script, "..", "codigo"))
    ruta_resultados = os.path.abspath(os.path.join(ruta_script, "..", "resultados", "clasificacion_yolov8n", "weights"))

    candidatos = [
        os.path.join(ruta_resultados, "best.pt"),
        os.path.join(ruta_resultados, "last.pt"),
        os.path.join(ruta_codigo, "yolov8n-cls.pt"),
    ]

    modelo_path = next((ruta for ruta in candidatos if os.path.exists(ruta)), None)
    if modelo_path is None:
        print("No se encontró un modelo YOLO entrenado ni el modelo base.")
        return None

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"No se pudo importar ultralytics: {exc}")
        return None

    return YOLO(modelo_path)


modelo = cargar_modelo_yolo()


def extraer_crop_rect(imagen: np.ndarray, rect) -> np.ndarray:
    """Genera un recorte rectificado a partir de un rectángulo rotado."""
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
    return cv2.warpPerspective(imagen, transformacion, (ancho, alto))


def clasificar_rect(imagen: np.ndarray, rect) -> dict | None:
    """Clasifica un rectángulo usando el modelo YOLO de clasificación."""
    if modelo is None:
        return None

    try:
        crop = extraer_crop_rect(imagen, rect)
        if crop.size == 0:
            return None

        resultados = modelo([crop], stream=False, verbose=False)
        if not resultados:
            return None

        resultado = resultados[0]
        if not hasattr(resultado, "probs") or not hasattr(resultado, "names"):
            return None

        top1 = int(getattr(resultado.probs, "top1", 0))
        nombre = resultado.names[top1]
        confianza = float(resultado.probs.top1conf)
        return {"nombre": nombre, "confianza": confianza}
    except Exception as exc:
        print(f"No se pudo clasificar el rectángulo: {exc}")
        return None


def crear_colores(nombres_clases, nombres_presentes=None):
    if nombres_presentes is None:
        nombres_presentes = nombres_clases

    nombres_presentes = [
        nombre for nombre in dict.fromkeys(nombres_presentes)
        if nombre in nombres_clases
    ]

    return {
        nombre: PALETA_COLORES[i % len(PALETA_COLORES)]
        for i, nombre in enumerate(nombres_presentes)
    }


def dibujar_rects(imagen: np.ndarray, rects: list, clasificaciones: list | None = None, colores: dict | None = None) -> np.ndarray:
    """Dibuja los rectángulos rotados sobre una copia de la imagen con color por clase."""
    resultado = imagen.copy()
    clasificaciones = clasificaciones or []
    colores = colores or {}

    for i, rect in enumerate(rects):
        box = np.intp(cv2.boxPoints(rect))
        color = (0, 255, 0)

        if i < len(clasificaciones) and clasificaciones[i] is not None:
            nombre_clase = clasificaciones[i]["nombre"]
            color = colores.get(nombre_clase, (0, 255, 0))

        largo = max(
            preprocesamiento._longitud_cm(box[0], box[1]),
            preprocesamiento._longitud_cm(box[0], box[2]),
        )
        cv2.putText(
            resultado,
            f"{largo:2.2f}cm",
            (int(rect[0][0]), int(rect[0][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            color,
            4,
        )
        cv2.drawContours(resultado, [box], 0, color, 3)
    return resultado


def dibujar_leyenda(imagen: np.ndarray, colores: dict) -> np.ndarray:
    """Dibuja una leyenda de clases presentes con su color correspondiente."""
    resultado = imagen.copy()
    x, y = 10, 40
    alto_linea = 30

    for nombre, color in colores.items():
        cv2.rectangle(resultado, (x - 5, y - 20), (x + 25, y + 5), color, -1)
        cv2.putText(
            resultado,
            nombre,
            (x + 35, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += alto_linea

    return resultado


def obtener_clasificaciones(imagen: np.ndarray, rects: list) -> tuple[list, list]:
    """Clasifica todos los rectángulos y devuelve las etiquetas y los nombres presentes."""
    clasificaciones = []
    nombres_presentes = []

    for rect in rects:
        clasificacion = clasificar_rect(imagen, rect)
        clasificaciones.append(clasificacion)
        if clasificacion is not None:
            nombres_presentes.append(clasificacion["nombre"])

    return clasificaciones, nombres_presentes



def run(path_img, mostrar: bool = True) -> tuple:
    path = path_img
    imagen = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if imagen is None:
        raise FileNotFoundError(f"No se encontró la imagen: {path}")

    mascara = preprocesamiento.segmentar(imagen)
    rects = preprocesamiento.detectar_individuos(mascara)
    clasificaciones, nombres_presentes = obtener_clasificaciones(imagen, rects)
    colores = crear_colores(nombres_presentes, nombres_presentes)

    print(f"Imagen: {path}")
    for i, rect in enumerate(rects):
        (cx, cy), (w, h), angulo = rect
        nombre_clase = f" | clase={clasificaciones[i]['nombre']} | confianza={clasificaciones[i]['confianza']:.2f}"
        print(
            f"  - individuo {i}: centro=(cx={cx:.0f}, cy={cy:.0f}) "
            f"tamaño=({w:.0f}x{h:.0f}) angulo={angulo:.1f}{nombre_clase}"
        )

    if mostrar:
        bboxes = dibujar_rects(imagen, rects, clasificaciones, colores)
        bboxes = dibujar_leyenda(bboxes, colores)

        cv2.namedWindow("Mascara", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Individuos detectados", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mascara", 800, 600)
        cv2.resizeWindow("Individuos detectados", 800, 600)

        print("Procesamiento completado. Presione 'q' para cerrar las ventanas de visualización.")
        while True:
            cv2.imshow("Mascara", mascara)
            cv2.imshow("Individuos detectados", bboxes)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        
        cv2.destroyAllWindows()

    return imagen, rects


def main():
    ruta_imagen = input("Ingrese la ruta de la imagen dentro de PDI_TP/data/: ").strip().strip('"').strip("'")
    if not ruta_imagen:
        print("No se ingresó una ruta válida.")
        return

    if not os.path.isabs(ruta_imagen):
        ruta_pruebas = os.path.dirname(os.path.abspath(__file__))
        ruta_proyecto = os.path.abspath(os.path.join(ruta_pruebas, ".."))
        ruta_data = os.path.join(ruta_proyecto, "data")
        ruta_imagen = os.path.join(ruta_data, ruta_imagen)

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print("No se pudo cargar la imagen.")
        return

    run(path_img=ruta_imagen, mostrar=True)



if __name__ == "__main__":
    while True:
        main()
        continuar = input("¿Desea procesar otra imagen? (s/n): ").strip().lower()
        if continuar != "s":
            break