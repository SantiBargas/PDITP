# Etapa 01 — Carga de imagen y anotaciones del dataset AutoFish

import json
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CATEGORIAS = {
    0: "horse_mackerel",
    1: "whiting",
    2: "haddock",
    3: "cod",
    4: "hake",
    5: "saithe",
    6: "other",
}


def cargar_anotaciones() -> dict:
    """Carga annotations.json y arma un índice: image_id -> info de imagen + lista de peces."""
    with open(DATA_DIR / "annotations.json") as f:
        data = json.load(f)

    indice = {}
    for img in data["images"]:
        indice[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "group": img["group"],
            "peces": [],
        }

    for ann in data["annotations"]:
        indice[ann["image_id"]]["peces"].append({
            "fish_id": ann["fish_id"],
            "bbox": ann["bbox"],
            "segmentation": ann["segmentation"],
            "length": ann["length"],
            "category_id": ann["category_id"],
            "especie": CATEGORIAS[ann["category_id"]],
            "side_up": ann["side_up"],
        })

    return indice


def cargar_imagen(group_id: int, image_id: int, anotaciones: dict) -> tuple:
    """Carga la imagen group_XX/000NN.png y devuelve (imagen, info_anotaciones)."""
    file_name = f"group_{group_id:02d}/{image_id:05d}.png"

    info = next(
        (v for v in anotaciones.values() if v["file_name"] == file_name),
        None,
    )
    if info is None:
        raise ValueError(f"No hay anotaciones para {file_name}")

    path = DATA_DIR / file_name
    imagen = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if imagen is None:
        raise FileNotFoundError(f"No se encontró la imagen: {path}")

    return imagen, info


def run(group_id: int = 1, image_id: int = 1, mostrar: bool = True) -> tuple:
    anotaciones = cargar_anotaciones()
    imagen, info = cargar_imagen(group_id, image_id, anotaciones)

    print(f"Imagen: {info['file_name']}")
    print(f"Dimensiones: {info['width']}x{info['height']}")
    print(f"Cantidad de peces: {len(info['peces'])}")
    for pez in info["peces"]:
        x, y, w, h = pez["bbox"]
        print(
            f"  - fish_id={pez['fish_id']:>4} "
            f"especie={pez['especie']:<14} "
            f"largo={pez['length']:.1f}cm "
            f"side_up={pez['side_up']} "
            f"bbox=(x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f})"
        )

    if mostrar:
        imagen_anotada = imagen.copy()
        for pez in info["peces"]:
            x, y, w, h = (int(v) for v in pez["bbox"])
            cv2.rectangle(imagen_anotada, (x, y), (x + w, y + h), (0, 255, 0), 4)
            etiqueta = f"{pez['fish_id']} {pez['especie']}"
            cv2.putText(
                imagen_anotada, etiqueta, (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
            )

        coords = {"x": 0, "y": 0}

        def actualizar_mouse(event, x, y, flags, param):
            coords["x"], coords["y"] = x, y

        cv2.namedWindow("Imagen original", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Imagen original", 800, 600)
        cv2.setMouseCallback("Imagen original", actualizar_mouse)

        while True:
            frame = imagen_anotada.copy()
            texto = f"x={coords['x']}, y={coords['y']}"
            cv2.putText(
                frame, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3
            )

            cv2.imshow("Imagen original", frame)

            if cv2.waitKey(30) != -1:
                break
            if cv2.getWindowProperty("Imagen original", cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    return imagen, info


if __name__ == "__main__":
    run()
