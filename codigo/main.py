import cv2
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MIN_CONTOUR_AREA = 25_000
MORPHOLOGY_KERNEL_SIZE = (35, 35)
OUTPUT_DIR = DATA_DIR / "fish_crops"


def load_image(path: str = "group_01/00003.png") -> np.ndarray:
    full_path = DATA_DIR / path
    image = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se encontró la imagen: {full_path}")
    return image


def build_binary_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, MORPHOLOGY_KERNEL_SIZE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_fish_rectangles(mask: np.ndarray) -> list:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        rect = cv2.minAreaRect(contour)
        rects.append((contour, rect, area))

    return rects


def draw_bounding_boxes(image: np.ndarray, rects: list) -> np.ndarray:
    result = image.copy()

    for _, rect, _ in rects:
        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(result, [box], 0, (0, 0, 255), 2)

    return result


def compute_warp_matrix(rect) -> tuple:
    box = np.intp(cv2.boxPoints(rect)).astype(np.float32)

    side_a = np.linalg.norm(box[1] - box[0])
    side_b = np.linalg.norm(box[2] - box[1])

    if side_a >= side_b:
        long_side = int(round(side_a))
        short_side = int(round(side_b))
        dst_pts = np.array(
            [
                [0, short_side - 1],
                [long_side - 1, short_side - 1],
                [long_side - 1, 0],
                [0, 0],
            ],
            dtype=np.float32,
        )
    else:
        long_side = int(round(side_b))
        short_side = int(round(side_a))
        dst_pts = np.array(
            [
                [0, 0],
                [0, short_side - 1],
                [long_side - 1, short_side - 1],
                [long_side - 1, 0],
            ],
            dtype=np.float32,
        )

    transform = cv2.getPerspectiveTransform(box, dst_pts)
    return transform, (long_side, short_side)


def save_fish_crops( group_id: int, image_id: int, image: np.ndarray, rects: list, output_dir: Path = OUTPUT_DIR) -> list:
    output_dir.mkdir(exist_ok=True)
    saved_paths = []

    for index, (_, rect, _) in enumerate(rects):
        transform, (width, height) = compute_warp_matrix(rect)
        warped = cv2.warpPerspective(image, transform, (width, height))

        output_path = output_dir / f"pez_{group_id}_{image_id}_{index}.png"
        cv2.imwrite(str(output_path), warped)
        saved_paths.append(output_path)
        print(f"Guardado: {output_path} (ancho={width}, alto={height})")

    return saved_paths


def show_results(mask: np.ndarray, bounding_boxes: np.ndarray) -> None:
    cv2.namedWindow("Máscara", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Bounding Boxes", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Máscara", 800, 600)
    cv2.resizeWindow("Bounding Boxes", 800, 600)

    cv2.imshow("Máscara", mask)
    cv2.imshow("Bounding Boxes", bounding_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(group_id: int, image_id: int, show: bool = False) -> None:
    path = f"group_{group_id:02d}/{image_id:05d}.png"
    image = load_image(path)
    mask = build_binary_mask(image)
    rects = find_fish_rectangles(mask)
    bounding_boxes = draw_bounding_boxes(image, rects)
    save_fish_crops(group_id, image_id, image, rects)
    if show:
        show_results(mask, bounding_boxes)


if __name__ == "__main__":
    for group_id in range(1, 25+1):
        for image_id in range(1, 40+1):
            main(group_id=group_id, image_id=image_id)

