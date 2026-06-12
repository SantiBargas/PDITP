# Corre una evaluación más sobre el test ciego (sin reentrenar) y genera sus figuras.
# Cada corrida queda en resultados/clasificacion_yolov8n/evaluaciones/testN/.

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "codigo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

clasificacion = importlib.import_module("04_clasificacion")
figuras = importlib.import_module("figuras_informe")


def run() -> None:
    _, carpeta = clasificacion.evaluar()
    figuras.figura_matriz_confusion(out_dir=carpeta)
    figuras.figura_predicciones(out_dir=carpeta)
    print(f"\nResultados en {carpeta}")


if __name__ == "__main__":
    run()
