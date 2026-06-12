# Etapa 06 — Salida estructurada: especie + medidas por individuo

import json
from pathlib import Path

RESULTADOS_DIR = Path(__file__).resolve().parent.parent / "resultados"


def generar(resultados_rect: list, predicciones: list, medidas: list, group_id: int, image_id: int) -> list:
    """Combina la salida de las etapas 03 (recortes + datos del pez), 04 (especie
    predicha) y 05 (medidas en píxeles) en una lista de individuos, la imprime y
    la guarda como JSON en resultados/."""
    salida = []
    for (_, pez), pred, medida in zip(resultados_rect, predicciones, medidas):
        item = {
            "fish_id": pez["fish_id"],
            "especie_real": pez["especie"],
            "especie_predicha": pred["especie"],
            "confianza": round(pred["confianza"], 3),
            "largo_px": medida["largo_px"],
            "ancho_px": medida["ancho_px"],
        }
        salida.append(item)
        print(
            f"  pez {item['fish_id']:>4}: real={item['especie_real']:<14} "
            f"predicha={item['especie_predicha']:<14} (conf={item['confianza']}) "
            f"largo={item['largo_px']}px ancho={item['ancho_px']}px"
        )

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTADOS_DIR / f"salida_group_{group_id:02d}_{image_id:05d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"\nSalida guardada en {out_path}")

    return salida


def run() -> None:
    print("Etapa 06: usar generar(resultados_rect, predicciones, medidas, group_id, image_id).")


if __name__ == "__main__":
    run()
