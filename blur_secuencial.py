"""Versión secuencial (CPU) del filtro de desenfoque 3×3.

Cada píxel del resultado es el promedio de sí mismo y sus 8 vecinos, por canal. En los bordes se
replica el píxel de la orilla (clamp). Implementación con ciclos de Python, sin vectorizar ni
paralelizar: sigue el diagrama diagramas/blur_secuencial.drawio y sirve como tiempo de referencia
para calcular el speedup de la versión CUDA.

Uso:
    python blur_secuencial.py            # imagen 8K completa (tarda varios minutos)
    python blur_secuencial.py --prueba   # recorte de 270 × 480 para probar rápido
    python blur_secuencial.py --entrada data/gallina_8k.npy
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"


def blur_secuencial(I):
    H, W, C = I.shape
    R = np.zeros_like(I)
    paso_progreso = max(1, H // 10)
    for y in range(H):
        # Las 3 filas que toca la ventana (y-1, y, y+1 con clamp), convertidas a listas de Python:
        # indexar listas es mucho más rápido que indexar un arreglo de NumPy elemento por elemento.
        filas = [I[min(max(y + dy, 0), H - 1)].tolist() for dy in (-1, 0, 1)]
        fila_R = [[0] * C for _ in range(W)]
        for x in range(W):
            for c in range(C):
                suma = 0
                for dy in (-1, 0, 1):
                    fila = filas[dy + 1]
                    for dx in (-1, 0, 1):
                        xx = min(max(x + dx, 0), W - 1)
                        suma += fila[xx][c]
                fila_R[x][c] = (suma + 4) // 9  # suma / 9 redondeado al entero más cercano
        R[y] = fila_R
        if (y + 1) % paso_progreso == 0:
            print(f"  {100 * (y + 1) // H:3d} % ({y + 1}/{H} filas)", flush=True)
    return R


def guardar_tiempo(imagen, clave, datos):
    ruta = DATA / "tiempos.json"
    tiempos = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    tiempos.setdefault(imagen, {})[clave] = datos
    ruta.write_text(json.dumps(tiempos, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrada", type=Path, default=DATA / "imagen_8k.npy")
    parser.add_argument("--prueba", action="store_true", help="procesa solo un recorte de 270 × 480")
    args = parser.parse_args()

    I = np.load(args.entrada)
    if args.prueba:
        I = np.ascontiguousarray(I[:270, :480])
    H, W, C = I.shape
    print(f"Entrada: {H} × {W} × {C} ({I.nbytes / 1024**2:.2f} MiB)")

    inicio = time.perf_counter()
    R = blur_secuencial(I)
    t_cpu = time.perf_counter() - inicio

    sufijo = "_prueba" if args.prueba else ""
    nombre = args.entrada.stem
    salida = DATA / f"{nombre}_blur_cpu{sufijo}.npy"
    np.save(salida, R)
    guardar_tiempo(nombre, f"cpu{sufijo}", {"segundos": t_cpu, "alto": H, "ancho": W, "canales": C})

    print(f"Tiempo CPU (secuencial): {t_cpu:.2f} s")
    print(f"Resultado guardado en {salida}")


if __name__ == "__main__":
    main()
