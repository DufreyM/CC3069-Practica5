"""Experimento: ¿por qué la copia GPU → Host tarda más que Host → GPU?

Compara las transferencias y el kernel según cómo se maneja la memoria:
    - copy_to_host() a un arreglo nuevo (lo que hace blur_cuda.py) vs. a uno ya reservado
    - memoria normal (pageable) vs. memoria fijada (pinned)
    - kernel escribiendo en un d_R recién reservado vs. en uno reutilizado

Uso:
    python experimento_memoria.py
"""
import json
import statistics
import time
from pathlib import Path

import numpy as np
from numba import cuda

from blur_cuda import blur_kernel, configurar_grid

RAIZ = Path(__file__).parent
BLOQUE = (16, 16)


def mediana_ms(funcion, repeticiones=10):
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion()
        tiempos.append(time.perf_counter() - inicio)
    return statistics.median(tiempos) * 1000


def main():
    I = np.load(RAIZ / "data" / "imagen_8k.npy")
    H, W, _ = I.shape
    grid = configurar_grid(H, W, BLOQUE)

    cuda.get_current_device()
    d_I = cuda.to_device(I)
    d_R = cuda.device_array_like(d_I)
    blur_kernel[grid, BLOQUE](d_I, d_R)  # compilación JIT fuera de la medición
    cuda.synchronize()

    destino = np.empty_like(I)
    fijada = cuda.pinned_array(I.shape, dtype=np.uint8)
    fijada[:] = I

    def kernel(salida):
        blur_kernel[grid, BLOQUE](d_I, salida)
        cuda.synchronize()

    def kernel_d_r_nuevo():
        salida = cuda.device_array_like(d_I)  # la reserva queda fuera de la medición
        inicio = time.perf_counter()
        kernel(salida)
        return time.perf_counter() - inicio

    resultados = {
        "h2d_pageable_ms": mediana_ms(lambda: (cuda.to_device(I), cuda.synchronize())),
        "h2d_pinned_ms": mediana_ms(lambda: (cuda.to_device(fijada), cuda.synchronize())),
        "d2h_arreglo_nuevo_ms": mediana_ms(lambda: d_R.copy_to_host()),
        "d2h_arreglo_reusado_ms": mediana_ms(lambda: d_R.copy_to_host(destino)),
        "d2h_pinned_ms": mediana_ms(lambda: d_R.copy_to_host(fijada)),
        "kernel_d_r_reusado_ms": mediana_ms(lambda: kernel(d_R)),
        "kernel_d_r_nuevo_ms": statistics.median(kernel_d_r_nuevo() for _ in range(10)) * 1000,
    }
    for clave, valor in resultados.items():
        print(f"{clave:<26}{valor:8.2f} ms")

    (RAIZ / "resultados").mkdir(exist_ok=True)
    (RAIZ / "resultados" / "experimento_memoria.json").write_text(json.dumps(resultados, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
