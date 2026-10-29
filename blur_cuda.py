"""Versión paralela (GPU) del filtro de desenfoque 3×3 con Numba CUDA.

Un hilo por píxel en una grid 2D: cada hilo calcula los 3 canales de su píxel con la misma ventana
3×3 y el mismo clamp de bordes que la versión secuencial. Mide por separado las tres fases del
diagrama diagramas/blur_cuda.drawio: copia Host→GPU, kernel y copia GPU→Host.

Uso:
    python blur_cuda.py                              # imagen 8K, bloques de 16 × 16
    python blur_cuda.py --entrada data/gallina_8k.npy
    python blur_cuda.py --prueba                     # recorte de 270 × 480
    python blur_cuda.py --explorar-bloques           # además compara tamaños de bloque
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
from numba import cuda

DATA = Path(__file__).parent / "data"
BLOQUES_A_EXPLORAR = [(8, 8), (16, 16), (32, 8), (32, 16), (32, 32)]


@cuda.jit
def blur_kernel(I, R):
    x, y = cuda.grid(2)  # índice global del hilo: x = columna, y = fila
    H, W, C = I.shape
    if x < W and y < H:  # los hilos sobrantes del último bloque no hacen nada
        for c in range(C):
            suma = 0
            for dy in range(-1, 2):
                yy = min(max(y + dy, 0), H - 1)
                for dx in range(-1, 2):
                    xx = min(max(x + dx, 0), W - 1)
                    suma += I[yy, xx, c]
            R[y, x, c] = (suma + 4) // 9  # suma / 9 redondeado, igual que en la CPU


def configurar_grid(H, W, bloque):
    """Bloques necesarios para cubrir la imagen: ⌈W / bloque_x⌉ × ⌈H / bloque_y⌉."""
    return (math.ceil(W / bloque[0]), math.ceil(H / bloque[1]))


def blur_cuda(I, bloque):
    """Una corrida completa. Regresa la imagen resultado y el tiempo de cada fase en segundos."""
    H, W, _ = I.shape

    inicio = time.perf_counter()
    d_I = cuda.to_device(I)
    cuda.synchronize()
    t_h2d = time.perf_counter() - inicio

    d_R = cuda.device_array_like(d_I)
    grid = configurar_grid(H, W, bloque)

    inicio = time.perf_counter()
    blur_kernel[grid, bloque](d_I, d_R)
    cuda.synchronize()  # el lanzamiento es asíncrono: hay que esperar a que terminen todos los hilos
    t_kernel = time.perf_counter() - inicio

    inicio = time.perf_counter()
    R = d_R.copy_to_host()
    t_d2h = time.perf_counter() - inicio

    return R, {"h2d": t_h2d, "kernel": t_kernel, "d2h": t_d2h, "total": t_h2d + t_kernel + t_d2h}


def tiempo_kernel(d_I, d_R, bloque, repeticiones):
    """Mediana del tiempo del kernel solo, con los datos ya en la GPU."""
    H, W, _ = d_I.shape
    grid = configurar_grid(H, W, bloque)
    blur_kernel[grid, bloque](d_I, d_R)  # calentamiento
    cuda.synchronize()
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        blur_kernel[grid, bloque](d_I, d_R)
        cuda.synchronize()
        tiempos.append(time.perf_counter() - inicio)
    return statistics.median(tiempos)


def guardar_tiempo(imagen, clave, datos):
    ruta = DATA / "tiempos.json"
    tiempos = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    tiempos.setdefault(imagen, {})[clave] = datos
    ruta.write_text(json.dumps(tiempos, indent=2, ensure_ascii=False), encoding="utf-8")


def ms(segundos):
    return f"{segundos * 1000:9.3f} ms"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrada", type=Path, default=DATA / "imagen_8k.npy")
    parser.add_argument("--prueba", action="store_true", help="procesa solo un recorte de 270 × 480")
    parser.add_argument("--bloque", type=int, nargs=2, default=(16, 16), metavar=("X", "Y"))
    parser.add_argument("--repeticiones", type=int, default=10)
    parser.add_argument("--explorar-bloques", action="store_true")
    args = parser.parse_args()

    I = np.load(args.entrada)
    if args.prueba:
        I = np.ascontiguousarray(I[:270, :480])
    H, W, C = I.shape
    bloque = tuple(args.bloque)
    grid = configurar_grid(H, W, bloque)
    hilos = grid[0] * grid[1] * bloque[0] * bloque[1]

    dispositivo = cuda.get_current_device()  # crea el contexto CUDA antes de medir
    print(f"GPU: {dispositivo.name.decode() if isinstance(dispositivo.name, bytes) else dispositivo.name}")
    print(f"Entrada: {H} × {W} × {C} ({I.nbytes / 1024**2:.2f} MiB)")
    print(f"Bloque {bloque[0]} × {bloque[1]} = {bloque[0] * bloque[1]} hilos | grid {grid[0]} × {grid[1]} = "
          f"{grid[0] * grid[1]:,} bloques | {hilos:,} hilos para {H * W:,} píxeles")

    # La primera corrida incluye la compilación JIT del kernel; se reporta aparte
    R, primera = blur_cuda(I, bloque)
    corridas = [blur_cuda(I, bloque)[1] for _ in range(args.repeticiones)]
    mediana = {fase: statistics.median(c[fase] for c in corridas) for fase in primera}

    sufijo = "_prueba" if args.prueba else ""
    nombre = args.entrada.stem
    ruta_cpu = DATA / f"{nombre}_blur_cpu{sufijo}.npy"
    correcto = bool(np.array_equal(R, np.load(ruta_cpu))) if ruta_cpu.exists() else None

    print(f"\n{'Fase':<22}{'1.ª corrida':>14}{f'mediana de {args.repeticiones}':>16}")
    for fase, etiqueta in [("h2d", "Host → GPU"), ("kernel", "Kernel"), ("d2h", "GPU → Host"), ("total", "Total CUDA")]:
        print(f"{etiqueta:<22}{ms(primera[fase]):>14}{ms(mediana[fase]):>16}")

    datos = {"bloque": list(bloque), "grid": list(grid), "hilos": hilos, "repeticiones": args.repeticiones,
             "primera": primera, "mediana": mediana, "igual_a_cpu": correcto}

    if args.explorar_bloques:
        d_I = cuda.to_device(I)
        d_R = cuda.device_array_like(d_I)
        datos["kernel_por_bloque"] = {}
        print("\nKernel según tamaño de bloque (mediana):")
        for b in BLOQUES_A_EXPLORAR:
            t = tiempo_kernel(d_I, d_R, b, args.repeticiones)
            datos["kernel_por_bloque"][f"{b[0]}x{b[1]}"] = t
            print(f"  {b[0]:>2} × {b[1]:<2} ({b[0] * b[1]:4d} hilos){ms(t):>14}")

    salida = DATA / f"{nombre}_blur_cuda{sufijo}.npy"
    np.save(salida, R)
    guardar_tiempo(nombre, f"cuda{sufijo}", datos)

    if correcto is None:
        print(f"\nNo se encontró {ruta_cpu.name}: corran blur_secuencial.py para verificar el resultado.")
    else:
        print(f"\n¿Resultado idéntico al de la CPU? {'Sí' if correcto else 'NO'}")
    cpu = json.loads((DATA / "tiempos.json").read_text(encoding="utf-8"))[nombre].get(f"cpu{sufijo}")
    if cpu:
        print(f"Speedup = t_CPU / t_CUDA = {cpu['segundos']:.3f} s / {mediana['total']:.4f} s "
              f"= {cpu['segundos'] / mediana['total']:,.1f}×")
    print(f"Resultado guardado en {salida}")


if __name__ == "__main__":
    main()
