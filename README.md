# CC3069 · Práctica Guiada 5: Paralelizando algoritmos con CUDA

Algoritmo elegido: **Filtro de desenfoque (blur) 3×3** sobre una imagen 8K (7680 × 4320, RGB).
Cada píxel del resultado es el promedio de sí mismo y sus 8 vecinos inmediatos; en los bordes se
replica el píxel de la orilla (clamp).

## Estructura

| Carpeta / archivo | Contenido |
|---|---|
| `generar_imagen.py` | Genera la imagen de entrada 8K en `data/` (sintética o a partir de una foto) |
| `blur_secuencial.py` | Versión secuencial en CPU (referencia de tiempo) |
| `blur_cuda.py` | Versión paralela en GPU con Numba CUDA; verifica el resultado contra la CPU |
| `analisis.py` | Speedup, gráficas y comparación visual del antes/después |
| `experimento_memoria.py` | Experimento de transferencias: memoria nueva, reutilizada y pinned |
| `imagenes/` | Fotos originales para usar como entrada |
| `diagramas/` | Diagramas de flujo en formato draw.io (`.drawio`) |
| `resultados/` | Gráficas y tiempos medidos (sí se suben al repo) |
| `requirements.txt` | Dependencias de Python |
| `data/` | Imágenes y tiempos generados (no se sube al repo) |

## Cómo correr

```powershell
python generar_imagen.py              # imagen sintética 8K en data/imagen_8k.npy
python generar_imagen.py --desde imagenes/gallina.png --salida data/gallina_8k.npy
python blur_secuencial.py --prueba    # recorte de 270 × 480, menos de 1 s
python blur_secuencial.py             # imagen 8K completa, ~4 min
python blur_secuencial.py --entrada data/gallina_8k.npy
python blur_cuda.py --explorar-bloques   # 8K en GPU + comparación de tamaños de bloque
python blur_cuda.py --entrada data/gallina_8k.npy
python experimento_memoria.py
python analisis.py                    # tablas y gráficas en resultados/
```

Cada corrida guarda su resultado en `data/<imagen>_blur_cpu.npy` o `data/<imagen>_blur_cuda.npy` y sus tiempos en
`data/tiempos.json`, agrupados por imagen. `blur_cuda.py` compara su resultado con el de la CPU, así que hay que
correr primero `blur_secuencial.py` con la misma imagen.

### Datos de entrada

| | |
|---|---|
| Dimensiones | 4320 × 7680 × 3 (alto × ancho × canales RGB) |
| Cantidad de datos | 33 177 600 píxeles = 99 532 800 valores `uint8` |
| Tamaño en memoria | 99 532 800 bytes ≈ 94.92 MiB |

La imagen por defecto es sintética, con aspecto de foto satelital: ruido fractal con semilla fija, así todos generan la misma imagen.
Con `--desde` se puede usar una foto: se recorta al centro a 16:9 (o 9:16 si es vertical) y se lleva a 8K.
Una foto vertical queda en 7680 × 4320, con la misma cantidad de píxeles y bytes.

## Entorno

GPU usada: NVIDIA GeForce RTX 4060 Laptop (compute capability 8.9).

```powershell
python -m venv C:\ruta\a\cuda_env
C:\ruta\a\cuda_env\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "from numba import cuda; cuda.detect()"
```

`numpy` se fija en `<2.4` porque `numba-cuda` todavía usa `np.row_stack`, que fue eliminado en NumPy 2.4.

## Diagramas

Se abren con [app.diagrams.net](https://app.diagrams.net) o con la extensión *Draw.io Integration* de VS Code.

Los `.drawio` se generan con `python diagramas/gen_diagramas.py` (requiere `pillow` solo para la opción `--preview`).
Si editan un diagrama a mano en draw.io, ya no lo regeneren con el script o se perderán los cambios.

Para exportar de nuevo a PNG con draw.io Desktop:

```powershell
& "$env:LOCALAPPDATA\Programs\draw.io\draw.io.exe" --export --format png --scale 2 --border 20 --output diagramas/blur_secuencial.png diagramas/blur_secuencial.drawio
```

### Algoritmo secuencial en CPU (sección III)

Fuente editable: [`diagramas/blur_secuencial.drawio`](diagramas/blur_secuencial.drawio)

![Diagrama de flujo secuencial](diagramas/blur_secuencial.png)

### Flujo con CUDA: Host ↔ Device (sección IV)

Fuente editable: [`diagramas/blur_cuda.drawio`](diagramas/blur_cuda.drawio)

![Diagrama de flujo CUDA](diagramas/blur_cuda.png)

## Imágenes

- `imagenes/gallina.png`: foto de [Unsplash](https://unsplash.com) (800 × 1200, vertical), ampliada 6.4 veces a 4320 × 7680.
