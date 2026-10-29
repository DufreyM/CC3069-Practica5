# CC3069 · Práctica Guiada 5: Paralelizando algoritmos con CUDA

Algoritmo elegido: **Filtro de desenfoque (blur) 3×3** sobre una imagen 8K (7680 × 4320, RGB).
Cada píxel del resultado es el promedio de sí mismo y sus 8 vecinos inmediatos; en los bordes se
replica el píxel de la orilla (clamp).

## Estructura

| Carpeta / archivo | Contenido |
|---|---|
| `diagramas/` | Diagramas de flujo en formato draw.io (`.drawio`) |
| `requirements.txt` | Dependencias de Python |

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

- `diagramas/blur_secuencial.drawio`: algoritmo secuencial en CPU (sección III).
