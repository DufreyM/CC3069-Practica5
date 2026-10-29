# Práctica Guiada 5: Paralelizando algoritmos con CUDA

**CC3069 · Computación Paralela y Distribuida**
Integrantes: Leonardo Mejía (`DufreyM`), María José (`tismajo`), Isabella (`Isabella1911`)
Algoritmo elegido: **Filtro de desenfoque de imagen (blur) 3×3**

---

## II. Preparación del entorno de trabajo

**¿Les funcionó el set up del env local o debieron utilizar alguna alternativa? ¿Cuál utilizaron?**

Trabajamos en local, con una GPU propia, sin usar Google Colab. El equipo no tenía conda, así que en lugar de
`conda create` usamos un entorno virtual de Python (`venv`) con `pip`:

```powershell
python -m venv cuda_env
cuda_env\Scripts\Activate.ps1
pip install -r requirements.txt     # numpy<2.4, numba-cuda[cu12], pillow, matplotlib
python -c "from numba import cuda; cuda.detect()"
```

En vez de `cudatoolkit` instalamos `numba-cuda[cu12]`, como en la guía de Colab. Este paquete trae por pip las
librerías de CUDA que necesita (NVRTC, runtime, nvJitLink), así que no hace falta instalar el CUDA Toolkit
completo. `cuda.detect()` reconoció la GPU como `[SUPPORTED]`.

**Problema encontrado:** con la versión más reciente de NumPy (2.5), la primera compilación de un kernel fallaba
con `AttributeError: module 'numpy' has no attribute 'row_stack'`. `numba-cuda` 0.30.4 todavía usa esa función,
que NumPy eliminó en la versión 2.4. Lo resolvimos fijando `numpy<2.4` (quedó la 2.3.5).

| Componente | Versión |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB, compute capability 8.9, driver 581.42 (CUDA 13.0) |
| CPU | Intel Core i9-13900HX (24 núcleos / 32 hilos); la versión secuencial usa solo uno |
| Python | 3.14.5 |
| Librerías | numba 0.68.1, numba-cuda 0.30.4, numpy 2.3.5, pillow 13.0.0, matplotlib 3.11.2 |

---

## III. Identificación del algoritmo secuencial

### Diagrama de flujo

![Diagrama de flujo secuencial](diagramas/blur_secuencial.png)

Fuente editable: [`diagramas/blur_secuencial.drawio`](diagramas/blur_secuencial.drawio)

### Tabla del algoritmo

| | |
|---|---|
| **Nombre del algoritmo elegido** | Filtro de desenfoque de imagen (box blur 3×3) |
| **Justificación de la selección** | Es un problema SIMD claro: se aplica la misma operación (promediar una ventana de 3×3) a cada uno de los 33 millones de píxeles, y cada píxel del resultado se calcula sin depender de los demás. Eso permite asignar un hilo por píxel, sin sincronización entre hilos y sin condiciones de carrera, porque cada hilo escribe solo su propio píxel. Además, una imagen se mapea de forma natural a una grid 2D, lo que nos permitió practicar la configuración de bloques y grid en dos dimensiones. |
| **Datos de entrada** | Imagen `I` de 4320 × 7680 píxeles RGB: un arreglo de 4320 × 7680 × 3 valores `uint8` (0–255). |
| **Datos de salida** | Imagen `R` de las mismas dimensiones y tipo, donde cada valor es el promedio redondeado de la ventana 3×3 centrada en ese píxel, para ese canal. |
| **Operaciones y procesos** | Para cada píxel `(y, x)` y cada canal `c`: (1) recorrer los 9 vecinos `(y+dy, x+dx)` con `dy, dx ∈ {−1, 0, 1}`; (2) en los bordes, llevar el índice al rango válido con *clamp*: `min(max(i, 0), n−1)`, lo que equivale a replicar el píxel de la orilla; (3) sumar los 9 valores; (4) dividir entre 9 redondeando al entero más cercano con aritmética entera: `(suma + 4) // 9`; (5) escribir el resultado en `R`, sin sobrescribir `I`. |

**Decisiones de diseño:**
- **Bordes:** el enunciado no especifica qué hacer en los bordes. Elegimos replicar el píxel de la orilla porque así
  todos los píxeles se dividen entre 9 y todos los hilos de la GPU hacen exactamente el mismo trabajo, sin casos
  especiales.
- **Redondeo:** usamos `(suma + 4) // 9` en lugar de `round(suma / 9)`, porque `round` de Python redondea al par en
  los empates (*banker's rounding*). Con aritmética entera, la CPU y la GPU dan resultados idénticos bit a bit, y
  eso nos permitió verificar la versión CUDA comparando ambas imágenes con `np.array_equal`.

### Implementación secuencial

**Archivo:** [`blur_secuencial.py`](blur_secuencial.py). La imagen de entrada se genera con [`generar_imagen.py`](generar_imagen.py).

La implementación es propia, con ciclos de Python y sin vectorización ni paralelismo: no usa `scipy.ndimage`,
OpenCV ni las operaciones vectorizadas de NumPy para el filtro. Sigue el diagrama con cinco ciclos anidados
(`y`, `x`, `c`, `dy`, `dx`).

Una optimización que no cambia el algoritmo: en cada fila `y`, las 3 filas que toca la ventana se convierten a
listas de Python. Indexar listas es aproximadamente el doble de rápido que indexar un arreglo de NumPy elemento por
elemento: medimos unos 4 min contra unos 8 min estimados para 8K. Además, convertir solo 3 filas evita cargar la
imagen completa como listas anidadas, que ocuparía varios GB de RAM.

### Conjunto de datos

| | |
|---|---|
| **Cantidad de datos** | 33 177 600 píxeles × 3 canales = **99 532 800 valores** |
| **Dimensiones** | **4320 × 7680 × 3** (alto × ancho × canales RGB), resolución 8K UHD |
| **Tamaño total en memoria** | **99 532 800 bytes ≈ 94.92 MiB** (1 byte por valor) |
| **Trabajo** | 99 532 800 valores × 9 lecturas ≈ **896 millones de sumas** |

Usamos dos imágenes del mismo tamaño:
1. **`imagen_8k`** (principal): imagen sintética de 7680 × 4320 con aspecto de foto satelital (agua, costa,
   vegetación, montaña y nieve). Se genera con ruido fractal y una semilla fija, así cualquiera puede reproducirla
   ejecutando `python generar_imagen.py`.
2. **`gallina_8k`**: una foto real de [Unsplash](https://unsplash.com) (`imagenes/gallina.png`, 800 × 1200), ampliada a 4320 × 7680
   (vertical). Tiene la misma cantidad de píxeles y bytes que la principal. Sirve para comprobar que el tiempo no
   depende del contenido de la imagen.
