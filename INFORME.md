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
