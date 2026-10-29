"""Genera la imagen de entrada de 8K (7680 × 4320, RGB) para la práctica.

Por defecto crea una imagen sintética con aspecto de foto satelital (agua, costa, vegetación,
montaña y nieve) a partir de ruido fractal con semilla fija, así todos obtienen la misma imagen.
Con --desde se puede usar una foto real, que se reescala a 8K.

Uso:
    python generar_imagen.py
    python generar_imagen.py --desde imagenes/gallina.png --salida data/gallina_8k.npy
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ANCHO, ALTO = 7680, 4320
DATA = Path(__file__).parent / "data"

# Paleta de "alturas" → color: (altura normalizada, R, G, B)
PALETA = np.array([
    (0.00, 8, 24, 60),      # mar profundo
    (0.42, 28, 78, 128),    # agua poco profunda
    (0.47, 194, 178, 128),  # arena / costa
    (0.52, 86, 125, 58),    # pasto
    (0.66, 38, 82, 36),     # bosque
    (0.80, 112, 100, 86),   # roca
    (0.90, 150, 140, 128),  # montaña alta
    (0.96, 240, 240, 240),  # nieve
], dtype=np.float32)


def ruido_fractal(alto, ancho, rng, octavas=8):
    """Suma capas de ruido aleatorio reescalado (cada capa con el doble de detalle y la mitad de peso)."""
    total = np.zeros((alto, ancho), dtype=np.float32)
    amplitud, peso_total = 1.0, 0.0
    for o in range(octavas):
        columnas = 2 ** (o + 2)
        filas = max(2, columnas * alto // ancho)
        base = rng.random((filas, columnas), dtype=np.float32)
        capa = Image.fromarray(base).resize((ancho, alto), Image.BICUBIC)
        total += amplitud * np.asarray(capa)
        peso_total += amplitud
        amplitud *= 0.5
    return total / peso_total


def imagen_sintetica(semilla=3069):
    rng = np.random.default_rng(semilla)
    altura = ruido_fractal(ALTO, ANCHO, rng)
    bajo, alto = np.percentile(altura, [1, 99])
    altura = np.clip((altura - bajo) / (alto - bajo), 0, 1)

    rgb = np.empty((ALTO, ANCHO, 3), dtype=np.float32)
    for c in range(3):
        rgb[..., c] = np.interp(altura, PALETA[:, 0], PALETA[:, c + 1])
    # Textura fina píxel a píxel para que el desenfoque sea visible
    rgb += rng.normal(0, 10, size=(ALTO, ANCHO, 1)).astype(np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def imagen_desde_archivo(ruta):
    # Las imágenes satelitales suelen pasar del límite anti "decompression bomb" de Pillow (~178 MP)
    Image.MAX_IMAGE_PIXELS = None
    foto = Image.open(ruta).convert("RGB")
    w, h = foto.size
    # Una foto vertical se lleva a 4320 × 7680: misma cantidad de píxeles que 8K horizontal
    ancho, alto = (ALTO, ANCHO) if h > w else (ANCHO, ALTO)
    # Recorte centrado a la proporción destino para no deformar la imagen
    if w * alto > h * ancho:
        nuevo_w = h * ancho // alto
        foto = foto.crop(((w - nuevo_w) // 2, 0, (w + nuevo_w) // 2, h))
    else:
        nuevo_h = w * alto // ancho
        foto = foto.crop((0, (h - nuevo_h) // 2, w, (h + nuevo_h) // 2))
    if foto.size[0] < ancho:
        print(f"Aviso: la foto ({w} × {h}) es más pequeña que 8K; se va a ampliar {ancho / foto.size[0]:.1f} veces.")
    return np.asarray(foto.resize((ancho, alto), Image.LANCZOS), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--desde", type=Path, help="foto a usar en lugar de la imagen sintética")
    parser.add_argument("--salida", type=Path, default=DATA / "imagen_8k.npy")
    args = parser.parse_args()

    imagen = imagen_desde_archivo(args.desde) if args.desde else imagen_sintetica()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.salida, imagen)

    vista = args.salida.with_name(args.salida.stem + "_vista.png")
    alto, ancho, _ = imagen.shape
    Image.fromarray(imagen).resize((ancho // 4, alto // 4), Image.LANCZOS).save(vista)

    alto, ancho, canales = imagen.shape
    print(f"Imagen guardada en {args.salida}")
    print(f"Vista previa (1/4) en {vista}")
    print(f"  Dimensiones:      {alto} × {ancho} × {canales} (alto × ancho × canales)")
    print(f"  Píxeles:          {alto * ancho:,}")
    print(f"  Valores totales:  {imagen.size:,} ({imagen.dtype}, 0–255)")
    print(f"  Tamaño en memoria: {imagen.nbytes:,} bytes = {imagen.nbytes / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
