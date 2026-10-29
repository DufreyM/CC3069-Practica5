"""Análisis de resultados: speedup, gráficas y comparación visual del antes/después.

Lee data/tiempos.json (generado por blur_secuencial.py y blur_cuda.py) y escribe en resultados/:
    tiempos.json               copia de los tiempos medidos
    tiempos_cpu_vs_cuda.png    CPU secuencial vs CUDA (escala logarítmica)
    desglose_cuda.png          en qué se va el tiempo de CUDA: copias vs kernel
    kernel_por_bloque.png      tiempo del kernel según el tamaño de bloque
    comparacion_<imagen>.png   imagen completa antes/después y un acercamiento con su mapa de diferencias

Uso:
    python analisis.py
"""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from PIL import Image

RAIZ = Path(__file__).parent
DATA, RESULTADOS = RAIZ / "data", RAIZ / "resultados"

# Acercamientos elegidos a mano: (fila, columna, lado, descripción). Las demás imágenes usan la zona con más bordes.
ACERCAMIENTOS = {"gallina_8k": (3802, 3062, 256, "ojo y pico de la gallina")}

# Paleta categórica validada (orden fijo) y tokens de texto/superficie
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SUPERFICIE, TEXTO, TEXTO_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "text.color": TEXTO, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def formato_tiempo(s):
    if s >= 1:
        return f"{s:,.1f} s"
    return f"{s * 1000:,.1f} ms" if s >= 1e-3 else f"{s * 1e6:,.0f} µs"


def grafica_cpu_vs_cuda(nombre, t):
    cpu, cuda = t["cpu"]["segundos"], t["cuda"]["mediana"]
    filas = [("CPU secuencial", cpu), ("CUDA total\n(copias + kernel)", cuda["total"]), ("CUDA solo kernel", cuda["kernel"])]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(filas))[::-1]
    ax.barh(y, [v for _, v in filas], color=AZUL, height=0.5)
    ax.set_xscale("log")
    ax.set_yticks(y, [f for f, _ in filas])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: formato_tiempo(v)))
    for yi, (_, v) in zip(y, filas):
        ax.text(v * 1.15, yi, formato_tiempo(v), va="center", color=TEXTO)
    ax.set_xlim(min(v for _, v in filas) / 3, cpu * 12)
    ax.set_title(f"Blur 3×3 sobre 8K ({nombre}): speedup total {cpu / cuda['total']:,.0f}×", loc="left")
    ax.set_xlabel("Tiempo (escala logarítmica)")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "tiempos_cpu_vs_cuda.png", dpi=200)
    plt.close(fig)


def grafica_desglose(nombre, t):
    m = t["cuda"]["mediana"]
    fases = [("Host → GPU", m["h2d"], AZUL), ("Kernel", m["kernel"], NARANJA), ("GPU → Host", m["d2h"], AQUA)]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    izquierda = 0
    for etiqueta, v, color in fases:
        ms = v * 1000
        ax.barh(0, ms, left=izquierda, color=color, height=0.5, edgecolor=SUPERFICIE, linewidth=2, label=etiqueta)
        if ms / (m["total"] * 1000) > 0.08:
            ax.text(izquierda + ms / 2, 0, f"{etiqueta}\n{ms:.1f} ms ({v / m['total']:.0%})",
                    ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        izquierda += ms
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, izquierda * 1.01)
    ax.set_xlabel("Tiempo (ms)")
    ax.set_title(f"Desglose del tiempo total de CUDA: {m['total'] * 1000:.1f} ms ({nombre})", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTADOS / "desglose_cuda.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def grafica_bloques(nombre, t):
    por_bloque = t["cuda"].get("kernel_por_bloque")
    if not por_bloque:
        return
    etiquetas = [f"{b.replace('x', ' × ')}\n({int(b.split('x')[0]) * int(b.split('x')[1])} hilos)" for b in por_bloque]
    valores = [v * 1000 for v in por_bloque.values()]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(valores))
    ax.bar(x, valores, color=AZUL, width=0.55)
    ax.set_xticks(x, etiquetas)
    ax.grid(axis="x", visible=False)
    mejor = int(np.argmin(valores))
    for xi, v in enumerate(valores):
        ax.text(xi, v + max(valores) * 0.02, f"{v:.2f} ms", ha="center", color=TEXTO,
                fontweight="bold" if xi == mejor else "normal")
    ax.set_ylim(0, max(valores) * 1.15)
    ax.set_ylabel("Tiempo del kernel (ms)")
    ax.set_title(f"Tiempo del kernel según el tamaño de bloque ({nombre})", loc="left")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "kernel_por_bloque.png", dpi=200)
    plt.close(fig)


def recorte_con_detalle(I, lado=64, escala=4):
    """Ubica la zona con más bordes para que el desenfoque sea visible.

    Los bordes se buscan en la imagen reducida `escala` veces, así cuentan las estructuras (costas,
    contornos) y no el ruido píxel a píxel.
    """
    H, W = I.shape[0] // lado * lado, I.shape[1] // lado * lado
    gris = I[:H, :W].mean(axis=2).reshape(H // escala, escala, W // escala, escala).mean(axis=(1, 3))
    bordes = np.zeros_like(gris)
    bordes[1:, :] += np.abs(np.diff(gris, axis=0))
    bordes[:, 1:] += np.abs(np.diff(gris, axis=1))
    c = lado // escala
    celdas = bordes.reshape(gris.shape[0] // c, c, gris.shape[1] // c, c).sum(axis=(1, 3))
    fy, fx = np.unravel_index(np.argmax(celdas), celdas.shape)
    return slice(fy * lado, (fy + 1) * lado), slice(fx * lado, (fx + 1) * lado)


def reducir(img, factor=8):
    alto, ancho, _ = img.shape
    return np.asarray(Image.fromarray(img).resize((ancho // factor, alto // factor), Image.LANCZOS))


def comparacion(nombre):
    """Imagen completa antes/después y, debajo, un acercamiento con su mapa de diferencias."""
    rutas = [DATA / f"{nombre}.npy", DATA / f"{nombre}_blur_cuda.npy"]
    if not all(r.exists() for r in rutas):
        return
    I, R = (np.load(r) for r in rutas)
    if nombre in ACERCAMIENTOS:
        fila, col, lado, descripcion = ACERCAMIENTOS[nombre]
        filas, cols = slice(fila, fila + lado), slice(col, col + lado)
    else:
        filas, cols = recorte_con_detalle(I)
        lado, descripcion = filas.stop - filas.start, "zona con más bordes"
    diferencia = np.abs(I[filas, cols].astype(np.int16) - R[filas, cols]).mean(axis=2)

    vertical = I.shape[0] > I.shape[1]
    fig = plt.figure(figsize=(13, 15.5) if vertical else (13, 9.5))
    gs = fig.add_gridspec(2, 6, height_ratios=(1.9, 1) if vertical else (1.2, 1))
    factor = 8
    for i, (img, titulo) in enumerate([(I, "Original completa"), (R, "Desenfocada completa (GPU)")]):
        ax = fig.add_subplot(gs[0, 3 * i:3 * i + 3])
        ax.imshow(reducir(img, factor))
        ax.add_patch(plt.Rectangle((cols.start / factor, filas.start / factor), lado / factor, lado / factor,
                                   fill=False, edgecolor=NARANJA, linewidth=2))
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")
    paneles = [(I[filas, cols], "Acercamiento: original", {}),
               (R[filas, cols], "Acercamiento: desenfocada", {}),
               (diferencia, f"|Original − Desenfocada| (máx. {diferencia.max():.0f})",
                {"cmap": "Blues", "vmin": 0, "vmax": max(1, np.percentile(diferencia, 99))})]
    for i, (img, titulo, opciones) in enumerate(paneles):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        ax.imshow(img, interpolation="nearest", **opciones)
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{nombre} ({I.shape[0]} × {I.shape[1]}): recuadro naranja = acercamiento de {lado} × {lado} px "
                 f"({descripcion}; filas {filas.start}–{filas.stop}, columnas {cols.start}–{cols.stop})",
                 fontsize=11, color=TEXTO_2)
    fig.tight_layout()
    fig.savefig(RESULTADOS / f"comparacion_{nombre}.png", dpi=150)
    plt.close(fig)


def main():
    RESULTADOS.mkdir(exist_ok=True)
    tiempos = json.loads((DATA / "tiempos.json").read_text(encoding="utf-8"))
    shutil.copy(DATA / "tiempos.json", RESULTADOS / "tiempos.json")

    print("| Imagen | t_CPU | t_H2D | t_kernel | t_D2H | t_CUDA total | Speedup total | Speedup solo kernel | ¿Igual a CPU? |")
    print("|---|---|---|---|---|---|---|---|---|")
    for nombre, t in tiempos.items():
        if "cpu" not in t or "cuda" not in t:
            continue
        cpu, m = t["cpu"]["segundos"], t["cuda"]["mediana"]
        print(f"| {nombre} | {cpu:.2f} s | {m['h2d'] * 1000:.2f} ms | {m['kernel'] * 1000:.2f} ms | "
              f"{m['d2h'] * 1000:.2f} ms | {m['total'] * 1000:.2f} ms | {cpu / m['total']:,.0f}× | "
              f"{cpu / m['kernel']:,.0f}× | {'Sí' if t['cuda']['igual_a_cpu'] else 'No'} |")
        comparacion(nombre)

    principal = "imagen_8k"
    grafica_cpu_vs_cuda(principal, tiempos[principal])
    grafica_desglose(principal, tiempos[principal])
    grafica_bloques(principal, tiempos[principal])
    print(f"\nGráficas guardadas en {RESULTADOS}")


if __name__ == "__main__":
    main()
