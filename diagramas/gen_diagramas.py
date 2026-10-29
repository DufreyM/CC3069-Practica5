"""Genera los diagramas .drawio de la práctica.

Uso:
    python gen_diagramas.py                 # escribe los .drawio junto a este script
    python gen_diagramas.py --preview DIR   # además guarda una vista previa PNG aproximada en DIR

Las coordenadas están fijas a propósito: los ciclos alternan de lado para que ninguna flecha se cruce.
Las imágenes finales se exportan con draw.io Desktop (ver README).
"""
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr
from PIL import Image, ImageDraw, ImageFont

STYLES = {
    "term": "rounded=1;arcSize=50;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontStyle=1;",
    "io": "shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;size=20;fillColor=#fff2cc;strokeColor=#d6b656;",
    "proc": "rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;",
    "dec": "rhombus;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;",
    "note": "shape=note;whiteSpace=wrap;html=1;size=16;align=left;verticalAlign=top;spacingLeft=10;spacingTop=6;fillColor=#f5f5f5;strokeColor=#666666;fontColor=#333333;",
    "title": "text;html=1;align=center;verticalAlign=middle;fontSize=18;fontStyle=1;",
    "lane": "rounded=1;arcSize=2;whiteSpace=wrap;html=1;verticalAlign=top;fontStyle=1;fontSize=15;spacingTop=8;",
    "store": "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;size=10;fillColor=#e1d5e7;strokeColor=#9673a6;",
}
PORTS = {"t": (0.5, 0), "b": (0.5, 1), "l": (0, 0.5), "r": (1, 0.5)}


class Diagram:
    def __init__(self, name):
        self.name, self.nodes, self.edges = name, {}, []

    def node(self, nid, kind, label, cx, y, w, h, extra=""):
        self.nodes[nid] = dict(kind=kind, label=label, x=cx - w / 2, y=y, w=w, h=h, extra=extra)

    def edge(self, src, dst, ex="b", en="t", pts=(), label="", dashed=False):
        self.edges.append(dict(src=src, dst=dst, ex=ex, en=en, pts=list(pts), label=label, dashed=dashed))

    def port(self, nid, p):
        n = self.nodes[nid]
        fx, fy = PORTS[p]
        return n["x"] + fx * n["w"], n["y"] + fy * n["h"]

    def to_xml(self):
        out = ['<mxfile host="app.diagrams.net">', f'  <diagram name={quoteattr(self.name)}>',
               '    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
               'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1700" pageHeight="1800" math="0" shadow="0">',
               '      <root>', '        <mxCell id="0"/>', '        <mxCell id="1" parent="0"/>']
        for nid, n in self.nodes.items():
            out.append(f'        <mxCell id="{nid}" value={quoteattr(n["label"])} '
                       f'style={quoteattr(STYLES[n["kind"]] + n["extra"])} vertex="1" parent="1">')
            out.append(f'          <mxGeometry x="{n["x"]:g}" y="{n["y"]:g}" width="{n["w"]}" height="{n["h"]}" as="geometry"/>')
            out.append('        </mxCell>')
        for i, e in enumerate(self.edges):
            (exx, exy), (enx, eny) = PORTS[e["ex"]], PORTS[e["en"]]
            style = (f"edgeStyle=none;rounded=0;html=1;endArrow=block;endFill=1;fontStyle=1;labelBackgroundColor=#ffffff;"
                     f"exitX={exx};exitY={exy};exitDx=0;exitDy=0;entryX={enx};entryY={eny};entryDx=0;entryDy=0;")
            if e["dashed"]:
                style += "dashed=1;strokeColor=#9673a6;fontColor=#9673a6;"
            out.append(f'        <mxCell id="e{i}" value={quoteattr(e["label"])} style={quoteattr(style)} '
                       f'edge="1" parent="1" source="{e["src"]}" target="{e["dst"]}">')
            out.append('          <mxGeometry relative="1" as="geometry">')
            if e["pts"]:
                out.append('            <Array as="points">')
                out += [f'              <mxPoint x="{x}" y="{y}"/>' for x, y in e["pts"]]
                out.append('            </Array>')
            out.append('          </mxGeometry>')
            out.append('        </mxCell>')
        out += ['      </root>', '    </mxGraphModel>', '  </diagram>', '</mxfile>', '']
        return "\n".join(out)

    def preview(self, path):
        W = int(max(n["x"] + n["w"] for n in self.nodes.values()) + 40)
        H = int(max(n["y"] + n["h"] for n in self.nodes.values()) + 40)
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 12)
        for n in self.nodes.values():
            x, y, w, h = n["x"], n["y"], n["w"], n["h"]
            if n["kind"] == "dec":
                d.polygon([(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)], outline="orange")
            elif n["kind"] != "title":
                d.rectangle([x, y, x + w, y + h], outline="gray" if n["kind"] == "lane" else "blue")
            txt = n["label"].replace("<br>", "\n").replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
            if n["kind"] in ("note", "lane"):
                d.multiline_text((x + 8, y + 8), txt, fill="black", font=font)
            else:
                d.multiline_text((x + w / 2, y + h / 2), txt, fill="black", font=font, anchor="mm", align="center")
        for e in self.edges:
            pts = [self.port(e["src"], e["ex"]), *e["pts"], self.port(e["dst"], e["en"])]
            d.line(pts, fill="purple" if e["dashed"] else "red", width=2)
            ax, ay = pts[-1]
            d.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill="red")
            if e["label"]:
                mx, my = (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2
                d.text((mx, my), e["label"], fill="darkred", font=font, anchor="mm")
        img.save(path)


def secuencial():
    g = Diagram("Blur secuencial")
    C = 700
    g.node("title", "title", "Diagrama de flujo · Filtro de desenfoque 3×3 (versión secuencial en CPU)", C, 20, 900, 40)
    g.node("ini", "term", "Inicio", C, 100, 140, 50)
    g.node("leer", "io", "Entrada: imagen I<br>H × W × 3 (uint8)", C, 180, 260, 60)
    g.node("crear", "proc", "Crear matriz resultado R<br>H × W × 3 (ceros)", C, 270, 260, 60)
    g.node("y0", "proc", "y ← 0", C, 360, 140, 40)
    g.node("d1", "dec", "¿y &lt; H?", C, 430, 160, 80)
    g.node("x0", "proc", "x ← 0", C, 540, 140, 40)
    g.node("d2", "dec", "¿x &lt; W?", C, 610, 160, 80)
    g.node("c0", "proc", "c ← 0", C, 720, 140, 40)
    g.node("d3", "dec", "¿c &lt; 3?", C, 790, 160, 80)
    g.node("s0", "proc", "suma ← 0<br>dy ← −1", C, 900, 200, 50)
    g.node("d4", "dec", "¿dy ≤ 1?", C, 980, 160, 80)
    g.node("dx0", "proc", "dx ← −1", C, 1090, 140, 40)
    g.node("d5", "dec", "¿dx ≤ 1?", C, 1160, 160, 80)
    g.node("clamp", "proc", "yy ← min(max(y + dy, 0), H − 1)<br>xx ← min(max(x + dx, 0), W − 1)", C, 1270, 300, 70)
    g.node("sumar", "proc", "suma ← suma + I[yy, xx, c]", C, 1370, 260, 40)
    g.node("dx1", "proc", "dx ← dx + 1", C, 1440, 140, 40)
    g.node("salida", "io", "Salida: imagen R<br>(desenfocada)", C, 1530, 260, 60)
    g.node("fin", "term", "Fin", C, 1620, 140, 50)
    # incrementos de los ciclos externos (alternan lados para que no se crucen las flechas)
    g.node("y1", "proc", "y ← y + 1", 930, 630, 130, 40)
    g.node("x1", "proc", "x ← x + 1", 470, 810, 130, 40)
    g.node("wr", "proc", "R[y, x, c] ← redondear(suma / 9)<br>c ← c + 1", 990, 990, 240, 60)
    g.node("dy1", "proc", "dy ← dy + 1", 470, 1180, 130, 40)
    g.node("nota", "note",
           "<b>Notas</b><br><br>"
           "• Cada píxel nuevo es el promedio de 9 valores:<br>&nbsp;&nbsp;él mismo y sus 8 vecinos (ventana 3×3).<br><br>"
           "• <b>Bordes:</b> se replica el píxel de la orilla<br>&nbsp;&nbsp;(clamp de yy y xx), así siempre se divide entre 9.<br><br>"
           "• Se lee de I y se escribe en R; nunca se<br>&nbsp;&nbsp;sobrescribe la imagen de entrada.<br><br>"
           "• Imagen 8K: H = 4320, W = 7680<br>&nbsp;&nbsp;→ 33 177 600 píxeles × 3 canales × 9 lecturas<br>&nbsp;&nbsp;≈ 896 millones de sumas.",
           1340, 100, 380, 230)

    for a, b in [("ini", "leer"), ("leer", "crear"), ("crear", "y0"), ("y0", "d1"), ("x0", "d2"), ("c0", "d3"),
                 ("s0", "d4"), ("dx0", "d5"), ("clamp", "sumar"), ("sumar", "dx1"), ("salida", "fin")]:
        g.edge(a, b)
    for a, b in [("d1", "x0"), ("d2", "c0"), ("d3", "s0"), ("d4", "dx0"), ("d5", "clamp")]:
        g.edge(a, b, label="Sí")
    # ciclo dx
    g.edge("dx1", "d5", "r", "t", [(900, 1460), (900, 1145), (C, 1145)])
    # ciclo dy
    g.edge("d5", "dy1", "l", "r", label="No")
    g.edge("dy1", "d4", "t", "t", [(470, 965), (C, 965)])
    # ciclo c
    g.edge("d4", "wr", "r", "l", label="No")
    g.edge("wr", "d3", "t", "t", [(990, 775), (C, 775)])
    # ciclo x
    g.edge("d3", "x1", "l", "r", label="No")
    g.edge("x1", "d2", "t", "t", [(470, 595), (C, 595)])
    # ciclo y
    g.edge("d2", "y1", "r", "l", label="No")
    g.edge("y1", "d1", "t", "t", [(930, 415), (C, 415)])
    # fin del recorrido
    g.edge("d1", "salida", "l", "l", [(300, 470), (300, 1560)], label="No")
    return g


def cuda():
    g = Diagram("Blur CUDA")
    H, D, K = 320, 980, 1050
    g.node("title", "title", "Diagrama de flujo · Filtro de desenfoque 3×3 con CUDA (Host ↔ Device)", 680, 20, 1000, 40)
    g.node("laneH", "lane", "Host (CPU · RAM)", H, 80, 560, 1040, "fillColor=#f8f9fb;strokeColor=#6c8ebf;")
    g.node("laneD", "lane", "Device (GPU · VRAM)", D, 80, 640, 1040, "fillColor=#fbf8fc;strokeColor=#9673a6;")

    g.node("ini", "term", "Inicio", H, 130, 140, 50)
    g.node("leer", "io", "Entrada: imagen I<br>H × W × 3 (uint8)", H, 210, 260, 60)
    g.node("h2d", "proc", "d_I ← cuda.to_device(I)<br>cuda.synchronize()<br><i>medir t_H2D</i>", H, 300, 300, 70)
    g.node("alloc", "proc", "d_R ← cuda.device_array((H, W, 3), uint8)", H, 400, 300, 50)
    g.node("cfg", "proc", "bloque ← (16, 16)<br>grid ← (⌈W / 16⌉, ⌈H / 16⌉)", H, 480, 300, 60)
    g.node("launch", "proc", "blur_kernel[grid, bloque](d_I, d_R)", H, 570, 300, 50)
    g.node("sync", "proc", "cuda.synchronize()<br>(espera a que terminen todos los hilos)<br><i>medir t_kernel</i>", H, 650, 300, 70)
    g.node("d2h", "proc", "R ← d_R.copy_to_host()<br><i>medir t_D2H</i>", H, 750, 300, 60)
    g.node("speed", "proc", "t_CUDA ← t_H2D + t_kernel + t_D2H<br>Speedup ← t_CPU / t_CUDA", H, 840, 300, 70)
    g.node("out", "io", "Salida: imagen R y tiempos", H, 940, 260, 60)
    g.node("fin", "term", "Fin", H, 1030, 140, 50)

    g.node("vin", "store", "VRAM: d_I<br>(copia de la entrada)", D, 290, 220, 90)
    g.node("vout", "store", "VRAM: d_R<br>(resultado)", 770, 735, 140, 90)
    g.node("k0", "term", "blur_kernel: cada hilo", K, 570, 240, 50)
    g.node("kidx", "proc", "(x, y) ← cuda.grid(2)", K, 650, 240, 50)
    g.node("kdec", "dec", "¿x &lt; W<br>y y &lt; H?", K, 730, 200, 90)
    g.node("kno", "proc", "Hilo sobrante:<br>no hace nada", 1240, 750, 100, 50)
    g.node("kloop", "proc", "Para c = 0, 1, 2:<br>suma de la ventana 3×3 de d_I<br>(índices con clamp)", K, 860, 280, 70)
    g.node("kwr", "proc", "d_R[y, x, c] ← redondear(suma / 9)", K, 960, 280, 50)
    g.node("kend", "term", "Fin del hilo", K, 1040, 160, 50)
    g.node("nota", "note",
           "<b>Configuración para 8K (7680 × 4320)</b><br><br>"
           "• Bloque 16 × 16 = 256 hilos<br>"
           "• Grid = 480 × 270 = 129 600 bloques<br>"
           "• 33 177 600 hilos: uno por píxel<br><br>"
           "• El if del kernel descarta los hilos que<br>&nbsp;&nbsp;caen fuera de la imagen cuando W o H<br>&nbsp;&nbsp;no son múltiplos de 16.<br><br>"
           "• t_CPU viene de la versión secuencial.",
           1500, 100, 320, 210)

    for a, b in [("ini", "leer"), ("leer", "h2d"), ("h2d", "alloc"), ("alloc", "cfg"), ("cfg", "launch"),
                 ("launch", "sync"), ("sync", "d2h"), ("d2h", "speed"), ("speed", "out"), ("out", "fin"),
                 ("k0", "kidx"), ("kidx", "kdec"), ("kloop", "kwr"), ("kwr", "kend")]:
        g.edge(a, b)
    g.edge("kdec", "kloop", label="Sí")
    g.edge("kdec", "kno", "r", "l", label="No")
    g.edge("kno", "kend", "b", "r", [(1240, 1065)])
    g.edge("h2d", "vin", "r", "l", label="copia H→D", dashed=True)
    g.edge("launch", "k0", "r", "l", label="lanza grid × bloque hilos", dashed=True)
    g.edge("kwr", "vout", "l", "b", [(770, 985)], label="escribe", dashed=True)
    g.edge("vout", "d2h", "l", "r", label="copia D→H", dashed=True)
    return g


if __name__ == "__main__":
    outdir = Path(__file__).parent
    prevdir = Path(sys.argv[sys.argv.index("--preview") + 1]) if "--preview" in sys.argv else None
    for fname, g in [("blur_secuencial", secuencial()), ("blur_cuda", cuda())]:
        (outdir / f"{fname}.drawio").write_text(g.to_xml(), encoding="utf-8", newline="\n")
        if prevdir:
            g.preview(prevdir / f"{fname}_preview.png")
        print("ok", fname)
