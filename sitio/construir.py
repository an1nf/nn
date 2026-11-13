import html
import re
import shutil
import sys
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else RAIZ / "_sitio"
AUTOR, ENLACE_AUTOR = "", ""

CABECERA = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titulo}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif:ital,wght@0,400;0,700;1,400&display=swap">
<link rel="stylesheet" href="{raiz}estilo.css">
<script>MathJax = {{options: {{ignoreHtmlClass: ".*", processHtmlClass: "arithmatex"}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
<main>
"""

PIE = """</main>
</body>
</html>
"""

MERMAID = """<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
mermaid.initialize({startOnLoad: false, securityLevel: "loose", theme: "neutral",
  themeVariables: {fontFamily: "Noto Serif, Georgia, serif", fontSize: "14px"},
  flowchart: {htmlLabels: false, wrappingWidth: 230}});
await mermaid.run();
</script>
"""


def convertir(texto):
    return markdown.markdown(texto, extensions=[
        "tables", "sane_lists", "codehilite", "pymdownx.arithmatex", "pymdownx.superfences",
    ], extension_configs={
        "codehilite": {"guess_lang": False},
        "pymdownx.arithmatex": {"generic": True},
        "pymdownx.superfences": {"custom_fences": [{
            "name": "mermaid", "class": "mermaid",
            "format": lambda fuente, *_, **__: f'<pre class="mermaid">{html.escape(fuente)}</pre>',
        }]},
    })


def ajustar_enlaces(cuerpo):
    # Directorios y README.md → index.html: el sitio funciona también sin servidor.
    def cambiar(m):
        destino = m.group(2)
        if re.match(r"(https?:|mailto:|#)", destino):
            return m.group(0)
        base, _, ancla = destino.partition("#")
        base = re.sub(r"(README\.md|/)$", lambda f: "index.html" if f.group(1) == "README.md" else "/index.html", base)
        return f'{m.group(1)}="{base}{"#" + ancla if ancla else ""}"'

    return re.sub(r'(href|src)="([^"]*)"', cambiar, cuerpo)


def secciones():
    resultado = []
    for carpeta in sorted(RAIZ.glob("capitulo-*/*/README.md")):
        numero, titulo = re.match(r"#\s+(\d+\.\d+)\s+(.+)", carpeta.read_text(encoding="utf-8")).groups()
        resultado.append({"carpeta": carpeta.parent, "numero": numero, "titulo": titulo,
                          "ruta": carpeta.parent.relative_to(RAIZ).as_posix()})
    return sorted(resultado, key=lambda s: [int(p) for p in s["numero"].split(".")])


def pagina_seccion(lista, i):
    s = lista[i]
    texto = "\n".join(l for l in (s["carpeta"] / "README.md").read_text(encoding="utf-8").splitlines()
                      if not l.startswith("[Índice]"))
    cuerpo = ajustar_enlaces(convertir(texto))
    for programa in sorted(s["carpeta"].glob("*.py")):
        fuente = programa.read_text(encoding="utf-8")
        cuerpo += (f'\n<h2>Código completo</h2>\n<p><a href="{programa.name}">{programa.name}</a> '
                   f'({len(fuente.splitlines())} líneas)</p>\n<details>\n<summary>Mostrar el código</summary>\n'
                   f'{highlight(fuente, PythonLexer(), HtmlFormatter(cssclass="codehilite"))}</details>\n')

    enlaces = ['<a href="../../index.html">Índice</a>']
    if i > 0:
        enlaces.append(f'<a href="../../{lista[i-1]["ruta"]}/index.html">Anterior: {lista[i-1]["numero"]} {lista[i-1]["titulo"]}</a>')
    if i + 1 < len(lista):
        enlaces.append(f'<a href="../../{lista[i+1]["ruta"]}/index.html">Siguiente: {lista[i+1]["numero"]} {lista[i+1]["titulo"]}</a>')
    nav = " · ".join(enlaces)

    destino = SALIDA / s["ruta"]
    shutil.copytree(s["carpeta"], destino, ignore=shutil.ignore_patterns("README.md", "__pycache__"))
    (destino / "index.html").write_text(
        CABECERA.format(titulo=f'{s["numero"]} {s["titulo"]}', raiz="../../")
        + f'<p class="navegacion">{nav}</p>\n{cuerpo}\n<p class="navegacion abajo">{nav}</p>\n' + PIE,
        encoding="utf-8")


def portada(lista):
    texto = (RAIZ / "README.md").read_text(encoding="utf-8")
    titulo = texto.splitlines()[0].lstrip("# ")
    texto = texto.split("\n## Resultados")[0]
    texto = re.sub(r"^### .*\n", "", texto, flags=re.M)
    texto = re.sub(r"^  \* ", "    * ", texto, flags=re.M)
    if AUTOR:
        autor = f"[{AUTOR}]({ENLACE_AUTOR})" if ENLACE_AUTOR else AUTOR
        texto = texto.replace("\n", f"\n\nAutor: {autor}\n", 1)

    # Nodos del mapa con número de sección → enlace a la sección.
    rutas = {s["numero"]: s["ruta"] for s in lista}

    def clics(m):
        nodos = dict(re.findall(r'(\w+)\["(\d+\.\d+)\s', m.group(1)))
        lineas = [f'    click {n} "{rutas[k]}/index.html"' for n, k in nodos.items() if k in rutas]
        return "```mermaid\n" + m.group(1) + "\n".join(lineas) + "\n```"

    texto = re.sub(r"```mermaid\n(.*?)```", clics, texto, flags=re.S)
    (SALIDA / "index.html").write_text(
        CABECERA.format(titulo=titulo, raiz="") + ajustar_enlaces(convertir(texto)) + MERMAID + PIE,
        encoding="utf-8")


def main():
    shutil.rmtree(SALIDA, ignore_errors=True)
    SALIDA.mkdir(parents=True)
    estilo = (Path(__file__).parent / "estilo.css").read_text(encoding="utf-8")
    (SALIDA / "estilo.css").write_text(estilo + "\n" + HtmlFormatter(style="friendly").get_style_defs(".codehilite"),
                                       encoding="utf-8")
    (SALIDA / ".nojekyll").touch()
    lista = secciones()
    for i in range(len(lista)):
        pagina_seccion(lista, i)
    portada(lista)
    print(f"{len(lista) + 1} páginas en {SALIDA}")


if __name__ == "__main__":
    main()
