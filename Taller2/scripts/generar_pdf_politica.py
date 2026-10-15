"""Genera `data/politicas_devolucion.pdf` a partir de su fuente editable en Markdown.

En EcoMarket la política oficial se publica como PDF (es el documento que firma
la Dirección de Experiencia al Cliente). Este script reproduce ese escenario a
partir de `data/fuentes/politicas_devolucion.md`, e incluye a propósito lo que
hace "sucio" a un PDF real: encabezado y pie de página repetidos en cada hoja,
tablas y secciones que quedan partidas entre páginas. El cargador
(`src/cargadores.py`) tiene que limpiar todo eso antes de partir el texto.

Solo hace falta ejecutarlo si cambia la fuente; el PDF ya está en el repositorio.

Uso:
    python -m scripts.generar_pdf_politica
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR  # noqa: E402

FUENTE = DATA_DIR / "fuentes" / "politicas_devolucion.md"
DESTINO = DATA_DIR / "politicas_devolucion.pdf"

ENCABEZADO = "EcoMarket S.A.S. - Política de Devoluciones, Cambios y Reembolsos - v3.2"
PIE = "Documento controlado. Uso interno y publicación en el centro de ayuda."

ESTILOS = getSampleStyleSheet()
CUERPO = ParagraphStyle("cuerpo", parent=ESTILOS["BodyText"], fontSize=10, leading=14)
VINETA = ParagraphStyle("vineta", parent=CUERPO, leftIndent=14, bulletIndent=4)
TITULO = ParagraphStyle("titulo", parent=ESTILOS["Title"], fontSize=16)
SECCION = ParagraphStyle("seccion", parent=ESTILOS["Heading2"], fontSize=12, spaceBefore=10)
META = ParagraphStyle("meta", parent=CUERPO, fontSize=9, textColor=colors.grey)


def _en_linea(texto: str) -> str:
    """Convierte el Markdown en línea (negrita, cursiva, código) a las etiquetas de ReportLab."""
    # Las fuentes estándar del PDF no traen la flecha; se usa un equivalente ASCII.
    texto = texto.replace("→", ">").replace("&", "&amp;").replace("<", "&lt;")
    texto = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", texto)
    texto = re.sub(r"\*(.+?)\*", r"<i>\1</i>", texto)
    texto = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", texto)
    return texto


def _tabla(filas: list[str]) -> Table:
    celdas = [
        [Paragraph(_en_linea(c.strip()), CUERPO) for c in fila.strip("|").split("|")]
        for fila in filas
        if not re.fullmatch(r"\|[\s\-|]+\|", fila.strip())
    ]
    tabla = Table(celdas, repeatRows=1, hAlign="LEFT")
    tabla.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0E8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return tabla


def construir_flowables(markdown: str) -> list:
    flowables: list = []
    lineas = markdown.splitlines()
    i = 0
    while i < len(lineas):
        linea = lineas[i].rstrip()
        if not linea or linea == "---":
            i += 1
            continue
        if linea.startswith("# "):
            flowables.append(Paragraph(_en_linea(linea[2:]), TITULO))
        elif linea.startswith("## "):
            flowables.append(Paragraph(_en_linea(linea[3:]), SECCION))
        elif linea.startswith("> "):
            flowables.append(Paragraph(_en_linea(linea[2:]), META))
        elif linea.startswith("|"):
            filas = []
            while i < len(lineas) and lineas[i].startswith("|"):
                filas.append(lineas[i])
                i += 1
            flowables += [_tabla(filas), Spacer(1, 6)]
            continue
        elif linea.lstrip().startswith("- ") or re.match(r"\s*\d+\. ", linea):
            # Ítems de lista; las líneas de continuación se pegan al ítem.
            item = linea.strip()
            while i + 1 < len(lineas) and lineas[i + 1].startswith("  ") and lineas[i + 1].strip():
                i += 1
                item += " " + lineas[i].strip()
            vineta = "•" if item.startswith("- ") else item.split(" ", 1)[0]
            flowables.append(Paragraph(_en_linea(item.split(" ", 1)[1]), VINETA, bulletText=vineta))
        else:
            # Párrafo: se unen las líneas consecutivas.
            parrafo = linea
            while i + 1 < len(lineas) and lineas[i + 1].strip() and not re.match(
                r"(#|>|\||-|\s*\d+\. )", lineas[i + 1].lstrip()
            ):
                i += 1
                parrafo += " " + lineas[i].strip()
            flowables.append(Paragraph(_en_linea(parrafo), CUERPO))
        i += 1
    return flowables


def _encabezado_y_pie(lienzo, documento) -> None:
    lienzo.saveState()
    lienzo.setFont("Helvetica", 8)
    lienzo.setFillColor(colors.grey)
    lienzo.drawString(documento.leftMargin, LETTER[1] - 1.2 * cm, ENCABEZADO)
    lienzo.drawString(documento.leftMargin, 1.2 * cm, PIE)
    lienzo.drawRightString(LETTER[0] - documento.rightMargin, 1.2 * cm, f"Página {documento.page}")
    lienzo.restoreState()


def main() -> int:
    documento = SimpleDocTemplate(
        str(DESTINO), pagesize=LETTER, topMargin=2 * cm, bottomMargin=2 * cm,
        title="Política de Devoluciones EcoMarket", author="EcoMarket S.A.S.",
    )
    flowables = construir_flowables(FUENTE.read_text(encoding="utf-8"))
    documento.build(flowables, onFirstPage=_encabezado_y_pie, onLaterPages=_encabezado_y_pie)
    print(f"PDF generado: {DESTINO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
