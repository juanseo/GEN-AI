"""Carga de los tres tipos de documentos de la base de conocimiento.

Cada formato llega con sus propios problemas, y el cargador los resuelve antes
de que el texto llegue al *chunking*:

| Documento | Formato | Problema | Qué hace el cargador |
|---|---|---|---|
| Política de devoluciones | PDF | encabezado y pie repetidos en cada hoja; secciones partidas entre páginas | quita el ruido y une las páginas en un solo texto |
| Catálogo de productos | CSV | una fila es un registro, no un texto | convierte cada fila en una ficha legible y deja los campos como metadata |
| Preguntas frecuentes | JSON | estructura anidada | un documento por par pregunta-respuesta |

Los pedidos (`pedidos.json`) no se cargan aquí: son datos transaccionales con PII
y se consultan por *function calling* (ver `src/herramientas.py`).

Se usan `pypdf` y la librería estándar en lugar de los cargadores de
`langchain-community`, que está en proceso de retiro; el resultado es el mismo
objeto `Document` de LangChain.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from .config import DATA_DIR

RUTA_POLITICA = DATA_DIR / "politicas_devolucion.pdf"
RUTA_CATALOGO = DATA_DIR / "catalogo_productos.csv"
RUTA_FAQ = DATA_DIR / "faq.json"

# Líneas que el PDF repite en cada página y que no aportan significado. Si se
# indexaran, aparecerían en todos los fragmentos y los harían más parecidos
# entre sí, lo que degrada la búsqueda por similitud.
_RUIDO_PDF = [
    re.compile(r"^EcoMarket S\.A\.S\. - Política de Devoluciones.*$"),
    re.compile(r"^Documento controlado\..*$"),
    re.compile(r"^Página \d+$"),
]

# Traducción de los códigos internos de política a lenguaje natural. Se agrega a
# cada ficha de producto para que el fragmento sea autosuficiente: el código
# `NO_DEV_PERECEDERO` por sí solo no se parece semánticamente a "¿me la reciben
# de vuelta?", pero "no admite devolución" sí.
RESUMEN_POLITICAS = {
    "DEV_ESTANDAR": "devolución hasta 30 días después de la entrega",
    "DEV_CONDICIONAL_SELLO": "devolución solo con el sello de seguridad intacto",
    "DEV_TEXTIL": "un cambio de talla gratuito por pedido dentro de los 30 días",
    "DEV_GARANTIA_12M": "30 días por retracto y 12 meses de garantía por defectos de fabricación",
    "NO_DEV_HIGIENE": "no admite devolución por retracto (producto de higiene personal)",
    "NO_DEV_PERECEDERO": "no admite devolución por retracto (alimento perecedero)",
    "NO_DEV_VIVO": "no admite devolución (material vegetal vivo)",
    "NO_DEV_PERSONALIZADO": "no admite devolución (producto personalizado)",
    "NO_DEV_DIGITAL": "no admite devolución (bien digital de consumo inmediato)",
}


def _formatear_cop(valor: str) -> str:
    return "COP " + f"{int(valor):,}".replace(",", ".")


def limpiar_texto_pdf(paginas: list[str]) -> str:
    """Quita encabezados y pies de página y une las páginas en un solo texto."""
    lineas: list[str] = []
    for pagina in paginas:
        for linea in pagina.splitlines():
            linea = linea.strip().lstrip("•").strip()
            if not linea or any(patron.match(linea) for patron in _RUIDO_PDF):
                continue
            lineas.append(linea)
    return "\n".join(lineas)


def cargar_politica_pdf(ruta: Path = RUTA_POLITICA) -> Document:
    """Lee la política en PDF y la devuelve como un único documento limpio.

    Se unen las páginas porque la unidad de significado es la sección, no la
    hoja: la sección 3 empieza en la página 1 y su excepción obligatoria está en
    la página 2. Partir por página separaría la regla de su excepción.
    """
    lector = PdfReader(str(ruta))
    paginas = [pagina.extract_text() or "" for pagina in lector.pages]
    return Document(
        page_content=limpiar_texto_pdf(paginas),
        metadata={
            "fuente": ruta.name,
            "tipo": "politica",
            "paginas": len(paginas),
            "version": "3.2",
        },
    )


def cargar_catalogo_csv(ruta: Path = RUTA_CATALOGO) -> list[Document]:
    """Convierte cada fila del catálogo en una ficha de producto legible."""
    documentos: list[Document] = []
    with open(ruta, encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            politica = fila["politica_devolucion"]
            texto = (
                f"Producto: {fila['nombre']} (SKU {fila['sku']})\n"
                f"Categoría: {fila['categoria']}\n"
                f"Precio: {_formatear_cop(fila['precio_cop'])}\n"
                f"Política de devolución: {politica}, "
                f"{RESUMEN_POLITICAS.get(politica, 'ver política de devoluciones')}\n"
                f"Disponibilidad: {fila['disponibilidad']}\n"
                f"Descripción: {fila['descripcion']}"
            )
            documentos.append(
                Document(
                    page_content=texto,
                    metadata={
                        "fuente": ruta.name,
                        "tipo": "producto",
                        "sku": fila["sku"],
                        "categoria": fila["categoria"],
                        "politica": politica,
                        "precio_cop": int(fila["precio_cop"]),
                    },
                )
            )
    return documentos


def cargar_faq_json(ruta: Path = RUTA_FAQ) -> list[Document]:
    """Un documento por par pregunta-respuesta del centro de ayuda."""
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return [
        Document(
            page_content=(
                f"Pregunta frecuente ({item['categoria']}): {item['pregunta']}\n"
                f"Respuesta: {item['respuesta']}"
            ),
            metadata={
                "fuente": ruta.name,
                "tipo": "faq",
                "id": item["id"],
                "categoria": item["categoria"],
                "actualizado": item["actualizado"],
            },
        )
        for item in datos["preguntas"]
    ]


def cargar_documentos() -> list[Document]:
    """Carga la base de conocimiento completa: política + catálogo + FAQ."""
    return [cargar_politica_pdf(), *cargar_catalogo_csv(), *cargar_faq_json()]
