"""Segmentación (*chunking*) de la base de conocimiento.

La estrategia que usa el asistente es la **estructural**: cada tipo de documento
se parte por su unidad natural de significado.

- **Política (PDF)**: por sección numerada. Una sección es una regla completa
  ("3. Categorías NO sujetas a devolución" incluye su tabla y su excepción).
  Si una sección supera `TAMANO_MAXIMO`, se sub-divide con un splitter
  recursivo (párrafo → línea → oración → palabra) con solape. Cada fragmento
  lleva un encabezado contextual con el nombre de su sección.
- **Catálogo (CSV)**: un fragmento por producto. Partir una ficha mezclaría el
  precio de un SKU con la política de otro.
- **FAQ (JSON)**: un fragmento por pregunta. La pregunta ya está redactada como
  la escribiría un cliente, que es lo que mejor se parece a una consulta.

Las estrategias `parrafos` y `fijo` existen solo para el experimento de
`scripts/evaluar_retrieval.py`: tratan cada documento como un texto plano, que
es lo que haría un pipeline genérico que no mira la estructura.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from .cargadores import cargar_documentos

ESTRATEGIAS = ("estructural", "parrafos", "fijo")

# Con 1.200 caracteres (~300 tokens) todas las secciones de la política caben
# enteras (la más larga, la 3, tiene ~1.180). El límite real lo pone el modelo
# de embeddings: bge-m3 acepta 8.192 tokens, pero un modelo con ventana de 128
# tokens truncaría el fragmento y su vector solo representaría el comienzo.
TAMANO_MAXIMO = 1200
SOLAPE = 150

TAMANO_FIJO = 500
TAMANO_PARRAFOS = 400

_TITULO_POLITICA = "Política de devoluciones de EcoMarket"
_LINEA_NUMERADA = re.compile(r"^(\d+)\. (.+)$")


def _es_titulo_de_seccion(linea: str, siguiente: int) -> bool:
    """Distingue "3. Categorías NO sujetas..." de un paso "3. El sistema valida...".

    Un título lleva el número de sección que sigue, es corto y no termina en
    punto. Los pasos del procedimiento (sección 8) reinician la numeración en 1
    y son oraciones completas, así que no cumplen la condición.
    """
    coincidencia = _LINEA_NUMERADA.match(linea)
    return bool(
        coincidencia
        and int(coincidencia.group(1)) == siguiente
        and len(linea) < 80
        and not linea.endswith(".")
    )


def _secciones_de_politica(texto: str) -> list[tuple[str, str]]:
    """Devuelve [(título, cuerpo)] de la política, en orden."""
    secciones: list[tuple[str, list[str]]] = [("Encabezado", [])]
    for linea in texto.splitlines():
        if _es_titulo_de_seccion(linea, siguiente=len(secciones)):
            secciones.append((linea, []))
        secciones[-1][1].append(linea)
    return [(titulo, "\n".join(cuerpo)) for titulo, cuerpo in secciones if cuerpo]


def partir_politica(
    politica: Document, tamano: int = TAMANO_MAXIMO, solape: int = SOLAPE
) -> list[Document]:
    """Parte la política por secciones; sub-divide solo las que exceden `tamano`."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamano, chunk_overlap=solape, separators=["\n\n", "\n", ". ", " ", ""]
    )
    fragmentos: list[Document] = []
    for numero, (titulo, cuerpo) in enumerate(_secciones_de_politica(politica.page_content)):
        partes = [cuerpo] if len(cuerpo) <= tamano else splitter.split_text(cuerpo)
        for indice, parte in enumerate(partes):
            fragmentos.append(
                Document(
                    # El encabezado contextual le dice al embedding y al LLM de
                    # dónde viene el texto, aun si es la segunda mitad de una
                    # sección.
                    page_content=f"{_TITULO_POLITICA} > {titulo}\n{parte}",
                    metadata={
                        **politica.metadata,
                        "seccion": titulo,
                        "id": f"politica#s{numero}-p{indice}",
                    },
                )
            )
    return fragmentos


def _estructural(documentos: list[Document]) -> list[Document]:
    chunks: list[Document] = []
    for documento in documentos:
        tipo = documento.metadata["tipo"]
        if tipo == "politica":
            chunks.extend(partir_politica(documento))
        elif tipo == "producto":
            chunks.append(_con_id(documento, f"catalogo#{documento.metadata['sku']}"))
        elif tipo == "faq":
            chunks.append(_con_id(documento, f"faq#{documento.metadata['id']}"))
    return chunks


def _como_texto_plano(documentos: list[Document]) -> list[Document]:
    """Concatena los registros de cada fuente: lo que vería un pipeline genérico."""
    por_fuente: dict[str, list[Document]] = {}
    for documento in documentos:
        por_fuente.setdefault(documento.metadata["fuente"], []).append(documento)
    return [
        Document(
            page_content="\n\n".join(d.page_content for d in grupo),
            metadata={"fuente": fuente, "tipo": grupo[0].metadata["tipo"]},
        )
        for fuente, grupo in por_fuente.items()
    ]


def _con_splitter(documentos: list[Document], splitter, nombre: str) -> list[Document]:
    chunks: list[Document] = []
    for documento in _como_texto_plano(documentos):
        stem = documento.metadata["fuente"].split(".")[0]
        for indice, trozo in enumerate(splitter.split_documents([documento])):
            chunks.append(_con_id(trozo, f"{nombre}:{stem}#{indice:03d}"))
    return chunks


def _con_id(documento: Document, identificador: str) -> Document:
    return Document(page_content=documento.page_content, metadata={**documento.metadata, "id": identificador})


def construir_chunks(
    documentos: list[Document] | None = None, estrategia: str = "estructural"
) -> list[Document]:
    """Punto de entrada: carga (si hace falta) y parte la base de conocimiento."""
    if estrategia not in ESTRATEGIAS:
        raise ValueError(f"Estrategia {estrategia!r} desconocida; opciones: {ESTRATEGIAS}")
    documentos = documentos if documentos is not None else cargar_documentos()

    if estrategia == "estructural":
        return _estructural(documentos)
    if estrategia == "parrafos":
        # Agrupa líneas completas hasta ~400 caracteres. En un PDF no hay líneas
        # en blanco entre párrafos, así que "párrafo" termina siendo "bloque de
        # líneas": la estructura de secciones se pierde.
        splitter = CharacterTextSplitter(separator="\n", chunk_size=TAMANO_PARRAFOS, chunk_overlap=0)
        return _con_splitter(documentos, splitter, "parrafos")
    # "fijo": corta cada 500 caracteres sin mirar el contenido, aunque parta una
    # palabra o una regla a la mitad.
    splitter = CharacterTextSplitter(separator="", chunk_size=TAMANO_FIJO, chunk_overlap=0)
    return _con_splitter(documentos, splitter, "fijo")
