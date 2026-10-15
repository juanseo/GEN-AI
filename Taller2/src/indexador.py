"""Indexación: fragmentos → vectores → base vectorial (ChromaDB).

El proceso completo es:

1. `construir_chunks()` carga los tres documentos y los parte (src/chunking.py).
2. El modelo de embeddings convierte cada fragmento en un vector.
3. Chroma guarda vector + texto + metadata y construye un índice HNSW para
   búsqueda aproximada de vecinos más cercanos con distancia coseno.

Decisiones:

- **Una colección por modelo y por estrategia de chunking.** Los vectores de
  modelos distintos no son comparables entre sí; mezclarlos en una colección
  daría resultados sin sentido.
- **IDs estables** (`catalogo#ECO-BOT-500`, `faq#PAG-02`, `politica#s3-p0`).
  Chroma hace *upsert*, así que reindexar actualiza en lugar de duplicar.
- **Manifiesto con huella de las fuentes.** Si un documento de `data/` cambia,
  el índice se reconstruye solo la próxima vez que se abra. Así se cumple el
  requisito del Taller 1 de que un cambio de política esté vigente en minutos.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .chunking import construir_chunks
from .config import DATA_DIR, INDICE_DIR
from .embeddings import resolver_embeddings

FUENTES = ("politicas_devolucion.pdf", "catalogo_productos.csv", "faq.json")
MANIFIESTO = INDICE_DIR / "manifiesto.json"


def nombre_coleccion(embeddings: str, estrategia: str = "estructural") -> str:
    return f"ecomarket_{embeddings}_{estrategia}".replace(" ", "_")


def huella_fuentes() -> str:
    """Hash de los documentos fuente: cambia si cualquiera de ellos cambia."""
    sha = hashlib.sha256()
    for nombre in FUENTES:
        sha.update((DATA_DIR / nombre).read_bytes())
    return sha.hexdigest()[:16]


def _leer_manifiesto() -> dict:
    if MANIFIESTO.is_file():
        return json.loads(MANIFIESTO.read_text(encoding="utf-8"))
    return {}


def _guardar_manifiesto(coleccion: str, embeddings: str, estrategia: str, n_chunks: int) -> None:
    manifiesto = _leer_manifiesto()
    manifiesto[coleccion] = {
        "embeddings": embeddings,
        "estrategia": estrategia,
        "fragmentos": n_chunks,
        "huella_fuentes": huella_fuentes(),
        "indexado": datetime.now().isoformat(timespec="seconds"),
    }
    MANIFIESTO.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")


def indexar(vectorstore: Chroma, chunks: list[Document]) -> int:
    """Calcula los embeddings de los fragmentos y los carga en la colección."""
    ids = [chunk.metadata["id"] for chunk in chunks]
    vectorstore.add_documents(chunks, ids=ids)
    return len(chunks)


def abrir_indice(
    embeddings: str, estrategia: str = "estructural", reconstruir: bool = False
) -> tuple[Chroma, str]:
    """Abre (y si hace falta construye) el índice. Devuelve (vectorstore, modelo efectivo)."""
    nombre_efectivo, modelo = resolver_embeddings(embeddings)
    coleccion = nombre_coleccion(nombre_efectivo, estrategia)

    INDICE_DIR.mkdir(exist_ok=True)
    vectorstore = Chroma(
        collection_name=coleccion,
        embedding_function=modelo,
        client=chromadb.PersistentClient(path=str(INDICE_DIR)),
        # Distancia coseno: el score de relevancia queda como similitud coseno
        # (1 - distancia), comparable contra el umbral de config.py.
        collection_configuration={"hnsw": {"space": "cosine"}},
    )

    registro = _leer_manifiesto().get(coleccion, {})
    desactualizado = registro.get("huella_fuentes") != huella_fuentes()
    # TF-IDF aprende su vocabulario al indexar y no se guarda en disco: su
    # índice no se puede reabrir sin volver a ajustarlo.
    reconstruir = reconstruir or nombre_efectivo == "tfidf"
    if reconstruir or desactualizado or vectorstore._collection.count() == 0:
        vectorstore.reset_collection()
        chunks = construir_chunks(estrategia=estrategia)
        indexar(vectorstore, chunks)
        _guardar_manifiesto(coleccion, nombre_efectivo, estrategia, len(chunks))

    return vectorstore, nombre_efectivo
