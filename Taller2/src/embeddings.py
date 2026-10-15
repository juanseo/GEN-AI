"""Fábrica de modelos de embeddings.

Todos los modelos se exponen con la interfaz `Embeddings` de LangChain
(`embed_documents` para indexar, `embed_query` para buscar), así que la base
vectorial y la cadena no saben cuál se está usando. Cambiar de modelo es
cambiar `ECOMARKET_EMBEDDINGS`; lo único que no se puede mezclar es el índice:
cada modelo produce un espacio vectorial distinto, así que cada uno tiene su
propia colección en Chroma (ver `src/indexador.py`).

| Nombre | Dónde corre | Para qué |
|---|---|---|
| `bge-m3` | Ollama local | modelo open-source multilingüe (elegido) |
| `paraphrase-multilingual` | Ollama local | open-source, más pequeño: para comparar |
| `openai-3-small` | API de OpenAI | propietario: para comparar |
| `tfidf` | en memoria | línea base léxica del Taller 1 |
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import MODELOS_EMBEDDINGS, OLLAMA_LOCAL_URL, PROVEEDORES, cargar_api_key


class EmbeddingsNoDisponibles(RuntimeError):
    """El modelo de embeddings pedido no se puede usar en este equipo."""


class TfidfEmbeddings(Embeddings):
    """TF-IDF de n-gramas de caracteres, como en el Taller 1, con la interfaz de LangChain.

    No es un modelo semántico: compara fragmentos de palabras. Sirve como línea
    base para medir cuánto aporta un modelo de embeddings real. A diferencia de
    un modelo preentrenado, su vocabulario se aprende del corpus en
    `embed_documents`, así que hay que indexar antes de consultar.
    """

    def __init__(self) -> None:
        self._vectorizador = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._ajustado = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matriz = self._vectorizador.fit_transform(texts)
        self._ajustado = True
        return matriz.toarray().astype(np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        if not self._ajustado:
            raise RuntimeError("TfidfEmbeddings necesita indexar los documentos antes de consultar.")
        return self._vectorizador.transform([text]).toarray()[0].astype(np.float32).tolist()


def ollama_tiene_modelo(modelo: str) -> bool:
    """True si el servidor local de Ollama responde y ya descargó `modelo`."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_LOCAL_URL}/api/tags", timeout=3) as respuesta:
            modelos = json.load(respuesta).get("models", [])
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False
    return any(m.get("name", "").split(":")[0] == modelo for m in modelos)


def crear_embeddings(nombre: str) -> Embeddings:
    """Instancia el modelo `nombre` del catálogo `MODELOS_EMBEDDINGS`."""
    if nombre not in MODELOS_EMBEDDINGS:
        raise ValueError(f"Modelo de embeddings desconocido: {nombre!r}")
    info = MODELOS_EMBEDDINGS[nombre]

    if info["proveedor"] == "ollama_local":
        from langchain_ollama import OllamaEmbeddings

        if not ollama_tiene_modelo(info["modelo"]):
            raise EmbeddingsNoDisponibles(
                f"Ollama no está corriendo en {OLLAMA_LOCAL_URL} o falta el modelo. "
                f"Instala Ollama (https://ollama.com) y ejecuta: ollama pull {info['modelo']}"
            )
        return OllamaEmbeddings(model=info["modelo"], base_url=OLLAMA_LOCAL_URL)

    if info["proveedor"] == "openai":
        from langchain_openai import OpenAIEmbeddings

        api_key = cargar_api_key(PROVEEDORES["openai"]["api_key_names"])
        if not api_key:
            raise EmbeddingsNoDisponibles("Falta OPENAI_API_KEY en el entorno o en el archivo .env.")
        return OpenAIEmbeddings(model=info["modelo"], api_key=api_key)

    return TfidfEmbeddings()


def resolver_embeddings(nombre: str) -> tuple[str, Embeddings]:
    """Devuelve (nombre efectivo, modelo), con respaldo en OpenAI si el local no está.

    El respaldo cambia el espacio vectorial, por eso se devuelve el nombre
    efectivo: el índice y el umbral de relevancia deben ser los de ese modelo.
    """
    try:
        return nombre, crear_embeddings(nombre)
    except EmbeddingsNoDisponibles as error:
        if MODELOS_EMBEDDINGS[nombre]["proveedor"] != "ollama_local":
            raise
        try:
            respaldo = crear_embeddings("openai-3-small")
        except EmbeddingsNoDisponibles:
            raise error from None
        print(f"[aviso] {error}\n[aviso] Se usa el respaldo: openai-3-small.")
        return "openai-3-small", respaldo
