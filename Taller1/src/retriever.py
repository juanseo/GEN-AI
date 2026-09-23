"""Capa de *retrieval* del sistema RAG.

Implementa, sobre los documentos de `data/`, las cuatro piezas que vimos en la
Clase 2:

1. **Chunking** de los documentos base (por sección de Markdown, con solape).
2. **Búsqueda vectorial** sobre embeddings (TF-IDF local por defecto, o
   embeddings de la API si `ECOMARKET_RETRIEVAL=openai`).
3. **Búsqueda híbrida**: combinación ponderada de la señal vectorial con una
   señal léxica tipo BM25, que rescata coincidencias exactas de términos raros
   (SKU, códigos de política, nombres propios) que el vector suaviza.
4. **Re-ranking**: una segunda pasada que reordena los candidatos antes de
   enviarlos al LLM, para gastar el *context window* solo en lo relevante.

Nota de alcance: en producción esta capa se reemplaza por una base vectorial
gestionada (pgvector, Pinecone, Azure AI Search) con índice HNSW y un
*cross-encoder* como re-ranker. Aquí se implementa en memoria para que el
taller sea ejecutable sin infraestructura, conservando la misma arquitectura.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import CONFIG, DATA_DIR

# Documentos no estructurados que se indexan. Los pedidos NO están aquí: son
# datos estructurados y con PII, y se consultan por function calling.
DOCUMENTOS_INDEXABLES = ("politicas_devolucion.md", "catalogo_productos.md", "faq_envios.md")

TAMANO_CHUNK = 900       # caracteres aproximados por fragmento
SOLAPE_CHUNK = 150       # solape para no partir una idea a la mitad

_PALABRAS_VACIAS = {
    "de", "la", "el", "los", "las", "un", "una", "y", "o", "que", "en", "a",
    "por", "para", "con", "del", "al", "se", "es", "mi", "me", "lo", "su",
    "puedo", "quiero", "como", "cual", "cuales", "si", "no", "the", "of",
}


def _normalizar(texto: str) -> list[str]:
    """Tokeniza en minúsculas, sin tildes y sin palabras vacías."""
    texto = texto.lower()
    for original, reemplazo in zip("áéíóúüñ", "aeiouun"):
        texto = texto.replace(original, reemplazo)
    tokens = re.findall(r"[a-z0-9\-]+", texto)
    return [t for t in tokens if t not in _PALABRAS_VACIAS and len(t) > 1]


@dataclass
class Fragmento:
    """Un chunk indexable, con la metadata que permite citar la fuente."""

    id: str
    documento: str
    seccion: str
    texto: str

    @property
    def cita(self) -> str:
        return f"{self.documento} › {self.seccion}"

    def como_contexto(self) -> str:
        return f"[{self.id}] Fuente: {self.cita}\n{self.texto}"


@dataclass
class Resultado:
    """Un fragmento recuperado con sus puntajes, para poder auditar el retrieval."""

    fragmento: Fragmento
    score_vectorial: float
    score_lexico: float
    score_hibrido: float
    score_rerank: float = 0.0


def _partir_seccion(texto: str) -> list[str]:
    """Parte una sección larga en trozos con solape, cortando en saltos de línea."""
    texto = texto.strip()
    if len(texto) <= TAMANO_CHUNK:
        return [texto] if texto else []

    trozos: list[str] = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + TAMANO_CHUNK, len(texto))
        if fin < len(texto):
            corte = texto.rfind("\n", inicio + TAMANO_CHUNK // 2, fin)
            if corte != -1:
                fin = corte
        trozos.append(texto[inicio:fin].strip())
        if fin >= len(texto):
            break
        inicio = max(fin - SOLAPE_CHUNK, inicio + 1)
    return [t for t in trozos if t]


def trocear_markdown(ruta: Path) -> list[Fragmento]:
    """Divide un Markdown por encabezados `##`, respetando el título como metadata."""
    contenido = ruta.read_text(encoding="utf-8")
    lineas = contenido.splitlines()

    secciones: list[tuple[str, list[str]]] = []
    seccion_actual = "Introducción"
    buffer: list[str] = []

    for linea in lineas:
        if linea.startswith("## "):
            if buffer:
                secciones.append((seccion_actual, buffer))
            seccion_actual = linea.lstrip("#").strip()
            buffer = []
        else:
            buffer.append(linea)
    if buffer:
        secciones.append((seccion_actual, buffer))

    fragmentos: list[Fragmento] = []
    for seccion, cuerpo in secciones:
        texto_seccion = "\n".join(cuerpo).strip()
        for indice, trozo in enumerate(_partir_seccion(texto_seccion)):
            # El título de la sección se repite dentro del chunk: mejora el
            # retrieval y le da al LLM el contexto de dónde viene el fragmento.
            texto = f"## {seccion}\n{trozo}"
            fragmentos.append(
                Fragmento(
                    id=f"{ruta.stem}#{len(fragmentos):02d}",
                    documento=ruta.name,
                    seccion=seccion,
                    texto=texto,
                )
            )
    return fragmentos


class BM25:
    """BM25 mínimo, para la mitad léxica de la búsqueda híbrida."""

    def __init__(self, corpus_tokenizado: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokenizado
        self.n_docs = len(corpus_tokenizado)
        self.long_docs = [len(d) for d in corpus_tokenizado]
        self.long_promedio = sum(self.long_docs) / max(self.n_docs, 1)
        self.frecuencias = [Counter(d) for d in corpus_tokenizado]

        apariciones: Counter[str] = Counter()
        for documento in corpus_tokenizado:
            apariciones.update(set(documento))
        self.idf = {
            termino: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for termino, n in apariciones.items()
        }

    def puntuar(self, consulta: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs)
        for i, frecuencia in enumerate(self.frecuencias):
            longitud = self.long_docs[i]
            total = 0.0
            for termino in consulta:
                if termino not in frecuencia:
                    continue
                f = frecuencia[termino]
                numerador = f * (self.k1 + 1)
                denominador = f + self.k1 * (
                    1 - self.b + self.b * longitud / max(self.long_promedio, 1e-9)
                )
                total += self.idf.get(termino, 0.0) * numerador / denominador
            scores[i] = total
        return scores


def _normalizar_scores(scores: np.ndarray) -> np.ndarray:
    """Lleva los puntajes a [0, 1] para poder sumarlos entre sí."""
    if scores.size == 0:
        return scores
    minimo, maximo = float(scores.min()), float(scores.max())
    if maximo - minimo < 1e-9:
        return np.zeros_like(scores)
    return (scores - minimo) / (maximo - minimo)


class BaseConocimiento:
    """Índice en memoria con búsqueda híbrida y re-ranking."""

    def __init__(self, cliente_llm=None, directorio: Path | None = None) -> None:
        directorio = directorio or DATA_DIR
        self.fragmentos: list[Fragmento] = []
        for nombre in DOCUMENTOS_INDEXABLES:
            ruta = directorio / nombre
            if ruta.is_file():
                self.fragmentos.extend(trocear_markdown(ruta))

        if not self.fragmentos:
            raise FileNotFoundError(f"No se encontraron documentos en {directorio}")

        textos = [f.texto for f in self.fragmentos]
        self._bm25 = BM25([_normalizar(t) for t in textos])

        self.motor = CONFIG.motor_retrieval
        if (
            self.motor == "openai"
            and CONFIG.proveedor == "openai"
            and cliente_llm is not None
        ):
            # Embeddings reales: la búsqueda vectorial captura sinónimos
            # ("me arrepentí" ~ "retracto") que TF-IDF no alcanza.
            self._matriz = np.array(cliente_llm.embeddings(textos))
            self._cliente = cliente_llm
            self._vectorizador = None
        else:
            # TF-IDF con n-gramas de caracteres: aproximación local y gratuita
            # a la búsqueda por similitud, suficiente para el taller.
            self.motor = "tfidf"
            self._vectorizador = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
            )
            self._matriz = self._vectorizador.fit_transform(textos)
            self._cliente = None

    def _scores_vectoriales(self, consulta: str) -> np.ndarray:
        if self._vectorizador is not None:
            vector = self._vectorizador.transform([consulta])
            return cosine_similarity(vector, self._matriz)[0]
        vector = np.array(self._cliente.embeddings([consulta]))
        return cosine_similarity(vector, self._matriz)[0]

    def _rerank(self, consulta: str, candidatos: list[Resultado]) -> list[Resultado]:
        """Segunda pasada de relevancia sobre los candidatos recuperados.

        Sustituye al cross-encoder de producción con tres señales baratas:
        cobertura de los términos de la consulta, coincidencia con el título de
        la sección y presencia de identificadores exactos (SKU, códigos).
        """
        tokens_consulta = set(_normalizar(consulta))
        if not tokens_consulta:
            return candidatos

        for resultado in candidatos:
            tokens_fragmento = set(_normalizar(resultado.fragmento.texto))
            cobertura = len(tokens_consulta & tokens_fragmento) / len(tokens_consulta)

            tokens_seccion = set(_normalizar(resultado.fragmento.seccion))
            bonus_seccion = 0.15 if tokens_consulta & tokens_seccion else 0.0

            # Identificadores exactos: un SKU o un código de política en la
            # consulta debe dominar cualquier señal semántica.
            identificadores = {
                t for t in tokens_consulta if re.search(r"\d", t) or t.isupper()
            }
            bonus_id = 0.25 if identificadores & tokens_fragmento else 0.0

            resultado.score_rerank = (
                0.60 * resultado.score_hibrido
                + 0.40 * cobertura
                + bonus_seccion
                + bonus_id
            )

        return sorted(candidatos, key=lambda r: r.score_rerank, reverse=True)

    def buscar(
        self, consulta: str, top_k: int | None = None, top_n: int | None = None
    ) -> list[Resultado]:
        """Búsqueda híbrida + re-ranking. Devuelve los `top_n` mejores fragmentos."""
        top_k = top_k or CONFIG.top_k_recuperacion
        top_n = top_n or CONFIG.top_n_reranking

        vectoriales = _normalizar_scores(self._scores_vectoriales(consulta))
        lexicos = _normalizar_scores(self._bm25.puntuar(_normalizar(consulta)))
        hibridos = CONFIG.peso_vectorial * vectoriales + (1 - CONFIG.peso_vectorial) * lexicos

        mejores = np.argsort(hibridos)[::-1][:top_k]
        candidatos = [
            Resultado(
                fragmento=self.fragmentos[i],
                score_vectorial=float(vectoriales[i]),
                score_lexico=float(lexicos[i]),
                score_hibrido=float(hibridos[i]),
            )
            for i in mejores
        ]
        return self._rerank(consulta, candidatos)[:top_n]

    def buscar_con_expansion(
        self, consulta: str, top_n: int | None = None
    ) -> list[Resultado]:
        """Búsqueda en dos pasos con *query expansion* sobre códigos de política.

        Problema que resuelve: si el cliente escribe "quiero devolver la miel",
        la primera búsqueda encuentra la ficha del producto (que dice
        `NO_DEV_PERECEDERO`) pero no la sección de la política que explica esa
        exclusión, porque el cliente nunca escribió "alimentos perecederos".
        Sin esa sección el modelo no tiene con qué justificar la negativa.

        La solución es leer los códigos de política presentes en los primeros
        resultados y lanzar una segunda búsqueda con ellos, fusionando ambos
        conjuntos. Es el mismo principio de las *subqueries* que mencionan
        Microsoft y AWS: expandir la consulta del usuario con el vocabulario
        real de los documentos.
        """
        top_n = top_n or CONFIG.top_n_reranking
        resultados = self.buscar(consulta, top_n=top_n)

        codigos: set[str] = set()
        for resultado in resultados[:2]:
            codigos.update(re.findall(r"\b(?:NO_DEV|DEV)_[A-Z0-9_]+\b", resultado.fragmento.texto))

        if not codigos:
            return resultados

        vistos = {r.fragmento.id for r in resultados}
        for extra in self.buscar(" ".join(sorted(codigos)), top_n=top_n):
            if extra.fragmento.id not in vistos:
                resultados.append(extra)
                vistos.add(extra.fragmento.id)

        return resultados[: top_n + 2]

    def construir_contexto(self, resultados: list[Resultado]) -> str:
        """Arma el bloque de contexto que se inyecta en el prompt."""
        return "\n\n---\n\n".join(r.fragmento.como_contexto() for r in resultados)

    def __len__(self) -> int:
        return len(self.fragmentos)
