"""El sistema RAG: recuperar → decidir si hay contexto → aumentar el prompt → generar.

Así se conectan las tres piezas que pide el taller:

    consulta ──► modelo de embeddings ──► vector de la consulta
                                              │
                         ChromaDB (HNSW, coseno) ◄── fragmentos indexados
                                              │
                          top-k fragmentos + similitud
                                              │
                    ¿similitud máxima ≥ umbral del modelo?
                       │ no                         │ sí
                       ▼                            ▼
        MENSAJE_FUERA_DE_ALCANCE        cadena LCEL (chaining):
        (texto fijo, 0 tokens)          prompt | LLM | parser de texto

La cadena se arma con LCEL, el lenguaje de composición de LangChain: cada pieza
es un `Runnable` y el operador `|` conecta la salida de una con la entrada de la
siguiente. `ChatPromptTemplate` recibe {consulta, contexto} y produce mensajes;
`ChatOpenAI` produce un mensaje de IA; `StrOutputParser` extrae el texto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from . import prompts
from .config import CONFIG, MODELOS_EMBEDDINGS
from .indexador import abrir_indice
from .llm import crear_llm


@dataclass
class Recuperado:
    """Un fragmento recuperado con su similitud coseno contra la consulta."""

    documento: Document
    score: float
    motivo: str = "similitud"  # "similitud" o "codigo_politica" (expansión)

    @property
    def id(self) -> str:
        return self.documento.metadata["id"]

    @property
    def cita(self) -> str:
        meta = self.documento.metadata
        detalle = meta.get("seccion") or meta.get("sku") or meta.get("categoria") or ""
        return f"{meta['fuente']} > {detalle}".strip()


@dataclass
class RespuestaRAG:
    texto: str
    recuperados: list[Recuperado] = field(default_factory=list)
    en_alcance: bool = True


def supera_umbral(recuperados: list[Recuperado], umbral: float) -> bool:
    """La consulta está en alcance si el MEJOR fragmento es suficientemente parecido.

    Se usa el máximo y no el promedio: una pregunta legítima suele tener uno o
    dos fragmentos muy relevantes y el resto de relleno.
    """
    return bool(recuperados) and max(r.score for r in recuperados) >= umbral


def formatear_contexto(recuperados: list[Recuperado]) -> str:
    """Arma el bloque de contexto: cada fragmento con su id citable y su fuente."""
    return "\n\n---\n\n".join(
        f"[{r.id}] Fuente: {r.cita}\n{r.documento.page_content}" for r in recuperados
    )


class SistemaRAG:
    """Retriever sobre Chroma + cadena de generación con LangChain."""

    def __init__(
        self,
        embeddings: str | None = None,
        llm: BaseChatModel | None = None,
        top_k: int | None = None,
        umbral: float | None = None,
    ) -> None:
        self.vectorstore, self.embeddings = abrir_indice(embeddings or CONFIG.embeddings)
        # El umbral es el del modelo efectivo (puede ser el de respaldo).
        self.umbral = umbral if umbral is not None else MODELOS_EMBEDDINGS[self.embeddings]["umbral"]
        self.top_k = top_k or CONFIG.top_k
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = crear_llm()
        return self._llm

    # -- Recuperación ---------------------------------------------------------

    def recuperar(self, consulta: str, k: int | None = None, filtro: dict | None = None) -> list[Recuperado]:
        pares = self.vectorstore.similarity_search_with_relevance_scores(
            consulta, k=k or self.top_k, filter=filtro
        )
        return [Recuperado(documento, score) for documento, score in pares]

    def expandir_con_politica(self, consulta: str, recuperados: list[Recuperado]) -> list[Recuperado]:
        """Agrega la sección de la política que define el código de cada producto recuperado.

        La ficha de la miel dice `NO_DEV_PERECEDERO`, pero la razón y la excepción
        están en la sección 3 de la política, que la búsqueda semántica no siempre
        trae (el cliente nunca escribe "alimentos perecederos"). Se resuelve con
        dos capacidades de la base vectorial: filtro por metadata
        (`tipo = politica`) y filtro de texto exacto (`$contains` el código).
        Es la misma *query expansion* del Taller 1, ahora nativa en Chroma.
        """
        vistos = {r.id for r in recuperados}
        codigos = {r.documento.metadata["politica"] for r in recuperados if r.documento.metadata.get("politica")}
        extra: list[Recuperado] = []
        for codigo in sorted(codigos):
            pares = self.vectorstore.similarity_search_with_relevance_scores(
                consulta, k=1, filter={"tipo": "politica"}, where_document={"$contains": codigo}
            )
            for documento, score in pares:
                if documento.metadata["id"] not in vistos:
                    extra.append(Recuperado(documento, score, motivo="codigo_politica"))
                    vistos.add(documento.metadata["id"])
        return recuperados + extra

    # -- Generación -----------------------------------------------------------

    def cadena(self, plantilla: str = prompts.RAG_GENERAL, sistema: str = prompts.SISTEMA) -> Runnable:
        """prompt | llm | parser: la cadena LCEL que genera la respuesta."""
        prompt = ChatPromptTemplate.from_messages([("system", sistema), ("human", plantilla)])
        return prompt | self.llm | StrOutputParser()

    def responder(
        self,
        consulta: str,
        plantilla: str = prompts.RAG_GENERAL,
        k: int | None = None,
        usar_umbral: bool = True,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> RespuestaRAG:
        recuperados = self.recuperar(consulta, k)
        if usar_umbral and not supera_umbral(recuperados, self.umbral):
            return RespuestaRAG(prompts.MENSAJE_FUERA_DE_ALCANCE, recuperados, en_alcance=False)
        texto = self.cadena(plantilla).invoke(
            {"consulta": consulta, "contexto": formatear_contexto(recuperados)},
            config={"callbacks": callbacks or []},
        )
        return RespuestaRAG(texto, recuperados, en_alcance=True)
