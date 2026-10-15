"""RAG mínimo de EcoMarket con LangChain, en cinco secciones.

Es la versión didáctica del sistema: un solo archivo, de arriba hacia abajo,
con las mismas decisiones de las Fases 1 y 2. El asistente completo
(`src/asistente.py`) usa exactamente estas piezas y les agrega el router, la
herramienta de pedidos y los guardrails del Taller 1.

    SECCIÓN 1  Documentos  → fragmentos      (Fase 2: carga y chunking)
    SECCIÓN 2  Fragmentos  → vectores        (Fase 1: modelo de embeddings)
    SECCIÓN 3  Vectores    → ChromaDB        (Fase 1: base vectorial)
    SECCIÓN 4  Consulta    → contexto        (retriever con umbral de relevancia)
    SECCIÓN 5  Contexto    → respuesta       (chaining LCEL: prompt | LLM | parser)

Uso:
    python -m scripts.rag_ejemplo                       # 3 preguntas de ejemplo
    python -m scripts.rag_ejemplo "¿puedo pagar a cuotas?"
    python -m scripts.rag_ejemplo --sin-llm             # solo recuperación, sin API key
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.runnables import RunnablePassthrough  # noqa: E402

from src import prompts  # noqa: E402
from src.chunking import construir_chunks  # noqa: E402
from src.config import CONFIG, MODELOS_EMBEDDINGS  # noqa: E402
from src.embeddings import resolver_embeddings  # noqa: E402
from src.llm import crear_llm  # noqa: E402

PREGUNTAS = [
    "¿Le puedo pagar en efectivo al mensajero cuando me traiga el paquete?",
    "Ya no quiero la miel que compré, ¿me la reciben de vuelta?",
    "¿Cuál es la capital de Australia?",  # fuera de alcance
]


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG mínimo de EcoMarket.")
    parser.add_argument("preguntas", nargs="*", help="Preguntas a responder.")
    parser.add_argument("--sin-llm", action="store_true", help="Muestra solo los fragmentos recuperados.")
    args = parser.parse_args()
    logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)

    # -----------------------------------------------------------------------
    # SECCIÓN 1 — Cargar y partir los documentos (Fase 2)
    # Tres formatos: política en PDF, catálogo en CSV y FAQ en JSON. Cada uno
    # se parte por su unidad de significado: sección, producto, pregunta.
    # -----------------------------------------------------------------------
    chunks = construir_chunks(estrategia="estructural")
    print(f"[1] {len(chunks)} fragmentos. Ejemplo ({chunks[11].metadata['id']}):")
    print("    " + chunks[11].page_content.replace("\n", "\n    ")[:300] + "\n")

    # -----------------------------------------------------------------------
    # SECCIÓN 2 — Modelo de embeddings (Fase 1)
    # bge-m3 (open-source, multilingüe) servido por Ollama local. Convierte
    # texto en vectores de 1.024 dimensiones donde "ya no la quiero" queda
    # cerca de "devolución" aunque no compartan palabras.
    # -----------------------------------------------------------------------
    nombre_embeddings, embeddings = resolver_embeddings(CONFIG.embeddings)
    vector = embeddings.embed_query("¿puedo devolver un producto?")
    print(f"[2] Embeddings: {nombre_embeddings} → vector de {len(vector)} dimensiones "
          f"[{vector[0]:+.3f}, {vector[1]:+.3f}, ...]\n")

    # -----------------------------------------------------------------------
    # SECCIÓN 3 — Base vectorial (Fase 1)
    # ChromaDB guarda cada fragmento con su vector y su metadata, y construye
    # un índice HNSW con distancia coseno. Aquí la colección vive en memoria;
    # el asistente usa la versión persistente en disco (src/indexador.py).
    # -----------------------------------------------------------------------
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=[c.metadata["id"] for c in chunks],
        collection_name="rag_ejemplo",
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    print(f"[3] ChromaDB: {vectorstore._collection.count()} vectores indexados\n")

    # -----------------------------------------------------------------------
    # SECCIÓN 4 — Retriever
    # Devuelve los k=4 fragmentos más parecidos, pero solo si superan el umbral
    # de similitud calibrado para el modelo. Si ninguno lo supera, el contexto
    # llega vacío y el prompt obliga al modelo a decir que no tiene la información.
    # -----------------------------------------------------------------------
    umbral = MODELOS_EMBEDDINGS[nombre_embeddings]["umbral"]
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": CONFIG.top_k, "score_threshold": umbral},
    )

    def formatear(documentos) -> str:
        if not documentos:
            return "(no se encontró información relevante en las fuentes de EcoMarket)"
        return "\n\n---\n\n".join(f"[{d.metadata['id']}] {d.page_content}" for d in documentos)

    preguntas = args.preguntas or PREGUNTAS
    if args.sin_llm:
        for pregunta in preguntas:
            documentos = retriever.invoke(pregunta)
            print(f"[4] «{pregunta}» → {[d.metadata['id'] for d in documentos] or 'sin contexto'}")
        return 0

    # -----------------------------------------------------------------------
    # SECCIÓN 5 — Chaining con LCEL
    # El diccionario del inicio corre dos ramas con la misma pregunta:
    #   "contexto": pregunta → retriever → formatear   (lo recuperado, como texto)
    #   "consulta": pregunta tal cual                  (RunnablePassthrough)
    # Después: prompt (sistema + plantilla) → LLM (gpt-4o-mini) → texto.
    # -----------------------------------------------------------------------
    prompt = ChatPromptTemplate.from_messages(
        [("system", prompts.SISTEMA), ("human", prompts.RAG_GENERAL)]
    )
    cadena = (
        {"contexto": retriever | formatear, "consulta": RunnablePassthrough()}
        | prompt
        | crear_llm()
        | StrOutputParser()
    )

    for pregunta in preguntas:
        fuentes = [d.metadata["id"] for d in retriever.invoke(pregunta)]
        print(f"[4] Pregunta: {pregunta}\n    Fragmentos: {fuentes or 'ninguno supera el umbral'}")
        print(f"[5] Respuesta:\n    {cadena.invoke(pregunta).replace(chr(10), chr(10) + '    ')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
