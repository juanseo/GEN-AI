"""Construye (o reconstruye) el índice vectorial de la base de conocimiento.

No hace falta ejecutarlo antes de usar el asistente: el índice se construye solo
la primera vez y se reconstruye solo si cambia algún documento de `data/`. Sirve
para forzar una reindexación o para indexar con otro modelo de embeddings.

Uso:
    python -m scripts.indexar                       # modelo por defecto (bge-m3)
    python -m scripts.indexar --embeddings openai-3-small
    python -m scripts.indexar --todos               # todos los modelos disponibles
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG, INDICE_DIR, MODELOS_EMBEDDINGS  # noqa: E402
from src.embeddings import EmbeddingsNoDisponibles  # noqa: E402
from src.indexador import abrir_indice  # noqa: E402

logging.getLogger("langchain_text_splitters").setLevel(logging.ERROR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Indexa la base de conocimiento en ChromaDB.")
    parser.add_argument("--embeddings", default=CONFIG.embeddings, choices=list(MODELOS_EMBEDDINGS))
    parser.add_argument("--todos", action="store_true", help="Indexa con todos los modelos.")
    args = parser.parse_args()

    modelos = [m for m in MODELOS_EMBEDDINGS if m != "tfidf"] if args.todos else [args.embeddings]
    for modelo in modelos:
        inicio = time.perf_counter()
        try:
            vectorstore, efectivo = abrir_indice(modelo, reconstruir=True)
        except EmbeddingsNoDisponibles as error:
            print(f"[{modelo}] omitido: {error}")
            continue
        datos = vectorstore._collection.get(include=["metadatas"])
        por_tipo = Counter(m["tipo"] for m in datos["metadatas"])
        print(
            f"[{efectivo}] {len(datos['ids'])} fragmentos indexados en {time.perf_counter() - inicio:.1f} s "
            f"({', '.join(f'{t}: {n}' for t, n in sorted(por_tipo.items()))}) → {INDICE_DIR}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
