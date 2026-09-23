"""Inspección de la capa de retrieval, sin llamar al LLM.

Permite ver el chunking, los puntajes de la búsqueda vectorial y léxica, la
fusión híbrida y el efecto del re-ranking. No consume tokens ni requiere API
key cuando `ECOMARKET_RETRIEVAL=tfidf` (el valor por defecto).

Uso:
    python -m scripts.demo_retrieval
    python -m scripts.demo_retrieval "puedo devolver la miel organica"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import BaseConocimiento  # noqa: E402

CONSULTAS_DEMO = [
    "quiero devolver la miel organica que compre",
    "el shampoo en barra me llego partido, que hago",
    "cuanto tarda un envio a Leticia",
    "puedo cambiar la talla de una camiseta",
    "que significa que mi pedido este EN_PREPARACION",
]


def main() -> int:
    base = BaseConocimiento()
    print(f"\nFragmentos indexados: {len(base)}  |  motor: {base.motor}\n")

    consultas = sys.argv[1:] or CONSULTAS_DEMO
    for consulta in consultas:
        print("=" * 78)
        print(f"CONSULTA: {consulta}")
        print("=" * 78)
        for posicion, resultado in enumerate(base.buscar(consulta), start=1):
            fragmento = resultado.fragmento
            print(
                f"\n#{posicion}  {fragmento.id}  |  {fragmento.cita}\n"
                f"    vectorial={resultado.score_vectorial:.3f}  "
                f"lexico={resultado.score_lexico:.3f}  "
                f"hibrido={resultado.score_hibrido:.3f}  "
                f"rerank={resultado.score_rerank:.3f}"
            )
            extracto = " ".join(fragmento.texto.split())[:220]
            print(f"    {extracto}...")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
