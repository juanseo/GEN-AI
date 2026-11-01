"""Evaluación del retrieval: compara modelos de embeddings y estrategias de chunking.

Responde con datos las dos preguntas de las Fases 1 y 2:

1. ¿Qué modelo de embeddings recupera mejor en español para EcoMarket?
   Se comparan la línea base TF-IDF del Taller 1, dos modelos open-source
   servidos por Ollama local (paraphrase-multilingual y bge-m3) y el modelo
   propietario text-embedding-3-small de OpenAI.
2. ¿Qué estrategia de chunking funciona mejor? Se compara la estructural
   (la del asistente) contra la de párrafos y la de tamaño fijo.

Además calibra, para cada modelo, el umbral de similitud a partir del cual una
consulta se considera "fuera de alcance".

Métricas, sobre `data/evaluacion/golden_set.json`:
- hit@1 / hit@4: % de consultas con un fragmento relevante en la posición 1 / entre los 4 primeros
  (4 = `top_k`, lo que realmente llega al prompt).
- MRR: promedio de 1/posición del primer fragmento relevante.
- Separación: similitud máxima de las consultas en alcance vs. fuera de alcance.

La búsqueda se hace con similitud coseno exacta en memoria (numpy). Con 55
fragmentos, el índice HNSW de Chroma devuelve el mismo orden; lo que se mide
aquí es el modelo y el chunking, no la base vectorial.

Uso:
    python -m scripts.evaluar_retrieval
    python -m scripts.evaluar_retrieval --modelos bge-m3 tfidf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import ESTRATEGIAS, construir_chunks  # noqa: E402
from src.config import DATA_DIR, MODELOS_EMBEDDINGS, OUTPUTS_DIR  # noqa: E402
from src.embeddings import EmbeddingsNoDisponibles, crear_embeddings  # noqa: E402

logging.getLogger("langchain_text_splitters").setLevel(logging.ERROR)

K_PROMPT = 4


def _normalizar(texto: str) -> str:
    return " ".join(texto.split())


def _matriz(vectores: list[list[float]]) -> np.ndarray:
    matriz = np.array(vectores, dtype=np.float32)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    return matriz / np.where(normas == 0, 1, normas)


def evaluar(nombre_modelo: str, estrategia: str, golden: dict) -> dict:
    chunks = construir_chunks(estrategia=estrategia)
    textos = [_normalizar(c.page_content) for c in chunks]
    modelo = crear_embeddings(nombre_modelo)
    if nombre_modelo != "tfidf":
        modelo.embed_query("calentamiento")  # la primera llamada carga el modelo en memoria

    inicio = time.perf_counter()
    docs = _matriz(modelo.embed_documents([c.page_content for c in chunks]))
    segundos_indexar = time.perf_counter() - inicio

    resultados, latencias = [], []
    for item in golden["en_alcance"] + golden["fuera_de_alcance"]:
        inicio = time.perf_counter()
        consulta = _matriz([modelo.embed_query(item["consulta"])])[0]
        latencias.append(time.perf_counter() - inicio)
        scores = docs @ consulta
        orden = np.argsort(-scores)

        relevantes = item.get("relevantes")
        posicion = None
        if relevantes:
            claves = [_normalizar(r) for r in relevantes]
            for rango, indice in enumerate(orden[:10], start=1):
                if any(clave in textos[indice] for clave in claves):
                    posicion = rango
                    break
        resultados.append({
            "id": item["id"],
            "consulta": item["consulta"],
            "en_alcance": bool(relevantes),
            "posicion": posicion,
            "score_max": float(scores[orden[0]]),
            "top1": chunks[orden[0]].metadata["id"],
        })

    en = [r for r in resultados if r["en_alcance"]]
    fuera = [r for r in resultados if not r["en_alcance"]]
    return {
        "modelo": nombre_modelo,
        "estrategia": estrategia,
        "fragmentos": len(chunks),
        "hit@1": sum(r["posicion"] == 1 for r in en) / len(en),
        f"hit@{K_PROMPT}": sum(r["posicion"] is not None and r["posicion"] <= K_PROMPT for r in en) / len(en),
        "mrr": sum(1 / r["posicion"] for r in en if r["posicion"]) / len(en),
        "latencia_consulta_ms": 1000 * float(np.median(latencias)),
        "segundos_indexar": segundos_indexar,
        "score_en_alcance": [r["score_max"] for r in en],
        "score_fuera": [r["score_max"] for r in fuera],
        "fallos": [r for r in en if r["posicion"] != 1],
        "detalle": resultados,
    }


def calibrar_umbral(en_alcance: list[float], fuera: list[float], holgura: float = 0.05) -> dict:
    """Elige el umbral de similitud para declarar una consulta "fuera de alcance".

    Política de calibración: **no rechazar ninguna pregunta legítima**. El umbral
    se pone por debajo de la consulta en alcance con menor similitud, menos una
    holgura de 0,05. La holgura existe porque el golden set es pequeño: con
    0,01, una variante de redacción de una pregunta legítima («¿Qué tienen para
    llevar el tinto calientico a la oficina?», similitud 0,44) quedó por debajo
    del umbral en la demo y fue rechazada. Rechazar una pregunta válida deja al cliente sin
    respuesta; dejar pasar una fuera de alcance solo la envía al LLM, cuyo
    prompt estricto le ordena decir que no tiene la información (segunda
    barrera). Se reporta qué proporción de las consultas fuera de alcance
    queda atajada por la primera barrera.
    """
    umbral = float(np.floor((min(en_alcance) - holgura) * 100) / 100)
    return {
        "umbral": umbral,
        "en_alcance_aceptadas": float(np.mean(np.array(en_alcance) >= umbral)),
        "fuera_rechazadas": float(np.mean(np.array(fuera) < umbral)),
        "margen": float(min(en_alcance) - max(fuera)),
    }


def evaluar_asistente(nombre_modelo: str, golden: dict) -> dict:
    """Recuperación tal como la hace el asistente: Chroma (HNSW) + expansión por código de política."""
    from src.rag import SistemaRAG

    rag = SistemaRAG(embeddings=nombre_modelo)
    filas = []
    for item in golden["en_alcance"]:
        basicos = rag.recuperar(item["consulta"])
        expandidos = rag.expandir_con_politica(item["consulta"], basicos)
        claves = [_normalizar(r) for r in item["relevantes"]]

        def acierta(recuperados):
            return any(any(c in _normalizar(r.documento.page_content) for c in claves) for r in recuperados)

        filas.append({"id": item["id"], "sin_expansion": acierta(basicos), "con_expansion": acierta(expandidos),
                      "fragmentos": len(expandidos)})
    return {
        "modelo": rag.embeddings,
        f"hit@{K_PROMPT}_chroma": sum(f["sin_expansion"] for f in filas) / len(filas),
        "hit_con_expansion": sum(f["con_expansion"] for f in filas) / len(filas),
        "fragmentos_promedio": sum(f["fragmentos"] for f in filas) / len(filas),
        "rescatadas": [f["id"] for f in filas if f["con_expansion"] and not f["sin_expansion"]],
    }


def _tabla(filas: list[list[str]], encabezados: list[str]) -> list[str]:
    return [
        "| " + " | ".join(encabezados) + " |",
        "|" + "|".join("---" for _ in encabezados) + "|",
        *("| " + " | ".join(fila) + " |" for fila in filas),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modelos", nargs="+", default=list(MODELOS_EMBEDDINGS))
    parser.add_argument("--modelo-chunking", default="bge-m3",
                        help="Modelo con el que se comparan las estrategias de chunking.")
    args = parser.parse_args()

    golden = json.loads((DATA_DIR / "evaluacion" / "golden_set.json").read_text(encoding="utf-8"))
    corridas: list[dict] = []
    omitidos: dict[str, str] = {}

    for modelo in args.modelos:
        estrategias = ESTRATEGIAS if modelo == args.modelo_chunking else ("estructural",)
        for estrategia in estrategias:
            print(f"Evaluando {modelo} / {estrategia} ...", flush=True)
            try:
                corridas.append(evaluar(modelo, estrategia, golden))
            except EmbeddingsNoDisponibles as error:
                omitidos[modelo] = str(error)
                print(f"  omitido: {error}")
                break

    estructural = [c for c in corridas if c["estrategia"] == "estructural"]
    for corrida in estructural:
        corrida["calibracion"] = calibrar_umbral(corrida["score_en_alcance"], corrida["score_fuera"])

    lineas = [
        "# Evaluación del retrieval: modelos de embeddings y estrategias de chunking",
        "",
        "> Generado por `python -m scripts.evaluar_retrieval` sobre "
        f"`data/evaluacion/golden_set.json` ({len(golden['en_alcance'])} consultas en alcance, "
        f"{len(golden['fuera_de_alcance'])} fuera de alcance).  ",
        f"> Fecha: {time.strftime('%Y-%m-%d %H:%M')} · CPU, sin GPU.",
        "",
        "## 1. Modelos de embeddings (chunking estructural)",
        "",
    ]
    lineas += _tabla(
        [[c["modelo"], MODELOS_EMBEDDINGS[c["modelo"]]["licencia"],
          f"{c['hit@1']:.0%}", f"{c[f'hit@{K_PROMPT}']:.0%}", f"{c['mrr']:.3f}",
          f"{c['latencia_consulta_ms']:.0f} ms", f"{c['segundos_indexar']:.1f} s"]
         for c in estructural],
        ["Modelo", "Licencia", "hit@1", f"hit@{K_PROMPT}", "MRR", "Latencia por consulta (mediana)", "Indexar 55 fragmentos"],
    )
    lineas += [
        "",
        "## 2. Detección de consultas fuera de alcance",
        "",
        "Similitud coseno del **mejor** fragmento recuperado. Si el modelo separa bien, "
        "las consultas en alcance quedan por encima y las de fuera por debajo, y existe "
        "un umbral que las distingue.",
        "",
    ]
    lineas += _tabla(
        [[c["modelo"],
          f"{np.min(c['score_en_alcance']):.3f} – {np.max(c['score_en_alcance']):.3f}",
          f"{np.min(c['score_fuera']):.3f} – {np.max(c['score_fuera']):.3f}",
          f"{c['calibracion']['margen']:+.3f}",
          f"{c['calibracion']['umbral']:.2f}",
          f"{c['calibracion']['en_alcance_aceptadas']:.0%}",
          f"{c['calibracion']['fuera_rechazadas']:.0%}"]
         for c in estructural],
        ["Modelo", "En alcance (mín – máx)", "Fuera (mín – máx)", "Margen", "Umbral",
         "En alcance aceptadas", "Fuera de alcance atajadas"],
    )
    lineas += [
        "",
        "Margen negativo = hay consultas fuera de alcance más parecidas a la base que "
        "algunas legítimas: ningún umbral las separa todas. El umbral se fija para no "
        "rechazar ninguna pregunta legítima; las que se cuelan las ataja el prompt "
        "estricto (ver la sección C de la demo de la Fase 3, `outputs/ejecucion_taller2*.md`).",
    ]

    chunking = [c for c in corridas if c["modelo"] == args.modelo_chunking]
    if len(chunking) > 1:
        lineas += ["", f"## 3. Estrategias de chunking (modelo {args.modelo_chunking})", ""]
        lineas += _tabla(
            [[c["estrategia"], str(c["fragmentos"]), f"{c['hit@1']:.0%}",
              f"{c[f'hit@{K_PROMPT}']:.0%}", f"{c['mrr']:.3f}"] for c in chunking],
            ["Estrategia", "Fragmentos", "hit@1", f"hit@{K_PROMPT}", "MRR"],
        )

    lineas += ["", "## 4. Consultas que no quedaron en la posición 1", ""]
    for c in corridas:
        if not c["fallos"]:
            continue
        lineas.append(f"**{c['modelo']} / {c['estrategia']}**")
        lineas.append("")
        for fallo in c["fallos"]:
            posicion = fallo["posicion"] or ">10"
            lineas.append(f"- {fallo['id']} «{fallo['consulta']}» → posición {posicion}; top-1: `{fallo['top1']}`")
        lineas.append("")

    if args.modelo_chunking in [c["modelo"] for c in estructural]:
        asistente = evaluar_asistente(args.modelo_chunking, golden)
        exacto = next(c[f"hit@{K_PROMPT}"] for c in estructural if c["modelo"] == args.modelo_chunking)
        lineas += [
            f"## 5. Recuperación del asistente ({asistente['modelo']} en ChromaDB)",
            "",
            f"- hit@{K_PROMPT} con el índice HNSW de Chroma: **{asistente[f'hit@{K_PROMPT}_chroma']:.0%}** "
            f"(cálculo exacto de la sección 1: {exacto:.0%}"
            + ("; a esta escala HNSW no pierde vecinos)." if abs(exacto - asistente[f'hit@{K_PROMPT}_chroma']) < 1e-9
               else "; la diferencia es la aproximación de HNSW)."),
            f"- Con la expansión por código de política (rama de devoluciones): "
            f"**{asistente['hit_con_expansion']:.0%}**, con {asistente['fragmentos_promedio']:.1f} "
            "fragmentos promedio en el contexto.",
            f"- Consultas rescatadas por la expansión: {', '.join(asistente['rescatadas']) or 'ninguna'}.",
            "",
        ]

    if omitidos:
        lineas += ["## Modelos omitidos", ""] + [f"- `{m}`: {e}" for m, e in omitidos.items()] + [""]

    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "evaluacion_retrieval.md").write_text("\n".join(lineas), encoding="utf-8")
    resumen = [{k: v for k, v in c.items() if k not in ("detalle", "fallos")} for c in corridas]
    (OUTPUTS_DIR / "evaluacion_retrieval.json").write_text(
        json.dumps({"corridas": resumen, "detalle": {f"{c['modelo']}/{c['estrategia']}": c["detalle"] for c in corridas}},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n".join(lineas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
