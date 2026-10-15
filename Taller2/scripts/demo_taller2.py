"""Demostración ejecutable de la Fase 3: el asistente del Taller 1 con el sistema RAG.

Ejecuta casos reales y escribe la transcripción en `outputs/ejecucion_taller2.md`:

A. Con RAG vs. sin RAG: la misma pregunta, con y sin la base de conocimiento.
B. Preguntas que el Taller 1 no podía responder (pagos, cuenta, catálogo ampliado).
C. Fuera de alcance: el asistente reconoce que no tiene la información.
D. Integración con la cadena del Taller 1: pedido (function calling),
   devolución (RAG + expansión) y escalamiento.
E. Experimentos: se modifica UNA pieza del código o del prompt y se observa el
   cambio de comportamiento (top_k, prompt estricto vs. permisivo, umbral,
   modelo de embeddings).

Uso:
    python -m scripts.demo_taller2
    python -m scripts.demo_taller2 --solo A      # una sola sección
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AuthenticationError  # noqa: E402

from src import prompts  # noqa: E402
from src.asistente import AsistenteEcoMarket, Respuesta  # noqa: E402
from src.config import CONFIG, OUTPUTS_DIR  # noqa: E402
from src.llm import ApiKeyFaltante, resumen_uso  # noqa: E402
from src.rag import SistemaRAG  # noqa: E402

logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)

CASOS_A = [
    ("A.1 — Medio de pago que EcoMarket no ofrece",
     "¿Le puedo pagar en efectivo al mensajero cuando me traiga el paquete?",
     "Sin RAG, el modelo responde con lo típico de un e-commerce colombiano y puede "
     "prometer pago contra entrega. Con RAG, la FAQ dice que no existe."),
    ("A.2 — Producto personalizado: no devolvible",
     "Compré una agenda con mi nombre grabado y ya no la quiero, ¿la puedo devolver?",
     "Sin RAG, lo usual es prometer 30 días de devolución. Con RAG, la ficha del "
     "producto (`NO_DEV_PERSONALIZADO`) y la sección 3 de la política dicen que no."),
    ("A.3 — Programa propio de la empresa",
     "¿Tienen algún programa para reciclar los frascos vacíos?",
     "Un modelo general no puede conocer el programa Vuelve a Casa ni su cupón."),
]

CASOS_B = [
    ("B.1 — Facturación a nombre de empresa (intención PAGO_CUENTA, nueva)",
     "Necesito que la factura salga a nombre de mi empresa, ¿cómo hago?"),
    ("B.2 — Catálogo con lenguaje coloquial",
     "¿Qué tienen para llevar el tinto calientico a la oficina?"),
    ("B.3 — Derecho de supresión de datos (Ley 1581)",
     "Quiero que borren toda mi información de su base de datos."),
]

CASOS_C = [
    ("C.1 — Totalmente fuera del dominio",
     "¿Cuál es la capital de Australia?",
     "Ningún fragmento supera el umbral: responde el mensaje fijo, sin llamar al LLM de generación."),
    ("C.2 — Cerca del dominio: pasa el umbral",
     "¿Tienen vacantes de trabajo en EcoMarket?",
     "Menciona a EcoMarket, así que el retrieval encuentra fragmentos parecidos. "
     "La segunda barrera es el prompt estricto: el contexto no habla de empleo."),
    ("C.3 — Producto que EcoMarket no vende",
     "Se me dañó la lavadora, ¿cómo la arreglo?",
     "Se parece a la sección de garantías de electrodomésticos, pero no es un producto de EcoMarket."),
]

CASOS_D = [
    ("D.1 — Estado de pedido retrasado (function calling)",
     "Hola, ¿cómo va mi pedido ECO-10003? Ya se pasó de la fecha."),
    ("D.2 — Producto excluido PERO defectuoso (RAG + expansión por código)",
     "El shampoo del pedido ECO-10009 me llegó partido y con moho, quiero que me lo repongan."),
    ("D.3 — Escalamiento determinista",
     "Esto es un pésimo servicio, me cobraron dos veces y voy a poner una queja en la SIC."),
]


def _detalle(respuesta: Respuesta) -> list[str]:
    datos = [f"intención: `{respuesta.intencion}`", f"llamadas al LLM: {respuesta.llamadas_llm}"]
    if respuesta.escalada:
        datos.append(f"**escalado a humano**: {respuesta.motivo_escalamiento}")
    if not respuesta.en_alcance:
        datos.append("**fuera de alcance** (ningún fragmento supera el umbral)")
    if respuesta.herramienta_invocada:
        datos.append("herramienta invocada por el modelo: `consultar_pedido`")
    if respuesta.recuperados:
        datos.append("fragmentos (similitud): " + ", ".join(
            f"`{r.id}` {r.score:.2f}" + (" *(expansión)*" if r.motivo == "codigo_politica" else "")
            for r in respuesta.recuperados))
    if respuesta.anclaje:
        estado = "OK" if respuesta.anclaje["anclada"] else "REVISAR"
        datos.append(f"anclaje: {estado} (cobertura {respuesta.anclaje['cobertura']})")
    return ["", "> " + "  \n> ".join(datos)]


def _bloque(titulo: str, texto: str) -> list[str]:
    return [f"**{titulo}**", "", *("> " + linea if linea else ">" for linea in texto.strip().splitlines())]


def _tokens(antes: dict, despues: dict) -> str:
    return f"{despues['tokens_entrada'] - antes['tokens_entrada']} / {despues['tokens_salida'] - antes['tokens_salida']}"


def seccion_a(asistente: AsistenteEcoMarket) -> list[str]:
    lineas = ["## A. Con RAG vs. sin RAG", "",
              f"Misma pregunta, mismo modelo (`{CONFIG.modelo}`). La única diferencia es el acceso a la base de conocimiento.", ""]
    for titulo, consulta, nota in CASOS_A:
        print(f"  {titulo}")
        sin = asistente.responder(consulta, modo="sin_rag")
        con = asistente.responder(consulta, modo="rag")
        lineas += [f"### {titulo}", "", f"**Cliente:** «{consulta}»", "", f"*Qué se espera:* {nota}", ""]
        lineas += _bloque("Sin RAG", sin.texto) + [""]
        lineas += _bloque("Con RAG", con.texto) + _detalle(con) + [""]
    return lineas


def seccion_simple(titulo: str, intro: str, casos, asistente: AsistenteEcoMarket) -> list[str]:
    lineas = [f"## {titulo}", "", intro, ""]
    for caso in casos:
        nombre, consulta, *nota = caso
        print(f"  {nombre}")
        respuesta = asistente.responder(consulta)
        lineas += [f"### {nombre}", "", f"**Cliente:** «{consulta}»", ""]
        if nota:
            lineas += [f"*Qué se espera:* {nota[0]}", ""]
        lineas += _bloque("Asistente", respuesta.texto) + _detalle(respuesta) + [""]
    return lineas


def seccion_e(asistente: AsistenteEcoMarket) -> list[str]:
    rag: SistemaRAG = asistente.rag
    cb = {"callbacks": [asistente.uso]}
    lineas = ["## E. Experimentos: modificar una pieza y observar el cambio", "",
              "Cada experimento cambia **una sola** variable y deja todo lo demás igual. "
              "Tokens = entrada / salida de esa ejecución.", ""]

    # E.1 — top_k
    consulta = "¿Cuánto me cuesta el envío a Bogotá y en cuántos días me llega?"
    print("  E.1 top_k")
    lineas += ["### E.1 — `top_k`: cuántos fragmentos llegan al prompt", "",
               f"**Cliente:** «{consulta}»", "",
               "La respuesta necesita dos fragmentos distintos (costo en `faq#ENV-02`, tiempo en `faq#ENV-01`).", ""]
    for k in (1, 4):
        antes = resumen_uso(asistente.uso)
        r = rag.responder(consulta, k=k, **cb)
        lineas += _bloque(f"top_k = {k} — fragmentos: {', '.join(x.id for x in r.recuperados)} — tokens {_tokens(antes, resumen_uso(asistente.uso))}", r.texto) + [""]

    # E.2 — prompt estricto vs permisivo
    consulta = "¿Cuántos años dura la batería de la lámpara solar antes de tener que cambiarla?"
    print("  E.2 prompt")
    lineas += ["### E.2 — Prompt estricto vs. permisivo (una sola instrucción cambia)", "",
               f"**Cliente:** «{consulta}»", "",
               "La ficha de `ECO-LAM-SOL` dice \"hasta 8 h de luz\" pero no habla de la vida útil de la batería.", ""]
    for nombre, plantilla in (("Estricto (`RAG_GENERAL`)", prompts.RAG_GENERAL), ("Permisivo (`RAG_PERMISIVO`)", prompts.RAG_PERMISIVO)):
        r = rag.responder(consulta, plantilla=plantilla, **cb)
        lineas += _bloque(nombre, r.texto) + [""]

    # E.3 — umbral de relevancia
    consulta = "¿Cuál es la capital de Australia?"
    print("  E.3 umbral")
    lineas += ["### E.3 — Umbral de relevancia activado vs. desactivado", "",
               f"**Cliente:** «{consulta}»", ""]
    configuraciones = (
        ("Umbral activado", dict(usar_umbral=True, plantilla=prompts.RAG_GENERAL)),
        ("Umbral desactivado + prompt estricto", dict(usar_umbral=False, plantilla=prompts.RAG_GENERAL)),
        ("Umbral desactivado + prompt permisivo", dict(usar_umbral=False, plantilla=prompts.RAG_PERMISIVO)),
    )
    for nombre, opciones in configuraciones:
        antes = resumen_uso(asistente.uso)
        r = rag.responder(consulta, **opciones, **cb)
        mejor = max(x.score for x in r.recuperados)
        lineas += _bloque(f"{nombre} — similitud máxima {mejor:.2f} (umbral {rag.umbral:.2f}) — tokens {_tokens(antes, resumen_uso(asistente.uso))}", r.texto) + [""]

    # E.4 — modelo de embeddings (solo recuperación)
    print("  E.4 embeddings")
    consultas = ["¿A qué horas puedo hablar con una persona de verdad?", "¿Pueden mandarle un pedido a mi hermana que vive en Miami?"]
    lineas += ["### E.4 — Cambiar el modelo de embeddings (solo recuperación)", "",
               "Mismo índice, mismos fragmentos; cambia el modelo que los vectoriza. "
               "Se muestran los 3 primeros fragmentos de cada uno.", "",
               "| Consulta | TF-IDF (Taller 1) | bge-m3 |", "|---|---|---|"]
    tfidf = SistemaRAG(embeddings="tfidf")
    for consulta in consultas:
        fila = [f"«{consulta}»"]
        for sistema in (tfidf, rag):
            fila.append("<br>".join(f"`{x.id}` {x.score:.2f}" for x in sistema.recuperar(consulta, k=3)))
        lineas.append("| " + " | ".join(fila) + " |")
    lineas += ["", "TF-IDF compara pedazos de palabras; bge-m3 compara significado. La consulta de Miami "
               "es un fallo conocido de ambos: la FAQ dice \"envíos internacionales\" y ningún modelo "
               "conecta \"Miami\" con \"internacional\" con suficiente fuerza (ver Fase 3, limitaciones).", ""]
    return lineas


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo de la Fase 3 del Taller 2.")
    parser.add_argument("--solo", choices=list("ABCDE"), help="Ejecuta una sola sección.")
    args = parser.parse_args()

    try:
        asistente = AsistenteEcoMarket()
    except ApiKeyFaltante as error:
        print(f"\n[ERROR] {error}\n", file=sys.stderr)
        return 1

    secciones = {
        "A": lambda: seccion_a(asistente),
        "B": lambda: seccion_simple("B. Preguntas que el Taller 1 no podía responder",
                                    "Pagos, cuenta y catálogo ampliado: ahora están en la base de conocimiento.",
                                    CASOS_B, asistente),
        "C": lambda: seccion_simple("C. Fuera de alcance: reconocer que no se tiene la información",
                                    "El enunciado pide responder cualquier solicitud o reconocer que no se "
                                    "puede atender. Hay dos barreras: el umbral de similitud (sin gastar "
                                    "tokens de generación) y el prompt estricto.", CASOS_C, asistente),
        "D": lambda: seccion_simple("D. Integración con la cadena del Taller 1",
                                    "Pedidos por function calling, devoluciones con RAG y escalamiento por reglas.",
                                    CASOS_D, asistente),
        "E": lambda: seccion_e(asistente),
    }
    elegidas = [args.solo] if args.solo else list(secciones)

    inicio = time.perf_counter()
    lineas = [
        "# Fase 3 — Ejecución del asistente con RAG",
        "",
        "> Generado por `python -m scripts.demo_taller2`.  ",
        f"> Fecha: {datetime.now():%Y-%m-%d %H:%M} · Fecha de referencia del asistente (\"hoy\"): {prompts.FECHA_REFERENCIA}  ",
        f"> LLM: `{CONFIG.modelo}` ({CONFIG.proveedor}) · Embeddings: `{asistente.rag.embeddings}` · "
        f"Base vectorial: ChromaDB ({asistente.rag.vectorstore._collection.count()} fragmentos) · "
        f"top_k = {asistente.rag.top_k} · umbral = {asistente.rag.umbral:.2f}",
        "",
    ]
    try:
        for letra in elegidas:
            print(f"Sección {letra}")
            lineas += ["---", "", *secciones[letra]()]
    except AuthenticationError:
        print(f"\n[ERROR] El proveedor '{CONFIG.proveedor}' rechazó la API key (401).\n", file=sys.stderr)
        return 1

    uso = asistente.uso_sesion
    lineas += ["---", "", "## Consumo de la ejecución", "",
               f"- Tokens de entrada: **{uso['tokens_entrada']:,}** · de salida: **{uso['tokens_salida']:,}**",
               f"- Costo estimado: **USD {uso['costo_usd']:.4f}** (precios de referencia en `src/llm.py`)",
               f"- Duración: {time.perf_counter() - inicio:.0f} s", ""]

    OUTPUTS_DIR.mkdir(exist_ok=True)
    # La corrida oficial es la del proveedor cloud (ollama u openai); la del
    # respaldo local (Ollama de la máquina, sin key) lleva sufijo.
    sufijo = f"_{CONFIG.proveedor}" if CONFIG.proveedor == "local" else ""
    nombre = f"ejecucion_taller2{sufijo}{'_' + args.solo if args.solo else ''}.md"
    destino = OUTPUTS_DIR / nombre
    destino.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\nTranscripción guardada en: {destino}\nCosto estimado: USD {uso['costo_usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
