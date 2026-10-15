"""Chat interactivo con el asistente de EcoMarket (Taller 2, con RAG).

Uso:
    python -m scripts.chat            # modo RAG (por defecto)
    python -m scripts.chat --debug    # muestra intención, fragmentos, similitud y anclaje

Comandos dentro del chat: /debug  /modo (rag <-> sin_rag)  /pedidos  /costo  /salir
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AuthenticationError  # noqa: E402

from src.asistente import AsistenteEcoMarket, Respuesta  # noqa: E402
from src.config import CONFIG  # noqa: E402
from src.herramientas import PEDIDOS  # noqa: E402
from src.llm import ApiKeyFaltante  # noqa: E402

logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


def imprimir_debug(respuesta: Respuesta) -> None:
    print(f"   [intencion] {respuesta.intencion}  [llamadas LLM] {respuesta.llamadas_llm}")
    if respuesta.pii_detectada:
        print(f"   [pii] enmascarada: {', '.join(respuesta.pii_detectada)}")
    if respuesta.escalada:
        print(f"   [escalamiento] {respuesta.motivo_escalamiento}")
    if not respuesta.en_alcance:
        print("   [fuera de alcance] ningún fragmento supera el umbral")
    if respuesta.datos_pedido:
        origen = "el modelo" if respuesta.herramienta_invocada else "respaldo determinista"
        print(f"   [herramienta] consultar_pedido (invocada por {origen})")
    for r in respuesta.recuperados:
        print(f"   [fuente] {r.score:.3f}  {r.id:<26} {r.cita}" + ("  (expansión)" if r.motivo == "codigo_politica" else ""))
    if respuesta.anclaje:
        print(f"   [anclaje] {'OK' if respuesta.anclaje['anclada'] else 'REVISAR'} "
              f"cobertura={respuesta.anclaje['cobertura']} no_soportados={respuesta.anclaje['no_soportados']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat con el asistente de EcoMarket.")
    parser.add_argument("--debug", action="store_true", help="Muestra el detalle interno.")
    args = parser.parse_args()

    try:
        asistente = AsistenteEcoMarket()
    except ApiKeyFaltante as error:
        print(f"\n[ERROR] {error}\n", file=sys.stderr)
        return 1

    modo, debug = "rag", args.debug
    print(f"\nEco · LLM {CONFIG.modelo} · embeddings {asistente.rag.embeddings} · "
          f"umbral {asistente.rag.umbral:.2f} · comandos: /debug /modo /pedidos /costo /salir")
    print("Prueba: '¿puedo pagar a cuotas?', '¿cómo va mi pedido ECO-10003?', '¿venden celulares?'\n")

    while True:
        try:
            entrada = input("Tú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not entrada:
            continue
        if entrada in {"/salir", "/exit"}:
            break
        if entrada == "/debug":
            debug = not debug
            print(f"   debug = {debug}\n")
            continue
        if entrada == "/modo":
            modo = "sin_rag" if modo == "rag" else "rag"
            print(f"   modo = {modo}\n")
            continue
        if entrada == "/pedidos":
            for tracking, pedido in PEDIDOS.items():
                print(f"   {tracking}  {pedido['estado']:<20} {pedido['ciudad_destino']}")
            print()
            continue
        if entrada == "/costo":
            uso = asistente.uso_sesion
            print(f"   entrada={uso['tokens_entrada']} salida={uso['tokens_salida']} USD~{uso['costo_usd']:.5f}\n")
            continue

        try:
            respuesta = asistente.responder(entrada, modo=modo)
        except AuthenticationError:
            print(f"\n[ERROR] El proveedor '{CONFIG.proveedor}' rechazó la API key (401).\n", file=sys.stderr)
            return 1
        print(f"\nEco > {respuesta.texto}\n")
        if debug:
            imprimir_debug(respuesta)
            print()

    print(f"Costo estimado de la sesión: USD {asistente.uso_sesion['costo_usd']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
