"""Chat interactivo con el asistente de EcoMarket.

Uso:
    python -m scripts.chat            # modo optimizado (por defecto)
    python -m scripts.chat --basico   # modo básico, para comparar
    python -m scripts.chat --debug    # muestra intención, fuentes y anclaje

Comandos dentro del chat: /salir, /debug, /modo, /pedidos, /costo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chain import AsistenteEcoMarket  # noqa: E402
from src.config import CONFIG  # noqa: E402
from src.herramientas import PEDIDOS  # noqa: E402
from openai import AuthenticationError  # noqa: E402
from src.llm_client import ApiKeyFaltante  # noqa: E402

BANNER = """
==========================================================
  Eco - Asistente de atencion al cliente de EcoMarket
==========================================================
  Modelo: {modelo} | Retrieval: {retrieval} | Modo: {modo}
  Comandos: /salir  /debug  /modo  /pedidos  /costo
==========================================================
"""


def imprimir_debug(respuesta) -> None:
    print(f"   [intencion] {respuesta.intencion}")
    if respuesta.pii_detectada:
        print(f"   [pii] enmascarada: {', '.join(respuesta.pii_detectada)}")
    if respuesta.escalada:
        print(f"   [escalamiento] {respuesta.motivo_escalamiento}")
    if respuesta.datos_pedido:
        encontrado = respuesta.datos_pedido.get("encontrado")
        print(f"   [herramienta] consultar_pedido -> encontrado={encontrado}")
    for resultado in respuesta.fragmentos:
        print(
            f"   [fuente] {resultado.fragmento.id} | {resultado.fragmento.cita} "
            f"| vect={resultado.score_vectorial:.2f} lex={resultado.score_lexico:.2f} "
            f"rerank={resultado.score_rerank:.2f}"
        )
    if respuesta.anclaje:
        print(
            f"   [anclaje] {'OK' if respuesta.anclaje['anclada'] else 'REVISAR'} "
            f"cobertura={respuesta.anclaje['cobertura']} "
            f"no_soportados={respuesta.anclaje['no_soportados']}"
        )
    print(f"   [llamadas] {respuesta.llamadas_llm}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat con el asistente de EcoMarket.")
    parser.add_argument("--basico", action="store_true", help="Usa el prompt básico.")
    parser.add_argument("--debug", action="store_true", help="Muestra el detalle interno.")
    args = parser.parse_args()

    try:
        asistente = AsistenteEcoMarket()
    except ApiKeyFaltante as error:
        print(f"\n[ERROR] {error}\n", file=sys.stderr)
        return 1

    modo = "basico" if args.basico else "optimizado"
    debug = args.debug

    print(BANNER.format(modelo=CONFIG.modelo, retrieval=asistente.base.motor, modo=modo))
    print("Prueba con: 'como va mi pedido ECO-10003?' o 'quiero devolver la miel'\n")

    while True:
        try:
            entrada = input("Tu > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not entrada:
            continue
        if entrada in {"/salir", "/exit", "/quit"}:
            break
        if entrada == "/debug":
            debug = not debug
            print(f"   debug = {debug}\n")
            continue
        if entrada == "/modo":
            modo = "basico" if modo == "optimizado" else "optimizado"
            print(f"   modo = {modo}\n")
            continue
        if entrada == "/pedidos":
            for tracking, pedido in PEDIDOS.items():
                print(f"   {tracking}  {pedido['estado']:<20} {pedido['ciudad_destino']}")
            print()
            continue
        if entrada == "/costo":
            uso = asistente.cliente.uso
            print(
                f"   llamadas={uso.llamadas}  entrada={uso.tokens_entrada}  "
                f"salida={uso.tokens_salida}  USD~{asistente.costo_sesion_usd:.5f}\n"
            )
            continue

        try:
            respuesta = asistente.responder(entrada, modo=modo)
        except AuthenticationError:
            print(
                "\n[ERROR] OpenAI rechazo la API key (401). La clave se leyo del "
                ".env pero no es valida o fue revocada. Genera una nueva en "
                "https://platform.openai.com/api-keys y actualiza el .env.\n",
                file=sys.stderr,
            )
            return 1

        print(f"\nEco > {respuesta.texto}\n")
        if debug:
            imprimir_debug(respuesta)
            print()

    print(f"\nCosto estimado de la sesion: USD {asistente.costo_sesion_usd:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
