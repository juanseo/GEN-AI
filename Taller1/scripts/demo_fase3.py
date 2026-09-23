"""Demostración ejecutable de la Fase 3 — Ingeniería de Prompts.

Ejecuta los dos ejercicios del taller sobre casos reales de la base de prueba,
comparando el prompt básico contra el prompt optimizado, y escribe la
transcripción completa en `outputs/ejecucion_fase3.md`.

Uso:
    python -m scripts.demo_fase3              # ejecuta todos los casos
    python -m scripts.demo_fase3 --solo 3     # ejecuta solo el caso 3
    python -m scripts.demo_fase3 --sin-basico # omite la versión básica (más barato)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chain import AsistenteEcoMarket  # noqa: E402
from src.config import CONFIG, OUTPUTS_DIR  # noqa: E402
from openai import AuthenticationError  # noqa: E402
from src.llm_client import ApiKeyFaltante  # noqa: E402

CASOS = [
    {
        "titulo": "Ejercicio 1.a — Pedido en tránsito, caso feliz",
        "ejercicio": 1,
        "consulta": "Hola, buenas. ¿Me pueden decir cómo va mi pedido ECO-10001?",
        "que_demuestra": (
            "El prompt básico no tiene acceso al sistema de pedidos: o inventa un "
            "estado o responde con evasivas. El optimizado invoca la herramienta "
            "`consultar_pedido` y responde con el estado, la transportadora, la "
            "fecha estimada y el enlace de rastreo reales."
        ),
    },
    {
        "titulo": "Ejercicio 1.b — Pedido retrasado, requiere disculpa y explicación",
        "ejercicio": 1,
        "consulta": "Mi pedido ECO-10003 ya se pasó de la fecha, ¿qué pasó?",
        "que_demuestra": (
            "El prompt optimizado detecta `retrasado: true`, ofrece la disculpa, "
            "explica el motivo real (cierre de vía por clima) y contrasta la fecha "
            "original con la nueva. El básico no puede hacer nada de eso."
        ),
    },
    {
        "titulo": "Ejercicio 1.c — Pedido inexistente (prueba anti-alucinación)",
        "ejercicio": 1,
        "consulta": "Quiero saber el estado del pedido ECO-99999 por favor.",
        "que_demuestra": (
            "Es la prueba crítica. El prompt básico tiende a inventar un estado "
            "plausible. El optimizado recibe `encontrado: false` y la instrucción "
            "explícita de no inventar, así que pide verificar el número."
        ),
    },
    {
        "titulo": "Ejercicio 2.a — Devolución de producto EXCLUIDO (alimento)",
        "ejercicio": 2,
        "consulta": (
            "Compré la miel orgánica en el pedido ECO-10008 y ya no la quiero. "
            "¿Cómo la devuelvo?"
        ),
        "que_demuestra": (
            "El reto del taller. El prompt básico aplica la política genérica de "
            "cualquier e-commerce y promete una devolución que EcoMarket no puede "
            "cumplir. El optimizado recupera la política, identifica la categoría "
            "`NO_DEV_PERECEDERO`, dice que no con empatía y ofrece alternativa."
        ),
    },
    {
        "titulo": "Ejercicio 2.b — Producto excluido PERO defectuoso (excepción de garantía)",
        "ejercicio": 2,
        "consulta": (
            "El shampoo en barra del pedido ECO-10009 me llegó partido y con moho. "
            "Quiero que me lo repongan."
        ),
        "que_demuestra": (
            "Verifica que el modelo no aplique la exclusión de forma mecánica: la "
            "excepción por producto defectuoso prevalece sobre la categoría de "
            "higiene, y debe ofrecer reposición o reembolso total."
        ),
    },
    {
        "titulo": "Ejercicio 2.c — Producto con sello condicional (no asumir)",
        "ejercicio": 2,
        "consulta": (
            "Quiero devolver el jabón de caléndula que venía en el pedido ECO-10003."
        ),
        "que_demuestra": (
            "El producto es `DEV_CONDICIONAL_SELLO`: el prompt obliga al modelo a "
            "preguntar por el sello de seguridad antes de prometer la devolución, "
            "en lugar de asumir una respuesta."
        ),
    },
    {
        "titulo": "Ejercicio 2.d — Devolución imposible: el pedido no ha llegado",
        "ejercicio": 2,
        "consulta": "Quiero devolver la compostera del pedido ECO-10004.",
        "que_demuestra": (
            "El pedido está `EN_PREPARACION`. El paso 5 del árbol de decisión hace "
            "que el modelo aclare que corresponde una cancelación, no una "
            "devolución, y explique la diferencia."
        ),
    },
    {
        "titulo": "Guardrail — Escalamiento automático a agente humano",
        "ejercicio": 0,
        "consulta": (
            "Esto es un pésimo servicio, me cobraron dos veces y voy a poner una "
            "queja en la SIC."
        ),
        "que_demuestra": (
            "El guardrail determinista intercepta la consulta antes de llamar al "
            "modelo: hay mención de cobro duplicado y de la SIC. No se gasta un "
            "token y el caso va directo a una persona."
        ),
    },
]


def formatear_respuesta(respuesta) -> str:
    partes = [respuesta.texto.strip()]
    detalles = []
    if respuesta.intencion:
        detalles.append(f"intención detectada: `{respuesta.intencion}`")
    if respuesta.escalada:
        detalles.append(f"**escalado a humano** — {respuesta.motivo_escalamiento}")
    if respuesta.pii_detectada:
        detalles.append(f"PII enmascarada: {', '.join(respuesta.pii_detectada)}")
    if respuesta.fuentes:
        detalles.append("fragmentos recuperados: " + "; ".join(respuesta.fuentes))
    if respuesta.anclaje:
        estado = "OK" if respuesta.anclaje["anclada"] else "REVISAR"
        detalles.append(
            f"anclaje: {estado} (cobertura {respuesta.anclaje['cobertura']})"
        )
    detalles.append(f"llamadas al LLM: {respuesta.llamadas_llm}")
    if detalles:
        partes.append("")
        partes.append("> " + "  \n> ".join(detalles))
    return "\n".join(partes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo de la Fase 3 del Taller 1.")
    parser.add_argument("--solo", type=int, help="Ejecuta solo el caso N (1-indexado).")
    parser.add_argument(
        "--sin-basico", action="store_true", help="Omite la versión con prompt básico."
    )
    args = parser.parse_args()

    try:
        asistente = AsistenteEcoMarket()
    except ApiKeyFaltante as error:
        print(f"\n[ERROR] {error}\n", file=sys.stderr)
        return 1

    casos = CASOS if args.solo is None else [CASOS[args.solo - 1]]

    lineas = [
        "# Fase 3 — Ejecución real de los prompts",
        "",
        "> Salida generada automáticamente por `scripts/demo_fase3.py`.",
        f"> Fecha de ejecución: {datetime.now():%Y-%m-%d %H:%M}  ",
        f"> Proveedor: `{CONFIG.proveedor}` · Modelo: `{CONFIG.modelo}` · Motor de retrieval: `{asistente.base.motor}` · "
        f"Fragmentos indexados: {len(asistente.base)}",
        "",
        "En cada caso se ejecuta la **misma consulta** con dos prompts distintos. "
        "La diferencia entre las dos respuestas es, literalmente, la ingeniería de prompts.",
        "",
    ]

    try:
        for indice, caso in enumerate(casos, start=1):
            print(f"[{indice}/{len(casos)}] {caso['titulo']}")
            lineas += [
                "---",
                "",
                f"## {caso['titulo']}",
                "",
                f"**Consulta del cliente:** «{caso['consulta']}»",
                "",
                f"**Qué demuestra este caso:** {caso['que_demuestra']}",
                "",
            ]

            if not args.sin_basico and caso["ejercicio"] > 0:
                basica = asistente.responder(caso["consulta"], modo="basico")
                lineas += [
                    "### Prompt BÁSICO",
                    "",
                    formatear_respuesta(basica),
                    "",
                ]

            optimizada = asistente.responder(caso["consulta"], modo="optimizado")
            lineas += [
                "### Prompt OPTIMIZADO",
                "",
                formatear_respuesta(optimizada),
                "",
            ]
    except AuthenticationError:
        print(
            f"\n[ERROR] El proveedor '{CONFIG.proveedor}' rechazó la API key "
            "(401). La clave se leyó del .env, pero no es válida o fue "
            "revocada. Genera una nueva y actualiza el archivo .env.\n",
            file=sys.stderr,
        )
        return 1

    uso = asistente.cliente.uso
    lineas += [
        "---",
        "",
        "## Consumo de la ejecución",
        "",
        f"- Llamadas al API: **{uso.llamadas}**",
        f"- Tokens de entrada: **{uso.tokens_entrada:,}**",
        f"- Tokens de salida: **{uso.tokens_salida:,}**",
        f"- Costo estimado: **USD {asistente.costo_sesion_usd:.5f}**",
        "",
        "> El costo se calcula con los precios de referencia declarados en "
        "`src/llm_client.py`. Verificar los precios vigentes antes de extrapolar. "
        "Los modelos de Ollama Cloud se cobran por suscripción, no por token, "
        "por lo que su costo aparece como 0.",
        "",
    ]

    OUTPUTS_DIR.mkdir(exist_ok=True)
    destino = OUTPUTS_DIR / "ejecucion_fase3.md"
    destino.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\nTranscripción guardada en: {destino}")
    print(f"Costo estimado de esta corrida: USD {asistente.costo_sesion_usd:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
