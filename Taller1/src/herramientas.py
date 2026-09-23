"""Acceso a datos estructurados de pedidos vía *function calling*.

Decisión de arquitectura: el estado de un pedido **no** se resuelve con
búsqueda vectorial. Un pedido es un registro transaccional, exacto y volátil:
la similitud semántica podría devolver el pedido ECO-10002 cuando el cliente
preguntó por el ECO-10012, y eso es un error inaceptable. RAG se reserva para
el conocimiento no estructurado (políticas, catálogo, FAQ) y los pedidos se
consultan con una función determinista expuesta al modelo como herramienta.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from .config import DATA_DIR

PATRON_TRACKING = re.compile(r"\bECO[\s\-]?(\d{5})\b", re.IGNORECASE)

ETIQUETAS_ESTADO = {
    "PAGO_PENDIENTE": "Pago pendiente de confirmación",
    "EN_PREPARACION": "En preparación en bodega",
    "EN_TRANSITO": "En tránsito con la transportadora",
    "RETRASADO": "Retrasado por una novedad logística",
    "ENTREGADO": "Entregado",
    "DEVOLUCION_EN_CURSO": "Con una devolución en curso",
    "CANCELADO": "Cancelado",
}


def cargar_pedidos(ruta: Path | None = None) -> dict[str, dict[str, Any]]:
    ruta = ruta or DATA_DIR / "pedidos.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return {p["tracking"]: p for p in datos["pedidos"]}


PEDIDOS = cargar_pedidos()


def normalizar_tracking(texto: str) -> str | None:
    """Extrae y normaliza un número de seguimiento de un texto libre."""
    coincidencia = PATRON_TRACKING.search(texto or "")
    return f"ECO-{coincidencia.group(1)}" if coincidencia else None


def consultar_pedido(tracking: str) -> dict[str, Any]:
    """Herramienta expuesta al modelo: devuelve el estado de un pedido.

    Si el pedido no existe devuelve `encontrado: False` en lugar de lanzar una
    excepción, para que el modelo pueda responder con honestidad en vez de
    inventar un estado.
    """
    codigo = normalizar_tracking(tracking) or (tracking or "").strip().upper()
    pedido = PEDIDOS.get(codigo)

    if pedido is None:
        return {
            "encontrado": False,
            "tracking_consultado": codigo,
            "mensaje": (
                "No existe un pedido con ese número de seguimiento en el sistema. "
                "No inventes un estado: pide al cliente que verifique el número."
            ),
        }

    resultado = {
        "encontrado": True,
        "tracking": pedido["tracking"],
        "estado": pedido["estado"],
        "estado_legible": ETIQUETAS_ESTADO.get(pedido["estado"], pedido["estado"]),
        "fecha_compra": pedido["fecha_compra"],
        "ciudad_destino": pedido["ciudad_destino"],
        "transportadora": pedido["transportadora"],
        "fecha_estimada_entrega": pedido["fecha_estimada_entrega"],
        "retrasado": pedido["retrasado"],
        "motivo_retraso": pedido["motivo_retraso"],
        "url_rastreo": pedido["url_rastreo"],
        "productos": [
            {"sku": i["sku"], "nombre": i["nombre"], "cantidad": i["cantidad"]}
            for i in pedido["items"]
        ],
        "total_cop": pedido["total_cop"],
        "fecha_consulta": date.today().isoformat(),
    }
    for opcional in ("fecha_entrega_real", "fecha_estimada_original",
                     "nota_devolucion", "nota_cancelacion"):
        if opcional in pedido:
            resultado[opcional] = pedido[opcional]
    return resultado


# Esquema de la herramienta en el formato que espera la API de OpenAI.
ESQUEMA_HERRAMIENTAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "consultar_pedido",
            "description": (
                "Consulta el sistema de pedidos de EcoMarket y devuelve el estado "
                "real de un pedido a partir de su número de seguimiento. Úsala "
                "SIEMPRE que el cliente pregunte por un pedido; nunca respondas "
                "sobre un pedido sin haber llamado esta función."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking": {
                        "type": "string",
                        "description": "Número de seguimiento, formato ECO-#####.",
                    }
                },
                "required": ["tracking"],
            },
        },
    }
]

FUNCIONES_DISPONIBLES = {"consultar_pedido": consultar_pedido}
