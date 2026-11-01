"""Motor de reglas de devolución: la política v3.2 traducida a código.

Decisión de arquitectura: el agente **no** decide si un producto se puede
devolver. Esa decisión tiene efectos (una guía emitida, un cobro de COP 12.000,
un derecho del consumidor negado) y un LLM se puede equivocar o ser manipulado
con una inyección de prompt. El modelo solo extrae de la conversación el pedido,
el producto y el motivo; esta función pura aplica la política y devuelve el
resultado con las cláusulas que lo sustentan, para que el agente lo explique.

Las reglas se evalúan en orden y gana la primera que aplica. El orden importa:
la excepción del §3 (producto defectuoso) va antes que las exclusiones, porque
la política dice que prevalece sobre ellas.

    1. Estado distinto de ENTREGADO           -> NO_ELEGIBLE + alternativa   (§2, §7)
    2. Defecto dentro del plazo de garantía   -> ELEGIBLE vía garantía        (§3, §4)
    3. Defecto fuera de ese plazo             -> REVISION_HUMANA              (§9)
    4. Categoría NO_DEV_*                      -> NO_ELEGIBLE                  (§3)
    5. Más de 30 días desde la entrega        -> NO_ELEGIBLE                  (§1)
    6. Higiene con sello sin confirmar o roto -> REQUIERE_CONFIRMACION / NO_ELEGIBLE (§4)
    7. Cambio de talla en textiles            -> ELEGIBLE, gratis la 1.a vez  (§4, §5)
    8. Hasta 5 días hábiles                    -> ELEGIBLE por retracto        (§1, Ley 1480)
    9. Día hábil 6 a día calendario 30        -> ELEGIBLE, paga el cliente    (§5)

La regla 6 se aplica después de calcular la vía (8 o 9) para que, al pedir la
confirmación del sello, el agente pueda informar de una vez quién paga el retorno.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .config import DATA_DIR, FECHA_REFERENCIA

COSTO_RETORNO_CLIENTE_COP = 12000      # §5: tarifa plana cuando paga el cliente
PLAZO_DEVOLUCION_DIAS = 30             # §1: días calendario desde la entrega
PLAZO_RETRACTO_DIAS_HABILES = 5        # §1: Ley 1480 de 2011, art. 47
PLAZO_GARANTIA_EXTENDIDA_DIAS = 365    # §4: DEV_GARANTIA_12M

MOTIVOS_DEFECTO = {"DEFECTUOSO", "EQUIVOCADO", "INCOMPLETO", "VENCIDO"}
MOTIVOS_VALIDOS = MOTIVOS_DEFECTO | {"CAMBIO_TALLA", "ARREPENTIMIENTO"}

# Festivos de Colombia en 2026 (Ley 51 de 1983): no cuentan como días hábiles
# para el retracto. Los pedidos de prueba son de 2026.
FESTIVOS_CO = [
    "2026-01-01", "2026-01-12", "2026-03-23", "2026-04-02", "2026-04-03",
    "2026-05-01", "2026-05-18", "2026-06-08", "2026-06-15", "2026-06-29",
    "2026-07-20", "2026-08-07", "2026-08-17", "2026-10-12", "2026-11-02",
    "2026-11-16", "2026-12-08", "2026-12-25",
]

_NO_HA_LLEGADO = (
    "El pedido todavía no ha llegado: el cliente puede rechazar la entrega o "
    "pedir la devolución cuando lo reciba."
)
_SE_PUEDE_CANCELAR = "El pedido aún no se ha despachado: se puede cancelar sin costo."
ALTERNATIVA_POR_ESTADO = {
    "PAGO_PENDIENTE": _SE_PUEDE_CANCELAR,
    "EN_PREPARACION": _SE_PUEDE_CANCELAR,
    "EN_TRANSITO": _NO_HA_LLEGADO,
    "RETRASADO": _NO_HA_LLEGADO,
    "DEVOLUCION_EN_CURSO": "El pedido ya tiene una devolución en curso; no hace falta abrir otra.",
    "CANCELADO": "El pedido está cancelado: no hay productos que devolver.",
}
ALTERNATIVA_GARANTIA = (
    "Si el producto llegó defectuoso, vencido, incompleto o distinto al comprado, "
    "sí aplica la garantía."
)


@dataclass
class Evaluacion:
    """Decisión del motor, con todo lo que el agente necesita para explicarla.

    `ok=False` significa que la entrada no se pudo evaluar (`codigo` dice por qué);
    con `ok=True`, `resultado` es ELEGIBLE, NO_ELEGIBLE, REQUIERE_CONFIRMACION o
    REVISION_HUMANA.
    """

    ok: bool
    codigo: str | None = None
    resultado: str | None = None
    via: str | None = None                 # GARANTIA, CAMBIO_TALLA, RETRACTO_5D, ARREPENTIMIENTO_30D
    politica: str | None = None            # código de la política del SKU en el catálogo
    dias_desde_entrega: int | None = None
    paga_retorno: str | None = None        # ECOMARKET o CLIENTE
    costo_retorno_cop: int = 0
    clausulas: list[str] = field(default_factory=list)
    detalle: str = ""
    alternativa: str | None = None
    pregunta_pendiente: str | None = None


def cargar_politicas(ruta: Path | None = None) -> dict[str, str]:
    """SKU -> código de política de devolución, según el catálogo."""
    ruta = ruta or DATA_DIR / "catalogo_productos.csv"
    with ruta.open(encoding="utf-8", newline="") as archivo:
        return {fila["sku"]: fila["politica_devolucion"] for fila in csv.DictReader(archivo)}


POLITICAS = cargar_politicas()


def dias_habiles_entre(desde: date, hasta: date) -> int:
    """Días hábiles posteriores a `desde`, hasta `hasta` inclusive."""
    inicio, fin = desde + timedelta(days=1), hasta + timedelta(days=1)
    return int(np.busday_count(inicio, fin, holidays=FESTIVOS_CO))


def evaluar(
    pedido: dict[str, Any] | None,
    sku: str,
    motivo: str,
    sello_intacto: bool | None = None,
    *,
    cambios_previos: int = 0,
    hoy: date | None = None,
) -> Evaluacion:
    """Aplica la política v3.2 a un producto de un pedido (registro de pedidos.json)."""
    hoy = hoy or date.fromisoformat(FECHA_REFERENCIA)
    sku = (sku or "").strip().upper()
    motivo = (motivo or "").strip().upper()

    if pedido is None:
        return Evaluacion(ok=False, codigo="PEDIDO_NO_EXISTE",
                          detalle="No existe un pedido con ese número de seguimiento.")
    if sku not in {item["sku"] for item in pedido["items"]}:
        return Evaluacion(ok=False, codigo="SKU_NO_PERTENECE",
                          detalle=f"El producto {sku} no hace parte del pedido {pedido['tracking']}.")
    if motivo not in MOTIVOS_VALIDOS:
        return Evaluacion(ok=False, codigo="MOTIVO_INVALIDO",
                          detalle=f"Motivo no reconocido. Valores válidos: {', '.join(sorted(MOTIVOS_VALIDOS))}.")

    politica = POLITICAS.get(sku)

    # 1. Solo se devuelven pedidos entregados.
    estado = pedido["estado"]
    if estado != "ENTREGADO":
        return Evaluacion(
            ok=True, politica=politica, resultado="NO_ELEGIBLE", codigo="ESTADO_NO_ADMITE",
            clausulas=["§2", "§7"], alternativa=ALTERNATIVA_POR_ESTADO.get(estado),
            detalle=f"El pedido está en estado {estado}; solo se devuelven pedidos entregados.",
        )

    entrega = date.fromisoformat(pedido["fecha_entrega_real"])
    dias = (hoy - entrega).days
    base ={"ok": True, "politica": politica, "dias_desde_entrega": dias}

    # 2 y 3. Defectos: la excepción del §3 prevalece sobre cualquier exclusión.
    if motivo in MOTIVOS_DEFECTO:
        plazo = PLAZO_GARANTIA_EXTENDIDA_DIAS if politica == "DEV_GARANTIA_12M" else PLAZO_DEVOLUCION_DIAS
        if dias <= plazo:
            return Evaluacion(
                **base, resultado="ELEGIBLE", via="GARANTIA", paga_retorno="ECOMARKET",
                clausulas=["§3", "§5"],
                detalle="Producto con defecto o error de despacho: aplica la garantía y EcoMarket paga el retorno.",
            )
        return Evaluacion(
            **base, resultado="REVISION_HUMANA", codigo="GARANTIA_FUERA_DE_PLAZO", clausulas=["§3", "§9"],
            detalle=(f"El defecto se reporta {dias} días después de la entrega; la política no cubre "
                     "ese caso y debe revisarlo una persona."),
        )

    # 4. Categorías excluidas por razones sanitarias, de inocuidad o de fabricación.
    if politica and politica.startswith("NO_DEV_"):
        return Evaluacion(
            **base, resultado="NO_ELEGIBLE", codigo="CATEGORIA_EXCLUIDA", clausulas=["§3"],
            alternativa=ALTERNATIVA_GARANTIA,
            detalle=f"El producto pertenece a una categoría sin devolución ({politica}).",
        )

    # 5. Plazo general de 30 días calendario.
    if dias > PLAZO_DEVOLUCION_DIAS:
        return Evaluacion(
            **base, resultado="NO_ELEGIBLE", codigo="FUERA_DE_PLAZO", clausulas=["§1"],
            alternativa=("Si presenta un defecto de fabricación, aplica la garantía de 12 meses."
                         if politica == "DEV_GARANTIA_12M" else None),
            detalle=f"Han pasado {dias} días desde la entrega y el plazo es de {PLAZO_DEVOLUCION_DIAS}.",
        )

    # 7. Cambio de talla: solo textiles, gratis la primera vez por pedido.
    if motivo == "CAMBIO_TALLA":
        if politica != "DEV_TEXTIL":
            return Evaluacion(
                **base, resultado="NO_ELEGIBLE", codigo="MOTIVO_NO_APLICA", clausulas=["§4"],
                alternativa="Se puede tramitar como devolución por arrepentimiento.",
                detalle="El cambio de talla aplica solo a textiles y prendas.",
            )
        gratis = cambios_previos == 0
        return Evaluacion(
            **base, resultado="ELEGIBLE", via="CAMBIO_TALLA",
            paga_retorno="ECOMARKET" if gratis else "CLIENTE",
            costo_retorno_cop=0 if gratis else COSTO_RETORNO_CLIENTE_COP, clausulas=["§4", "§5"],
            detalle="Primer cambio de talla del pedido: sin costo." if gratis
            else "Ya hubo un cambio de talla en este pedido: el retorno lo paga el cliente.",
        )

    # 8 y 9. Retracto (5 días hábiles) o arrepentimiento (hasta el día 30).
    if dias_habiles_entre(entrega, hoy) <= PLAZO_RETRACTO_DIAS_HABILES:
        via, paga, costo = "RETRACTO_5D", "ECOMARKET", 0
        detalle = "Dentro del retracto de 5 días hábiles: reembolso total, incluido el envío original."
    else:
        via, paga, costo = "ARREPENTIMIENTO_30D", "CLIENTE", COSTO_RETORNO_CLIENTE_COP
        tarifa = f"{COSTO_RETORNO_CLIENTE_COP:,}".replace(",", ".")
        detalle = f"Devolución por arrepentimiento: el retorno cuesta COP {tarifa}."
    con_sello = politica == "DEV_CONDICIONAL_SELLO"
    economia = {
        "via": via, "paga_retorno": paga, "costo_retorno_cop": costo,
        "clausulas": ["§1", "§4", "§5"] if con_sello else ["§1", "§5"],
    }

    # 6. Higiene con sello: solo si el sello no se ha roto.
    if con_sello and sello_intacto is None:
        return Evaluacion(
            **base, **economia, resultado="REQUIERE_CONFIRMACION",
            pregunta_pendiente="¿El sello de seguridad del producto está intacto?",
            detalle="Producto de higiene: se acepta solo con el sello de seguridad intacto. " + detalle,
        )
    if con_sello and not sello_intacto:
        return Evaluacion(
            **base, resultado="NO_ELEGIBLE", codigo="SELLO_ROTO", clausulas=["§3", "§4"],
            alternativa=ALTERNATIVA_GARANTIA,
            detalle="Producto de higiene con el sello de seguridad roto: no se puede devolver.",
        )

    return Evaluacion(**base, **economia, resultado="ELEGIBLE", detalle=detalle)
