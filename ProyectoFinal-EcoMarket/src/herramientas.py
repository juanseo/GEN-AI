"""Herramientas del agente: los únicos puntos donde el asistente toca un sistema.

Decisión de arquitectura (Taller 2): el estado de un pedido **no** se resuelve
con búsqueda vectorial. Un pedido es un registro transaccional, exacto y
volátil: la similitud semántica podría devolver el pedido ECO-10002 cuando el
cliente preguntó por el ECO-10012, y eso es un error inaceptable. RAG se reserva
para el conocimiento no estructurado (políticas, catálogo, FAQ) y los pedidos se
consultan con una función determinista expuesta al modelo como herramienta.

Proyecto Final: a `consultar_pedido` se suman las herramientas del agente de
devoluciones. Todas son funciones simuladas sobre los datos del Taller 2 y
siguen las mismas tres reglas:

1. **Nunca lanzan excepciones hacia el modelo.** Un error vuelve como
   `{"ok": false, "codigo": ..., "detalle": ...}`, igual que `consultar_pedido`
   devuelve `encontrado: false`. Así el agente puede explicar el problema en
   lugar de inventar.
2. **El modelo extrae argumentos; el código decide.** La elegibilidad la
   decide `reglas_devolucion.evaluar`, no el LLM.
3. **Dos compuertas en código antes de la única acción con efecto.**
   `generar_etiqueta_devolucion` solo acepta un `id_verificacion` emitido por
   el servidor con resultado ELEGIBLE, en la misma conversación y vigente. Si
   el retorno tiene costo, además exige que el cliente lo haya aceptado en un
   turno *posterior* a aquel en que se le informó: el modelo no puede emitir la
   guía en el mismo turno en que conoce el costo.

| Herramienta                      | Sistema que simula          | Efecto            |
|----------------------------------|-----------------------------|-------------------|
| consultar_pedido                 | OMS (pedidos.json)          | solo lectura      |
| verificar_elegibilidad_producto  | motor de reglas             | solo lectura      |
| generar_etiqueta_devolucion      | transportadora + registro   | crea una guía     |
| escalar_a_humano                 | mesa de ayuda               | crea un ticket    |
| buscar_politica (fábrica)        | índice Chroma del Taller 2  | solo lectura      |
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from .config import DATA_DIR, FECHA_REFERENCIA
from .guardrails import enmascarar_pii
from .reglas_devolucion import FESTIVOS_CO, Evaluacion, evaluar

if TYPE_CHECKING:
    from .rag import SistemaRAG

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


# =============================================================================
# consultar_pedido (Taller 2, sin cambios)
# =============================================================================

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
        "fecha_consulta": FECHA_REFERENCIA,
    }
    for opcional in ("fecha_entrega_real", "fecha_estimada_original",
                     "nota_devolucion", "nota_cancelacion"):
        if opcional in pedido:
            resultado[opcional] = pedido[opcional]
    return resultado


# Herramienta expuesta al LLM. La descripción es parte del prompt: le dice al
# modelo cuándo debe llamarla. El esquema de argumentos (tracking: str) se
# infiere de la firma de `consultar_pedido`.
HERRAMIENTA_CONSULTAR_PEDIDO = StructuredTool.from_function(
    func=consultar_pedido,
    name="consultar_pedido",
    description=(
        "Consulta el sistema de pedidos de EcoMarket y devuelve el estado real de "
        "un pedido a partir de su número de seguimiento (formato ECO-#####). Úsala "
        "SIEMPRE que el cliente pregunte por un pedido; nunca respondas sobre un "
        "pedido sin haber llamado esta función."
    ),
)

FUNCIONES_DISPONIBLES = {"consultar_pedido": consultar_pedido}


# =============================================================================
# Registro simulado de devoluciones (verificaciones, guías y tickets)
# =============================================================================

VIGENCIA_VERIFICACION_S = 30 * 60      # un id_verificacion vale 30 minutos
PLAZO_ENTREGA_GUIA_DIAS_HABILES = 5    # §8, paso 5: entregar el paquete en 5 días hábiles
MAX_FALLAS_TRANSPORTADORA = 2          # al segundo fallo, el agente escala
SLA_HUMANO_HORAS_HABILES = 2           # §9


class RegistroDevoluciones:
    """El "sistema de devoluciones" que las herramientas consultan y modifican.

    Las verificaciones viven en memoria (son efímeras: 30 minutos). Las guías y
    los tickets son el registro de las acciones del agente; si hay `ruta`, se
    persisten en JSON (por defecto `data/devoluciones.json`, que no se versiona).
    El registro también lleva el número de turno de cada conversación (hilo):
    con eso la compuerta del costo exige que la aceptación llegue en un turno
    posterior a aquel en que el cliente conoció el valor.
    """

    def __init__(self, ruta: Path | None = None) -> None:
        self.ruta = ruta
        self.verificaciones: dict[str, dict[str, Any]] = {}
        self.fallas: dict[str, int] = {}
        self.turnos: dict[str, int] = {}
        datos = json.loads(ruta.read_text(encoding="utf-8")) if ruta and ruta.is_file() else {}
        self.guias: list[dict[str, Any]] = datos.get("guias", [])
        self.tickets: list[dict[str, Any]] = datos.get("tickets", [])

    def _guardar(self) -> None:
        if self.ruta:
            contenido = {"guias": self.guias, "tickets": self.tickets}
            self.ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8")

    def vaciar(self) -> None:
        """Borra guías, tickets y verificaciones: vuelve a los datos de prueba sin acciones."""
        self.verificaciones.clear()
        self.fallas.clear()
        self.guias.clear()
        self.tickets.clear()
        self._guardar()

    # -- Turnos de conversación -----------------------------------------------

    def iniciar_turno(self, hilo: str) -> int:
        self.turnos[hilo] = self.turnos.get(hilo, 0) + 1
        return self.turnos[hilo]

    def turno_actual(self, hilo: str | None) -> int | None:
        return self.turnos.get(hilo) if hilo else None

    # -- Verificaciones ---------------------------------------------------------

    def registrar_verificacion(self, tracking: str, sku: str, motivo: str,
                               evaluacion: Evaluacion, hilo: str | None) -> str:
        id_verificacion = f"VER-{tracking.removeprefix('ECO-')}-{secrets.token_hex(2)}"
        self.verificaciones[id_verificacion] = {
            "tracking": tracking, "sku": sku, "motivo": motivo, "evaluacion": evaluacion,
            "hilo": hilo, "turno": self.turno_actual(hilo), "creada": time.monotonic(),
        }
        return id_verificacion

    def verificacion(self, id_verificacion: str) -> dict[str, Any] | None:
        """La verificación si existe y sigue vigente; None si no."""
        registro = self.verificaciones.get((id_verificacion or "").strip())
        if registro and time.monotonic() - registro["creada"] <= VIGENCIA_VERIFICACION_S:
            return registro
        return None

    def cambios_de_talla(self, tracking: str) -> int:
        return sum(1 for g in self.guias if g["tracking"] == tracking and g["via"] == "CAMBIO_TALLA")

    # -- Guías -------------------------------------------------------------------

    def guia_existente(self, tracking: str, sku: str) -> dict[str, Any] | None:
        return next((g for g in self.guias if g["tracking"] == tracking and g["sku"] == sku), None)

    def registrar_falla(self, id_verificacion: str) -> int:
        self.fallas[id_verificacion] = self.fallas.get(id_verificacion, 0) + 1
        return self.fallas[id_verificacion]

    def crear_guia(self, guia: dict[str, Any]) -> dict[str, Any]:
        consecutivo = sum(1 for g in self.guias if g["tracking"] == guia["tracking"]) + 1
        guia = {"numero_guia": f"RET-{guia['tracking']}-{consecutivo:02d}", **guia}
        guia["url_etiqueta"] = f"https://envios.ecomarket.co/{guia['numero_guia']}.pdf"
        self.guias.append(guia)
        self._guardar()
        return guia

    # -- Tickets -----------------------------------------------------------------

    def crear_ticket(self, categoria: str, resumen: str, tracking: str | None) -> dict[str, Any]:
        anio = FECHA_REFERENCIA[:4]
        ticket = {
            "ticket": f"TCK-{anio}-{len(self.tickets) + 1:04d}",
            "categoria": categoria, "resumen": resumen, "tracking": tracking,
            "fecha": FECHA_REFERENCIA,
        }
        self.tickets.append(ticket)
        self._guardar()
        return ticket


REGISTRO = RegistroDevoluciones(DATA_DIR / "devoluciones.json")


def reiniciar_registro(ruta: Path | None = None) -> RegistroDevoluciones:
    """Reemplaza el registro global (las pruebas y la evaluación usan uno en memoria)."""
    global REGISTRO
    REGISTRO = RegistroDevoluciones(ruta)
    return REGISTRO


def _hilo(config: RunnableConfig | None) -> str | None:
    """El thread_id de la conversación; LangChain inyecta `config` en la herramienta."""
    return ((config or {}).get("configurable") or {}).get("thread_id")


def _error(codigo: str, detalle: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "codigo": codigo, "detalle": detalle, **extra}


def _cop(valor: int) -> str:
    return f"COP {valor:,}".replace(",", ".")


# =============================================================================
# verificar_elegibilidad_producto
# =============================================================================

Motivo = Literal["DEFECTUOSO", "EQUIVOCADO", "INCOMPLETO", "VENCIDO", "CAMBIO_TALLA", "ARREPENTIMIENTO"]


class ArgsVerificar(BaseModel):
    # Un argumento desconocido (p. ej. "selo_intacto") es un error que el modelo ve y corrige,
    # no un dato que se pierde en silencio.
    model_config = ConfigDict(extra="forbid")

    tracking: str = Field(description="Número de seguimiento del pedido, formato ECO-#####.")
    motivo: Motivo | None = Field(None, description=(
        "Motivo de la devolución según lo que dice el cliente: ARREPENTIMIENTO (ya no lo "
        "quiere o no lo necesita), DEFECTUOSO (llegó dañado, roto o no funciona), "
        "EQUIVOCADO (llegó otro producto), INCOMPLETO, VENCIDO o CAMBIO_TALLA (solo prendas). "
        "Si el cliente aún no lo ha dicho, omítelo: la herramienta te dirá si hace falta."
    ))
    sku: str | None = Field(None, description=(
        "SKU del producto a devolver, tal como lo devuelve consultar_pedido. Se puede "
        "omitir solo si el pedido tiene un único producto."
    ))
    sello_intacto: bool | None = Field(None, description=(
        "Solo para productos de higiene con sello: true si el cliente confirmó que el "
        "sello está intacto, false si dijo que está roto. Omítelo si el cliente no lo ha dicho."
    ))


def _siguiente_paso(evaluacion: Evaluacion) -> str:
    """Lo que el agente debe hacer con este resultado, en una línea."""
    costo = evaluacion.costo_retorno_cop
    if evaluacion.resultado == "ELEGIBLE" and costo:
        return (f"Informa al cliente que el retorno cuesta {_cop(costo)} y pregúntale si lo acepta. "
                "No generes la guía en este turno: espera su respuesta y solo con un sí explícito "
                "llama generar_etiqueta_devolucion con cliente_acepta_costo=true.")
    if evaluacion.resultado == "ELEGIBLE":
        return "El retorno es gratis: ya puedes llamar generar_etiqueta_devolucion con este id_verificacion."
    if evaluacion.resultado == "REQUIERE_CONFIRMACION":
        aviso = f" Avísale también que el retorno costaría {_cop(costo)}." if costo else ""
        return (f"Pregunta al cliente: «{evaluacion.pregunta_pendiente}».{aviso} Con su respuesta, "
                "vuelve a llamar esta herramienta con sello_intacto.")
    if evaluacion.resultado == "REVISION_HUMANA":
        return ("No decidas tú: llama escalar_a_humano con categoria FUERA_DE_POLITICA y explica "
                "al cliente que una persona revisará su caso.")
    return "Explica el motivo citando la cláusula, ofrece la alternativa si la hay y no generes guía."


def verificar_elegibilidad_producto(
    tracking: str,
    motivo: str | None = None,
    sku: str | None = None,
    sello_intacto: bool | None = None,
    config: RunnableConfig = None,  # LangChain la inyecta solo si el tipo es exactamente RunnableConfig
) -> dict[str, Any]:
    """Aplica la política v3.2 a un producto de un pedido y emite un id_verificacion.

    Si el pedido no se ha entregado, la respuesta es la misma para cualquier
    producto y cualquier motivo (regla 1: se cancela o se espera la entrega), así
    que no se le exige al agente preguntar datos que no cambian el resultado.
    """
    codigo = normalizar_tracking(tracking) or (tracking or "").strip().upper()
    pedido = PEDIDOS.get(codigo)
    if pedido is None:
        return _error("PEDIDO_NO_EXISTE", f"No existe un pedido {codigo}: pide al cliente que verifique el número.")

    productos = [{"sku": i["sku"], "nombre": i["nombre"]} for i in pedido["items"]]
    entregado = pedido["estado"] == "ENTREGADO"
    if not sku:
        if len(productos) != 1 and entregado:
            return _error("SKU_REQUERIDO", "El pedido tiene varios productos: pregunta al cliente cuál quiere devolver.",
                          productos=productos)
        sku = productos[0]["sku"]
    motivo_informado = motivo
    if not motivo:
        if entregado:
            return _error("MOTIVO_REQUERIDO", (
                "El pedido sí fue entregado: pregunta al cliente por qué quiere devolverlo (ya no lo "
                "quiere, llegó dañado, llegó otro producto, incompleto, vencido o cambio de talla)."
            ))
        motivo = "ARREPENTIMIENTO"  # la regla 1 no depende del motivo

    evaluacion = evaluar(pedido, sku, motivo, sello_intacto, cambios_previos=REGISTRO.cambios_de_talla(codigo))
    if not evaluacion.ok:
        extra = {"productos": productos} if evaluacion.codigo == "SKU_NO_PERTENECE" else {}
        return _error(evaluacion.codigo, evaluacion.detalle, **extra)

    sku = sku.strip().upper()
    id_verificacion = REGISTRO.registrar_verificacion(codigo, sku, motivo, evaluacion, _hilo(config))
    respuesta = {
        "ok": True,
        "id_verificacion": id_verificacion,
        "tracking": codigo,
        "sku": sku,
        "producto": next(p["nombre"] for p in productos if p["sku"] == sku),
        "motivo": motivo_informado,
        "resultado": evaluacion.resultado,
        "codigo": evaluacion.codigo,
        "via": evaluacion.via,
        "politica": evaluacion.politica,
        "dias_desde_entrega": evaluacion.dias_desde_entrega,
        "paga_retorno": evaluacion.paga_retorno,
        "costo_retorno_cop": evaluacion.costo_retorno_cop,
        "clausulas": evaluacion.clausulas,
        "detalle": evaluacion.detalle,
        "alternativa": evaluacion.alternativa,
        "pregunta_pendiente": evaluacion.pregunta_pendiente,
        "siguiente_paso": _siguiente_paso(evaluacion),
    }
    return {clave: valor for clave, valor in respuesta.items() if valor is not None}


HERRAMIENTA_VERIFICAR_ELEGIBILIDAD = StructuredTool.from_function(
    func=verificar_elegibilidad_producto,
    name="verificar_elegibilidad_producto",
    args_schema=ArgsVerificar,
    description=(
        "Decide si un producto de un pedido se puede devolver, aplicando la política de "
        "devoluciones vigente. Llámala SIEMPRE antes de decirle al cliente si puede o no "
        "devolver algo: tú no decides la elegibilidad. Devuelve el resultado (ELEGIBLE, "
        "NO_ELEGIBLE, REQUIERE_CONFIRMACION o REVISION_HUMANA), quién paga el retorno, las "
        "cláusulas que lo sustentan, el siguiente paso y un id_verificacion."
    ),
)


# =============================================================================
# generar_etiqueta_devolucion
# =============================================================================

class ArgsGenerar(BaseModel):
    # Un argumento desconocido (p. ej. "selo_intacto") es un error que el modelo ve y corrige,
    # no un dato que se pierde en silencio.
    model_config = ConfigDict(extra="forbid")

    id_verificacion: str = Field(description=(
        "El id_verificacion que devolvió verificar_elegibilidad_producto con resultado ELEGIBLE."
    ))
    cliente_acepta_costo: bool = Field(False, description=(
        "true solo si el cliente aceptó de forma explícita pagar el costo del retorno. "
        "Si el retorno es gratis, déjalo en false."
    ))


def generar_etiqueta_devolucion(
    id_verificacion: str,
    cliente_acepta_costo: bool = False,
    config: RunnableConfig = None,  # LangChain la inyecta solo si el tipo es exactamente RunnableConfig
) -> dict[str, Any]:
    """Crea la guía de retorno con la transportadora del pedido y la registra."""
    hilo = _hilo(config)
    verificacion = REGISTRO.verificacion(id_verificacion)
    if verificacion is None or (hilo and verificacion["hilo"] and verificacion["hilo"] != hilo):
        return _error("VERIFICACION_INVALIDA", (
            "Ese id_verificacion no existe, venció o no pertenece a esta conversación. "
            "Vuelve a verificar la elegibilidad con verificar_elegibilidad_producto."
        ))
    evaluacion: Evaluacion = verificacion["evaluacion"]
    if evaluacion.resultado != "ELEGIBLE":
        return _error("VERIFICACION_INVALIDA",
                      f"La verificación dio {evaluacion.resultado}: no se puede generar una guía.")

    tracking, sku = verificacion["tracking"], verificacion["sku"]
    if previa := REGISTRO.guia_existente(tracking, sku):
        return {"ok": True, "codigo": "YA_EXISTE",
                "detalle": "Ya existía una guía para este producto; no se creó otra.",
                **{k: v for k, v in previa.items() if k != "id_verificacion"}}

    costo = evaluacion.costo_retorno_cop
    if costo:
        turno = REGISTRO.turno_actual(hilo)
        informado_antes = turno is None or verificacion["turno"] is None or verificacion["turno"] < turno
        if not cliente_acepta_costo or not informado_antes:
            return _error("CONFIRMACION_FALTANTE", (
                f"El retorno cuesta {_cop(costo)}. Infórmalo y espera a que el cliente lo acepte "
                "en su siguiente mensaje; solo entonces llama de nuevo con cliente_acepta_costo=true."
            ), costo_retorno_cop=costo)

    if os.environ.get("ECOMARKET_SIMULAR_FALLA") == "1":
        intentos = REGISTRO.registrar_falla(id_verificacion)
        paso = ("Puedes reintentar una vez." if intentos < MAX_FALLAS_TRANSPORTADORA else
                "No reintentes: llama escalar_a_humano con categoria FALLA_SISTEMA y discúlpate con el cliente.")
        return _error("TRANSPORTADORA_NO_DISPONIBLE", "El sistema de la transportadora no respondió.",
                      intentos_fallidos=intentos, siguiente_paso=paso)

    pedido = PEDIDOS[tracking]
    hoy = date.fromisoformat(FECHA_REFERENCIA)
    limite = str(np.busday_offset(hoy, PLAZO_ENTREGA_GUIA_DIAS_HABILES, roll="forward", holidays=FESTIVOS_CO))
    guia = REGISTRO.crear_guia({
        "tracking": tracking,
        "sku": sku,
        "producto": next(i["nombre"] for i in pedido["items"] if i["sku"] == sku),
        "via": evaluacion.via,
        "transportadora": pedido["transportadora"],
        "costo_cliente_cop": costo,
        "entregar_antes_de": limite,
        "instrucciones": (
            "Empaca el producto en su empaque original y entrégalo en cualquier punto de la "
            "transportadora. El reembolso se procesa después de la inspección en bodega "
            "(hasta 3 días hábiles desde que llega)."
        ),
        "fecha_emision": FECHA_REFERENCIA,
        "id_verificacion": id_verificacion,
    })
    return {"ok": True, **{k: v for k, v in guia.items() if k != "id_verificacion"}}


HERRAMIENTA_GENERAR_ETIQUETA = StructuredTool.from_function(
    func=generar_etiqueta_devolucion,
    name="generar_etiqueta_devolucion",
    args_schema=ArgsGenerar,
    description=(
        "Genera la guía de retorno de un producto. Solo funciona con un id_verificacion "
        "cuyo resultado fue ELEGIBLE. Si el retorno tiene costo, úsala únicamente después "
        "de que el cliente lo haya aceptado de forma explícita. Devuelve el número de guía, "
        "la transportadora, la fecha límite de entrega y las instrucciones."
    ),
)


# =============================================================================
# escalar_a_humano
# =============================================================================

class ArgsEscalar(BaseModel):
    # Un argumento desconocido (p. ej. "selo_intacto") es un error que el modelo ve y corrige,
    # no un dato que se pierde en silencio.
    model_config = ConfigDict(extra="forbid")

    categoria: Literal["SALUD", "DISPUTA", "FUERA_DE_POLITICA", "SOLICITUD_CLIENTE", "FALLA_SISTEMA"] = Field(
        description="SALUD, DISPUTA, FUERA_DE_POLITICA (caso no cubierto o revisión humana), "
                    "SOLICITUD_CLIENTE (pidió una persona) o FALLA_SISTEMA (una herramienta falló dos veces)."
    )
    resumen: str = Field(description="Resumen breve del caso para la persona que lo atenderá, sin datos personales.")
    tracking: str | None = Field(None, description="Número de seguimiento del pedido, si lo hay.")


def escalar_a_humano(categoria: str, resumen: str, tracking: str | None = None) -> dict[str, Any]:
    """Abre un ticket en la mesa de ayuda con el resumen del caso."""
    resumen_limpio, _ = enmascarar_pii(resumen)   # defensa en profundidad: el ticket no guarda PII
    ticket = REGISTRO.crear_ticket(categoria, resumen_limpio, normalizar_tracking(tracking or ""))
    return {
        "ok": True,
        "ticket": ticket["ticket"],
        "sla_horas_habiles": SLA_HUMANO_HORAS_HABILES,
        "detalle": "Una persona del equipo contactará al cliente en máximo 2 horas hábiles "
                   "(lunes a viernes, 8:00 a 18:00).",
    }


HERRAMIENTA_ESCALAR = StructuredTool.from_function(
    func=escalar_a_humano,
    name="escalar_a_humano",
    args_schema=ArgsEscalar,
    description=(
        "Abre un ticket para que una persona del equipo atienda el caso. Úsala cuando la "
        "verificación dé REVISION_HUMANA, cuando una herramienta falle dos veces o cuando el "
        "cliente pida hablar con una persona. Devuelve el número de ticket y el tiempo de respuesta."
    ),
)


# =============================================================================
# buscar_politica: el RAG del Taller 2 como herramienta
# =============================================================================

def crear_herramienta_buscar_politica(rag: SistemaRAG) -> StructuredTool:
    """Envuelve el retriever del Taller 2 (mismo índice, mismo umbral) como herramienta.

    Es una fábrica porque necesita el `SistemaRAG` abierto por el asistente: el
    índice de Chroma se abre una sola vez por proceso.
    """
    from .rag import supera_umbral   # import diferido: las pruebas de herramientas no abren Chroma

    def buscar_politica(consulta: str) -> dict[str, Any]:
        recuperados = rag.recuperar(consulta)
        if not supera_umbral(recuperados, rag.umbral):
            return {"en_alcance": False,
                    "detalle": "No hay información sobre eso en las fuentes de EcoMarket. "
                               "No respondas con conocimiento general."}
        recuperados = rag.expandir_con_politica(consulta, recuperados)
        return {
            "en_alcance": True,
            "fragmentos": [{"id": r.id, "similitud": round(r.score, 2), "texto": r.documento.page_content}
                           for r in recuperados],
        }

    return StructuredTool.from_function(
        func=buscar_politica,
        name="buscar_politica",
        description=(
            "Busca en las fuentes oficiales de EcoMarket (política de devoluciones, catálogo y "
            "preguntas frecuentes). Úsala para preguntas informativas (plazos, reembolsos, costos, "
            "condiciones) y para citar la cláusula exacta. No sirve para consultar pedidos."
        ),
    )


HERRAMIENTAS_TRANSACCIONALES = [
    HERRAMIENTA_CONSULTAR_PEDIDO,
    HERRAMIENTA_VERIFICAR_ELEGIBILIDAD,
    HERRAMIENTA_GENERAR_ETIQUETA,
    HERRAMIENTA_ESCALAR,
]
