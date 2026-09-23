"""Cadena de orquestación del asistente de EcoMarket.

Flujo de una consulta en modo optimizado:

    mensaje del cliente
        -> minimización de PII
        -> guardrail de escalamiento (reglas deterministas)
        -> router de intención (LLM pequeño, 1 token de salida)
        -> rama especializada:
             ESTADO_PEDIDO  -> function calling sobre el sistema de pedidos
             DEVOLUCION     -> RAG (híbrido + re-ranking) + árbol de decisión
             resto          -> RAG general
        -> verificación de anclaje de la respuesta
        -> respuesta al cliente

El modo básico existe solo para comparación: envía el prompt ingenuo sin
contexto, sin herramientas y sin controles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .config import CONFIG
from .guardrails import (
    MENSAJE_ESCALAMIENTO,
    enmascarar_pii,
    evaluar_escalamiento,
    verificar_anclaje,
)
from .herramientas import (
    ESQUEMA_HERRAMIENTAS,
    FUNCIONES_DISPONIBLES,
    consultar_pedido,
    normalizar_tracking,
)
from .llm_client import ClienteLLM
from .retriever import BaseConocimiento, Resultado

INTENCIONES_VALIDAS = {
    "ESTADO_PEDIDO", "DEVOLUCION", "PRODUCTO", "ENVIO", "QUEJA", "OTRO"
}


@dataclass
class Respuesta:
    """Resultado de una consulta, con todo lo necesario para auditarla."""

    texto: str
    intencion: str = "OTRO"
    modo: str = "optimizado"
    escalada: bool = False
    motivo_escalamiento: str | None = None
    pii_detectada: list[str] = field(default_factory=list)
    fragmentos: list[Resultado] = field(default_factory=list)
    datos_pedido: dict[str, Any] | None = None
    anclaje: dict[str, Any] | None = None
    llamadas_llm: int = 0

    @property
    def fuentes(self) -> list[str]:
        return [r.fragmento.cita for r in self.fragmentos]


class AsistenteEcoMarket:
    """Asistente de atención al cliente con RAG + function calling."""

    def __init__(self, cliente: ClienteLLM | None = None) -> None:
        self.cliente = cliente or ClienteLLM()
        # La base de conocimiento se construye una sola vez por proceso: el
        # chunking y la vectorización son el costo fijo del arranque.
        self.base = BaseConocimiento(cliente_llm=self.cliente)

    # -- Router -------------------------------------------------------------

    def clasificar(self, consulta: str) -> str:
        """Clasifica la intención. Ante cualquier duda, devuelve OTRO."""
        etiqueta = self.cliente.texto(
            [{"role": "user", "content": prompts.ROUTER.format(consulta=consulta)}],
            temperatura=0.0,
            max_tokens=8,
        )
        etiqueta = etiqueta.strip().upper().strip(".")
        return etiqueta if etiqueta in INTENCIONES_VALIDAS else "OTRO"

    # -- Ramas --------------------------------------------------------------

    def _rama_pedido(self, consulta: str, respuesta: Respuesta) -> str:
        """Resuelve el estado de un pedido usando function calling."""
        mensajes = [
            {"role": "system", "content": prompts.SISTEMA_OPTIMIZADO},
            {"role": "user", "content": consulta},
        ]
        mensaje_modelo = self.cliente.completar(
            mensajes, herramientas=ESQUEMA_HERRAMIENTAS, max_tokens=200
        )
        respuesta.llamadas_llm += 1

        datos: dict[str, Any] | None = None
        for llamada in getattr(mensaje_modelo, "tool_calls", None) or []:
            funcion = FUNCIONES_DISPONIBLES.get(llamada.function.name)
            if not funcion:
                continue
            argumentos = json.loads(llamada.function.arguments or "{}")
            datos = funcion(**argumentos)
            break

        # Respaldo: si el modelo no invocó la herramienta pero el mensaje trae
        # un número de seguimiento, lo resolvemos nosotros. Nunca dejamos que
        # el modelo conteste sobre un pedido sin datos verificados.
        if datos is None:
            tracking = normalizar_tracking(consulta)
            if tracking:
                datos = consultar_pedido(tracking)

        if datos is None:
            return (
                "Con gusto reviso tu pedido. ¿Me compartes el número de "
                "seguimiento? Tiene el formato ECO-#####  y lo encuentras en el "
                "correo de confirmación de tu compra."
            )

        respuesta.datos_pedido = datos
        prompt = prompts.E1_OPTIMIZADO.format(
            consulta=consulta,
            datos_pedido=json.dumps(datos, ensure_ascii=False, indent=2),
        )
        respuesta.llamadas_llm += 1
        return self.cliente.texto(
            [
                {"role": "system", "content": prompts.SISTEMA_OPTIMIZADO},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
        )

    def _rama_devolucion(self, consulta: str, respuesta: Respuesta) -> str:
        """Resuelve una devolución combinando RAG con los datos del pedido."""
        # Si el cliente dio el número de seguimiento, los productos del pedido
        # enriquecen la consulta de recuperación: así el retriever encuentra la
        # ficha del SKU correcto y no solo la política genérica.
        datos: dict[str, Any] = {}
        tracking = normalizar_tracking(consulta)
        consulta_retrieval = consulta
        if tracking:
            datos = consultar_pedido(tracking)
            if datos.get("encontrado"):
                respuesta.datos_pedido = datos
                nombres = " ".join(p["nombre"] for p in datos["productos"])
                consulta_retrieval = f"{consulta} {nombres} devolucion politica"

        # Expansión de consulta: garantiza que la sección de la política que
        # sustenta la exclusión llegue al contexto, aunque el cliente no haya
        # usado el vocabulario del documento.
        respuesta.fragmentos = self.base.buscar_con_expansion(consulta_retrieval)
        contexto = self.base.construir_contexto(respuesta.fragmentos)

        prompt = prompts.E2_OPTIMIZADO.format(
            consulta=consulta,
            contexto=contexto,
            datos_pedido=json.dumps(datos, ensure_ascii=False, indent=2) if datos else "{}",
        )
        respuesta.llamadas_llm += 1
        return self.cliente.texto(
            [
                {"role": "system", "content": prompts.SISTEMA_OPTIMIZADO},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )

    def _rama_general(self, consulta: str, respuesta: Respuesta) -> str:
        """Consultas de producto, envíos y otros: RAG estándar."""
        respuesta.fragmentos = self.base.buscar(consulta)
        contexto = self.base.construir_contexto(respuesta.fragmentos)
        prompt = prompts.RAG_GENERAL.format(consulta=consulta, contexto=contexto)
        respuesta.llamadas_llm += 1
        return self.cliente.texto(
            [
                {"role": "system", "content": prompts.SISTEMA_OPTIMIZADO},
                {"role": "user", "content": prompt},
            ],
            max_tokens=450,
        )

    # -- Punto de entrada ---------------------------------------------------

    def responder(self, consulta: str, modo: str = "optimizado") -> Respuesta:
        """Atiende una consulta. `modo` es 'optimizado' o 'basico'."""
        respuesta = Respuesta(texto="", modo=modo)

        if modo == "basico":
            # Sin PII masking, sin router, sin RAG, sin guardrails: exactamente
            # lo que se obtiene al escribir el primer prompt que a uno se le
            # ocurre. Es el punto de comparación de la Fase 3.
            tracking = normalizar_tracking(consulta) or "12345"
            es_devolucion = any(
                palabra in consulta.lower()
                for palabra in ("devolver", "devolución", "devolucion", "cambio", "reembolso")
            )
            plantilla = prompts.E2_BASICO if es_devolucion else prompts.E1_BASICO
            prompt = plantilla.format(consulta=consulta, tracking=tracking)
            respuesta.llamadas_llm = 1
            respuesta.texto = self.cliente.texto(
                [
                    {"role": "system", "content": prompts.SISTEMA_BASICO},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
            )
            return respuesta

        # 1. Minimización de PII antes de que el texto salga hacia el proveedor.
        consulta_limpia, pii = enmascarar_pii(consulta)
        respuesta.pii_detectada = pii

        # 2. Guardrail de escalamiento: por reglas, no por criterio del modelo.
        decision = evaluar_escalamiento(consulta_limpia)
        if decision.escalar:
            respuesta.escalada = True
            respuesta.intencion = "QUEJA"
            respuesta.motivo_escalamiento = f"[{decision.categoria}] {decision.motivo}"
            respuesta.texto = MENSAJE_ESCALAMIENTO
            return respuesta

        # 3. Router de intención.
        respuesta.intencion = self.clasificar(consulta_limpia)
        respuesta.llamadas_llm += 1

        # 4. Rama especializada.
        if respuesta.intencion == "ESTADO_PEDIDO":
            respuesta.texto = self._rama_pedido(consulta_limpia, respuesta)
        elif respuesta.intencion == "DEVOLUCION":
            respuesta.texto = self._rama_devolucion(consulta_limpia, respuesta)
        elif respuesta.intencion == "QUEJA":
            respuesta.escalada = True
            respuesta.motivo_escalamiento = "El router clasificó la consulta como queja."
            respuesta.texto = MENSAJE_ESCALAMIENTO
            return respuesta
        else:
            respuesta.texto = self._rama_general(consulta_limpia, respuesta)

        # 5. Verificación de anclaje contra el material realmente disponible.
        contexto_total = self.base.construir_contexto(respuesta.fragmentos)
        if respuesta.datos_pedido:
            contexto_total += "\n" + json.dumps(
                respuesta.datos_pedido, ensure_ascii=False
            )
        respuesta.anclaje = verificar_anclaje(respuesta.texto, contexto_total)

        return respuesta

    @property
    def costo_sesion_usd(self) -> float:
        return self.cliente.uso.costo_estimado_usd()
