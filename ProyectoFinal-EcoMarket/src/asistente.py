"""Cadena completa del asistente de EcoMarket: Taller 1 + RAG del Taller 2 + agente del Proyecto Final.

Flujo de una consulta (modo "rag"):

    mensaje del cliente
        -> minimización de PII                      (guardrails.py, regex)
        -> ¿escalar ya? reglas deterministas        (guardrails.py, 0 tokens)
        -> ¿el hilo ya está con el agente?          (ruta pegajosa, 0 tokens)
        -> router de intención                      (LLM, temperatura 0)
        -> ruta:
             QUEJA                     -> agente humano
             ESTADO_PEDIDO, DEVOLUCION -> agente de devoluciones (agente.py)
             o mensaje con ECO-#####      si falla: rama del Taller 2, sin acciones
             resto                     -> RAG general (con umbral de fuera de alcance)
        -> verificación de anclaje de la respuesta  (guardrails.py)

Arquitectura router + agente (Fase 1): el router del Taller 2 se conserva como
puerta de entrada porque cuesta un llamado corto y su evaluación sigue valiendo.
Las preguntas informativas siguen yendo a la cadena RAG; solo la ruta del agente
puede ejecutar acciones. Dentro del agente el mismo índice aparece como la
herramienta `buscar_politica`.

Ruta pegajosa: el router clasifica cada turno por separado, y un «sí, el sello
está intacto» saldría como OTRO. Por eso, una vez que una conversación (`hilo`)
entra al agente, sus turnos siguientes van directo a él. Los controles
deterministas (PII y escalamiento por reglas) se aplican siempre antes.

Respaldo: si el agente falla (proveedor caído, límite de pasos, respuesta vacía),
responde la rama del Taller 2 correspondiente, que informa pero no ejecuta
acciones. El cliente nunca queda sin respuesta.

El modo "sin_rag" existe solo para comparar: mismo modelo, sin contexto.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from . import prompts
from .agente import AgenteDevoluciones, PasoHerramienta
from .guardrails import MENSAJE_ESCALAMIENTO, enmascarar_pii, evaluar_escalamiento, verificar_anclaje
from .herramientas import FUNCIONES_DISPONIBLES, HERRAMIENTA_CONSULTAR_PEDIDO, consultar_pedido, normalizar_tracking
from .llm import crear_llm, resumen_uso
from .rag import Recuperado, SistemaRAG, formatear_contexto, supera_umbral

INTENCIONES_VALIDAS = {"ESTADO_PEDIDO", "DEVOLUCION", "PRODUCTO", "ENVIO", "PAGO_CUENTA", "QUEJA", "OTRO"}
INTENCIONES_DEL_AGENTE = {"ESTADO_PEDIDO", "DEVOLUCION"}


@dataclass
class Respuesta:
    """Resultado de una consulta, con todo lo necesario para auditarla."""

    texto: str
    modo: str = "rag"
    intencion: str = "OTRO"
    escalada: bool = False
    motivo_escalamiento: str | None = None
    en_alcance: bool = True
    pii_detectada: list[str] = field(default_factory=list)
    recuperados: list[Recuperado] = field(default_factory=list)
    datos_pedido: dict[str, Any] | None = None
    herramienta_invocada: bool = False
    anclaje: dict[str, Any] | None = None
    llamadas_llm: int = 0
    # Proyecto Final: por dónde pasó la consulta y qué hizo el agente.
    ruta: str = "rag"                      # rag | agente | respaldo | humano | sin_rag
    hilo: str | None = None
    herramientas: list[PasoHerramienta] = field(default_factory=list)
    error_agente: str | None = None

    @property
    def fuentes(self) -> list[str]:
        return [f"{r.id} ({r.score:.2f})" for r in self.recuperados]


class AsistenteEcoMarket:
    """Asistente de atención al cliente: router + RAG + agente de devoluciones + guardrails."""

    def __init__(self, rag: SistemaRAG | None = None, con_agente: bool = True) -> None:
        self.uso = UsageMetadataCallbackHandler()
        self._config = {"callbacks": [self.uso]}
        self.llm = crear_llm()
        # La base vectorial se abre una vez por proceso: cargar el índice es el
        # costo fijo del arranque, no de cada consulta.
        self.rag = rag or SistemaRAG(llm=self.llm)
        # con_agente=False deja el comportamiento exacto del Taller 2 (para comparar).
        self.agente = AgenteDevoluciones(self.rag, callbacks=[self.uso]) if con_agente else None
        self._hilos_con_agente: set[str] = set()

        # Router: una etiqueta, temperatura 0, pocos tokens.
        self.router = (
            ChatPromptTemplate.from_template(prompts.ROUTER)
            | crear_llm(temperatura=0.0, max_tokens=8)
            | StrOutputParser()
        )
        # Function calling: el modelo decide si llama a consultar_pedido y con qué argumento.
        self.llm_con_herramientas = self.llm.bind_tools([HERRAMIENTA_CONSULTAR_PEDIDO])

    def _generar(self, plantilla: str, sistema: str = prompts.SISTEMA, **variables: str) -> str:
        cadena = ChatPromptTemplate.from_messages([("system", sistema), ("human", plantilla)]) | self.llm | StrOutputParser()
        return cadena.invoke(variables, config=self._config)

    # -- Router ---------------------------------------------------------------

    def clasificar(self, consulta: str) -> str:
        """Clasifica la intención. Ante cualquier duda, devuelve OTRO."""
        etiqueta = self.router.invoke({"consulta": consulta}, config=self._config)
        etiqueta = etiqueta.strip().upper().strip(".")
        return etiqueta if etiqueta in INTENCIONES_VALIDAS else "OTRO"

    # -- Ramas ----------------------------------------------------------------

    def _rama_pedido(self, consulta: str, respuesta: Respuesta) -> str:
        mensaje = self.llm_con_herramientas.invoke(
            [("system", prompts.SISTEMA), ("human", consulta)], config=self._config
        )
        respuesta.llamadas_llm += 1

        datos: dict[str, Any] | None = None
        for llamada in mensaje.tool_calls:
            funcion = FUNCIONES_DISPONIBLES.get(llamada["name"])
            if funcion:
                datos = funcion(**llamada["args"])
                respuesta.herramienta_invocada = True
                break

        # Respaldo determinista: si el modelo no invocó la herramienta pero el
        # mensaje trae un número de seguimiento, lo consultamos nosotros. Nunca
        # se responde sobre un pedido sin datos verificados.
        if datos is None and (tracking := normalizar_tracking(consulta)):
            datos = consultar_pedido(tracking)
        if datos is None:
            return (
                "Con gusto reviso tu pedido. ¿Me compartes el número de seguimiento? "
                "Tiene el formato ECO-##### y lo encuentras en el correo de confirmación."
            )

        respuesta.datos_pedido = datos
        respuesta.llamadas_llm += 1
        return self._generar(
            prompts.PEDIDO, consulta=consulta, datos_pedido=json.dumps(datos, ensure_ascii=False, indent=2)
        )

    def _rama_devolucion(self, consulta: str, respuesta: Respuesta) -> str:
        # Si el cliente dio el número de seguimiento, el nombre de los productos
        # del pedido enriquece la consulta: el retriever encuentra la ficha del
        # SKU correcto y no solo la política genérica.
        datos: dict[str, Any] = {}
        consulta_retrieval = consulta
        if tracking := normalizar_tracking(consulta):
            datos = consultar_pedido(tracking)
            if datos.get("encontrado"):
                respuesta.datos_pedido = datos
                nombres = ", ".join(p["nombre"] for p in datos["productos"])
                consulta_retrieval = f"{consulta} (productos: {nombres})"

        recuperados = self.rag.recuperar(consulta_retrieval)
        if not datos.get("encontrado") and not supera_umbral(recuperados, self.rag.umbral):
            # Sin pedido verificado y sin nada parecido en la base: no hay con qué decidir.
            respuesta.recuperados = recuperados
            respuesta.en_alcance = False
            return prompts.MENSAJE_FUERA_DE_ALCANCE
        respuesta.recuperados = self.rag.expandir_con_politica(consulta_retrieval, recuperados)
        respuesta.llamadas_llm += 1
        return self._generar(
            prompts.DEVOLUCION,
            consulta=consulta,
            contexto=formatear_contexto(respuesta.recuperados),
            datos_pedido=json.dumps(datos, ensure_ascii=False, indent=2) if datos else "{}",
        )

    def _rama_general(self, consulta: str, respuesta: Respuesta) -> str:
        respuesta.recuperados = self.rag.recuperar(consulta)
        if not supera_umbral(respuesta.recuperados, self.rag.umbral):
            # Nada en la base se parece a la consulta: se reconoce sin llamar al LLM.
            respuesta.en_alcance = False
            return prompts.MENSAJE_FUERA_DE_ALCANCE
        respuesta.llamadas_llm += 1
        return self._generar(
            prompts.RAG_GENERAL, consulta=consulta, contexto=formatear_contexto(respuesta.recuperados)
        )

    def _ruta_agente(self, consulta: str, respuesta: Respuesta) -> str:
        """Un turno del agente de devoluciones; si falla, la rama del Taller 2 responde."""
        respuesta.ruta = "agente"
        try:
            turno = self.agente.responder(consulta, respuesta.hilo)
        except Exception as error:  # noqa: BLE001 — cualquier falla del agente activa el respaldo
            respuesta.ruta = "respaldo"
            respuesta.error_agente = f"{type(error).__name__}: {error}"[:300]
            self._hilos_con_agente.discard(respuesta.hilo)
            if respuesta.intencion == "ESTADO_PEDIDO":
                return self._rama_pedido(consulta, respuesta)
            return self._rama_devolucion(consulta, respuesta)

        self._hilos_con_agente.add(respuesta.hilo)
        respuesta.herramientas = turno.pasos
        respuesta.herramienta_invocada = bool(turno.pasos)
        respuesta.llamadas_llm += turno.llamadas_llm
        for paso in turno.pasos:
            if paso.nombre == "consultar_pedido" and isinstance(paso.resultado, dict):
                respuesta.datos_pedido = paso.resultado
            if paso.nombre == "escalar_a_humano" and paso.ok:
                respuesta.escalada = True
                respuesta.motivo_escalamiento = (
                    f"[agente:{paso.argumentos.get('categoria')}] ticket {paso.resultado.get('ticket')}"
                )
        return turno.texto

    # -- Punto de entrada -----------------------------------------------------

    def responder(self, consulta: str, modo: str = "rag", hilo: str | None = None) -> Respuesta:
        """Atiende una consulta. `modo` es 'rag' (el asistente) o 'sin_rag' (comparación).

        `hilo` identifica la conversación: con el mismo hilo, el agente recuerda
        los turnos anteriores. Sin hilo, cada consulta es independiente.
        """
        respuesta = Respuesta(texto="", modo=modo, hilo=hilo or f"consulta-{uuid4().hex[:8]}")

        if modo == "sin_rag":
            # El mismo modelo con un rol, pero sin acceso a la base de
            # conocimiento: el "modelo de propósito general" del enunciado.
            respuesta.ruta = "sin_rag"
            respuesta.llamadas_llm = 1
            respuesta.texto = self._generar(prompts.SIN_RAG, sistema=prompts.SISTEMA_SIN_RAG, consulta=consulta)
            return respuesta

        # 1. Minimización de PII antes de que el texto salga hacia el proveedor.
        consulta_limpia, respuesta.pii_detectada = enmascarar_pii(consulta)

        # 2. Escalamiento por reglas: no depende del criterio del modelo.
        decision = evaluar_escalamiento(consulta_limpia)
        if decision.escalar:
            respuesta.escalada = True
            respuesta.ruta = "humano"
            respuesta.intencion = "QUEJA"
            respuesta.motivo_escalamiento = f"[{decision.categoria}] {decision.motivo}"
            respuesta.texto = MENSAJE_ESCALAMIENTO
            return respuesta

        # 3. Ruta pegajosa: la conversación ya está con el agente, sin pasar por el router.
        if self.agente and respuesta.hilo in self._hilos_con_agente:
            respuesta.intencion = "CONTINUACION"
            respuesta.texto = self._ruta_agente(consulta_limpia, respuesta)
        else:
            # 4. Router de intención.
            respuesta.intencion = self.clasificar(consulta_limpia)
            respuesta.llamadas_llm += 1

            # 5. Ruta: agente (con acciones) o ramas informativas del Taller 2.
            #    Un número de seguimiento en el mensaje lleva al agente aunque el
            #    router diga otra cosa: el router es un LLM y en la evaluación
            #    clasificó el mismo mensaje como DEVOLUCION en una corrida y OTRO en
            #    otra. Un mensaje sobre un pedido concreto necesita las herramientas.
            sobre_un_pedido = normalizar_tracking(consulta_limpia) is not None
            if respuesta.intencion == "QUEJA":
                respuesta.escalada = True
                respuesta.ruta = "humano"
                respuesta.motivo_escalamiento = "El router clasificó la consulta como queja."
                respuesta.texto = MENSAJE_ESCALAMIENTO
                return respuesta
            if self.agente and (respuesta.intencion in INTENCIONES_DEL_AGENTE or sobre_un_pedido):
                respuesta.texto = self._ruta_agente(consulta_limpia, respuesta)
            elif respuesta.intencion == "ESTADO_PEDIDO":
                respuesta.texto = self._rama_pedido(consulta_limpia, respuesta)
            elif respuesta.intencion == "DEVOLUCION":
                respuesta.texto = self._rama_devolucion(consulta_limpia, respuesta)
            else:
                respuesta.texto = self._rama_general(consulta_limpia, respuesta)

        # 6. Verificación de anclaje contra el material realmente disponible:
        #    fragmentos recuperados, datos del pedido y resultados de las herramientas.
        if respuesta.en_alcance:
            contexto = formatear_contexto(respuesta.recuperados)
            if respuesta.datos_pedido:
                contexto += "\n" + json.dumps(respuesta.datos_pedido, ensure_ascii=False)
            for paso in respuesta.herramientas:
                contexto += "\n" + json.dumps(paso.resultado, ensure_ascii=False)
            respuesta.anclaje = verificar_anclaje(respuesta.texto, contexto)
        return respuesta

    @property
    def uso_sesion(self) -> dict:
        return resumen_uso(self.uso)
