"""El agente de devoluciones: `create_agent` de LangChain sobre las herramientas del proyecto.

Por qué LangChain (Fase 1): todo el Taller 2 ya está escrito en LangChain 1.x
(retriever sobre Chroma, cadenas LCEL, `StructuredTool`, `ChatOpenAI`). Con
`create_agent` el agente reutiliza esas piezas tal como están: el mismo cliente
LLM de `llm.py`, la misma herramienta `consultar_pedido` y el mismo índice,
ahora envuelto como la herramienta `buscar_politica`.

Cómo funciona un turno:

    mensaje del cliente (ya sin PII)
        -> modelo con 5 herramientas  ──► ¿pide herramientas? ─ sí ─► ejecutarlas
                ▲                                                       │
                └──────────────── resultados (JSON) ◄───────────────────┘
        -> no pide más: ese texto es la respuesta del turno

`create_agent` arma ese bucle (ReAct) como un grafo de LangGraph. Tres
decisiones de diseño:

- **Memoria por conversación.** `InMemorySaver` guarda los mensajes por
  `thread_id`: una devolución puede tomar varios turnos (sello, costo, sí).
- **Límite de pasos.** `recursion_limit` corta un agente en bucle; si se
  alcanza, el asistente responde con la rama del Taller 2 (ver `asistente.py`).
- **Traza auditable.** Cada turno devuelve las herramientas llamadas, con sus
  argumentos y resultados, para la interfaz, la evaluación y el monitoreo.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from . import herramientas, prompts
from .llm import crear_llm

if TYPE_CHECKING:
    from .rag import SistemaRAG

# Un turno normal usa 2-3 rondas de herramientas (5-7 pasos del grafo). El caso
# más largo previsto (consultar, verificar, generar, reintentar y escalar) usa
# 11. Con 16 queda margen sin permitir bucles largos.
LIMITE_PASOS = 16

# Herramientas con efecto en otro sistema. Las demás solo leen.
HERRAMIENTAS_DE_ACCION = {"generar_etiqueta_devolucion", "escalar_a_humano"}


@dataclass
class PasoHerramienta:
    """Una llamada a herramienta dentro de un turno: qué pidió el modelo y qué obtuvo."""

    nombre: str
    argumentos: dict[str, Any]
    resultado: Any  # dict si la herramienta respondió JSON; texto si el modelo envió argumentos inválidos

    @property
    def ok(self) -> bool:
        return isinstance(self.resultado, dict) and self.resultado.get("ok", True) is not False

    @property
    def resumen(self) -> str:
        """Una línea para la traza: el dato clave del resultado."""
        if not isinstance(self.resultado, dict):
            return str(self.resultado)[:120]
        r = self.resultado
        for clave in ("numero_guia", "ticket", "resultado", "codigo", "estado"):
            if clave in r:
                return f"{clave}={r[clave]}"
        if "en_alcance" in r:
            return f"en_alcance={r['en_alcance']}, fragmentos={len(r.get('fragmentos', []))}"
        if r.get("encontrado") is False:
            return "encontrado=False"
        return "ok"


@dataclass
class TurnoAgente:
    texto: str
    pasos: list[PasoHerramienta] = field(default_factory=list)
    llamadas_llm: int = 0


def _extraer_pasos(mensajes: list[BaseMessage]) -> list[PasoHerramienta]:
    """Empareja cada tool_call del modelo con el ToolMessage que la respondió."""
    llamadas = {}
    pasos: list[PasoHerramienta] = []
    for mensaje in mensajes:
        if isinstance(mensaje, AIMessage):
            for llamada in mensaje.tool_calls:
                llamadas[llamada["id"]] = llamada
        elif isinstance(mensaje, ToolMessage):
            llamada = llamadas.get(mensaje.tool_call_id, {})
            contenido = mensaje.content
            try:
                resultado = json.loads(contenido) if isinstance(contenido, str) else contenido
            except json.JSONDecodeError:
                resultado = contenido
            pasos.append(PasoHerramienta(mensaje.name or llamada.get("name", "?"), llamada.get("args", {}), resultado))
    return pasos


class AgenteDevoluciones:
    """Agente con memoria por conversación sobre las herramientas de `herramientas.py`."""

    def __init__(
        self,
        rag: SistemaRAG,
        llm: BaseChatModel | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> None:
        self.herramientas = [
            *herramientas.HERRAMIENTAS_TRANSACCIONALES,
            herramientas.crear_herramienta_buscar_politica(rag),
        ]
        self.grafo = create_agent(
            # Temperatura 0: en un flujo con acciones preferimos decisiones repetibles.
            model=llm or crear_llm(temperatura=0.0, max_tokens=700),
            tools=self.herramientas,
            system_prompt=prompts.SISTEMA_AGENTE,
            checkpointer=InMemorySaver(),
        )
        self.callbacks = callbacks or []
        # Si un turno falla a mitad de camino, el historial puede quedar con una
        # llamada a herramienta sin respuesta. Se abre un hilo interno nuevo.
        self._generacion: dict[str, int] = defaultdict(int)

    def _hilo_interno(self, hilo: str) -> str:
        return f"{hilo}#{self._generacion[hilo]}"

    def responder(self, mensaje: str, hilo: str) -> TurnoAgente:
        """Ejecuta un turno de la conversación `hilo` y devuelve el texto y la traza."""
        hilo_interno = self._hilo_interno(hilo)
        config = {
            "configurable": {"thread_id": hilo_interno},
            "recursion_limit": LIMITE_PASOS,
            "callbacks": self.callbacks,
        }
        previos = len(self.grafo.get_state({"configurable": {"thread_id": hilo_interno}}).values.get("messages", []))
        herramientas.REGISTRO.iniciar_turno(hilo_interno)
        try:
            salida = self.grafo.invoke({"messages": [("user", mensaje)]}, config=config)
        except Exception:
            self._generacion[hilo] += 1
            raise

        nuevos = salida["messages"][previos:]
        respuestas = [m for m in nuevos if isinstance(m, AIMessage)]
        texto = respuestas[-1].text.strip() if respuestas else ""
        if not texto:
            self._generacion[hilo] += 1
            raise RuntimeError("El agente terminó el turno sin texto para el cliente.")
        return TurnoAgente(texto=texto, pasos=_extraer_pasos(nuevos), llamadas_llm=len(respuestas))
