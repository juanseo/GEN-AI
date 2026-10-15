"""Cliente del LLM de generación (LangChain `ChatOpenAI`) y contabilidad de uso.

Toda la dependencia del proveedor queda en este archivo: OpenAI y Ollama Cloud
exponen la misma API, así que solo cambian la URL, la key y el modelo
(`src/config.py`). La cadena, el retriever y los guardrails no lo notan.
"""

from __future__ import annotations

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_openai import ChatOpenAI

from .config import CONFIG, PROVEEDORES


class ApiKeyFaltante(RuntimeError):
    """No hay API key del proveedor del LLM en el entorno ni en el .env."""


# Precios de lista de referencia en USD por millón de tokens. Verificar los
# vigentes en https://openai.com/api/pricing/ antes de usarlos en un caso de
# negocio. Los modelos de Ollama Cloud se cobran por suscripción: costo 0 aquí.
PRECIOS_USD_POR_MILLON = {
    "gpt-4o-mini": {"entrada": 0.15, "salida": 0.60},
    "gpt-4o": {"entrada": 2.50, "salida": 10.00},
}


def crear_llm(temperatura: float | None = None, max_tokens: int = 500) -> ChatOpenAI:
    """Instancia el modelo de chat configurado."""
    api_key = CONFIG.api_key
    if not api_key:
        nombres = " o ".join(PROVEEDORES[CONFIG.proveedor]["api_key_names"])
        raise ApiKeyFaltante(
            f"No se encontró la API key del proveedor '{CONFIG.proveedor}'. Define {nombres} "
            "en el entorno o en un archivo .env en Taller2/ (ver .env.example)."
        )
    parametros = {
        "model": CONFIG.modelo,
        "temperature": CONFIG.temperatura if temperatura is None else temperatura,
        "max_tokens": max_tokens + CONFIG.margen_razonamiento,
        "api_key": api_key,
        "base_url": CONFIG.base_url,
    }
    if CONFIG.esfuerzo_razonamiento:
        parametros["reasoning_effort"] = CONFIG.esfuerzo_razonamiento
    return ChatOpenAI(**parametros)


def resumen_uso(uso: UsageMetadataCallbackHandler) -> dict:
    """Tokens y costo estimado acumulados por un `UsageMetadataCallbackHandler`."""
    entrada = salida = 0
    costo = 0.0
    for modelo, datos in uso.usage_metadata.items():
        entrada += datos.get("input_tokens", 0)
        salida += datos.get("output_tokens", 0)
        # La API devuelve el nombre con fecha (gpt-4o-mini-2024-07-18).
        precio = next((p for nombre, p in sorted(PRECIOS_USD_POR_MILLON.items(), key=lambda x: -len(x[0]))
                       if modelo.startswith(nombre)), None)
        if precio:
            costo += datos.get("input_tokens", 0) / 1e6 * precio["entrada"]
            costo += datos.get("output_tokens", 0) / 1e6 * precio["salida"]
    return {"tokens_entrada": entrada, "tokens_salida": salida, "costo_usd": costo}
