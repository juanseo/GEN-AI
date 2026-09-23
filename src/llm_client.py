"""Cliente del LLM (OpenAI) con conteo de tokens y estimación de costo.

Se aísla aquí toda la dependencia del proveedor para que cambiar de modelo o de
proveedor no obligue a tocar la cadena de prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .config import CONFIG


class ApiKeyFaltante(RuntimeError):
    """Se lanza cuando no hay API key disponible en el entorno ni en el .env."""


# Precios de lista de referencia en USD por 1 millón de tokens.
# OJO: son valores de referencia a la fecha del taller; verificar siempre los
# precios vigentes en https://openai.com/api/pricing/ antes de usarlos en un
# caso de negocio real.
PRECIOS_USD_POR_MILLON = {
    "gpt-4o-mini": {"entrada": 0.15, "salida": 0.60},
    "gpt-4o": {"entrada": 2.50, "salida": 10.00},
    "text-embedding-3-small": {"entrada": 0.02, "salida": 0.0},
}


@dataclass
class Uso:
    """Acumulador de consumo de tokens de una sesión."""

    tokens_entrada: int = 0
    tokens_salida: int = 0
    llamadas: int = 0
    por_modelo: dict[str, dict[str, int]] = field(default_factory=dict)

    def registrar(self, modelo: str, entrada: int, salida: int) -> None:
        self.tokens_entrada += entrada
        self.tokens_salida += salida
        self.llamadas += 1
        acumulado = self.por_modelo.setdefault(
            modelo, {"entrada": 0, "salida": 0, "llamadas": 0}
        )
        acumulado["entrada"] += entrada
        acumulado["salida"] += salida
        acumulado["llamadas"] += 1

    def costo_estimado_usd(self) -> float:
        total = 0.0
        for modelo, datos in self.por_modelo.items():
            precios = PRECIOS_USD_POR_MILLON.get(modelo)
            if not precios:
                continue
            total += datos["entrada"] / 1_000_000 * precios["entrada"]
            total += datos["salida"] / 1_000_000 * precios["salida"]
        return total


class ClienteLLM:
    """Envoltura delgada sobre el SDK de OpenAI."""

    def __init__(self, modelo: str | None = None) -> None:
        api_key = CONFIG.api_key
        if not api_key:
            raise ApiKeyFaltante(
                "No se encontró la API key. Define OPENAI_API_KEY (o "
                "OPENAI-API-KEY) en el entorno o en un archivo .env en este "
                "directorio o en alguno superior."
            )
        self._cliente = OpenAI(api_key=api_key)
        self.modelo = modelo or CONFIG.modelo
        self.uso = Uso()

    def completar(
        self,
        mensajes: list[dict[str, str]],
        *,
        temperatura: float | None = None,
        max_tokens: int = 700,
        herramientas: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Llama al modelo y devuelve el mensaje de respuesta completo."""
        kwargs: dict[str, Any] = {
            "model": self.modelo,
            "messages": mensajes,
            "temperature": CONFIG.temperatura if temperatura is None else temperatura,
            "max_tokens": max_tokens,
        }
        if herramientas:
            kwargs["tools"] = herramientas
            kwargs["tool_choice"] = "auto"

        respuesta = self._cliente.chat.completions.create(**kwargs)
        if respuesta.usage:
            self.uso.registrar(
                self.modelo,
                respuesta.usage.prompt_tokens,
                respuesta.usage.completion_tokens,
            )
        return respuesta.choices[0].message

    def texto(self, mensajes: list[dict[str, str]], **kwargs: Any) -> str:
        """Atajo para cuando solo interesa el texto de la respuesta."""
        return (self.completar(mensajes, **kwargs).content or "").strip()

    def embeddings(self, textos: list[str]) -> list[list[float]]:
        """Genera embeddings. Solo se usa con ECOMARKET_RETRIEVAL=openai."""
        respuesta = self._cliente.embeddings.create(
            model=CONFIG.modelo_embeddings, input=textos
        )
        if respuesta.usage:
            self.uso.registrar(
                CONFIG.modelo_embeddings, respuesta.usage.prompt_tokens, 0
            )
        return [dato.embedding for dato in respuesta.data]
