"""Configuración central del asistente de EcoMarket.

Carga la API key desde un archivo `.env`, buscándolo hacia arriba desde este
paquete. Soporta dos proveedores que exponen la misma API (la de OpenAI):

- `ollama` (por defecto): modelos open-source en Ollama Cloud, con
  `OLLAMA_API_KEY`. El taller permite explícitamente un modelo open-source para
  la Fase 3.
- `openai`: modelos de OpenAI, con `OPENAI_API_KEY` (o `OPENAI-API-KEY`, el
  nombre con guiones que usa el `.env` del curso).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

# Raíz del entregable: .../GEN-AI/Taller1
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Proveedor de inferencia: "ollama" (Ollama Cloud) u "openai".
PROVEEDOR = os.environ.get("ECOMARKET_PROVEEDOR", "ollama").strip().lower()

# Por proveedor: variables de la API key (en orden de prioridad), URL base del
# endpoint compatible con OpenAI, modelo de generación por defecto y ajustes de
# razonamiento. Los modelos de Ollama Cloud (glm, gpt-oss) razonan: los tokens
# que "piensan" cuentan contra max_tokens, así que sin margen extra el router
# (8 tokens) y la llamada a herramientas (200) se quedan sin espacio y devuelven
# texto vacío.
PROVEEDORES = {
    "ollama": {
        "api_key_names": ("OLLAMA_API_KEY",),
        "base_url": "https://ollama.com/v1",
        "modelo": "glm-5.3-flash",
        "esfuerzo_razonamiento": "low",
        "margen_razonamiento": 1024,
    },
    "openai": {
        "api_key_names": ("OPENAI_API_KEY", "OPENAI-API-KEY"),
        "base_url": None,
        "modelo": "gpt-4o-mini",
        "esfuerzo_razonamiento": None,
        "margen_razonamiento": 0,
    },
}
if PROVEEDOR not in PROVEEDORES:
    raise ValueError(
        f"ECOMARKET_PROVEEDOR={PROVEEDOR!r} no es válido; usa 'ollama' u 'openai'."
    )

# Nombres de variable aceptados para el proveedor activo.
API_KEY_NAMES = PROVEEDORES[PROVEEDOR]["api_key_names"]


def _buscar_archivos_env() -> list[Path]:
    """Devuelve los .env candidatos, del más cercano al más lejano."""
    candidatos = [PROJECT_ROOT / ".env"]
    for directorio in PROJECT_ROOT.parents:
        candidatos.append(directorio / ".env")
    return [ruta for ruta in candidatos if ruta.is_file()]


def cargar_api_key() -> str | None:
    """Busca la API key en el entorno y luego en los .env de la jerarquía."""
    for nombre in API_KEY_NAMES:
        valor = os.environ.get(nombre)
        if valor:
            return valor.strip()

    for ruta_env in _buscar_archivos_env():
        valores = dotenv_values(ruta_env)
        for nombre in API_KEY_NAMES:
            valor = valores.get(nombre)
            if valor:
                return valor.strip()
    return None


@dataclass(frozen=True)
class Config:
    """Parámetros de ejecución del asistente."""

    # Proveedor activo y URL del endpoint (None = la de OpenAI).
    proveedor: str = PROVEEDOR
    base_url: str | None = PROVEEDORES[PROVEEDOR]["base_url"]

    # Modelo de generación. Con Ollama, glm-5.3-flash: open-source, con buen
    # soporte de function calling y el más rápido y económico en tokens de los
    # probados (gpt-oss:120b da respuestas equivalentes con ~50% más salida).
    # Con OpenAI, gpt-4o-mini: pequeño y económico porque el 80% del tráfico es
    # repetitivo.
    modelo: str = os.environ.get("ECOMARKET_MODELO", PROVEEDORES[PROVEEDOR]["modelo"])

    # Solo para modelos de razonamiento: esfuerzo pedido (None = no se envía) y
    # tokens extra que se suman al max_tokens de cada llamada para el
    # razonamiento, de modo que los límites de la cadena sigan midiendo solo
    # la respuesta visible.
    esfuerzo_razonamiento: str | None = PROVEEDORES[PROVEEDOR]["esfuerzo_razonamiento"]
    margen_razonamiento: int = PROVEEDORES[PROVEEDOR]["margen_razonamiento"]

    # Modelo de embeddings, usado solo si motor_retrieval == "openai" y el
    # proveedor es openai.
    modelo_embeddings: str = os.environ.get(
        "ECOMARKET_MODELO_EMBEDDINGS", "text-embedding-3-small"
    )

    # "tfidf" (offline, sin costo) o "openai" (embeddings reales de la API).
    motor_retrieval: str = os.environ.get("ECOMARKET_RETRIEVAL", "tfidf")

    # Temperatura baja: en atención al cliente preferimos respuestas
    # consistentes y apegadas al contexto por encima de la creatividad.
    temperatura: float = 0.2

    # Fragmentos recuperados antes y después del re-ranking.
    top_k_recuperacion: int = 8
    top_n_reranking: int = 4

    # Peso de la búsqueda vectorial en la búsqueda híbrida (el resto es léxica).
    peso_vectorial: float = 0.6

    @property
    def api_key(self) -> str | None:
        return cargar_api_key()

    @property
    def hay_api_key(self) -> bool:
        return bool(self.api_key)


CONFIG = Config()
