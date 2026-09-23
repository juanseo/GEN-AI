"""Configuración central del asistente de EcoMarket.

Carga la API key desde un archivo `.env`, buscándolo hacia arriba desde este
paquete. Se aceptan dos nombres de variable porque el `.env` del curso usa
`OPENAI-API-KEY` (con guiones) mientras que la convención del SDK es
`OPENAI_API_KEY`.
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

# Nombres de variable aceptados, en orden de prioridad.
API_KEY_NAMES = ("OPENAI_API_KEY", "OPENAI-API-KEY")


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

    # Modelo de generación. El taller permite cualquier modelo; usamos un
    # modelo pequeño y económico porque el 80% del tráfico es repetitivo.
    modelo: str = os.environ.get("ECOMARKET_MODELO", "gpt-4o-mini")

    # Modelo de embeddings, usado solo si motor_retrieval == "openai".
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
