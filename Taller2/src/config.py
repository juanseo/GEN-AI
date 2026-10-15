"""Configuración central del asistente de EcoMarket con RAG (Taller 2).

Agrupa en un solo lugar las tres decisiones que el resto del código consume:

- **LLM de generación** (`ECOMARKET_PROVEEDOR`): `openai` (`gpt-4o-mini`, por
  defecto en este taller) u `ollama` (`glm-5.3-flash` en Ollama Cloud, el
  modelo de la Fase 3 del Taller 1). Ambos exponen la API de OpenAI.
- **Modelo de embeddings** (`ECOMARKET_EMBEDDINGS`): ver `MODELOS_EMBEDDINGS`.
  Los modelos open-source corren en el Ollama **local** del equipo
  (`http://localhost:11434`), que es distinto de Ollama Cloud.
- **Parámetros del retrieval**: `top_k` y el umbral de relevancia por modelo,
  calibrado con `scripts/evaluar_retrieval.py`.

La API key se lee del entorno o de un archivo `.env` en este directorio o en
cualquiera de los superiores.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

# Raíz del entregable: .../GEN-AI/Taller2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INDICE_DIR = PROJECT_ROOT / "indice_chroma"

# "Hoy" para el asistente. Los pedidos de prueba son una foto del sistema al
# corte de pedidos.json; si se usara la fecha real, cada día que pasa todos los
# pedidos parecerían más atrasados y los plazos de devolución vencerían. Por
# eso, salvo que se indique otra cosa, hoy = fecha de corte de los datos.
FECHA_REFERENCIA = os.environ.get("ECOMARKET_FECHA") or json.loads(
    (DATA_DIR / "pedidos.json").read_text(encoding="utf-8")
)["_meta"]["fecha_corte"]

# --- LLM de generación ------------------------------------------------------

PROVEEDOR = os.environ.get("ECOMARKET_PROVEEDOR", "openai").strip().lower()
OLLAMA_LOCAL_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

PROVEEDORES = {
    "openai": {
        "api_key_names": ("OPENAI_API_KEY", "OPENAI-API-KEY"),
        "base_url": None,
        "modelo": "gpt-4o-mini",
        "esfuerzo_razonamiento": None,
        "margen_razonamiento": 0,
    },
    # Modelo de razonamiento: sus tokens de "pensamiento" cuentan contra
    # max_tokens, por eso se suma un margen (ver Taller 1, Fase 1, sección 4.4).
    "ollama": {
        "api_key_names": ("OLLAMA_API_KEY",),
        "base_url": "https://ollama.com/v1",
        "modelo": "glm-5.3-flash",
        "esfuerzo_razonamiento": "low",
        "margen_razonamiento": 1024,
    },
    # Respaldo sin API key ni costo: un modelo pequeño servido por el Ollama
    # local del equipo (el mismo que sirve los embeddings). Sirve para ejecutar
    # y verificar la cadena completa sin depender de un proveedor externo; la
    # calidad de redacción es menor que la de gpt-4o-mini.
    "local": {
        "api_key_names": (),
        "base_url": f"{OLLAMA_LOCAL_URL}/v1",
        "modelo": "llama3.2:3b",
        "esfuerzo_razonamiento": None,
        "margen_razonamiento": 0,
    },
}
if PROVEEDOR not in PROVEEDORES:
    raise ValueError(f"ECOMARKET_PROVEEDOR={PROVEEDOR!r} no es válido; opciones: {', '.join(PROVEEDORES)}.")

# --- Modelos de embeddings ----------------------------------------------------

# Catálogo de modelos comparados en la Fase 1. `umbral` es la similitud coseno
# mínima del mejor fragmento para considerar que la consulta está dentro del
# alcance de la base de conocimiento: la mínima observada en las preguntas en
# alcance del golden set, menos 0,05 de holgura. Cada modelo distribuye sus similitudes en
# un rango distinto, por eso el umbral se calibra por modelo y no es universal
# (ver outputs/evaluacion_retrieval.md).
MODELOS_EMBEDDINGS: dict[str, dict] = {
    "bge-m3": {
        "proveedor": "ollama_local",
        "modelo": "bge-m3",
        "dimensiones": 1024,
        "licencia": "MIT (open-source, BAAI)",
        "umbral": 0.41,
    },
    "paraphrase-multilingual": {
        "proveedor": "ollama_local",
        "modelo": "paraphrase-multilingual",
        "dimensiones": 768,
        "licencia": "Apache 2.0 (open-source, sentence-transformers)",
        "umbral": 0.34,
    },
    "openai-3-small": {
        "proveedor": "openai",
        "modelo": "text-embedding-3-small",
        "dimensiones": 1536,
        "licencia": "Propietario (API de OpenAI)",
        # Provisional: no se calibró por falta de API key en el entorno de
        # desarrollo. Recalibrar con scripts/evaluar_retrieval.py.
        "umbral": 0.35,
    },
    "tfidf": {
        "proveedor": "local",
        "modelo": "TF-IDF de n-gramas de caracteres (línea base del Taller 1)",
        "dimensiones": None,
        "licencia": "scikit-learn",
        "umbral": 0.06,
    },
}

EMBEDDINGS_POR_DEFECTO = "bge-m3"


# --- API keys -------------------------------------------------------------------

def _buscar_archivos_env() -> list[Path]:
    """Devuelve los .env candidatos, del más cercano al más lejano."""
    candidatos = [PROJECT_ROOT / ".env", *(d / ".env" for d in PROJECT_ROOT.parents)]
    return [ruta for ruta in candidatos if ruta.is_file()]


def cargar_api_key(nombres: tuple[str, ...]) -> str | None:
    """Busca la primera variable de `nombres` en el entorno y luego en los .env."""
    for nombre in nombres:
        if os.environ.get(nombre):
            return os.environ[nombre].strip()
    for ruta_env in _buscar_archivos_env():
        valores = dotenv_values(ruta_env)
        for nombre in nombres:
            if valores.get(nombre):
                return valores[nombre].strip()
    return None


@dataclass(frozen=True)
class Config:
    """Parámetros de ejecución del asistente."""

    proveedor: str = PROVEEDOR
    base_url: str | None = PROVEEDORES[PROVEEDOR]["base_url"]
    modelo: str = os.environ.get("ECOMARKET_MODELO", PROVEEDORES[PROVEEDOR]["modelo"])
    esfuerzo_razonamiento: str | None = PROVEEDORES[PROVEEDOR]["esfuerzo_razonamiento"]
    margen_razonamiento: int = PROVEEDORES[PROVEEDOR]["margen_razonamiento"]

    # Temperatura baja: en atención al cliente preferimos respuestas
    # consistentes y apegadas al contexto por encima de la creatividad.
    temperatura: float = 0.2

    embeddings: str = os.environ.get("ECOMARKET_EMBEDDINGS", EMBEDDINGS_POR_DEFECTO)

    # Fragmentos que se inyectan en el prompt. 4 cubre las preguntas que
    # combinan dos documentos (p. ej. ficha del producto + política) sin
    # llenar el context window de ruido.
    top_k: int = 4

    @property
    def api_key(self) -> str | None:
        nombres = PROVEEDORES[self.proveedor]["api_key_names"]
        # Ollama local no valida la key, pero el cliente de OpenAI exige una.
        return cargar_api_key(nombres) if nombres else "ollama-local"

    @property
    def umbral_relevancia(self) -> float:
        return MODELOS_EMBEDDINGS[self.embeddings]["umbral"]


CONFIG = Config()
if CONFIG.embeddings not in MODELOS_EMBEDDINGS:
    raise ValueError(
        f"ECOMARKET_EMBEDDINGS={CONFIG.embeddings!r} no es válido; "
        f"opciones: {', '.join(MODELOS_EMBEDDINGS)}."
    )
