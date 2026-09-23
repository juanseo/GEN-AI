"""Controles de seguridad alrededor del modelo.

Los riesgos que identificamos en la Fase 2 no se mitigan con un buen prompt
solamente: hacen falta controles deterministas antes y después de la llamada al
LLM. Este módulo implementa los tres más baratos y de mayor impacto:

- **Minimización de PII** antes de enviar texto al proveedor del modelo.
- **Detección de escalamiento** a un agente humano por reglas, no por criterio
  del modelo (un modelo que decide si escalar puede decidir no hacerlo).
- **Verificación de anclaje** de la respuesta contra el contexto recuperado,
  como señal de posible alucinación.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- 1. Minimización de PII -------------------------------------------------

PATRONES_PII = [
    (re.compile(r"\b[\w.\-+]+@[\w\-]+\.[\w.\-]+\b"), "[CORREO]"),
    (re.compile(r"\b(?:\+57[\s\-]?)?3\d{2}[\s\-]?\d{3}[\s\-]?\d{4}\b"), "[CELULAR]"),
    (re.compile(r"\b(?:\d[ \-]?){13,16}\b"), "[TARJETA]"),
    (re.compile(r"\b(?:cc|c\.c\.|cédula|cedula)[\s:]*\d{6,12}\b", re.IGNORECASE), "[DOCUMENTO]"),
    (re.compile(r"\b(?:calle|carrera|cra|cll|avenida|av|diagonal|transversal)\.?\s*"
                r"\d+[\w\s#\-]{2,30}\b", re.IGNORECASE), "[DIRECCION]"),
]


def enmascarar_pii(texto: str) -> tuple[str, list[str]]:
    """Reemplaza datos personales por marcadores. Devuelve (texto, tipos hallados)."""
    hallados: list[str] = []
    limpio = texto
    for patron, marcador in PATRONES_PII:
        limpio, reemplazos = patron.subn(marcador, limpio)
        if reemplazos:
            hallados.append(marcador)
    return limpio, hallados


# --- 2. Escalamiento a humano ----------------------------------------------

DISPARADORES_ESCALAMIENTO = {
    "legal": ["demanda", "demandar", "abogado", "sic ", "superintendencia",
              "tutela", "denuncia", "fraude", "estafa"],
    "seguridad": ["intoxicación", "intoxicacion", "alergia", "quemadura",
                  "me enfermé", "me enferme", "hospital", "incendio"],
    "financiero": ["cobro duplicado", "me cobraron dos veces", "doble cobro",
                   "no reconozco el cargo", "cargo no autorizado"],
    "emocional": ["pésimo servicio", "pesimo servicio", "estafadores",
                  "nunca más", "nunca mas", "indignado", "indignada", "furioso"],
    "explicito": ["hablar con una persona", "hablar con un humano",
                  "agente humano", "supervisor", "quiero un asesor"],
}


@dataclass
class DecisionEscalamiento:
    escalar: bool
    motivo: str | None = None
    categoria: str | None = None


def evaluar_escalamiento(mensaje: str) -> DecisionEscalamiento:
    """Decide por reglas si el caso debe ir a un agente humano."""
    texto = mensaje.lower()
    for categoria, disparadores in DISPARADORES_ESCALAMIENTO.items():
        for disparador in disparadores:
            if disparador in texto:
                return DecisionEscalamiento(
                    escalar=True,
                    motivo=f"Se detectó el patrón '{disparador.strip()}'.",
                    categoria=categoria,
                )
    return DecisionEscalamiento(escalar=False)


MENSAJE_ESCALAMIENTO = (
    "Entiendo la situación y lamento mucho lo ocurrido. Este caso lo va a "
    "atender directamente una persona del equipo de EcoMarket, que tendrá todo "
    "el detalle de tu conversación a la mano. Te contactaremos dentro de las "
    "próximas 2 horas hábiles. Gracias por tu paciencia."
)


# --- 3. Verificación de anclaje (anti-alucinación) --------------------------

def verificar_anclaje(respuesta: str, contexto: str, umbral: float = 0.55) -> dict:
    """Mide qué proporción de los datos duros de la respuesta aparece en el contexto.

    Es una heurística, no una garantía: busca cifras, fechas, códigos y URLs en
    la respuesta y verifica que existan en el material recuperado. Sirve como
    señal de monitoreo y para marcar respuestas que deben revisarse.
    """
    datos_duros = set(
        re.findall(r"\b(?:ECO-\d+|\d{4}-\d{2}-\d{2}|https?://\S+|\d[\d.,]{2,})\b", respuesta)
    )
    if not datos_duros:
        return {"anclada": True, "cobertura": 1.0, "no_soportados": []}

    contexto_normalizado = contexto.replace(".", "").replace(",", "")
    no_soportados = [
        dato for dato in datos_duros
        if dato not in contexto and dato.replace(".", "").replace(",", "") not in contexto_normalizado
    ]
    cobertura = 1 - len(no_soportados) / len(datos_duros)
    return {
        "anclada": cobertura >= umbral,
        "cobertura": round(cobertura, 2),
        "no_soportados": no_soportados,
    }
