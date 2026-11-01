# Proyecto Final — Agente de devoluciones para EcoMarket

**Asignatura:** IA Generativa — Agentes de IA para automatización de tareas
**Maestría en Inteligencia Artificial — Universidad ICESI**

---

## El problema en una línea

El asistente RAG del [Taller 2](../Taller2/) informa sobre la política de
devoluciones, pero no puede actuar. El proyecto final lo convierte en un agente
que verifica si un producto se puede devolver y genera la guía de retorno.

## La solución en un párrafo

El router del Taller 2 sigue enviando las preguntas informativas a la cadena
RAG. Las devoluciones, las consultas de pedido y cualquier mensaje con un número
de seguimiento pasan a un agente de **LangChain** (`create_agent`) con cuatro
herramientas transaccionales y el RAG como quinta herramienta para citar la
política. La elegibilidad la decide un **motor de reglas en código**, nunca el
modelo. La guía solo se genera con una verificación válida de la misma
conversación y, si el retorno tiene costo, con la aceptación del cliente en un
turno posterior a aquel en que conoció el valor. La interfaz será una app de
**Streamlit**.

**Arquitectura completa:** abrir [`docs/arquitectura.html`](docs/arquitectura.html)
en el navegador (diagramas, herramientas, reglas, flujo, riesgos y plan).

---

## Estado

| Parte | Estado |
|---|---|
| Diseño de la arquitectura (Fase 1) | Listo en `docs/arquitectura.html` |
| Motor de reglas `src/reglas_devolucion.py` + 35 pruebas | Listo |
| Herramientas del agente `src/herramientas.py` + 20 pruebas | Listo |
| Agente `src/agente.py` e integración en `src/asistente.py` (Fase 2) | Listo |
| Evaluación del comportamiento: 12 casos de punta a punta | Listo: 12/12 en [`outputs/evaluacion_agente.md`](outputs/evaluacion_agente.md) |
| Respuestas textuales de las fases (`docs/fase1…fase4-*.md`) | Pendiente (el contenido de las fases 1 y 3 ya está en el HTML) |
| Interfaz Streamlit (Fase 4) | Pendiente |
| Video (máximo 10 minutos) | Pendiente (guion en el HTML) |

---

## Las herramientas

| Herramienta | Qué hace | Efecto |
|---|---|---|
| `consultar_pedido` | Estado, fecha de entrega y productos (SKU) de un pedido. Es la del Taller 2. | Solo lectura |
| `verificar_elegibilidad_producto` | Aplica la política v3.2 con el motor de reglas y emite un `id_verificacion` | Solo lectura |
| `generar_etiqueta_devolucion` | Crea la guía de retorno. Exige una verificación ELEGIBLE de la misma conversación | Crea una guía |
| `escalar_a_humano` | Abre un ticket con el resumen del caso, sin datos personales | Crea un ticket |
| `buscar_politica` | El retriever del Taller 2 (mismo índice y umbral) como herramienta | Solo lectura |

Ninguna herramienta lanza excepciones hacia el modelo: los errores vuelven como
`{"ok": false, "codigo": ..., "detalle": ...}` para que el agente los explique.

---

## Cómo ejecutar

Requisitos: Python 3.11+, [Ollama](https://ollama.com) local con `bge-m3` para los
embeddings y una key de Ollama Cloud para el LLM (`glm-5.3-flash`).

```bash
cd ProyectoFinal-EcoMarket
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
ollama pull bge-m3

# Pruebas (76; no requieren API key, Ollama ni índice)
python -m pytest -q

# Indexar la base de conocimiento (en Windows, con salida UTF-8)
set PYTHONIOENCODING=utf-8
python -m scripts.indexar

# Evaluación de punta a punta: 12 casos → outputs/evaluacion_agente.md (~1,5 min)
python -m scripts.evaluar_agente
python -m scripts.evaluar_agente --solo 4     # un caso, solo en consola

# Chat por consola; --debug muestra la ruta y las herramientas de cada turno
python -m scripts.chat --debug
```

La key va en un archivo `.env` (ver `.env.example`), en esta carpeta o en
cualquier carpeta superior: `OLLAMA_API_KEY=...`. Para usar OpenAI:
`ECOMARKET_PROVEEDOR=openai` y `OPENAI_API_KEY`.

Variables útiles: `ECOMARKET_SIMULAR_FALLA=1` hace fallar a la transportadora
(caso 12 de la evaluación) y `ECOMARKET_FECHA` fija el "hoy" (por defecto
2026-09-22, el corte de `pedidos.json`).

---

## Decisiones que conviene revisar

- **Defecto reportado después de 30 días en productos sin garantía de 12 meses.**
  La política no lo cubre. El motor lo manda a revisión humana (§9) en lugar de
  negarlo, porque negarlo en automático podría ir contra la garantía legal.
- **Un número de seguimiento en el mensaje lleva al agente aunque el router diga
  otra cosa.** En la evaluación, el router (un LLM) clasificó el mismo mensaje
  como `DEVOLUCION` en una corrida y como `OTRO` en otra.
- **La aceptación del costo debe llegar en un turno posterior.** El booleano
  `cliente_acepta_costo` lo llena el modelo; la regla del turno es la parte que
  no depende de él.

---

## Estructura

```
ProyectoFinal-EcoMarket/
├── docs/arquitectura.html          ← diseño completo del agente (Fases 1, 3 y 4)
├── data/                           ← base de conocimiento y pedidos (ficticios, del Taller 2)
├── src/
│   ├── reglas_devolucion.py        ← NUEVO: política v3.2 como código
│   ├── herramientas.py             ← CAMBIA: + 4 herramientas y el registro simulado
│   ├── agente.py                   ← NUEVO: create_agent con memoria por conversación
│   ├── asistente.py                ← CAMBIA: ruta al agente, ruta pegajosa y respaldo
│   ├── prompts.py                  ← CAMBIA: + SISTEMA_AGENTE
│   └── …                           ← RAG, router y guardrails del Taller 2
├── scripts/
│   ├── evaluar_agente.py           ← NUEVO: los 12 casos de punta a punta
│   ├── chat.py                     ← CAMBIA: una conversación por sesión
│   └── …                           ← indexar y demos del Taller 2
├── tests/
│   ├── test_reglas_devolucion.py   ← NUEVO: 35 pruebas del motor
│   ├── test_herramientas_agente.py ← NUEVO: 20 pruebas de las herramientas
│   └── …                           ← 21 pruebas del Taller 2
└── outputs/evaluacion_agente.md    ← generado por evaluar_agente.py
```

No se versionan el índice (`indice_chroma/`), el entorno (`.venv/`) ni el
registro de guías y tickets (`data/devoluciones.json`): se regeneran.
