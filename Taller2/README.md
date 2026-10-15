# Taller Práctico #2 — Sistema RAG para la Atención al Cliente de EcoMarket

**Caso:** Optimización de la atención al cliente en una empresa de e-commerce
**Asignatura:** IA Generativa — Generación Aumentada por Recuperación (RAG)
**Maestría en Inteligencia Artificial — Universidad ICESI**

---

## El problema en una línea

El asistente del [Taller 1](../Taller1/) atendía bien dos tipos de consulta, pero su
"base de conocimiento" era una búsqueda TF-IDF en memoria sobre tres archivos
Markdown. Para responder **cualquier** solicitud, o reconocer con honestidad que no
puede atenderla, necesita un sistema RAG de verdad.

## La solución en un párrafo

Un modelo de embeddings **open-source y multilingüe (`bge-m3`)**, servido localmente
con Ollama, convierte en vectores una base de conocimiento de **tres documentos en
tres formatos** (política en PDF, catálogo en CSV, FAQ en JSON), partidos por su
estructura natural. Los vectores viven en **ChromaDB** (HNSW, coseno). Con
**LangChain**, la cadena del Taller 1 (guardrails → router → *function calling* para
pedidos / RAG para lo demás) consulta esa base antes de que el LLM de la nube genere la
respuesta: `glm-5.3-flash` en Ollama Cloud por defecto (el modelo del Taller 1),
`gpt-4o-mini` como alternativa. Si nada en la base se parece a la pregunta, un **umbral de similitud
calibrado** hace que el asistente lo diga, sin inventar ni gastar tokens. Todas las
decisiones están respaldadas por una evaluación con 34 preguntas reales.

---

## Entregables por fase

| Fase | Documento |
|---|---|
| **1. Selección de componentes** (embeddings + base vectorial) | [`docs/fase1-seleccion-componentes.md`](docs/fase1-seleccion-componentes.md) |
| **2. Base de conocimiento** (documentos, *chunking*, indexación) | [`docs/fase2-base-de-conocimiento.md`](docs/fase2-base-de-conocimiento.md) |
| **3. Integración y código** (explicación, experimentos, limitaciones y supuestos) | [`docs/fase3-integracion-codigo.md`](docs/fase3-integracion-codigo.md) + código ejecutable |

**Evidencia ejecutable:**

| Archivo | Lo genera | Contiene |
|---|---|---|
| [`outputs/evaluacion_retrieval.md`](outputs/evaluacion_retrieval.md) | `python -m scripts.evaluar_retrieval` | Comparación de 3 modelos de embeddings (más OpenAI si hay key) y 3 estrategias de *chunking*; calibración del umbral |
| [`outputs/ejecucion_taller2_local.md`](outputs/ejecucion_taller2_local.md) | `python -m scripts.demo_taller2` con `ECOMARKET_PROVEEDOR=local` | La corrida completa de 5 secciones (A–E: con/sin RAG, preguntas nuevas, fuera de alcance, integración con el Taller 1, 4 experimentos), ejecutada con `llama3.2:3b` local |
| `outputs/ejecucion_taller2.md` | `python -m scripts.demo_taller2` | La misma corrida con el proveedor por defecto: `glm-5.3-flash` en Ollama Cloud (requiere `OLLAMA_API_KEY`) |

## Resultados principales

| Medición | Resultado |
|---|---|
| Recuperación en español coloquial, hit@1 (26 preguntas) | **bge-m3 88%**, paraphrase-multilingual 62%, TF-IDF del Taller 1 58% |
| Mismo modelo, distinto *chunking* (hit@1) | **Estructural 88%**, tamaño fijo 58%, por párrafos 38% |
| Expansión por código de política con filtros de Chroma (hit@4) | 92% → **96%** |
| Preguntas legítimas rechazadas por el umbral | **0%** (umbral 0,41) |

---

## Cómo ejecutar

### 1. Requisitos

- Python 3.11+
- [Ollama](https://ollama.com) corriendo localmente, con el modelo de embeddings:

```bash
ollama pull bge-m3
```

### 2. Instalar dependencias

```bash
cd Taller2
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

### 3. Configurar la API key del LLM

```bash
cp .env.example .env            # y completa OLLAMA_API_KEY
```

> Por defecto el LLM es `glm-5.3-flash` en Ollama Cloud, el mismo proveedor del
> Taller 1 (la key se crea en https://ollama.com/settings/keys). Con
> `ECOMARKET_PROVEEDOR=openai` y `OPENAI_API_KEY` se usa `gpt-4o-mini` en su lugar.

> **Sin API key:** todo el sistema corre localmente con `ECOMARKET_PROVEEDOR=local`
> (LLM `llama3.2:3b` en Ollama: `ollama pull llama3.2:3b`). La redacción es de menor
> calidad, pero la cadena es la misma. Ver la Fase 3, sección de limitaciones.

### 4. Ejecutar

```bash
# Pruebas (no requieren API key ni Ollama para el chunking)
python -m pytest -q

# Indexar la base de conocimiento (opcional: se hace sola la primera vez)
python -m scripts.indexar

# RAG mínimo en 5 secciones: el recorrido didáctico del código
python -m scripts.rag_ejemplo
python -m scripts.rag_ejemplo --sin-llm        # solo recuperación, sin API key

# Evaluación de embeddings y chunking → outputs/evaluacion_retrieval.md
python -m scripts.evaluar_retrieval

# Demo completa de la Fase 3 → outputs/ejecucion_taller2.md
python -m scripts.demo_taller2

# Chat interactivo, con el detalle interno visible
python -m scripts.chat --debug
```

Preguntas para probar en el chat:

```
¿Le puedo pagar en efectivo al mensajero cuando me traiga el paquete?
Necesito que la factura salga a nombre de mi empresa
¿Cómo va mi pedido ECO-10003?
El shampoo del pedido ECO-10009 me llegó partido y con moho
¿Cuál es la capital de Australia?
```

---

## Estructura del repositorio

```
Taller2/
├── README.md                       ← este archivo
├── requirements.txt
├── .env.example
│
├── docs/                           ← FASES 1, 2 y 3 (respuestas textuales)
│   ├── fase1-seleccion-componentes.md
│   ├── fase2-base-de-conocimiento.md
│   └── fase3-integracion-codigo.md
│
├── data/                           ← base de conocimiento (datos ficticios)
│   ├── politicas_devolucion.pdf    ← documento 1: PDF   (se indexa)
│   ├── catalogo_productos.csv      ← documento 2: CSV   (se indexa)
│   ├── faq.json                    ← documento 3: JSON  (se indexa)
│   ├── pedidos.json                ← NO se indexa: function calling
│   ├── fuentes/politicas_devolucion.md   ← fuente editable del PDF
│   └── evaluacion/golden_set.json  ← 26 preguntas en alcance + 8 fuera
│
├── src/
│   ├── config.py                   ← proveedores, modelos, umbrales, fecha de referencia
│   ├── cargadores.py               ← PDF / CSV / JSON → Document (limpieza)
│   ├── chunking.py                 ← segmentación estructural (+ fija y por párrafos)
│   ├── embeddings.py               ← bge-m3 · paraphrase · OpenAI · TF-IDF
│   ├── indexador.py                ← ChromaDB: colección, upsert, reindexación
│   ├── rag.py                      ← retriever con umbral, expansión, cadena LCEL
│   ├── llm.py                      ← ChatOpenAI (OpenAI / Ollama) y costo
│   ├── prompts.py                  ← prompts del Taller 1 + RAG + fuera de alcance
│   ├── asistente.py                ← cadena completa: guardrails, router, ramas
│   ├── herramientas.py             ← consultar_pedido como herramienta de LangChain
│   └── guardrails.py               ← PII, escalamiento, anclaje (del Taller 1)
│
├── scripts/
│   ├── rag_ejemplo.py              ← RAG mínimo en 5 secciones
│   ├── indexar.py · evaluar_retrieval.py · demo_taller2.py · chat.py
│   └── generar_pdf_politica.py     ← genera el PDF desde su fuente
│
├── tests/                          ← 21 pruebas: cargadores, chunking, umbral
└── outputs/                        ← evaluación y transcripciones generadas
```

El índice vectorial (`indice_chroma/`) no se versiona: se regenera desde `data/` en
segundos.

---

## Fuentes

1. Ávila, D. (2026). *Clase 2 — RAG: Retrieval Augmented Generation*. ICESI.
2. LangChain. *Documentación de LangChain 1.x* (LCEL, retrievers, vector stores). https://python.langchain.com/
3. BAAI. *bge-m3*. https://huggingface.co/BAAI/bge-m3
4. Chroma. *Documentación*. https://docs.trychroma.com/
5. Congreso de Colombia. *Ley 1480 de 2011* (Estatuto del Consumidor) y *Ley 1581 de 2012* (habeas data).
