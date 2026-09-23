# Taller Práctico #1 — Solución de IA Generativa para EcoMarket

**Caso:** Optimización de la atención al cliente en una empresa de e-commerce
**Asignatura:** IA Generativa — Sesión 2 (RAG)
**Maestría en Inteligencia Artificial — Universidad ICESI**

---

## El problema en una línea

EcoMarket responde en 24 horas. El 80% de sus miles de consultas diarias son
repetitivas y verificables contra sus propios sistemas; el 20% restante necesita
empatía humana. La solución no es "ponerle un chatbot": es **separar esos dos
flujos** y darle al primero acceso confiable a los datos de la empresa.

## La solución en un párrafo

Una **arquitectura híbrida de dos niveles**. El Nivel 1 atiende el 80%
repetitivo con un LLM de gama económica (`gpt-4o-mini`) que nunca responde de
memoria: el estado de un pedido se resuelve con **function calling** contra el
sistema transaccional, y las políticas, el catálogo y las FAQ se resuelven con
**RAG** (búsqueda híbrida + re-ranking + expansión de consulta). El Nivel 2 son
los agentes humanos, a quienes el sistema **escala por reglas deterministas** y
asiste con un copiloto. El conocimiento vive **fuera del modelo**, por lo que
actualizar una política es editar un documento, no reentrenar.

---

## Entregables por fase

| Fase | Documento | Puntos |
|---|---|---|
| **1. Selección y justificación del modelo** | [`docs/fase1-seleccion-y-justificacion.md`](docs/fase1-seleccion-y-justificacion.md) | 2 |
| **2. Fortalezas, limitaciones y riesgos éticos** | [`docs/fase2-fortalezas-limitaciones-riesgos.md`](docs/fase2-fortalezas-limitaciones-riesgos.md) | 2 |
| **3. Ingeniería de prompts** | [`docs/fase3-ingenieria-de-prompts.md`](docs/fase3-ingenieria-de-prompts.md) + código ejecutable | 1 |

**Salida verificable de la Fase 3:** [`outputs/ejecucion_fase3.md`](outputs/)
(se genera al ejecutar `python -m scripts.demo_fase3`).

---

## Cómo ejecutar

### 1. Instalar dependencias

```bash
cd Entregables/sesion2
pip install -r requirements.txt
```

### 2. Configurar la API key

```bash
cp .env.example .env
```

Y edita el `.env`:

```
OPENAI_API_KEY=sk-...
```

> El código acepta tanto `OPENAI_API_KEY` como `OPENAI-API-KEY`, y busca el
> archivo `.env` en este directorio y en todos los directorios superiores, así
> que un `.env` ya existente en la raíz del curso también funciona.

### 3. Ejecutar

```bash
# Inspeccionar la capa de retrieval — NO requiere API key ni consume tokens
python -m scripts.demo_retrieval

# Ejecutar los 8 casos de la Fase 3 (prompt básico vs. optimizado)
python -m scripts.demo_fase3

# Chat interactivo, con el detalle interno visible
python -m scripts.chat --debug
```

Dentro del chat: `/debug` `/modo` `/pedidos` `/costo` `/salir`.

Consultas de ejemplo para probar:

```
¿Cómo va mi pedido ECO-10001?
Mi pedido ECO-10003 ya se pasó de la fecha, ¿qué pasó?
Quiero saber del pedido ECO-99999
Compré la miel del pedido ECO-10008 y ya no la quiero, ¿cómo la devuelvo?
El shampoo del pedido ECO-10009 llegó partido y con moho
¿Cuánto tarda un envío a Leticia?
```

---

## Estructura del repositorio

```
Entregables/sesion2/
├── README.md                     ← este archivo
├── requirements.txt
├── .env.example
│
├── docs/                         ← FASES 1 y 2 (y la explicación de la 3)
│   ├── fase1-seleccion-y-justificacion.md
│   ├── fase2-fortalezas-limitaciones-riesgos.md
│   └── fase3-ingenieria-de-prompts.md
│
├── data/                         ← fuentes autorizadas (grounding data)
│   ├── pedidos.json              ← 12 pedidos de prueba (dato estructurado)
│   ├── politicas_devolucion.md   ← indexado en la base vectorial
│   ├── catalogo_productos.md     ← indexado en la base vectorial
│   └── faq_envios.md             ← indexado en la base vectorial
│
├── src/
│   ├── config.py                 ← configuración y carga de la API key
│   ├── llm_client.py             ← cliente del LLM + conteo de tokens y costo
│   ├── retriever.py              ← RAG: chunking, híbrida, re-ranking, expansión
│   ├── herramientas.py           ← function calling contra el OMS
│   ├── prompts.py                ← catálogo de prompts (básicos y optimizados)
│   ├── guardrails.py             ← PII, escalamiento, verificación de anclaje
│   └── chain.py                  ← orquestación de la cadena
│
├── scripts/
│   ├── demo_fase3.py             ← ejecuta los 8 casos y genera la transcripción
│   ├── demo_retrieval.py         ← inspecciona el retrieval, sin API
│   └── chat.py                   ← chat interactivo
│
└── outputs/
    └── ejecucion_fase3.md        ← generado por demo_fase3.py
```

---

## Las tres decisiones de diseño que sostienen la solución

### 1. RAG en vez de fine-tuning, por una razón operativa y otra ética

EcoMarket crece rápido: el catálogo, los precios y las políticas cambian cada
semana. El fine-tuning congela el conocimiento en los pesos y actualizarlo toma
días. RAG lo mantiene en documentos que se editan y reindexan en minutos.

La razón ética pesa igual: un dato personal incorporado a los pesos de un modelo
**no se puede borrar selectivamente**, y el titular tiene derecho a la supresión
(Ley 1581 de 2012). Mantener el conocimiento fuera del modelo es lo que hace ese
derecho ejercible.

### 2. RAG no es para todo

| Dato | Mecanismo | Por qué |
|---|---|---|
| Políticas, catálogo, FAQ | **RAG** (vector + BM25 + re-ranking) | La consulta del cliente no coincide léxicamente con el documento |
| Estado de un pedido | **Function calling** | Un pedido se busca por clave primaria. La similitud semántica podría devolver el ECO-10002 cuando se pidió el ECO-10012 |
| Tono de marca | **Prompt de sistema** | Es lo único que el fine-tuning enseña bien, y no cambia cada semana |

Vectorizar los pedidos habría sido el error de diseño más caro del proyecto:
impreciso, y además obligaría a replicar datos personales fuera del sistema
transaccional.

### 3. Los controles críticos son deterministas, no instrucciones al modelo

Un prompt es una instrucción que el modelo *puede* no seguir; un `if` no. El
escalamiento a un humano, el enmascaramiento de PII y el acceso de solo lectura
al OMS están implementados como código, fuera del modelo. Un cliente que menciona
la SIC o un cobro duplicado se escala **antes** de que se gaste un solo token.

---

## Conceptos de la Clase 2 aplicados en el código

| Concepto de la clase | Dónde está implementado |
|---|---|
| Chunking de los documentos base | `retriever.py::trocear_markdown` (por sección, con solape de 150 caracteres) |
| Modelo de embeddings | `retriever.py` (`text-embedding-3-small`, o TF-IDF local sin costo) |
| Búsqueda vectorial | `retriever.py::_scores_vectoriales` |
| Búsqueda híbrida | `retriever.py::buscar` (60% vectorial + 40% BM25) |
| Optimización de la búsqueda | `retriever.py::buscar_con_expansion` (query expansion sobre códigos de política) |
| Re-ranking | `retriever.py::_rerank` (cobertura + sección + identificadores exactos) |
| Cantidad de documentos a recuperar | `config.py` (`top_k=8` → `top_n=4`, por el *context window*) |
| Aumento del prompt | `chain.py` + `prompts.py` |
| Trade-off con/sin RAG | Demostrado en `outputs/ejecucion_fase3.md`: misma consulta, con y sin contexto |

---

## Limitaciones declaradas de esta entrega

Está en el espíritu del taller decir también lo que **no** es:

- La base vectorial es **en memoria** (TF-IDF o embeddings + coseno exacto), no
  un motor con índice HNSW. La arquitectura es la misma; la escala no. En
  producción sería pgvector, Pinecone o Azure AI Search.
- El re-ranker es una **heurística**, no un *cross-encoder*. Se documenta como
  tal en `retriever.py`.
- La verificación de anclaje es una **señal de monitoreo**, no una garantía
  anti-alucinación.
- Los datos de `data/` son **ficticios**, creados para el taller.
- Las cifras de costo usan **precios de lista de referencia** declarados en
  `src/llm_client.py`; deben verificarse antes de usarse en un caso de negocio.
- No hay un *golden set* de evaluación automatizada: es la primera deuda técnica
  que asumiría el proyecto, y así se declara en la Fase 2.

---

## Fuentes

1. Ávila, D. (2026). *Clase 2 — RAG: Retrieval Augmented Generation*. ICESI.
2. AWS. [*¿Qué es RAG?*](https://aws.amazon.com/es/what-is/retrieval-augmented-generation/)
3. Microsoft Learn. [*RAG and generative AI — Azure AI Search*](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview?tabs=docs)
4. Aryani, A. [*A Brief Introduction to RAG*](https://medium.com/@amiraryani/a-brief-introduction-to-retrieval-augmented-generation-rag-4bd6e50da532). Medium.
5. Congreso de Colombia. *Ley 1480 de 2011* (Estatuto del Consumidor) y *Ley 1581 de 2012* (habeas data).
