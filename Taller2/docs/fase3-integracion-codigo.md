# Fase 3 — Integración y Ejecución del Código

**Caso:** EcoMarket — Sistema RAG para la atención al cliente
**Taller Práctico #2 — IA Generativa — Maestría en Inteligencia Artificial, ICESI**

---

## 1. Qué se construyó

Se ajustó el asistente del Taller 1 para que su conocimiento salga de un **sistema
RAG real** construido con **LangChain**: la base de conocimiento de la Fase 2,
vectorizada con el modelo de embeddings y guardada en la base vectorial elegidos en
la Fase 1. El código está en dos niveles:

| Nivel | Archivo | Para qué |
|---|---|---|
| **RAG mínimo** | [`scripts/rag_ejemplo.py`](../scripts/rag_ejemplo.py) | Cumple el papel del `rag_ejemplo.py` del enunciado: un archivo lineal en 5 secciones que muestra cómo se conectan las piezas |
| **Asistente completo** | [`src/asistente.py`](../src/asistente.py) y módulos de `src/` | La cadena del Taller 1 (guardrails, router, *function calling*) con el RAG en el lugar del TF-IDF |

Comandos (detalle en el [README](../README.md)):

```bash
python -m scripts.rag_ejemplo          # RAG mínimo
python -m scripts.demo_taller2         # casos y experimentos → outputs/
python -m scripts.chat --debug         # chat con el detalle interno
```

---

## 2. Cómo se conectan las piezas

Un sistema RAG tiene dos tiempos. Las piezas que elegimos en la Fase 1 aparecen en
los dos:

```
 TIEMPO DE INDEXACIÓN (una vez, y cada vez que cambia un documento)
 ──────────────────────────────────────────────────────────────────
 PDF / CSV / JSON ─► cargadores ─► chunking ─► [ bge-m3 ] ─► vectores ─► [ ChromaDB ]
                     (limpieza)   (Fase 2)     embed_documents          upsert por id
                                                                        HNSW · coseno

 TIEMPO DE CONSULTA (cada mensaje del cliente)
 ──────────────────────────────────────────────────────────────────
 pregunta ─► [ bge-m3 ] ─► vector ─► [ ChromaDB ] ─► top-4 fragmentos + similitud
             embed_query            búsqueda ANN            │
                                                            ▼
                                          ¿similitud máx. ≥ 0,41?
                                    no ◄───────┴───────► sí
                                    │                    │
                    mensaje fijo "no tengo        ┌──── CHAINING (LCEL) ─────────────┐
                    esa información"              │ ChatPromptTemplate               │
                    (0 tokens)                    │   sistema + {contexto}+{consulta}│
                                                  │        │                         │
                                                  │        ▼                         │
                                                  │ ChatOpenAI (glm-5.3-flash)      │
                                                  │        │                         │
                                                  │        ▼                         │
                                                  │ StrOutputParser → texto          │
                                                  └──────────────────────────────────┘
                                                                   │
                                                                   ▼
                                                   respuesta con citas [faq#PAG-02]
```

**El mismo modelo de embeddings tiene que vectorizar los documentos y las
preguntas.** Si se indexa con bge-m3 y se consulta con OpenAI, los vectores viven en
espacios distintos y la similitud no significa nada. Por eso el índice guarda una
colección por modelo (`src/indexador.py`).

En LangChain, cada pieza es una abstracción intercambiable, y por eso cambiar de
modelo o de base es cambiar una línea:

| Pieza | Abstracción de LangChain | Implementación en este proyecto |
|---|---|---|
| Fragmento | `Document(page_content, metadata)` | `src/cargadores.py`, `src/chunking.py` |
| Modelo de embeddings | `Embeddings` (`embed_documents`, `embed_query`) | `OllamaEmbeddings("bge-m3")`, `OpenAIEmbeddings`, `TfidfEmbeddings` (`src/embeddings.py`) |
| Base vectorial | `VectorStore` | `Chroma` (`src/indexador.py`) |
| Recuperador | `VectorStoreRetriever` / `similarity_search_with_relevance_scores` | `src/rag.py` |
| Prompt | `ChatPromptTemplate` | `src/prompts.py` |
| LLM | `BaseChatModel` | `ChatOpenAI` (OpenAI, Ollama Cloud u Ollama local) (`src/llm.py`) |
| Herramienta | `StructuredTool` + `llm.bind_tools` | `consultar_pedido` (`src/herramientas.py`) |
| Composición | `Runnable` con el operador `\|` (LCEL) | `prompt \| llm \| StrOutputParser()` |

---

## 3. El RAG mínimo, sección por sección (`scripts/rag_ejemplo.py`)

### Sección 1 — Documentos → fragmentos

```python
chunks = construir_chunks(estrategia="estructural")
```

Carga los tres documentos y los parte con la estrategia de la Fase 2: la política
por sección, el catálogo por producto y la FAQ por pregunta. Resultado: 55 objetos
`Document`, cada uno con su texto y una metadata que incluye un `id` estable
(`catalogo#ECO-CEP-BAM`). Ese `id` sirve tres veces: Chroma lo usa para el
*upsert*, el LLM lo cita en la respuesta y la evaluación lo usa para auditar.

### Sección 2 — Fragmentos → vectores

```python
nombre_embeddings, embeddings = resolver_embeddings(CONFIG.embeddings)
vector = embeddings.embed_query("¿puedo devolver un producto?")   # 1.024 números
```

`resolver_embeddings` devuelve el modelo configurado (bge-m3 en Ollama local). Si
Ollama no responde, cae a OpenAI. `embed_query` muestra lo que hace el modelo: una
frase entra y salen 1.024 números. Dos frases con significado parecido quedan con
vectores cercanos, aunque no compartan palabras.

### Sección 3 — Vectores → base vectorial

```python
vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embeddings,
    ids=[c.metadata["id"] for c in chunks],
    collection_name="rag_ejemplo",
    collection_configuration={"hnsw": {"space": "cosine"}},
)
```

`from_documents` hace dos cosas: llama a `embeddings.embed_documents` sobre los 55
textos y guarda en Chroma cada vector con su texto y su metadata. `space: cosine`
define la métrica del índice HNSW. Con ella, el *score* de relevancia que devuelve
LangChain es la similitud coseno (1 − distancia), comparable contra un umbral. En
el ejemplo la colección vive en memoria; el asistente usa la versión persistente
con reindexación automática (`src/indexador.py::abrir_indice`).

### Sección 4 — Consulta → contexto

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.41},
)
```

El *retriever* es la interfaz que la cadena usa para buscar. Vectoriza la pregunta
con el **mismo** modelo, pide a Chroma los 4 vecinos más cercanos y descarta los
que no superan el umbral calibrado. Con «¿Cuál es la capital de Australia?» no
devuelve nada (la mejor similitud es 0,25), y esa es la señal de que la pregunta
está fuera de alcance.

### Sección 5 — Contexto → respuesta (*chaining*)

```python
cadena = (
    {"contexto": retriever | formatear, "consulta": RunnablePassthrough()}
    | prompt          # ChatPromptTemplate: SISTEMA + RAG_GENERAL
    | crear_llm()     # ChatOpenAI (el modelo activo de config.py)
    | StrOutputParser()
)
cadena.invoke("¿Le puedo pagar en efectivo al mensajero?")
```

Así se lee la cadena LCEL, de izquierda a derecha:

1. El diccionario corre **dos ramas en paralelo** con la misma pregunta. La rama
   `contexto` pasa la pregunta por el *retriever* y convierte los `Document`
   recuperados en texto con sus identificadores (`formatear`). La rama `consulta`
   deja la pregunta tal cual (`RunnablePassthrough`).
2. `prompt` recibe `{contexto, consulta}` y arma los mensajes: el prompt de sistema
   del Taller 1 (rol, reglas, formato) y la plantilla con el contexto delimitado
   entre `<contexto>…</contexto>`.
3. El LLM genera un mensaje y `StrOutputParser` extrae el texto.

Este es el **aumento** de la Generación Aumentada por Recuperación: el modelo no
responde con lo que memorizó, responde con lo que la cadena le puso en el prompt.

---

## 4. Del ejemplo al asistente: cómo se ajustó el modelo del Taller 1

`src/asistente.py` conserva el flujo del Taller 1 (el mismo del diagrama
`cadena-ecomarket`) y cambia lo que el RAG mejora:

```
mensaje → enmascarar PII → ¿escalar? (reglas) → router (LLM)
   ├─ ESTADO_PEDIDO → llm.bind_tools([consultar_pedido]) → datos verificados → respuesta
   ├─ DEVOLUCION    → RAG + expansión por código de política + datos del pedido → árbol de decisión
   ├─ QUEJA         → agente humano
   └─ PRODUCTO / ENVIO / PAGO_CUENTA / OTRO → RAG general con umbral
                                                   → verificación de anclaje
```

| Cambio | Taller 1 | Taller 2 | Dónde |
|---|---|---|---|
| Recuperación | TF-IDF + BM25 en memoria | bge-m3 + ChromaDB persistente | `src/rag.py`, `src/indexador.py` |
| Fuera de alcance | Solo una regla en el prompt | **Umbral de similitud** antes del LLM + regla en el prompt (en devoluciones, el umbral aplica si no hay un pedido verificado) | `src/rag.py::supera_umbral`, `asistente.py::_rama_general` y `_rama_devolucion` |
| Expansión de consulta | Regex sobre el texto recuperado | **Filtros nativos de Chroma**: `filter={"tipo": "politica"}` + `where_document={"$contains": código}` | `src/rag.py::SistemaRAG.expandir_con_politica` |
| Llamadas al LLM | SDK de OpenAI, a mano | Cadenas LCEL; uso de tokens con `UsageMetadataCallbackHandler` | `src/asistente.py`, `src/llm.py` |
| Herramienta de pedidos | Esquema JSON escrito a mano | `StructuredTool` + `bind_tools` (el esquema se infiere de la firma) | `src/herramientas.py` |
| Router | 6 intenciones | + `PAGO_CUENTA`, porque la nueva FAQ cubre pagos, facturas y datos personales | `src/prompts.py::ROUTER` |
| Proveedores de LLM | OpenAI, Ollama Cloud | + **Ollama local** sin API key (`ECOMARKET_PROVEEDOR=local`) | `src/config.py` |

**Cómo sigue una consulta de devolución por el código.** Para «El shampoo del
pedido ECO-10009 me llegó partido y con moho»:

1. `enmascarar_pii` no encuentra datos personales.
2. `evaluar_escalamiento` no encuentra disparadores.
3. El router responde `DEVOLUCION`.
4. `normalizar_tracking` extrae `ECO-10009` y `consultar_pedido` trae el pedido
   (shampoo en barra, `ENTREGADO`).
5. El nombre del producto enriquece la consulta de recuperación.
6. Chroma devuelve la ficha `catalogo#ECO-SHA-BAR` (0,71) y otras.
7. `expandir_con_politica` ve los códigos de esos productos y trae la sección 3
   (`politica#s3-p0`, la de la *excepción obligatoria*).
8. El prompt `DEVOLUCION` aplica el árbol de decisión del Taller 1: defectuoso →
   excepción de garantía.

---

## 5. Modificaciones al código y al prompt, y su efecto observado

La rúbrica pide modificar el código o el prompt y observar el cambio. Cada
experimento de `scripts/demo_taller2.py` (sección E) cambia **una sola** variable.
Los resultados citados son de
[`outputs/ejecucion_taller2_local.md`](../outputs/ejecucion_taller2_local.md),
ejecutado con el LLM local `llama3.2:3b` (ver sección 6). La redacción exacta
cambia entre corridas (temperatura 0,2 y un modelo pequeño); la recuperación, los
umbrales y los tokens de entrada son deterministas. La corrida oficial
([`outputs/ejecucion_taller2.md`](../outputs/ejecucion_taller2.md)) se genera con el
mismo comando y ahora corre `glm-5.3-flash` (Ollama Cloud), el proveedor por defecto;
con `ECOMARKET_PROVEEDOR=openai`, corre `gpt-4o-mini`.

### 5.1 Quitar el RAG (sección A de la demo)

La misma pregunta, el mismo modelo, con y sin la base de conocimiento:

| Pregunta | Sin RAG | Con RAG |
|---|---|---|
| ¿Le puedo pagar en efectivo al mensajero? | Dice que no, pero **inventa** alternativas: pagar en efectivo "en nuestra tienda física" (EcoMarket no tiene) o con PayPal (no lo acepta) | "No ofrecemos pago contra entrega ni pago en efectivo al recibir", citando `faq#PAG-02` |
| Agenda con mi nombre grabado, ¿la puedo devolver? | **Inventa** "un reembolso parcial o un descuento en tu próxima compra" | No es devolvible por ser personalizada, y pregunta si llegó defectuosa (excepción de garantía) |
| ¿Programa para reciclar frascos? | **Inventa** un programa con "10% de descuento por frasco" | Programa Vuelve a Casa: cupón de COP 5.000 por cada 5 envases (`faq#SOS-03`) |

Es la alucinación que describe el enunciado: el modelo general no conoce la empresa
y completa con lo plausible. Con RAG, cada dato tiene un fragmento que lo respalda.

### 5.2 `top_k` = 1 vs. 4 (E.1)

«¿Cuánto me cuesta el envío a Bogotá y en cuántos días me llega?» necesita dos
fragmentos: el costo está en `faq#ENV-02` y el tiempo en `faq#ENV-01`.

- `top_k = 1`: solo llega `ENV-01`. Sin el dato del costo, el modelo lo confunde
  con el tiempo ("el costo del envío… varía entre 2 y 4 días hábiles") y remite a
  la página de envíos.
- `top_k = 4`: llegan ambos y aparece el costo (gratis sobre COP 150.000; tarifa de
  COP 14.000). Pero también llega ruido: el cuarto fragmento (`politica#s6-p0`,
  tiempos de reembolso) termina en la respuesta sin que nadie lo preguntara. El
  prompt crece de 948 a 1.329 tokens y la salida de 130 a 318.

`top_k` cambia recall por costo y ruido. Con poco contexto el modelo improvisa; con
mucho, se distrae. 4 es el punto que elegimos para preguntas que combinan dos
fragmentos, y un re-ranker (sección 7) ayudaría a filtrar el que sobra.

### 5.3 Prompt estricto vs. permisivo (E.2 y E.3)

Se cambia **una sola instrucción**: "usa únicamente el contexto" pasa a "si no
alcanza, complétalo con tu conocimiento general".

- «¿Cuántos años dura la batería de la lámpara solar?» (la ficha no lo dice). Con el
  prompt estricto, el modelo reconoce que no tiene el dato y ofrece un agente
  humano. Con el permisivo, presenta "8 horas de luz" como si fuera la "vida útil"
  de la batería: una respuesta plausible y falsa.
- «¿Cuál es la capital de Australia?» sin umbral. En una corrida previa (no
  incluida en el repositorio, sobrescrita por la final), el prompt permisivo
  respondió "la capital de Australia es Canberra **[faq#ENV-02]**": contestó fuera
  del rol del asistente y además **inventó una cita**. En la corrida incluida no lo
  hizo. Que el resultado cambie entre corridas es en sí la razón para no confiar la
  contención solo al prompt.

### 5.4 Umbral de relevancia activado vs. desactivado (E.3)

| Configuración | Tokens (entrada/salida) | Resultado |
|---|---|---|
| Umbral activado (0,41) | **0 / 0** | Mensaje fijo: "no encontré información sobre eso…" |
| Sin umbral, prompt estricto | 1.204 / 67 | El modelo dice que no puede ayudar (llegó a la misma conclusión pagando ~1.200 tokens) |
| Sin umbral, prompt permisivo | 1.164 / 55 | Variable entre corridas (ver 5.3) |

El umbral convierte una decisión probabilística (lo que el LLM decida hacer con
contexto irrelevante) en una determinista, y además es gratis.

### 5.5 Recalibrar el umbral después de observar un error

La primera calibración puso el umbral en 0,45, justo debajo de la pregunta legítima
con menor similitud del *golden set*. En la demo, «¿Qué tienen para llevar el tinto
calientico a la oficina?», una variante de redacción de una pregunta del *golden
set*, sacó 0,44 y **fue rechazada como fuera de alcance**. Se cambió la holgura de
calibración de 0,01 a 0,05 (`scripts/evaluar_retrieval.py::calibrar_umbral`) y el
umbral quedó en **0,41**. La pregunta ahora pasa. El costo: el umbral ataja el 38%
de las preguntas fuera de alcance en lugar del 50%, y el resto queda a cargo del
prompt estricto.

### 5.6 Cambiar el modelo de embeddings y el *chunking* (E.4 y evaluación)

- Con TF-IDF (Taller 1), «¿A qué horas puedo hablar con una persona de verdad?»
  trae primero la ficha del cepillo de dientes (0,12). Con bge-m3 trae el horario de
  atención (`faq#ATN-01`, 0,59).
- En todo el *golden set*: bge-m3 88% de hit@1 contra 58% de TF-IDF. Con el mismo
  modelo, el *chunking* estructural llega a 88% contra 58% del tamaño fijo y 38% de
  párrafos. Detalle en las Fases 1 y 2 y en
  [`outputs/evaluacion_retrieval.md`](../outputs/evaluacion_retrieval.md).

### 5.7 Agregar la expansión por código de política

Sin la expansión, «el shampoo me llegó roto y con hongos» no recupera la excepción
de garantía en los 10 primeros resultados. Con la expansión (dos filtros de Chroma),
la sección 3 entra al contexto, y el acierto entre los fragmentos que llegan al
prompt sube de 92% a **96%**.

---

## 6. Limitaciones y suposiciones

El enunciado pide declarar las limitaciones de recursos que impidieron ejecutar la
propuesta tal cual, y los supuestos para entender la arquitectura.

### 6.1 Limitaciones de recursos y cómo se resolvieron

| Limitación | Efecto | Qué se hizo | Supuesto para entender la arquitectura |
|---|---|---|---|
| **No hubo API key de OpenAI** en el entorno de desarrollo | No se pudo ejecutar `gpt-4o-mini` ni medir `text-embedding-3-small` | Se agregó el proveedor `local` (`llama3.2:3b` en Ollama, sin key ni costo) y con él se ejecutó y verificó la cadena completa (`outputs/ejecucion_taller2_local.md`). `python -m scripts.demo_taller2` y `python -m scripts.evaluar_retrieval` producen la corrida oficial apenas exista `OPENAI_API_KEY` | En producción, el LLM es `gpt-4o-mini` (decisión del Taller 1). El modelo de 3B solo demuestra que la cadena funciona; **su calidad no representa la del sistema propuesto** |
| **Windows Control de aplicaciones** bloquea la DLL de `tokenizers` | No corren `sentence-transformers` ni `transformers`, así que no se pudo probar `multilingual-e5` | Los modelos open-source se sirven con **Ollama**, que trae su propio *runtime*. No se modificó la política de seguridad del equipo | En producción, bge-m3 se sirve igual (Ollama o un servidor de inferencia equivalente) en el servidor del asistente |
| **Solo CPU**, en un portátil | Latencias mayores que en un servidor | Se midió igual: ~30 ms por consulta para el embedding, ~8 s para indexar todo | Las métricas de calidad no dependen del hardware |

### 6.2 Lo que mostró el LLM pequeño (y por qué importa)

La corrida con `llama3.2:3b` sirvió también como **prueba de estrés de los
guardrails**, porque un modelo pequeño falla más:

- **D.1:** el modelo inventó una fecha de entrega (2026-10-10). La verificación de
  anclaje del Taller 1 lo detectó: `anclaje: REVISAR (cobertura 0.5)`.
- **C.3 «Se me dañó la lavadora»:** pasó el umbral (0,47, se parece a la sección de
  garantías) y el modelo, en lugar de decir que EcoMarket no vende lavadoras,
  inventó una devolución de la compostera. **La segunda barrera (el prompt) depende
  de la capacidad del LLM.** Es la razón para no reemplazar `gpt-4o-mini` por un
  modelo pequeño en producción, y para mantener la verificación de anclaje y el
  escalamiento como controles fuera del modelo.
- **Router:** clasificó «¿le puedo pagar en efectivo al mensajero?» como `ENVIO` en
  lugar de `PAGO_CUENTA`. No afectó la respuesta, porque ambas intenciones van al RAG
  general, pero muestra que la clasificación no es perfecta.
- **Variabilidad:** el mismo experimento (E.3 permisivo) dio respuestas distintas en
  dos corridas.

### 6.3 Supuestos

- **"Hoy" es el 2026-09-22**, la fecha de corte de `pedidos.json`
  (`ECOMARKET_FECHA` la cambia). Con la fecha real, todos los pedidos de prueba
  parecerían vencidos.
- **Los datos son ficticios**, creados para el taller. El PDF se genera desde su
  fuente Markdown (`scripts/generar_pdf_politica.py`) para reproducir los problemas
  de un PDF real.
- **La disponibilidad del catálogo es una foto.** El stock en tiempo real se
  consultaría con una herramienta, igual que los pedidos.
- **Chroma corre embebido en un solo proceso.** Con varias réplicas del asistente
  habría que pasar a su modo cliente-servidor o a pgvector (Fase 1, sección 4.3).
- **El *golden set* es pequeño** (26 + 8 preguntas). Las diferencias grandes son
  claras; las pequeñas no son concluyentes. El umbral de 0,41 es una calibración
  inicial que hay que revisar con tráfico real.
- **Los precios** de `src/llm.py` son de lista, de referencia; verificar antes de un
  caso de negocio.

### 6.4 Fallos conocidos del retrieval

- **«¿Pueden mandarle un pedido a mi hermana que vive en Miami?»** no recupera la
  FAQ de envíos internacionales con ningún modelo: ninguno conecta "Miami" con
  "internacional" con suficiente fuerza. La respuesta resultante es segura (no
  encuentra el dato y ofrece un agente), pero no ideal. Mejora posible: reescribir
  la consulta con el LLM antes de buscar (*multi-query* o *HyDE*).
- **Tablas del PDF:** la extracción las rompe celda por celda (Fase 2, sección
  3.1). La expansión lo mitiga; la solución de fondo es mejorar el documento.
- **Preguntas cercanas al dominio** (vacantes en EcoMarket, la lavadora) pasan el
  umbral por diseño, y dependen del prompt y del LLM (sección 6.2).

---

## 7. Siguientes pasos

1. Medir `text-embedding-3-small` con `scripts/evaluar_retrieval.py` y recalibrar
   su umbral (0,35 provisional). La corrida oficial ya está generada con el
   proveedor por defecto: [`outputs/ejecucion_taller2.md`](../outputs/ejecucion_taller2.md),
   con `glm-5.3-flash`.
2. Hacer crecer el *golden set* con preguntas reales del copiloto interno y
   recalibrar el umbral.
3. Agregar un re-ranker *cross-encoder* multilingüe (por ejemplo `bge-reranker-v2-m3`)
   entre Chroma y el prompt. Los fallos de la sección 6.4 son del orden de los
   resultados, que es lo que corrige un re-ranker.
4. Reescritura de consulta con el LLM para casos como el de Miami.
