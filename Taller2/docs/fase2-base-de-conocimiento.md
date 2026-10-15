# Fase 2 — Creación de la Base de Conocimiento de Documentos

**Caso:** EcoMarket — Sistema RAG para la atención al cliente
**Taller Práctico #2 — IA Generativa — Maestría en Inteligencia Artificial, ICESI**

---

## 1. Por qué esta fase decide la calidad del sistema

Un sistema RAG no puede responder mejor que lo que recupera. El LLM recibe cuatro
fragmentos y una instrucción: "responde solo con esto". Si el fragmento correcto no
está entre esos cuatro, el modelo tiene dos opciones, y las dos son malas: decir que
no sabe (aunque la empresa sí tenga la respuesta) o completar con su conocimiento
general (alucinar).

La evaluación de este taller lo muestra con números. Con **el mismo modelo de
embeddings (bge-m3), la misma base vectorial y los mismos documentos**, solo
cambiando la forma de partirlos, el porcentaje de preguntas cuyo primer fragmento
recuperado es el correcto va de **38% a 88%** (sección 4.3). Ninguna decisión de la
Fase 1 mueve tanto la aguja como esta.

Dos propiedades de los documentos determinan ese resultado:

- **Calidad:** texto limpio, completo y autosuficiente. Un encabezado de página
  repetido o un código interno sin explicar degradan el vector del fragmento.
- **Segmentación:** cada fragmento debe contener **una** idea completa. Si mezcla
  dos, su vector queda en un punto intermedio que no se parece a ninguna pregunta;
  si parte una idea, ninguna de las mitades responde.

---

## 2. Identificación de documentos

### 2.1 Los tres documentos que se indexan

Elegimos los tres tipos que sugiere el enunciado porque cubren las intenciones del
80% repetitivo del caso, y porque **cada uno llega en un formato distinto**: así la
base de conocimiento enfrenta los problemas reales de un e-commerce, donde la
información no vive en un solo sistema.

| # | Documento | Formato | Dueño en la empresa | Cambia | Responde preguntas de… | Archivo |
|---|---|---|---|---|---|---|
| 1 | Política de devoluciones, cambios y reembolsos | **PDF** (documento oficial firmado) | Dirección de Experiencia al Cliente | Trimestral | Devoluciones, retracto, garantías, reembolsos, cancelaciones, escalamiento | `data/politicas_devolucion.pdf` |
| 2 | Catálogo de productos / inventario | **CSV** (exportación de la hoja de cálculo del ERP) | Comercial | Semanal | Precio, materiales, categoría, política de cada SKU, disponibilidad | `data/catalogo_productos.csv` |
| 3 | Preguntas frecuentes del centro de ayuda | **JSON** (exportación del CMS) | Servicio al cliente | Mensual | Envíos, pagos, facturación, cuenta y datos personales, sostenibilidad, canales | `data/faq.json` |

Respecto al Taller 1, la base de conocimiento creció para que el asistente pueda
atender **cualquier tipo de solicitud** y no solo los dos ejercicios de ese taller:

- El catálogo pasó de 13 a **22 productos**. Ahora cubre todas las categorías de
  la política: alimentos, higiene, textiles, garantía, productos vivos,
  personalizados y digitales.
- La FAQ pasó de 7 preguntas de envíos a **23 preguntas** en 5 categorías:
  envíos, pagos, cuenta, sostenibilidad y atención.
- La política pasó de Markdown a **PDF**, que es como la publica la empresa.

### 2.2 Lo que **no** se indexa, y por qué

| Dato | Dónde vive | Por qué no va a la base vectorial |
|---|---|---|
| Estado de los pedidos (`pedidos.json`) | Sistema de pedidos (OMS) | Es exacto, cambia cada hora y tiene datos personales. Se busca por clave primaria con *function calling*: la similitud semántica podría devolver el pedido ECO-10002 cuando se pidió el ECO-10012 (decisión del Taller 1, que se mantiene). |
| Stock en tiempo real | Inventario del ERP | El campo `disponibilidad` del CSV es una **foto** con fecha de corte. Para "¿cuántas unidades quedan?" la fuente correcta sería una herramienta contra el inventario, igual que con los pedidos. |
| Historial de conversaciones | CRM | Contiene datos personales. Indexarlo replicaría PII fuera del sistema que la custodia (Ley 1581 de 2012). |

La regla que aplicamos: **a la base vectorial solo va conocimiento no estructurado,
semi-estable y sin datos personales.** Lo exacto y volátil se consulta con
herramientas.

---

## 3. Calidad de los documentos: problemas encontrados y cómo se trataron

Cada formato trajo un problema distinto. Los tratamos en el **cargador**
(`src/cargadores.py`), antes del *chunking*, porque un error en la entrada se
propaga a cada fragmento.

### 3.1 PDF: ruido de página, secciones partidas y tablas rotas

El PDF de la política (`scripts/generar_pdf_politica.py` lo genera como lo
publicaría la empresa) tiene los tres problemas típicos de un PDF real:

| Problema | Efecto si no se trata | Tratamiento |
|---|---|---|
| Encabezado y pie repetidos en cada hoja ("Documento controlado…", "Página 2") | El mismo texto en todos los fragmentos los hace parecerse entre sí y confunde la búsqueda | Se eliminan por patrón (`_RUIDO_PDF`) |
| Secciones que cruzan páginas: la tabla de exclusiones queda en la página 1 y su **excepción obligatoria** en la página 2 | Si se partiera por página, la regla quedaría separada de su excepción | Se unen las páginas en un solo texto antes de partir |
| Las tablas se extraen **una celda por línea** ("Alimentos perecederos y / frescos / NO_DEV_PERECEDERO / …") | El fragmento de la sección 3 queda como una lista de palabras sueltas y su vector pierde significado | No se pudo reparar de forma genérica. Se mitigó en el retrieval (sección 3.4) |

El tercer problema tuvo un efecto medible. La consulta «el shampoo me llegó roto y
con hongos, ¿qué opciones tengo?» debería recuperar la *excepción obligatoria*
(producto defectuoso → reposición total), pero ningún tamaño de fragmento logró que
apareciera entre los 10 primeros (probamos 1.200, 800, 600 y 400 caracteres). En
producción la solución de fondo es **mejorar el documento**: extraer las tablas con
una herramienta que respete su estructura (pdfplumber, Docling) o indexar la fuente
editable en lugar del PDF. Es la lección principal de esta fase: la calidad del
documento limita lo que cualquier modelo puede hacer.

### 3.2 CSV: un registro no es un texto

Una fila como `ECO-MIE-500,Miel orgánica…,Alimentos,54000,NO_DEV_PERECEDERO,…`
no se parece semánticamente a ninguna pregunta. El cargador convierte cada fila en
una **ficha legible** y conserva los campos como metadata filtrable:

```
Producto: Miel orgánica de abejas nativas 500 g (SKU ECO-MIE-500)
Categoría: Alimentos
Precio: COP 54.000
Política de devolución: NO_DEV_PERECEDERO, no admite devolución por retracto (alimento perecedero)
Disponibilidad: Disponible
Descripción: Miel cruda sin pasteurizar…
```

Fíjate en la línea de política. El código `NO_DEV_PERECEDERO` es un identificador
interno que ningún cliente escribe. El cargador le agrega su significado en
lenguaje natural ("no admite devolución…"), así que la ficha queda cerca, en el
espacio vectorial, de «¿me la reciben de vuelta?». Es **enriquecimiento del
fragmento**: cada fragmento debe entenderse sin los demás.

### 3.3 JSON: la estructura ayuda

El JSON del centro de ayuda ya viene dividido en pares pregunta–respuesta, y la
pregunta está redactada como la escribiría un cliente, que es exactamente lo que
mejor se parece a una consulta. El cargador solo aplana cada par a texto y guarda
`id`, `categoria` y `actualizado` como metadata.

### 3.4 Mitigación en el retrieval: expansión por código de política

Para compensar la tabla rota del PDF, la rama de devoluciones del asistente hace una
segunda búsqueda con dos filtros nativos de la base vectorial. Si entre los
fragmentos recuperados hay fichas de producto, se trae la sección de la política
que **contiene** su código (`where_document={"$contains": "NO_DEV_HIGIENE"}`,
`filter={"tipo": "politica"}`). Así la sección 3, con su excepción, llega al
contexto de la pregunta del shampoo
(`src/rag.py::SistemaRAG.expandir_con_politica`). Es la misma expansión de consulta
del Taller 1, ahora resuelta por la base de datos y no en memoria.

---

## 4. Segmentación (*chunking*)

### 4.1 Estrategias consideradas

| Estrategia | Cómo parte | Ventaja | Desventaja para EcoMarket |
|---|---|---|---|
| **Tamaño fijo** | Cada *N* caracteres, sin mirar el contenido | Simple, fragmentos homogéneos | Corta palabras y reglas a la mitad; mezcla el final de un producto con el inicio del siguiente |
| **Por párrafos** | En los saltos de párrafo | Respeta unidades de redacción | El PDF extraído no tiene líneas en blanco entre párrafos: "párrafo" degenera en "bloque de líneas" y se pierden las secciones. En el CSV y el JSON no hay párrafos |
| **Recursiva** | Intenta separar por párrafo → línea → oración → palabra hasta cumplir el tamaño | Respeta la estructura cuando puede, con un tope de tamaño | Sola, no sabe qué es una sección de la política ni qué es un registro |
| **Estructural (elegida)** | Por la unidad natural de cada documento, y recursiva **solo** si esa unidad excede el tope | Cada fragmento es una idea completa y autosuficiente | Requiere un cargador por formato (es el costo de hacerlo bien) |

### 4.2 La estrategia elegida, por documento

| Documento | Unidad | Tamaño | Solape | Metadata | Fragmentos |
|---|---|---|---|---|---|
| Política (PDF) | **Sección numerada** (encabezado + 9 secciones); recursiva si excede el tope | Tope 1.200 caracteres (~300 tokens) | 150 caracteres, solo si una sección se subdivide | `fuente`, `tipo`, `seccion`, `version`, `paginas`, `id` | 10 |
| Catálogo (CSV) | **Un producto** (una fila) | 280–420 caracteres | Ninguno: cada ficha es independiente | `sku`, `categoria`, `politica`, `precio_cop`, `id` | 22 |
| FAQ (JSON) | **Una pregunta** con su respuesta | 150–600 caracteres | Ninguno | `id`, `categoria`, `actualizado` | 23 |

Por qué estos valores:

- **Tope de 1.200 caracteres.** Todas las secciones de la política caben enteras
  (la más larga, la 3, tiene ~1.180). Una regla y sus condiciones viajan juntas al
  prompt. El límite real lo pone el modelo de embeddings: bge-m3 acepta 8.192
  tokens, pero `paraphrase-multilingual` se usa en Ollama con 128 tokens (~500
  caracteres) y **trunca** todo lo que sobra. Con ese modelo, el vector de una
  sección de 1.200 caracteres solo representa su comienzo. El tamaño del fragmento
  y el modelo de embeddings se deciden juntos.
- **Solape solo dentro de una sección subdividida.** El solape existe para no
  perder una idea que cae en el corte. Entre fichas de producto o preguntas de la
  FAQ no hay idea compartida que proteger: el solape solo duplicaría texto de un
  registro ajeno.
- **Encabezado contextual.** Cada fragmento de la política empieza con
  `Política de devoluciones de EcoMarket > 3. Categorías NO sujetas a devolución`.
  Si una sección se partiera, la segunda mitad seguiría diciendo de dónde viene,
  tanto para el embedding como para el LLM que la lee.
- **Pasos numerados que no son secciones.** El procedimiento (sección 8) tiene
  pasos "1.", "2.", …, que un separador ingenuo confundiría con secciones nuevas.
  El chunker solo acepta como título la línea con el **número de sección que
  sigue**, corta y sin punto final (`_es_titulo_de_seccion`). Hay una prueba que
  lo verifica (`tests/test_chunking.py`).

### 4.3 Evidencia: la estrategia cambia el resultado

Evaluación con 26 preguntas reales redactadas como las escribiría un cliente
(`data/evaluacion/golden_set.json`), con el modelo bge-m3. *hit@1* = el primer
fragmento recuperado es el correcto; *hit@4* = el correcto está entre los 4 que
llegan al prompt. Detalle en [`outputs/evaluacion_retrieval.md`](../outputs/evaluacion_retrieval.md).

| Estrategia | Fragmentos | hit@1 | hit@4 | MRR |
|---|---|---|---|---|
| **Estructural (elegida)** | 55 | **88%** | **92%** | **0,894** |
| Tamaño fijo (500 caracteres) | 42 | 58% | 88% | 0,704 |
| Por párrafos (~400 caracteres) | 58 | 38% | 62% | 0,505 |

Los fallos explican los números:

- **Tamaño fijo.** «¿puedo devolver la agenda que mandé a grabar con mi nombre?»
  no recupera la ficha de la agenda entre los 10 primeros: el corte a 500
  caracteres la partió y mezcló sus mitades con los productos vecinos.
- **Párrafos.** Junta en un mismo bloque preguntas de la FAQ que no tienen relación
  (pagos con cuenta, envíos con estados). Su vector es un promedio que no se
  parece a ninguna consulta concreta.

---

## 5. Indexación

### 5.1 El proceso

```
 data/politicas_devolucion.pdf ─┐
 data/catalogo_productos.csv  ──┼─► 1. CARGAR       src/cargadores.py
 data/faq.json                ──┘      limpiar ruido · fila→ficha · par→texto
                                              │
                                              ▼
                                       2. PARTIR         src/chunking.py
                                          sección · producto · pregunta
                                          + encabezado contextual + metadata + id estable
                                              │  55 Document(page_content, metadata)
                                              ▼
                                       3. VECTORIZAR     src/embeddings.py
                                          bge-m3 (Ollama local) → 55 vectores de 1.024 dimensiones
                                              │
                                              ▼
                                       4. CARGAR EN LA BASE VECTORIAL   src/indexador.py
                                          ChromaDB · colección "ecomarket_bge-m3_estructural"
                                          índice HNSW · distancia coseno · upsert por id
                                              │
                                              ▼
                                       5. REGISTRAR      indice_chroma/manifiesto.json
                                          huella (hash) de las fuentes, fecha, n.º de fragmentos
```

En código, los pasos 3 y 4 son una sola línea de LangChain: `Chroma` llama al
modelo de embeddings por cada lote de fragmentos y guarda vector, texto y metadata:

```python
vectorstore = Chroma(collection_name=coleccion, embedding_function=modelo,
                     client=chromadb.PersistentClient(path="indice_chroma"),
                     collection_configuration={"hnsw": {"space": "cosine"}})
vectorstore.add_documents(chunks, ids=[c.metadata["id"] for c in chunks])
```

Indexar los 55 fragmentos con bge-m3 en CPU toma unos 8 segundos
(`python -m scripts.indexar`).

### 5.2 Decisiones de la indexación

| Decisión | Por qué |
|---|---|
| **Distancia coseno** | Los modelos de embeddings de texto se entrenan para que el ángulo entre vectores mida el parecido semántico. Con coseno, el *score* que devuelve Chroma es directamente la similitud (1 − distancia) y se puede comparar contra un umbral. |
| **IDs estables** (`catalogo#ECO-MIE-500`, `faq#PAG-02`, `politica#s3-p0`) | Chroma hace *upsert*: si el precio de la miel cambia, se reemplaza ese vector y no se duplica. Además, el ID es lo que el LLM cita en su respuesta, así que cada afirmación es trazable a su fuente. |
| **Una colección por modelo de embeddings** | Los vectores de bge-m3 y los de OpenAI viven en espacios distintos, y compararlos entre sí no tiene sentido. Cambiar de modelo implica reindexar. |
| **Reindexación automática** | El manifiesto guarda un hash de las fuentes. Si alguien edita el CSV o publica una nueva versión del PDF, el índice se reconstruye solo la próxima vez que arranca el asistente. Así se cumple el requisito del Taller 1 de que un cambio de política esté vigente en minutos (RNF-5). |
| **Metadata filtrable** | Permite búsquedas restringidas (`tipo = politica`, una categoría de FAQ, un SKU) sin otro sistema. Es la base de la expansión de la sección 3.4. |

### 5.3 Mantenimiento de la base

| Documento | Disparador de actualización | Responsable |
|---|---|---|
| Política (PDF) | Nueva versión aprobada | Experiencia al Cliente publica el PDF y la reindexación es automática |
| Catálogo (CSV) | Exportación nocturna del ERP | Proceso automático (en producción, un *job* programado) |
| FAQ (JSON) | Publicación en el CMS | Servicio al cliente |

En los tres casos, la versión y la fecha viajan en la metadata del fragmento, de
modo que una respuesta siempre se puede auditar contra el documento vigente en ese
momento.

---

## Fuentes

1. Ávila, D. (2026). *Clase 2 — RAG: Retrieval Augmented Generation*. ICESI.
2. LangChain. *Text splitters* — `RecursiveCharacterTextSplitter`, `CharacterTextSplitter`.
   https://python.langchain.com/docs/concepts/text_splitters/
3. Chroma. *Collections, metadata filtering and full-text search* (`where`, `where_document`).
   https://docs.trychroma.com/
4. Chen, J. et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality,
   Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. arXiv:2402.03216.
5. Congreso de Colombia. *Ley 1581 de 2012* — protección de datos personales.
