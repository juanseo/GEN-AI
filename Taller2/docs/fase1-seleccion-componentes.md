# Fase 1 — Selección de Componentes Clave del Sistema RAG

**Caso:** EcoMarket — Sistema RAG para la atención al cliente
**Taller Práctico #2 — IA Generativa — Maestría en Inteligencia Artificial, ICESI**

---

## 0. La decisión en una tabla

| Componente | Elección | Alternativa preparada | Razón principal |
|---|---|---|---|
| **Modelo de embeddings** | **`bge-m3`** (BAAI, open-source, licencia MIT), servido localmente con Ollama | `text-embedding-3-small` (OpenAI), activable por configuración y como respaldo automático | Fue el mejor en español coloquial en nuestra evaluación (88% de aciertos en primera posición contra 62% del otro modelo abierto y 58% de la línea base), sin costo por consulta y sin sacar texto de la infraestructura |
| **Base de datos vectorial** | **ChromaDB** persistente (índice HNSW, distancia coseno) | pgvector (ruta de producción del Taller 1), Weaviate o Pinecone si cambia la escala | La base de conocimiento de EcoMarket es pequeña (decenas a miles de fragmentos). Chroma da filtros por metadata y por texto, persistencia y *upsert* sin operar un servidor adicional |
| LLM de generación *(sin cambios)* | **`glm-5.3-flash`** (Ollama Cloud, la misma del Taller 1) | `gpt-4o-mini` (OpenAI) o `llama3.2:3b` local | Decisión del Taller 1: si el dato correcto llega en el prompt, basta un modelo de gama económica |
| Framework | LangChain 1.x (LCEL) | — | Abstrae embeddings y base vectorial detrás de interfaces comunes: cambiar cualquiera de los dos es cambiar una clase |

---

## 1. Qué le exige EcoMarket a estos dos componentes

Los requisitos salen del caso y de los requisitos no funcionales que fijamos en el
Taller 1 (Fase 1, sección 2):

| Requisito | De dónde viene | Qué le exige al embedding | Qué le exige a la base vectorial |
|---|---|---|---|
| Clientes que escriben en **español coloquial colombiano** ("¿me cobran el domicilio?", "el tinto calientico") | Caso: chat, correo y redes | Entender sinónimos y regionalismos, no solo palabras exactas | — |
| Vocabulario técnico propio: SKU (`ECO-MIE-500`), códigos de política (`NO_DEV_PERECEDERO`) | Documentos de la empresa | No perder los identificadores exactos | Filtro por texto exacto y por metadata |
| ~3.000 consultas diarias, 80% repetitivas; picos de 10× | Caso + RNF-4 | Latencia baja y costo marginal casi nulo por consulta | Búsqueda en milisegundos, escalable |
| Cambios semanales de catálogo y políticas; vigentes en < 1 hora | Caso + RNF-5 | Reindexar barato | *Upsert* por ID, persistencia, reindexación incremental |
| Trazabilidad de cada respuesta | RNF-6 | — | Guardar texto y metadata junto al vector, para citar la fuente |
| Ley 1581 de 2012 (datos personales) | RNF-7 | Preferiblemente procesar el texto **dentro** de la infraestructura | Residencia de datos controlable |
| Equipo pequeño, empresa en crecimiento | Caso | Operación simple | Operación simple, sin un clúster que mantener |

---

## 2. Cómo impacta cada componente en la eficacia del RAG

Antes de comparar productos conviene tener claro qué decide cada pieza, porque de
eso dependen los criterios:

- **El modelo de embeddings decide qué es "parecido".** Convierte la pregunta y los
  fragmentos en vectores, y la búsqueda devuelve los fragmentos cuyos vectores
  quedan más cerca. Si el modelo no sabe que «¿le puedo pagar al mensajero?» se
  parece a «¿puedo pagar contra entrega?», el fragmento correcto nunca llega al
  prompt, y ningún LLM ni prompt puede compensarlo: el modelo responde sin el dato
  (alucina) o dice que no sabe. Determina el **recall** del sistema.
  También decide qué tan bien se puede detectar una pregunta **fuera de alcance**,
  porque el umbral de similitud depende de cómo distribuye el modelo sus puntajes
  (sección 3.3).
- **La base vectorial decide cómo se busca, no qué es parecido.** Con el mismo
  modelo, una búsqueda exacta y una aproximada (HNSW) devuelven prácticamente el
  mismo orden. Medimos que, a nuestra escala, Chroma no pierde ningún vecino
  (sección 4.4). Lo que la base aporta a la eficacia son las **capacidades
  alrededor de la similitud**: filtros por metadata, filtros por texto exacto,
  *upsert*, persistencia y escalabilidad. En este proyecto una de esas
  capacidades, el filtro por texto, subió el acierto de 92% a 96% (sección 4.4).

Por eso evaluamos el modelo de embeddings **con datos** (es lo que cambia la
calidad) y la base vectorial **por capacidades y operación** (es lo que cambia el
costo y el riesgo).

---

## 3. Modelo de embeddings

### 3.1 Candidatos

| Modelo | Tipo | Dimensiones | Contexto máx. | Idiomas | Costo | Dónde corre |
|---|---|---|---|---|---|---|
| TF-IDF de n-gramas de caracteres | Léxico (línea base del Taller 1) | ~20.000 (disperso) | Ilimitado | Cualquiera, sin semántica | $0 | En memoria |
| `paraphrase-multilingual` (mpnet-base-v2, sentence-transformers) | Denso, open-source (Apache 2.0) | 768 | 512, pero Ollama lo usa con **128** tokens | 50+ | $0 por consulta | Ollama local, CPU |
| **`bge-m3`** (BAAI) | Denso, open-source (MIT), 567 M parámetros | 1.024 | **8.192** | **100+** | $0 por consulta | Ollama local, CPU |
| `text-embedding-3-small` | Denso, propietario (OpenAI) | 1.536 | 8.191 | Multilingüe (MIRACL 44,0 vs. 31,4 de ada-002) | USD 0,02 / millón de tokens | API de OpenAI |
| `multilingual-e5` (intfloat) | Denso, open-source | 384–1.024 | 512 | 100 | $0 | *Considerado, no ejecutado* (ver sección 6) |

Sobre la pregunta del enunciado, **¿código abierto o propietario?**: la respuesta
para EcoMarket es **código abierto, servido localmente**, con el propietario como
respaldo. Las razones están en las secciones 3.2 a 3.4. En resumen: rinde mejor en
nuestro caso, cuesta cero por consulta y los textos no salen de la empresa.

### 3.2 Evaluación empírica en español

Construimos un *golden set* de **26 preguntas reales en español coloquial**,
escritas a propósito sin el vocabulario de los documentos ("¿le puedo pagar en
efectivo al mensajero?" en lugar de "pago contra entrega"), más **8 preguntas fuera
de alcance**. Medimos sobre la base de conocimiento de la Fase 2 (55 fragmentos,
*chunking* estructural). Script: `python -m scripts.evaluar_retrieval`; resultado
completo en [`outputs/evaluacion_retrieval.md`](../outputs/evaluacion_retrieval.md).

| Modelo | hit@1 | hit@4 | MRR | Latencia por consulta (CPU) |
|---|---|---|---|---|
| TF-IDF (Taller 1) | 58% | 88% | 0,713 | 1 ms |
| `paraphrase-multilingual` | 62% | 85% | 0,732 | 28 ms |
| **`bge-m3`** | **88%** | **92%** | **0,894** | 32 ms |

*hit@1*: el primer fragmento recuperado es el correcto. *hit@4*: el correcto está
entre los 4 que llegan al prompt (`top_k = 4`). *MRR*: promedio de 1/posición del
primer acierto.

Qué explica la diferencia:

- **bge-m3 entiende la intención y no solo las palabras.** «¿A qué horas puedo
  hablar con una persona de verdad?» recupera el horario de atención en primera
  posición. TF-IDF trae primero la ficha del cepillo de dientes, porque comparte
  fragmentos de palabras.
- **`paraphrase-multilingual` trunca.** En Ollama trabaja con una ventana de 128
  tokens (~500 caracteres): de una sección de la política solo "ve" el comienzo.
  Es consistente con sus fallos: «¿quién paga el envío si devuelvo algo porque no
  me gustó?» se responde con la penúltima fila de la tabla de la sección 5, y el
  modelo no la recupera ni entre los 10 primeros. Un modelo con ventana corta obligaría a partir en fragmentos mucho
  más pequeños, con los problemas que muestra la Fase 2.
- **TF-IDF no es tan malo en hit@4 (88%)** porque los documentos comparten
  vocabulario con muchas preguntas. Pero pone el fragmento correcto de primero
  solo en el 58% de los casos, así que el contexto llega con más ruido.

`text-embedding-3-small` no se midió en la corrida incluida en el repositorio (ver
sección 6). Al ejecutar el mismo script con `OPENAI_API_KEY` configurada, el reporte
lo agrega automáticamente a la tabla.

### 3.3 Detección de preguntas fuera de alcance

El enunciado pide que el asistente atienda cualquier solicitud **o reconozca que no
puede**. La primera barrera es un umbral: si el mejor fragmento recuperado tiene
una similitud menor, el asistente responde un mensaje fijo sin llamar al LLM.

| Modelo | Similitud del mejor fragmento, preguntas en alcance | Ídem, fuera de alcance | Umbral | Fuera de alcance atajadas por el umbral |
|---|---|---|---|---|
| TF-IDF | 0,119 – 0,384 | 0,097 – 0,202 | 0,06 | 0% |
| `paraphrase-multilingual` | 0,398 – 0,786 | 0,137 – 0,634 | 0,34 | 50% |
| **`bge-m3`** | 0,462 – 0,684 | 0,255 – 0,651 | **0,41** | **38%** |

Tres lecciones:

1. **El umbral depende del modelo.** Cada modelo distribuye sus similitudes en un
   rango distinto (TF-IDF nunca pasa de 0,4, `paraphrase` llega a 0,79). Un umbral
   copiado de otro proyecto no sirve: hay que calibrarlo. Por eso vive en
   `src/config.py` junto a cada modelo.
2. **Ningún umbral separa todo.** «¿Tienen vacantes de trabajo en EcoMarket?»
   obtiene 0,651 con bge-m3, porque menciona a EcoMarket y se parece a «¿tienen
   tiendas físicas?». Por eso el umbral se calibró para **no rechazar ninguna
   pregunta legítima**: la similitud mínima en alcance (0,462) menos una holgura
   de 0,05 → **0,41**. Las que se cuelan las ataja la **segunda barrera**: el
   prompt estricto, que ordena decir "no tengo esa información" si el contexto no
   responde. Lo verificamos en la Fase 3.

   La holgura tiene historia. La primera calibración usó 0,01 (umbral 0,45). En
   la demo, una variante de redacción de una pregunta del *golden set*, «¿Qué
   tienen para llevar el tinto calientico a la oficina?», obtuvo 0,44 y el
   asistente la rechazó como fuera de alcance. Con un conjunto de evaluación tan
   pequeño, el umbral no puede quedar pegado al mínimo observado.
3. **Rechazar una pregunta válida es peor que dejar pasar una inválida.** La
   primera deja al cliente sin respuesta; la segunda solo cuesta una llamada al LLM
   que termina diciendo que no sabe.

### 3.4 Costo, escalabilidad y ventajas/desventajas

**Costo.** Con 90.000 consultas al mes y ~20 tokens por consulta, vectorizar las
preguntas con `text-embedding-3-small` costaría **≈ USD 0,04 al mes**. Indexar toda
la base (55 fragmentos, ~15.000 tokens) cuesta fracciones de centavo. **El costo
directo no decide entre las dos opciones.** Lo que sí pesa es el costo indirecto: la
dependencia de un proveedor, la latencia de red en cada consulta y la salida de
datos. bge-m3 necesita un servidor con ~2 GB de RAM; en CPU responde en ~30 ms por
consulta, holgado para 3.000 consultas diarias e incluso para picos de 10×.

| Criterio | `bge-m3` (elegido) | `text-embedding-3-small` |
|---|---|---|
| Rendimiento en español (nuestro *golden set*) | **88% hit@1** | No medido en esta corrida; MIRACL multilingüe 44,0 (dato de OpenAI) |
| Costo por consulta | $0 (costo fijo del servidor) | ≈ $0,0000004 |
| Latencia | ~30 ms, local | Ida y vuelta por internet (típicamente 100–300 ms) |
| Privacidad (Ley 1581) | El texto no sale de la infraestructura | El texto de cada consulta viaja a un tercero |
| Dependencia | Ninguna: pesos abiertos con licencia MIT | *Vendor lock-in*: si el modelo se retira, hay que reindexar con otro |
| Contexto máximo | 8.192 tokens | 8.191 tokens |
| Operación | Hay que operar Ollama (o un servicio de inferencia) | Cero operación |
| Escalabilidad | Vertical/horizontal a cargo de EcoMarket | Elástica, la escala el proveedor |
| Riesgo técnico encontrado | Requiere un *runtime* de inferencia que funcione en el equipo (sección 6) | Requiere API key y conectividad |

**Decisión:** `bge-m3`. Gana en la métrica que determina la calidad (recuperar el
fragmento correcto en español coloquial), no tiene costo marginal y cumple mejor la
minimización de datos. Su desventaja, operar el servidor del modelo, es aceptable
porque Ollama ya sirve otros modelos en la misma máquina. `text-embedding-3-small`
queda como respaldo automático: si Ollama no responde, el sistema cambia a OpenAI,
con su propia colección y su propio umbral, todavía provisional (0,35, no calibrado
en el golden set; recalibrar con `scripts/evaluar_retrieval.py`). El respaldo vive
en `src/embeddings.py::resolver_embeddings`.

---

## 4. Base de datos vectorial

### 4.1 Qué necesita EcoMarket de la base vectorial

- **Escala pequeña:** hoy 55 fragmentos. Aun con 12.000 SKU y toda la
  documentación, la base estaría en el orden de **decenas de miles de vectores**, no
  millones.
- **Pocas consultas por segundo:** 3.000 al día equivalen a ~0,03 por segundo en
  promedio, y unas pocas por segundo en un pico.
- **Filtros:** por tipo de documento, categoría o SKU (metadata) y por código exacto
  (texto).
- **Actualización frecuente:** *upsert* por ID y reindexación en minutos.
- **Operación simple y costo bajo:** EcoMarket no tiene un equipo de plataforma.

### 4.2 Comparación

| Criterio | **ChromaDB** (elegida) | Pinecone | Weaviate | pgvector (propuesta del Taller 1) |
|---|---|---|---|---|
| Modelo de despliegue | Embebida en el proceso o cliente-servidor; open-source (Apache 2.0) | SaaS gestionado (serverless) | Open-source auto-hospedado o nube gestionada | Extensión de PostgreSQL |
| Escalabilidad | Hasta millones de vectores en un nodo; sin *sharding* nativo en la versión open-source | La más alta: escala elástica y gestionada | Alta: *sharding* y replicación | Buena hasta millones con índice HNSW; escala como escala Postgres |
| Costo | $0 en licencia; corre en el mismo servidor del asistente | Pago por uso / plan; el más caro a gran escala | $0 auto-hospedado (más el costo de operarlo) o pago gestionado | $0 adicional si ya existe Postgres |
| Facilidad de uso | **Muy alta:** `pip install`, sin servidor, persistencia en disco | Alta: API simple, pero exige cuenta, red y API key | Media: esquema, módulos y configuración del clúster | Media: SQL e índices; requiere acceso al Postgres de producción |
| Filtros por metadata | Sí (`where`) | Sí | Sí (GraphQL / filtros) | Sí (SQL `WHERE`, el más expresivo) |
| Búsqueda por texto / híbrida | Filtro de texto (`where_document`), sin BM25 nativo | Híbrida con vectores dispersos | **Híbrida BM25 + vector nativa** | Full-text de Postgres + vector, combinados a mano |
| Residencia de datos | Local | En la nube del proveedor | Local o nube | Local |
| Integración con LangChain | `langchain-chroma` | `langchain-pinecone` | `langchain-weaviate` | `langchain-postgres` |

**Ventajas y desventajas para EcoMarket:**

- **ChromaDB.** A favor: arranca en segundos, sin infraestructura nueva. Tiene
  justo las capacidades que necesitamos (coseno, HNSW, metadata, filtro de texto,
  *upsert*) y los datos quedan en casa. En contra: no trae búsqueda híbrida BM25
  nativa y el modo embebido no está pensado para varias réplicas del asistente
  escribiendo a la vez. Para eso habría que pasar a su modo cliente-servidor o a
  otra base.
- **Pinecone.** A favor: cero operación y escala prácticamente ilimitada. En contra:
  para unas decenas de miles de vectores es pagar capacidad que no se usa. Además,
  los textos de la política y el catálogo quedan en un tercero y el sistema depende
  de la conectividad con un proveedor más.
- **Weaviate.** A favor: la búsqueda híbrida nativa reemplazaría el BM25 que el
  Taller 1 tuvo que implementar a mano. En contra: operar un clúster (o pagarlo) es
  desproporcionado para la escala actual.
- **pgvector.** A favor: no introduce un sistema nuevo si EcoMarket ya opera
  PostgreSQL, y permite combinar vectores con SQL. En contra: acopla el asistente a
  la base de datos transaccional, que es justo la que el Taller 1 decidió no tocar
  más que en modo lectura. Además, las pruebas y el desarrollo requieren acceso a
  esa base.

### 4.3 Decisión y ruta de migración

**ChromaDB persistente**, en el mismo servidor del asistente. Esto ajusta la
recomendación del Taller 1, que proponía pgvector sin haber comparado opciones. Hay
un argumento que baja el riesgo de la elección: **la base vectorial es un índice
derivado, no un sistema de registro.** Los documentos fuente viven en sus sistemas
(PDF, ERP, CMS) y reconstruir el índice completo toma ~8 segundos. Migrar a otra
base es cambiar la clase `Chroma` por la de otra integración de LangChain y
reindexar, no migrar datos.

| Señal | Migrar a |
|---|---|
| Varias réplicas del asistente o alta disponibilidad | Chroma en modo cliente-servidor, o pgvector |
| Se quiere consolidar en el PostgreSQL existente | pgvector |
| Se necesita búsqueda híbrida BM25 nativa a escala | Weaviate |
| Millones de vectores (p. ej., indexar reseñas o el historial) sin equipo de operación | Pinecone |

### 4.4 Lo que medimos de la base vectorial

- **HNSW no pierde calidad a esta escala.** El acierto entre los 4 primeros con el
  índice HNSW de Chroma (92%) es igual al de la búsqueda exacta.
- **Los filtros sí cambian la calidad.** La expansión por código de política, una
  búsqueda filtrada por metadata (`tipo = politica`) y por texto exacto (`$contains`
  el código del producto), sube el acierto de **92% a 96%**. Rescata la pregunta
  que el modelo solo no resolvía («el shampoo me llegó roto y con hongos»; ver
  Fase 2, sección 3).
- **Persistencia y *upsert*.** Reabrir el índice toma ~3 s y reindexar todo ~8 s
  en CPU. Un cambio de política queda vigente en el siguiente arranque, porque el
  índice detecta por hash que las fuentes cambiaron.

---

## 5. Cómo encaja en la arquitectura del Taller 1

La arquitectura híbrida de dos niveles del Taller 1 no cambia. Lo que cambia es la
pieza que estaba marcada como limitación ("la base vectorial es en memoria, TF-IDF"):

| Pieza | Taller 1 | Taller 2 |
|---|---|---|
| Modelo de embeddings | TF-IDF de caracteres (o `text-embedding-3-small` opcional) | **`bge-m3`** open-source, local |
| Almacenamiento y búsqueda | Matriz en memoria, coseno exacto | **ChromaDB** persistente, HNSW, coseno |
| Documentos | 3 Markdown (13 productos, 7 FAQ) | **PDF + CSV + JSON** (22 productos, 23 FAQ, política completa) |
| *Chunking* | Por encabezado de Markdown | **Estructural por tipo de documento** (Fase 2) |
| Fuera de alcance | Solo el prompt | **Umbral de similitud calibrado + prompt estricto** |
| Expansión de consulta | Regex sobre códigos, en memoria | **Filtros de metadata y texto de Chroma** |
| Orquestación | SDK de OpenAI, llamadas manuales | **LangChain LCEL** (`prompt | llm | parser`, `bind_tools`) |
| Pedidos, guardrails, router | Function calling, reglas, LLM | Sin cambios de fondo (se agregó la intención `PAGO_CUENTA`) |

---

## 6. Supuestos y limitaciones de esta selección

- **El *golden set* es pequeño** (26 + 8 preguntas). Las diferencias grandes, como
  88% contra 58–62% o la del *chunking*, son claras; las de pocos puntos no son
  concluyentes. En producción, el *golden set* debe crecer con preguntas reales del
  copiloto interno (Taller 1, hoja de ruta, fase 1).
- **`multilingual-e5` no se pudo ejecutar.** La política de Control de aplicaciones
  de Windows del equipo de desarrollo bloquea la librería nativa `tokenizers`, de
  la que dependen `transformers` y `sentence-transformers`. En lugar de modificar
  una política de seguridad del sistema, servimos los modelos abiertos con Ollama,
  que trae su propio *runtime*. Es también un hallazgo de operación: un modelo
  open-source "gratis" exige un *runtime* que funcione en la infraestructura real.
- **`text-embedding-3-small` no se midió** en la corrida incluida, porque no se
  contó con una API key de OpenAI en el entorno de desarrollo. La comparación con él
  se apoya en los datos públicos del proveedor. El script lo evalúa automáticamente
  cuando la key está disponible.
- **Las mediciones son en CPU** de un portátil. En un servidor, las latencias
  bajan; las métricas de calidad no cambian.

---

## Fuentes

1. Ávila, D. (2026). *Clase 2 — RAG: Retrieval Augmented Generation*. ICESI.
2. BAAI. *bge-m3* — tarjeta del modelo (100+ idiomas, 8.192 tokens, 1.024 dimensiones, MIT).
   https://huggingface.co/BAAI/bge-m3
3. Chen, J. et al. (2024). *M3-Embedding: Multi-Linguality, Multi-Functionality,
   Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. arXiv:2402.03216.
4. OpenAI. *text-embedding-3-small* — documentación del modelo y precios.
   https://developers.openai.com/api/docs/models/text-embedding-3-small
5. Vercel AI Gateway. *text-embedding-3-small* — MTEB 62,3; MIRACL 44,0 vs. 31,4 de ada-002.
   https://vercel.com/ai-gateway/models/text-embedding-3-small/faq
6. Reimers, N. y Gurevych, I. (2020). *Making Monolingual Sentence Embeddings Multilingual
   using Knowledge Distillation* (modelos `paraphrase-multilingual`). EMNLP.
7. Documentación de ChromaDB, Pinecone, Weaviate y pgvector, e integraciones de LangChain.
8. Congreso de Colombia. *Ley 1581 de 2012* — protección de datos personales.
