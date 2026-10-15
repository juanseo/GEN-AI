# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es

Entregas de los talleres del curso de IA Generativa (Maestría en IA, Universidad ICESI). Cada taller vive en una carpeta autocontenida, con su README, sus dependencias y sus instrucciones; un taller nuevo es una carpeta propia más una fila en la tabla del `README.md` de la raíz. El caso de trabajo es el mismo en todos: un asistente de atención al cliente para EcoMarket (e-commerce ficticio colombiano, datos ficticios). El código de cada carpeta es la implementación ejecutable de la Fase 3 de ese taller; `Taller*/docs/` contiene los entregables escritos de las tres fases.

| Taller | Qué aporta | Carpeta |
|---|---|---|
| 1 | El asistente base: RAG en memoria (TF-IDF), function calling e ingeniería de prompts | [`Taller1/`](Taller1/) |
| 2 | Sistema RAG de verdad: embeddings `bge-m3`, ChromaDB, base PDF/CSV/JSON e integración con LangChain | [`Taller2/`](Taller2/) |

Convención de ejecución (los dos talleres): los scripts corren como módulos — `python -m scripts.<script>` desde la carpeta del taller; los comandos de cada sección asumen ese directorio de trabajo. Cada script inserta la raíz del taller en `sys.path` antes de sus imports `src.*`, por lo que también funcionan ejecutados como archivos sueltos desde cualquier directorio. La API key se resuelve en `config.py`: primero el entorno, luego el `.env` más cercano — el del taller o el de cualquier directorio superior. El `.env` está en `.gitignore`.

## Convenciones (ambos talleres)

- Todo el contenido — comentarios, docstrings, prompts, respuestas y mensajes de commit — está en español, y también la mayoría de los identificadores (`esfuerzo_razonamiento`, `motor_retrieval`). Mantener el idioma en código nuevo.
- Los docstrings de módulo documentan decisiones de diseño (el porqué, el alcance, referencias de la clase); hay que preservarlos al editar el código.
- Los archivos de `outputs/` son artefactos generados por scripts; no editarlos a mano. Ojo con `--solo`: en el Taller 1 sobrescribe el entregable completo con la corrida parcial (el original se recupera con `git restore`); en el Taller 2 escribe un archivo aparte con la sección (`ejecucion_taller2_<sección>.md`), sin tocar el entregable completo.
- Los pedidos de `pedidos.json` (transaccionales y con PII) **nunca se indexan** en ninguna base vectorial: se consultan por function calling. La similitud semántica podría devolver el pedido ECO-10002 cuando el cliente pidió el ECO-10012, y la base replicaría PII fuera del sistema transaccional.

---

## Taller 1 (`Taller1/`)

Asistente de atención al cliente para EcoMarket que combina RAG, function calling e ingeniería de prompts.

### Comandos

```bash
pip install -r requirements.txt            # openai, python-dotenv, scikit-learn, numpy

python -m scripts.demo_retrieval           # inspecciona el RAG: sin API key ni tokens
python -m scripts.demo_retrieval "puedo devolver la miel organica"  # consulta libre opcional
python -m scripts.demo_fase3               # los 8 casos de la Fase 3; regenera outputs/ejecucion_fase3.md (~24 llamadas, ~1 min)
python -m scripts.demo_fase3 --solo 3      # un solo caso (1-indexado)
python -m scripts.demo_fase3 --sin-basico  # omite el prompt básico (~7 de las ~24 llamadas)
python -m scripts.chat                     # chat interactivo (modo optimizado)
python -m scripts.chat --basico            # punto de comparación: sin RAG ni guardrails
python -m scripts.chat --debug             # muestra intención, fuentes, anclaje y llamadas
```

Comandos dentro del chat: `/salir` `/debug` `/modo` `/pedidos` `/costo` (también acepta `/exit` y `/quit`; `/modo` y `/debug` alternan en caliente, sin reiniciar).

No hay tests, lint ni build en este taller. La única verificación sin API key es `demo_retrieval` (con `ECOMARKET_RETRIEVAL=tfidf`, el valor por defecto); aun así exige las dependencias instaladas, porque todo import de `src.*` pasa por `src/__init__.py`, que reexporta `chain` y arrastra las cuatro. Un 401 (API key inválida o revocada) termina el script con un mensaje explicativo.

### Configuración

- `cp .env.example .env` (en `Taller1/`) y completar `OLLAMA_API_KEY` (se crea en https://ollama.com/settings/keys). El proveedor por defecto es Ollama Cloud: modelo `glm-5.3-flash` (endpoint `https://ollama.com/v1`, compatible con la API de OpenAI) y retrieval `tfidf` (offline y gratuito).
- OpenAI como alternativa: `ECOMARKET_PROVEEDOR=openai` → `gpt-4o-mini` con `OPENAI_API_KEY` (o `OPENAI-API-KEY`, el nombre que usa el `.env` del curso); habilita `ECOMARKET_RETRIEVAL=openai` para embeddings reales (`text-embedding-3-small`); esa opción solo toma efecto con el proveedor `openai` — en cualquier otra combinación el motor de retrieval cae silenciosamente a `tfidf`.
- Otras variables: `ECOMARKET_MODELO`, `ECOMARKET_MODELO_EMBEDDINGS`.

### Arquitectura

Decisión central: **arquitectura híbrida donde el conocimiento vive fuera del modelo**, con un mecanismo distinto por tipo de dato:

| Dato | Mecanismo | Módulo |
|---|---|---|
| Políticas, catálogo y FAQ (`data/*.md`, no estructurado) | RAG | `retriever.py` |
| Pedidos (`data/pedidos.json`, transaccional y con PII) | Function calling | `herramientas.py` |
| Tono de marca | Prompt de sistema | `prompts.py` |

Esa separación es una decisión de diseño, no un detalle: los pedidos nunca se indexan en la base vectorial (ver convenciones generales).

Flujo de una consulta en `AsistenteEcoMarket.responder()` (`src/chain.py`), modo optimizado:

1. `enmascarar_pii` (regex, `guardrails.py`) — minimización de PII antes de que el texto salga al proveedor.
2. `evaluar_escalamiento` (reglas) — si hay disparador (legal, salud, cobro duplicado, emocional, pedido explícito), se escala a humano **sin gastar un token**.
3. Router de intención — LLM con `prompts.ROUTER` (temperatura 0, 8 tokens): devuelve una etiqueta de {ESTADO_PEDIDO, DEVOLUCION, PRODUCTO, ENVIO, QUEJA, OTRO}; cualquier respuesta fuera del set se trata como OTRO.
4. Rama especializada: `ESTADO_PEDIDO` → `consultar_pedido` por function calling (con respaldo regex si el modelo no invoca la herramienta); `DEVOLUCION` → RAG + datos del pedido + `prompts.E2_OPTIMIZADO` (árbol de decisión y few-shot que incluye el caso negativo); `QUEJA` → escalamiento; el resto → RAG general con `RAG_GENERAL`.
5. `verificar_anclaje` — los datos duros de la respuesta (ECO-####, fechas, URLs, cifras) se verifican contra el contexto recuperado. Es señal de monitoreo, no garantía anti-alucinación.

Los controles críticos son deterministas (código, no instrucciones al modelo): "un prompt es una instrucción que el modelo puede no seguir; un `if` no". Modificar `guardrails.py` o el orden de esta pipeline cambia garantías de seguridad, no solo comportamiento.

### Dualidad básico/optimizado

Los prompts de los ejercicios (`SISTEMA`, `E1`, `E2`) existen dos veces (`*_BASICO` / `*_OPTIMIZADO`); `ROUTER` y `RAG_GENERAL` solo existen para el modo optimizado. `responder(modo="basico")` corta la pipeline: sin PII, sin router, sin RAG, sin guardrails. Ese modo existe solo para que la comparación de la Fase 3 sea medible (`outputs/ejecucion_fase3.md`); no es el modo normal.

### Independencia de proveedor

`src/llm_client.py` concentra toda la dependencia del proveedor: el SDK de OpenAI con `base_url` cambiada según `PROVEEDORES` en `config.py`. Cambiar de proveedor no toca la cadena de prompts. Detalle no obvio: los modelos de Ollama Cloud razonan y esos tokens cuentan contra `max_tokens`; por eso `margen_razonamiento` (+1024) se suma a cada límite — sin ese margen, el router (8 tokens) devuelve texto vacío. Ollama Cloud se cobra por suscripción, así que el costo se reporta como 0 aunque sí se consuman tokens (no confundir con "no consumió nada"); los precios de `PRECIOS_USD_POR_MILLON` aplican solo a modelos OpenAI.

### Capa de retrieval (`src/retriever.py`)

`BaseConocimiento` se construye una sola vez por proceso (el chunking y la vectorización son el costo fijo del arranque):

- Chunking por secciones `##` de Markdown, re-corte a ~900 caracteres con solape de 150; el título de la sección se repite dentro del chunk (mejora el retrieval y le da la cita al LLM).
- Búsqueda híbrida: 60% vectorial + 40% BM25 (implementado a mano), ambos normalizados a [0,1]. La señal vectorial es TF-IDF de n-gramas de caracteres (3–5); los embeddings de la API solo entran si se cumplen a la vez motor `openai` + proveedor `openai` + cliente disponible (`retriever.py`), y sin esas condiciones el motor cae silenciosamente a TF-IDF. Con embeddings de la API, construir `BaseConocimiento` hace esa llamada al arranque, fuera del manejo del 401 de los scripts.
- Re-ranking heurístico (sustituye al cross-encoder de producción): cobertura de términos + bonus por coincidencia con el título de sección + bonus fuerte por identificadores con dígitos (SKUs; los códigos de política `NO_DEV_*` no disparan ese bonus porque la normalización pasa todo a minúsculas y parte los guiones bajos).
- `buscar_con_expansion` (*query expansion*): lee códigos `NO_DEV_*`/`DEV_*` en los 2 primeros resultados y lanza una segunda búsqueda con ellos, para traer la sección de la política que cita la exclusión (si el cliente dice "quiero devolver la miel", sin expansión no llega el contexto que explica "alimentos perecederos").
- Los IDs de fragmento (`<documento>#NN`, contador secuencial por documento indexado) son el formato de cita que la cadena exige al modelo. Solo se indexan los archivos de `DOCUMENTOS_INDEXABLES`: un documento nuevo en `data/` no entra al índice sin avisar hasta añadirlo a esa tupla (`src/retriever.py`).
- El índice en memoria es una decisión de alcance del taller; en producción sería pgvector/Pinecone con cross-encoder.

### Convenciones propias

- Sin framework de agentes (no LangChain): la orquestación es Python plano y la única dependencia de red es el SDK `openai`.

---

## Taller 2 (`Taller2/`)

Sistema RAG para el mismo asistente: los "tres archivos Markdown con TF-IDF en memoria" del Taller 1 se reemplazan por una base de conocimiento de tres formatos (política en PDF, catálogo en CSV, FAQ en JSON), embeddings open-source servidos con Ollama local y ChromaDB; las cadenas se expresan con LangChain LCEL. El pipeline de guardrails/router/ramas/anclaje es el del Taller 1, reutilizado.

### Comandos

```bash
pip install -r requirements.txt            # con el .venv de Taller2/ (cómo crearlo en el README)

python -m pytest -q                        # 21 pruebas: cargadores, chunking, umbral (sin API key ni Ollama)
python -m scripts.indexar                  # construye indice_chroma/ (opcional: se indexa solo la primera vez)
python -m scripts.rag_ejemplo              # RAG mínimo en 5 secciones: el recorrido didáctico del código
python -m scripts.rag_ejemplo --sin-llm    # solo recuperación, sin API key
python -m scripts.evaluar_retrieval        # comparación de embeddings y chunking + calibración → outputs/evaluacion_retrieval.md
python -m scripts.demo_taller2             # demo de la Fase 3 → outputs/ejecucion_taller2.md
python -m scripts.demo_taller2 --solo A    # una sola sección (A–E)
python -m scripts.chat --debug             # chat interactivo, con el detalle interno visible
python -m scripts.generar_pdf_politica     # regenera data/politicas_devolucion.pdf desde data/fuentes/*.md
```

### Configuración (`src/config.py`)

- LLM por defecto: `ollama` → `glm-5.3-flash` en Ollama Cloud (`https://ollama.com/v1`, key `OLLAMA_API_KEY`, la misma del Taller 1). Alternativa: `ECOMARKET_PROVEEDOR=openai` → `gpt-4o-mini` con `OPENAI_API_KEY`. Respaldo: `ECOMARKET_PROVEEDOR=local` sin key ni costo → `llama3.2:3b` en el Ollama de la máquina (calidad de redacción menor; sirve para verificar la cadena sin proveedor externo).
- Embeddings por defecto: `bge-m3`, corriendo en el Ollama **local** (`http://localhost:11434`, variable `OLLAMA_HOST`), que es distinto de Ollama Cloud: requiere `ollama pull bge-m3`. Opciones con `ECOMARKET_EMBEDDINGS`: `bge-m3` | `paraphrase-multilingual` | `openai-3-small` (API de OpenAI; requiere su key) | `tfidf` (línea base del Taller 1). Cada modelo tiene su propia colección: cambiar de modelo implica reindexar.
- Si el Ollama local no responde o le falta el modelo, `resolver_embeddings` (`embeddings.py`) cae automáticamente a `openai-3-small` **solo** si hay key de OpenAI; si no, lanza `EmbeddingsNoDisponibles`. El nombre efectivo del modelo manda: colección, umbral y dimensiones cambian con él.
- El umbral de relevancia se calibra **por modelo** (`MODELOS_EMBEDDINGS[...]["umbral"]`): la mínima similitud coseno observada en las preguntas en alcance del golden set, menos 0,05 de holgura. Con bge-m3 (umbral 0,41) no se rechaza ninguna pregunta legítima. Excepción: el de `openai-3-small` (0,35) es provisional — no se calibró por falta de API key y queda pendiente (`scripts/evaluar_retrieval.py`).
- `ECOMARKET_FECHA` fija el "hoy" del asistente; por defecto es la fecha de corte de `pedidos.json` (con la fecha real, cada día que pasa los plazos de los pedidos de prueba vencerían).

### Arquitectura

El pipeline de una consulta es el del Taller 1 (secciones anteriores): PII → escalamiento por reglas → router → rama → anclaje. Lo que cambia es la capa de conocimiento:

- **Cargadores** (`src/cargadores.py`): `pypdf` + csv + json → `Document` de LangChain (no se usan los de `langchain-community`, en proceso de retiro). El del PDF quita encabezados/pies repetidos de cada hoja y une las páginas (la sección no puede partirse de la excepción que le sigue); la ficha de cada producto traduce su código de política a lenguaje natural (`RESUMEN_POLITICAS`) para que el fragmento sea autosuficiente.
- **Chunking** (`src/chunking.py`): la estrategia elegida es la **estructural** — política por sección numerada (sub-división recursiva con solape si excede 1.200 caracteres, encabezado contextual en cada fragmento), catálogo por producto, FAQ por pregunta. `parrafos` y `fijo` existen solo para el experimento comparativo de `evaluar_retrieval`.
- **Indexación** (`src/indexador.py`): ChromaDB persistente en `indice_chroma/` — no se versiona, está en `.gitignore` y se regenera desde `data/`. HNSW con distancia coseno; una colección por modelo de embeddings y estrategia de chunking; IDs estables (`catalogo#ECO-BOT-500`, `faq#ENV-02`, `politica#s3-p0`) con upsert; un **manifiesto guarda la huella SHA de los tres documentos fuente** — si cambia cualquiera, el índice se reconstruye solo al abrirlo la próxima vez.
- **Sistema RAG** (`src/rag.py`): retriever con similitud coseno → si el MEJOR fragmento (no el promedio) no supera el umbral del modelo, `MENSAJE_FUERA_DE_ALCANCE` (texto fijo, 0 tokens de generación); si pasa, cadena LCEL `prompt | llm | parser`. **Expansión por código de política**: sobre los fragmentos con metadata `politica`, una segunda búsqueda con filtro `tipo=politica` + `$contains` trae la sección de la política que explica la exclusión (la query expansion del Taller 1, ahora nativa de Chroma).
- **Asistente** (`src/asistente.py`): la intención nueva es `PAGO_CUENTA`. La rama `DEVOLUCION` enriquece la consulta de retrieval con los nombres de los productos del pedido verificado (así encuentra la ficha del SKU correcto y no solo la política genérica). El modo `sin_rag` (mismo modelo, sin acceso a la base) existe para la comparación A de la demo — el equivalente a la dualidad básico/optimizado del Taller 1.
- **Costos** (`src/llm.py`): `langchain_openai.ChatOpenAI` con `base_url` del proveedor activo — toda la dependencia del proveedor vive en `config.py` + `llm.py`. Los precios de `PRECIOS_USD_POR_MILLON` aplican solo a modelos OpenAI; Ollama Cloud es por suscripción (costo 0 reportado aunque sí se consuman tokens). El `margen_razonamiento` (+1024) sigue sumándose: GLM razona y esos tokens cuentan contra `max_tokens`.

### Ojo

- La corrida "oficial" (`outputs/ejecucion_taller2.md`) es la del proveedor cloud activo; la del respaldo `local` lleva sufijo (`outputs/ejecucion_taller2_local.md`) y también es entregable.
- La indexación usa el `bge-m3` del Ollama LOCAL: si `ollama list` no lo muestra, `ollama pull bge-m3` antes de ejecutar cualquier cosa que abra el índice.