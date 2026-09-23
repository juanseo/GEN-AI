# Fase 1 — Selección y Justificación del Modelo de IA

**Caso:** EcoMarket — Optimización de la atención al cliente
**Taller Práctico #1 — Sesión 2, IA Generativa — Maestría en Inteligencia Artificial, ICESI**

---

## 1. Del problema de negocio a los requisitos técnicos

Antes de elegir un modelo hay que traducir lo que dice el caso en requisitos
verificables. El enunciado entrega cinco hechos y cada uno restringe la decisión:

| Hecho del caso | Requisito técnico que impone | Consecuencia sobre la elección |
|---|---|---|
| Miles de consultas diarias por chat, correo y redes | Alto volumen, multicanal, con picos | El costo por consulta y la latencia dominan la decisión; descarta modelos premium para el tráfico masivo |
| 80% repetitivas: estado del pedido, devoluciones, características del producto | Consultas **acotadas y verificables** contra fuentes internas | El problema no es de creatividad sino de **acceso a datos**: es un problema de *retrieval*, no de capacidad del modelo |
| 20% complejas: quejas, problemas técnicos, sugerencias | Requieren empatía, criterio y responsabilidad legal | Exige un **router** y una ruta humana; automatizarlas es el riesgo más grande del proyecto |
| Tiempo de respuesta promedio: 24 horas | Objetivo: segundos para el 80% | Descarta arquitecturas con muchos saltos o con re-entrenamiento en el camino |
| Empresa en rápido crecimiento | El catálogo, los precios y las políticas cambian cada semana | **Descarta el fine-tuning como mecanismo de conocimiento**: reentrenar cada vez que cambia un precio es inviable |

**El requisito crítico es el último.** La información que necesita el asistente
(el estado de un pedido, el precio de un SKU, la política de devoluciones vigente)
es **volátil y exacta**. Un modelo que "sabe" esa información porque la memorizó
durante el entrenamiento queda desactualizado el día siguiente y, peor aún, no
puede decir que no sabe: la completa con una alucinación plausible.

---

## 2. Requisitos no funcionales que usaremos como criterio de decisión

| # | Requisito | Meta |
|---|---|---|
| RNF-1 | **Precisión factual** en datos de pedido y política | 0% de datos inventados; toda afirmación debe ser trazable a una fuente |
| RNF-2 | **Latencia** percibida por el cliente | p95 < 5 segundos |
| RNF-3 | **Costo** por consulta resuelta | < USD 0,002 (≈ COP 8) |
| RNF-4 | **Escalabilidad** ante picos (Black Friday, campañas) | 10× el tráfico base sin re-arquitectura |
| RNF-5 | **Actualización del conocimiento** | Un cambio de política debe estar vigente en el asistente en < 1 hora |
| RNF-6 | **Trazabilidad** | Toda respuesta debe poder citar el documento y la versión que la sustenta |
| RNF-7 | **Cumplimiento** (Ley 1581 de 2012, habeas data) | Minimización de datos personales enviados a terceros |
| RNF-8 | **Derivación segura** al humano | 100% de los casos legales, de salud o de cobro deben llegar a una persona |

---

## 3. Alternativas evaluadas

### Opción A — LLM de propósito general, sin acceso a datos (solo prompt)

Un GPT-4o o similar respondiendo con su conocimiento paramétrico.

- ✅ Integración trivial, disponible en horas.
- ❌ **Descartada.** No conoce ningún pedido de EcoMarket. Viola RNF-1 de forma
  estructural: ante "¿dónde está mi pedido ECO-10001?" su única salida es
  inventar o evadir. Es exactamente el escenario que el material de AWS describe
  como "generar información falsa cuando desconoce la respuesta".

### Opción B — Modelo pequeño afinado (*fine-tuning*) con datos de EcoMarket

Afinar un Llama 3 8B o un `gpt-4o-mini` con el histórico de conversaciones.

- ✅ Costo de inferencia bajo, control total del tono de marca, datos en casa.
- ❌ **Descartada como mecanismo principal.** El fine-tuning enseña *forma*, no
  *hechos actualizables*. El conocimiento queda congelado en los pesos: cada
  cambio de precio, política o estado de pedido exigiría reentrenar. Además:
  - Requiere un dataset etiquetado grande que EcoMarket todavía no tiene curado.
  - Meter datos de clientes en los pesos del modelo crea un riesgo de
    **memorización y fuga de PII** difícil de revertir (no existe el "olvido"
    selectivo), lo que choca con el derecho de supresión del habeas data.
  - Tiempo de actualización: días. Viola RNF-5 por dos órdenes de magnitud.

### Opción C — Sistema RAG sobre un LLM de propósito general

Un LLM mediano cuyo prompt se **aumenta** en tiempo real con los fragmentos
relevantes de las fuentes autorizadas de EcoMarket.

- ✅ Resuelve RNF-1, RNF-5 y RNF-6 de raíz: la respuesta se ancla en documentos
  vigentes y citables. Actualizar el conocimiento es actualizar un documento.
- ✅ Sin reentrenamiento: cambiar la política de devoluciones es editar un
  archivo y reindexarlo.
- ⚠️ Introduce un componente nuevo que hay que operar bien: la calidad del
  sistema pasa a depender de la calidad del *retrieval*.

### Opción D — Arquitectura híbrida por niveles (RAG + enrutamiento + copiloto humano) ← **SELECCIONADA**

La Opción C, más dos decisiones que el caso exige explícitamente:

1. **Separar el 80% del 20%** con un clasificador de intención barato, porque el
   enunciado dice que el 20% "requiere un toque humano y empatía".
2. **Separar el dato estructurado del no estructurado**: los pedidos se
   consultan con *function calling* contra el sistema transaccional; las
   políticas, el catálogo y las FAQ se resuelven con RAG.

---

## 4. La decisión: arquitectura híbrida de dos niveles con RAG

> **Modelo seleccionado:** `gpt-4o-mini` (LLM de propósito general, de gama
> económica) como motor del Nivel 1, orquestado por un sistema RAG con búsqueda
> híbrida y re-ranking, más *function calling* hacia el OMS. `gpt-4o` (modelo de
> gama alta) se reserva para el Nivel 2, como copiloto del agente humano.
>
> **Modelo de la implementación (Fase 3):** `glm-5.3-flash`, un LLM
> open-source de la misma gama, servido en línea por Ollama Cloud. Ocupa
> exactamente el lugar de `gpt-4o-mini` en la arquitectura; ver la sección 4.4.

### 4.1 Por qué no un solo modelo para todo

Usar un modelo premium para las 2.400 consultas diarias repetitivas es pagar
capacidad de razonamiento que la tarea no necesita: resumir un JSON de pedido en
tono amable no requiere el modelo más capaz del mercado, requiere el **dato
correcto**. Y usar un modelo económico para gestionar una queja con implicación
legal es ahorrar en el punto exacto donde el error cuesta más. La segmentación
del tráfico es, en sí misma, la decisión de mayor impacto económico del diseño.

### 4.2 Diagrama de la arquitectura

```
                          ┌───────────────────────────────┐
   Chat / Correo /        │   Capa de canales (API GW)    │
   Redes sociales ───────►│   normaliza el mensaje        │
                          └───────────────┬───────────────┘
                                          ▼
                          ┌───────────────────────────────┐
                          │  GUARDRAILS DE ENTRADA        │
                          │  · Minimización de PII        │
                          │  · Reglas de escalamiento     │  ──► NIVEL 2
                          │  · Anti prompt-injection      │      (humano)
                          └───────────────┬───────────────┘
                                          ▼
                          ┌───────────────────────────────┐
                          │  ROUTER DE INTENCIÓN          │
                          │  gpt-4o-mini, temp=0, 1 token │
                          └──┬────────────┬───────────┬───┘
                             │            │           │
         ESTADO_PEDIDO ◄─────┘   DEVOLUCION│      QUEJA└──────────► NIVEL 2
                │                          │
                ▼                          ▼
   ┌────────────────────────┐   ┌──────────────────────────────────┐
   │  FUNCTION CALLING      │   │  PIPELINE RAG                    │
   │  consultar_pedido()    │   │  ┌────────────────────────────┐  │
   │      │                 │   │  │ 1. Expansión de consulta   │  │
   │      ▼                 │   │  │ 2. Búsqueda HÍBRIDA        │  │
   │  ┌──────────────┐      │   │  │    vectorial (HNSW/ANN)    │  │
   │  │ OMS / ERP    │      │   │  │    + léxica (BM25)         │  │
   │  │ (PostgreSQL) │      │   │  │ 3. Re-ranking (cross-enc.) │  │
   │  └──────────────┘      │   │  │ 4. Top-N al context window │  │
   │  Dato exacto, no       │   │  └──────────────┬─────────────┘  │
   │  vectorizado           │   │                 ▼                │
   └───────────┬────────────┘   │    ┌──────────────────────────┐  │
               │                │    │  VECTOR DB (pgvector)    │  │
               │                │    │  · políticas             │  │
               │                │    │  · catálogo              │  │
               │                │    │  · FAQ / manuales        │  │
               │                │    └──────────────────────────┘  │
               │                └────────────────┬─────────────────┘
               └──────────────┬──────────────────┘
                              ▼
               ┌──────────────────────────────┐
               │  PROMPT AUMENTADO            │
               │  rol + reglas + CONTEXTO +   │
               │  datos verificados + formato │
               └──────────────┬───────────────┘
                              ▼
               ┌──────────────────────────────┐
               │  NIVEL 1 — gpt-4o-mini       │
               │  temperatura 0.2             │
               └──────────────┬───────────────┘
                              ▼
               ┌──────────────────────────────┐
               │  GUARDRAILS DE SALIDA        │
               │  · Verificación de anclaje   │
               │  · Detección de promesas     │  ──► si falla ──► NIVEL 2
               │  · Citación de fuentes       │
               └──────────────┬───────────────┘
                              ▼
                         Cliente

   NIVEL 2 — AGENTE HUMANO CON COPILOTO (gpt-4o)
   · Resumen del caso y del histórico
   · Borrador de respuesta sugerido
   · El humano edita, aprueba y envía  ← el humano tiene la última palabra
   · Cada aprobación alimenta el dataset de evaluación
```

### 4.3 Decisión de diseño clave: RAG **no** es para todo

Esta es la distinción que más nos importa dejar explícita, y está implementada
así en el código de la Fase 3:

| Tipo de dato | Ejemplo | Mecanismo | Razón |
|---|---|---|---|
| **No estructurado, semi-estable** | Política de devoluciones, fichas de producto, FAQ | **RAG** (vector DB + búsqueda híbrida) | La consulta del cliente no coincide léxicamente con el documento; se necesita similitud semántica |
| **Estructurado, transaccional, exacto** | Estado del pedido ECO-10003 | **Function calling** contra el OMS | La similitud semántica es peligrosa aquí: podría devolver el pedido ECO-10002 cuando se pidió el ECO-10012. Un pedido se busca por clave primaria, no por parecido |
| **Tono y estilo de marca** | Forma de saludar, nivel de formalidad | **Prompt de sistema** (y, si se justifica, fine-tuning ligero en la fase 3 del roadmap) | Es lo único que el fine-tuning enseña bien, y no cambia cada semana |

Meter los pedidos en la base vectorial sería el error de diseño más costoso de
este proyecto: además de ser impreciso, obligaría a vectorizar datos personales
de clientes y a replicarlos fuera del sistema transaccional.

### 4.4 El modelo de la implementación: `glm-5.3-flash` (open-source)

El taller permite usar un modelo open-source para la Fase 3, y lo aprovechamos
para poner a prueba la tesis de la sección 4.1: si el dato correcto llega en el
prompt, **un modelo de gama económica basta, sea propietario o abierto**. El
código ejecutable usa `glm-5.3-flash`, servido por Ollama Cloud, en todos los
puntos donde la arquitectura dice `gpt-4o-mini`: router, *function calling* y
generación del Nivel 1.

Lo elegimos por los mismos criterios de la sección 2, contrastado con
`gpt-oss:120b` (el otro candidato open-source evaluado) sobre los 8 casos de
`outputs/ejecucion_fase3.md`:

| Criterio | `glm-5.3-flash` | `gpt-oss:120b` |
|---|---|---|
| Calidad (RNF-1) | Clasifica bien las 8 consultas, invoca `consultar_pedido` con el argumento correcto y todas sus respuestas pasan la verificación de anclaje | Equivalente: mismas clasificaciones y respuestas |
| Tokens de salida en la corrida | **2.619** | 4.014 (~50% más) |
| Latencia de la corrida completa | **~45 s** | Varios minutos |
| Integración (RNF-2) | Endpoint compatible con la API de OpenAI: mismo SDK y mismo código | Igual |

La diferencia no está en la calidad sino en la eficiencia: `glm-5.3-flash`
razona menos para llegar a la misma respuesta, y en un sistema cuyo costo escala
con cada consulta eso es lo que decide.

Esto tiene tres consecuencias para la decisión:

1. **La portabilidad de la sección 5.4 está demostrada, no solo afirmada.**
   Pasar de `gpt-4o-mini` a `glm-5.3-flash` solo exigió tocar la capa del
   proveedor (`src/config.py` y `src/llm_client.py`): la URL del endpoint, la
   API key y el margen de razonamiento del punto 3. La cadena de prompts, el
   retriever y los guardrails no se tocaron. Desde entonces, cambiar de modelo
   o de proveedor es cambiar dos variables de entorno (`ECOMARKET_PROVEEDOR`,
   `ECOMARKET_MODELO`).
2. **Un modelo abierto es una alternativa real para producción.** Reduce el
   *vendor lock-in* y, si se auto-hospeda, permite que los datos de los clientes
   no salgan de la infraestructura de EcoMarket, lo cual pesa en el análisis de
   privacidad de la Fase 2. El precio es asumir la operación del modelo, que es
   justo la desventaja de escalabilidad que la sección 5.3 señala para las
   opciones auto-hospedadas.
3. **Los modelos de razonamiento imponen un ajuste de ingeniería.**
   `glm-5.3-flash` "piensa" antes de responder, y esos tokens cuentan contra el
   límite de salida. El router, que responde una sola palabra con un límite de
   8 tokens, devolvía una cadena vacía hasta que se agregó un margen de tokens
   para el razonamiento (`src/config.py`). Es la clase de detalle que solo
   aparece al ejecutar, y la razón por la que la hoja de ruta de la sección 7
   empieza con un copiloto interno.

Mantenemos `gpt-4o-mini` como recomendación para la primera versión en
producción porque su costo por token es público y permite el cálculo de la
sección 5.2. Ollama Cloud cobra por suscripción, no por token, así que el costo
de `glm-5.3-flash` a escala debe cotizarse, o estimarse como costo de GPU si se
auto-hospeda, antes de compararlo en igualdad de condiciones.

---

## 5. Justificación por criterio

### 5.1 Calidad de la respuesta esperada (RNF-1, RNF-6)

El material de la clase lo resume en el *trade-off* de retrieval: con RAG la
precisión es alta porque el modelo accede a información externa autorizada, y la
transparencia es alta porque se pueden citar las fuentes; sin RAG la precisión
es baja y el modelo "puede alucinar".

Tres decisiones concretas elevan la precisión por encima de un RAG ingenuo:

1. **Búsqueda híbrida** (vectorial + BM25). La búsqueda vectorial sola falla con
   los términos raros que abundan en este dominio: `ECO-MIE-500`,
   `NO_DEV_PERECEDERO`, "caléndula". La léxica sola falla cuando el cliente
   escribe "ya no la quiero" y el documento dice "derecho de retracto". Azure
   recomienda exactamente esta combinación "para máximo recall".
2. **Re-ranking**. Recuperamos 8 candidatos y enviamos los 4 mejores. El
   *context window* es un recurso escaso y caro: llenarlo de fragmentos
   irrelevantes degrada la respuesta además de encarecerla.
3. **Expansión de consulta**. Si el cliente escribe "quiero devolver la miel", la
   primera búsqueda trae la ficha del producto (que dice `NO_DEV_PERECEDERO`)
   pero no la sección de la política que explica esa exclusión, porque el
   cliente nunca escribió "alimentos perecederos". Una segunda búsqueda con los
   códigos de política hallados cierra ese hueco de *recall*. Está implementado
   en `src/retriever.py::buscar_con_expansion`.

### 5.2 Costo (RNF-3)

**Supuestos declarados** (deben validarse con el área financiera de EcoMarket):
3.000 consultas/día ≈ 90.000/mes; 80% automatizable = 72.000 consultas/mes;
~2.500 tokens de entrada y ~250 de salida por consulta resuelta (incluye el
router, el prompt de sistema y el contexto recuperado); TRM de referencia
COP 4.000/USD. Precios de lista de referencia a la fecha del taller —
**verificar los vigentes antes de decidir**.

| Concepto | gpt-4o-mini (elegido) | gpt-4o (alternativa premium) |
|---|---|---|
| Entrada: 180M tokens/mes | 180 × USD 0,15 = **USD 27,00** | 180 × USD 2,50 = USD 450,00 |
| Salida: 18M tokens/mes | 18 × USD 0,60 = **USD 10,80** | 18 × USD 10,00 = USD 180,00 |
| Embeddings de consulta (1,4M tokens) | USD 0,03 | USD 0,03 |
| **Total mensual del Nivel 1** | **≈ USD 38** | ≈ USD 630 |
| **Costo por consulta resuelta** | **≈ USD 0,00053 (COP ~2)** | ≈ USD 0,0088 (COP ~35) |

El modelo elegido cumple RNF-3 con casi cuatro veces de margen y cuesta **~17×
menos** que la alternativa premium para una tarea en la que la diferencia de
capacidad no se percibe, porque el dato correcto ya viene en el prompt.

A esto hay que sumarle el costo de la infraestructura de RAG (base vectorial,
reindexación, observabilidad), del orden de USD 100–300/mes en una instancia
gestionada pequeña, y el costo de operación del Nivel 2, que **no desaparece**.

**Comparación con el estado actual (orden de magnitud, no cifra exacta):** con el
supuesto de que un agente resuelve ~40 consultas diarias, atender 2.400 consultas
repetitivas al día requiere del orden de 60 agentes dedicados solo a eso. El
ahorro relevante no es despedir a esos agentes, sino **liberarlos hacia el 20%
complejo**, que hoy espera 24 horas. Esa es la tesis de negocio del proyecto, y
se conecta directamente con la discusión de impacto laboral de la Fase 2.

### 5.3 Escalabilidad (RNF-4)

- **El modelo escala por API**: no hay GPUs que aprovisionar ni modelos que
  mantener calientes. Un pico de Black Friday es un aumento de gasto, no un
  incidente de capacidad. Esta es la ventaja decisiva frente a la Opción B, donde
  un modelo afinado y auto-hospedado exigiría dimensionar GPUs para el pico.
- **La base vectorial escala por índice**: con un índice ANN tipo HNSW la
  búsqueda se mantiene en milisegundos aunque el catálogo crezca de 12 a 12.000
  SKU, tal como señala el material de la clase sobre bases vectoriales.
- **El costo escala linealmente y es predecible** por consulta, lo que permite
  ponerle un presupuesto y una alarma.
- **Límite conocido:** los *rate limits* del proveedor. Se mitiga con caché de
  respuestas para las consultas más frecuentes, colas con reintentos y un
  proveedor secundario configurado.

### 5.4 Facilidad de integración (RNF-2, RNF-5)

- **Integración con el OMS sin tocarlo.** `consultar_pedido()` es una llamada de
  solo lectura a la API que EcoMarket ya expone para su app. No se replica la
  base de pedidos, no se migra nada, no se vectoriza PII.
- **Integración con el conocimiento sin proyecto de datos.** Las políticas y el
  catálogo ya existen como documentos. Indexarlos es un *pipeline* de ingesta,
  no una migración. Actualizar la política es editar el documento y reindexar:
  minutos, no días. Esto es lo que cumple RNF-5.
- **Portabilidad del proveedor.** Toda la dependencia del proveedor está aislada
  en `src/llm_client.py` y `src/config.py`. Cambiar a otro proveedor que exponga
  la API de OpenAI (Ollama, vLLM, la mayoría de los servicios de modelos
  abiertos) es cambiar la configuración; uno que no la exponga exige
  reimplementar una clase de ~100 líneas. En ambos casos, la cadena de prompts,
  el retriever y los guardrails no se tocan. Así se hizo en la Fase 3, que corre
  sobre el modelo open-source `glm-5.3-flash` (sección 4.4). Esto evita el
  *vendor lock-in*, que es la objeción legítima más fuerte contra elegir un
  modelo propietario.
- **Latencia.** El router gasta ~1 token de salida; la búsqueda híbrida corre en
  milisegundos; la generación es una sola llamada. El p95 objetivo de 5 segundos
  es holgado para un modelo de gama económica con respuestas de ~200 palabras.

### 5.5 Riesgo y reversibilidad

Ninguna de las decisiones es irreversible: el modelo se cambia por
configuración, el conocimiento vive fuera del modelo y el enrutamiento es un
umbral ajustable. Un sistema donde el conocimiento está *en los pesos* (Opción B)
no tiene esa propiedad: para corregir un dato mal aprendido hay que reentrenar.

---

## 6. Resumen de la selección por componente

| Componente | Modelo / tecnología | Por qué |
|---|---|---|
| Router de intención | `gpt-4o-mini`, temp. 0, salida de 1 token | Clasificar 6 categorías no requiere un modelo grande; el costo marginal es despreciable |
| Generación Nivel 1 | `gpt-4o-mini`, temp. 0,2 | Calidad suficiente cuando el contexto es correcto; 17× más barato que la gama alta |
| Router + Nivel 1 en la implementación (Fase 3) | `glm-5.3-flash` (open-source, Ollama Cloud), temp. 0,2 | Misma calidad que `gpt-oss:120b` con ~35% menos tokens de salida; demuestra la portabilidad de la arquitectura |
| Embeddings | `text-embedding-3-small` | Costo casi nulo, dimensionalidad manejable, calidad suficiente para un corpus de este tamaño |
| Búsqueda | Híbrida: vectorial (HNSW) + BM25, con re-ranking | Recupera tanto sinónimos como identificadores exactos |
| Base vectorial | pgvector sobre el PostgreSQL existente | Evita introducir un sistema nuevo; migrable a Pinecone/Azure AI Search si el corpus crece |
| Datos de pedidos | *Function calling* al OMS (solo lectura) | Exactitud por clave primaria; no se vectoriza PII |
| Copiloto Nivel 2 | `gpt-4o` | Aquí sí se justifica la gama alta: menos volumen, más consecuencia |
| Tono de marca | Prompt de sistema (fine-tuning ligero, opcional, fase 3) | El fine-tuning enseña estilo, no hechos |

---

## 7. Hoja de ruta de implementación

| Fase | Alcance | Duración estimada | Criterio de avance |
|---|---|---|---|
| 0. Línea base | Medir volumen real, mezcla de intenciones, tiempo de resolución y CSAT actuales | 2 semanas | Sin línea base no hay forma de demostrar mejora |
| 1. Copiloto interno (*human-in-the-loop*) | El asistente **solo sugiere**; el agente humano aprueba y envía. Cero exposición al cliente | 6 semanas | ≥ 70% de borradores aprobados sin edición mayor |
| 2. Piloto público acotado | Se libera al cliente solo `ESTADO_PEDIDO` y `ENVIO`, en un canal, con botón visible de "hablar con una persona" | 4 semanas | Tasa de alucinación < 1%, CSAT ≥ línea base |
| 3. Ampliación | Se agrega `DEVOLUCION` y `PRODUCTO`; se evalúa fine-tuning ligero de tono con los casos aprobados en la fase 1 | 6 semanas | Tasa de contención ≥ 60% sin caída de CSAT |
| 4. Operación continua | Monitoreo, reindexación automática, revisión trimestral de sesgos | Permanente | Tablero de métricas y auditoría trimestral |

Empezar por el copiloto interno no es cautela excesiva: es la forma de generar el
dataset de evaluación con supervisión humana **antes** de exponer el sistema a
clientes, y de convertir a los agentes en dueños de la herramienta en lugar de
víctimas de ella.

---

## 8. Métricas de éxito

| Métrica | Línea base | Meta a 6 meses |
|---|---|---|
| Tiempo de primera respuesta (80% repetitivo) | 24 h | < 30 s |
| Tiempo de primera respuesta (20% complejo) | 24 h | < 4 h |
| Tasa de contención (resueltas sin humano) | 0% | 55–65% |
| Tasa de alucinación (auditoría de muestra semanal) | — | < 0,5% |
| CSAT del canal automatizado | — | ≥ CSAT humano actual |
| Costo por consulta resuelta | — | < USD 0,002 |
| Tasa de escalamiento correcto (casos sensibles) | — | 100% |

La métrica que **no** usaremos como objetivo es "porcentaje de consultas sin
intervención humana" en términos absolutos: optimizar eso incentiva a no escalar
casos que sí deberían escalarse.

---

## 9. Fuentes

1. Ávila, D. (2026). *Clase 2 — RAG: Retrieval Augmented Generation*. Maestría en
   Inteligencia Artificial, ICESI. (Arquitectura RAG, mecanismo de retrieval,
   bases vectoriales y sus consideraciones, búsqueda híbrida, re-ranking con
   *cross-encoders*, tabla de *trade-off* con y sin RAG.)
2. AWS. *¿Qué es la generación aumentada por recuperación (RAG)?*
   https://aws.amazon.com/es/what-is/retrieval-augmented-generation/
   (Definición de RAG como referencia a una base de conocimientos autorizada
   fuera de los datos de entrenamiento; los cuatro pasos: datos externos,
   recuperación relevante, aumento del prompt y actualización.)
3. Microsoft Learn. *RAG and generative AI — Azure AI Search*.
   https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview
   (Desafíos del RAG empresarial: comprensión de la consulta, acceso multi-fuente,
   restricciones de tokens, tiempo de respuesta y seguridad; recomendación de
   búsqueda híbrida con *semantic ranking* y de *security trimming* a nivel de
   documento.)
4. Aryani, A. *A Brief Introduction to Retrieval Augmented Generation (RAG)*.
   Medium. https://medium.com/@amiraryani/a-brief-introduction-to-retrieval-augmented-generation-rag-4bd6e50da532
   (Referencia conceptual sobre el esquema *retriever* + *generator*.)
5. Congreso de Colombia. *Ley 1480 de 2011* (Estatuto del Consumidor), art. 47 —
   derecho de retracto; *Ley 1581 de 2012* — protección de datos personales.
