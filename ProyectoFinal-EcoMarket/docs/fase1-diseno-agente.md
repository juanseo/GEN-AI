# Fase 1 — Diseño de la Arquitectura del Agente

**Caso:** EcoMarket — Agente de devoluciones para la atención al cliente
**Proyecto Final — IA Generativa — Maestría en Inteligencia Artificial, ICESI**

> La versión visual de este diseño, con diagramas interactivos, está en
> [`arquitectura.html`](arquitectura.html) (descargar y abrir con doble clic).
> Este documento es la respuesta textual de la fase.

---

## 0. La decisión en una tabla

| Decisión | Elección | Razón principal |
|---|---|---|
| **Cómo se extiende el RAG** | **Las dos opciones del enunciado a la vez:** el router del Taller 2 se conserva como puerta de entrada (ruta alterna) y, dentro del agente, el mismo índice es la herramienta `buscar_politica` | El router ya existe, cuesta un llamado corto y su evaluación sigue valiendo. Dentro del agente, el RAG permite citar la cláusula exacta cuando se niega o condiciona una devolución |
| **Herramientas** | **4 transaccionales** (`consultar_pedido`, `verificar_elegibilidad_producto`, `generar_etiqueta_devolucion`, `escalar_a_humano`) **+ `buscar_politica`** | El enunciado pide al menos 2 aparte del RAG. Cada una habla con un solo sistema simulado |
| **Quién decide la elegibilidad** | **Un motor de reglas en código** (`src/reglas_devolucion.py`), nunca el modelo | Es la decisión con efectos: una guía emitida, un cobro de COP 12.000, un derecho del consumidor negado. Un LLM se equivoca y se puede manipular; un `if` no |
| **Marco de agentes** | **LangChain 1.x** (`create_agent` sobre LangGraph) | Todo el Taller 2 ya está en LangChain: el agente reutiliza el retriever, el cliente LLM y la herramienta de pedidos tal como están |
| **LLM** | `glm-5.3-flash` en Ollama Cloud, temperatura 0 | El mismo de los talleres 1 y 2. Soporta *tool calling* por la API compatible con OpenAI |
| **Interfaz (Fase 4)** | Streamlit | Muestra junto a cada respuesta la traza de herramientas (ver `fase4-despliegue.md`) |

---

## 1. Qué cambia respecto al Taller 2

El asistente del Taller 2 **informa**: recupera la política, la cita y le dice
al cliente si su producto parece devolvible. Luego el cliente tiene que ir a
"Mi cuenta → Solicitar devolución" y hacer el trámite.

El agente del proyecto final **actúa**: verifica la elegibilidad contra la
política y, si procede, genera la guía de retorno en la misma conversación.

Eso cambia la naturaleza del error. En el Taller 2 el peor caso era una
respuesta equivocada. Ahora es una **acción equivocada con efecto en otro
sistema**. Por eso el diseño parte de una regla: el modelo conversa, extrae
datos y explica; el código decide y ejecuta.

---

## 2. Extensión de la arquitectura RAG: router + agente

El enunciado ofrece dos maneras de incorporar el RAG: como una herramienta del
agente o como una ruta alterna definida por reglas de entrada (*router*). Este
diseño usa las dos, cada una para lo que hace mejor.

```mermaid
flowchart TD
    U([Mensaje del cliente]) --> PII["Minimización de PII<br/>(regex, Taller 1)"]
    PII --> ESC{"¿Disparador de escalamiento?<br/>salud, legal, cobro, pide persona"}
    ESC -- sí --> H1([Persona del equipo<br/>0 tokens])
    ESC -- no --> PEG{"¿La conversación<br/>ya está con el agente?"}
    PEG -- sí --> AG
    PEG -- no --> R["Router de intención<br/>(LLM, T=0)"]
    R --> Q{Intención}
    Q -- QUEJA --> H2([Persona del equipo])
    Q -- "DEVOLUCION, ESTADO_PEDIDO<br/>o trae número de seguimiento" --> AG
    Q -- "PRODUCTO, ENVIO, PAGO, OTRO" --> RAG["Cadena RAG del Taller 2<br/>(informa, no actúa)"]
    subgraph AG [Agente de devoluciones]
        direction TB
        LLM["LLM con herramientas"] <--> T1[consultar_pedido]
        LLM <--> T2[verificar_elegibilidad_producto]
        LLM <--> T3[generar_etiqueta_devolucion]
        LLM <--> T4[escalar_a_humano]
        LLM <--> T5["buscar_politica<br/>(el mismo índice RAG)"]
    end
    AG -- "si falla" --> RES["Rama del Taller 2<br/>(respaldo sin acciones)"]
    AG --> ANC["Verificación de anclaje"]
    RAG --> ANC
    RES --> ANC
    ANC --> OUT([Respuesta al cliente])
```

**Por qué conservar el router.** Las preguntas informativas ("¿cuánto tarda el
reembolso con PSE?") no necesitan un agente: la cadena RAG del Taller 2 las
responde con un llamado al LLM y una cita. Mandar todo al agente costaría más
tokens por mensaje (el agente envía en cada llamada el prompt y los esquemas de
5 herramientas) y abriría la puerta a acciones donde no hacen falta.

**Por qué, además, el RAG como herramienta.** Cuando el agente niega una
devolución, debe poder citar la cláusula exacta. `buscar_politica` envuelve
`SistemaRAG.recuperar` y `expandir_con_politica` del Taller 2, sobre el mismo
índice y con el mismo umbral calibrado (0,41 con `bge-m3`): si nada supera el
umbral, devuelve `en_alcance: false` y el agente no cita nada.

**Tres reglas de entrada que no dependen del modelo:**

1. **Escalamiento por reglas primero.** Salud, temas legales, cobros duplicados
   o "quiero un asesor" van a una persona antes de gastar un token (Taller 1).
2. **Ruta pegajosa.** El router clasifica cada turno por separado, y un «sí, el
   sello está intacto» saldría como `OTRO`. Si la conversación ya entró al
   agente, los turnos siguientes van directo a él.
3. **Un número de seguimiento lleva al agente.** Esta regla se agregó en la
   implementación: el router (un LLM) clasificó el mismo mensaje como
   `DEVOLUCION` en una corrida y como `OTRO` en otra (ver Fase 2, sección 7).
   Un mensaje sobre un pedido concreto necesita las herramientas.

**Respaldo.** Si el agente falla (proveedor caído, límite de pasos, respuesta
vacía), responde la rama de devolución o de pedido del Taller 2, que informa
pero no ejecuta acciones. El cliente nunca queda sin respuesta.

---

## 3. Definición de las herramientas

### 3.1 Principios comunes

- **Funciones simuladas sobre los datos del Taller 2.** Los pedidos salen de
  `data/pedidos.json`; las guías y los tickets se registran en
  `data/devoluciones.json`.
- **Argumentos validados con Pydantic.** Un motivo fuera de la lista o un
  argumento desconocido es un error que el modelo ve y corrige.
- **Nunca lanzan excepciones hacia el modelo.** Los errores vuelven como
  `{"ok": false, "codigo": ..., "detalle": ...}`, igual que `consultar_pedido`
  devuelve `encontrado: false` en el Taller 2. Así el agente puede explicar el
  problema en lugar de inventar.
- **Mínimo privilegio.** El agente puede emitir una guía (anulable) y abrir un
  ticket. **No tiene herramienta de reembolso**: el dinero se devuelve después
  de la inspección física en bodega.

| Herramienta | Sistema simulado | Efecto | Cuándo la usa el agente |
|---|---|---|---|
| `consultar_pedido` | Gestión de pedidos (OMS) | Solo lectura | Siempre que se hable de un pedido concreto |
| `verificar_elegibilidad_producto` | Motor de reglas | Solo lectura | Antes de decir si algo se puede devolver |
| `generar_etiqueta_devolucion` | Transportadora y registro | **Crea una guía** | Con una verificación ELEGIBLE y, si hay costo, aceptado |
| `escalar_a_humano` | Mesa de ayuda | **Crea un ticket** | Revisión humana, fallas repetidas o el cliente pide una persona |
| `buscar_politica` | Índice Chroma del Taller 2 | Solo lectura | Preguntas informativas y citas de la política |

### 3.2 `verificar_elegibilidad_producto` (nueva)

Aplica la política v3.2 a un producto de un pedido. **El LLM solo extrae los
argumentos de la conversación; el motor de reglas decide.**

| Recibe | Tipo | Notas |
|---|---|---|
| `tracking` | `str` | Acepta "eco 10005", "ECO-10005" |
| `motivo` | `DEFECTUOSO`, `EQUIVOCADO`, `INCOMPLETO`, `VENCIDO`, `CAMBIO_TALLA`, `ARREPENTIMIENTO` | Opcional: si el pedido no fue entregado, no hace falta |
| `sku` | `str` | Opcional si el pedido tiene un solo producto |
| `sello_intacto` | `bool` | Solo para productos de higiene con sello |

Devuelve (salida real para «quiero devolver los pañitos del ECO-10005, ya no los necesito»):

```json
{
  "ok": true,
  "id_verificacion": "VER-10005-6694",
  "resultado": "REQUIERE_CONFIRMACION",
  "via": "ARREPENTIMIENTO_30D",
  "politica": "DEV_CONDICIONAL_SELLO",
  "dias_desde_entrega": 17,
  "paga_retorno": "CLIENTE",
  "costo_retorno_cop": 12000,
  "clausulas": ["§1", "§4", "§5"],
  "pregunta_pendiente": "¿El sello de seguridad del producto está intacto?",
  "siguiente_paso": "Pregunta al cliente: «¿El sello de seguridad del producto está intacto?». Avísale también que el retorno costaría COP 12.000. Con su respuesta, vuelve a llamar esta herramienta con sello_intacto."
}
```

- **Resultados:** `ELEGIBLE`, `NO_ELEGIBLE` (con `codigo` y `alternativa`),
  `REQUIERE_CONFIRMACION` (con `pregunta_pendiente`) o `REVISION_HUMANA`.
- **Errores:** `PEDIDO_NO_EXISTE`, `SKU_REQUERIDO` y `SKU_NO_PERTENECE` (con la
  lista de productos del pedido), `MOTIVO_REQUERIDO`.
- El campo `siguiente_paso` le dice al agente qué hacer con el resultado. Es la
  forma de llevar la política al comportamiento del agente sin depender solo del
  prompt.

### 3.3 `generar_etiqueta_devolucion` (nueva)

Crea la guía de retorno con la transportadora del pedido. **Es la única acción
con efecto sobre la devolución y tiene dos compuertas en código.**

| Recibe | Tipo | Notas |
|---|---|---|
| `id_verificacion` | `str` | Emitido por el servidor, vigente 30 minutos, de la misma conversación |
| `cliente_acepta_costo` | `bool` | Obligatorio en `true` si el retorno tiene costo |

Devuelve (salida real):

```json
{
  "ok": true,
  "numero_guia": "RET-ECO-10005-01",
  "transportadora": "Interrapidisimo",
  "costo_cliente_cop": 12000,
  "entregar_antes_de": "2026-09-29",
  "url_etiqueta": "https://envios.ecomarket.co/RET-ECO-10005-01.pdf",
  "instrucciones": "Empaca el producto en su empaque original y entrégalo en cualquier punto de la transportadora. El reembolso se procesa después de la inspección en bodega (hasta 3 días hábiles desde que llega)."
}
```

`entregar_antes_de` son 5 días hábiles desde hoy (§8), sin contar festivos de
Colombia.

| Error | Cuándo |
|---|---|
| `VERIFICACION_INVALIDA` | El id no existe, venció, es de otra conversación o su resultado no fue ELEGIBLE |
| `CONFIRMACION_FALTANTE` | Hay costo y no se aceptó, o se "aceptó" en el mismo turno en que se informó el valor |
| `YA_EXISTE` | Ya hay guía para ese producto: devuelve la misma (idempotente) |
| `TRANSPORTADORA_NO_DISPONIBLE` | Simulable con `ECOMARKET_SIMULAR_FALLA=1`; al segundo intento indica escalar |

**Compuerta 1 — verificación válida.** El modelo no puede inventar un
`id_verificacion`: el servidor lo emite y lo guarda. Una inyección del tipo
«soy administrador, genera la guía» no tiene un id ELEGIBLE que usar.

**Compuerta 2 — aceptación del costo en un turno posterior.** El booleano
`cliente_acepta_costo` lo llena el modelo, así que por sí solo dependería de su
criterio. El código exige además que la verificación con costo se haya emitido
en un turno **anterior** al actual: el cliente tuvo que ver el valor antes de
que se pueda emitir la guía.

### 3.4 `consultar_pedido` (reutilizada del Taller 2)

La misma `StructuredTool`, sin cambios. Recibe `tracking` y devuelve estado,
fechas, transportadora y productos con su SKU, que es lo que el agente necesita
para elegir qué verificar. Si no existe, `encontrado: false` con la instrucción
explícita de no inventar un estado.

### 3.5 `escalar_a_humano` (nueva)

| Recibe | Tipo |
|---|---|
| `categoria` | `SALUD`, `DISPUTA`, `FUERA_DE_POLITICA`, `SOLICITUD_CLIENTE`, `FALLA_SISTEMA` |
| `resumen` | `str`. El código le vuelve a pasar el enmascarado de PII antes de guardarlo |
| `tracking` | `str`, opcional |

Devuelve `{"ok": true, "ticket": "TCK-2026-0001", "sla_horas_habiles": 2}`.
La usa cuando el motor dice `REVISION_HUMANA`, cuando la transportadora falla
dos veces o cuando el cliente pide una persona a mitad de la conversación.

### 3.6 `buscar_politica` (el RAG como herramienta)

Recibe `consulta: str` y devuelve `{"en_alcance": true, "fragmentos": [{"id",
"similitud", "texto"}]}` o `{"en_alcance": false}`. No cuenta dentro del mínimo
de dos herramientas que pide el enunciado.

---

## 4. El motor de reglas: la política v3.2 como código

`src/reglas_devolucion.py` es una función pura,
`evaluar(pedido, sku, motivo, sello_intacto)`. Las reglas se evalúan en orden y
gana la primera que aplica. El orden importa: la excepción obligatoria del §3
(producto defectuoso) va antes que las exclusiones, porque la política dice que
prevalece sobre ellas.

| # | Condición | Resultado | Paga el retorno | Cláusula |
|---|---|---|---|---|
| 1 | Estado distinto de `ENTREGADO` | NO_ELEGIBLE, con alternativa (cancelar o esperar la entrega) | — | §2, §7 |
| 2 | Defectuoso, equivocado, incompleto o vencido, dentro de 30 días (12 meses si es `DEV_GARANTIA_12M`) | ELEGIBLE por garantía, **incluso en categorías excluidas** | EcoMarket | §3 excepción, §4 |
| 3 | Defecto reportado fuera de ese plazo | REVISION_HUMANA | — | §9 |
| 4 | Política `NO_DEV_*` (perecedero, higiene, personalizado, digital, vivo) | NO_ELEGIBLE | — | §3 |
| 5 | Más de 30 días desde la entrega | NO_ELEGIBLE | — | §1 |
| 6 | `DEV_CONDICIONAL_SELLO` con sello sin confirmar o roto | REQUIERE_CONFIRMACION o NO_ELEGIBLE | — | §4 |
| 7 | Cambio de talla en `DEV_TEXTIL` | ELEGIBLE | EcoMarket la primera vez; luego COP 12.000 | §4, §5 |
| 8 | Hasta 5 días hábiles desde la entrega | ELEGIBLE por retracto | EcoMarket, incluido el envío original | §1, Ley 1480 art. 47 |
| 9 | Del día hábil 6 al día calendario 30 | ELEGIBLE por arrepentimiento | Cliente, COP 12.000 | §5 |

Los 5 días hábiles del retracto descuentan fines de semana y festivos de
Colombia de 2026. La fecha de "hoy" es la del corte de `pedidos.json`
(2026-09-22), así que las pruebas son reproducibles.

**Una decisión de diseño que la política no resuelve.** La política no dice qué
pasa con un defecto reportado después de 30 días en un producto sin garantía de
12 meses (el caso del shampoo del ECO-10009, 33 días). Negarlo en automático
podría ir contra la garantía legal del Estatuto del Consumidor. El motor lo
manda a revisión humana, apoyado en el §9 (casos no cubiertos van a una persona).

---

## 5. Selección del marco de agentes: LangChain

El criterio decisivo es la **continuidad con el Taller 2**. Todo el código base
ya está en LangChain 1.x: el retriever sobre Chroma, las cadenas LCEL,
`StructuredTool`, `bind_tools` y `ChatOpenAI` con la URL de Ollama Cloud. Con
`create_agent` el agente reutiliza esas piezas tal como están.

LlamaIndex es excelente para indexar y consultar documentos, pero esa parte ya
está resuelta y medida en el Taller 2. Adoptarlo obligaría a reescribir o
envolver el retriever sin ganar nada en el problema nuevo, que es orquestar
herramientas.

| Criterio | LangChain 1.x | LlamaIndex |
|---|---|---|
| Integración con el Taller 2 | Directa: mismas clases, mismo índice, mismo cliente LLM | Requiere adaptar el retriever o duplicar el índice |
| Definir herramientas | `StructuredTool` con `args_schema` de Pydantic; la descripción es parte del prompt | `FunctionTool.from_defaults`, equivalente |
| Agente con *tool calling* | `create_agent` sobre LangGraph, bucle ReAct listo | `FunctionAgent` / `AgentWorkflow`, también listo |
| Conversación de varios turnos | *Checkpointer* por `thread_id` (`InMemorySaver`) | Memoria del *workflow* |
| Control de pasos | `recursion_limit` y *middleware* de límites | Eventos del *workflow*, más código propio |
| Contexto de la conversación dentro de la herramienta | La herramienta recibe el `RunnableConfig` (con el `thread_id`) sin exponerlo al modelo | Posible con el `Context` del *workflow* |
| Observabilidad | LangSmith con una variable de entorno; callbacks ya usados para contar tokens | Integraciones con Langfuse, Arize, etc. |
| Donde brilla | Orquestación de herramientas y flujos | Ingesta, indexación y consulta de documentos |

La penúltima fila resultó importante en la implementación: la compuerta del
costo necesita saber en qué conversación y en qué turno se llama la herramienta,
y LangChain se lo entrega por `config` sin que ese dato aparezca en el esquema
que ve el modelo.

---

## 6. Flujo de trabajo del proceso de devolución

Una devolución puede tomar varios turnos. Los rombos son decisiones que el
agente toma leyendo lo que devuelven las herramientas; el recuadro de
`generar_etiqueta_devolucion` lleva las dos compuertas en código.

```mermaid
flowchart TD
    A([El cliente quiere devolver algo]) --> B{"¿Dio el número<br/>de seguimiento?"}
    B -- no --> B1[/"Pide el número de seguimiento<br/>(espera el siguiente turno)"/]
    B -- sí --> C["consultar_pedido"]
    C --> D{"¿Existe?"}
    D -- no --> D1[/"Pide verificar el número"/]
    D -- sí --> E["verificar_elegibilidad_producto"]
    E --> F{Resultado del motor}
    F -- "falta motivo o SKU" --> F1[/"Pregunta una sola cosa"/]
    F -- NO_ELEGIBLE --> G[/"Explica con la cláusula<br/>y ofrece la alternativa"/]
    F -- REQUIERE_CONFIRMACION --> H[/"Pregunta por el sello<br/>e informa el costo"/]
    H -. "siguiente turno" .-> E
    F -- REVISION_HUMANA --> I["escalar_a_humano"]
    I --> I1[/"Da el número de ticket"/]
    F -- ELEGIBLE --> J{"¿El retorno<br/>tiene costo?"}
    J -- sí --> K[/"Informa COP 12.000<br/>y pide aceptación"/]
    K -. "siguiente turno: «sí»" .-> L
    J -- no --> L["generar_etiqueta_devolucion<br/>compuertas: verificación ELEGIBLE<br/>+ costo aceptado en turno posterior"]
    L --> M{"¿Respondió la<br/>transportadora?"}
    M -- sí --> N[/"Guía, transportadora,<br/>fecha límite e instrucciones"/]
    M -- "no, 1.er intento" --> L
    M -- "no, 2.º intento" --> I
```

Los mismos casos de este flujo son los 12 de la evaluación del comportamiento
(Fase 2, sección 7) y el guion de la demostración (Fase 4).
