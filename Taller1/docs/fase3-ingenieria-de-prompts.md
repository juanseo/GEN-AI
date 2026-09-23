# Fase 3 — Aplicación de la Ingeniería de Prompts

**Caso:** EcoMarket — Optimización de la atención al cliente
**Código:** `src/prompts.py` (prompts), `src/chain.py` (cadena), `scripts/demo_fase3.py` (ejecución)
**Salida verificable:** `outputs/ejecucion_fase3.md`

---

## 1. Qué se construyó

El taller pide "ver, en código, cómo se estructura una cadena de prompts". Lo que
hay en este repositorio no es un prompt suelto, sino la cadena completa:

```
mensaje del cliente
   │
   ├─► [1] Minimización de PII              (guardrails.py, determinista)
   ├─► [2] Reglas de escalamiento           (guardrails.py, determinista)
   ├─► [3] PROMPT DE ROUTER                 (prompts.ROUTER, temp=0, 1 token)
   │
   ├─► [4] Rama según la intención:
   │        ESTADO_PEDIDO → function calling → PROMPT E1_OPTIMIZADO
   │        DEVOLUCION    → RAG híbrido      → PROMPT E2_OPTIMIZADO
   │        resto         → RAG general      → PROMPT RAG_GENERAL
   │        QUEJA         → escalamiento, sin generación
   │
   └─► [5] Verificación de anclaje          (guardrails.py, determinista)
```

Cada prompt del catálogo existe en **dos versiones**: la básica (la que uno
escribe primero) y la optimizada. `scripts/demo_fase3.py` ejecuta las dos sobre
la misma consulta para que la diferencia sea observable y no solo argumentada.

---

## 2. Anatomía de los prompts optimizados

Los seis elementos que aplicamos en cada prompt:

| Elemento | Qué aporta | Dónde se ve |
|---|---|---|
| **Rol** | Fija registro, idioma y punto de vista. "Eres Eco, el asistente de EcoMarket" produce un texto distinto a "eres un asistente" | `SISTEMA_OPTIMIZADO` |
| **Tarea** | Define el entregable exacto del turno | Encabezado de cada plantilla |
| **Contexto** | El material recuperado por RAG, delimitado con `<contexto>` para que el modelo distinga datos de instrucciones | `E2_OPTIMIZADO`, `RAG_GENERAL` |
| **Reglas y restricciones** | Sobre todo lo que **no** debe hacer. Es la parte que más cambia el comportamiento | Reglas 1–9 de `SISTEMA_OPTIMIZADO` |
| **Formato de salida** | Longitud, estructura y cierre. Sin esto, el modelo produce muros de texto con encabezados | Bloque `# FORMATO DE SALIDA` |
| **Few-shot** | Casos límite resueltos, **incluido el caso negativo** | Ejemplos A, B y C de `E2_OPTIMIZADO` |

Tres decisiones transversales, con su razón:

- **Temperatura 0,2.** En atención al cliente la consistencia vale más que la
  variedad: dos clientes con el mismo problema deben recibir la misma política.
- **Delimitadores explícitos** (`<contexto>`, bloques ```json```). Reducen la
  confusión entre "lo que el modelo debe usar" y "lo que el modelo debe hacer", y
  dificultan la inyección de instrucciones desde el mensaje del cliente.
- **La regla negativa antes que la positiva.** "No inventes; si no está en el
  contexto, dilo" cambia el comportamiento más que cualquier instrucción de tono.

---

## 3. Ejercicio 1 — Prompt de solicitud de pedido

### 3.1 La "base de datos" de prueba

El taller pide un documento con el estado de mínimo 10 pedidos. Está en
[`data/pedidos.json`](../data/pedidos.json): **12 pedidos** que cubren
deliberadamente todos los estados del ciclo de vida, incluidos los casos
incómodos:

| Tracking | Estado | Por qué está en el set |
|---|---|---|
| ECO-10001 | `EN_TRANSITO` | Caso feliz |
| ECO-10002 | `ENTREGADO` | Entrega a tiempo |
| ECO-10003 | `RETRASADO` | Retraso vigente, con motivo; obliga a disculpa y explicación |
| ECO-10004 | `EN_PREPARACION` | Todavía cancelable: no aplica devolución |
| ECO-10005 | `ENTREGADO` con retraso pasado | El retraso ya ocurrió: el tono debe ser distinto al de ECO-10003 |
| ECO-10006 | `PAGO_PENDIENTE` | Sin fecha de entrega: campo `null` que el modelo no debe rellenar |
| ECO-10007 | `DEVOLUCION_EN_CURSO` | Ya hay una gestión activa |
| ECO-10008 | `EN_TRANSITO` | Multi-ítem, incluye un producto no devolvible |
| ECO-10009 | `ENTREGADO` | Producto de higiene: alimenta el Ejercicio 2 |
| ECO-10010 | `EN_PREPARACION` | Producto `NO_DEV_VIVO` |
| ECO-10011 | `CANCELADO` | Con reembolso aprobado |
| ECO-10012 | `RETRASADO` | Retraso por causa **del cliente** (no ubicado): el tono no puede ser el mismo que el de un retraso por culpa de la empresa |

> **Decisión de diseño:** el JSON **no se inyecta completo** en el prompt. Se
> consulta un solo registro por clave primaria mediante `consultar_pedido()`. En
> un taller con 12 pedidos inyectar todo funcionaría; con 50.000 pedidos reales
> es imposible, y además obligaría a exponer los datos de todos los clientes en
> cada consulta. La versión que funciona con 12 y con 50.000 es la misma: la
> herramienta.

### 3.2 Prompt básico

```text
Dame el estado del pedido 12345.
```

**Por qué falla:** el modelo no tiene forma de conocer ese pedido. Sus dos
salidas posibles son (a) inventar un estado plausible —alucinación— o (b) una
evasiva genérica. Ninguna resuelve la consulta del cliente. El problema no es de
redacción: **es de arquitectura**. Ningún prompt arregla la falta de datos.

### 3.3 Prompt optimizado

Ver `src/prompts.py::E1_OPTIMIZADO`. Sus decisiones:

| Decisión | Problema que resuelve |
|---|---|
| Los datos llegan como **JSON verificado** del OMS, no como texto libre | El modelo no tiene que interpretar ni adivinar; solo redactar |
| `"si un campo es null, ese dato no existe y no debes inventarlo"` | Evita que el modelo rellene la fecha de entrega de un pedido en `PAGO_PENDIENTE` |
| Rama explícita para `encontrado: false` | Da una respuesta correcta al caso "el pedido no existe", que es donde más se alucina |
| Rama explícita para `retrasado: true` | El taller lo pide: disculpa + explicación. Añadimos comparar `fecha_estimada_original` contra la nueva, para que el cliente dimensione el cambio |
| Obligación de incluir `url_rastreo` | El taller pide el enlace de rastreo en tiempo real |
| Manejo de `PAGO_PENDIENTE`, `CANCELADO`, `DEVOLUCION_EN_CURSO` | Estados que un prompt ingenuo ignora y que dejan al cliente sin siguiente paso |
| Límite de 150 palabras | Un cliente en un chat no lee un muro de texto |

---

## 4. Ejercicio 2 — Prompt de devolución de producto

### 4.1 El desafío del taller

> *"Diseñar el prompt para que el modelo sea capaz de distinguir entre productos
> que pueden devolverse y los que no (ej: productos perecederos, productos de
> higiene). La respuesta debe ser clara y empática, incluso si la devolución no
> es posible."*

Hay **tres dificultades** encadenadas, y la tercera es la que la mayoría de
soluciones pasa por alto:

1. El modelo debe saber qué categoría tiene el producto → se resuelve con RAG
   sobre el catálogo.
2. El modelo debe saber qué dice la política sobre esa categoría → se resuelve
   con RAG sobre la política.
3. El modelo debe saber que **la excepción por producto defectuoso prevalece
   sobre la exclusión**. Una miel que el cliente ya no quiere no se devuelve;
   una miel que llegó vencida **sí**. Un modelo que solo aprende "alimentos = no
   devolución" le niega al cliente un derecho que la ley le da, y eso es un
   problema legal, no un problema de calidad de respuesta.

### 4.2 Prompt básico

```text
El cliente quiere devolver un producto. Explícale cómo hacerlo.
```

**Por qué falla:** sin la política de EcoMarket, el modelo responde con la
política genérica de "cualquier e-commerce" que aprendió en el entrenamiento:
30 días, empaque original, reembolso completo. Para la miel eso es **falso**, y
crea una expectativa que EcoMarket no va a poder cumplir. Es el caso de manual de
una alucinación con consecuencia comercial y jurídica.

### 4.3 Prompt optimizado

Ver `src/prompts.py::E2_OPTIMIZADO`. La pieza central es un **árbol de decisión
de cinco pasos con orden obligatorio**:

```
1. ¿Llegó defectuoso / vencido / averiado / incompleto / equivocado?
   └─ SÍ → GARANTÍA: reposición o reembolso total, AUNQUE la categoría esté excluida
2. ¿Es categoría NO_DEV_*?
   └─ SÍ → no devolvible: reconocer + explicar el motivo real + ofrecer alternativa
3. ¿Es DEV_CONDICIONAL_SELLO?
   └─ SÍ → PREGUNTAR por el sello antes de prometer nada
4. ¿Está dentro del plazo (30 días / 5 días hábiles)?
   └─ SÍ → devolvible: pasos + quién paga el retorno
5. ¿El pedido no está ENTREGADO?
   └─ SÍ → corresponde CANCELACIÓN, no devolución
```

**Por qué un árbol y no una descripción en prosa.** Un LLM al que se le da una
política en prosa aplica la regla más saliente del contexto y se detiene. El
orden explícito fuerza a evaluar la excepción de garantía **antes** que la
exclusión de categoría, que es exactamente el error que queremos evitar. El orden
de los pasos codifica la jerarquía legal de las reglas.

**Los tres ejemplos few-shot** están elegidos para cubrir el espacio de fallo:

| Ejemplo | Enseña |
|---|---|
| A — miel, sin defecto | Cómo decir **no** con empatía y con el motivo real |
| B — shampoo defectuoso | Que la excepción **prevalece** sobre la exclusión |
| C — jabón con sello | A **preguntar** en vez de asumir cuando falta un dato |

**Las reglas de redacción para la negativa** — reconocer → explicar → ofrecer
alternativa — no son cortesía decorativa. Son la contramedida al riesgo de
asimetría de poder identificado en la [Fase 2](fase2-fortalezas-limitaciones-riesgos.md#35-riesgos-adicionales-que-el-enunciado-no-lista-pero-el-caso-implica):
una negativa automatizada sin explicación y sin ruta de salida deja al cliente
sin nadie a quien apelar.

---

## 5. El prompt de router

```text
Clasifica el mensaje ... Responde solo con la etiqueta, en mayúsculas.
```

Tres decisiones de eficiencia y una de seguridad:

- `temperatura = 0` y `max_tokens = 8`: la clasificación debe ser determinista y
  costar prácticamente nada.
- Categorías **mutuamente excluyentes** con una descripción de una línea cada una.
- Categoría `OTRO` como salida por defecto: cualquier etiqueta no reconocida cae
  ahí, en lugar de romper la cadena.
- La etiqueta `QUEJA` no genera respuesta al cliente: **escala**. La decisión de
  no responder es tan importante como la de responder.

---

## 6. Controles que no son prompts

Una lección de la Fase 2 traducida a código: **hay cosas que no se le piden a un
modelo, se le imponen desde afuera.** Un prompt es una instrucción que el modelo
puede no seguir; un `if` no.

| Control | Por qué es determinista y no una instrucción del prompt |
|---|---|
| **Escalamiento** (`evaluar_escalamiento`) | Un modelo que decide si escalar puede decidir no hacerlo. Un cliente que menciona la SIC o un cobro duplicado se escala por la mención, sin pasar por el modelo. Además ahorra tokens: la consulta nunca llega a la API |
| **Minimización de PII** (`enmascarar_pii`) | Pedirle al modelo "no uses datos sensibles" es inútil: para entonces el dato ya salió del perímetro de EcoMarket |
| **Verificación de anclaje** (`verificar_anclaje`) | Extrae cifras, fechas, códigos y URLs de la respuesta y verifica que existan en el contexto. Es una heurística de monitoreo, no una garantía, y así está declarado |
| **Solo lectura sobre el OMS** | Aunque una inyección de prompt convenciera al modelo de aprobar un reembolso, no existe la función para ejecutarlo |

---

## 7. Cómo verificar los resultados

```bash
# 1. Inspeccionar el retrieval sin gastar tokens (no requiere API key)
python -m scripts.demo_retrieval

# 2. Ejecutar los 8 casos, básico vs. optimizado, y generar la transcripción
python -m scripts.demo_fase3

# 3. Probar consultas propias
python -m scripts.chat --debug
```

`scripts/demo_fase3.py` cubre ocho casos, elegidos para que cada uno pueda
fallar de una forma distinta:

| # | Caso | Qué está probando |
|---|---|---|
| 1 | Pedido en tránsito | Camino feliz del Ejercicio 1 |
| 2 | Pedido retrasado | Disculpa + explicación + comparación de fechas |
| 3 | **Pedido inexistente** | **Prueba crítica anti-alucinación** |
| 4 | Devolución de alimento | El desafío del Ejercicio 2: decir no con empatía |
| 5 | **Producto excluido pero defectuoso** | **Que la excepción prevalece sobre la exclusión** |
| 6 | Producto con sello condicional | Que pregunta en vez de asumir |
| 7 | Devolución de pedido no entregado | Que distingue cancelación de devolución |
| 8 | Queja con mención legal | Que el guardrail escala sin llamar al modelo |

La salida queda en `outputs/ejecucion_fase3.md`, con la intención detectada, los
fragmentos recuperados, el resultado de la verificación de anclaje, el número de
llamadas al LLM y el costo estimado de la corrida.

---

## 8. Qué demuestra la comparación

| Aspecto | Prompt básico | Prompt optimizado |
|---|---|---|
| Acceso a datos reales | Ninguno | *Function calling* + RAG |
| Comportamiento ante un dato que no tiene | Inventa o evade | Lo dice y pide verificar |
| Política de devoluciones | La genérica de internet | La de EcoMarket, citada |
| Producto excluido | Promete una devolución imposible | Niega con motivo y ofrece alternativa |
| Excepción por defecto de fabricación | No la conoce | La aplica con prioridad |
| Trazabilidad | Ninguna | Cita `[id_fragmento]` |
| Formato | Variable | Acotado y consistente |
| Escalamiento | No existe | Determinista, antes del modelo |

La conclusión del ejercicio es que **la ingeniería de prompts es necesaria pero
no suficiente**. El salto de calidad más grande entre las dos columnas no lo
produce una mejor redacción: lo produce que el prompt optimizado tenga los datos
correctos dentro. Es la tesis de RAG que vimos en clase —convertir al modelo de
un "libro cerrado" en un "experto con biblioteca"— y por eso la Fase 1 eligió una
arquitectura y no solamente un modelo.
