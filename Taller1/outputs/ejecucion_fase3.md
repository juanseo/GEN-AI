# Fase 3 — Ejecución real de los prompts

> Salida generada automáticamente por `scripts/demo_fase3.py`.
> Fecha de ejecución: 2026-09-22 22:48  
> Proveedor: `ollama` · Modelo: `glm-5.3-flash` · Motor de retrieval: `tfidf` · Fragmentos indexados: 33

En cada caso se ejecuta la **misma consulta** con dos prompts distintos. La diferencia entre las dos respuestas es, literalmente, la ingeniería de prompts.

---

## Ejercicio 1.a — Pedido en tránsito, caso feliz

**Consulta del cliente:** «Hola, buenas. ¿Me pueden decir cómo va mi pedido ECO-10001?»

**Qué demuestra este caso:** El prompt básico no tiene acceso al sistema de pedidos: o inventa un estado o responde con evasivas. El optimizado invoca la herramienta `consultar_pedido` y responde con el estado, la transportadora, la fecha estimada y el enlace de rastreo reales.

### Prompt BÁSICO

No tengo acceso a un sistema de pedidos ni a información sobre el pedido ECO-10001, por lo que no puedo consultar su estado.

Para obtener esa información, te recomiendo:

- **Revisar el correo de confirmación** del pedido, que suele incluir un enlace de seguimiento.
- **Acceder a tu cuenta** en el sitio web de la tienda, en la sección "Mis pedidos".
- **Contactar al servicio al cliente** de la tienda con el número de pedido.

¿Hay algo más en lo que pueda ayudarte?

> intención detectada: `OTRO`  
> llamadas al LLM: 1

### Prompt OPTIMIZADO

¡Hola! 👋 Con gusto te cuento cómo va tu pedido ECO-10001:

- **Estado:** En tránsito con la transportadora.
- **Transportadora:** Coordinadora, con destino a Cali.
- **Fecha estimada de entrega:** 24 de septiembre de 2026.
- **Productos:** 2 botellas de acero inoxidable 500 ml (total: $178.000 COP).
- Puedes seguirlo en tiempo real aquí: https://track.ecomarket.co/ECO-10001

Todo va en tiempo, sin retrasos registrados. ¿Te ayudo con algo más? 😊

> intención detectada: `ESTADO_PEDIDO`  
> anclaje: OK (cobertura 1.0)  
> llamadas al LLM: 3

---

## Ejercicio 1.b — Pedido retrasado, requiere disculpa y explicación

**Consulta del cliente:** «Mi pedido ECO-10003 ya se pasó de la fecha, ¿qué pasó?»

**Qué demuestra este caso:** El prompt optimizado detecta `retrasado: true`, ofrece la disculpa, explica el motivo real (cierre de vía por clima) y contrasta la fecha original con la nueva. El básico no puede hacer nada de eso.

### Prompt BÁSICO

# Estado del Pedido ECO-10003

Lo siento, no tengo acceso a un sistema de pedidos ni a información de clientes, por lo que no puedo consultar el estado de tu pedido ECO-10003.

Para obtener esa información, te recomiendo:

- **Verificar en tu cuenta** en el sitio web de la tienda
- **Revisar el correo de confirmación** del pedido, que suele incluir un enlace de seguimiento
- **Contactar al servicio al cliente** de la tienda con tu número de pedido

¿Hay algo más en lo que pueda ayudarte?

> intención detectada: `OTRO`  
> llamadas al LLM: 1

### Prompt OPTIMIZADO

🌿 Entiendo tu preocupación, tu pedido sí lleva unos días de retraso y te cuento exactamente qué pasó:

- **Estado:** Retrasado por una novedad logística.
- **Motivo:** La vía fluvial de última milla hacia Leticia está cerrada temporalmente por condiciones climáticas.
- **Transportadora:** Coordinadora.
- **Fecha original de entrega:** 18 de septiembre de 2026.
- **Nueva fecha estimada:** 26 de septiembre de 2026 (4 días después de la original).

Lamentamos sinceramente este inconveniente; el clima en la ruta está fuera de nuestro control, pero tu pedido sigue en movimiento y llegará apenas se reabra el paso.

Puedes seguirlo en tiempo real aquí: https://track.ecomarket.co/ECO-10003

¿Te ayudo con algo más sobre tu pedido?

> intención detectada: `ESTADO_PEDIDO`  
> anclaje: OK (cobertura 1.0)  
> llamadas al LLM: 3

---

## Ejercicio 1.c — Pedido inexistente (prueba anti-alucinación)

**Consulta del cliente:** «Quiero saber el estado del pedido ECO-99999 por favor.»

**Qué demuestra este caso:** Es la prueba crítica. El prompt básico tiende a inventar un estado plausible. El optimizado recibe `encontrado: false` y la instrucción explícita de no inventar, así que pide verificar el número.

### Prompt BÁSICO

Lo siento, no tengo acceso a un sistema de pedidos ni a información de clientes, por lo que no puedo consultar el estado del pedido ECO-99999.

Para conocer su estado, te recomiendo:

1. **Revisar el correo de confirmación** — suele incluir un enlace de seguimiento.
2. **Acceder a tu cuenta** en el sitio web de la tienda, en la sección "Mis pedidos".
3. **Contactar al servicio al cliente** de la tienda con tu número de pedido.

Si me das más contexto (por ejemplo, si tienes información del pedido a mano), puedo ayudarte a interpretarla o con cualquier otra duda.

> intención detectada: `OTRO`  
> llamadas al LLM: 1

### Prompt OPTIMIZADO

🌱 ¡Hola! Con gusto te ayudo a revisar tu pedido.

Busqué el número **ECO-99999** en nuestro sistema, pero no logré ubicar ningún pedido con ese número de seguimiento. Puede que haya una letra o dígito diferente, o que el pedido esté registrado con otro número.

Te sugiero:
- Verificar el número de seguimiento en el correo de confirmación de tu compra.
- Revisar que no falten guiones o caracteres al escribirlo.

Con gusto vuelvo a consultarlo en cuanto me confirmes el número correcto. Si prefieres, también puedo escalar tu caso a un agente humano (disponible de lunes a viernes, 8:00 a 18:00).

¿Me confirmas el número de seguimiento para revisarlo de nuevo?

> intención detectada: `ESTADO_PEDIDO`  
> anclaje: OK (cobertura 1.0)  
> llamadas al LLM: 3

---

## Ejercicio 2.a — Devolución de producto EXCLUIDO (alimento)

**Consulta del cliente:** «Compré la miel orgánica en el pedido ECO-10008 y ya no la quiero. ¿Cómo la devuelvo?»

**Qué demuestra este caso:** El reto del taller. El prompt básico aplica la política genérica de cualquier e-commerce y promete una devolución que EcoMarket no puede cumplir. El optimizado recupera la política, identifica la categoría `NO_DEV_PERECEDERO`, dice que no con empatía y ofrece alternativa.

### Prompt BÁSICO

Lo siento, no tengo acceso a un sistema de pedidos ni a información de clientes, por lo que no puedo consultar el estado del pedido ECO-10008.

Para obtener esa información te recomiendo:

- **Revisar el correo de confirmación** que recibiste al hacer el pedido, ya que suele incluir un enlace de seguimiento.
- **Contactar directamente al servicio al cliente** de la tienda con tu número de pedido.
- **Iniciar sesión en tu cuenta** en el sitio web de la tienda y revisar la sección "Mis pedidos".

¿Hay algo más en lo que pueda ayudarte?

> intención detectada: `OTRO`  
> llamadas al LLM: 1

### Prompt OPTIMIZADO

🌿 Entiendo que ya no quieres la miel orgánica de tu pedido ECO-10008; te explico tu situación con claridad.

- Tu pedido aún está **en tránsito** con Inter Rapidísimo (entrega estimada: 23 de septiembre de 2026). Como no ha sido entregado, lo que corresponde sería una **cancelación**, no una devolución.
- Además, la miel orgánica pertenece a la categoría de alimentos perecederos, que **no admite devolución ni cambio por retracto**, por razones de inocuidad alimentaria y cadena de frío [politicas_devolucion#03] [catalogo_productos#09].
- Importante: si el producto llegara **vencido, dañado o en mal estado**, sí aplica la excepción de garantía con reposición o reembolso total, aunque sea un producto no devolvible.
- Las bolsas de tela de tu pedido sí son devolvibles dentro de los 30 días [catalogo_productos#10].

¿Quieres que te ayude con la cancelación del pedido antes de su entrega, o el producto llegó con algún problema de calidad?

> intención detectada: `DEVOLUCION`  
> fragmentos recuperados: catalogo_productos.md › ECO-MIE-500 — Miel orgánica de abejas nativas 500 g; catalogo_productos.md › ECO-BOL-TEL — Bolsa de tela reutilizable; catalogo_productos.md › ECO-BOT-500 — Botella de acero inoxidable 500 ml; catalogo_productos.md › ECO-TER-CAF — Termo de café reutilizable 350 ml; politicas_devolucion.md › 3. Categorías NO sujetas a devolución  
> anclaje: OK (cobertura 1.0)  
> llamadas al LLM: 2

---

## Ejercicio 2.b — Producto excluido PERO defectuoso (excepción de garantía)

**Consulta del cliente:** «El shampoo en barra del pedido ECO-10009 me llegó partido y con moho. Quiero que me lo repongan.»

**Qué demuestra este caso:** Verifica que el modelo no aplique la exclusión de forma mecánica: la excepción por producto defectuoso prevalece sobre la categoría de higiene, y debe ofrecer reposición o reembolso total.

### Prompt BÁSICO

# Estado del Pedido ECO-10009

Lo siento, pero no tengo acceso a un sistema de pedidos ni a información de clientes, por lo que no puedo consultar el estado de tu pedido ECO-10009.

Te recomiendo:

- **Revisar tu correo electrónico** — probablemente recibiste confirmaciones y actualizaciones del pedido.
- **Contactar al servicio al cliente** de la tienda donde realizaste la compra, proporcionando el número de pedido.
- **Consultar la web de la tienda** — muchas tiendas tienen un apartado de "Seguimiento de pedidos" donde puedes ingresar tu número.

¿Hay algo más en lo que pueda ayudarte?

> intención detectada: `OTRO`  
> llamadas al LLM: 1

### Prompt OPTIMIZADO

Lamento mucho que el shampoo en barra te haya llegado partido y con moho; entiendo tu preocupación y vamos a resolverlo. 🌱

- Tu pedido ECO-10009 figura como **Entregado** el 2026-08-20 en Cartagena, con 2 unidades del shampoo en barra (COP 76.000 en total).
- Como el producto llegó averiado, aplica la **excepción por defecto o avería**: puedes acceder a una **reposición o reembolso total**, aunque los productos de higiene normalmente solo se devuelven con el sello intacto [politicas_devolucion#05].
- El procedimiento es: entra a **Mi cuenta → Mis pedidos**, selecciona el pedido con el número de seguimiento, elige **"Solicitar devolución"** e indica el motivo. El sistema validará la elegibilidad y generará la guía de retorno; luego debes entregar el producto en la transportadora dentro de los **5 días hábiles** siguientes [politicas_devolucion#09].
- Al recibirlo, EcoMarket hará la inspección y te notificará el resultado por correo [politicas_devolucion#09].

¿Quieres que te guíe paso a paso para iniciar la solicitud en este momento?

> intención detectada: `DEVOLUCION`  
> fragmentos recuperados: catalogo_productos.md › ECO-SHA-BAR — Shampoo en barra sin sulfatos; faq_envios.md › ¿Qué significa cada estado del pedido?; politicas_devolucion.md › 8. Procedimiento de devolución (pasos para el cliente); catalogo_productos.md › ECO-JAB-SOL — Jabón sólido artesanal de caléndula; politicas_devolucion.md › 4. Categorías con condiciones especiales; catalogo_productos.md › ECO-PAN-REU — Pañitos reutilizables de algodón orgánico (x10)  
> anclaje: OK (cobertura 1.0)  
> llamadas al LLM: 2

---

## Ejercicio 2.c — Producto con sello condicional (no asumir)

**Consulta del cliente:** «Quiero devolver el jabón de caléndula que venía en el pedido ECO-10003.»

**Qué demuestra este caso:** El producto es `DEV_CONDICIONAL_SELLO`: el prompt obliga al modelo a preguntar por el sello de seguridad antes de prometer la devolución, en lugar de asumir una respuesta.

### Prompt BÁSICO

Hola, gracias por contactarnos. Con gusto te explico cómo devolver el jabón de caléndula de tu pedido ECO-10003:

**Pasos para la devolución:**

1. **Solicita la devolución** desde tu cuenta en la sección "Mis pedidos", seleccionando el pedido ECO-10003 y el producto a devolver. También puedes hacerlo respondiendo a este mensaje.

2. **Prepara el producto:** debe estar sin usar (o en su estado original), con su embalaje y etiquetas intactas.

3. **Empaqueta el artículo** de forma segura, idealmente en la caja original.

4. **Envía el paquete** a la dirección que te indicaremos al confirmar la solicitud. En muchos casos te enviaremos una etiqueta de envío prepagada.

5. **Recibe tu reembolso:** una vez que recibamos y verifiquemos el producto, procesaremos el reembolso a tu método de pago original en un plazo de 5-10 días hábiles.

**Ten en cuenta:**
- Por motivos de higiene, algunos productos de cuidado personal solo pueden devolverse si están sin abrir y precintados.
- El plazo habitual para devoluciones es de 30 días desde la recepción del pedido.

¿Quieres que inicie la solicitud de devolución ahora? Solo confírmame y te enviaré la etiqueta de envío. 😊

> intención detectada: `OTRO`  
> llamadas al LLM: 1

### Prompt OPTIMIZADO

Entiendo tu intención de devolver el jabón de caléndula del pedido ECO-10003, y te explico con transparencia lo que aplica en este momento:

- Tu pedido aún **no ha sido entregado**: está en estado `RETRASADO` por un cierre temporal de la vía fluvial hacia Leticia, con nueva fecha estimada de entrega para el 26 de septiembre [faq_envios#03].
- Como no lo has recibido, no corresponde una **devolución** (eso aplica a productos ya en tu poder), sino una **cancelación** si prefieres no esperar.
- Ten en cuenta que la cancelación sin costo solo está garantizada mientras el pedido está en `EN_PREPARACION`; en tu estado actual no puedo confirmarte si es posible sin costo, así que lo ideal es que un agente humano lo revise.
- Cuando el pedido llegue, el jabón sí admite devolución, pero solo con el **sello de seguridad intacto** [catalogo_productos#03].

Puedes rastrear tu pedido aquí: https://track.ecomarket.co/ECO-10003

¿Quieres que escale tu caso a un agente humano para evaluar la cancelación del pedido?

> intención detectada: `DEVOLUCION`  
> fragmentos recuperados: catalogo_productos.md › ECO-JAB-SOL — Jabón sólido artesanal de caléndula; catalogo_productos.md › ECO-ESP-KON — Esponja vegetal konjac; faq_envios.md › ¿Qué significa cada estado del pedido?; politicas_devolucion.md › 8. Procedimiento de devolución (pasos para el cliente); catalogo_productos.md › ECO-SHA-BAR — Shampoo en barra sin sulfatos; catalogo_productos.md › ECO-PAN-REU — Pañitos reutilizables de algodón orgánico (x10)  
> anclaje: OK (cobertura 1.0)  
> llamadas al LLM: 2

---

## Ejercicio 2.d — Devolución imposible: el pedido no ha llegado

**Consulta del cliente:** «Quiero devolver la compostera del pedido ECO-10004.»

**Qué demuestra este caso:** El pedido está `EN_PREPARACION`. El paso 5 del árbol de decisión hace que el modelo aclare que corresponde una cancelación, no una devolución, y explique la diferencia.

### Prompt BÁSICO

# Devolución de tu compostera – Pedido ECO-10004

¡Hola! Con gusto te explico el proceso de devolución:

## Pasos a seguir

1. **Solicita la devolución** desde tu cuenta en la sección "Mis pedidos", seleccionando el pedido ECO-10004 y la compostera. También puedes escribirnos por este canal para gestionarlo.

2. **Plazo**: Tienes **30 días naturales** desde la recepción del pedido para solicitar la devolución.

3. **Estado del producto**: La compostera debe estar sin usar, en su embalaje original y con todos sus accesorios y manuales incluidos.

4. **Recogida o envío**: 
   - Te enviaremos una **etiqueta de devolución** por correo electrónico.
   - Puedes llevar el paquete a un punto de recogida o acordar la recogida a domicilio (según disponibilidad en tu zona).

5. **Reembolso**: Una vez recibido y revisado el producto, te devolveremos el importe en un plazo de **5 a 7 días hábiles** por el mismo método de pago utilizado en la compra.

## Datos que necesitamos de ti

Para agilizar el proceso, indícanos:
- Motivo de la devolución (producto defectuoso, no cumple expectativas, etc.)
- Si el paquete está sin abrir o ya fue abierto

¿Hay algo más en lo que pueda ayudarte? 😊

> intención detectada: `OTRO`  
> llamadas al LLM: 1

### Prompt OPTIMIZADO

🌿 Entiendo que quieres devolver la compostera del pedido ECO-10004, y te ayudo con gusto.

Revisé tu pedido y encontré algo importante:

- **Estado actual:** En preparación en bodega; aún no ha sido despachado ni entregado [faq_envios#03].
- **Qué aplica:** Como el producto no ha llegado a tu dirección, lo que corresponde no es una devolución, sino una **cancelación**. La diferencia es que la devolución aplica cuando ya recibiste el producto, mientras que la cancelación anula el pedido antes del despacho.
- **Buena noticia:** Los pedidos en estado `EN_PREPARACION` todavía se pueden cancelar **sin costo** [faq_envios#03].
- Si prefieres esperar la entrega y luego devolverla, la compostera tiene 30 días por retracto y 12 meses de garantía por defectos de fabricación [politicas_devolucion#05].

¿Quieres que procedamos con la cancelación sin costo del pedido ECO-10004, o prefieres esperar la entrega?

> intención detectada: `DEVOLUCION`  
> fragmentos recuperados: catalogo_productos.md › ECO-COM-13L — Compostera doméstica 13 L; faq_envios.md › ¿Qué significa cada estado del pedido?; politicas_devolucion.md › 8. Procedimiento de devolución (pasos para el cliente); catalogo_productos.md › Introducción; politicas_devolucion.md › 4. Categorías con condiciones especiales; politicas_devolucion.md › 3. Categorías NO sujetas a devolución  
> anclaje: OK (cobertura 1.0)  
> llamadas al LLM: 2

---

## Guardrail — Escalamiento automático a agente humano

**Consulta del cliente:** «Esto es un pésimo servicio, me cobraron dos veces y voy a poner una queja en la SIC.»

**Qué demuestra este caso:** El guardrail determinista intercepta la consulta antes de llamar al modelo: hay mención de cobro duplicado y de la SIC. No se gasta un token y el caso va directo a una persona.

### Prompt OPTIMIZADO

Entiendo la situación y lamento mucho lo ocurrido. Este caso lo va a atender directamente una persona del equipo de EcoMarket, que tendrá todo el detalle de tu conversación a la mano. Te contactaremos dentro de las próximas 2 horas hábiles. Gracias por tu paciencia.

> intención detectada: `QUEJA`  
> **escalado a humano** — [financiero] Se detectó el patrón 'me cobraron dos veces'.  
> llamadas al LLM: 0

---

## Consumo de la ejecución

- Llamadas al API: **24**
- Tokens de entrada: **18,337**
- Tokens de salida: **3,156**
- Costo estimado: **USD 0.00000**

> El costo se calcula con los precios de referencia declarados en `src/llm_client.py`. Verificar los precios vigentes antes de extrapolar. Los modelos de Ollama Cloud se cobran por suscripción, no por token, por lo que su costo aparece como 0.
