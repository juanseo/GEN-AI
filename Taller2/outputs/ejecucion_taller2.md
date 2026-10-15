# Fase 3 — Ejecución del asistente con RAG

> Generado por `python -m scripts.demo_taller2`.  
> Fecha: 2026-10-15 18:29 · Fecha de referencia del asistente ("hoy"): 2026-09-22  
> LLM: `glm-5.3-flash` (ollama) · Embeddings: `bge-m3` · Base vectorial: ChromaDB (55 fragmentos) · top_k = 4 · umbral = 0.41

---

## A. Con RAG vs. sin RAG

Misma pregunta, mismo modelo (`glm-5.3-flash`). La única diferencia es el acceso a la base de conocimiento.

### A.1 — Medio de pago que EcoMarket no ofrece

**Cliente:** «¿Le puedo pagar en efectivo al mensajero cuando me traiga el paquete?»

*Qué se espera:* Sin RAG, el modelo responde con lo típico de un e-commerce colombiano y puede prometer pago contra entrega. Con RAG, la FAQ dice que no existe.

**Sin RAG**

> ¡Hola! 😊 Gracias por escribir a EcoMarket.
>
> Sí, ¡claro que sí! Ofrecemos la opción de **pago contra entrega** en efectivo. Cuando nuestro mensajero llegue con tu paquete, puedes pagarle directamente en efectivo al recibir tu pedido.
>
> Algunos detalles a tener en cuenta:
>
> - 💵 Esta opción está disponible para la mayoría de destinos dentro de Colombia.
> - 📍 Te recomendamos tener el monto exacto o cercano, ya que el mensajero podría no llevar cambio suficiente.
> - 📦 Verifica tu pedido en el momento de la entrega antes de realizar el pago.
>
> Si prefieres otras formas de pago, también aceptamos tarjetas de crédito/débito, transferencias por Nequi, Daviplata y PSE.
>
> ¿Hay algo más en lo que pueda ayudarte? ¡Que tengas un día lleno de verde! 🌱

**Con RAG**

> Entiendo tu duda sobre las opciones de pago al recibir tu pedido.
>
> Lamentablemente, no es posible: EcoMarket no ofrece pago contra entrega ni pago en efectivo al recibir. El pedido se despacha solo cuando la pasarela confirma el pago; antes de eso queda en estado PAGO_PENDIENTE [faq#PAG-02].
>
> ¿Te gustaría que te ayude con algo más sobre tu pedido o tu método de pago?

> intención: `PAGO_CUENTA`  
> llamadas al LLM: 2  
> fragmentos (similitud): `faq#PAG-02` 0.65, `faq#ENV-02` 0.61, `faq#ENV-04` 0.57, `faq#ENV-05` 0.57  
> anclaje: OK (cobertura 1.0)

### A.2 — Producto personalizado: no devolvible

**Cliente:** «Compré una agenda con mi nombre grabado y ya no la quiero, ¿la puedo devolver?»

*Qué se espera:* Sin RAG, lo usual es prometer 30 días de devolución. Con RAG, la ficha del producto (`NO_DEV_PERSONALIZADO`) y la sección 3 de la política dicen que no.

**Sin RAG**

> ¡Hola! 🌿 Gracias por escribir a EcoMarket.
>
> Entiendo tu situación, y lamento que la agenda ya no sea de tu agrado. Sin embargo, debo informarte que los **productos personalizados** (como agendas con nombre grabado) **no son aptos para devolución o cambio**, ya que fueron fabricados especialmente para ti y no pueden revenderse.
>
> Esto está indicado en nuestra política de devoluciones, que puedes consultar en la sección "Políticas de la tienda" de nuestra página web.
>
> **Excepciones importantes:** sí puedes solicitar cambio o reembolso si el producto presenta alguna de estas situaciones:
> - ❌ Errores en la personalización (nombre mal escrito, diferente a lo solicitado)
> - ❌ Defectos de fabricación o daños al recibirlo
> - ❌ Producto diferente al que compraste
>
> Si aplica alguna de estas opciones, escríbenos con una foto del producto y tu número de orden, y con gusto te ayudaremos. 📸
>
> ¿Hay algo más en lo que pueda ayudarte? 💚

**Con RAG**

> Entiendo que ya no quieras la agenda, y lamento que no sea la noticia que esperabas.
>
> - La agenda con nombre grabado (SKU ECO-AGE-PER) es un producto personalizado, fabricado exclusivamente para ti, por lo que no admite devolución ni cambio [catalogo#ECO-AGE-PER] [politica#s3-p0].
> - Excepción importante: si la agenda llegó defectuosa, averiada, incompleta o distinta a la comprada, sí tienes derecho a reposición o reembolso total mediante el flujo de garantía [politica#s3-p0].
>
> Como alternativa, puedo ayudarte a revisar otras agendas de papelería sin personalización, que sí serían devolvibles dentro del plazo general de 30 días desde la entrega [politica#s1-p0].
>
> ¿La agenda llegó en buen estado y conforme a lo pedido, o presentó algún defecto?

> intención: `DEVOLUCION`  
> llamadas al LLM: 2  
> fragmentos (similitud): `catalogo#ECO-AGE-PER` 0.67, `faq#CTA-01` 0.57, `politica#s1-p0` 0.51, `politica#s8-p0` 0.50, `politica#s3-p0` 0.50 *(expansión)*  
> anclaje: OK (cobertura 1.0)

### A.3 — Programa propio de la empresa

**Cliente:** «¿Tienen algún programa para reciclar los frascos vacíos?»

*Qué se espera:* Un modelo general no puede conocer el programa Vuelve a Casa ni su cupón.

**Sin RAG**

> ¡Hola! 🌱 Gracias por escribir a EcoMarket, qué gusto saludarte.
>
> ¡Claro que sí! Nos encanta que pienses en darle una segunda vida a tus frascos. 💚 Te cuento sobre nuestro **Programa "Retorna y Renueva"**:
>
> **¿Cómo funciona?**
> 1. **Acumula tus frascos vacíos** de productos EcoMarket (limpios y sin etiquetas si es posible).
> 2. **Tráelos o envíalos** a nuestro punto de recolección en Bogotá, o solicita una recogida a domicilio para pedidos superiores a 5 frascos.
> 3. **Gana beneficios**: por cada frasco que retornes, acumulas **puntos Eco** que puedes canjear por descuentos en tus próximas compras (¡hasta 10% de descuento!).
>
> **Además**, los frascos en buen estado son reacondicionados y reutilizados en nuestra línea a granel, y los que no, se reciclan adecuadamente con nuestros aliados locales. ♻️
>
> ¿Te gustaría que te comparta la dirección del punto de recolección más cercano o los detalles para programar una recogida? 😊
>
> ¡Gracias por ser parte del cambio hacia un consumo más consciente!
>
> Con cariño,
> **Eco** 🌿
> Asistente Virtual de EcoMarket

**Con RAG**

> ¡Qué buena iniciativa la de reciclar tus frascos! Sí tenemos un programa para eso:
>
> - Se llama "Vuelve a Casa": llevas los frascos y envases vacíos de productos EcoMarket a los puntos aliados en Cali, Bogotá y Medellín.
> - Por cada 5 envases que entregues, recibes un cupón de COP 5.000 para tu próxima compra [faq#SOS-03].
>
> ¿Te gustaría que te conecte con un agente humano para conocer las direcciones exactas de los puntos aliados más cercanos?

> intención: `OTRO`  
> llamadas al LLM: 2  
> fragmentos (similitud): `faq#SOS-03` 0.69, `faq#SOS-01` 0.53, `catalogo#ECO-BOL-TEL` 0.50, `catalogo#ECO-COM-13L` 0.50  
> anclaje: OK (cobertura 1.0)

---

## B. Preguntas que el Taller 1 no podía responder

Pagos, cuenta y catálogo ampliado: ahora están en la base de conocimiento.

### B.1 — Facturación a nombre de empresa (intención PAGO_CUENTA, nueva)

**Cliente:** «Necesito que la factura salga a nombre de mi empresa, ¿cómo hago?»

**Asistente**

> Entiendo, necesitas que la factura quede a nombre de tu empresa. Te cuento cómo funciona:
>
> - Si la compra aún no se ha realizado: ingresa el NIT y la razón social en el checkout antes de pagar, y la factura electrónica validada por la DIAN llegará a tu correo dentro de las 24 horas siguientes a la confirmación del pago [faq#PAG-04].
> - Si la factura ya fue emitida: no se puede modificar. El cambio de titular requiere una nota crédito que gestiona un agente humano, y solo puede hacerse dentro del mismo mes de la compra [faq#PAG-04].
>
> ¿Tu factura ya fue emitida o estás por hacer una nueva compra? Si ya está emitida, con gusto te conecto con un agente humano para gestionar la nota crédito.

> intención: `PAGO_CUENTA`  
> llamadas al LLM: 2  
> fragmentos (similitud): `faq#PAG-04` 0.62, `faq#CTA-01` 0.54, `faq#CTA-03` 0.51, `faq#CTA-04` 0.51  
> anclaje: OK (cobertura 1.0)

### B.2 — Catálogo con lenguaje coloquial

**Cliente:** «¿Qué tienen para llevar el tinto calientico a la oficina?»

**Asistente**

> ¡Qué buena idea llevar el tinto calientico a la oficina! Te tengo esta opción en nuestro catálogo:
>
> - Termo de café reutilizable 350 ml (SKU ECO-TER-CAF): COP 98.000. Tiene tapa antiderrame, interior cerámico y mantiene las bebidas calientes hasta 5 horas. Está disponible y puedes devolverlo hasta 30 días después de la entrega [catalogo#ECO-TER-CAF].
>
> Si también quieres algo para el almuerzo en la oficina, tenemos el set de cubiertos de bambú portátil a COP 34.000, aunque con pocas unidades disponibles [catalogo#ECO-CUB-BAM].
>
> ¿Te gustaría que te ayude con algo más sobre el termo o tu pedido?

> intención: `PRODUCTO`  
> llamadas al LLM: 2  
> fragmentos (similitud): `faq#ENV-01` 0.44, `catalogo#ECO-CUB-BAM` 0.43, `catalogo#ECO-JAB-SOL` 0.41, `catalogo#ECO-TER-CAF` 0.41  
> anclaje: OK (cobertura 1.0)

### B.3 — Derecho de supresión de datos (Ley 1581)

**Cliente:** «Quiero que borren toda mi información de su base de datos.»

**Asistente**

> Entiendo tu solicitud de eliminar tus datos personales, y con gusto te explico cómo hacerlo.
>
> - Puedes pedir la supresión de tus datos desde Mi cuenta > Mis datos, o escribiendo a datos@ecomarket.co [faq#CTA-03].
> - La solicitud se responde en un máximo de 15 días hábiles [faq#CTA-03].
> - Ten en cuenta que los datos que la ley obliga a conservar, como las facturas, se mantienen bloqueados durante el plazo legal [faq#CTA-03].
>
> ¿Quieres que te conecte con un agente humano para acompañarte en el proceso?

> intención: `PAGO_CUENTA`  
> llamadas al LLM: 2  
> fragmentos (similitud): `faq#CTA-03` 0.63, `faq#CTA-04` 0.47, `faq#PAG-05` 0.44, `politica#s0-p0` 0.44  
> anclaje: OK (cobertura 1.0)

---

## C. Fuera de alcance: reconocer que no se tiene la información

El enunciado pide responder cualquier solicitud o reconocer que no se puede atender. Hay dos barreras: el umbral de similitud (sin gastar tokens de generación) y el prompt estricto.

### C.1 — Totalmente fuera del dominio

**Cliente:** «¿Cuál es la capital de Australia?»

*Qué se espera:* Ningún fragmento supera el umbral: responde el mensaje fijo, sin llamar al LLM de generación.

**Asistente**

> Gracias por escribirnos. No encontré información sobre eso en las fuentes de EcoMarket, así que prefiero no darte una respuesta que podría ser incorrecta. Puedo ayudarte con pedidos, envíos, devoluciones, pagos, tu cuenta y nuestros productos. Si lo prefieres, te conecto con una persona del equipo (lunes a viernes, 8:00 a 18:00). ¿Quieres que lo haga?

> intención: `OTRO`  
> llamadas al LLM: 1  
> **fuera de alcance** (ningún fragmento supera el umbral)  
> fragmentos (similitud): `faq#ENV-02` 0.25, `catalogo#ECO-AGE-PER` 0.25, `catalogo#ECO-KIT-HUE` 0.25, `faq#ATN-02` 0.24

### C.2 — Cerca del dominio: pasa el umbral

**Cliente:** «¿Tienen vacantes de trabajo en EcoMarket?»

*Qué se espera:* Menciona a EcoMarket, así que el retrieval encuentra fragmentos parecidos. La segunda barrera es el prompt estricto: el contexto no habla de empleo.

**Asistente**

> Gracias por tu interés en EcoMarket. Lamentablemente, no tengo información sobre vacantes laborales ni procesos de reclutamiento, ya que mi función es atender consultas sobre pedidos, envíos, devoluciones, pagos, productos y cuentas.
>
> Si deseas conocer sobre oportunidades de empleo, te recomiendo contactar al equipo humano por estos canales:
> - Chat del sitio web o de la app
> - WhatsApp o redes sociales
> - Correo a servicio@ecomarket.co [faq#ATN-02]
>
> ¿Te gustaría que te ayude con alguna consulta sobre tus pedidos o nuestros productos?

> intención: `OTRO`  
> llamadas al LLM: 2  
> fragmentos (similitud): `faq#ATN-03` 0.66, `faq#ENV-06` 0.62, `faq#ATN-02` 0.58, `faq#SOS-03` 0.57  
> anclaje: OK (cobertura 1.0)

### C.3 — Producto que EcoMarket no vende

**Cliente:** «Se me dañó la lavadora, ¿cómo la arreglo?»

*Qué se espera:* Se parece a la sección de garantías de electrodomésticos, pero no es un producto de EcoMarket.

**Asistente**

> Entiendo tu preocupación con la lavadora dañada, y quiero ayudarte con esto.
>
> Sin embargo, no tengo información suficiente para resolver tu caso:
>
> - La lavadora no aparece en el catálogo de EcoMarket que tengo disponible, y no recibí datos de tu pedido (¿me compartes el número de seguimiento?).
> - Si el equipo se dañó por un defecto de fabricación, los electrodomésticos cuentan con garantía legal de 12 meses, pero necesito verificar tu compra para confirmarlo [politica#s4-p0].
> - Si el daño no es de fabricación, lamentablemente no puedo ofrecerte una solución con la información actual.
>
> Para avanzar, por favor compárteme el número de seguimiento de tu pedido, o si prefieres, te conecto con un agente humano del equipo (disponible lunes a viernes, 8:00 a 18:00) para revisar tu caso en detalle.
>
> ¿Cuentas con el número de seguimiento a la mano?

> intención: `DEVOLUCION`  
> llamadas al LLM: 2  
> fragmentos (similitud): `politica#s4-p0` 0.47, `faq#ENV-04` 0.46, `catalogo#ECO-COM-13L` 0.45, `politica#s8-p0` 0.44  
> anclaje: OK (cobertura 1.0)

---

## D. Integración con la cadena del Taller 1

Pedidos por function calling, devoluciones con RAG y escalamiento por reglas.

### D.1 — Estado de pedido retrasado (function calling)

**Cliente:** «Hola, ¿cómo va mi pedido ECO-10003? Ya se pasó de la fecha.»

**Asistente**

> Hola, gracias por escribirnos y lamento la espera de tu pedido.
>
> - Estado: Retrasado por una novedad logística.
> - Transportadora: Coordinadora, con destino a Leticia.
> - Motivo: un cierre temporal de la vía por condiciones climáticas en la ruta fluvial de última milla.
> - La fecha estimada original era el 18 de septiembre; la nueva fecha de entrega es el 26 de septiembre de 2026.
> - Puedes seguirlo en todo momento aquí: https://track.ecomarket.co/ECO-10003
>
> Te pido disculpas sinceras por la demora, entendemos lo molesto que es esperar más de lo previsto.
>
> ¿Te ayudo con algo más sobre tu pedido o tu cuenta?

> intención: `ESTADO_PEDIDO`  
> llamadas al LLM: 3  
> herramienta invocada por el modelo: `consultar_pedido`  
> anclaje: OK (cobertura 1.0)

### D.2 — Producto excluido PERO defectuoso (RAG + expansión por código)

**Cliente:** «El shampoo del pedido ECO-10009 me llegó partido y con moho, quiero que me lo repongan.»

**Asistente**

> Lamento mucho que el shampoo te haya llegado partido y con moho, entiendo lo frustrante que debe ser.
>
> Tengo buenas noticias: aunque los productos de higiene personal normalmente no admiten devolución, cuando un producto llega averiado o defectuoso aplica la excepción de garantía, y tienes derecho a reposición o reembolso total [politica#s3-p0].
>
> Sobre tu pedido ECO-10009:
> - Producto: Shampoo en barra sin sulfatos (x2), entregado el 2026-08-20 en Cartagena.
> - Como pediste reposición, puedo gestionar el envío de un reemplazo.
>
> Para continuar, confírmame por favor:
> - ¿Prefieres reposición de las dos unidades o reembolso total?
> - ¿Puedes describir si el daño afectó ambas barras o solo una?
>
> Con tu confirmación genero la solicitud por el flujo de garantía y te comparto los siguientes pasos.

> intención: `DEVOLUCION`  
> llamadas al LLM: 2  
> fragmentos (similitud): `catalogo#ECO-SHA-BAR` 0.71, `catalogo#ECO-CEP-BAM` 0.55, `politica#s4-p0` 0.52, `catalogo#ECO-DES-SOL` 0.52, `politica#s3-p0` 0.47 *(expansión)*  
> anclaje: OK (cobertura 1.0)

### D.3 — Escalamiento determinista

**Cliente:** «Esto es un pésimo servicio, me cobraron dos veces y voy a poner una queja en la SIC.»

**Asistente**

> Entiendo la situación y lamento mucho lo ocurrido. Este caso lo va a atender directamente una persona del equipo de EcoMarket, que tendrá todo el detalle de tu conversación a la mano. Te contactaremos dentro de las próximas 2 horas hábiles. Gracias por tu paciencia.

> intención: `QUEJA`  
> llamadas al LLM: 0  
> **escalado a humano**: [financiero] Se detectó el patrón 'me cobraron dos veces'.

---

## E. Experimentos: modificar una pieza y observar el cambio

Cada experimento cambia **una sola** variable y deja todo lo demás igual. Tokens = entrada / salida de esa ejecución.

### E.1 — `top_k`: cuántos fragmentos llegan al prompt

**Cliente:** «¿Cuánto me cuesta el envío a Bogotá y en cuántos días me llega?»

La respuesta necesita dos fragmentos distintos (costo en `faq#ENV-02`, tiempo en `faq#ENV-01`).

**top_k = 1 — fragmentos: faq#ENV-01 — tokens 897 / 128**

> ¡Claro que te ayudo con tu envío a Bogotá!
>
> - Tiempo de entrega: 2 a 4 días hábiles desde el despacho [faq#ENV-01].
> - La preparación del pedido en bodega toma entre 1 y 2 días hábiles adicionales [faq#ENV-01].
>
> Sobre el costo del envío a Bogotá, no tengo esa información en mis fuentes autorizadas en este momento, así que no puedo confirmarte el valor.
>
> ¿Quieres que te conecte con un agente humano para que te confirme el costo del envío?

**top_k = 4 — fragmentos: faq#ENV-01, faq#ENV-02, faq#ENV-04, politica#s6-p0 — tokens 1259 / 149**

> ¡Claro que te cuento! Estos son los datos para envíos a Bogotá:
>
> - Costo: el envío es gratis en compras superiores a COP 150.000; por debajo de ese valor, la tarifa plana es de COP 14.000 para ciudades principales como Bogotá [faq#ENV-02].
> - Tiempo de entrega: de 2 a 4 días hábiles desde el despacho [faq#ENV-01].
> - Ten en cuenta que la preparación del pedido en bodega toma entre 1 y 2 días hábiles adicionales [faq#ENV-01].
>
> ¿Quieres que revise el estado de algún pedido que ya tengas en camino?

### E.2 — Prompt estricto vs. permisivo (una sola instrucción cambia)

**Cliente:** «¿Cuántos años dura la batería de la lámpara solar antes de tener que cambiarla?»

La ficha de `ECO-LAM-SOL` dice "hasta 8 h de luz" pero no habla de la vida útil de la batería.

**Estricto (`RAG_GENERAL`)**

> Entiendo tu interés en la durabilidad de la lámpara solar antes de comprarla.
>
> Lamentablemente, en la información disponible del producto no se especifica la vida útil de la batería (cuántos años dura antes de cambiarla). Lo que sí puedo contarte de la Lámpara solar recargable para exterior (ECO-LAM-SOL):
>
> - Batería de litio recargable, con hasta 8 h de luz por carga [catalogo#ECO-LAM-SOL]
> - Panel solar monocristalino y resistencia a la lluvia (IP65) [catalogo#ECO-LAM-SOL]
> - Cuenta con 12 meses de garantía por defectos de fabricación [catalogo#ECO-LAM-SOL]
>
> ¿Quieres que te conecte con un agente humano para que te confirme la vida útil exacta de la batería?

**Permisivo (`RAG_PERMISIVO`)**

> Entiendo tu interés en la durabilidad de la lámpara solar antes de comprarla.
>
> Sobre la lámpara solar recargable para exterior (SKU ECO-LAM-SOL), el catálogo indica que usa batería de litio recargable y ofrece hasta 8 horas de luz por carga [catalogo#ECO-LAM-SOL]. Sin embargo, no tengo información sobre la vida útil de la batería en años ni sobre cuándo debería reemplazarse, así que no puedo darte ese dato con certeza.
>
> Lo que sí puedo confirmarte:
> - Cuenta con 12 meses de garantía por defectos de fabricación [catalogo#ECO-LAM-SOL].
> - Si la batería presentara fallas dentro de ese periodo, estaría cubierta por la garantía [catalogo#ECO-LAM-SOL].
>
> ¿Te gustaría que conecte tu pregunta sobre la vida útil de la batería con un agente humano del equipo, disponible de lunes a viernes de 8:00 a 18:00?

### E.3 — Umbral de relevancia activado vs. desactivado

**Cliente:** «¿Cuál es la capital de Australia?»

**Umbral activado — similitud máxima 0.25 (umbral 0.41) — tokens 0 / 0**

> Gracias por escribirnos. No encontré información sobre eso en las fuentes de EcoMarket, así que prefiero no darte una respuesta que podría ser incorrecta. Puedo ayudarte con pedidos, envíos, devoluciones, pagos, tu cuenta y nuestros productos. Si lo prefieres, te conecto con una persona del equipo (lunes a viernes, 8:00 a 18:00). ¿Quieres que lo haga?

**Umbral desactivado + prompt estricto — similitud máxima 0.25 (umbral 0.41) — tokens 1156 / 118**

> Gracias por escribir a EcoMarket. Aunque tu pregunta es interesante, no está relacionada con nuestros productos, pedidos, envíos o servicios, y no tengo información autorizada para responderla.
>
> En EcoMarket puedo ayudarte con temas como:
> - Costos de envío [faq#ENV-02]
> - Consultas sobre productos de nuestro catálogo [catalogo#ECO-AGE-PER]
> - Canales de contacto con nuestro equipo [faq#ATN-02]
>
> ¿Hay algo sobre tu experiencia de compra en EcoMarket en lo que pueda ayudarte?

**Umbral desactivado + prompt permisivo — similitud máxima 0.25 (umbral 0.41) — tokens 1117 / 125**

> ¡Gracias por escribir a EcoMarket! Tu pregunta sobre la capital de Australia está fuera de lo que puedo ayudarte, ya que soy el asistente de atención al cliente de EcoMarket y solo cuento con información sobre pedidos, envíos, pagos, productos y cuentas de la tienda.
>
> Si tienes alguna consulta sobre nuestros productos sostenibles, envíos o tu cuenta, con gusto te ayudo. Por ejemplo, puedo contarte sobre tarifas de envío [faq#ENV-02] o sobre nuestros productos disponibles.
>
> ¿Hay algo relacionado con EcoMarket en lo que pueda ayudarte hoy?

### E.4 — Cambiar el modelo de embeddings (solo recuperación)

Mismo índice, mismos fragmentos; cambia el modelo que los vectoriza. Se muestran los 3 primeros fragmentos de cada uno.

| Consulta | TF-IDF (Taller 1) | bge-m3 |
|---|---|---|
| «¿A qué horas puedo hablar con una persona de verdad?» | `catalogo#ECO-CEP-BAM` 0.12<br>`faq#PAG-05` 0.12<br>`faq#PAG-04` 0.12 | `faq#ATN-01` 0.59<br>`politica#s9-p0` 0.54<br>`faq#PAG-04` 0.46 |
| «¿Pueden mandarle un pedido a mi hermana que vive en Miami?» | `faq#ENV-05` 0.18<br>`faq#PAG-02` 0.15<br>`faq#CTA-01` 0.13 | `faq#ENV-05` 0.50<br>`faq#CTA-01` 0.48<br>`faq#ENV-01` 0.47 |

TF-IDF compara pedazos de palabras; bge-m3 compara significado. La consulta de Miami es un fallo conocido de ambos: la FAQ dice "envíos internacionales" y ningún modelo conecta "Miami" con "internacional" con suficiente fuerza (ver Fase 3, limitaciones).

---

## Consumo de la ejecución

- Tokens de entrada: **25,499** · de salida: **3,463**
- Costo estimado: **USD 0.0000** (precios de referencia en `src/llm.py`)
- Duración: 32 s
