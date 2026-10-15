# Fase 3 — Ejecución del asistente con RAG

> Generado por `python -m scripts.demo_taller2`.  
> Fecha: 2026-10-15 09:28 · Fecha de referencia del asistente ("hoy"): 2026-09-22  
> LLM: `llama3.2:3b` (local) · Embeddings: `bge-m3` · Base vectorial: ChromaDB (55 fragmentos) · top_k = 4 · umbral = 0.41

---

## A. Con RAG vs. sin RAG

Misma pregunta, mismo modelo (`llama3.2:3b`). La única diferencia es el acceso a la base de conocimiento.

### A.1 — Medio de pago que EcoMarket no ofrece

**Cliente:** «¿Le puedo pagar en efectivo al mensajero cuando me traiga el paquete?»

*Qué se espera:* Sin RAG, el modelo responde con lo típico de un e-commerce colombiano y puede prometer pago contra entrega. Con RAG, la FAQ dice que no existe.

**Sin RAG**

> ¡Hola! Gracias por contactarnos. Entiendo que deseas pagar en efectivo al mensajero cuando te traiga el paquete. Sin embargo, por razones de seguridad y cumplimiento, no podemos aceptar pagos en efectivo directamente a nuestro mensajero.
>
> Sin embargo, podemos ofrecerte una solución alternativa. Puedes pagar en efectivo en nuestra tienda física, si la tienes cerca de tu ubicación, o puedes utilizar una de nuestras opciones de pago en línea, como PayPal o tarjeta de crédito, que te permitirán pagar en línea y que la cantidad se procesará de manera segura.
>
> Si prefieres pagar en efectivo al mensajero, puedo ofrecerte una opción de pago en línea y te enviaré un código de pago que puedes utilizar para pagar en línea. ¿Te gustaría que te ayude con eso?

**Con RAG**

> Tú estás preguntando si puedes pagar en efectivo al mensajero cuando te traiga el paquete. [id_fragmento: pag-02]. Según nuestra política, no ofrecemos pago contra entrega ni pago en efectivo al recibir. [id_fragmento: pag-02]. El pedido se despacha cuando la pasarela confirma el pago; mientras tanto queda en estado PAGO_PENDIENTE. [id_fragmento: pag-02]. Por lo tanto, no es posible pagar en efectivo al mensajero. ¿Quieres saber más sobre nuestros pagos?

> intención: `ENVIO`  
> llamadas al LLM: 2  
> fragmentos (similitud): `faq#PAG-02` 0.65, `faq#ENV-02` 0.60, `faq#ENV-04` 0.57, `faq#PAG-03` 0.57  
> anclaje: OK (cobertura 1.0)

### A.2 — Producto personalizado: no devolvible

**Cliente:** «Compré una agenda con mi nombre grabado y ya no la quiero, ¿la puedo devolver?»

*Qué se espera:* Sin RAG, lo usual es prometer 30 días de devolución. Con RAG, la ficha del producto (`NO_DEV_PERSONALIZADO`) y la sección 3 de la política dicen que no.

**Sin RAG**

> ¡Hola de nuevo! Me alegra que hayas contactado con nosotros. Entiendo que la agenda que compraste no es para ti, y quieres devolverla.
>
> En EcoMarket, nos esforzamos por ofrecer productos de alta calidad y atención al cliente excepcional. Sin embargo, debido a la naturaleza personalizada de la agenda (con tu nombre grabado), no podemos aceptar devoluciones o reembolsos completos.
>
> Sin embargo, podemos ofrecerte una solución alternativa. ¿Te gustaría que te ayudemos a encontrar una solución para que puedas utilizar la agenda de otra manera? Por ejemplo, podríamos sugerirte algunas ideas creativas para utilizar la agenda como un cuaderno o un diario, o incluso podríamos ofrecerte un reembolso parcial o un descuento en tu próxima compra.
>
> ¿Te gustaría explorar alguna de estas opciones? Estoy aquí para ayudarte.

**Con RAG**

> Tú estás interesado en devolver tu Agenda de papel reciclado con nombre grabado. Entiendo que quieres devolverlo debido a que ya no lo quieres.
>
> Primero, quiero informarte que nuestra política de devolución establece que los productos personalizados no son devolvibles. Esto se debe a que cada uno de nuestros productos es único y personalizado según las necesidades del cliente. [politica#s1-p0]
>
> Sin embargo, si el producto llegó defectuoso, vencido, averiado, incompleto o distinto al comprado, tienes derecho a reposición o reembolso total. ¿Podrías confirmar si el producto tiene algún problema o si simplemente lo quieres devolver?

> intención: `DEVOLUCION`  
> llamadas al LLM: 2  
> fragmentos (similitud): `catalogo#ECO-AGE-PER` 0.67, `faq#CTA-01` 0.57, `politica#s1-p0` 0.51, `politica#s8-p0` 0.50, `politica#s3-p0` 0.50 *(expansión)*  
> anclaje: OK (cobertura 1.0)

### A.3 — Programa propio de la empresa

**Cliente:** «¿Tienen algún programa para reciclar los frascos vacíos?»

*Qué se espera:* Un modelo general no puede conocer el programa Vuelve a Casa ni su cupón.

**Sin RAG**

> ¡Hola! Gracias por contactarnos. En EcoMarket, nos esforzamos por reducir nuestro impacto ambiental y promover prácticas sostenibles.
>
> Sí, tenemos un programa de reciclaje de frascos vacíos. Estamos trabajando en colaboración con una empresa especializada en reciclaje de plásticos para ofrecer un servicio de reciclaje de frascos vacíos a nuestros clientes.
>
> Aquí te presento los detalles del programa:
>
> *   Puedes enviar tus frascos vacíos a nuestra dirección de envío indicada en la página de pago de tu pedido.
> *   Nuestro equipo de reciclaje se encargará de recopilar y procesar los frascos para reciclarlos.
> *   En cada envío, recibirás un código de reciclaje que te permitirá obtener un descuento en tu próximo pedido.
>
> Además, también ofrecemos un pequeño incentivo para cada frasco reciclado: 10% de descuento en tu próximo pedido.
>
> Si tienes alguna pregunta o necesitas más información, no dudes en preguntar. Estamos aquí para ayudarte.
>
> ¡Gracias por ser parte de nuestro programa de reciclaje!

**Con RAG**

> "Entendido, cliente. Sí, tenemos un programa de reciclaje de envases llamado Vuelve a Casa. Por cada 5 envases vacíos, recibes un cupón de COP 5.000 para tu próxima compra. [faq#SOS-03]. Además, todos nuestros empaques están diseñados con sostenibilidad en mente, utilizando cartón reciclado, papel kraft y cinta de fibra natural. [faq#SOS-01]. ¿Quieres saber más sobre cómo participar en este programa?"

> intención: `PRODUCTO`  
> llamadas al LLM: 2  
> fragmentos (similitud): `faq#SOS-03` 0.69, `faq#SOS-01` 0.53, `catalogo#ECO-BOL-TEL` 0.50, `catalogo#ECO-COM-13L` 0.50  
> anclaje: OK (cobertura 1.0)

---

## B. Preguntas que el Taller 1 no podía responder

Pagos, cuenta y catálogo ampliado: ahora están en la base de conocimiento.

### B.1 — Facturación a nombre de empresa (intención PAGO_CUENTA, nueva)

**Cliente:** «Necesito que la factura salga a nombre de mi empresa, ¿cómo hago?»

**Asistente**

> "Entiendo que necesitas que la factura salga a nombre de tu empresa. Según nuestra política de pago, debes ingresar el NIT y la razón social en el checkout antes de pagar. Esto se indica en la pregunta frecuente [faq#PAG-04]. 
>
> Además, debes registrarte con el mismo correo que usaste en la compra para poder solicitar devoluciones desde Mi cuenta > Mis pedidos. Esto se menciona en la pregunta frecuente [faq#CTA-01]. 
>
> Si necesitas ayuda con el proceso o tienes alguna pregunta adicional, puedo ofrecerte conectar con un agente humano para asistirte. ¿Quieres que te ayude con eso?"

> intención: `PAGO_CUENTA`  
> llamadas al LLM: 2  
> fragmentos (similitud): `faq#PAG-04` 0.62, `faq#CTA-01` 0.54, `faq#CTA-03` 0.51, `faq#CTA-04` 0.51  
> anclaje: OK (cobertura 1.0)

### B.2 — Catálogo con lenguaje coloquial

**Cliente:** «¿Qué tienen para llevar el tinto calientico a la oficina?»

**Asistente**

> Lo siento, pero no tengo información sobre el estado del envío de un pedido específico. Para obtener más información sobre tu pedido, te recomiendo consultar el número de seguimiento o contactar con nuestro equipo de atención al cliente. ¿Quieres que te ayude con algo más?

> intención: `OTRO`  
> llamadas al LLM: 2  
> fragmentos (similitud): `faq#ENV-01` 0.44, `catalogo#ECO-CUB-BAM` 0.43, `catalogo#ECO-JAB-SOL` 0.42, `catalogo#ECO-TER-CAF` 0.41  
> anclaje: OK (cobertura 1.0)

### B.3 — Derecho de supresión de datos (Ley 1581)

**Cliente:** «Quiero que borren toda mi información de su base de datos.»

**Asistente**

> "Entiendo que desea eliminar su información de nuestra base de datos. Según nuestra política de privacidad, puede solicitar la supresión de sus datos personales desde Mi cuenta > Mis datos o escribiendo a datos@ecomarket.co. La solicitud se responde en un máximo de 15 días hábiles. [faq#CTA-03] 
>
> Si desea dejar de recibir correos promocionales, puede utilizar el enlace Cancelar suscripción al final de cualquier correo promocional o desactivar la opción en Mi cuenta > Preferencias. [faq#CTA-04] 
>
> Sin embargo, no tengo información sobre cómo cancelar un pedido o solicitar un reembolso. Si necesita ayuda con eso, puedo conectarlo con un agente humano para asistirle. ¿Le gustaría que lo haga?"

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

> intención: `PRODUCTO`  
> llamadas al LLM: 1  
> **fuera de alcance** (ningún fragmento supera el umbral)  
> fragmentos (similitud): `faq#ENV-02` 0.25, `catalogo#ECO-AGE-PER` 0.25, `catalogo#ECO-KIT-HUE` 0.25, `faq#ATN-02` 0.24

### C.2 — Cerca del dominio: pasa el umbral

**Cliente:** «¿Tienen vacantes de trabajo en EcoMarket?»

*Qué se espera:* Menciona a EcoMarket, así que el retrieval encuentra fragmentos parecidos. La segunda barrera es el prompt estricto: el contexto no habla de empleo.

**Asistente**

> Tú estás buscando información sobre vacantes de trabajo en EcoMarket. [id_fragmento: atn-03]. Sin embargo, según nuestra política de no tener tiendas físicas, no tenemos vacantes de trabajo en tiendas físicas. [id_fragmento: env-06]. Además, dado que solo realizamos envíos dentro de Colombia continental y San Andrés, no tenemos vacantes de trabajo relacionadas con envíos internacionales. [id_fragmento: atn-02]. Si tienes alguna otra pregunta o necesitas ayuda con algo más, estaré encantado de ayudarte. ¿Podrías decirme en qué puedo asistirte hoy?

> intención: `OTRO`  
> llamadas al LLM: 2  
> fragmentos (similitud): `faq#ATN-03` 0.66, `faq#ENV-06` 0.62, `faq#ATN-02` 0.58, `faq#SOS-03` 0.57  
> anclaje: OK (cobertura 1.0)

### C.3 — Producto que EcoMarket no vende

**Cliente:** «Se me dañó la lavadora, ¿cómo la arreglo?»

*Qué se espera:* Se parece a la sección de garantías de electrodomésticos, pero no es un producto de EcoMarket.

**Asistente**

> Hola, gracias por contactarnos. Entiendo que la lavadora se dañó y deseas devolverla. Según nuestra política de devoluciones, los productos de higiene con sello intacto no se aceptan, pero en este caso, la compostera doméstica 13 L pertenece a la categoría de huerta y jardín, que tiene una política de devolución de 30 días desde la entrega.
>
> ¿Está dentro del plazo de 30 días desde la entrega? [politica#s8-p0] [catalogo#ECO-COM-13L] La fecha de entrega real es [fecha_entrega_real]. Si es así, necesitas seguir los pasos del procedimiento de devolución: 1) Ingresar a Mi cuenta > Mis pedidos y seleccionar el pedido con su número de seguimiento, 2) Elegir "Solicitar devolución" e indicar el motivo, 3) Empacar el producto en su empaque original y entregarlo en cualquier punto de la transportadora indicada, dentro de los 5 días hábiles siguientes a la generación de la guía.
>
> ¿Quieres seguir con el procedimiento de devolución?

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

> Hola, gracias por contactarnos. Para tu pedido ECO-10003, encontramos que se encuentra en estado de retrasado por una novedad logística [motivo_retraso#Cierre temporal de la via por condiciones climaticas en la ruta fluvial de ultima milla]. La transportadora es Coordinadora y la ciudad de destino es Leticia. La fecha estimada de entrega ahora es el 2026-10-10, lo que significa que se ha retrasado aproximadamente 6 días desde la fecha original. Si tienes alguna pregunta o necesitas ayuda con algo más, no dudes en preguntar.

> intención: `ESTADO_PEDIDO`  
> llamadas al LLM: 3  
> herramienta invocada por el modelo: `consultar_pedido`  
> anclaje: REVISAR (cobertura 0.5)

### D.2 — Producto excluido PERO defectuoso (RAG + expansión por código)

**Cliente:** «El shampoo del pedido ECO-10009 me llegó partido y con moho, quiero que me lo repongan.»

**Asistente**

> Tú estás pasando por un problema con tu pedido ECO-10009. Entiendo que el shampoo llegó partido y con moho. 
>
> Primero, quiero aclararte que, aunque el shampoo pertenece a una categoría con condiciones especiales, tenemos una excepción para productos que llegan defectuosos. 
>
> Para proceder con la devolución, necesito confirmar si el sello de seguridad sigue intacto. ¿Puedes confirmarme eso para mí? [politica#s4-p0]

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

**top_k = 1 — fragmentos: faq#ENV-01 — tokens 948 / 130**

> "Entendido, cliente. Para saber el costo del envío a Bogotá, necesitaría verificar la política de envío de EcoMarket. Según la información disponible, el costo del envío a ciudades principales como Bogotá varía entre 2 y 4 días hábiles. [id_fragmento: envios#PAG-02]
>
> Para obtener un costo más preciso, te recomiendo consultar nuestra página de envíos o contactar con nuestro equipo de atención al cliente. ¿Quieres que te ayude a encontrar la información necesaria o prefieres hablar con un agente humano?"

**top_k = 4 — fragmentos: faq#ENV-01, faq#ENV-02, faq#ENV-04, politica#s6-p0 — tokens 1329 / 318**

> "Entiendo que estás buscando información sobre el envío a Bogotá. ¿Podrías decirme cuánto cuesta el envío a Bogotá y en cuántos días me llega?"
>
> *   El envío es gratis en compras superiores a COP 150.000.
> *   Para compras inferiores a ese valor, se cobra una tarifa plana de COP 14.000 para ciudades principales e intermedias y de COP 28.000 para zonas de difícil acceso. [faq#ENV-02]
> *   El tiempo de entrega varía según la ciudad, pero en general, se cuenta en días hábiles desde el despacho. [faq#ENV-01]
> *   La preparación del pedido en bodega toma entre 1 y 2 días hábiles adicionales. [faq#ENV-01]
> *   Si no estás cuando llega el domicilio, la transportadora hace hasta tres intentos de entrega en días distintos. [faq#ENV-04]
> *   Una vez el producto llega a la bodega de EcoMarket y pasa la inspección de calidad, el tiempo de reembolso varía según el método de pago: entre 5 y 15 días hábiles para tarjeta de crédito, entre 3 y 8 días hábiles para PSE o transferencia bancaria, y inmediato para saldo a favor en EcoMarket. [politica#s6-p0]

### E.2 — Prompt estricto vs. permisivo (una sola instrucción cambia)

**Cliente:** «¿Cuántos años dura la batería de la lámpara solar antes de tener que cambiarla?»

La ficha de `ECO-LAM-SOL` dice "hasta 8 h de luz" pero no habla de la vida útil de la batería.

**Estricto (`RAG_GENERAL`)**

> Reconocemos su consulta y nos gustaría ayudarle.
>
> Para determinar la respuesta a su pregunta, necesitamos verificar la información del producto específico que está interesado en conocer. Sin embargo, el contexto proporcionado solo incluye información general sobre los productos de la tienda y no especifica el producto que está buscando.
>
> Para obtener la información precisa sobre la batería de la lámpara solar, le recomendamos conectar con un agente humano que pueda ayudarle a encontrar la información específica que necesita. ¿Le gustaría hablar con uno de nuestros agentes?

**Permisivo (`RAG_PERMISIVO`)**

> "¡Hola! Entiendo que estás buscando información sobre la duración de la batería de la lámpara solar de EcoMarket. Desafortunadamente, no encontré ninguna información específica sobre la duración de la batería en el contexto proporcionado. Sin embargo, puedo decirte que la lámpara solar recargable para exterior de EcoMarket tiene una batería de litio recargable que se estima tiene una vida útil de hasta 8 horas de luz.
>
> Si deseas obtener más información sobre la duración de la batería o cualquier otra característica de la lámpara solar, te recomiendo consultar el folleto de producto o contactar con nuestro equipo de atención al cliente. ¿Necesitas ayuda con algo más?"

### E.3 — Umbral de relevancia activado vs. desactivado

**Cliente:** «¿Cuál es la capital de Australia?»

**Umbral activado — similitud máxima 0.25 (umbral 0.41) — tokens 0 / 0**

> Gracias por escribirnos. No encontré información sobre eso en las fuentes de EcoMarket, así que prefiero no darte una respuesta que podría ser incorrecta. Puedo ayudarte con pedidos, envíos, devoluciones, pagos, tu cuenta y nuestros productos. Si lo prefieres, te conecto con una persona del equipo (lunes a viernes, 8:00 a 18:00). ¿Quieres que lo haga?

**Umbral desactivado + prompt estricto — similitud máxima 0.25 (umbral 0.41) — tokens 1204 / 67**

> Lo siento, pero no puedo proporcionar información sobre la capital de Australia. El contexto proporcionado solo contiene información sobre envíos, productos y políticas de devolución de EcoMarket, y no parece tener relación con esta pregunta. ¿Podría conectarme con un agente humano para ayudar con esa información?

**Umbral desactivado + prompt permisivo — similitud máxima 0.25 (umbral 0.41) — tokens 1164 / 55**

> Reconocemos tu necesidad de ayuda, pero lamentablemente, no tenemos información sobre la capital de Australia. No podemos proporcionar una respuesta precisa a tu pregunta. ¿Podrías considerar conectarte con nuestro equipo humano para obtener asistencia adicional?

### E.4 — Cambiar el modelo de embeddings (solo recuperación)

Mismo índice, mismos fragmentos; cambia el modelo que los vectoriza. Se muestran los 3 primeros fragmentos de cada uno.

| Consulta | TF-IDF (Taller 1) | bge-m3 |
|---|---|---|
| «¿A qué horas puedo hablar con una persona de verdad?» | `catalogo#ECO-CEP-BAM` 0.12<br>`faq#PAG-05` 0.12<br>`faq#PAG-04` 0.12 | `faq#ATN-01` 0.59<br>`politica#s9-p0` 0.54<br>`faq#PAG-04` 0.46 |
| «¿Pueden mandarle un pedido a mi hermana que vive en Miami?» | `faq#ENV-05` 0.18<br>`faq#PAG-02` 0.15<br>`faq#CTA-01` 0.13 | `faq#ENV-05` 0.50<br>`faq#CTA-01` 0.48<br>`faq#ENV-01` 0.47 |

TF-IDF compara pedazos de palabras; bge-m3 compara significado. La consulta de Miami es un fallo conocido de ambos: la FAQ dice "envíos internacionales" y ningún modelo conecta "Miami" con "internacional" con suficiente fuerza (ver Fase 3, limitaciones).

---

## Consumo de la ejecución

- Tokens de entrada: **26,796** · de salida: **3,012**
- Costo estimado: **USD 0.0000** (precios de referencia en `src/llm.py`)
- Duración: 45 s
