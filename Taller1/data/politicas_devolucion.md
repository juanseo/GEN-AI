# Política de Devoluciones, Cambios y Reembolsos — EcoMarket

> Documento fuente autorizado (*grounding data*) para el asistente de atención al cliente.
> Versión 3.2 — Vigente desde: 2026-08-01 — Responsable: Dirección de Experiencia al Cliente.
> Documento ficticio, creado exclusivamente para el Taller 1 de la asignatura.

---

## 1. Derecho de retracto (regla general)

El cliente puede solicitar la devolución de un producto dentro de los **30 días calendario**
siguientes a la fecha de entrega registrada por la transportadora, siempre que el producto
cumpla las condiciones del apartado 2.

Para compras realizadas por canales digitales aplica adicionalmente el **derecho de retracto de
5 días hábiles** contado desde la entrega, conforme al Estatuto del Consumidor colombiano
(Ley 1480 de 2011, art. 47). Durante ese plazo el reembolso es total, incluyendo el valor del
envío original.

---

## 2. Condiciones que debe cumplir el producto

Para que una devolución sea aprobada, el producto debe:

- Estar **sin uso**, salvo la manipulación razonable necesaria para verificarlo.
- Conservar **empaque original**, etiquetas, manuales y accesorios.
- No presentar daños atribuibles al uso indebido por parte del cliente.
- Estar asociado a un pedido en estado `ENTREGADO`. Los pedidos en estado
  `EN_PREPARACION`, `EN_TRANSITO` o `PAGO_PENDIENTE` **no se devuelven: se cancelan**
  (ver apartado 7).

---

## 3. Categorías NO sujetas a devolución

Por razones sanitarias, de inocuidad alimentaria o de seguridad del consumidor, **no se acepta
devolución ni cambio** de los siguientes grupos, aun dentro de los 30 días:

| Categoría | Código interno | Motivo de la exclusión |
|---|---|---|
| Alimentos perecederos y frescos | `NO_DEV_PERECEDERO` | Cadena de frío e inocuidad alimentaria |
| Productos de higiene personal y cuidado íntimo | `NO_DEV_HIGIENE` | Riesgo sanitario una vez abierto el sello |
| Cosméticos y cremas con sello de seguridad roto | `NO_DEV_HIGIENE` | Imposibilidad de re-comercialización sanitaria |
| Productos personalizados o hechos a la medida | `NO_DEV_PERSONALIZADO` | Fabricación exclusiva para el cliente |
| Tarjetas de regalo y bonos digitales | `NO_DEV_DIGITAL` | Bien digital de consumo inmediato |
| Semillas vivas, plantas y material vegetal | `NO_DEV_VIVO` | Material biológico vivo, normativa fitosanitaria |

**Excepción obligatoria:** si un producto de estas categorías llega **defectuoso, vencido,
averiado, incompleto o distinto al comprado**, el cliente **sí tiene derecho** a reposición o
reembolso total. Esta excepción prevalece sobre cualquier exclusión de la tabla anterior y se
gestiona por el flujo de *garantía*, no por el de *retracto*.

---

## 4. Categorías con condiciones especiales

- **Productos de higiene con sello intacto** (`DEV_CONDICIONAL_SELLO`): se aceptan únicamente si
  el sello de seguridad **no ha sido roto**. El agente debe confirmarlo con el cliente antes de
  generar la guía de retorno.
- **Textiles y prendas** (`DEV_TEXTIL`): se aceptan cambios de talla sin costo **una sola vez**
  por pedido, dentro de los 30 días.
- **Electrodomésticos y equipos** (`DEV_GARANTIA_12M`): 30 días por retracto; después aplica
  garantía legal de 12 meses por defectos de fabricación.

---

## 5. Costos del envío de retorno

| Causa de la devolución | ¿Quién paga el retorno? |
|---|---|
| Producto defectuoso, averiado o equivocado | EcoMarket (100%) |
| Retracto dentro de los 5 días hábiles | EcoMarket (100%) |
| Cambio de talla en textiles (primera vez) | EcoMarket (100%) |
| Arrepentimiento entre el día 6 y el día 30 | Cliente (tarifa plana de COP 12.000) |
| Segundo cambio o posteriores del mismo pedido | Cliente (tarifa plana de COP 12.000) |

---

## 6. Tiempos de reembolso

Una vez el producto llega a la bodega de EcoMarket y pasa la inspección de calidad:

- **Tarjeta de crédito:** entre 5 y 15 días hábiles, sujeto a los tiempos del banco emisor.
- **PSE / transferencia bancaria:** entre 3 y 8 días hábiles.
- **Saldo a favor en EcoMarket:** inmediato, una vez aprobada la inspección.

La inspección de calidad en bodega toma hasta **3 días hábiles** desde la recepción.

---

## 7. Cancelaciones antes del despacho

Un pedido en estado `EN_PREPARACION` o `PAGO_PENDIENTE` puede cancelarse sin costo desde la
cuenta del cliente o solicitándolo al asistente. Si el pedido ya está `EN_TRANSITO`, no puede
cancelarse: el cliente debe rechazar la entrega o iniciar una devolución estándar una vez
recibido.

---

## 8. Procedimiento de devolución (pasos para el cliente)

1. Ingresar a **Mi cuenta → Mis pedidos** y seleccionar el pedido con su número de seguimiento.
2. Elegir **"Solicitar devolución"** e indicar el motivo.
3. El sistema valida automáticamente la elegibilidad según este documento.
4. Si es elegible, se genera una **guía de retorno prepagada o con cobro**, según el apartado 5.
5. Empacar el producto en su empaque original y entregarlo en cualquier punto de la
   transportadora indicada, dentro de los **5 días hábiles** siguientes a la generación de la guía.
6. Al recibir el producto, EcoMarket realiza la inspección y notifica el resultado por correo.

---

## 9. Escalamiento a un agente humano

El asistente **debe transferir la conversación a un agente humano** cuando:

- El cliente manifieste inconformidad, reclamo formal, queja ante la SIC o mención de acciones legales.
- Exista un cobro duplicado, un fraude presunto o una disputa sobre el valor pagado.
- El producto haya causado un **daño a la salud o a la propiedad** del cliente.
- El caso no esté cubierto por este documento o la información recuperada sea contradictoria.
- El cliente lo solicite explícitamente.
- Se detecte angustia emocional significativa en el mensaje del cliente.

Tiempo objetivo de respuesta del agente humano tras el escalamiento: **2 horas hábiles**.
