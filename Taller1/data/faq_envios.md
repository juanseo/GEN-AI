# Preguntas frecuentes — Envíos y entregas — EcoMarket

> Documento fuente autorizado (*grounding data*). Versión 2026-09. Datos ficticios para el Taller 1.

## ¿Cuánto tarda un envío?

| Destino | Tiempo estimado (días hábiles) |
|---|---|
| Cali y área metropolitana | 1 a 2 |
| Ciudades principales (Bogotá, Medellín, Barranquilla, Cartagena, Bucaramanga, Pereira, Manizales) | 2 a 4 |
| Ciudades intermedias | 4 a 6 |
| Zonas de difícil acceso (Amazonas, Chocó, Guainía, Vaupés, Vichada, San Andrés) | 8 a 15 |

Los tiempos se cuentan desde el **despacho**, no desde la compra. La preparación del pedido
toma entre 1 y 2 días hábiles adicionales.

## ¿Cuánto cuesta el envío?

- Envío **gratis** en compras superiores a COP 150.000.
- Tarifa plana de COP 14.000 para ciudades principales e intermedias.
- Tarifa de COP 28.000 para zonas de difícil acceso.

## ¿Qué significa cada estado del pedido?

- `PAGO_PENDIENTE`: la transacción aún no ha sido confirmada por la pasarela de pagos.
- `EN_PREPARACION`: el pedido está siendo alistado en bodega. Todavía se puede cancelar sin costo.
- `EN_TRANSITO`: el pedido fue entregado a la transportadora y está en ruta.
- `RETRASADO`: la entrega superó la fecha estimada original por una novedad logística.
- `ENTREGADO`: la transportadora confirmó la entrega en la dirección registrada.
- `DEVOLUCION_EN_CURSO`: hay una solicitud de devolución o cambio activa sobre el pedido.
- `CANCELADO`: el pedido fue anulado antes del despacho.

## ¿Qué pasa si no estoy cuando llega el domicilio?

La transportadora realiza hasta **tres intentos de entrega** en días distintos. Después del
tercer intento fallido, el pedido regresa a la bodega de EcoMarket y se contacta al cliente para
coordinar un nuevo envío (con costo) o el reembolso.

## ¿Puedo cambiar la dirección de entrega?

Solo mientras el pedido esté en estado `EN_PREPARACION`. Una vez `EN_TRANSITO`, el cambio de
dirección depende de la transportadora y debe gestionarlo un agente humano.

## ¿Hacen envíos internacionales?

No. EcoMarket opera únicamente dentro de Colombia continental y San Andrés.

## ¿Los empaques son sostenibles?

Sí. Todos los pedidos se despachan en cajas de cartón reciclado FSC, con relleno de papel kraft
y cinta de fibra natural. No se usa plástico de un solo uso en ningún punto del empaque.
