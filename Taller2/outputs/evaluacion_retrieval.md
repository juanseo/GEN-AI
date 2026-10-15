# Evaluación del retrieval: modelos de embeddings y estrategias de chunking

> Generado por `python -m scripts.evaluar_retrieval` sobre `data/evaluacion/golden_set.json` (26 consultas en alcance, 8 fuera de alcance).  
> Fecha: 2026-10-15 09:34 · CPU, sin GPU.

## 1. Modelos de embeddings (chunking estructural)

| Modelo | Licencia | hit@1 | hit@4 | MRR | Latencia por consulta (mediana) | Indexar 55 fragmentos |
|---|---|---|---|---|---|---|
| tfidf | scikit-learn | 58% | 88% | 0.713 | 1 ms | 0.1 s |
| paraphrase-multilingual | Apache 2.0 (open-source, sentence-transformers) | 62% | 85% | 0.732 | 28 ms | 1.0 s |
| bge-m3 | MIT (open-source, BAAI) | 88% | 92% | 0.894 | 32 ms | 1.2 s |

## 2. Detección de consultas fuera de alcance

Similitud coseno del **mejor** fragmento recuperado. Si el modelo separa bien, las consultas en alcance quedan por encima y las de fuera por debajo, y existe un umbral que las distingue.

| Modelo | En alcance (mín – máx) | Fuera (mín – máx) | Margen | Umbral | En alcance aceptadas | Fuera de alcance atajadas |
|---|---|---|---|---|---|---|
| tfidf | 0.119 – 0.384 | 0.097 – 0.202 | -0.084 | 0.06 | 100% | 0% |
| paraphrase-multilingual | 0.398 – 0.786 | 0.137 – 0.634 | -0.236 | 0.34 | 100% | 50% |
| bge-m3 | 0.462 – 0.684 | 0.255 – 0.651 | -0.189 | 0.41 | 100% | 38% |

Margen negativo = hay consultas fuera de alcance más parecidas a la base que algunas legítimas: ningún umbral las separa todas. El umbral se fija para no rechazar ninguna pregunta legítima; las que se cuelan las ataja el prompt estricto (ver la sección C de la demo de la Fase 3, `outputs/ejecucion_taller2*.md`).

## 3. Estrategias de chunking (modelo bge-m3)

| Estrategia | Fragmentos | hit@1 | hit@4 | MRR |
|---|---|---|---|---|
| estructural | 55 | 88% | 92% | 0.894 |
| parrafos | 58 | 38% | 62% | 0.505 |
| fijo | 42 | 58% | 88% | 0.704 |

## 4. Consultas que no quedaron en la posición 1

**tfidf / estructural**

- Q01 «me arrepentí de comprar la botella, ¿hasta cuándo la puedo devolver?» → posición 2; top-1: `faq#CTA-01`
- Q02 «ya no quiero la miel que compré, ¿me la reciben de vuelta?» → posición 2; top-1: `faq#SOS-03`
- Q03 «el shampoo me llegó roto y con hongos, ¿qué opciones tengo?» → posición 6; top-1: `catalogo#ECO-SHA-BAR`
- Q05 «si pagué por PSE, ¿en cuánto tiempo me devuelven la plata?» → posición 2; top-1: `faq#PAG-01`
- Q09 «¿me cobran el domicilio o es gratis?» → posición 2; top-1: `faq#ENV-04`
- Q10 «¿pueden mandarle un pedido a mi hermana que vive en Miami?» → posición >10; top-1: `faq#ENV-05`
- Q15 «¿tienen algo para llevar el tinto calientico a la oficina?» → posición 5; top-1: `faq#SOS-03`
- Q17 «¿de qué material son los cepillos de dientes que venden?» → posición 4; top-1: `catalogo#ECO-PIT-ACE`
- Q21 «¿a qué horas puedo hablar con una persona de verdad?» → posición 4; top-1: `catalogo#ECO-CEP-BAM`
- Q22 «¿tienen algún local para ir a mirar los productos?» → posición 3; top-1: `faq#SOS-02`
- Q23 «en la app mi pedido dice EN_TRANSITO, ¿eso qué quiere decir?» → posición 3; top-1: `faq#ENV-05`

**paraphrase-multilingual / estructural**

- Q03 «el shampoo me llegó roto y con hongos, ¿qué opciones tengo?» → posición >10; top-1: `catalogo#ECO-SHA-BAR`
- Q04 «¿quién paga el envío si devuelvo algo porque simplemente no me gustó?» → posición >10; top-1: `faq#ENV-04`
- Q05 «si pagué por PSE, ¿en cuánto tiempo me devuelven la plata?» → posición 2; top-1: `faq#PAG-04`
- Q07 «mi compra todavía no ha salido de la bodega, ¿la puedo anular?» → posición 2; top-1: `faq#PAG-05`
- Q09 «¿me cobran el domicilio o es gratis?» → posición 2; top-1: `faq#ENV-04`
- Q10 «¿pueden mandarle un pedido a mi hermana que vive en Miami?» → posición 5; top-1: `faq#ENV-01`
- Q15 «¿tienen algo para llevar el tinto calientico a la oficina?» → posición 3; top-1: `faq#SOS-03`
- Q16 «¿qué pasa si la compostera se daña a los seis meses?» → posición 2; top-1: `faq#ENV-04`
- Q18 «¿venden matas o plantas para la casa?» → posición 2; top-1: `faq#ATN-03`
- Q26 «¿puedo devolver la agenda que mandé a grabar con mi nombre?» → posición >10; top-1: `faq#CTA-03`

**bge-m3 / estructural**

- Q03 «el shampoo me llegó roto y con hongos, ¿qué opciones tengo?» → posición >10; top-1: `catalogo#ECO-DES-SOL`
- Q10 «¿pueden mandarle un pedido a mi hermana que vive en Miami?» → posición >10; top-1: `faq#ENV-05`
- Q15 «¿tienen algo para llevar el tinto calientico a la oficina?» → posición 4; top-1: `catalogo#ECO-CUB-BAM`

**bge-m3 / parrafos**

- Q01 «me arrepentí de comprar la botella, ¿hasta cuándo la puedo devolver?» → posición 2; top-1: `parrafos:politicas_devolucion#009`
- Q03 «el shampoo me llegó roto y con hongos, ¿qué opciones tengo?» → posición >10; top-1: `parrafos:catalogo_productos#009`
- Q04 «¿quién paga el envío si devuelvo algo porque simplemente no me gustó?» → posición >10; top-1: `parrafos:faq#000`
- Q10 «¿pueden mandarle un pedido a mi hermana que vive en Miami?» → posición 4; top-1: `parrafos:faq#000`
- Q11 «¿le puedo pagar en efectivo al mensajero cuando me traiga el paquete?» → posición 6; top-1: `parrafos:faq#007`
- Q12 «necesito que la factura salga a nombre de mi empresa» → posición 5; top-1: `parrafos:faq#009`
- Q13 «¿puedo diferir la compra con mi tarjeta de crédito?» → posición >10; top-1: `parrafos:faq#008`
- Q14 «quiero que borren toda mi información de su base de datos» → posición 2; top-1: `parrafos:faq#014`
- Q15 «¿tienen algo para llevar el tinto calientico a la oficina?» → posición 7; top-1: `parrafos:faq#000`
- Q17 «¿de qué material son los cepillos de dientes que venden?» → posición 2; top-1: `parrafos:catalogo_productos#014`
- Q19 «¿qué hago con los frascos vacíos de lo que les compro?» → posición >10; top-1: `parrafos:faq#018`
- Q20 «¿mandan los pedidos envueltos en plástico?» → posición 4; top-1: `parrafos:faq#016`
- Q21 «¿a qué horas puedo hablar con una persona de verdad?» → posición 8; top-1: `parrafos:faq#020`
- Q23 «en la app mi pedido dice EN_TRANSITO, ¿eso qué quiere decir?» → posición >10; top-1: `parrafos:faq#003`
- Q25 «si no hay nadie en la casa cuando llegue el mensajero, ¿qué pasa?» → posición 2; top-1: `parrafos:faq#000`
- Q26 «¿puedo devolver la agenda que mandé a grabar con mi nombre?» → posición >10; top-1: `parrafos:faq#012`

**bge-m3 / fijo**

- Q02 «ya no quiero la miel que compré, ¿me la reciben de vuelta?» → posición 4; top-1: `fijo:catalogo_productos#006`
- Q03 «el shampoo me llegó roto y con hongos, ¿qué opciones tengo?» → posición 6; top-1: `fijo:catalogo_productos#002`
- Q11 «¿le puedo pagar en efectivo al mensajero cuando me traiga el paquete?» → posición 3; top-1: `fijo:faq#001`
- Q12 «necesito que la factura salga a nombre de mi empresa» → posición 3; top-1: `fijo:faq#006`
- Q13 «¿puedo diferir la compra con mi tarjeta de crédito?» → posición 3; top-1: `fijo:faq#007`
- Q14 «quiero que borren toda mi información de su base de datos» → posición 2; top-1: `fijo:faq#009`
- Q15 «¿tienen algo para llevar el tinto calientico a la oficina?» → posición 4; top-1: `fijo:faq#000`
- Q17 «¿de qué material son los cepillos de dientes que venden?» → posición 2; top-1: `fijo:catalogo_productos#010`
- Q21 «¿a qué horas puedo hablar con una persona de verdad?» → posición 2; top-1: `fijo:faq#013`
- Q23 «en la app mi pedido dice EN_TRANSITO, ¿eso qué quiere decir?» → posición 7; top-1: `fijo:faq#002`
- Q26 «¿puedo devolver la agenda que mandé a grabar con mi nombre?» → posición >10; top-1: `fijo:politicas_devolucion#008`

## 5. Recuperación del asistente (bge-m3 en ChromaDB)

- hit@4 con el índice HNSW de Chroma: **92%** (cálculo exacto de la sección 1: 92%; a esta escala HNSW no pierde vecinos).
- Con la expansión por código de política (rama de devoluciones): **96%**, con 4.4 fragmentos promedio en el contexto.
- Consultas rescatadas por la expansión: Q03.
