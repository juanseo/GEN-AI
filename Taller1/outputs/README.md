# Salidas de ejecución

Este directorio recibe las transcripciones que generan los scripts.

| Archivo | Lo genera | Contiene |
|---|---|---|
| `ejecucion_fase3.md` | `python -m scripts.demo_fase3` | Los 8 casos de la Fase 3 ejecutados con el prompt básico y con el optimizado, con la intención detectada, los fragmentos recuperados, la verificación de anclaje y el costo de la corrida |

Para generarlo:

```bash
cd Taller1
pip install -r requirements.txt
cp .env.example .env     # y completa OLLAMA_API_KEY
python -m scripts.demo_fase3
```

La corrida completa hace ~24 llamadas a `glm-5.3-flash` en Ollama Cloud y tarda
alrededor de un minuto. Ollama Cloud cobra por suscripción, no por token, así
que el costo aparece como 0; con `ECOMARKET_PROVEEDOR=openai` (`gpt-4o-mini`)
cuesta del orden de **USD 0,01** y el script imprime el costo exacto al terminar.

> Si solo quieres inspeccionar la capa de RAG sin consumir tokens ni configurar
> la API key, usa `python -m scripts.demo_retrieval`.
