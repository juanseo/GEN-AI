"""Pruebas de la lógica de decisión del RAG que no requieren modelo ni red."""

from langchain_core.documents import Document

from src.rag import Recuperado, formatear_contexto, supera_umbral


def _recuperado(identificador: str, score: float) -> Recuperado:
    documento = Document(
        page_content="Pregunta frecuente (pagos): ¿Puedo pagar contra entrega?\nRespuesta: No.",
        metadata={"id": identificador, "fuente": "faq.json", "categoria": "pagos"},
    )
    return Recuperado(documento, score)


def test_supera_umbral_usa_el_mejor_fragmento():
    recuperados = [_recuperado("faq#A", 0.30), _recuperado("faq#B", 0.62)]
    assert supera_umbral(recuperados, umbral=0.45)


def test_no_supera_umbral_si_ningun_fragmento_se_parece():
    recuperados = [_recuperado("faq#A", 0.30), _recuperado("faq#B", 0.25)]
    assert not supera_umbral(recuperados, umbral=0.45)


def test_sin_resultados_no_hay_contexto():
    assert not supera_umbral([], umbral=0.0)


def test_el_contexto_incluye_id_citable_y_fuente():
    contexto = formatear_contexto([_recuperado("faq#PAG-02", 0.6)])
    assert contexto.startswith("[faq#PAG-02] Fuente: faq.json > pagos")
    assert "contra entrega" in contexto
