"""Pruebas de los cargadores: cada formato produce documentos limpios y con metadata."""

import csv

from src.cargadores import (
    cargar_catalogo_csv,
    cargar_documentos,
    cargar_faq_json,
    cargar_politica_pdf,
)
from src.config import DATA_DIR


def test_pdf_elimina_encabezado_y_pie_de_pagina():
    texto = cargar_politica_pdf().page_content
    assert "Documento controlado" not in texto
    assert "Página 2" not in texto
    assert "EcoMarket S.A.S. - Política" not in texto


def test_pdf_une_las_paginas_en_un_solo_documento():
    politica = cargar_politica_pdf()
    assert politica.metadata["paginas"] == 3
    assert politica.metadata["fuente"] == "politicas_devolucion.pdf"
    # La sección 3 empieza en la página 1 y su excepción queda en la página 2.
    inicio = politica.page_content.index("3. Categorías NO sujetas")
    assert politica.page_content.index("Excepción obligatoria") > inicio


def test_catalogo_produce_un_documento_por_producto():
    with open(DATA_DIR / "catalogo_productos.csv", encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    productos = cargar_catalogo_csv()
    assert len(productos) == len(filas)
    botella = next(p for p in productos if p.metadata["sku"] == "ECO-BOT-500")
    assert "COP 89.000" in botella.page_content
    assert "devolución hasta 30 días" in botella.page_content
    assert botella.metadata["politica"] == "DEV_ESTANDAR"


def test_faq_produce_un_documento_por_pregunta():
    faq = cargar_faq_json()
    ids = [d.metadata["id"] for d in faq]
    assert len(faq) == 23
    assert len(set(ids)) == len(ids)
    contra_entrega = next(d for d in faq if d.metadata["id"] == "PAG-02")
    assert contra_entrega.page_content.startswith("Pregunta frecuente")
    assert "contra entrega" in contra_entrega.page_content


def test_cargar_documentos_reune_los_tres_tipos():
    tipos = {d.metadata["tipo"] for d in cargar_documentos()}
    assert tipos == {"politica", "producto", "faq"}
