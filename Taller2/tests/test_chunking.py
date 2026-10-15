"""Pruebas del chunking: cada estrategia respeta la unidad semántica que promete."""

import re

import pytest

from src.cargadores import cargar_politica_pdf
from src.chunking import ESTRATEGIAS, construir_chunks, partir_politica

ENCABEZADO_SECCION = re.compile(r"^\d+\. [^\n]+$", re.MULTILINE)


def _secciones_en(texto: str) -> set[str]:
    # Un título de sección es corto y no termina en punto; los pasos numerados
    # del procedimiento son oraciones largas.
    return {
        m.group(0) for m in ENCABEZADO_SECCION.finditer(texto)
        if not m.group(0).endswith(".") and len(m.group(0)) < 60
    }


def test_politica_se_parte_por_secciones_numeradas():
    chunks = partir_politica(cargar_politica_pdf())
    secciones = {c.metadata["seccion"] for c in chunks}
    numeradas = {s for s in secciones if s[0].isdigit()}
    assert len(numeradas) == 9
    assert "3. Categorías NO sujetas a devolución" in numeradas
    # El bloque inicial (título, versión, vigencia) se conserva como fragmento propio.
    assert "Encabezado" in secciones


def test_ningun_chunk_de_politica_mezcla_dos_secciones():
    for chunk in partir_politica(cargar_politica_pdf()):
        cuerpo = chunk.page_content.split("\n", 1)[1]  # sin el encabezado contextual
        assert len(_secciones_en(cuerpo)) <= 1, chunk.metadata["id"]


def test_los_pasos_numerados_no_se_confunden_con_secciones():
    chunks = partir_politica(cargar_politica_pdf())
    paso = next(c for c in chunks if "Ingresar a Mi cuenta" in c.page_content)
    assert paso.metadata["seccion"].startswith("8.")


def test_la_excepcion_queda_junto_a_la_tabla_de_exclusiones():
    chunks = partir_politica(cargar_politica_pdf())
    excepcion = next(c for c in chunks if "Excepción obligatoria" in c.page_content)
    assert "NO_DEV_PERECEDERO" in excepcion.page_content


def test_cada_chunk_lleva_el_encabezado_de_su_seccion():
    for chunk in partir_politica(cargar_politica_pdf()):
        assert chunk.page_content.startswith("Política de devoluciones de EcoMarket > ")


def test_estrategia_estructural_un_chunk_por_producto_y_por_pregunta():
    chunks = construir_chunks(estrategia="estructural")
    productos = [c for c in chunks if c.metadata["tipo"] == "producto"]
    faq = [c for c in chunks if c.metadata["tipo"] == "faq"]
    assert len(productos) == 22
    assert len(faq) == 23


def test_ids_son_unicos_y_estables():
    primera = [c.metadata["id"] for c in construir_chunks()]
    segunda = [c.metadata["id"] for c in construir_chunks()]
    assert primera == segunda
    assert len(set(primera)) == len(primera)
    assert "catalogo#ECO-BOT-500" in primera
    assert "faq#PAG-02" in primera


def test_estrategia_fijo_corta_a_tamano_constante():
    chunks = construir_chunks(estrategia="fijo")
    assert all(len(c.page_content) <= 500 for c in chunks)


@pytest.mark.parametrize("estrategia", ESTRATEGIAS)
def test_todas_las_estrategias_conservan_la_fuente(estrategia):
    fuentes = {c.metadata["fuente"] for c in construir_chunks(estrategia=estrategia)}
    assert fuentes == {"politicas_devolucion.pdf", "catalogo_productos.csv", "faq.json"}


def test_estrategia_desconocida_falla():
    with pytest.raises(ValueError):
        construir_chunks(estrategia="magica")
