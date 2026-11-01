"""Pruebas del motor de reglas de devolución: la política v3.2 traducida a código.

Usan los pedidos reales de data/pedidos.json con la fecha de corte 2026-09-22 y
pedidos sintéticos para lo que esos datos no cubren (retracto, festivos, cambio
de talla, garantía de 12 meses). No requieren modelo ni red.
"""

from datetime import date

import pytest

from src.config import FECHA_REFERENCIA
from src.herramientas import PEDIDOS
from src.reglas_devolucion import COSTO_RETORNO_CLIENTE_COP, evaluar

HOY = date(2026, 9, 22)


def _pedido(sku: str, entrega: str | None, estado: str = "ENTREGADO") -> dict:
    return {
        "tracking": "ECO-90000",
        "estado": estado,
        "fecha_entrega_real": entrega,
        "items": [{"sku": sku, "nombre": "Producto de prueba", "cantidad": 1}],
    }


# --- Entradas inválidas -----------------------------------------------------

def test_pedido_inexistente_es_un_error():
    evaluacion = evaluar(None, "ECO-BOT-500", "ARREPENTIMIENTO", hoy=HOY)
    assert not evaluacion.ok
    assert evaluacion.codigo == "PEDIDO_NO_EXISTE"


def test_sku_que_no_esta_en_el_pedido_es_un_error():
    evaluacion = evaluar(PEDIDOS["ECO-10002"], "ECO-BOT-500", "ARREPENTIMIENTO", hoy=HOY)
    assert not evaluacion.ok
    assert evaluacion.codigo == "SKU_NO_PERTENECE"


def test_motivo_desconocido_es_un_error():
    evaluacion = evaluar(PEDIDOS["ECO-10002"], "ECO-CEP-BAM", "NO_ME_GUSTA_EL_COLOR", hoy=HOY)
    assert not evaluacion.ok
    assert evaluacion.codigo == "MOTIVO_INVALIDO"


def test_sku_en_minusculas_se_normaliza():
    evaluacion = evaluar(PEDIDOS["ECO-10002"], " eco-cep-bam ", "DEFECTUOSO", hoy=HOY)
    assert evaluacion.ok


# --- Regla 1: solo se devuelven pedidos entregados (§2, §7) -----------------

@pytest.mark.parametrize(
    ("tracking", "sku"),
    [
        ("ECO-10001", "ECO-BOT-500"),  # EN_TRANSITO
        ("ECO-10003", "ECO-JAB-SOL"),  # RETRASADO
        ("ECO-10004", "ECO-COM-13L"),  # EN_PREPARACION
        ("ECO-10006", "ECO-TER-CAF"),  # PAGO_PENDIENTE
        ("ECO-10007", "ECO-CAM-ALG"),  # DEVOLUCION_EN_CURSO
        ("ECO-10011", "ECO-COM-13L"),  # CANCELADO
    ],
)
def test_pedido_no_entregado_no_admite_devolucion(tracking, sku):
    evaluacion = evaluar(PEDIDOS[tracking], sku, "ARREPENTIMIENTO", hoy=HOY)
    assert evaluacion.ok
    assert evaluacion.resultado == "NO_ELEGIBLE"
    assert evaluacion.codigo == "ESTADO_NO_ADMITE"
    assert evaluacion.alternativa


def test_pedido_en_preparacion_se_cancela_en_lugar_de_devolverse():
    evaluacion = evaluar(PEDIDOS["ECO-10004"], "ECO-COM-13L", "ARREPENTIMIENTO", hoy=HOY)
    assert "cancel" in evaluacion.alternativa.lower()
    assert "§7" in evaluacion.clausulas


# --- Reglas 2 y 3: defectos y garantía (§3 excepción, §4, §9) ---------------

def test_defecto_prevalece_sobre_la_exclusion_de_higiene():
    evaluacion = evaluar(PEDIDOS["ECO-10002"], "ECO-CEP-BAM", "DEFECTUOSO", hoy=HOY)
    assert evaluacion.resultado == "ELEGIBLE"
    assert evaluacion.via == "GARANTIA"
    assert evaluacion.paga_retorno == "ECOMARKET"
    assert evaluacion.costo_retorno_cop == 0


@pytest.mark.parametrize("motivo", ["EQUIVOCADO", "INCOMPLETO", "VENCIDO"])
def test_producto_equivocado_incompleto_o_vencido_va_por_garantia(motivo):
    evaluacion = evaluar(PEDIDOS["ECO-10002"], "ECO-CEP-BAM", motivo, hoy=HOY)
    assert evaluacion.resultado == "ELEGIBLE"
    assert evaluacion.via == "GARANTIA"


def test_defecto_reportado_despues_de_30_dias_va_a_revision_humana():
    evaluacion = evaluar(PEDIDOS["ECO-10009"], "ECO-SHA-BAR", "DEFECTUOSO", hoy=HOY)
    assert evaluacion.dias_desde_entrega == 33
    assert evaluacion.resultado == "REVISION_HUMANA"
    assert "§9" in evaluacion.clausulas


def test_garantia_de_12_meses_cubre_defectos_despues_de_30_dias():
    evaluacion = evaluar(_pedido("ECO-COM-13L", "2026-03-01"), "ECO-COM-13L", "DEFECTUOSO", hoy=HOY)
    assert evaluacion.resultado == "ELEGIBLE"
    assert evaluacion.via == "GARANTIA"


def test_garantia_de_12_meses_vencida_va_a_revision_humana():
    evaluacion = evaluar(_pedido("ECO-COM-13L", "2025-09-01"), "ECO-COM-13L", "DEFECTUOSO", hoy=HOY)
    assert evaluacion.resultado == "REVISION_HUMANA"


# --- Regla 4: categorías excluidas (§3) -------------------------------------

@pytest.mark.parametrize(
    "sku",
    [
        "ECO-CEP-BAM",  # NO_DEV_HIGIENE
        "ECO-MIE-500",  # NO_DEV_PERECEDERO
        "ECO-AGE-PER",  # NO_DEV_PERSONALIZADO
        "ECO-TAR-REG",  # NO_DEV_DIGITAL
        "ECO-PLA-ALO",  # NO_DEV_VIVO
    ],
)
def test_categorias_excluidas_no_se_devuelven_ni_dentro_del_retracto(sku):
    evaluacion = evaluar(_pedido(sku, "2026-09-21"), sku, "ARREPENTIMIENTO", hoy=HOY)
    assert evaluacion.resultado == "NO_ELEGIBLE"
    assert evaluacion.codigo == "CATEGORIA_EXCLUIDA"
    assert "§3" in evaluacion.clausulas


# --- Regla 5: plazo de 30 días (§1) -----------------------------------------

def test_arrepentimiento_despues_de_30_dias_esta_fuera_de_plazo():
    evaluacion = evaluar(PEDIDOS["ECO-10009"], "ECO-SHA-BAR", "ARREPENTIMIENTO", sello_intacto=True, hoy=HOY)
    assert evaluacion.resultado == "NO_ELEGIBLE"
    assert evaluacion.codigo == "FUERA_DE_PLAZO"
    assert "§1" in evaluacion.clausulas


def test_el_dia_30_todavia_esta_dentro_del_plazo():
    evaluacion = evaluar(_pedido("ECO-BOT-500", "2026-08-23"), "ECO-BOT-500", "ARREPENTIMIENTO", hoy=HOY)
    assert evaluacion.dias_desde_entrega == 30
    assert evaluacion.resultado == "ELEGIBLE"


# --- Regla 6: higiene con sello (§4) ----------------------------------------

def test_sello_sin_confirmar_pide_confirmacion_e_informa_el_costo():
    evaluacion = evaluar(PEDIDOS["ECO-10005"], "ECO-PAN-REU", "ARREPENTIMIENTO", hoy=HOY)
    assert evaluacion.resultado == "REQUIERE_CONFIRMACION"
    assert evaluacion.pregunta_pendiente
    assert evaluacion.dias_desde_entrega == 17
    assert evaluacion.via == "ARREPENTIMIENTO_30D"
    assert evaluacion.paga_retorno == "CLIENTE"
    assert evaluacion.costo_retorno_cop == COSTO_RETORNO_CLIENTE_COP == 12000
    assert evaluacion.clausulas == ["§1", "§4", "§5"]


def test_sello_roto_no_es_devolvible():
    evaluacion = evaluar(PEDIDOS["ECO-10005"], "ECO-PAN-REU", "ARREPENTIMIENTO", sello_intacto=False, hoy=HOY)
    assert evaluacion.resultado == "NO_ELEGIBLE"
    assert evaluacion.codigo == "SELLO_ROTO"


def test_sello_intacto_entre_el_dia_6_y_el_30_es_elegible_con_costo():
    evaluacion = evaluar(PEDIDOS["ECO-10005"], "ECO-PAN-REU", "ARREPENTIMIENTO", sello_intacto=True, hoy=HOY)
    assert evaluacion.resultado == "ELEGIBLE"
    assert evaluacion.via == "ARREPENTIMIENTO_30D"
    assert evaluacion.costo_retorno_cop == 12000


# --- Regla 7: cambio de talla en textiles (§4, §5) --------------------------

def test_primer_cambio_de_talla_en_textiles_es_gratis():
    evaluacion = evaluar(_pedido("ECO-CAM-ALG", "2026-09-10"), "ECO-CAM-ALG", "CAMBIO_TALLA", hoy=HOY)
    assert evaluacion.resultado == "ELEGIBLE"
    assert evaluacion.via == "CAMBIO_TALLA"
    assert evaluacion.paga_retorno == "ECOMARKET"
    assert evaluacion.costo_retorno_cop == 0


def test_segundo_cambio_de_talla_lo_paga_el_cliente():
    evaluacion = evaluar(
        _pedido("ECO-CAM-ALG", "2026-09-10"), "ECO-CAM-ALG", "CAMBIO_TALLA", cambios_previos=1, hoy=HOY
    )
    assert evaluacion.resultado == "ELEGIBLE"
    assert evaluacion.paga_retorno == "CLIENTE"
    assert evaluacion.costo_retorno_cop == 12000


def test_cambio_de_talla_solo_aplica_a_textiles():
    evaluacion = evaluar(_pedido("ECO-BOT-500", "2026-09-10"), "ECO-BOT-500", "CAMBIO_TALLA", hoy=HOY)
    assert evaluacion.resultado == "NO_ELEGIBLE"
    assert evaluacion.codigo == "MOTIVO_NO_APLICA"


# --- Reglas 8 y 9: retracto y arrepentimiento (§1, §5, Ley 1480) -----------

def test_retracto_dentro_de_5_dias_habiles_lo_paga_ecomarket():
    evaluacion = evaluar(_pedido("ECO-BOT-500", "2026-09-16"), "ECO-BOT-500", "ARREPENTIMIENTO", hoy=HOY)
    assert evaluacion.resultado == "ELEGIBLE"
    assert evaluacion.via == "RETRACTO_5D"
    assert evaluacion.paga_retorno == "ECOMARKET"
    assert evaluacion.costo_retorno_cop == 0


def test_los_festivos_no_cuentan_como_dias_habiles():
    # Entrega el jueves 13 de agosto; el lunes 17 es festivo (Asunción).
    # Del 14 al 21 hay 5 días hábiles: sigue siendo retracto.
    evaluacion = evaluar(
        _pedido("ECO-BOT-500", "2026-08-13"), "ECO-BOT-500", "ARREPENTIMIENTO", hoy=date(2026, 8, 21)
    )
    assert evaluacion.via == "RETRACTO_5D"


def test_el_sexto_dia_habil_ya_es_arrepentimiento():
    evaluacion = evaluar(
        _pedido("ECO-BOT-500", "2026-08-13"), "ECO-BOT-500", "ARREPENTIMIENTO", hoy=date(2026, 8, 24)
    )
    assert evaluacion.via == "ARREPENTIMIENTO_30D"
    assert evaluacion.paga_retorno == "CLIENTE"


# --- Fecha por defecto ------------------------------------------------------

def test_sin_fecha_explicita_usa_la_fecha_de_referencia():
    evaluacion = evaluar(PEDIDOS["ECO-10005"], "ECO-PAN-REU", "ARREPENTIMIENTO")
    esperado = (date.fromisoformat(FECHA_REFERENCIA) - date(2026, 9, 5)).days
    assert evaluacion.dias_desde_entrega == esperado
