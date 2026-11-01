"""Pruebas de las herramientas del agente: compuertas, errores estructurados e idempotencia.

Se invocan como las invoca el agente (`herramienta.invoke(args, config)`), con un
registro en memoria y un `thread_id` por conversación. No requieren modelo, red
ni índice: lo que se prueba es que las garantías estén en el código y no dependan
de que el LLM siga el prompt.
"""

import pytest
from pydantic import ValidationError

from src import herramientas as h


@pytest.fixture(autouse=True)
def registro_limpio(monkeypatch):
    monkeypatch.delenv("ECOMARKET_SIMULAR_FALLA", raising=False)
    h.reiniciar_registro(None)


def _config(hilo: str = "hilo-1") -> dict:
    return {"configurable": {"thread_id": hilo}}


def _turno(hilo: str = "hilo-1") -> None:
    h.REGISTRO.iniciar_turno(hilo)


def _verificar(hilo: str = "hilo-1", **args) -> dict:
    return h.HERRAMIENTA_VERIFICAR_ELEGIBILIDAD.invoke(args, config=_config(hilo))


def _generar(id_verificacion: str, acepta: bool = False, hilo: str = "hilo-1") -> dict:
    return h.HERRAMIENTA_GENERAR_ETIQUETA.invoke(
        {"id_verificacion": id_verificacion, "cliente_acepta_costo": acepta}, config=_config(hilo)
    )


# --- verificar_elegibilidad_producto ------------------------------------------

def test_verificar_pide_confirmar_el_sello_e_informa_el_costo():
    _turno()
    r = _verificar(tracking="eco 10005", motivo="ARREPENTIMIENTO")
    assert r["ok"] and r["resultado"] == "REQUIERE_CONFIRMACION"
    assert r["sku"] == "ECO-PAN-REU"            # pedido de un solo producto: el SKU se infiere
    assert r["costo_retorno_cop"] == 12000
    assert r["id_verificacion"].startswith("VER-10005-")
    assert "sello" in r["siguiente_paso"]


def test_verificar_pedido_inexistente_es_error_estructurado():
    r = _verificar(tracking="ECO-99999", motivo="ARREPENTIMIENTO")
    assert r == {"ok": False, "codigo": "PEDIDO_NO_EXISTE", "detalle": r["detalle"]}


def test_verificar_pedido_entregado_con_varios_productos_exige_sku(monkeypatch):
    pedido = {**h.PEDIDOS["ECO-10003"], "tracking": "ECO-90001", "estado": "ENTREGADO",
              "fecha_entrega_real": "2026-09-20"}
    monkeypatch.setitem(h.PEDIDOS, "ECO-90001", pedido)
    r = _verificar(tracking="ECO-90001", motivo="ARREPENTIMIENTO")
    assert r["codigo"] == "SKU_REQUERIDO"
    assert {p["sku"] for p in r["productos"]} == {"ECO-JAB-SOL", "ECO-ESP-KON"}


def test_verificar_sku_ajeno_devuelve_los_productos_del_pedido():
    r = _verificar(tracking="ECO-10002", motivo="DEFECTUOSO", sku="ECO-BOT-500")
    assert r["codigo"] == "SKU_NO_PERTENECE"
    assert r["productos"][0]["sku"] == "ECO-CEP-BAM"


def test_argumentos_invalidos_no_llegan_a_la_funcion():
    with pytest.raises(ValidationError):
        _verificar(tracking="ECO-10002", motivo="NO_ME_GUSTA")
    with pytest.raises(ValidationError):
        _verificar(tracking="ECO-10005", motivo="ARREPENTIMIENTO", selo_intacto=True)


def test_el_config_no_se_expone_al_modelo():
    for herramienta in (h.HERRAMIENTA_VERIFICAR_ELEGIBILIDAD, h.HERRAMIENTA_GENERAR_ETIQUETA):
        assert "config" not in herramienta.tool_call_schema.model_json_schema()["properties"]


# --- generar_etiqueta_devolucion: compuertas ----------------------------------

def test_generar_con_id_inventado_falla():
    assert _generar("VER-10005-abcd")["codigo"] == "VERIFICACION_INVALIDA"


def test_generar_con_verificacion_no_elegible_falla():
    _turno()
    r = _verificar(tracking="ECO-10002", motivo="ARREPENTIMIENTO")   # higiene: no devolvible
    assert r["resultado"] == "NO_ELEGIBLE"
    assert _generar(r["id_verificacion"])["codigo"] == "VERIFICACION_INVALIDA"
    assert h.REGISTRO.guias == []


def test_generar_con_id_de_otra_conversacion_falla():
    _turno("hilo-1")
    r = _verificar("hilo-1", tracking="ECO-10002", motivo="DEFECTUOSO")
    _turno("hilo-2")
    assert _generar(r["id_verificacion"], hilo="hilo-2")["codigo"] == "VERIFICACION_INVALIDA"


def test_generar_con_verificacion_vencida_falla(monkeypatch):
    _turno()
    r = _verificar(tracking="ECO-10002", motivo="DEFECTUOSO")
    creada = h.REGISTRO.verificaciones[r["id_verificacion"]]["creada"]
    monkeypatch.setattr(h.time, "monotonic", lambda: creada + h.VIGENCIA_VERIFICACION_S + 1)
    assert _generar(r["id_verificacion"])["codigo"] == "VERIFICACION_INVALIDA"


def test_garantia_sin_costo_genera_la_guia_en_el_mismo_turno():
    _turno()
    r = _verificar(tracking="ECO-10002", motivo="DEFECTUOSO")
    guia = _generar(r["id_verificacion"])
    assert guia["ok"] and guia["numero_guia"] == "RET-ECO-10002-01"
    assert guia["costo_cliente_cop"] == 0
    assert "id_verificacion" not in guia


def test_con_costo_exige_aceptacion_explicita():
    _turno()
    r = _verificar(tracking="ECO-10005", motivo="ARREPENTIMIENTO", sello_intacto=True)
    _turno()
    assert _generar(r["id_verificacion"], acepta=False)["codigo"] == "CONFIRMACION_FALTANTE"


def test_con_costo_no_acepta_el_si_en_el_mismo_turno_en_que_se_informo():
    _turno()
    r = _verificar(tracking="ECO-10005", motivo="ARREPENTIMIENTO", sello_intacto=True)
    assert _generar(r["id_verificacion"], acepta=True)["codigo"] == "CONFIRMACION_FALTANTE"
    assert h.REGISTRO.guias == []


def test_con_costo_aceptado_en_el_turno_siguiente_genera_la_guia():
    _turno()
    r = _verificar(tracking="ECO-10005", motivo="ARREPENTIMIENTO", sello_intacto=True)
    _turno()
    guia = _generar(r["id_verificacion"], acepta=True)
    assert guia["ok"]
    assert guia["numero_guia"] == "RET-ECO-10005-01"
    assert guia["costo_cliente_cop"] == 12000
    assert guia["entregar_antes_de"] == "2026-09-29"    # 5 días hábiles desde el 2026-09-22
    assert guia["transportadora"] == "Interrapidisimo"


def test_generar_dos_veces_devuelve_la_misma_guia():
    _turno()
    r = _verificar(tracking="ECO-10002", motivo="DEFECTUOSO")
    primera = _generar(r["id_verificacion"])
    segunda = _generar(r["id_verificacion"])
    assert segunda["codigo"] == "YA_EXISTE"
    assert segunda["numero_guia"] == primera["numero_guia"]
    assert len(h.REGISTRO.guias) == 1


def test_falla_de_la_transportadora_pide_escalar_al_segundo_intento(monkeypatch):
    monkeypatch.setenv("ECOMARKET_SIMULAR_FALLA", "1")
    _turno()
    r = _verificar(tracking="ECO-10002", motivo="DEFECTUOSO")
    primero = _generar(r["id_verificacion"])
    segundo = _generar(r["id_verificacion"])
    assert primero["codigo"] == segundo["codigo"] == "TRANSPORTADORA_NO_DISPONIBLE"
    assert "reintentar" in primero["siguiente_paso"]
    assert "escalar_a_humano" in segundo["siguiente_paso"]
    assert h.REGISTRO.guias == []


def test_la_guia_se_persiste_en_el_registro(tmp_path):
    ruta = tmp_path / "devoluciones.json"
    h.reiniciar_registro(ruta)
    _turno()
    r = _verificar(tracking="ECO-10002", motivo="DEFECTUOSO")
    _generar(r["id_verificacion"])
    assert h.RegistroDevoluciones(ruta).guias[0]["numero_guia"] == "RET-ECO-10002-01"


# --- escalar_a_humano ------------------------------------------------------------

def test_escalar_crea_ticket_sin_pii():
    r = h.HERRAMIENTA_ESCALAR.invoke({
        "categoria": "FUERA_DE_POLITICA",
        "resumen": "Shampoo partido reportado 33 días después; cliente ana@correo.com, 3001234567",
        "tracking": "ECO-10009",
    })
    assert r["ok"] and r["ticket"] == "TCK-2026-0001" and r["sla_horas_habiles"] == 2
    ticket = h.REGISTRO.tickets[0]
    assert "ana@correo.com" not in ticket["resumen"] and "3001234567" not in ticket["resumen"]
    assert ticket["tracking"] == "ECO-10009"


# --- Datos que no cambian el resultado no se le exigen al agente -----------------

def test_pedido_no_entregado_responde_sin_motivo_ni_sku():
    r = _verificar(tracking="ECO-10004")                  # EN_PREPARACION
    assert r["resultado"] == "NO_ELEGIBLE" and r["codigo"] == "ESTADO_NO_ADMITE"
    assert "cancel" in r["alternativa"].lower()
    assert "motivo" not in r                              # no se inventa un motivo
    r = _verificar(tracking="ECO-10003")                  # RETRASADO, dos productos
    assert r["resultado"] == "NO_ELEGIBLE"


def test_pedido_entregado_sin_motivo_pide_preguntarlo():
    r = _verificar(tracking="ECO-10002")
    assert r["codigo"] == "MOTIVO_REQUERIDO"
