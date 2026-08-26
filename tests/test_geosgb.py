"""Testes do cliente WFS do SGB.

Todos offline (httpx.MockTransport) — CI não depende de rede. O teste marcado
`network` só roda com `pytest -m network`.

Os três primeiros são REGRESSÃO dos bugs que faziam o cliente devolver
HTTP 200 com zero feature sem sinalizar nada.
"""
import asyncio
import json

import httpx
import pytest

import geological_layers as gl
from geological_layers import (
    CRS84_URN,
    GeoSGBClient,
    Geoquimica,
    TipoLitologia,
    _plausivel,
    azimute_segmento,
    classificar_litologia,
)


def _cliente_falso(handler):
    """GeoSGBClient com transporte controlado, sem abrir rede."""
    c = GeoSGBClient()
    c.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


def _json_resp(payload):
    return httpx.Response(200, json=payload)


# ---------------------------------------------------------------------------
# Regressão: a forma da requisição WFS 2.0
# ---------------------------------------------------------------------------

def test_usa_typenames_plural_e_bbox_com_crs():
    """WFS 2.0 exige `typeNames`; `typeName` (singular, sintaxe 1.x) devolve
    zero feature com HTTP 200. E o bbox sem CRS explícito é lido como
    (lat, lon), caindo fora da área. Ambos causavam vazio silencioso."""
    capturadas = []

    def handler(request):
        capturadas.append(request)
        return _json_resp({"features": []})

    c = _cliente_falso(handler)
    asyncio.run(c.buscar_depositos(-6.43, -50.05))

    assert capturadas, "nenhuma requisição foi feita"
    q = capturadas[0].url.params
    assert "typeNames" in q, "WFS 2.0 exige typeNames (plural)"
    assert "typeName" not in dict(q), "typeName singular devolve 0 feature"
    assert q["bbox"].endswith(CRS84_URN), "bbox precisa fixar o CRS (ordem lon,lat)"
    # ordem lon,lat: primeiro valor é longitude (~-50), segundo latitude (~-6)
    lon_min, lat_min = (float(v) for v in q["bbox"].split(",")[:2])
    assert -51 < lon_min < -49 and -7 < lat_min < -6


def test_vazio_nao_e_silencioso():
    """O bug original: 200 OK + zero feature era indistinguível de 'não há
    nada aqui'. Agora toda consulta vazia deixa motivo no diagnóstico."""
    c = _cliente_falso(lambda r: _json_resp({"features": []}))
    asyncio.run(c.buscar_depositos(-6.43, -50.05))
    assert c.diagnostico, "consulta vazia precisa registrar o motivo"
    assert "sem feature" in c.diagnostico[0]


def test_excecao_xml_com_http_200_nao_vira_sucesso():
    """O GeoServer devolve ExceptionReport em XML com status 200. O código
    anterior engolia isso num `except: pass`."""
    xml = '<?xml version="1.0"?><ows:ExceptionReport>camada inexistente</ows:ExceptionReport>'
    c = _cliente_falso(lambda r: httpx.Response(200, text=xml))
    estruturas = asyncio.run(c.buscar_estruturas(-6.43, -50.05))
    assert estruturas == []
    assert any("XML" in d for d in c.diagnostico)


def test_http_500_registrado():
    c = _cliente_falso(lambda r: httpx.Response(500, text="erro"))
    asyncio.run(c.buscar_depositos(-6.43, -50.05))
    assert any("HTTP 500" in d for d in c.diagnostico)


# ---------------------------------------------------------------------------
# Mapeamento de campos (nomes reais do serviço)
# ---------------------------------------------------------------------------

OCORRENCIA_REAL = {
    "features": [{
        "geometry": {"type": "Point", "coordinates": [-50.0518, -6.4386]},
        "properties": {
            "toponimia": "Sossego",
            "substancias": "Cobre, Ouro",
            "importancia": "Depósito",
            "classes_utilitarias": "Metais não ferrosos e semimetais",
            "rochas_hosp": "Brecha",
            "morfologia": "Brechada, Stockwork",
            "status_econ": "Lavra",
        },
    }]
}

LITO_REAL = {
    "features": [{
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {
            "nome_unida": "Grão Pará", "sigla_unid": "A4gp", "hierarquia": "Grupo",
            "litotipo1": "Jaspilito", "litotipo2": None,
            "era_maxima": "Neoarqueano", "eon_idad_m": "Arqueano",
        },
    }]
}


def test_mapeia_campos_reais_de_ocorrencia():
    """Os nomes antigos ('nome', 'substancia', 'denominacao') não existem na
    camada. Os reais são 'toponimia', 'substancias', 'importancia'."""
    c = _cliente_falso(lambda r: _json_resp(OCORRENCIA_REAL))
    deps = asyncio.run(c.buscar_depositos(-6.43, -50.05))
    assert len(deps) == 1
    d = deps[0]
    assert d.nome == "Sossego"
    assert d.substancia == "Cobre, Ouro"
    assert d.importancia == "Depósito"      # rótulo para PU learning (Fase 3)
    assert d.distancia_m < 2000


def test_mapeia_campos_reais_de_litologia():
    c = _cliente_falso(lambda r: _json_resp(LITO_REAL))
    lito = asyncio.run(c.buscar_litologia(-6.43, -50.05))
    assert lito.unidade == "Grão Pará"
    assert lito.sigla == "A4gp"
    assert lito.codigo == TipoLitologia.JASPILITO
    assert lito.fonte == "litotipo"
    # A escala precisa viajar junto: 1:2.5M é contexto regional, não prospecto.
    assert lito.escala_nominal == 2_500_000


def test_litologia_cai_para_nome_da_unidade_quando_litotipo_nulo():
    payload = json.loads(json.dumps(LITO_REAL))
    payload["features"][0]["properties"]["litotipo1"] = None
    payload["features"][0]["properties"]["nome_unida"] = "Formação Águas Claras"
    c = _cliente_falso(lambda r: _json_resp(payload))
    lito = asyncio.run(c.buscar_litologia(-6.43, -50.05))
    assert lito.codigo == TipoLitologia.FORMACAO_AGUAS_CLARAS
    assert lito.fonte == "nome_unidade"     # inferência mais fraca, declarada


def test_gnaisse_do_embasamento_e_classificado():
    # Regressão: 'Gnaisse' (Complexo Carapanã) voltava DESCONHECIDO e inflava
    # artificialmente a favorabilidade via `_score_lit` (0.5 em vez de 0.3).
    assert classificar_litologia("Gnaisse") == TipoLitologia.EMBASAMENTO_GNAISSICO


# ---------------------------------------------------------------------------
# Guarda de plausibilidade da geoquímica
# ---------------------------------------------------------------------------

def test_descarta_geoquimica_fisicamente_impossivel():
    """A camada publica valores como fe_pct=2310 (2310% de ferro). Sem guarda,
    isso viraria alvo fantasma no scoring."""
    payload = {"features": [{
        "geometry": {"type": "Point", "coordinates": [-50.05, -6.43]},
        "properties": {"cu_ppm": 1.56, "au_ppm": 0.005, "fe_pct": 2310,
                       "as_ppm": 5000, "num_campo": "AS-S-662", "Tipo": "Sedimento de corrente"},
    }]}
    c = _cliente_falso(lambda r: _json_resp(payload))
    g = asyncio.run(c.buscar_geoquimica(-6.43, -50.05))
    assert g.cu_ppm == 1.56
    assert g.au_ppb == 5.0          # au_ppm -> ppb
    assert g.fe_pct is None
    assert g.as_ppm is None
    assert len(g.descartados) == 2


def test_valores_baixos_sao_background_valido():
    # Não confundir "baixo" com "impossível": abaixo do limite de detecção o
    # SGB grava metade do LD, que é um número pequeno e legítimo.
    assert _plausivel("as_ppm", 0.05)
    assert _plausivel("cu_ppm", 0.0)
    assert not _plausivel("as_ppm", -5)      # convenção de "menor que"
    assert not _plausivel("fe_pct", 2310)


# ---------------------------------------------------------------------------
# Azimute derivado da geometria (a camada não tem campo de direção)
# ---------------------------------------------------------------------------

def test_azimute_segmento():
    assert azimute_segmento(0, 0, 0, 1) == pytest.approx(0, abs=1)      # N-S
    assert azimute_segmento(0, 0, 1, 0) == pytest.approx(90, abs=1)     # E-W
    assert azimute_segmento(0, 0, -1, 0) == pytest.approx(90, abs=1)    # sem sentido


def test_estrutura_recebe_azimute_calculado():
    payload = {"features": [{
        "geometry": {"type": "LineString", "coordinates": [[-50.1, -6.5], [-50.1, -6.4]]},
        "properties": {"tipo_estru": "Zona de cisalhamento compressional",
                       "nmestrutur": "Cinzento"},
    }]}
    c = _cliente_falso(lambda r: _json_resp(payload))
    ests = asyncio.run(c.buscar_estruturas(-6.43, -50.05))
    assert ests and ests[0].tipo.startswith("Zona de cisalhamento")
    assert ests[0].azimute == pytest.approx(0, abs=5)   # linha N-S


# ---------------------------------------------------------------------------
# Integração real (opt-in): pytest -m network
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_servico_real_responde_em_carajas():
    gl.GeoSGBClient._typenames_cache = None
    ctx = asyncio.run(gl.obter_contexto_geologico(-6.43, -50.05))
    nomes = [d.nome for d in ctx.depositos]
    assert any("Sossego" in n for n in nomes), f"esperava Sossego, veio {nomes}"
    assert ctx.consultas_com_dado() >= 3
