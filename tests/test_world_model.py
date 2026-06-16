import asyncio

import world_model
from world_model import WorldModel
from geological_layers import ContextoGeologico, Litologia, Geoquimica, Estrutura
from metalogenic_context import TipoDeposito


def _ctx_rico(lat, lon):
    g = ContextoGeologico(latitude=lat, longitude=lon)
    g.litologia = Litologia(codigo=7, nome="Metabasalto", unidade="Grupo Grão Pará")
    g.geoquimica = Geoquimica(cu_ppm=800.0, au_ppb=150.0, distancia_m=120.0)
    est = Estrutura(tipo="cisalhamento", nome="Falha Carajás", distancia_m=300.0)
    g.estruturas = [est]
    g.estrutura_mais_proxima = est
    return g


def _ctx_vazio(lat, lon):
    return ContextoGeologico(latitude=lat, longitude=lon)


def _processar(monkeypatch_ctx_fn, lat=-6.06, lon=-50.18, mag=26500.0):
    async def fake(la, lo):
        return monkeypatch_ctx_fn(la, lo)
    orig = world_model.obter_contexto_geologico
    world_model.obter_contexto_geologico = fake
    try:
        model = WorldModel(usar_ia=False)
        return asyncio.run(model.processar(lat, lon, mag))
    finally:
        world_model.obter_contexto_geologico = orig


def test_sem_dados_nao_fabrica_confianca():
    ctx = _processar(_ctx_vazio)
    assert ctx.completude_dados == 0.0
    assert ctx.incerteza == 1.0
    assert ctx.decisao.confianca == 0.0
    assert ctx.decisao.acao == "BAIXA_PRIORIDADE"


def test_dados_ricos_reduzem_incerteza():
    ctx = _processar(_ctx_rico)
    assert ctx.completude_dados >= 0.5
    assert ctx.incerteza < 1.0
    assert ctx.synapse_index > 0.0


def test_nunca_recomenda_perfurar_imediato():
    # A decisão é triagem/priorização; jamais um veredito de furo super-confiante.
    ctx = _processar(_ctx_rico)
    assert ctx.decisao.acao != "PERFURAR_IMEDIATO"
    assert 0.0 <= ctx.decisao.confianca <= 1.0


def test_confianca_atrelada_a_incerteza():
    ctx = _processar(_ctx_rico)
    assert abs(ctx.decisao.confianca - round(1.0 - ctx.incerteza, 2)) < 1e-6


def test_usa_anomalia_residual():
    ctx = _processar(_ctx_rico, mag=26500.0)
    # A anomalia residual difere do campo total absoluto.
    assert ctx.anomalia_nt != ctx.mag_nt_proprio
    assert ctx.baseline_metodo in ("igrf", "regional_local", "igrf_aprox")
