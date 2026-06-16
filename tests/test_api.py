from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api

client = TestClient(api.app)
AUTH = {"Authorization": "Bearer test-key"}


def _fake_ctx():
    decisao = SimpleNamespace(justificativa="parecer de teste", acao="MONITORAR", confianca=0.4)
    ctx = SimpleNamespace(
        synapse_index=55.0, synapse_ajustado=50.0,
        risk_tier="MEDIUM", tier_code="T3", decisao=decisao,
    )
    ctx.resumo = lambda: {
        "filtrado_mag_nt": 26500.0, "anomalia_nt": 1700.0, "anomalia_confiavel": False,
        "cprm_litologia": None, "cprm_cu_ppm": None, "cprm_au_ppb": None,
        "dist_estrutura_m": None, "tipo_deposito": "IOCG", "score_metalogenico": 0.6,
        "anomalia_persistente": False, "completude_dados": 0.25, "incerteza": 0.7,
        "acao": "MONITORAR", "confianca": 0.4,
    }
    return ctx


def _leitura():
    return {"id": "PT-1", "timestamp": "2026-06-16T00:00:00Z",
            "latitude": -6.06, "longitude": -50.18, "mag_nt": 26500.0}


def test_health_publico():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == api.config.VERSION


def test_auth_obrigatoria():
    r = client.post("/api/v1/readings", json=_leitura())
    assert r.status_code in (401, 403)  # sem Authorization


def test_token_invalido():
    r = client.post("/api/v1/readings", json=_leitura(), headers={"Authorization": "Bearer errado"})
    assert r.status_code == 401


def test_validacao_latitude(monkeypatch):
    async def fake(**kw):
        return _fake_ctx()
    monkeypatch.setattr(api, "processar_ponto", fake)
    ruim = _leitura() | {"latitude": 200.0}
    r = client.post("/api/v1/readings", json=ruim, headers=AUTH)
    assert r.status_code == 422


def test_validacao_mag_nt(monkeypatch):
    async def fake(**kw):
        return _fake_ctx()
    monkeypatch.setattr(api, "processar_ponto", fake)
    ruim = _leitura() | {"mag_nt": -5.0}
    r = client.post("/api/v1/readings", json=ruim, headers=AUTH)
    assert r.status_code == 422


def test_processamento_ok(monkeypatch):
    async def fake(**kw):
        return _fake_ctx()
    monkeypatch.setattr(api, "processar_ponto", fake)
    r = client.post("/api/v1/readings", json=_leitura(), headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["risk_tier"] == "MEDIUM"
    assert body["incerteza"] == 0.7
    assert body["confianca"] == 0.4
    assert body["tipo_deposito"] == "IOCG"


def test_erro_interno_nao_vaza(monkeypatch):
    async def boom(**kw):
        raise RuntimeError("segredo interno do stack")
    monkeypatch.setattr(api, "processar_ponto", boom)
    r = client.post("/api/v1/readings", json=_leitura(), headers=AUTH)
    assert r.status_code == 500
    assert "segredo interno" not in r.text  # não vaza detalhe interno
