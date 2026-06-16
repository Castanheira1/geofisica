#!/usr/bin/env python3

import hmac
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, field_validator

import config
from spatial_memory import memoria
from world_model import processar_ponto

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("prospector.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fail-fast em produção: recusa subir com configuração insegura."""
    problemas = config.validar_producao()
    if problemas:
        msg = "Configuração de produção inválida: " + "; ".join(problemas)
        logger.error(msg)
        raise RuntimeError(msg)
    if not config.IS_PRODUCTION:
        logger.warning("Rodando em modo DESENVOLVIMENTO (APP_ENV=%s).", config.APP_ENV)
    if not config.USAR_IA:
        logger.info("Interpretação por IA desativada (USAR_IA=false ou sem OPENAI_API_KEY).")
    yield


app = FastAPI(title="PROSPECTOR-AI", version=config.VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=True)


class LeituraSensor(BaseModel):
    id: str
    timestamp: str
    latitude: float
    longitude: float
    mag_nt: float
    altitude: Optional[float] = 0
    satellites: Optional[int] = 0

    @field_validator("id", "timestamp")
    @classmethod
    def _nao_vazio(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("campo obrigatório vazio")
        return v

    @field_validator("latitude")
    @classmethod
    def _lat_valida(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise ValueError("latitude fora de [-90, 90]")
        return v

    @field_validator("longitude")
    @classmethod
    def _lon_valida(cls, v: float) -> float:
        if not (-180.0 <= v <= 180.0):
            raise ValueError("longitude fora de [-180, 180]")
        return v

    @field_validator("mag_nt")
    @classmethod
    def _mag_plausivel(cls, v: float) -> float:
        # Campo magnético total terrestre plausível (margem larga). Rejeita NaN/absurdos.
        if not (0.0 < v < 100000.0):
            raise ValueError("mag_nt fora de faixa plausível (0-100000 nT)")
        return v


class ResultadoProcessado(BaseModel):
    id: str
    timestamp: str
    latitude: float
    longitude: float
    proprio_mag_nt: float
    filtrado_mag_nt: float
    anomalia_nt: Optional[float] = None
    anomalia_confiavel: bool = False
    cprm_litologia: Optional[str] = None
    cprm_cu_ppm: Optional[float] = None
    cprm_au_ppb: Optional[float] = None
    dist_estrutura_m: Optional[float] = None
    tipo_deposito: Optional[str] = None
    score_metalogenico: float
    anomalia_persistente: bool
    synapse_index: float            # favorabilidade RELATIVA (0-100), não probabilidade
    synapse_ajustado: float
    risk_tier: str
    tier_code: str
    completude_dados: float = 0.0   # fração da evidência esperada presente (0-1)
    incerteza: float = 1.0          # 0-1 (1 = máxima)
    acao: Optional[str] = None
    justificativa: Optional[str] = None
    confianca: float                # honesta: derivada da incerteza, não fixada


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not config.API_KEY:
        # Em produção isto não acontece (startup falha); defensivo.
        raise HTTPException(status_code=503, detail="Servidor não configurado")
    # Comparação em tempo constante para evitar timing attack.
    if not hmac.compare_digest(credentials.credentials, config.API_KEY):
        raise HTTPException(status_code=401, detail="Token inválido")
    return credentials.credentials


async def salvar_no_supabase(resultado: ResultadoProcessado):
    """Persistência no Supabase (best-effort, em background)."""
    if not config.SUPABASE_KEY or not config.SUPABASE_URL:
        logger.info("SUPABASE não configurado; resultado %s não persistido.", resultado.id)
        return

    async with httpx.AsyncClient(timeout=10) as client:
        headers = {
            "apikey": config.SUPABASE_KEY,
            "Authorization": f"Bearer {config.SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }
        try:
            resp = await client.post(
                f"{config.SUPABASE_URL}/rest/v1/readings_processed",
                headers=headers,
                json=resultado.model_dump(),
            )
            if resp.status_code not in (200, 201):
                logger.error("Supabase retornou %s ao salvar %s", resp.status_code, resultado.id)
        except Exception:
            logger.exception("Falha na conexão com Supabase ao salvar %s", resultado.id)


@app.get("/")
async def root():
    if os.path.exists("dashboard.html"):
        return FileResponse("dashboard.html")
    return JSONResponse({"service": "PROSPECTOR-AI", "version": config.VERSION})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": config.VERSION,
        "env": config.APP_ENV,
        "supabase_connected": bool(config.SUPABASE_KEY and config.SUPABASE_URL),
        "ia_ativa": config.USAR_IA,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/readings", response_model=ResultadoProcessado)
async def processar_leitura(
    leitura: LeituraSensor,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token),
):
    try:
        ctx = await processar_ponto(
            lat=leitura.latitude,
            lon=leitura.longitude,
            mag_nt=leitura.mag_nt,
            usar_ia=config.USAR_IA,
        )
        resumo = ctx.resumo()

        resultado = ResultadoProcessado(
            id=leitura.id,
            timestamp=leitura.timestamp,
            latitude=leitura.latitude,
            longitude=leitura.longitude,
            proprio_mag_nt=leitura.mag_nt,
            filtrado_mag_nt=resumo.get("filtrado_mag_nt", leitura.mag_nt),
            anomalia_nt=resumo.get("anomalia_nt"),
            anomalia_confiavel=resumo.get("anomalia_confiavel", False),
            cprm_litologia=resumo.get("cprm_litologia"),
            cprm_cu_ppm=resumo.get("cprm_cu_ppm"),
            cprm_au_ppb=resumo.get("cprm_au_ppb"),
            dist_estrutura_m=resumo.get("dist_estrutura_m"),
            tipo_deposito=resumo.get("tipo_deposito"),
            score_metalogenico=resumo.get("score_metalogenico", 0),
            anomalia_persistente=resumo.get("anomalia_persistente", False),
            synapse_index=ctx.synapse_index,
            synapse_ajustado=ctx.synapse_ajustado,
            risk_tier=ctx.risk_tier,
            tier_code=ctx.tier_code,
            completude_dados=resumo.get("completude_dados", 0.0),
            incerteza=resumo.get("incerteza", 1.0),
            acao=resumo.get("acao"),
            justificativa=ctx.decisao.justificativa if ctx.decisao else None,
            confianca=resumo.get("confianca", 0),
        )

        # Persiste em background para não atrasar a resposta.
        background_tasks.add_task(salvar_no_supabase, resultado)
        return resultado
    except Exception:
        # Loga o detalhe internamente; não vaza stack/mensagem para o cliente.
        logger.exception("Erro ao processar leitura %s", getattr(leitura, "id", "?"))
        raise HTTPException(status_code=500, detail="Erro interno de processamento")


@app.get("/api/v1/results", response_model=List[ResultadoProcessado])
async def listar_resultados(limit: int = 100, token: str = Depends(verify_token)):
    """Busca resultados diretamente do Supabase."""
    limit = max(1, min(limit, 1000))
    if not config.SUPABASE_KEY or not config.SUPABASE_URL:
        raise HTTPException(status_code=503, detail="Serviço de banco de dados não configurado")

    async with httpx.AsyncClient(timeout=10) as client:
        headers = {
            "apikey": config.SUPABASE_KEY,
            "Authorization": f"Bearer {config.SUPABASE_KEY}",
        }
        try:
            resp = await client.get(
                f"{config.SUPABASE_URL}/rest/v1/readings_processed"
                f"?limit={limit}&order=timestamp.desc",
                headers=headers,
            )
            if resp.status_code == 200:
                return resp.json()
            logger.error("Supabase retornou %s ao listar resultados", resp.status_code)
            return []
        except Exception:
            logger.exception("Falha ao consultar Supabase")
            raise HTTPException(status_code=502, detail="Falha ao consultar banco de dados")


@app.get("/api/v1/hotspots")
async def listar_hotspots(min_synapse: float = 70, min_dias: int = 2, token: str = Depends(verify_token)):
    hotspots = memoria.hotspots(min_synapse, min_dias)
    return [{
        "centro": c.centro(),
        "media_synapse": c.media_synapse(),
        "media_mag": c.media_mag(),
        "leituras": len(c.leituras),
        "dias": c.dias_distintos(),
    } for c in hotspots[:20]]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
