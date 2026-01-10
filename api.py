#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import asyncio
import os
import httpx

from world_model import processar_ponto, WorldContext
from spatial_memory import memoria

app = FastAPI(title="PROSPECTOR-AI", version="4.1")

# Configurações via Variáveis de Ambiente (Segurança)
API_KEY = os.environ.get('API_KEY', 'dev-key-12345')
SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

class LeituraSensor(BaseModel):
    id: str
    timestamp: str
    latitude: float
    longitude: float
    mag_nt: float
    altitude: Optional[float] = 0
    satellites: Optional[int] = 0

class ResultadoProcessado(BaseModel):
    id: str
    timestamp: str
    latitude: float
    longitude: float
    proprio_mag_nt: float
    filtrado_mag_nt: float
    cprm_litologia: Optional[str]
    cprm_cu_ppm: Optional[float]
    cprm_au_ppb: Optional[float]
    dist_estrutura_m: Optional[float]
    tipo_deposito: Optional[str]
    score_metalogenico: float
    anomalia_persistente: bool
    synapse_index: float
    synapse_ajustado: float
    risk_tier: str
    tier_code: str
    acao: Optional[str]
    justificativa: Optional[str]
    confianca: float

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Token inválido")
    return credentials.credentials

async def salvar_no_supabase(resultado: ResultadoProcessado):
    """Persistência real no Supabase"""
    if not SUPABASE_KEY:
        print("Aviso: SUPABASE_KEY não configurada. Dados não persistidos.")
        return

    async with httpx.AsyncClient(timeout=10) as client:
        data = resultado.model_dump()
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        }
        try:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/readings_processed",
                headers=headers,
                json=data
            )
            if resp.status_code not in [200, 201]:
                print(f"Erro ao salvar no Supabase: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Falha na conexão com Supabase: {e}")

@app.get("/")
async def root():
    return FileResponse("dashboard.html")

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "version": "4.1", 
        "supabase_connected": bool(SUPABASE_KEY),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/readings", response_model=ResultadoProcessado)
async def processar_leitura(leitura: LeituraSensor, background_tasks: BackgroundTasks, token: str = Depends(verify_token)):
    try:
        ctx = await processar_ponto(
            lat=leitura.latitude,
            lon=leitura.longitude,
            mag_nt=leitura.mag_nt,
            usar_ia=True
        )
        
        resumo = ctx.resumo()
        
        resultado = ResultadoProcessado(
            id=leitura.id,
            timestamp=leitura.timestamp,
            latitude=leitura.latitude,
            longitude=leitura.longitude,
            proprio_mag_nt=leitura.mag_nt,
            filtrado_mag_nt=resumo.get('filtrado_mag_nt', leitura.mag_nt),
            cprm_litologia=resumo.get('cprm_litologia'),
            cprm_cu_ppm=resumo.get('cprm_cu_ppm'),
            cprm_au_ppb=resumo.get('cprm_au_ppb'),
            dist_estrutura_m=resumo.get('dist_estrutura_m'),
            tipo_deposito=resumo.get('tipo_deposito'),
            score_metalogenico=resumo.get('score_metalogenico', 0),
            anomalia_persistente=resumo.get('anomalia_persistente', False),
            synapse_index=ctx.synapse_index,
            synapse_ajustado=ctx.synapse_ajustado,
            risk_tier=ctx.risk_tier,
            tier_code=ctx.tier_code,
            acao=resumo.get('acao'),
            justificativa=ctx.decisao.justificativa if ctx.decisao else None,
            confianca=resumo.get('confianca', 0)
        )
        
        # Salva no banco de dados em background para não atrasar a resposta da API
        background_tasks.add_task(salvar_no_supabase, resultado)
        
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno de processamento: {str(e)}")

@app.get("/api/v1/results", response_model=List[ResultadoProcessado])
async def listar_resultados(limit: int = 100, token: str = Depends(verify_token)):
    """Busca resultados diretamente do Supabase"""
    if not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Serviço de banco de dados não configurado")

    async with httpx.AsyncClient(timeout=10) as client:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/readings_processed?limit={limit}&order=timestamp.desc",
            headers=headers
        )
        if resp.status_code == 200:
            return resp.json()
        return []

@app.get("/api/v1/hotspots")
async def listar_hotspots(min_synapse: float = 70, min_dias: int = 2, token: str = Depends(verify_token)):
    hotspots = memoria.hotspots(min_synapse, min_dias)
    return [{
        'centro': c.centro(),
        'media_synapse': c.media_synapse(),
        'media_mag': c.media_mag(),
        'leituras': len(c.leituras),
        'dias': c.dias_distintos()
    } for c in hotspots[:20]]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
