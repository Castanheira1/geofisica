#!/usr/bin/env python3

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("prospector.ia")

# Inicialização preguiçosa: importar este módulo NÃO deve falhar nem exigir chave.
# O cliente só é criado quando há OPENAI_API_KEY e a IA é efetivamente usada.
_client = None


def _get_client():
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            return None
        from openai import OpenAI
        _client = OpenAI()
    return _client

@dataclass
class ContextoPonto:
    latitude: float
    longitude: float
    mag_nt: float
    litologia: str
    unidade_geologica: str
    distancia_estrutura_m: float
    tipo_estrutura: str
    cu_ppm: Optional[float]
    au_ppb: Optional[float]
    deposito_proximo: str
    distancia_deposito_m: float
    synapse_index: float
    risk_tier: str
    tipo_deposito_provavel: str
    score_metalogenico: float
    anomalia_persistente: bool
    dias_observados: int

@dataclass
class InterpretacaoGeologica:
    interpretacao: str
    recomendacao: str
    confianca: str
    modelo_provavel: str
    proximos_passos: str

async def interpretar_ponto(ctx: ContextoPonto) -> InterpretacaoGeologica:
    """Gera um PARECER TEXTUAL (narrativa) a partir do contexto do ponto.

    [AVISO — ver RELATORIO_TECNICO.md] A saída do LLM é interpretação em linguagem
    natural para apoio à leitura humana; NÃO é uma medição nem uma fonte de evidência
    quantitativa, e o campo `confianca` aqui é qualitativo. A favorabilidade e a
    incerteza quantitativas vêm do WorldModel (dado + física), não do LLM.

    Nota de stack: este módulo usa a API OpenAI; `requirements.txt`/README mencionam
    `anthropic`. Padronizar o provedor é um item de limpeza pendente.
    """
    prompt = f"""
    Você é um Geólogo Sênior Especialista em Exploração Mineral (IOCG, Ouro Orogênico).
    Analise os dados deste ponto de prospecção em Carajás:

    LOCALIZAÇÃO: {ctx.latitude:.6f}, {ctx.longitude:.6f}
    MAGNETOMETRIA: {ctx.mag_nt:.0f} nT
    SYNAPSE INDEX: {ctx.synapse_index:.1f} ({ctx.risk_tier})
    LITOLOGIA: {ctx.litologia} ({ctx.unidade_geologica})
    ESTRUTURA: {ctx.tipo_estrutura} a {ctx.distancia_estrutura_m:.0f}m
    GEOQUÍMICA: Cu={ctx.cu_ppm if ctx.cu_ppm else 'N/D'} ppm, Au={ctx.au_ppb if ctx.au_ppb else 'N/D'} ppb
    MODELO METALOGÊNICO: {ctx.tipo_deposito_provavel} (Score: {ctx.score_metalogenico:.2f})
    PERSISTÊNCIA: {'Sim' if ctx.anomalia_persistente else 'Não'} ({ctx.dias_observados} dias)

    Forneça um parecer técnico estruturado em JSON com os campos:
    - interpretacao: Significado geológico dos dados.
    - recomendacao: O que fazer com este alvo.
    - confianca: Nível de confiança (Alto/Médio/Baixo).
    - modelo_provavel: Modelo de depósito mais compatível.
    - proximos_passos: Investigação prática a seguir.
    """
    
    client = _get_client()
    if client is None:
        return InterpretacaoGeologica(
            interpretacao="Interpretação por IA desativada (sem OPENAI_API_KEY).",
            recomendacao="Usar a favorabilidade e a incerteza quantitativas do WorldModel.",
            confianca="N/A",
            modelo_provavel=ctx.tipo_deposito_provavel,
            proximos_passos="Configurar OPENAI_API_KEY para habilitar o parecer textual.",
        )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Você é um sistema de IA geológica de alta precisão para exploração mineral."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content)
        
        return InterpretacaoGeologica(
            interpretacao=data.get('interpretacao', 'Análise inconclusiva'),
            recomendacao=data.get('recomendacao', 'Monitorar'),
            confianca=data.get('confianca', 'Médio'),
            modelo_provavel=data.get('modelo_provavel', ctx.tipo_deposito_provavel),
            proximos_passos=data.get('proximos_passos', 'Investigação adicional')
        )
    except Exception:
        logger.exception("Erro na interpretação da IA")
        return InterpretacaoGeologica(
            interpretacao="Erro no processamento da IA",
            recomendacao="Verificar logs",
            confianca="N/A",
            modelo_provavel=ctx.tipo_deposito_provavel,
            proximos_passos="Revisar conexão com API"
        )

def criar_contexto(
    lat: float, lon: float, mag_nt: float,
    litologia: str = '', unidade: str = '',
    dist_estrut: float = 99999, tipo_estrut: str = '',
    cu: Optional[float] = None, au: Optional[float] = None,
    deposito: str = '', dist_dep: float = 99999,
    synapse: float = 0, tier: str = 'BACKGROUND',
    tipo_dep: str = 'UNKNOWN', score_metal: float = 0,
    persistente: bool = False, dias: int = 0
) -> ContextoPonto:
    return ContextoPonto(
        lat, lon, mag_nt, litologia, unidade,
        dist_estrut, tipo_estrut, cu, au,
        deposito, dist_dep, synapse, tier,
        tipo_dep, score_metal, persistente, dias
    )
