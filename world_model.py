#!/usr/bin/env python3

import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

from geological_layers import obter_contexto_geologico, ContextoGeologico, TipoLitologia
from metalogenic_context import analisar_metalogenia, AnaliseMetalogenica, TipoDeposito
from spatial_memory import memoria, AnaliseEspacial
from ai_interpreter import interpretar_ponto, criar_contexto, InterpretacaoGeologica
from mag_preprocessing import MagPreProcessor, MagPoint

@dataclass
class DecisaoFinal:
    acao: str
    justificativa: str
    confianca: float
    prioridade: int

@dataclass
class WorldContext:
    latitude: float
    longitude: float
    mag_nt_proprio: float
    mag_nt_filtrado: float
    timestamp: str
    
    geologico: Optional[ContextoGeologico] = None
    metalogenico: Optional[AnaliseMetalogenica] = None
    espacial: Optional[AnaliseEspacial] = None
    interpretacao_ia: Optional[InterpretacaoGeologica] = None
    
    synapse_index: float = 0.0
    synapse_ajustado: float = 0.0
    risk_tier: str = 'BACKGROUND'
    tier_code: str = 'T5'
    
    decisao: Optional[DecisaoFinal] = None
    
    def resumo(self) -> dict:
        return {
            'id': f"PT-{self.timestamp}",
            'lat': self.latitude,
            'lon': self.longitude,
            'proprio_mag_nt': self.mag_nt_proprio,
            'filtrado_mag_nt': self.mag_nt_filtrado,
            'cprm_litologia': self.geologico.litologia.nome if self.geologico else None,
            'cprm_cu_ppm': self.geologico.geoquimica.cu_ppm if self.geologico else None,
            'cprm_au_ppb': self.geologico.geoquimica.au_ppb if self.geologico else None,
            'dist_estrutura_m': self.geologico.distancia_estrutura_m() if self.geologico else None,
            'tipo_deposito': self.metalogenico.tipo_mais_provavel.name if self.metalogenico else None,
            'score_metalogenico': self.metalogenico.score_maximo if self.metalogenico else 0,
            'anomalia_persistente': self.espacial.anomalia_persistente if self.espacial else False,
            'synapse_index': self.synapse_index,
            'synapse_ajustado': self.synapse_ajustado,
            'risk_tier': self.risk_tier,
            'tier_code': self.tier_code,
            'acao': self.decisao.acao if self.decisao else None,
            'confianca': self.decisao.confianca if self.decisao else 0
        }

class WorldModel:
    def __init__(self, usar_ia: bool = True):
        self.usar_ia = usar_ia
        self.preprocessor = MagPreProcessor()
        self.buffer_pontos = [] # Para micronivelamento em tempo real
    
    async def processar(self, lat: float, lon: float, mag_nt: float, timestamp: str = '') -> WorldContext:
        # 1. Pré-processamento e Filtragem
        # Adiciona ao buffer para análise de contexto local
        ponto_atual = MagPoint(id=f"TMP-{timestamp}", latitude=lat, longitude=lon, mag_nt=mag_nt)
        self.buffer_pontos.append(ponto_atual)
        
        # Se tivermos pontos suficientes, aplicamos micronivelamento
        mag_filtrado = mag_nt
        if len(self.buffer_pontos) >= 5:
            # Remove spikes e aplica filtro de mediana no buffer local
            pontos_limpos = self.preprocessor.remover_spikes(self.buffer_pontos)
            res = self.preprocessor.processar_completo(pontos_limpos, filtros=['mediana'])
            if 'mediana' in res['filtrados']:
                mag_filtrado = res['filtrados']['mediana'][-1].mag_filtered
            
            # Mantém buffer pequeno para performance
            if len(self.buffer_pontos) > 50:
                self.buffer_pontos.pop(0)

        ctx = WorldContext(
            latitude=lat,
            longitude=lon,
            mag_nt_proprio=mag_nt,
            mag_nt_filtrado=mag_filtrado,
            timestamp=timestamp or str(int(asyncio.get_event_loop().time()))
        )
        
        # 2. Enriquecimento Geológico (CPRM)
        ctx.geologico = await obter_contexto_geologico(lat, lon)
        
        # 3. Análise Metalogênica
        lit_codigo = int(ctx.geologico.litologia.codigo) if ctx.geologico else 0
        dist_estrut = ctx.geologico.distancia_estrutura_m() if ctx.geologico else 99999
        cu = ctx.geologico.geoquimica.cu_ppm if ctx.geologico else None
        au = ctx.geologico.geoquimica.au_ppb if ctx.geologico else None
        
        ctx.metalogenico = analisar_metalogenia(
            lat, lon, mag_filtrado,
            cu_ppm=cu, au_ppb=au,
            litologia_codigo=lit_codigo,
            distancia_estrutura_m=dist_estrut
        )
        
        # 4. Cálculo do Synapse Index
        ctx.synapse_index = self._calcular_synapse(ctx)
        
        # 5. Memória Espacial e Persistência
        ctx.espacial = memoria.analisar(lat, lon, mag_filtrado, ctx.synapse_index)
        
        # Ajuste por persistência temporal
        ctx.synapse_ajustado = ctx.synapse_index * ctx.espacial.fator_confianca
        ctx.synapse_ajustado = min(100, max(0, ctx.synapse_ajustado))
        
        # 6. Classificação de Risco
        ctx.risk_tier, ctx.tier_code = self._classificar_tier(ctx.synapse_ajustado)
        
        # Registro na memória
        memoria.registrar(
            id=f"PT-{ctx.timestamp}",
            lat=lat, lon=lon,
            mag_nt=mag_filtrado,
            synapse=ctx.synapse_ajustado,
            tier=ctx.risk_tier
        )
        
        # 7. Interpretação por IA (Claude)
        if self.usar_ia and ctx.synapse_ajustado >= 40: # Baixamos o threshold para IA ser mais proativa
            try:
                ia_ctx = criar_contexto(
                    lat=lat, lon=lon, mag_nt=mag_filtrado,
                    litologia=ctx.geologico.litologia.nome if ctx.geologico else '',
                    unidade=ctx.geologico.litologia.unidade if ctx.geologico else '',
                    dist_estrut=dist_estrut,
                    tipo_estrut=ctx.geologico.estrutura_mais_proxima.tipo if ctx.geologico and ctx.geologico.estrutura_mais_proxima else '',
                    cu=cu, au=au,
                    deposito=ctx.geologico.deposito_mais_proximo.nome if ctx.geologico and ctx.geologico.deposito_mais_proximo else '',
                    dist_dep=ctx.geologico.distancia_deposito_m() if ctx.geologico else 99999,
                    synapse=ctx.synapse_ajustado,
                    tier=ctx.risk_tier,
                    tipo_dep=ctx.metalogenico.tipo_mais_provavel.name if ctx.metalogenico else 'UNKNOWN',
                    score_metal=ctx.metalogenico.score_maximo if ctx.metalogenico else 0,
                    persistente=ctx.espacial.anomalia_persistente,
                    dias=ctx.espacial.dias_observados
                )
                ctx.interpretacao_ia = await interpretar_ponto(ia_ctx)
            except Exception as e:
                print(f"Erro na IA: {e}")
        
        # 8. Decisão Final
        ctx.decisao = self._decidir(ctx)
        
        return ctx
    
    def _calcular_synapse(self, ctx: WorldContext) -> float:
        score = 0.0
        
        # Componente Magnética (35%)
        mag = ctx.mag_nt_filtrado
        if 26000 <= mag <= 28500: # Faixa típica de IOCG em Carajás
            score += 35
        elif 28500 < mag <= 32000: # Anomalias fortes (BIF ou IOCG maciço)
            score += 30
        elif 24000 <= mag < 26000: # Anomalias de ouro orogênico
            score += 20
        else:
            score += 5
        
        # Componente Metalogênica (40%)
        if ctx.metalogenico:
            score += ctx.metalogenico.score_maximo * 40
        
        # Componente Geoquímica (25%)
        if ctx.geologico:
            geoq_score = 0
            if ctx.geologico.geoquimica.cu_ppm:
                if ctx.geologico.geoquimica.cu_ppm > 500: geoq_score += 15
                elif ctx.geologico.geoquimica.cu_ppm > 100: geoq_score += 8
            
            if ctx.geologico.geoquimica.au_ppb:
                if ctx.geologico.geoquimica.au_ppb > 100: geoq_score += 10
                elif ctx.geologico.geoquimica.au_ppb > 20: geoq_score += 5
            
            score += min(25, geoq_score)
        
        return min(100, max(0, score))
    
    def _classificar_tier(self, synapse: float) -> tuple:
        if synapse >= 85: return 'CRITICAL', 'T1'
        if synapse >= 70: return 'HIGH', 'T2'
        if synapse >= 50: return 'MEDIUM', 'T3'
        if synapse >= 30: return 'LOW', 'T4'
        return 'BACKGROUND', 'T5'
    
    def _decidir(self, ctx: WorldContext) -> DecisaoFinal:
        tier = ctx.risk_tier
        persistente = ctx.espacial.anomalia_persistente if ctx.espacial else False
        score_metal = ctx.metalogenico.score_maximo if ctx.metalogenico else 0
        tipo = ctx.metalogenico.tipo_mais_provavel if ctx.metalogenico else TipoDeposito.UNKNOWN
        
        if tier == 'CRITICAL':
            if persistente and score_metal > 0.8:
                return DecisaoFinal(
                    acao='PERFURAR_IMEDIATO',
                    justificativa=f'Alvo T1 de altíssima confiança. Assinatura {tipo.name} confirmada por persistência temporal.',
                    confianca=0.98,
                    prioridade=1
                )
            return DecisaoFinal(
                acao='DETALHAR_MALHA',
                justificativa='Alvo T1 detectado. Requer malha de 25m para locação de furo.',
                confianca=0.90,
                prioridade=1
            )
        
        if tier == 'HIGH':
            return DecisaoFinal(
                acao='AMOSTRAGEM_SOLO',
                justificativa=f'Alvo T2 compatível com {tipo.name}. Necessário validar extensão geoquímica.',
                confianca=0.80,
                prioridade=2
            )
        
        if tier == 'MEDIUM':
            return DecisaoFinal(
                acao='MONITORAR',
                justificativa='Anomalia T3. Manter monitoramento para verificar consistência.',
                confianca=0.65,
                prioridade=3
            )
        
        return DecisaoFinal(
            acao='ARQUIVAR',
            justificativa='Sem indicadores de mineralização econômica.',
            confianca=0.50,
            prioridade=5
        )

async def processar_ponto(lat: float, lon: float, mag_nt: float, usar_ia: bool = True) -> WorldContext:
    # Singleton-like para manter o buffer entre chamadas se necessário, 
    # mas aqui instanciamos por requisição para simplicidade.
    # Em produção, o WorldModel seria um serviço persistente.
    model = WorldModel(usar_ia=usar_ia)
    return await model.processar(lat, lon, mag_nt)
