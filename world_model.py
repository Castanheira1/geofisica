#!/usr/bin/env python3

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

logger = logging.getLogger("prospector.world")

from geological_layers import obter_contexto_geologico, ContextoGeologico, TipoLitologia
from metalogenic_context import analisar_metalogenia, AnaliseMetalogenica, TipoDeposito
from spatial_memory import memoria, AnaliseEspacial
from ai_interpreter import interpretar_ponto, criar_contexto, InterpretacaoGeologica
from mag_preprocessing import MagPreProcessor, MagPoint
from mag_anomaly import anomalia_residual, AnomaliaPontual

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

    # Anomalia magnética RESIDUAL (campo total - baseline). É esta, não o campo
    # total absoluto, que é interpretável (ver RELATORIO_TECNICO.md §1).
    anomalia_nt: float = 0.0
    anomalia_confiavel: bool = False
    baseline_metodo: str = 'indisponivel'

    geologico: Optional[ContextoGeologico] = None
    metalogenico: Optional[AnaliseMetalogenica] = None
    espacial: Optional[AnaliseEspacial] = None
    interpretacao_ia: Optional[InterpretacaoGeologica] = None

    # Favorabilidade RELATIVA (0-100). Não é probabilidade de minério nem veredito.
    synapse_index: float = 0.0
    synapse_ajustado: float = 0.0
    risk_tier: str = 'BACKGROUND'
    tier_code: str = 'T5'

    # Honestidade sobre o dado: quanto da evidência esperada está realmente
    # presente (0-1) e a incerteza resultante (0-1, 1 = máxima).
    completude_dados: float = 0.0
    incerteza: float = 1.0

    decisao: Optional[DecisaoFinal] = None

    def resumo(self) -> dict:
        return {
            'id': f"PT-{self.timestamp}",
            'lat': self.latitude,
            'lon': self.longitude,
            'proprio_mag_nt': self.mag_nt_proprio,
            'filtrado_mag_nt': self.mag_nt_filtrado,
            'anomalia_nt': self.anomalia_nt,
            'anomalia_confiavel': self.anomalia_confiavel,
            'baseline_metodo': self.baseline_metodo,
            'cprm_litologia': self.geologico.litologia.nome if self.geologico else None,
            'cprm_cu_ppm': self.geologico.geoquimica.cu_ppm if self.geologico else None,
            'cprm_au_ppb': self.geologico.geoquimica.au_ppb if self.geologico else None,
            'dist_estrutura_m': self.geologico.distancia_estrutura_m() if self.geologico else None,
            'tipo_deposito': self.metalogenico.tipo_mais_provavel.name if self.metalogenico else None,
            'score_metalogenico': self.metalogenico.score_maximo if self.metalogenico else 0,
            'anomalia_persistente': self.espacial.anomalia_persistente if self.espacial else False,
            'favorabilidade_relativa': self.synapse_index,
            'synapse_index': self.synapse_index,
            'synapse_ajustado': self.synapse_ajustado,
            'risk_tier': self.risk_tier,
            'tier_code': self.tier_code,
            'completude_dados': self.completude_dados,
            'incerteza': self.incerteza,
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

        # 1b. Anomalia RESIDUAL. O baseline regional é estimado pela mediana do
        # buffer local; isto realça variações de curto comprimento de onda, mas
        # NÃO substitui a remoção do IGRF (sinalizado em baseline_metodo). É a
        # anomalia, não o campo total, que entra no resto da análise.
        janela = [p.mag_nt for p in self.buffer_pontos]
        anomalia = anomalia_residual(mag_filtrado, janela_local_nt=janela)

        ctx = WorldContext(
            latitude=lat,
            longitude=lon,
            mag_nt_proprio=mag_nt,
            mag_nt_filtrado=mag_filtrado,
            anomalia_nt=anomalia.anomalia_nt,
            anomalia_confiavel=anomalia.confiavel,
            baseline_metodo=anomalia.baseline_metodo,
            timestamp=timestamp or str(int(asyncio.get_event_loop().time()))
        )

        # 2. Enriquecimento Geológico (CPRM)
        ctx.geologico = await obter_contexto_geologico(lat, lon)

        # 3. Análise Metalogênica (sobre a ANOMALIA, não o campo total)
        lit_codigo = int(ctx.geologico.litologia.codigo) if ctx.geologico else 0
        dist_estrut = ctx.geologico.distancia_estrutura_m() if ctx.geologico else 99999
        cu = ctx.geologico.geoquimica.cu_ppm if ctx.geologico else None
        au = ctx.geologico.geoquimica.au_ppb if ctx.geologico else None

        ctx.metalogenico = analisar_metalogenia(
            lat, lon, ctx.anomalia_nt,
            cu_ppm=cu, au_ppb=au,
            litologia_codigo=lit_codigo,
            distancia_estrutura_m=dist_estrut
        )
        
        # 3b. Completude do dado e incerteza (honestidade): quanto da evidência
        # esperada está realmente presente. Menos dado -> mais incerteza.
        ctx.completude_dados = self._avaliar_completude(ctx, cu, au, dist_estrut)
        ctx.incerteza = self._calcular_incerteza(ctx)

        # 4. Favorabilidade relativa (NÃO é probabilidade de minério)
        ctx.synapse_index = self._calcular_synapse(ctx)

        # 5. Memória Espacial e Persistência
        ctx.espacial = memoria.analisar(lat, lon, mag_filtrado, ctx.synapse_index)
        
        # Ajuste por persistência temporal
        ctx.synapse_ajustado = ctx.synapse_index * ctx.espacial.fator_confianca
        ctx.synapse_ajustado = min(100, max(0, ctx.synapse_ajustado))
        
        # 6. Classificação de Risco
        ctx.risk_tier, ctx.tier_code = self._classificar_tier(ctx.synapse_ajustado)

        # Recalcula a incerteza agora que a persistência espacial é conhecida.
        ctx.incerteza = self._calcular_incerteza(ctx)

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
            except Exception:
                logger.exception("Erro na interpretação por IA")
        
        # 8. Decisão Final
        ctx.decisao = self._decidir(ctx)
        
        return ctx
    
    def _avaliar_completude(self, ctx: WorldContext, cu, au, dist_estrut: float) -> float:
        """Fração (0-1) da evidência esperada que está realmente presente.
        Honestidade: pontuar alto com pouco dado é enganoso."""
        presentes = 0
        total = 4
        if ctx.anomalia_confiavel:                 # baseline magnético decente
            presentes += 1
        if (cu is not None) or (au is not None):   # geoquímica
            presentes += 1
        if dist_estrut is not None and dist_estrut < 50000:  # estrutura mapeada
            presentes += 1
        if ctx.geologico and ctx.geologico.litologia.codigo:  # litologia conhecida
            presentes += 1
        return round(presentes / total, 3)

    def _calcular_incerteza(self, ctx: WorldContext) -> float:
        """Incerteza (0-1, 1 = máxima). Cai com mais dado e com persistência
        temporal (evidência repetida em dias distintos)."""
        inc = 1.0 - 0.85 * ctx.completude_dados
        if not ctx.anomalia_confiavel:
            inc = min(1.0, inc + 0.1)
        if ctx.espacial and getattr(ctx.espacial, 'anomalia_persistente', False):
            inc *= 0.7
        return round(min(1.0, max(0.05, inc)), 3)

    def _calcular_synapse(self, ctx: WorldContext) -> float:
        """Favorabilidade RELATIVA (0-100) para priorização — NÃO é probabilidade
        de minério nem veredito. Usa a ANOMALIA residual (via análise metalogênica,
        que já compara anomalia + polaridade) e a geoquímica, amortecido pela
        completude do dado. Ver RELATORIO_TECNICO.md §1/§2."""
        # Componente metalogênica (anomalia + estrutura + litologia já ponderadas): 65%
        s_metal = ctx.metalogenico.score_maximo if ctx.metalogenico else 0.0
        score = s_metal * 65.0

        # Componente geoquímica direta: 35%
        if ctx.geologico:
            geoq = 0.0
            cu = ctx.geologico.geoquimica.cu_ppm
            au = ctx.geologico.geoquimica.au_ppb
            if cu:
                if cu > 500: geoq += 20
                elif cu > 100: geoq += 10
            if au:
                if au > 100: geoq += 15
                elif au > 20: geoq += 7
            score += min(35.0, geoq)

        # Amortecimento por completude: com pouco dado, puxa a favorabilidade
        # em direção ao "background" para não fabricar confiança.
        score *= (0.55 + 0.45 * ctx.completude_dados)
        return min(100.0, max(0.0, score))
    
    def _classificar_tier(self, synapse: float) -> tuple:
        if synapse >= 85: return 'CRITICAL', 'T1'
        if synapse >= 70: return 'HIGH', 'T2'
        if synapse >= 50: return 'MEDIUM', 'T3'
        if synapse >= 30: return 'LOW', 'T4'
        return 'BACKGROUND', 'T5'
    
    def _decidir(self, ctx: WorldContext) -> DecisaoFinal:
        """Recomenda o PRÓXIMO PASSO DE EXPLORAÇÃO (triagem/priorização), nunca um
        veredito de furo. A confiança é derivada da incerteza real do dado, não
        fixada. Furo só após validação multi-método (mag + gravimetria + geoquímica
        + estrutura) — ver RELATORIO_TECNICO.md §4 (magnetometria isolada é ambígua)."""
        tier = ctx.risk_tier
        persistente = ctx.espacial.anomalia_persistente if ctx.espacial else False
        tipo = ctx.metalogenico.tipo_mais_provavel if ctx.metalogenico else TipoDeposito.UNKNOWN

        # Confiança honesta: ligada à completude do dado e à persistência.
        confianca = round(1.0 - ctx.incerteza, 2)

        # Ressalvas sobre o que falta para subir a confiança.
        ressalvas = []
        if not ctx.anomalia_confiavel:
            ressalvas.append("anomalia mag. com baseline aproximado (sem IGRF)")
        if tipo == TipoDeposito.IOCG:
            ressalvas.append("IOCG pode ser máximo (magnetita) OU mínimo (hematita) magnético — requer gravimetria")
        if ctx.completude_dados < 0.75:
            ressalvas.append(f"completude de dados {ctx.completude_dados:.0%}")
        nota = (" Ressalvas: " + "; ".join(ressalvas) + ".") if ressalvas else ""

        if tier == 'CRITICAL':
            return DecisaoFinal(
                acao='PRIORIDADE_ALTA_VALIDAR',
                justificativa=(f'Alvo T1 prioritário, assinatura compatível com {tipo.name}. '
                               f'Validar com gravimetria + magnetometria terrestre + amostragem antes de locar furo.'
                               f'{nota}'),
                confianca=confianca,
                prioridade=1
            )

        if tier == 'HIGH':
            return DecisaoFinal(
                acao='AMOSTRAGEM_SOLO',
                justificativa=(f'Alvo T2 compatível com {tipo.name}. Validar extensão geoquímica '
                               f'e detalhar geofísica.{nota}'),
                confianca=confianca,
                prioridade=2
            )

        if tier == 'MEDIUM':
            return DecisaoFinal(
                acao='MONITORAR',
                justificativa=f'Anomalia T3. Monitorar e adquirir dado complementar para reduzir incerteza.{nota}',
                confianca=confianca,
                prioridade=3
            )

        return DecisaoFinal(
            acao='BAIXA_PRIORIDADE',
            justificativa=f'Sem indicadores robustos de mineralização econômica na evidência disponível.{nota}',
            confianca=confianca,
            prioridade=5
        )

_MODELO_SINGLETON: Optional[WorldModel] = None


async def processar_ponto(lat: float, lon: float, mag_nt: float, usar_ia: bool = True) -> WorldContext:
    """Mantém um WorldModel por PROCESSO para que o buffer de micronivelamento e a
    memória espacial PERSISTAM entre requisições (antes recriava-se a cada chamada,
    zerando o buffer e a persistência temporal).

    Limitação conhecida (Fase 2): este estado vive em memória do processo — com
    múltiplos workers ou após restart ele se perde. Para produção, externalizar o
    buffer/memória (ex.: banco/Redis). Ver RELATORIO_TECNICO.md §6.
    """
    global _MODELO_SINGLETON
    if _MODELO_SINGLETON is None or _MODELO_SINGLETON.usar_ia != usar_ia:
        _MODELO_SINGLETON = WorldModel(usar_ia=usar_ia)
    return await _MODELO_SINGLETON.processar(lat, lon, mag_nt)
