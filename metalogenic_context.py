#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum, auto

class TipoDeposito(Enum):
    IOCG = auto()
    VMS = auto()
    OROGENIC_GOLD = auto()
    PORPHYRY = auto()
    BIF_HOSTED = auto()
    UNKNOWN = auto()

class NivelConfianca(Enum):
    MUITO_ALTO = 5
    ALTO = 4
    MODERADO = 3
    BAIXO = 2
    MUITO_BAIXO = 1

@dataclass
class AssinaturaMagnetica:
    # Faixa de AMPLITUDE da ANOMALIA RESIDUAL esperada em nT (campo total menos
    # baseline/IGRF) -- NÃO o campo total absoluto. Ver RELATORIO_TECNICO.md §1.
    anomalia_nt: Tuple[float, float]
    # 'positiva'  -> magnetita-dominante (máximo magnético)
    # 'negativa'  -> alteração hematítica / não-magnética (mínimo magnético)
    # 'variavel'  -> depende de magnetita vs hematita; exige gravimetria p/ desambiguar
    polaridade: str
    descricao: str

@dataclass
class AssinaturaGeoquimica:
    cu_ppm: Tuple[float, float]
    au_ppb: Tuple[float, float]
    fe_pct: Tuple[float, float]
    elementos_patfinder: List[str]

@dataclass
class AssinaturaEstrutural:
    tipos_favoraveis: List[str]
    azimutes_favoraveis: List[Tuple[float, float]]
    distancia_maxima_m: float
    importancia: float

@dataclass
class ModeloDeposito:
    tipo: TipoDeposito
    nome: str
    assinatura_mag: AssinaturaMagnetica
    assinatura_geoq: AssinaturaGeoquimica
    assinatura_estrut: AssinaturaEstrutural
    litologias_hospedeiras: List[int]
    exemplos_carajas: List[str]

MODELOS: Dict[TipoDeposito, ModeloDeposito] = {
    TipoDeposito.IOCG: ModeloDeposito(
        tipo=TipoDeposito.IOCG,
        nome="Iron Oxide Copper-Gold",
        # Magnetita-dominante (Salobo) = forte anomalia POSITIVA; hematita-rico
        # (Cristalino raso) pode ser MÍNIMO magnético sobre máximo gravimétrico.
        # Polaridade 'variavel' -> não penalizar mínimo; sinalizar necessidade de gravimetria.
        assinatura_mag=AssinaturaMagnetica((300, 8000), "variavel", "Anomalia forte (magnetita) ou discreta/negativa (hematita)"),
        assinatura_geoq=AssinaturaGeoquimica((200, 10000), (50, 5000), (30, 65), ['Cu', 'Au', 'Fe', 'U', 'REE', 'Co', 'Ag']),
        assinatura_estrut=AssinaturaEstrutural(['falha', 'cisalhamento'], [(290, 340), (110, 160)], 1000, 0.9),
        litologias_hospedeiras=[4, 5, 7],
        exemplos_carajas=['Salobo', 'Sossego', 'Cristalino']
    ),
    TipoDeposito.OROGENIC_GOLD: ModeloDeposito(
        tipo=TipoDeposito.OROGENIC_GOLD,
        nome="Orogenic Gold",
        assinatura_mag=AssinaturaMagnetica((0, 400), "negativa", "Anomalia fraca, frequentemente negativa"),
        assinatura_geoq=AssinaturaGeoquimica((10, 200), (100, 50000), (5, 20), ['Au', 'As', 'Sb', 'W', 'Te']),
        assinatura_estrut=AssinaturaEstrutural(['cisalhamento'], [(0, 360)], 500, 0.95),
        litologias_hospedeiras=[4, 6, 7],
        exemplos_carajas=['Serra Pelada', 'Andorinhas']
    ),
    TipoDeposito.VMS: ModeloDeposito(
        tipo=TipoDeposito.VMS,
        nome="Volcanogenic Massive Sulfide",
        assinatura_mag=AssinaturaMagnetica((0, 600), "variavel", "Anomalia fraca a moderada (depende de pirrotita/magnetita)"),
        assinatura_geoq=AssinaturaGeoquimica((500, 20000), (10, 500), (20, 45), ['Cu', 'Zn', 'Pb', 'Ag']),
        assinatura_estrut=AssinaturaEstrutural(['sinclinal'], [(0, 360)], 2000, 0.5),
        litologias_hospedeiras=[7, 6],
        exemplos_carajas=['Pojuca', 'Gameleira']
    ),
    TipoDeposito.BIF_HOSTED: ModeloDeposito(
        tipo=TipoDeposito.BIF_HOSTED,
        nome="BIF-Hosted Iron",
        # Jaspilito/itabirito: anomalia POSITIVA muito forte, contínua e estratiforme.
        assinatura_mag=AssinaturaMagnetica((2000, 30000), "positiva", "Anomalia muito forte, contínua e estratiforme"),
        assinatura_geoq=AssinaturaGeoquimica((0, 50), (0, 10), (50, 68), ['Fe', 'Mn', 'P']),
        assinatura_estrut=AssinaturaEstrutural(['sinclinal', 'dobra'], [(0, 360)], 5000, 0.3),
        litologias_hospedeiras=[4, 5],
        exemplos_carajas=['S11D', 'N4', 'N5']
    ),
}

@dataclass
class CompatibilidadeDeposito:
    tipo: TipoDeposito
    score_total: float
    score_mag: float
    score_geoq: float
    score_estrut: float
    score_lit: float
    confianca: NivelConfianca
    justificativa: str

@dataclass
class AnaliseMetalogenica:
    latitude: float
    longitude: float
    anomalia_nt: float            # anomalia residual usada na análise (não campo total)
    compatibilidades: List[CompatibilidadeDeposito] = field(default_factory=list)
    tipo_mais_provavel: TipoDeposito = TipoDeposito.UNKNOWN
    score_maximo: float = 0.0

class MotorMetalogenico:
    def __init__(self):
        self.modelos = MODELOS
    
    def _score_mag(self, anomalia_nt: float, modelo: ModeloDeposito) -> float:
        """Compara a ANOMALIA RESIDUAL (não o campo total) com a assinatura do
        modelo, respeitando a polaridade esperada. Ver RELATORIO_TECNICO.md §1/§4."""
        amp_min, amp_max = modelo.assinatura_mag.anomalia_nt
        polaridade = modelo.assinatura_mag.polaridade
        amp = abs(anomalia_nt)

        # Penalidade de polaridade: um forte máximo magnético argumenta contra um
        # alvo de polaridade 'negativa', e vice-versa. Alvos 'variavel' (ex.: IOCG,
        # que pode ser magnetita-alto OU hematita-baixo) não são penalizados pelo sinal.
        if polaridade == "positiva" and anomalia_nt < -amp_max * 0.25:
            return 0.1
        if polaridade == "negativa" and anomalia_nt > amp_max:
            return 0.2

        # Score pela amplitude da anomalia dentro da faixa esperada.
        if amp_min <= amp <= amp_max:
            return 1.0
        if amp < amp_min:
            faixa = max(amp_min, 1e-6)
            return max(0.2, amp / faixa)
        return max(0.0, 1.0 - (amp - amp_max) / max(amp_max, 1.0))
    
    def _score_geoq(self, cu: Optional[float], au: Optional[float], modelo: ModeloDeposito) -> float:
        if cu is None and au is None:
            return 0.5
        scores = []
        assin = modelo.assinatura_geoq
        if cu is not None:
            if assin.cu_ppm[0] <= cu <= assin.cu_ppm[1]:
                scores.append(1.0)
            elif cu > assin.cu_ppm[1]:
                scores.append(0.8)
            else:
                scores.append(0.3)
        if au is not None:
            if assin.au_ppb[0] <= au <= assin.au_ppb[1]:
                scores.append(1.0)
            elif au > assin.au_ppb[1]:
                scores.append(0.8)
            else:
                scores.append(0.3)
        return sum(scores) / len(scores) if scores else 0.5
    
    def _score_estrut(self, dist_m: float, modelo: ModeloDeposito) -> float:
        max_d = modelo.assinatura_estrut.distancia_maxima_m
        imp = modelo.assinatura_estrut.importancia
        if dist_m <= max_d:
            return (1.0 - dist_m / max_d * 0.3) * imp + (1 - imp) * 0.5
        else:
            return max(0, (1.0 - (dist_m - max_d) / 5000) * imp)
    
    def _score_lit(self, lit_codigo: int, modelo: ModeloDeposito) -> float:
        if lit_codigo in modelo.litologias_hospedeiras:
            return 1.0
        elif lit_codigo == 0:
            return 0.5
        else:
            return 0.3
    
    def analisar(
        self,
        lat: float, lon: float, anomalia_nt: float,
        cu_ppm: Optional[float] = None,
        au_ppb: Optional[float] = None,
        litologia_codigo: int = 0,
        distancia_estrutura_m: float = 99999
    ) -> AnaliseMetalogenica:

        analise = AnaliseMetalogenica(latitude=lat, longitude=lon, anomalia_nt=anomalia_nt)
        tem_geoq = cu_ppm is not None or au_ppb is not None
        tem_estrut = distancia_estrutura_m < 50000

        for tipo, modelo in self.modelos.items():
            s_mag = self._score_mag(anomalia_nt, modelo)
            s_geoq = self._score_geoq(cu_ppm, au_ppb, modelo)
            s_estrut = self._score_estrut(distancia_estrutura_m, modelo)
            s_lit = self._score_lit(litologia_codigo, modelo)
            
            w_mag, w_geoq, w_estrut, w_lit = 0.35, 0.30 if tem_geoq else 0.10, 0.20 if tem_estrut else 0.10, 0.15
            total_w = w_mag + w_geoq + w_estrut + w_lit
            score = (s_mag*w_mag + s_geoq*w_geoq + s_estrut*w_estrut + s_lit*w_lit) / total_w
            
            if score >= 0.85 and tem_geoq:
                conf = NivelConfianca.MUITO_ALTO
            elif score >= 0.70:
                conf = NivelConfianca.ALTO
            elif score >= 0.55:
                conf = NivelConfianca.MODERADO
            elif score >= 0.40:
                conf = NivelConfianca.BAIXO
            else:
                conf = NivelConfianca.MUITO_BAIXO
            
            just = []
            if s_mag > 0.7:
                just.append(f"Anomalia mag. compatível ({anomalia_nt:+.0f}nT)")
            if s_geoq > 0.7 and tem_geoq:
                just.append("Geoquímica favorável")
            if s_estrut > 0.7:
                just.append(f"Estrutura próxima ({distancia_estrutura_m:.0f}m)")
            if s_lit > 0.7:
                just.append("Litologia hospedeira")
            
            analise.compatibilidades.append(CompatibilidadeDeposito(
                tipo=tipo, score_total=score,
                score_mag=s_mag, score_geoq=s_geoq, score_estrut=s_estrut, score_lit=s_lit,
                confianca=conf, justificativa="; ".join(just) if just else "Compatibilidade parcial"
            ))
        
        analise.compatibilidades.sort(key=lambda x: x.score_total, reverse=True)
        if analise.compatibilidades:
            analise.tipo_mais_provavel = analise.compatibilidades[0].tipo
            analise.score_maximo = analise.compatibilidades[0].score_total
        
        return analise

def analisar_metalogenia(lat: float, lon: float, anomalia_nt: float, **kwargs) -> AnaliseMetalogenica:
    """`anomalia_nt` = anomalia magnética RESIDUAL (campo total - baseline), não o campo total."""
    return MotorMetalogenico().analisar(lat, lon, anomalia_nt, **kwargs)
