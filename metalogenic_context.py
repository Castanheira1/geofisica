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
    intensidade_esperada: Tuple[float, float]
    anomalia_positiva: bool
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
        assinatura_mag=AssinaturaMagnetica((26000, 30000), True, "Forte anomalia positiva"),
        assinatura_geoq=AssinaturaGeoquimica((200, 10000), (50, 5000), (30, 65), ['Cu', 'Au', 'Fe', 'U', 'REE']),
        assinatura_estrut=AssinaturaEstrutural(['falha', 'cisalhamento'], [(290, 340), (110, 160)], 1000, 0.9),
        litologias_hospedeiras=[4, 5, 7],
        exemplos_carajas=['Salobo', 'Sossego', 'Cristalino']
    ),
    TipoDeposito.OROGENIC_GOLD: ModeloDeposito(
        tipo=TipoDeposito.OROGENIC_GOLD,
        nome="Orogenic Gold",
        assinatura_mag=AssinaturaMagnetica((24000, 26000), False, "Anomalia fraca/negativa"),
        assinatura_geoq=AssinaturaGeoquimica((10, 200), (100, 50000), (5, 20), ['Au', 'As', 'Sb', 'W']),
        assinatura_estrut=AssinaturaEstrutural(['cisalhamento'], [(0, 360)], 500, 0.95),
        litologias_hospedeiras=[4, 6, 7],
        exemplos_carajas=['Serra Pelada', 'Andorinhas']
    ),
    TipoDeposito.VMS: ModeloDeposito(
        tipo=TipoDeposito.VMS,
        nome="Volcanogenic Massive Sulfide",
        assinatura_mag=AssinaturaMagnetica((24500, 26500), False, "Anomalia fraca"),
        assinatura_geoq=AssinaturaGeoquimica((500, 20000), (10, 500), (20, 45), ['Cu', 'Zn', 'Pb', 'Ag']),
        assinatura_estrut=AssinaturaEstrutural(['sinclinal'], [(0, 360)], 2000, 0.5),
        litologias_hospedeiras=[7, 6],
        exemplos_carajas=['Pojuca', 'Gameleira']
    ),
    TipoDeposito.BIF_HOSTED: ModeloDeposito(
        tipo=TipoDeposito.BIF_HOSTED,
        nome="BIF-Hosted Iron",
        assinatura_mag=AssinaturaMagnetica((28000, 35000), True, "Anomalia muito forte"),
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
    mag_nt: float
    compatibilidades: List[CompatibilidadeDeposito] = field(default_factory=list)
    tipo_mais_provavel: TipoDeposito = TipoDeposito.UNKNOWN
    score_maximo: float = 0.0

class MotorMetalogenico:
    def __init__(self):
        self.modelos = MODELOS
    
    def _score_mag(self, mag_nt: float, modelo: ModeloDeposito) -> float:
        min_v, max_v = modelo.assinatura_mag.intensidade_esperada
        if min_v <= mag_nt <= max_v:
            return 1.0
        elif mag_nt < min_v:
            return max(0, 1.0 - (min_v - mag_nt) / 2000)
        else:
            return max(0, 1.0 - (mag_nt - max_v) / 2000)
    
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
        lat: float, lon: float, mag_nt: float,
        cu_ppm: Optional[float] = None,
        au_ppb: Optional[float] = None,
        litologia_codigo: int = 0,
        distancia_estrutura_m: float = 99999
    ) -> AnaliseMetalogenica:
        
        analise = AnaliseMetalogenica(latitude=lat, longitude=lon, mag_nt=mag_nt)
        tem_geoq = cu_ppm is not None or au_ppb is not None
        tem_estrut = distancia_estrutura_m < 50000
        
        for tipo, modelo in self.modelos.items():
            s_mag = self._score_mag(mag_nt, modelo)
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
                just.append(f"Mag compatível ({mag_nt:.0f}nT)")
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

def analisar_metalogenia(lat: float, lon: float, mag_nt: float, **kwargs) -> AnaliseMetalogenica:
    return MotorMetalogenico().analisar(lat, lon, mag_nt, **kwargs)
