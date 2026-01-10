#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import json
from pathlib import Path

@dataclass
class LeituraHistorica:
    id: str
    timestamp: datetime
    latitude: float
    longitude: float
    mag_nt: float
    synapse_index: float
    risk_tier: str

@dataclass
class CelulaEspacial:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    leituras: List[LeituraHistorica] = field(default_factory=list)
    
    def centro(self) -> Tuple[float, float]:
        return ((self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2)
    
    def adicionar(self, leitura: LeituraHistorica):
        self.leituras.append(leitura)
    
    def media_mag(self) -> float:
        return sum(l.mag_nt for l in self.leituras) / len(self.leituras) if self.leituras else 0
    
    def std_mag(self) -> float:
        if len(self.leituras) < 2:
            return 0
        m = self.media_mag()
        return (sum((l.mag_nt - m)**2 for l in self.leituras) / len(self.leituras))**0.5
    
    def media_synapse(self) -> float:
        return sum(l.synapse_index for l in self.leituras) / len(self.leituras) if self.leituras else 0
    
    def dias_distintos(self) -> int:
        return len(set(l.timestamp.date() for l in self.leituras)) if self.leituras else 0

@dataclass
class AnaliseEspacial:
    latitude: float
    longitude: float
    leituras_anteriores: int
    dias_observados: int
    media_historica_mag: float
    std_historica_mag: float
    media_historica_synapse: float
    anomalia_persistente: bool
    consistencia: float
    tendencia: str
    fator_confianca: float

class MemoriaEspacial:
    def __init__(self, resolucao: float = 0.001):
        self.resolucao = resolucao
        self.celulas: Dict[str, CelulaEspacial] = {}
    
    def _chave(self, lat: float, lon: float) -> str:
        return f"{int(lat/self.resolucao)}_{int(lon/self.resolucao)}"
    
    def registrar(self, id: str, lat: float, lon: float, mag_nt: float, synapse: float, tier: str):
        chave = self._chave(lat, lon)
        if chave not in self.celulas:
            lat_idx, lon_idx = int(lat/self.resolucao), int(lon/self.resolucao)
            self.celulas[chave] = CelulaEspacial(
                lat_idx*self.resolucao, (lat_idx+1)*self.resolucao,
                lon_idx*self.resolucao, (lon_idx+1)*self.resolucao
            )
        self.celulas[chave].adicionar(LeituraHistorica(id, datetime.utcnow(), lat, lon, mag_nt, synapse, tier))
    
    def analisar(self, lat: float, lon: float, mag_atual: float, synapse_atual: float) -> AnaliseEspacial:
        chave = self._chave(lat, lon)
        cel = self.celulas.get(chave)
        
        if not cel or not cel.leituras:
            return AnaliseEspacial(lat, lon, 0, 0, 0, 0, 0, False, 0.5, 'indeterminado', 1.0)
        
        media_mag, std_mag = cel.media_mag(), cel.std_mag()
        media_syn = cel.media_synapse()
        total, dias = len(cel.leituras), cel.dias_distintos()
        
        consistencia = max(0, 1 - abs(mag_atual - media_mag) / (std_mag * 3)) if std_mag > 0 else 0.8
        anomalia = dias >= 2 and media_syn >= 50 and consistencia >= 0.6
        
        if total >= 3:
            ordenadas = sorted(cel.leituras, key=lambda x: x.timestamp)
            m1 = sum(l.synapse_index for l in ordenadas[:len(ordenadas)//2]) / (len(ordenadas)//2)
            m2 = sum(l.synapse_index for l in ordenadas[len(ordenadas)//2:]) / (len(ordenadas) - len(ordenadas)//2)
            tendencia = 'crescente' if m2 - m1 > 10 else 'decrescente' if m1 - m2 > 10 else 'estável'
        else:
            tendencia = 'insuficiente'
        
        fator = 1.3 if anomalia and dias >= 3 else 1.15 if anomalia else 0.8 if consistencia < 0.4 else 1.0
        
        return AnaliseEspacial(lat, lon, total, dias, media_mag, std_mag, media_syn, anomalia, consistencia, tendencia, fator)
    
    def hotspots(self, min_synapse: float = 70, min_dias: int = 2) -> List[CelulaEspacial]:
        return sorted([c for c in self.celulas.values() if c.media_synapse() >= min_synapse and c.dias_distintos() >= min_dias], key=lambda x: x.media_synapse(), reverse=True)

memoria = MemoriaEspacial()
