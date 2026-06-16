#!/usr/bin/env python3

import httpx
import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from enum import IntEnum
import math

logger = logging.getLogger("prospector.geosgb")

# Endpoints do GeoSGB. ATENÇÃO: os nomes de camada (typeName) abaixo NÃO foram
# verificados contra o serviço real (ver RELATORIO_TECNICO.md §5). Por isso o
# cliente DESCOBRE as camadas em runtime via GetCapabilities e só recorre aos
# palpites como último recurso — emitindo aviso explícito em vez de falhar calado.
GEOSGB_WFS = "https://geosgb.cprm.gov.br/geosgb/gs/ows"
GEOSGB_WCS = "https://geosgb.cprm.gov.br/geosgb/gs/wcs"

class TipoLitologia(IntEnum):
    DESCONHECIDO = 0
    COBERTURA = 1
    GRANITO = 2
    GABRO_DIORITO = 3
    BIF_ITABIRITO = 4
    JASPILITO = 5
    METASSEDIMENTO = 6
    METABASALTO = 7
    BRECHA = 8
    FORMACAO_AGUAS_CLARAS = 9
    COMPLEXO_XINGU = 10

LITOLOGIA_KEYWORDS = {
    'cobertura': TipoLitologia.COBERTURA,
    'laterit': TipoLitologia.COBERTURA,
    'aluvio': TipoLitologia.COBERTURA,
    'granito': TipoLitologia.GRANITO,
    'granodiorito': TipoLitologia.GRANITO,
    'monzogranito': TipoLitologia.GRANITO,
    'gabro': TipoLitologia.GABRO_DIORITO,
    'diorito': TipoLitologia.GABRO_DIORITO,
    'formação ferrífera': TipoLitologia.BIF_ITABIRITO,
    'bif': TipoLitologia.BIF_ITABIRITO,
    'itabirito': TipoLitologia.BIF_ITABIRITO,
    'jaspilito': TipoLitologia.JASPILITO,
    'metassedimento': TipoLitologia.METASSEDIMENTO,
    'metarenito': TipoLitologia.METASSEDIMENTO,
    'metapelito': TipoLitologia.METASSEDIMENTO,
    'metabasalto': TipoLitologia.METABASALTO,
    'basalto': TipoLitologia.METABASALTO,
    'anfibolito': TipoLitologia.METABASALTO,
    'brecha': TipoLitologia.BRECHA,
    'águas claras': TipoLitologia.FORMACAO_AGUAS_CLARAS,
    'xingu': TipoLitologia.COMPLEXO_XINGU,
}

@dataclass
class Litologia:
    codigo: TipoLitologia = TipoLitologia.DESCONHECIDO
    nome: str = ''
    unidade: str = ''
    idade: str = ''
    provincia: str = ''

@dataclass
class Estrutura:
    tipo: str = ''
    nome: str = ''
    azimute: float = 0.0
    distancia_m: float = 0.0
    geometria: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class Deposito:
    nome: str = ''
    substancia: str = ''
    tipo: str = ''
    status: str = ''
    distancia_m: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0

@dataclass
class MagRegional:
    valor_nt: float = 0.0
    anomalia_nt: float = 0.0
    gradiente: float = 0.0
    fonte: str = 'cprm_aeromag'

@dataclass
class Geoquimica:
    cu_ppm: Optional[float] = None
    au_ppb: Optional[float] = None
    fe_pct: Optional[float] = None
    ag_ppm: Optional[float] = None
    as_ppm: Optional[float] = None
    distancia_m: float = 0.0
    amostra_id: str = ''

@dataclass
class ContextoGeologico:
    latitude: float
    longitude: float
    litologia: Litologia = field(default_factory=Litologia)
    estruturas: List[Estrutura] = field(default_factory=list)
    depositos: List[Deposito] = field(default_factory=list)
    mag_regional: MagRegional = field(default_factory=MagRegional)
    geoquimica: Geoquimica = field(default_factory=Geoquimica)
    estrutura_mais_proxima: Optional[Estrutura] = None
    deposito_mais_proximo: Optional[Deposito] = None
    
    def distancia_estrutura_m(self) -> float:
        if self.estrutura_mais_proxima:
            return self.estrutura_mais_proxima.distancia_m
        return 99999.0
    
    def distancia_deposito_m(self) -> float:
        if self.deposito_mais_proximo:
            return self.deposito_mais_proximo.distancia_m
        return 99999.0

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def classificar_litologia(texto: str) -> TipoLitologia:
    if not texto:
        return TipoLitologia.DESCONHECIDO
    texto_lower = texto.lower()
    for keyword, tipo in LITOLOGIA_KEYWORDS.items():
        if keyword in texto_lower:
            return tipo
    return TipoLitologia.DESCONHECIDO

class GeoSGBClient:
    # Cache de camadas disponíveis por processo. None = ainda não consultado;
    # set() vazio = GetCapabilities inacessível (não tenta mais nesta sessão).
    _typenames_cache: Optional[Set[str]] = None

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    def _bbox(self, lat: float, lon: float, buffer: float) -> str:
        return f"{lon-buffer},{lat-buffer},{lon+buffer},{lat+buffer}"

    async def _typenames_disponiveis(self) -> Set[str]:
        """Lê os FeatureTypes reais do serviço via GetCapabilities (cache por processo).
        Evita depender de nomes de camada chutados. Retorna set vazio se inacessível."""
        if GeoSGBClient._typenames_cache is not None:
            return GeoSGBClient._typenames_cache

        nomes: Set[str] = set()
        params = {"service": "WFS", "version": "2.0.0", "request": "GetCapabilities"}
        try:
            resp = await self.client.get(GEOSGB_WFS, params=params)
            if resp.status_code == 200:
                # Captura <Name>...</Name> dos FeatureType (com ou sem namespace prefix).
                nomes = set(re.findall(r"<(?:\w+:)?Name>\s*([^<\s][^<]*?)\s*</(?:\w+:)?Name>", resp.text))
            else:
                logger.warning("GeoSGB GetCapabilities HTTP %s", resp.status_code)
        except Exception as e:
            logger.warning("GeoSGB GetCapabilities inacessível: %s", e)

        GeoSGBClient._typenames_cache = nomes
        if not nomes:
            logger.warning("GeoSGB: nenhuma camada descoberta; usando palpites de typeName (podem retornar vazio).")
        return nomes

    async def _resolver_typenames(self, candidatos: List[str], palavras: List[str]) -> List[str]:
        """Resolve a lista de typeNames a consultar: prioriza camadas REAIS do serviço
        cujo nome contém as palavras-chave; cai nos palpites apenas se a descoberta
        falhar. Assim erros de nome viram aviso, não silêncio."""
        disponiveis = await self._typenames_disponiveis()
        if not disponiveis:
            return candidatos  # descoberta indisponível -> tenta palpites

        reais = [n for n in disponiveis if any(p in n.lower() for p in palavras)]
        if not reais:
            logger.warning("GeoSGB: nenhuma camada casa com %s. Disponíveis (amostra): %s",
                           palavras, sorted(disponiveis)[:15])
        return reais or candidatos
    
    async def _wfs_query(self, typename: str, bbox: str) -> List[dict]:
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": typename,
            "outputFormat": "application/json",
            "bbox": bbox,
            "srsName": "EPSG:4326"
        }
        try:
            resp = await self.client.get(GEOSGB_WFS, params=params)
            if resp.status_code == 200:
                data = resp.json()
                return data.get('features', [])
        except:
            pass
        return []
    
    async def buscar_litologia(self, lat: float, lon: float) -> Litologia:
        bbox = self._bbox(lat, lon, 0.005)
        typenames = await self._resolver_typenames(
            ["geosgb:unidades_litoestratigraficas", "geosgb:geologia"],
            ["litoestrat", "litolog", "geolog", "unidade"],
        )
        features = []
        for tn in typenames:
            features = await self._wfs_query(tn, bbox)
            if features:
                break

        if not features:
            return Litologia()
        
        props = features[0].get('properties', {})
        nome_lit = props.get('litologia_principal', '') or props.get('litologia', '') or props.get('descricao', '')
        
        return Litologia(
            codigo=classificar_litologia(nome_lit),
            nome=nome_lit,
            unidade=props.get('unidade', '') or props.get('nome_unidade', ''),
            idade=props.get('idade', '') or props.get('eon_era', ''),
            provincia=props.get('provincia_estrutural', '') or props.get('dominio', '')
        )
    
    async def buscar_estruturas(self, lat: float, lon: float, raio_km: float = 5.0) -> List[Estrutura]:
        buffer = raio_km / 111.0
        bbox = self._bbox(lat, lon, buffer)
        
        estruturas = []

        typenames = await self._resolver_typenames(
            ["geosgb:estruturas_geologicas", "geosgb:lineamentos", "geosgb:falhas"],
            ["estrutur", "lineament", "falha", "cisalh"],
        )
        for typename in typenames:
            features = await self._wfs_query(typename, bbox)
            
            for f in features:
                props = f.get('properties', {})
                geom = f.get('geometry', {})
                coords = geom.get('coordinates', [])
                
                min_dist = float('inf')
                pontos = []
                
                if geom.get('type') == 'LineString':
                    for coord in coords:
                        if len(coord) >= 2:
                            d = haversine(lat, lon, coord[1], coord[0])
                            min_dist = min(min_dist, d)
                            pontos.append((coord[0], coord[1]))
                elif geom.get('type') == 'MultiLineString':
                    for line in coords:
                        for coord in line:
                            if len(coord) >= 2:
                                d = haversine(lat, lon, coord[1], coord[0])
                                min_dist = min(min_dist, d)
                                pontos.append((coord[0], coord[1]))
                
                if min_dist < raio_km * 1000:
                    estruturas.append(Estrutura(
                        tipo=props.get('tipo', '') or props.get('tipo_estrutura', ''),
                        nome=props.get('nome', '') or props.get('denominacao', ''),
                        azimute=float(props.get('azimute', 0) or props.get('direcao', 0) or 0),
                        distancia_m=min_dist,
                        geometria=pontos
                    ))
        
        estruturas.sort(key=lambda x: x.distancia_m)
        return estruturas[:10]
    
    async def buscar_depositos(self, lat: float, lon: float, raio_km: float = 10.0) -> List[Deposito]:
        buffer = raio_km / 111.0
        bbox = self._bbox(lat, lon, buffer)
        
        depositos = []

        typenames = await self._resolver_typenames(
            ["geosgb:depositos_minerais", "geosgb:ocorrencias_minerais", "geosgb:recursos_minerais"],
            ["deposit", "ocorrenc", "recurso", "mineral"],
        )
        for typename in typenames:
            features = await self._wfs_query(typename, bbox)
            
            for f in features:
                props = f.get('properties', {})
                geom = f.get('geometry', {})
                coords = geom.get('coordinates', [])
                
                if len(coords) >= 2:
                    dep_lat, dep_lon = coords[1], coords[0]
                    dist = haversine(lat, lon, dep_lat, dep_lon)
                    
                    if dist < raio_km * 1000:
                        depositos.append(Deposito(
                            nome=props.get('nome', '') or props.get('denominacao', ''),
                            substancia=props.get('substancia', '') or props.get('substancia_principal', ''),
                            tipo=props.get('tipo_deposito', '') or props.get('modelo', ''),
                            status=props.get('situacao', '') or props.get('status', ''),
                            distancia_m=dist,
                            latitude=dep_lat,
                            longitude=dep_lon
                        ))
        
        depositos.sort(key=lambda x: x.distancia_m)
        return depositos[:10]
    
    async def buscar_geoquimica(self, lat: float, lon: float, raio_km: float = 2.0) -> Geoquimica:
        buffer = raio_km / 111.0
        bbox = self._bbox(lat, lon, buffer)
        
        typenames = await self._resolver_typenames(
            ["geosgb:geoquimica_sedimento_corrente", "geosgb:geoquimica_solo", "geosgb:geoquimica"],
            ["geoquim", "sedimento", "solo"],
        )
        for typename in typenames:
            features = await self._wfs_query(typename, bbox)

            if features:
                closest = None
                min_dist = float('inf')
                
                for f in features:
                    geom = f.get('geometry', {})
                    coords = geom.get('coordinates', [])
                    if len(coords) >= 2:
                        d = haversine(lat, lon, coords[1], coords[0])
                        if d < min_dist:
                            min_dist = d
                            closest = f
                
                if closest and min_dist < raio_km * 1000:
                    props = closest.get('properties', {})
                    return Geoquimica(
                        cu_ppm=self._safe_float(props.get('cu_ppm') or props.get('cu')),
                        au_ppb=self._safe_float(props.get('au_ppb') or props.get('au')),
                        fe_pct=self._safe_float(props.get('fe_pct') or props.get('fe')),
                        ag_ppm=self._safe_float(props.get('ag_ppm') or props.get('ag')),
                        as_ppm=self._safe_float(props.get('as_ppm') or props.get('as')),
                        distancia_m=min_dist,
                        amostra_id=props.get('id_amostra', '') or props.get('codigo', '')
                    )
        
        return Geoquimica()
    
    def _safe_float(self, val) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except:
            return None

async def obter_contexto_geologico(lat: float, lon: float) -> ContextoGeologico:
    ctx = ContextoGeologico(latitude=lat, longitude=lon)
    
    async with GeoSGBClient() as client:
        litologia, estruturas, depositos, geoquimica = await asyncio.gather(
            client.buscar_litologia(lat, lon),
            client.buscar_estruturas(lat, lon),
            client.buscar_depositos(lat, lon),
            client.buscar_geoquimica(lat, lon),
            return_exceptions=True
        )
        
        if isinstance(litologia, Litologia):
            ctx.litologia = litologia
        if isinstance(estruturas, list) and estruturas:
            ctx.estruturas = estruturas
            ctx.estrutura_mais_proxima = estruturas[0]
        if isinstance(depositos, list) and depositos:
            ctx.depositos = depositos
            ctx.deposito_mais_proximo = depositos[0]
        if isinstance(geoquimica, Geoquimica):
            ctx.geoquimica = geoquimica
    
    return ctx

if __name__ == '__main__':
    async def test():
        ctx = await obter_contexto_geologico(-6.0652, -50.1836)
        print(f"Litologia: {ctx.litologia.nome} ({ctx.litologia.codigo.name})")
        print(f"Unidade: {ctx.litologia.unidade}")
        print(f"Estruturas: {len(ctx.estruturas)}")
        if ctx.estrutura_mais_proxima:
            print(f"  Mais próxima: {ctx.estrutura_mais_proxima.tipo} a {ctx.estrutura_mais_proxima.distancia_m:.0f}m")
        print(f"Depósitos: {len(ctx.depositos)}")
        if ctx.deposito_mais_proximo:
            print(f"  Mais próximo: {ctx.deposito_mais_proximo.nome} ({ctx.deposito_mais_proximo.substancia}) a {ctx.deposito_mais_proximo.distancia_m:.0f}m")
        if ctx.geoquimica.cu_ppm:
            print(f"Geoquímica: Cu={ctx.geoquimica.cu_ppm}ppm, Au={ctx.geoquimica.au_ppb}ppb")
    
    asyncio.run(test())
