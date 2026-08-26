#!/usr/bin/env python3
"""
PROSPECTOR-AI — Cliente das camadas geológicas públicas do Serviço Geológico do
Brasil (SGB, ex-CPRM), via WFS/OGC.

ESTADO DOS ENDPOINTS (verificado em 2026-08-26 contra o serviço real):

  * O host antigo `geosgb.cprm.gov.br` está MORTO (connection reset). A CPRM foi
    renomeada para SGB e o serviço migrou. O `RELATORIO_TECNICO.md` §5 registrava
    os nomes de camada como "NÃO verificados" por suposto bloqueio de egress —
    o problema era o endereço, não a rede.
  * O endpoint vivo é `https://opendata.sgb.gov.br/geoserver/ows` (GeoServer,
    344 FeatureTypes no GetCapabilities).

ARMADILHAS DO WFS 2.0 QUE CAUSAVAM FALHA SILENCIOSA (HTTP 200 + zero feature):

  1. `typeName` (singular) é sintaxe WFS 1.x. O WFS 2.0 exige `typeNames`.
  2. `bbox` sem CRS explícito é interpretado em EPSG:4326, cuja ordem de eixos é
     (lat, lon) — invertida em relação ao (lon, lat) usual. Passamos o CRS84
     explicitamente no próprio bbox para fixar a ordem (lon, lat).
  3. Exceções do GeoServer chegam como XML com HTTP 200; `resp.json()` estoura e,
     no código anterior, era engolido por um `except: pass`.

Comparação medida na mesma camada e mesma bbox (Carajás):
    forma antiga  -> HTTP 200, 0 features
    forma correta -> HTTP 200, 345 features

LIMITAÇÃO DE ESCALA (importante para não repetir over-claim):
  As camadas de litoestratigrafia e estruturas são 1:2.500.000 (1 cm = 25 km).
  Isso é CONTEXTO REGIONAL, não litologia de prospecto. Cada resultado carrega
  `escala_nominal` para que o consumidor não trate como dado de detalhe.
"""

import httpx
import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from enum import IntEnum
import math

logger = logging.getLogger("prospector.geosgb")

# --- Endpoints verificados -------------------------------------------------
SGB_WFS = "https://opendata.sgb.gov.br/geoserver/ows"
# Aliases mantidos para compatibilidade com imports antigos.
GEOSGB_WFS = SGB_WFS

# CRS explícito no bbox: garante ordem (lon, lat). Sem isto o WFS 2.0 assume
# EPSG:4326 (lat, lon) e devolve zero feature sem erro.
CRS84_URN = "urn:ogc:def:crs:OGC:1.3:CRS84"

# --- Camadas verificadas (nomes reais lidos do GetCapabilities) ------------
LAYER_LITOESTRATIGRAFIA = "geonode:mgbrasil_litoestratigrafia_escala_1_2500000"
LAYER_ESTRUTURAS = "geonode:mgbrasil_estruturas_terrestres_escala_1_2500000"
LAYER_ESTRUTURAS_GEOFISICAS = "geonode:mgbrasil_estruturas_terrestres_geofisicas_escala_1_2500000"
LAYER_OCORRENCIAS = "p3m:vw_cprm_ocorr_min"
LAYER_GEOQUIMICA = "p3m:vw_cprm_geoq_b_d"

ESCALA_REGIONAL = 2_500_000

MAX_FEATURES = 500


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
    EMBASAMENTO_GNAISSICO = 11


# Classificação por descrição de litotipo (campos `litotipo1`/`litotipo2`).
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
    # Embasamento TTG/granulítico (ex.: Complexo Carapanã). Não é rocha
    # hospedeira de nenhum dos modelos, então `_score_lit` o pontua como
    # desfavorável — que é o comportamento correto.
    'gnaisse': TipoLitologia.EMBASAMENTO_GNAISSICO,
    'migmatito': TipoLitologia.EMBASAMENTO_GNAISSICO,
    'granulito': TipoLitologia.EMBASAMENTO_GNAISSICO,
    'tonalito': TipoLitologia.EMBASAMENTO_GNAISSICO,
    'trondhjemito': TipoLitologia.EMBASAMENTO_GNAISSICO,
}

# Inferência de SEGUNDA ORDEM: pelo NOME DA UNIDADE estratigráfica, usada só
# quando os campos de litotipo vêm nulos (o que é o caso na maior parte da
# camada 1:2.5M). É mais fraca que a classificação por litotipo — a unidade
# pode ser heterogênea — e por isso fica marcada em `Litologia.fonte`.
# Só entram unidades de Carajás com litologia dominante bem estabelecida.
UNIDADE_KEYWORDS = {
    'parauapebas': TipoLitologia.METABASALTO,
    'águas claras': TipoLitologia.FORMACAO_AGUAS_CLARAS,
    'aguas claras': TipoLitologia.FORMACAO_AGUAS_CLARAS,
    'xingu': TipoLitologia.COMPLEXO_XINGU,
    # Embasamento TTG/granulítico (ex.: Complexo Carapanã). Não é rocha
    # hospedeira de nenhum dos modelos, então `_score_lit` o pontua como
    # desfavorável — que é o comportamento correto.
    'gnaisse': TipoLitologia.EMBASAMENTO_GNAISSICO,
    'migmatito': TipoLitologia.EMBASAMENTO_GNAISSICO,
    'granulito': TipoLitologia.EMBASAMENTO_GNAISSICO,
    'tonalito': TipoLitologia.EMBASAMENTO_GNAISSICO,
    'trondhjemito': TipoLitologia.EMBASAMENTO_GNAISSICO,
}

# Faixas de plausibilidade para geoquímica de sedimento de corrente.
# MOTIVO: a camada `p3m:vw_cprm_geoq_b_d` contém valores fisicamente impossíveis
# (ex.: `fe_pct = 2310`, isto é, 2310% de ferro; `as_ppm = 5000`), aparentemente
# por erro de escala/unidade ou por marcadores de limite de detecção gravados
# como número. Alimentar isso direto no scoring produziria alvos fantasma, então
# valores fora da faixa são DESCARTADOS e registrados no diagnóstico.
# Só o limite SUPERIOR é diagnóstico: valor baixo é background legítimo, e
# abaixo do limite de detecção o SGB grava metade do LD (número pequeno e
# válido). Valor NEGATIVO é a convenção de "menor que" e também é descartado.
TETOS_PLAUSIVEIS = {
    'cu_ppm': 10000.0,
    'au_ppb': 50000.0,
    'fe_pct': 70.0,        # Fe puro = 100%; magnetita maciça ~72%
    'ag_ppm': 500.0,
    'as_ppm': 2000.0,
}


@dataclass
class Litologia:
    codigo: TipoLitologia = TipoLitologia.DESCONHECIDO
    nome: str = ''
    unidade: str = ''
    idade: str = ''
    provincia: str = ''
    sigla: str = ''
    hierarquia: str = ''
    fonte: str = ''              # 'litotipo' | 'nome_unidade' | ''
    escala_nominal: int = 0      # denominador da escala do mapa (0 = sem dado)


@dataclass
class Estrutura:
    tipo: str = ''
    nome: str = ''
    azimute: float = 0.0         # calculado do segmento mais próximo (0-180)
    distancia_m: float = 0.0
    geometria: List[Tuple[float, float]] = field(default_factory=list)
    origem: str = ''             # 'mapeada' | 'geofisica'
    escala_nominal: int = 0


@dataclass
class Deposito:
    nome: str = ''
    substancia: str = ''
    tipo: str = ''
    status: str = ''
    distancia_m: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    importancia: str = ''        # 'Depósito' | 'Indício' | 'Ocorrência' | ...
    rochas_hospedeiras: str = ''
    morfologia: str = ''


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
    tipo_amostra: str = ''
    descartados: List[str] = field(default_factory=list)


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
    # Trilha de proveniência: toda consulta que voltou vazia ou falhou aparece
    # aqui. Vazio silencioso deixa de ser indistinguível de "não há nada".
    diagnostico: List[str] = field(default_factory=list)

    def distancia_estrutura_m(self) -> float:
        if self.estrutura_mais_proxima:
            return self.estrutura_mais_proxima.distancia_m
        return 99999.0

    def distancia_deposito_m(self) -> float:
        if self.deposito_mais_proximo:
            return self.deposito_mais_proximo.distancia_m
        return 99999.0

    def consultas_com_dado(self) -> int:
        """Quantas das 4 consultas (lito, estrut, depósito, geoq) trouxeram dado."""
        n = 0
        if self.litologia.codigo != TipoLitologia.DESCONHECIDO or self.litologia.unidade:
            n += 1
        if self.estruturas:
            n += 1
        if self.depositos:
            n += 1
        if any(v is not None for v in (self.geoquimica.cu_ppm, self.geoquimica.au_ppb,
                                       self.geoquimica.fe_pct)):
            n += 1
        return n


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def azimute_segmento(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Azimute do segmento, dobrado para 0-180 (uma estrutura não tem sentido,
    só direção). Alimenta `AssinaturaEstrutural.azimutes` do motor metalogênico,
    já que a camada do SGB não traz campo de direção."""
    dlon = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    dlat = math.radians(lat2 - lat1)
    if dlon == 0 and dlat == 0:
        return 0.0
    az = math.degrees(math.atan2(dlon, dlat)) % 180.0
    return az


def classificar_litologia(texto: str) -> TipoLitologia:
    if not texto:
        return TipoLitologia.DESCONHECIDO
    texto_lower = texto.lower()
    for keyword, tipo in LITOLOGIA_KEYWORDS.items():
        if keyword in texto_lower:
            return tipo
    return TipoLitologia.DESCONHECIDO


def classificar_por_unidade(nome_unidade: str) -> TipoLitologia:
    """Inferência de segunda ordem pelo nome da unidade estratigráfica."""
    if not nome_unidade:
        return TipoLitologia.DESCONHECIDO
    alvo = nome_unidade.lower()
    for keyword, tipo in UNIDADE_KEYWORDS.items():
        if keyword in alvo:
            return tipo
    return TipoLitologia.DESCONHECIDO


def _plausivel(campo: str, valor: Optional[float]) -> bool:
    """True se o valor é fisicamente possível. Rejeita só o que é impossível:
    negativo (marcador de "abaixo do limite de detecção") e acima do teto
    físico do elemento. Valores baixos passam — são background real."""
    if valor is None:
        return False
    if valor < 0:
        return False
    teto = TETOS_PLAUSIVEIS.get(campo)
    if teto is None:
        return True
    return valor <= teto


class ResultadoWFS:
    """Resultado de uma consulta WFS, com o motivo quando não há feature."""
    __slots__ = ('features', 'erro')

    def __init__(self, features: List[dict], erro: str = ''):
        self.features = features
        self.erro = erro

    def __bool__(self) -> bool:
        return bool(self.features)


class GeoSGBClient:
    # Cache de FeatureTypes por processo. None = ainda não consultado.
    _typenames_cache: Optional[Set[str]] = None

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
        self.diagnostico: List[str] = []

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    def _bbox(self, lat: float, lon: float, buffer: float) -> str:
        """bbox no formato lon_min,lat_min,lon_max,lat_max + CRS explícito.
        O sufixo de CRS é obrigatório: sem ele o WFS 2.0 assume ordem (lat, lon)
        e a consulta cai no oceano, devolvendo zero feature sem erro."""
        return (f"{lon-buffer},{lat-buffer},{lon+buffer},{lat+buffer},"
                f"{CRS84_URN}")

    async def typenames_disponiveis(self) -> Set[str]:
        """FeatureTypes reais publicados pelo serviço (cache por processo)."""
        if GeoSGBClient._typenames_cache is not None:
            return GeoSGBClient._typenames_cache

        nomes: Set[str] = set()
        params = {"service": "WFS", "version": "2.0.0", "request": "GetCapabilities"}
        try:
            resp = await self.client.get(SGB_WFS, params=params)
            if resp.status_code == 200:
                nomes = set(re.findall(
                    r"<(?:\w+:)?Name>\s*([^<\s][^<]*?)\s*</(?:\w+:)?Name>", resp.text))
            else:
                logger.warning("SGB GetCapabilities HTTP %s", resp.status_code)
        except Exception as e:
            logger.warning("SGB GetCapabilities inacessível: %s", e)

        GeoSGBClient._typenames_cache = nomes
        return nomes

    async def _wfs_query(self, typename: str, bbox: str) -> ResultadoWFS:
        """GetFeature em WFS 2.0. Ao contrário da versão anterior, NUNCA engole
        o erro: toda falha vira motivo textual em `ResultadoWFS.erro`."""
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": typename,          # WFS 2.0: PLURAL. `typeName` = 0 features.
            "outputFormat": "application/json",
            "bbox": bbox,                   # já inclui o CRS
            "count": str(MAX_FEATURES),
        }
        try:
            resp = await self.client.get(SGB_WFS, params=params)
        except Exception as e:
            motivo = f"{typename}: falha de rede ({type(e).__name__})"
            logger.warning(motivo)
            return ResultadoWFS([], motivo)

        if resp.status_code != 200:
            motivo = f"{typename}: HTTP {resp.status_code}"
            logger.warning(motivo)
            return ResultadoWFS([], motivo)

        # O GeoServer devolve ExceptionReport em XML com HTTP 200.
        texto = resp.text.lstrip()
        if texto.startswith('<'):
            trecho = re.sub(r"\s+", " ", texto[:200])
            motivo = f"{typename}: serviço retornou XML/exceção — {trecho}"
            logger.warning(motivo)
            return ResultadoWFS([], motivo)

        try:
            data = resp.json()
        except Exception:
            motivo = f"{typename}: resposta não é JSON"
            logger.warning(motivo)
            return ResultadoWFS([], motivo)

        feats = data.get('features', [])
        if not feats:
            return ResultadoWFS([], f"{typename}: sem feature na bbox")
        return ResultadoWFS(feats)

    def _registrar(self, res: ResultadoWFS):
        if res.erro:
            self.diagnostico.append(res.erro)

    # --- consultas ---------------------------------------------------------

    async def buscar_litologia(self, lat: float, lon: float) -> Litologia:
        """Unidade litoestratigráfica 1:2.500.000.

        A bbox é pequena de propósito (~550 m): a pergunta é point-in-polygon
        ("em que unidade este ponto cai"), e uma caixa larga faria `features[0]`
        cair numa unidade VIZINHA. A ressalva de escala do mapa não se resolve
        alargando a consulta — ela viaja em `Litologia.escala_nominal`.
        """
        bbox = self._bbox(lat, lon, 0.005)
        res = await self._wfs_query(LAYER_LITOESTRATIGRAFIA, bbox)
        self._registrar(res)
        if not res:
            return Litologia()

        props = res.features[0].get('properties', {})

        # Campos REAIS da camada (a versão anterior procurava 'nome_unidade',
        # 'litologia_principal', 'eon_era' — nenhum deles existe).
        nome_unidade = props.get('nome_unida') or ''
        litotipo = props.get('litotipo1') or props.get('litotipo2') or ''

        codigo = classificar_litologia(litotipo)
        fonte = 'litotipo' if codigo != TipoLitologia.DESCONHECIDO else ''
        if codigo == TipoLitologia.DESCONHECIDO:
            codigo = classificar_por_unidade(nome_unidade)
            if codigo != TipoLitologia.DESCONHECIDO:
                fonte = 'nome_unidade'

        idade = ' / '.join(x for x in (props.get('era_maxima'), props.get('eon_idad_m')) if x)

        return Litologia(
            codigo=codigo,
            nome=litotipo or nome_unidade,
            unidade=nome_unidade,
            idade=idade,
            provincia='',
            sigla=props.get('sigla_unid') or '',
            hierarquia=props.get('hierarquia') or '',
            fonte=fonte,
            escala_nominal=ESCALA_REGIONAL,
        )

    async def buscar_estruturas(self, lat: float, lon: float, raio_km: float = 25.0) -> List[Estrutura]:
        """Estruturas mapeadas + lineamentos interpretados de geofísica.
        Raio padrão de 25 km: coerente com a escala 1:2.5M das camadas."""
        buffer = raio_km / 111.0
        bbox = self._bbox(lat, lon, buffer)
        estruturas: List[Estrutura] = []

        for typename, origem in ((LAYER_ESTRUTURAS, 'mapeada'),
                                 (LAYER_ESTRUTURAS_GEOFISICAS, 'geofisica')):
            res = await self._wfs_query(typename, bbox)
            self._registrar(res)

            for f in res.features:
                geom = f.get('geometry') or {}
                coords = geom.get('coordinates') or []
                props = f.get('properties', {})

                if geom.get('type') == 'LineString':
                    linhas = [coords]
                elif geom.get('type') == 'MultiLineString':
                    linhas = coords
                else:
                    continue

                min_dist = float('inf')
                pontos: List[Tuple[float, float]] = []
                az = 0.0

                for linha in linhas:
                    validos = [c for c in linha if len(c) >= 2]
                    for i, coord in enumerate(validos):
                        pontos.append((coord[0], coord[1]))
                        d = haversine(lat, lon, coord[1], coord[0])
                        if d < min_dist:
                            min_dist = d
                            # Vizinho para o segmento: o anterior, ou o seguinte
                            # quando o vértice mais próximo é o primeiro da linha
                            # (caso em que o azimute ficava 0 indevidamente).
                            viz = validos[i - 1] if i > 0 else (
                                validos[i + 1] if i + 1 < len(validos) else None)
                            az = azimute_segmento(viz[0], viz[1],
                                                  coord[0], coord[1]) if viz else 0.0

                if min_dist < raio_km * 1000:
                    estruturas.append(Estrutura(
                        # Campos reais: 'tipo_estru' e 'nmestrutur'.
                        tipo=props.get('tipo_estru') or '',
                        nome=props.get('nmestrutur') or '',
                        azimute=az,
                        distancia_m=min_dist,
                        geometria=pontos[:200],
                        origem=origem,
                        escala_nominal=ESCALA_REGIONAL,
                    ))

        estruturas.sort(key=lambda x: x.distancia_m)
        return estruturas[:10]

    async def buscar_depositos(self, lat: float, lon: float, raio_km: float = 25.0) -> List[Deposito]:
        """Ocorrências minerais cadastradas (RECMIN). É a fonte de GROUND TRUTH
        prevista no RELATORIO_TECNICO.md §6 (Fase 3): o campo `importancia`
        separa 'Depósito' de 'Indício'/'Ocorrência', e `substancias` diz a
        commodity — insumo direto para rotular positivos em PU learning."""
        buffer = raio_km / 111.0
        bbox = self._bbox(lat, lon, buffer)
        depositos: List[Deposito] = []

        res = await self._wfs_query(LAYER_OCORRENCIAS, bbox)
        self._registrar(res)

        for f in res.features:
            geom = f.get('geometry') or {}
            coords = geom.get('coordinates') or []
            if len(coords) < 2:
                continue
            props = f.get('properties', {})
            dep_lat, dep_lon = coords[1], coords[0]
            dist = haversine(lat, lon, dep_lat, dep_lon)
            if dist >= raio_km * 1000:
                continue

            depositos.append(Deposito(
                # Campos reais: 'toponimia', 'substancias', 'importancia', ...
                nome=props.get('toponimia') or '',
                substancia=props.get('substancias') or '',
                tipo=props.get('classes_utilitarias') or '',
                status=props.get('status_econ') or props.get('situ_mina') or '',
                distancia_m=dist,
                latitude=dep_lat,
                longitude=dep_lon,
                importancia=props.get('importancia') or '',
                rochas_hospedeiras=props.get('rochas_hosp') or '',
                morfologia=props.get('morfologia') or '',
            ))

        depositos.sort(key=lambda x: x.distancia_m)
        return depositos[:10]

    async def buscar_geoquimica(self, lat: float, lon: float, raio_km: float = 10.0) -> Geoquimica:
        """Geoquímica de sedimento de corrente.

        ATENÇÃO — QUALIDADE DO DADO: esta camada contém valores fisicamente
        impossíveis (medidos: `fe_pct = 2310`, `as_ppm = 5000`, `co_ppm = 3000`),
        provavelmente erro de escala/unidade ou limite de detecção gravado como
        número. Cada campo passa por `TETOS_PLAUSIVEIS`; o que não passa é
        DESCARTADO e listado em `Geoquimica.descartados`. Nenhum valor suspeito
        entra no scoring.
        """
        buffer = raio_km / 111.0
        bbox = self._bbox(lat, lon, buffer)

        res = await self._wfs_query(LAYER_GEOQUIMICA, bbox)
        self._registrar(res)
        if not res:
            return Geoquimica()

        closest = None
        min_dist = float('inf')
        for f in res.features:
            coords = (f.get('geometry') or {}).get('coordinates') or []
            if len(coords) >= 2:
                d = haversine(lat, lon, coords[1], coords[0])
                if d < min_dist:
                    min_dist = d
                    closest = f

        if closest is None or min_dist >= raio_km * 1000:
            return Geoquimica()

        props = closest.get('properties', {})
        au_ppm = self._safe_float(props.get('au_ppm'))

        brutos = {
            'cu_ppm': self._safe_float(props.get('cu_ppm')),
            # A camada publica ouro em ppm; o resto do sistema usa ppb.
            'au_ppb': (au_ppm * 1000.0) if au_ppm is not None else None,
            'fe_pct': self._safe_float(props.get('fe_pct')),
            'ag_ppm': self._safe_float(props.get('ag_ppm')),
            'as_ppm': self._safe_float(props.get('as_ppm')),
        }

        limpos: Dict[str, Optional[float]] = {}
        descartados: List[str] = []
        for campo, valor in brutos.items():
            if valor is None:
                limpos[campo] = None
            elif _plausivel(campo, valor):
                limpos[campo] = valor
            else:
                limpos[campo] = None
                descartados.append(f"{campo}={valor:g} fora da faixa plausível")

        if descartados:
            logger.warning("Geoquímica implausível descartada em %.4f,%.4f: %s",
                           lat, lon, '; '.join(descartados))

        return Geoquimica(
            distancia_m=min_dist,
            amostra_id=props.get('num_campo') or props.get('num_lab') or '',
            tipo_amostra=props.get('Tipo') or props.get('tipo') or '',
            descartados=descartados,
            **limpos,
        )

    def _safe_float(self, val) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None


async def obter_contexto_geologico(lat: float, lon: float) -> ContextoGeologico:
    ctx = ContextoGeologico(latitude=lat, longitude=lon)

    async with GeoSGBClient() as client:
        litologia, estruturas, depositos, geoquimica = await asyncio.gather(
            client.buscar_litologia(lat, lon),
            client.buscar_estruturas(lat, lon),
            client.buscar_depositos(lat, lon),
            client.buscar_geoquimica(lat, lon),
            return_exceptions=True,
        )

        for nome, r in (('litologia', litologia), ('estruturas', estruturas),
                        ('depositos', depositos), ('geoquimica', geoquimica)):
            if isinstance(r, BaseException):
                msg = f"{nome}: exceção {type(r).__name__}: {r}"
                logger.warning(msg)
                ctx.diagnostico.append(msg)

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
            ctx.diagnostico.extend(geoquimica.descartados)

        ctx.diagnostico.extend(client.diagnostico)

    return ctx


if __name__ == '__main__':
    async def test():
        logging.basicConfig(level=logging.INFO)
        # Sossego / Canaã dos Carajás
        ctx = await obter_contexto_geologico(-6.43, -50.05)
        print(f"Litologia : {ctx.litologia.nome} ({ctx.litologia.codigo.name})"
              f" [fonte={ctx.litologia.fonte or 'n/d'}, 1:{ctx.litologia.escala_nominal:,}]")
        print(f"Unidade   : {ctx.litologia.unidade} ({ctx.litologia.sigla}) — {ctx.litologia.idade}")
        print(f"Estruturas: {len(ctx.estruturas)}")
        for e in ctx.estruturas[:3]:
            print(f"   {e.tipo} [{e.origem}] az={e.azimute:.0f} a {e.distancia_m:.0f} m")
        print(f"Depósitos : {len(ctx.depositos)}")
        for d in ctx.depositos[:5]:
            print(f"   {d.nome} | {d.substancia} | {d.importancia} | {d.distancia_m:.0f} m")
        print(f"Geoquímica: Cu={ctx.geoquimica.cu_ppm} ppm  Au={ctx.geoquimica.au_ppb} ppb"
              f"  ({ctx.geoquimica.tipo_amostra})")
        print(f"Consultas com dado: {ctx.consultas_com_dado()}/4")
        if ctx.diagnostico:
            print("Diagnóstico:")
            for d in ctx.diagnostico:
                print(f"   - {d}")

    asyncio.run(test())
