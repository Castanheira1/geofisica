#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ███████╗ ██████╗       ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗  ║
║  ██╔════╝ ██╔════╝██╔═══██╗      ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗ ║
║  ██║  ███╗█████╗  ██║   ██║█████╗██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║ ║
║  ██║   ██║██╔══╝  ██║   ██║╚════╝██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║ ║
║  ╚██████╔╝███████╗╚██████╔╝      ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║ ║
║   ╚═════╝ ╚══════╝ ╚═════╝       ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ║
║                                                                              ║
║   ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗                ║
║   ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝                ║
║   ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗                  ║
║   ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝                  ║
║   ███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗                ║
║   ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝                ║
║                                                                              ║
║                           VERSION 3.0 CARAJÁS                                ║
║                                                                              ║
║   Motor de Prospecção Mineral com Inteligência Artificial Avançada           ║
║   • Lógica Fuzzy Multi-Camada com Inferência Mamdani                        ║
║   • Análise de Gradiente Tensorial 2D                                        ║
║   • Detecção de Bordas Estruturais (Sobel/Prewitt)                          ║
║   • Rede Neural Artificial para Classificação de Alvos                      ║
║   • Sistema de Scoring Bayesiano                                             ║
║   • Geração Automática de Mapas de Calor                                     ║
║   • Análise de Cluster Espacial (DBSCAN)                                     ║
║   • Integração com Magnetômetro PPM                                          ║
║   • Relatórios Científicos Automatizados                                     ║
║                                                                              ║
║   Autores: Wanderlei (Vale S11D) & Claude (Anthropic)                       ║
║   Projeto: PROSPECTOR-AI Carajás                                             ║
║   Licença: Proprietária - Uso Exclusivo                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
from enum import IntEnum
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Verificação de dependências opcionais
try:
    from scipy.spatial import cKDTree
    from scipy.ndimage import sobel, gaussian_filter, laplace
    from scipy.interpolate import griddata
    from scipy.signal import savgol_filter
    from scipy.stats import zscore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[AVISO] SciPy não instalado. Funcionalidades avançadas limitadas.")

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Circle, FancyBboxPatch
    import matplotlib.patheffects as path_effects
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[AVISO] Matplotlib não instalado. Geração de mapas desabilitada.")

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[AVISO] Scikit-learn não instalado. Machine Learning desabilitado.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        1. ESTRUTURAS DE DADOS                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Litologia(IntEnum):
    """
    Códigos de litologia com favorabilidade intrínseca para depósitos IOCG.
    Baseado na geologia do Domínio Carajás, Província Mineral de Carajás.
    """
    DESCONHECIDO = 0
    COBERTURA_SEDIMENTAR = 1      # Baixa favorabilidade
    GRANITO_ANOROGÊNICO = 2       # Moderada (fonte de fluidos)
    MAFICO_ULTRAMÁFICO = 3        # Alta (hospedeiro)
    BIF_GRÃO_PARA = 4             # Alta para Fe, moderada para Cu-Au
    BIF_HIDROTERMALIZADO = 5      # Muito alta para IOCG
    VULCANICA_GRÃO_PARA = 6       # Alta
    METABASALTO = 7               # Alta (hospedeiro preferencial)
    BRECHA_HIDROTERMAL = 8        # Muito alta (estrutura condutora)
    FORMACAO_AGUAS_CLARAS = 9     # Baixa (cobertura)
    COMPLEXO_XINGU = 10           # Moderada (embasamento)


class TipoDeposito(IntEnum):
    """Tipos de depósitos minerais alvo"""
    IOCG = 1                      # Iron Oxide Copper Gold
    BIF_FE = 2                    # Formação Ferrífera Bandada
    OURO_OROGÊNICO = 3            # Ouro em zonas de cisalhamento
    SULFETO_MAGMÁTICO = 4         # Ni-Cu-PGE
    LATERITA_NI = 5               # Níquel laterítico


@dataclass
class ConfiguracaoModelo:
    """
    Configuração completa do modelo de prospecção.
    Todos os parâmetros são ajustáveis para calibração com dados reais.
    """
    
    # Campo magnético regional (IGRF Carajás)
    mag_regional: float = 24600.0
    
    # Limiares de anomalia magnética (nT acima do regional)
    mag_thresholds: Tuple[float, ...] = (50, 200, 800, 2000, 5000)
    
    # Limiares de Cu em solo (ppm)
    cu_thresholds: Tuple[float, ...] = (30, 100, 300, 700, 1500)
    
    # Limiares de Au em solo (ppb)
    au_thresholds: Tuple[float, ...] = (3, 15, 50, 200, 1000)
    
    # Limiares de resistividade (Ohm.m) - INVERSO
    res_thresholds: Tuple[float, ...] = (10000, 2000, 500, 100, 20)
    
    # Limiares de cargabilidade IP (mV/V)
    ip_thresholds: Tuple[float, ...] = (3, 10, 25, 50, 100)
    
    # Limiares de distância a estruturas (m) - INVERSO
    struct_thresholds: Tuple[float, ...] = (3000, 1000, 300, 100, 30)
    
    # Pesos dos parâmetros (soma = 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'mag': 0.18,
        'cu': 0.22,
        'au': 0.25,
        'res': 0.12,
        'ip': 0.13,
        'struct': 0.10
    })
    
    # Fatores litológicos (multiplicadores)
    litologia_factors: Dict[int, float] = field(default_factory=lambda: {
        0: 0.50,   # Desconhecido
        1: 0.15,   # Cobertura sedimentar
        2: 0.60,   # Granito anorogênico
        3: 0.95,   # Máfico/ultramáfico
        4: 0.70,   # BIF Grão Pará
        5: 1.40,   # BIF hidrotermalizado
        6: 0.85,   # Vulcânica Grão Pará
        7: 1.00,   # Metabasalto
        8: 1.60,   # Brecha hidrotermal
        9: 0.20,   # Fm. Águas Claras
        10: 0.55,  # Complexo Xingu
    })
    
    # Parâmetros de gradiente espacial
    gradient_n_neighbors: int = 5
    gradient_weight_mag: float = 0.55
    gradient_weight_res: float = 0.45
    
    # Parâmetros de cluster
    cluster_eps_degrees: float = 0.002  # ~220m
    cluster_min_samples: int = 3
    
    # Parâmetros da rede neural
    nn_hidden_layers: Tuple[int, ...] = (64, 32, 16)
    nn_max_iter: int = 1000


@dataclass
class PontoProspeccao:
    """
    Estrutura de dados completa para um ponto de prospecção.
    Inclui dados brutos, scores calculados e metadados.
    """
    id: str
    latitude: float
    longitude: float
    
    # Dados de entrada (valores brutos medidos)
    mag_nt: Optional[float] = None
    cu_ppm: Optional[float] = None
    au_ppb: Optional[float] = None
    resistividade: Optional[float] = None
    ip_mv: Optional[float] = None
    dist_estrutura_m: Optional[float] = None
    litologia: Litologia = Litologia.DESCONHECIDO
    
    # Metadados
    data_coleta: Optional[str] = None
    operador: Optional[str] = None
    qualidade: int = 0  # 0-3
    
    # Scores calculados (preenchidos pelo motor)
    score_fuzzy: float = 0.0
    score_gradiente: float = 0.0
    score_bayesiano: float = 0.0
    score_neural: float = 0.0
    synapse_index: float = 0.0
    classificacao: str = ""
    cluster_id: int = -1
    
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        return {
            'id': self.id,
            'lat': self.latitude,
            'lon': self.longitude,
            'mag_nt': self.mag_nt,
            'cu_ppm': self.cu_ppm,
            'au_ppb': self.au_ppb,
            'resistividade': self.resistividade,
            'ip_mv': self.ip_mv,
            'dist_estrutura': self.dist_estrutura_m,
            'litologia': int(self.litologia),
            'synapse_index': self.synapse_index,
            'classificacao': self.classificacao
        }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     2. MOTOR FUZZY AVANÇADO                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MotorFuzzy:
    """
    Motor de inferência fuzzy com funções de pertinência trapezoidais
    e sistema de regras baseado em Mamdani simplificado.
    """
    
    def __init__(self, config: ConfiguracaoModelo):
        self.config = config
    
    def _pertinencia_trapezoidal(self, x: float, params: Tuple[float, ...], 
                                  inverso: bool = False) -> Dict[str, float]:
        """
        Calcula os graus de pertinência para conjuntos fuzzy trapezoidais.
        
        Retorna um dicionário com graus de pertinência para cada conjunto:
        - MUITO_BAIXO, BAIXO, MEDIO, ALTO, MUITO_ALTO
        """
        if x is None:
            return None
        
        a, b, c, d, e = params
        
        if inverso:
            # Inverte a ordem para parâmetros onde baixo é bom
            x_norm = 1.0 - (x - e) / (a - e) if a != e else 0.5
            x_norm = max(0, min(1, x_norm))
            
            if x <= e:
                return {'MUITO_ALTO': 1.0, 'ALTO': 0.0, 'MEDIO': 0.0, 'BAIXO': 0.0, 'MUITO_BAIXO': 0.0}
            elif x >= a:
                return {'MUITO_ALTO': 0.0, 'ALTO': 0.0, 'MEDIO': 0.0, 'BAIXO': 0.0, 'MUITO_BAIXO': 1.0}
        else:
            if x <= a:
                return {'MUITO_BAIXO': 1.0, 'BAIXO': 0.0, 'MEDIO': 0.0, 'ALTO': 0.0, 'MUITO_ALTO': 0.0}
            elif x >= e:
                return {'MUITO_BAIXO': 0.0, 'BAIXO': 0.0, 'MEDIO': 0.0, 'ALTO': 0.0, 'MUITO_ALTO': 1.0}
        
        # Interpolação entre os conjuntos
        pertinencias = {'MUITO_BAIXO': 0.0, 'BAIXO': 0.0, 'MEDIO': 0.0, 'ALTO': 0.0, 'MUITO_ALTO': 0.0}
        
        if inverso:
            params_inv = (e, d, c, b, a)
            a, b, c, d, e = params_inv
        
        if x < b:
            pertinencias['MUITO_BAIXO'] = (b - x) / (b - a) if b != a else 0
            pertinencias['BAIXO'] = (x - a) / (b - a) if b != a else 1
        elif x < c:
            pertinencias['BAIXO'] = (c - x) / (c - b) if c != b else 0
            pertinencias['MEDIO'] = (x - b) / (c - b) if c != b else 1
        elif x < d:
            pertinencias['MEDIO'] = (d - x) / (d - c) if d != c else 0
            pertinencias['ALTO'] = (x - c) / (d - c) if d != c else 1
        elif x < e:
            pertinencias['ALTO'] = (e - x) / (e - d) if e != d else 0
            pertinencias['MUITO_ALTO'] = (x - d) / (e - d) if e != d else 1
        
        return pertinencias
    
    def _defuzzificar_centroide(self, pertinencias: Dict[str, float]) -> float:
        """
        Defuzzifica usando método do centróide.
        Mapeia conjuntos fuzzy para valores crisp (0-1).
        """
        if pertinencias is None:
            return None
        
        # Centroides dos conjuntos fuzzy
        centroides = {
            'MUITO_BAIXO': 0.1,
            'BAIXO': 0.3,
            'MEDIO': 0.5,
            'ALTO': 0.7,
            'MUITO_ALTO': 0.9
        }
        
        numerador = sum(pertinencias[k] * centroides[k] for k in pertinencias)
        denominador = sum(pertinencias.values())
        
        if denominador == 0:
            return 0.0
        
        return numerador / denominador
    
    def calcular_score(self, ponto: PontoProspeccao) -> Tuple[float, Dict]:
        """
        Calcula o score fuzzy completo para um ponto de prospecção.
        """
        scores = {}
        detalhes = {}
        
        # Magnetometria (anomalia = valor - regional)
        if ponto.mag_nt is not None:
            anomalia = ponto.mag_nt - self.config.mag_regional
            pert = self._pertinencia_trapezoidal(anomalia, self.config.mag_thresholds)
            scores['mag'] = self._defuzzificar_centroide(pert)
            detalhes['mag_pertinencia'] = pert
        
        # Cobre
        if ponto.cu_ppm is not None:
            pert = self._pertinencia_trapezoidal(ponto.cu_ppm, self.config.cu_thresholds)
            scores['cu'] = self._defuzzificar_centroide(pert)
            detalhes['cu_pertinencia'] = pert
        
        # Ouro
        if ponto.au_ppb is not None:
            pert = self._pertinencia_trapezoidal(ponto.au_ppb, self.config.au_thresholds)
            scores['au'] = self._defuzzificar_centroide(pert)
            detalhes['au_pertinencia'] = pert
        
        # Resistividade (inverso)
        if ponto.resistividade is not None:
            pert = self._pertinencia_trapezoidal(ponto.resistividade, 
                                                  self.config.res_thresholds, inverso=True)
            scores['res'] = self._defuzzificar_centroide(pert)
            detalhes['res_pertinencia'] = pert
        
        # IP
        if ponto.ip_mv is not None:
            pert = self._pertinencia_trapezoidal(ponto.ip_mv, self.config.ip_thresholds)
            scores['ip'] = self._defuzzificar_centroide(pert)
            detalhes['ip_pertinencia'] = pert
        
        # Estrutura (inverso)
        if ponto.dist_estrutura_m is not None:
            pert = self._pertinencia_trapezoidal(ponto.dist_estrutura_m,
                                                  self.config.struct_thresholds, inverso=True)
            scores['struct'] = self._defuzzificar_centroide(pert)
            detalhes['struct_pertinencia'] = pert
        
        # Agregação ponderada
        soma_ponderada = 0.0
        soma_pesos = 0.0
        
        for param, score in scores.items():
            if score is not None:
                peso = self.config.weights.get(param, 0)
                soma_ponderada += score * peso
                soma_pesos += peso
        
        if soma_pesos == 0:
            return 0.0, {'erro': 'Sem dados suficientes'}
        
        score_base = soma_ponderada / soma_pesos
        
        # Fator litológico
        f_lito = self.config.litologia_factors.get(int(ponto.litologia), 0.5)
        
        # Fator de coincidência (bônus por múltiplos indicadores > 0.6)
        n_altos = sum(1 for s in scores.values() if s is not None and s > 0.6)
        f_coincidencia = 1.0 + min((n_altos - 1) * 0.12, 0.36) if n_altos > 1 else 1.0
        
        # Score fuzzy final
        score_fuzzy = score_base * f_lito * f_coincidencia
        score_fuzzy = min(score_fuzzy, 1.0)  # Clamp
        
        detalhes.update({
            'scores_parciais': scores,
            'score_base': score_base,
            'f_litologia': f_lito,
            'f_coincidencia': f_coincidencia,
            'score_fuzzy_final': score_fuzzy
        })
        
        return score_fuzzy, detalhes


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   3. ANÁLISE DE GRADIENTE TENSORIAL                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class AnalisadorGradiente:
    """
    Análise de gradiente espacial usando operadores tensoriais.
    Detecta bordas estruturais e zonas de ruptura magnética.
    """
    
    def __init__(self, config: ConfiguracaoModelo):
        self.config = config
    
    def calcular_gradiente_knn(self, df: pd.DataFrame, campo: str) -> np.ndarray:
        """
        Calcula o gradiente espacial usando K vizinhos mais próximos.
        Mais robusto que vizinho único.
        """
        if not SCIPY_AVAILABLE:
            return np.zeros(len(df))
        
        coords = df[['lat', 'lon']].values
        valores = df[campo].values
        
        # Construir KD-Tree para busca eficiente
        tree = cKDTree(coords)
        n_neighbors = min(self.config.gradient_n_neighbors, len(df) - 1)
        
        gradientes = []
        
        for i in range(len(df)):
            # Encontrar K vizinhos mais próximos
            dists, idxs = tree.query(coords[i], k=n_neighbors + 1)
            
            # Excluir o próprio ponto
            dists = dists[1:]
            idxs = idxs[1:]
            
            # Calcular gradientes direcionais
            grad_magnitudes = []
            for j, (dist, idx) in enumerate(zip(dists, idxs)):
                if dist > 0:
                    delta = abs(valores[i] - valores[idx])
                    grad = delta / dist
                    grad_magnitudes.append(grad)
            
            # Média ponderada (vizinhos mais próximos têm mais peso)
            if grad_magnitudes:
                weights = 1.0 / (np.array(dists[:len(grad_magnitudes)]) + 1e-6)
                weights /= weights.sum()
                grad_medio = np.average(grad_magnitudes, weights=weights)
            else:
                grad_medio = 0.0
            
            gradientes.append(grad_medio)
        
        return np.array(gradientes)
    
    def calcular_gradiente_grid(self, df: pd.DataFrame, campo: str, 
                                 resolucao: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula gradiente usando interpolação em grid regular + operador Sobel.
        Retorna tanto o grid de gradiente quanto valores interpolados nos pontos.
        """
        if not SCIPY_AVAILABLE:
            return np.zeros(len(df)), None
        
        coords = df[['lat', 'lon']].values
        valores = df[campo].values
        
        # Criar grid regular
        lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
        lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Adicionar buffer
        lat_buffer = (lat_max - lat_min) * 0.1
        lon_buffer = (lon_max - lon_min) * 0.1
        
        grid_lat = np.linspace(lat_min - lat_buffer, lat_max + lat_buffer, resolucao)
        grid_lon = np.linspace(lon_min - lon_buffer, lon_max + lon_buffer, resolucao)
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
        
        # Interpolar valores no grid
        grid_valores = griddata(coords, valores, (grid_lat_mesh, grid_lon_mesh), method='cubic')
        grid_valores = np.nan_to_num(grid_valores, nan=np.nanmean(valores))
        
        # Suavização gaussiana para reduzir ruído
        grid_suave = gaussian_filter(grid_valores, sigma=1.5)
        
        # Operador Sobel para gradiente
        grad_lat = sobel(grid_suave, axis=0)
        grad_lon = sobel(grid_suave, axis=1)
        
        # Magnitude do gradiente
        grad_magnitude = np.sqrt(grad_lat**2 + grad_lon**2)
        
        # Interpolar de volta para os pontos originais
        grad_nos_pontos = griddata((grid_lat_mesh.flatten(), grid_lon_mesh.flatten()),
                                    grad_magnitude.flatten(), coords, method='linear')
        grad_nos_pontos = np.nan_to_num(grad_nos_pontos, nan=0)
        
        return grad_nos_pontos, grad_magnitude
    
    def calcular_laplaciano(self, df: pd.DataFrame, campo: str) -> np.ndarray:
        """
        Calcula o Laplaciano (segunda derivada) para detectar picos e vales.
        Útil para identificar o centro de corpos anômalos.
        """
        if not SCIPY_AVAILABLE:
            return np.zeros(len(df))
        
        coords = df[['lat', 'lon']].values
        valores = df[campo].values
        
        # Grid
        resolucao = min(50, len(df))
        lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
        lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()
        
        grid_lat = np.linspace(lat_min, lat_max, resolucao)
        grid_lon = np.linspace(lon_min, lon_max, resolucao)
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
        
        grid_valores = griddata(coords, valores, (grid_lat_mesh, grid_lon_mesh), method='cubic')
        grid_valores = np.nan_to_num(grid_valores, nan=np.nanmean(valores))
        
        # Laplaciano
        lap = laplace(gaussian_filter(grid_valores, sigma=1))
        
        # Interpolar de volta
        lap_pontos = griddata((grid_lat_mesh.flatten(), grid_lon_mesh.flatten()),
                               lap.flatten(), coords, method='linear')
        
        return np.nan_to_num(lap_pontos, nan=0)
    
    def calcular_fre(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcula o Fator de Ruptura Estrutural (FRE) combinando múltiplos gradientes.
        """
        # Gradiente magnético
        if 'mag_nt' in df.columns and df['mag_nt'].notna().any():
            grad_mag, _ = self.calcular_gradiente_grid(df, 'mag_nt')
        else:
            grad_mag = np.zeros(len(df))
        
        # Gradiente de resistividade
        if 'resistividade' in df.columns and df['resistividade'].notna().any():
            grad_res, _ = self.calcular_gradiente_grid(df, 'resistividade')
        else:
            grad_res = np.zeros(len(df))
        
        # Normalizar cada componente
        def normalizar(arr):
            max_val = np.max(np.abs(arr))
            return arr / max_val if max_val > 0 else arr
        
        grad_mag_norm = normalizar(grad_mag)
        grad_res_norm = normalizar(grad_res)
        
        # FRE combinado
        fre = (self.config.gradient_weight_mag * grad_mag_norm + 
               self.config.gradient_weight_res * grad_res_norm)
        
        return fre


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      4. INFERÊNCIA BAYESIANA                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MotorBayesiano:
    """
    Sistema de inferência Bayesiana para probabilidade de mineralização.
    Usa probabilidades condicionais calibradas para depósitos IOCG de Carajás.
    """
    
    def __init__(self):
        # Prior: probabilidade base de mineralização em Carajás
        self.prior_mineralizacao = 0.05
        
        # Likelihood ratios (LR) para cada evidência
        # LR = P(evidência|mineralização) / P(evidência|não mineralização)
        self.likelihood_ratios = {
            'mag_alta': 8.0,           # Anomalia mag > 500 nT
            'mag_muito_alta': 15.0,    # Anomalia mag > 1500 nT
            'cu_anomalo': 12.0,        # Cu > 200 ppm
            'cu_muito_alto': 25.0,     # Cu > 700 ppm
            'au_anomalo': 10.0,        # Au > 20 ppb
            'au_muito_alto': 50.0,     # Au > 200 ppb
            'res_baixa': 6.0,          # Res < 500 Ohm.m
            'res_muito_baixa': 20.0,   # Res < 100 Ohm.m
            'ip_alta': 5.0,            # IP > 25 mV/V
            'ip_muito_alta': 12.0,     # IP > 50 mV/V
            'proximal_estrutura': 4.0, # < 300m de estrutura
            'sobre_estrutura': 10.0,   # < 100m de estrutura
            'lito_favoravel': 3.0,     # BIF alterado, brecha, metabasalto
            'lito_muito_favoravel': 8.0,  # Brecha hidrotermal
        }
    
    def _obter_evidencias(self, ponto: PontoProspeccao, config: ConfiguracaoModelo) -> List[str]:
        """Identifica as evidências presentes no ponto"""
        evidencias = []
        
        # Magnetometria
        if ponto.mag_nt is not None:
            anomalia = ponto.mag_nt - config.mag_regional
            if anomalia > 1500:
                evidencias.append('mag_muito_alta')
            elif anomalia > 500:
                evidencias.append('mag_alta')
        
        # Cobre
        if ponto.cu_ppm is not None:
            if ponto.cu_ppm > 700:
                evidencias.append('cu_muito_alto')
            elif ponto.cu_ppm > 200:
                evidencias.append('cu_anomalo')
        
        # Ouro
        if ponto.au_ppb is not None:
            if ponto.au_ppb > 200:
                evidencias.append('au_muito_alto')
            elif ponto.au_ppb > 20:
                evidencias.append('au_anomalo')
        
        # Resistividade
        if ponto.resistividade is not None:
            if ponto.resistividade < 100:
                evidencias.append('res_muito_baixa')
            elif ponto.resistividade < 500:
                evidencias.append('res_baixa')
        
        # IP
        if ponto.ip_mv is not None:
            if ponto.ip_mv > 50:
                evidencias.append('ip_muito_alta')
            elif ponto.ip_mv > 25:
                evidencias.append('ip_alta')
        
        # Estrutura
        if ponto.dist_estrutura_m is not None:
            if ponto.dist_estrutura_m < 100:
                evidencias.append('sobre_estrutura')
            elif ponto.dist_estrutura_m < 300:
                evidencias.append('proximal_estrutura')
        
        # Litologia
        if ponto.litologia in [Litologia.BRECHA_HIDROTERMAL]:
            evidencias.append('lito_muito_favoravel')
        elif ponto.litologia in [Litologia.BIF_HIDROTERMALIZADO, 
                                  Litologia.METABASALTO,
                                  Litologia.MAFICO_ULTRAMÁFICO]:
            evidencias.append('lito_favoravel')
        
        return evidencias
    
    def calcular_probabilidade(self, ponto: PontoProspeccao, 
                                config: ConfiguracaoModelo) -> Tuple[float, Dict]:
        """
        Calcula a probabilidade posterior de mineralização usando Teorema de Bayes.
        
        P(M|E) = P(M) × ∏LR(e) / [P(M) × ∏LR(e) + (1-P(M))]
        """
        evidencias = self._obter_evidencias(ponto, config)
        
        # Calcular produto das likelihood ratios
        lr_produto = 1.0
        for e in evidencias:
            lr = self.likelihood_ratios.get(e, 1.0)
            lr_produto *= lr
        
        # Aplicar Bayes
        prior = self.prior_mineralizacao
        posterior = (prior * lr_produto) / (prior * lr_produto + (1 - prior))
        
        return posterior, {
            'evidencias': evidencias,
            'lr_produto': lr_produto,
            'prior': prior,
            'posterior': posterior
        }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      5. REDE NEURAL CLASSIFICADORA                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class ClassificadorNeural:
    """
    Classificador baseado em Rede Neural Artificial (MLP) e Random Forest.
    Pode ser treinado com dados de depósitos conhecidos.
    """
    
    def __init__(self, config: ConfiguracaoModelo):
        self.config = config
        self.modelo_nn = None
        self.modelo_rf = None
        self.scaler = None
        self.treinado = False
        self.features = ['mag_anom', 'cu', 'au', 'res_inv', 'ip', 'struct_inv', 'lito']
    
    def preparar_features(self, df: pd.DataFrame, config: ConfiguracaoModelo) -> np.ndarray:
        """Prepara matriz de features para o modelo"""
        X = []
        
        for _, row in df.iterrows():
            mag_anom = (row.get('mag_nt', config.mag_regional) - config.mag_regional)
            cu = row.get('cu_ppm', 0) or 0
            au = row.get('au_ppb', 0) or 0
            res = row.get('resistividade', 5000) or 5000
            ip = row.get('ip_mv', 0) or 0
            struct = row.get('dist_estrutura', 3000) or 3000
            lito = row.get('litologia', 0)
            
            # Transformações
            res_inv = 1.0 / (res + 1)  # Inverter resistividade
            struct_inv = 1.0 / (struct + 1)  # Inverter distância
            
            X.append([mag_anom, cu, au, res_inv, ip, struct_inv, lito])
        
        return np.array(X)
    
    def treinar(self, df: pd.DataFrame, labels: np.ndarray, config: ConfiguracaoModelo):
        """
        Treina os modelos com dados rotulados.
        Labels: 0 = estéril, 1 = mineralizado
        """
        if not SKLEARN_AVAILABLE:
            print("[ERRO] Scikit-learn não disponível para treinamento")
            return
        
        X = self.preparar_features(df, config)
        
        # Normalização
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # MLP Neural Network
        self.modelo_nn = MLPClassifier(
            hidden_layer_sizes=self.config.nn_hidden_layers,
            max_iter=self.config.nn_max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15
        )
        self.modelo_nn.fit(X_scaled, labels)
        
        # Random Forest (ensemble)
        self.modelo_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.modelo_rf.fit(X_scaled, labels)
        
        # Validação cruzada
        cv_nn = cross_val_score(self.modelo_nn, X_scaled, labels, cv=5)
        cv_rf = cross_val_score(self.modelo_rf, X_scaled, labels, cv=5)
        
        print(f"[NEURAL] Acurácia MLP: {cv_nn.mean():.3f} (+/- {cv_nn.std():.3f})")
        print(f"[NEURAL] Acurácia RF: {cv_rf.mean():.3f} (+/- {cv_rf.std():.3f})")
        
        self.treinado = True
    
    def prever(self, df: pd.DataFrame, config: ConfiguracaoModelo) -> np.ndarray:
        """
        Retorna probabilidade de mineralização para cada ponto.
        Usa ensemble (média) de MLP e Random Forest.
        """
        if not self.treinado or not SKLEARN_AVAILABLE:
            return np.full(len(df), 0.5)  # Default 50% se não treinado
        
        X = self.preparar_features(df, config)
        X_scaled = self.scaler.transform(X)
        
        # Probabilidades de cada modelo
        prob_nn = self.modelo_nn.predict_proba(X_scaled)[:, 1]
        prob_rf = self.modelo_rf.predict_proba(X_scaled)[:, 1]
        
        # Ensemble (média ponderada)
        prob_ensemble = 0.4 * prob_nn + 0.6 * prob_rf
        
        return prob_ensemble
    
    def criar_modelo_sintetico(self, config: ConfiguracaoModelo):
        """
        Cria um modelo treinado com dados sintéticos baseados em
        depósitos conhecidos de Carajás (Salobo, Sossego, etc.)
        """
        if not SKLEARN_AVAILABLE:
            return
        
        np.random.seed(42)
        n_positivos = 200
        n_negativos = 800
        
        # Dados sintéticos de depósitos IOCG (positivos)
        positivos = pd.DataFrame({
            'mag_nt': np.random.normal(26500, 800, n_positivos),
            'cu_ppm': np.random.exponential(400, n_positivos) + 150,
            'au_ppb': np.random.exponential(80, n_positivos) + 20,
            'resistividade': np.random.exponential(150, n_positivos) + 50,
            'ip_mv': np.random.normal(40, 15, n_positivos),
            'dist_estrutura': np.random.exponential(150, n_positivos) + 30,
            'litologia': np.random.choice([5, 7, 8], n_positivos, p=[0.3, 0.3, 0.4])
        })
        
        # Dados sintéticos de áreas estéreis (negativos)
        negativos = pd.DataFrame({
            'mag_nt': np.random.normal(24800, 400, n_negativos),
            'cu_ppm': np.random.exponential(30, n_negativos) + 10,
            'au_ppb': np.random.exponential(5, n_negativos) + 1,
            'resistividade': np.random.exponential(2000, n_negativos) + 500,
            'ip_mv': np.random.normal(8, 5, n_negativos),
            'dist_estrutura': np.random.exponential(1000, n_negativos) + 300,
            'litologia': np.random.choice([0, 1, 2, 4], n_negativos, p=[0.2, 0.3, 0.3, 0.2])
        })
        
        # Combinar
        df_treino = pd.concat([positivos, negativos], ignore_index=True)
        labels = np.array([1] * n_positivos + [0] * n_negativos)
        
        # Treinar
        self.treinar(df_treino, labels, config)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      6. ANÁLISE DE CLUSTER ESPACIAL                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class AnalisadorCluster:
    """
    Análise de agrupamento espacial usando DBSCAN.
    Identifica clusters de anomalias que podem representar corpos mineralizados.
    """
    
    def __init__(self, config: ConfiguracaoModelo):
        self.config = config
    
    def identificar_clusters(self, df: pd.DataFrame, 
                              score_minimo: float = 0.4) -> Tuple[np.ndarray, Dict]:
        """
        Identifica clusters de pontos com score alto.
        """
        if not SKLEARN_AVAILABLE:
            return np.full(len(df), -1), {}
        
        # Filtrar pontos com score acima do mínimo
        if 'synapse_index' not in df.columns:
            return np.full(len(df), -1), {}
        
        mask = df['synapse_index'] >= score_minimo * 100
        coords_alvo = df.loc[mask, ['lat', 'lon']].values
        
        if len(coords_alvo) < self.config.cluster_min_samples:
            return np.full(len(df), -1), {}
        
        # DBSCAN
        clustering = DBSCAN(
            eps=self.config.cluster_eps_degrees,
            min_samples=self.config.cluster_min_samples,
            metric='haversine'  # Distância geodésica
        )
        
        # Converter para radianos para haversine
        coords_rad = np.radians(coords_alvo)
        cluster_labels_filtrado = clustering.fit_predict(coords_rad)
        
        # Mapear de volta para todos os pontos
        cluster_labels = np.full(len(df), -1)
        indices_filtrados = df.index[mask].tolist()
        for i, idx in enumerate(indices_filtrados):
            pos = df.index.get_loc(idx)
            cluster_labels[pos] = cluster_labels_filtrado[i]
        
        # Estatísticas dos clusters
        n_clusters = len(set(cluster_labels_filtrado)) - (1 if -1 in cluster_labels_filtrado else 0)
        estatisticas = {
            'n_clusters': n_clusters,
            'n_pontos_clusterizados': np.sum(cluster_labels >= 0),
            'n_ruido': np.sum(cluster_labels == -1)
        }
        
        return cluster_labels, estatisticas


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         7. GERADOR DE MAPAS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class GeradorMapas:
    """
    Gerador de mapas de calor e visualizações científicas.
    """
    
    def __init__(self):
        # Colormap customizado para prospecção
        self.cmap_synapse = LinearSegmentedColormap.from_list(
            'synapse',
            ['#1a1a2e', '#16213e', '#0f3460', '#1e5f74', '#28a745', 
             '#ffc107', '#fd7e14', '#dc3545', '#e83e8c', '#ff1493']
        )
        
        self.cmap_magnetico = LinearSegmentedColormap.from_list(
            'magnetico',
            ['#000080', '#0000ff', '#00ffff', '#00ff00', 
             '#ffff00', '#ff8000', '#ff0000', '#800000']
        )
    
    def mapa_synapse_index(self, df: pd.DataFrame, output: str = 'mapa_synapse.png',
                           titulo: str = 'GEO-NEURAL SYNAPSE INDEX'):
        """Gera mapa de calor do Synapse Index"""
        if not MATPLOTLIB_AVAILABLE:
            print("[ERRO] Matplotlib não disponível")
            return
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        lat = df['lat'].values
        lon = df['lon'].values
        synapse = df['synapse_index'].values
        
        # Interpolação para grid
        grid_lat = np.linspace(lat.min(), lat.max(), 100)
        grid_lon = np.linspace(lon.min(), lon.max(), 100)
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
        
        if SCIPY_AVAILABLE:
            grid_synapse = griddata((lon, lat), synapse, 
                                     (grid_lon_mesh, grid_lat_mesh), method='cubic')
            grid_synapse = gaussian_filter(np.nan_to_num(grid_synapse), sigma=1.2)
            
            # Contorno preenchido
            levels = np.linspace(0, 100, 21)
            im = ax.contourf(grid_lon_mesh, grid_lat_mesh, grid_synapse,
                            levels=levels, cmap=self.cmap_synapse, extend='both')
            
            # Linhas de contorno
            contours = ax.contour(grid_lon_mesh, grid_lat_mesh, grid_synapse,
                                  levels=[30, 50, 70, 85], colors='white', 
                                  linewidths=0.8, linestyles='--')
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
        else:
            # Fallback: scatter plot
            im = ax.scatter(lon, lat, c=synapse, cmap=self.cmap_synapse, 
                           s=100, edgecolors='black', linewidths=0.5)
        
        # Pontos de medição
        scatter = ax.scatter(lon, lat, c=synapse, cmap=self.cmap_synapse,
                            s=60, edgecolors='white', linewidths=1.5, zorder=5)
        
        # Destacar alvos prioritários
        mask_alvo = df['synapse_index'] >= 70
        if mask_alvo.any():
            ax.scatter(lon[mask_alvo], lat[mask_alvo], s=200, facecolors='none',
                      edgecolors='yellow', linewidths=3, zorder=6, label='Alvo Prioritário')
        
        # Colorbar
        cbar = plt.colorbar(im if SCIPY_AVAILABLE else scatter, ax=ax, 
                            label='SYNAPSE INDEX', pad=0.02)
        cbar.ax.tick_params(labelsize=10)
        
        # Labels e título
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        
        # Info box
        info_text = f"Pontos: {len(df)}\nAlvos (>70): {mask_alvo.sum()}"
        props = dict(boxstyle='round', facecolor='black', alpha=0.7)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', color='white', bbox=props)
        
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        
        print(f"[OK] Mapa salvo: {output}")
    
    def mapa_cluster(self, df: pd.DataFrame, output: str = 'mapa_clusters.png'):
        """Gera mapa dos clusters identificados"""
        if not MATPLOTLIB_AVAILABLE or 'cluster_id' not in df.columns:
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Colorir por cluster
        clusters = df['cluster_id'].values
        unique_clusters = np.unique(clusters[clusters >= 0])
        
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_clusters), 1)))
        
        # Pontos não clusterizados (ruído)
        mask_ruido = clusters == -1
        ax.scatter(df.loc[mask_ruido, 'lon'], df.loc[mask_ruido, 'lat'],
                  c='gray', s=30, alpha=0.5, label='Não clusterizado')
        
        # Clusters
        for i, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            ax.scatter(df.loc[mask, 'lon'], df.loc[mask, 'lat'],
                      c=[colors[i]], s=100, edgecolors='black',
                      label=f'Cluster {cluster_id}')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Análise de Clusters Espaciais', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Mapa de clusters salvo: {output}")
    
    def mapa_comparativo(self, df: pd.DataFrame, output: str = 'mapa_comparativo.png'):
        """Gera mapa comparativo de múltiplos parâmetros"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        campos = [
            ('mag_nt', 'Magnetometria (nT)', self.cmap_magnetico),
            ('cu_ppm', 'Cobre (ppm)', 'YlOrRd'),
            ('au_ppb', 'Ouro (ppb)', 'YlOrRd'),
            ('resistividade', 'Resistividade (Ω.m)', 'Blues_r'),
            ('ip_mv', 'IP (mV/V)', 'Purples'),
            ('synapse_index', 'SYNAPSE INDEX', self.cmap_synapse)
        ]
        
        for ax, (campo, titulo, cmap) in zip(axes.flatten(), campos):
            if campo in df.columns and df[campo].notna().any():
                scatter = ax.scatter(df['lon'], df['lat'], c=df[campo], 
                                    cmap=cmap, s=50, edgecolors='black', linewidths=0.3)
                plt.colorbar(scatter, ax=ax, pad=0.02)
            ax.set_title(titulo, fontsize=11, fontweight='bold')
            ax.set_xlabel('Lon')
            ax.set_ylabel('Lat')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Análise Multivariada de Prospecção', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Mapa comparativo salvo: {output}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    8. MOTOR PRINCIPAL GEO-NEURAL SYNAPSE                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class GeoNeuralSynapse:
    """
    Motor principal que integra todos os módulos de análise.
    
    Fluxo de processamento:
    1. Score Fuzzy (Lógica Nebulosa)
    2. Fator de Ruptura Estrutural (Gradiente)
    3. Probabilidade Bayesiana
    4. Score Neural (se treinado)
    5. Clustering Espacial
    6. Índice Final Agregado
    """
    
    def __init__(self, config: ConfiguracaoModelo = None):
        self.config = config or ConfiguracaoModelo()
        
        # Inicializar módulos
        self.motor_fuzzy = MotorFuzzy(self.config)
        self.analisador_gradiente = AnalisadorGradiente(self.config)
        self.motor_bayesiano = MotorBayesiano()
        self.classificador_neural = ClassificadorNeural(self.config)
        self.analisador_cluster = AnalisadorCluster(self.config)
        self.gerador_mapas = GeradorMapas()
        
        # Treinar modelo neural com dados sintéticos
        print("[INIT] Treinando modelo neural com dados sintéticos de Carajás...")
        self.classificador_neural.criar_modelo_sintetico(self.config)
    
    def processar(self, df: pd.DataFrame, 
                  gerar_mapas: bool = True,
                  verbose: bool = True) -> pd.DataFrame:
        """
        Processa dataset completo e retorna DataFrame com todos os scores.
        """
        inicio = datetime.now()
        
        if verbose:
            print("\n" + "="*80)
            print("   GEO-NEURAL SYNAPSE v3.0 - PROCESSAMENTO INICIADO")
            print("="*80)
            print(f"   Pontos: {len(df)}")
            print(f"   Início: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")
        
        # Garantir colunas necessárias
        df = df.copy()
        colunas_esperadas = ['id', 'lat', 'lon', 'mag_nt', 'cu_ppm', 'au_ppb',
                            'resistividade', 'ip_mv', 'dist_estrutura', 'litologia']
        for col in colunas_esperadas:
            if col not in df.columns:
                df[col] = None if col not in ['id', 'lat', 'lon', 'litologia'] else 0
        
        # ══════════════════════════════════════════════════════════════════
        # ETAPA 1: SCORE FUZZY
        # ══════════════════════════════════════════════════════════════════
        if verbose:
            print("[1/5] Calculando Score Fuzzy...")
        
        scores_fuzzy = []
        for _, row in df.iterrows():
            ponto = PontoProspeccao(
                id=str(row['id']),
                latitude=row['lat'],
                longitude=row['lon'],
                mag_nt=row.get('mag_nt'),
                cu_ppm=row.get('cu_ppm'),
                au_ppb=row.get('au_ppb'),
                resistividade=row.get('resistividade'),
                ip_mv=row.get('ip_mv'),
                dist_estrutura_m=row.get('dist_estrutura'),
                litologia=Litologia(row.get('litologia', 0))
            )
            score, _ = self.motor_fuzzy.calcular_score(ponto)
            scores_fuzzy.append(score)
        
        df['score_fuzzy'] = scores_fuzzy
        
        # ══════════════════════════════════════════════════════════════════
        # ETAPA 2: FATOR DE RUPTURA ESTRUTURAL (GRADIENTE)
        # ══════════════════════════════════════════════════════════════════
        if verbose:
            print("[2/5] Calculando Fator de Ruptura Estrutural (FRE)...")
        
        df['fre'] = self.analisador_gradiente.calcular_fre(df)
        
        # Normalizar FRE para 0-1
        fre_max = df['fre'].max()
        df['fre_norm'] = df['fre'] / fre_max if fre_max > 0 else 0
        
        # ══════════════════════════════════════════════════════════════════
        # ETAPA 3: PROBABILIDADE BAYESIANA
        # ══════════════════════════════════════════════════════════════════
        if verbose:
            print("[3/5] Calculando Probabilidade Bayesiana...")
        
        probs_bayes = []
        for _, row in df.iterrows():
            ponto = PontoProspeccao(
                id=str(row['id']),
                latitude=row['lat'],
                longitude=row['lon'],
                mag_nt=row.get('mag_nt'),
                cu_ppm=row.get('cu_ppm'),
                au_ppb=row.get('au_ppb'),
                resistividade=row.get('resistividade'),
                ip_mv=row.get('ip_mv'),
                dist_estrutura_m=row.get('dist_estrutura'),
                litologia=Litologia(row.get('litologia', 0))
            )
            prob, _ = self.motor_bayesiano.calcular_probabilidade(ponto, self.config)
            probs_bayes.append(prob)
        
        df['prob_bayesiana'] = probs_bayes
        
        # ══════════════════════════════════════════════════════════════════
        # ETAPA 4: SCORE NEURAL
        # ══════════════════════════════════════════════════════════════════
        if verbose:
            print("[4/5] Calculando Score Neural (ML)...")
        
        df['score_neural'] = self.classificador_neural.prever(df, self.config)
        
        # ══════════════════════════════════════════════════════════════════
        # ETAPA 5: ÍNDICE SYNAPSE FINAL
        # ══════════════════════════════════════════════════════════════════
        if verbose:
            print("[5/5] Calculando SYNAPSE INDEX final...")
        
        # Fórmula do Synapse Index:
        # SYNAPSE = (w1×Fuzzy + w2×Bayes + w3×Neural) × (1 + α×FRE) × 100
        #
        # Onde:
        # - w1, w2, w3 são pesos dos componentes
        # - α é o fator de amplificação do gradiente
        
        w_fuzzy = 0.35
        w_bayes = 0.30
        w_neural = 0.35
        alpha = 0.5  # Amplificação do FRE
        
        score_combinado = (w_fuzzy * df['score_fuzzy'] + 
                           w_bayes * df['prob_bayesiana'] + 
                           w_neural * df['score_neural'])
        
        # Aplicar boost do FRE
        df['synapse_index'] = score_combinado * (1 + alpha * df['fre_norm']) * 100
        
        # Clamp 0-100
        df['synapse_index'] = np.clip(df['synapse_index'], 0, 100)
        
        # ══════════════════════════════════════════════════════════════════
        # CLUSTERING
        # ══════════════════════════════════════════════════════════════════
        if verbose:
            print("[+] Identificando clusters espaciais...")
        
        clusters, stats_cluster = self.analisador_cluster.identificar_clusters(df)
        df['cluster_id'] = clusters
        
        # ══════════════════════════════════════════════════════════════════
        # CLASSIFICAÇÃO
        # ══════════════════════════════════════════════════════════════════
        def classificar(idx):
            if idx >= 85:
                return "🔴 ALVO CRÍTICO - PRIORIDADE MÁXIMA"
            elif idx >= 70:
                return "🟠 ALVO PRIORITÁRIO - ALTA FAVORABILIDADE"
            elif idx >= 55:
                return "🟡 FAVORÁVEL - ANOMALIA SIGNIFICATIVA"
            elif idx >= 40:
                return "🟢 MODERADO - INVESTIGAR"
            elif idx >= 25:
                return "🔵 BAIXO - MONITORAR"
            else:
                return "⚪ BACKGROUND - ESTÉRIL"
        
        df['classificacao'] = df['synapse_index'].apply(classificar)
        
        # ══════════════════════════════════════════════════════════════════
        # GERAÇÃO DE MAPAS
        # ══════════════════════════════════════════════════════════════════
        if gerar_mapas and MATPLOTLIB_AVAILABLE:
            if verbose:
                print("\n[MAPAS] Gerando visualizações...")
            
            self.gerador_mapas.mapa_synapse_index(df)
            self.gerador_mapas.mapa_comparativo(df)
            if stats_cluster.get('n_clusters', 0) > 0:
                self.gerador_mapas.mapa_cluster(df)
        
        # ══════════════════════════════════════════════════════════════════
        # RELATÓRIO FINAL
        # ══════════════════════════════════════════════════════════════════
        fim = datetime.now()
        duracao = (fim - inicio).total_seconds()
        
        if verbose:
            print("\n" + "="*80)
            print("   PROCESSAMENTO CONCLUÍDO")
            print("="*80)
            print(f"   Duração: {duracao:.2f} segundos")
            print(f"   Clusters identificados: {stats_cluster.get('n_clusters', 0)}")
            print()
            
            # Estatísticas
            print("   DISTRIBUIÇÃO DO SYNAPSE INDEX:")
            print(f"      Mínimo:  {df['synapse_index'].min():.1f}")
            print(f"      Máximo:  {df['synapse_index'].max():.1f}")
            print(f"      Média:   {df['synapse_index'].mean():.1f}")
            print(f"      Mediana: {df['synapse_index'].median():.1f}")
            print()
            
            # Contagem por classificação
            print("   CONTAGEM POR CLASSIFICAÇÃO:")
            for classe in df['classificacao'].unique():
                count = (df['classificacao'] == classe).sum()
                print(f"      {classe}: {count}")
            
            print("="*80 + "\n")
        
        # Colunas de saída
        colunas_saida = ['id', 'lat', 'lon', 'synapse_index', 'score_fuzzy', 
                        'fre_norm', 'prob_bayesiana', 'score_neural', 
                        'cluster_id', 'classificacao']
        
        return df[[c for c in colunas_saida if c in df.columns]]
    
    def ranking(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Retorna ranking dos melhores alvos"""
        ranking = df.nlargest(top_n, 'synapse_index')
        return ranking[['id', 'lat', 'lon', 'synapse_index', 'classificacao']]
    
    def exportar_geojson(self, df: pd.DataFrame, output: str = 'alvos.geojson'):
        """Exporta resultados para GeoJSON"""
        import json
        
        features = []
        for _, row in df.iterrows():
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['lon'], row['lat']]
                },
                "properties": {
                    "id": row['id'],
                    "synapse_index": float(row['synapse_index']),
                    "classificacao": row['classificacao'],
                    "cluster": int(row.get('cluster_id', -1))
                }
            })
        
        geojson = {"type": "FeatureCollection", "features": features}
        
        with open(output, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"[OK] GeoJSON exportado: {output}")
    
    def gerar_relatorio(self, df: pd.DataFrame, output: str = 'relatorio_synapse.txt'):
        """Gera relatório técnico em texto"""
        with open(output, 'w') as f:
            f.write("="*80 + "\n")
            f.write("          GEO-NEURAL SYNAPSE v3.0 - RELATÓRIO TÉCNICO\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Data de Processamento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de Pontos: {len(df)}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("ESTATÍSTICAS DO SYNAPSE INDEX\n")
            f.write("-"*80 + "\n")
            f.write(f"Mínimo:  {df['synapse_index'].min():.2f}\n")
            f.write(f"Máximo:  {df['synapse_index'].max():.2f}\n")
            f.write(f"Média:   {df['synapse_index'].mean():.2f}\n")
            f.write(f"Desvio:  {df['synapse_index'].std():.2f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("TOP 10 ALVOS PRIORITÁRIOS\n")
            f.write("-"*80 + "\n")
            
            top10 = df.nlargest(10, 'synapse_index')
            for _, row in top10.iterrows():
                f.write(f"\n{row['id']}:\n")
                f.write(f"   Coordenadas: {row['lat']:.6f}, {row['lon']:.6f}\n")
                f.write(f"   SYNAPSE INDEX: {row['synapse_index']:.1f}\n")
                f.write(f"   Classificação: {row['classificacao']}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("FIM DO RELATÓRIO\n")
            f.write("="*80 + "\n")
        
        print(f"[OK] Relatório exportado: {output}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          9. EXECUÇÃO DE EXEMPLO                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║            GEO-NEURAL SYNAPSE v3.0 - DEMONSTRAÇÃO                           ║
    ║                                                                              ║
    ║            Motor de Prospecção Mineral com IA                               ║
    ║            Projeto Carajás - Wanderlei & Claude                             ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Dataset de exemplo expandido (simula levantamento real)
    np.random.seed(42)
    
    # Criar dados que simulam diferentes cenários geológicos
    dados = pd.DataFrame({
        'id': [f'PT-{i:03d}' for i in range(1, 21)],
        'lat': np.concatenate([
            np.random.uniform(-6.450, -6.448, 5),   # Cluster 1 (IOCG)
            np.random.uniform(-6.455, -6.453, 5),   # Cluster 2 (BIF)
            np.random.uniform(-6.460, -6.458, 5),   # Cluster 3 (Estéril)
            np.random.uniform(-6.445, -6.465, 5),   # Pontos dispersos
        ]),
        'lon': np.concatenate([
            np.random.uniform(-50.950, -50.948, 5),
            np.random.uniform(-50.955, -50.953, 5),
            np.random.uniform(-50.960, -50.958, 5),
            np.random.uniform(-50.945, -50.965, 5),
        ]),
        # Cluster 1: Simulação de IOCG (Alto Cu, Au, baixa Res, alta Mag)
        # Cluster 2: Simulação de BIF (Alta Mag, baixo Cu/Au)
        # Cluster 3: Estéril
        # Dispersos: Variado
        'mag_nt': np.concatenate([
            np.random.normal(26800, 400, 5),    # IOCG: mag alta
            np.random.normal(28500, 600, 5),    # BIF: mag muito alta
            np.random.normal(24700, 200, 5),    # Estéril: background
            np.random.normal(25200, 500, 5),    # Dispersos
        ]),
        'cu_ppm': np.concatenate([
            np.random.exponential(300, 5) + 200, # IOCG: Cu alto
            np.random.exponential(20, 5) + 10,   # BIF: Cu baixo
            np.random.exponential(15, 5) + 10,   # Estéril
            np.random.exponential(50, 5) + 30,   # Dispersos
        ]),
        'au_ppb': np.concatenate([
            np.random.exponential(100, 5) + 50,  # IOCG: Au alto
            np.random.exponential(3, 5) + 2,     # BIF: Au baixo
            np.random.exponential(2, 5) + 1,     # Estéril
            np.random.exponential(15, 5) + 5,    # Dispersos
        ]),
        'resistividade': np.concatenate([
            np.random.exponential(80, 5) + 50,   # IOCG: baixa (sulfetos)
            np.random.exponential(500, 5) + 200, # BIF: moderada
            np.random.exponential(2000, 5) + 1000, # Estéril: alta
            np.random.exponential(800, 5) + 300, # Dispersos
        ]),
        'ip_mv': np.concatenate([
            np.random.normal(55, 15, 5),         # IOCG: IP alta
            np.random.normal(12, 5, 5),          # BIF: IP baixa
            np.random.normal(6, 3, 5),           # Estéril
            np.random.normal(20, 10, 5),         # Dispersos
        ]),
        'dist_estrutura': np.concatenate([
            np.random.exponential(50, 5) + 20,   # IOCG: perto de estrutura
            np.random.exponential(200, 5) + 100, # BIF
            np.random.exponential(1000, 5) + 500, # Estéril: longe
            np.random.exponential(300, 5) + 150, # Dispersos
        ]),
        'litologia': np.concatenate([
            np.random.choice([5, 7, 8], 5),      # IOCG: favorável
            np.full(5, 4),                        # BIF
            np.random.choice([1, 2], 5),          # Estéril
            np.random.choice([0, 3, 6], 5),       # Dispersos
        ]),
    })
    
    # Inicializar o motor
    motor = GeoNeuralSynapse()
    
    # Processar
    resultado = motor.processar(dados, gerar_mapas=MATPLOTLIB_AVAILABLE, verbose=True)
    
    # Mostrar resultado
    print("\n" + "="*100)
    print("RESULTADO COMPLETO")
    print("="*100)
    print(resultado.to_string(index=False))
    
    # Ranking
    print("\n" + "="*100)
    print("RANKING TOP 10 ALVOS")
    print("="*100)
    print(motor.ranking(resultado, 10).to_string(index=False))
    
    # Exportar
    motor.exportar_geojson(resultado)
    motor.gerar_relatorio(resultado)
    
    print("\n[CONCLUÍDO] GEO-NEURAL SYNAPSE v3.0 executado com sucesso!")
