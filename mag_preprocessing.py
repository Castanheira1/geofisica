#!/usr/bin/env python3
"""
PROSPECTOR-AI - Módulo de Pré-Processamento Aeromagnético
Filtros de Micronivelamento e Decorrugação com Suporte UTM
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.interpolate import griddata
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pyproj import CRS, Transformer
import json

@dataclass
class MagPoint:
    id: str
    latitude: float
    longitude: float
    mag_nt: float
    linha: int = 0
    
@dataclass
class FilteredPoint:
    id: str
    latitude: float
    longitude: float
    mag_raw: float
    mag_filtered: float
    correcao_nt: float
    filtro_aplicado: str

class MagPreProcessor:
    def __init__(self, cell_size_m: float = 50.0):
        self.cell_size = cell_size_m
        self.grid_raw = None
        self.x = None
        self.y = None
        self.transformer = None
        self.crs_utm = None
        
    def _get_utm_crs(self, lon: float, lat: float) -> CRS:
        zone = int((lon + 180) / 6) + 1
        hemisphere = 'south' if lat < 0 else 'north'
        return CRS.from_dict({'proj': 'utm', 'zone': zone, hemisphere: True})
    
    def _setup_projection(self, lons: np.ndarray, lats: np.ndarray):
        lon_med = np.median(lons)
        lat_med = np.median(lats)
        self.crs_utm = self._get_utm_crs(lon_med, lat_med)
        self.transformer = Transformer.from_crs("epsg:4326", self.crs_utm, always_xy=True)
        
    def _to_xy(self, lon, lat):
        return self.transformer.transform(lon, lat)
        
    def points_to_grid(self, points: List[MagPoint]) -> np.ndarray:
        lats = np.array([p.latitude for p in points])
        lons = np.array([p.longitude for p in points])
        vals = np.array([p.mag_nt for p in points])
        
        self._setup_projection(lons, lats)
        x, y = self._to_xy(lons, lats)
        
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        
        self.x = np.arange(xmin, xmax, self.cell_size)
        self.y = np.arange(ymin, ymax, self.cell_size)
        
        if len(self.x) < 3 or len(self.y) < 3:
            self.x = np.linspace(xmin, xmax, max(10, len(points)//2))
            self.y = np.linspace(ymin, ymax, max(10, len(points)//2))
        
        grid_x, grid_y = np.meshgrid(self.x, self.y)
        
        try:
            grid = griddata((x, y), vals, (grid_x, grid_y), method='cubic')
        except:
            grid = griddata((x, y), vals, (grid_x, grid_y), method='linear')
        
        if np.isnan(grid).any():
            grid[np.isnan(grid)] = np.nanmedian(vals)
            
        self.grid_raw = grid
        return grid
    
    def grid_to_points(self, grid: np.ndarray, original_points: List[MagPoint], filtro: str) -> List[FilteredPoint]:
        results = []
        for p in original_points:
            xp, yp = self._to_xy(p.longitude, p.latitude)
            
            ix = np.argmin(np.abs(self.x - xp))
            iy = np.argmin(np.abs(self.y - yp))
            
            ix = min(ix, grid.shape[1] - 1)
            iy = min(iy, grid.shape[0] - 1)
            
            val = grid[iy, ix]
            if np.isnan(val):
                val = p.mag_nt
                
            results.append(FilteredPoint(
                id=p.id,
                latitude=p.latitude,
                longitude=p.longitude,
                mag_raw=p.mag_nt,
                mag_filtered=float(val),
                correcao_nt=float(val - p.mag_nt),
                filtro_aplicado=filtro
            ))
        return results

    def filtro_minty(self, grid: np.ndarray, direcao_voo: float = 0, fator: float = 0.6) -> np.ndarray:
        f_grid = fftshift(fft2(grid))
        ny, nx = grid.shape
        cy, cx = ny // 2, nx // 2
        
        y, x = np.ogrid[:ny, :nx]
        y = y - cy
        x = x - cx
        
        theta = np.deg2rad(direcao_voo)
        dist = np.abs(x * np.cos(theta) - y * np.sin(theta))
        dmax = np.sqrt(cx**2 + cy**2)
        
        mask = 1 - fator * np.exp(-(dist**2) / (0.15 * dmax)**2)
        return np.real(ifft2(ifftshift(f_grid * mask)))
    
    def filtro_espectral_direcional(self, grid: np.ndarray, direcao: float = 0, largura: float = 30) -> np.ndarray:
        f_grid = fftshift(fft2(grid))
        ny, nx = grid.shape
        cy, cx = ny // 2, nx // 2
        
        y, x = np.ogrid[:ny, :nx]
        y = y - cy
        x = x - cx
        
        angulo = np.arctan2(y, x)
        theta = np.deg2rad(direcao)
        largura_rad = np.deg2rad(largura)
        
        diff1 = np.abs(angulo - theta)
        diff2 = np.abs(angulo - theta + np.pi)
        diff3 = np.abs(angulo - theta - np.pi)
        diff = np.minimum(np.minimum(diff1, diff2), diff3)
        
        mask = 1 - np.exp(-diff**2 / (2 * largura_rad**2))
        mask[cy, cx] = 1
        
        return np.real(ifft2(ifftshift(f_grid * mask)))
    
    def filtro_passa_banda(self, grid: np.ndarray, lambda_min_m: float = 100.0, lambda_max_m: float = 5000.0) -> np.ndarray:
        ny, nx = grid.shape
        
        freq_x = np.fft.fftfreq(nx, d=self.cell_size)
        freq_y = np.fft.fftfreq(ny, d=self.cell_size)
        fx, fy = np.meshgrid(freq_x, freq_y)
        freq_r = np.sqrt(fx**2 + fy**2)
        
        f_min = 1.0 / lambda_max_m
        f_max = 1.0 / lambda_min_m
        
        with np.errstate(divide='ignore', invalid='ignore'):
            mask_high = 1 - np.exp(-(freq_r**2) / (2 * f_min**2))
            mask_low = np.exp(-(freq_r**2) / (2 * f_max**2))
        
        mask = mask_high * mask_low
        mask[0, 0] = 0
        
        f_grid = fft2(grid)
        return np.real(ifft2(f_grid * mask))
    
    def filtro_estatistico_linhas(self, points: List[MagPoint]) -> List[FilteredPoint]:
        linhas = {}
        for p in points:
            if p.linha not in linhas:
                linhas[p.linha] = []
            linhas[p.linha].append(p)
        
        if len(linhas) <= 1:
            return [FilteredPoint(
                id=p.id, latitude=p.latitude, longitude=p.longitude,
                mag_raw=p.mag_nt, mag_filtered=p.mag_nt,
                correcao_nt=0, filtro_aplicado='estatistico_linhas'
            ) for p in points]
        
        medianas = {linha: np.median([p.mag_nt for p in pts]) for linha, pts in linhas.items()}
        mediana_global = np.median(list(medianas.values()))
        correcoes = {linha: mediana_global - med for linha, med in medianas.items()}
        
        return [FilteredPoint(
            id=p.id, latitude=p.latitude, longitude=p.longitude,
            mag_raw=p.mag_nt, mag_filtered=p.mag_nt + correcoes.get(p.linha, 0),
            correcao_nt=correcoes.get(p.linha, 0), filtro_aplicado='estatistico_linhas'
        ) for p in points]
    
    def filtro_mediana(self, grid: np.ndarray, size: int = 3) -> np.ndarray:
        return ndimage.median_filter(grid, size=size)
    
    def filtro_gaussiano(self, grid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return ndimage.gaussian_filter(grid, sigma=sigma)
    
    def remover_spikes(self, points: List[MagPoint], n_mad: float = 5.0) -> List[MagPoint]:
        vals = np.array([p.mag_nt for p in points])
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        
        if mad == 0:
            return points
        
        low, high = med - n_mad * mad, med + n_mad * mad
        
        return [p if low <= p.mag_nt <= high else 
                MagPoint(id=p.id, latitude=p.latitude, longitude=p.longitude, mag_nt=med, linha=p.linha)
                for p in points]

    def processar_completo(self, points: List[MagPoint], filtros: List[str] = None, direcao_voo: float = 0) -> dict:
        if filtros is None:
            filtros = ['minty', 'passa_banda', 'mediana']
        
        resultados = {'raw': points, 'filtrados': {}}
        
        if len(points) < 3:
            return resultados
        
        grid = self.points_to_grid(points)
        
        filtro_map = {
            'minty': lambda g: self.filtro_minty(g, direcao_voo),
            'espectral': lambda g: self.filtro_espectral_direcional(g, direcao_voo),
            'passa_banda': lambda g: self.filtro_passa_banda(g),
            'mediana': lambda g: self.filtro_mediana(g),
            'gaussiano': lambda g: self.filtro_gaussiano(g),
        }
        
        for filtro in filtros:
            if filtro == 'estatistico':
                resultados['filtrados']['estatistico'] = self.filtro_estatistico_linhas(points)
                continue
                
            if filtro not in filtro_map:
                continue
                
            grid_f = filtro_map[filtro](grid)
            resultados['filtrados'][filtro] = self.grid_to_points(grid_f, points, filtro)
        
        return resultados
