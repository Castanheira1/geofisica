#!/usr/bin/env python3
"""
PROSPECTOR-AI - Processamento de anomalia magnética (campo potencial)

Substitui o raciocínio incorreto por VALOR ABSOLUTO de campo total (nT) por
quantidades fisicamente interpretáveis: a ANOMALIA RESIDUAL e produtos
direção-independentes (sinal analítico, tilt, gradiente horizontal total).

Base teórica e por que isto importa (ver RELATORIO_TECNICO.md, seção 1):

  * O campo total (TMI) é dominado pelo IGRF (campo principal da Terra,
    ~25.000-60.000 nT), que varia com latitude, altitude e época. O valor
    absoluto NÃO é assinatura de rocha; só a anomalia residual (pós-IGRF,
    pós-correção diurna) representa as fontes crustais.
    - Blakely (1995), "Potential Theory in Gravity and Magnetic Applications".
    - Geoscience Australia, TMI Grid of Australia 2019 (remoção de IGRF + diurna).

  * Em Carajás (~-6 deg, perto do equador magnético) a Redução ao Polo (RTP)
    é instável (< ~15 deg de inclinacao). Por isso usamos o SINAL ANALÍTICO
    (gradiente total), cuja amplitude pica sobre as bordas das fontes e é
    INDEPENDENTE da direcao de magnetizacao.
    - Silva (1986), GEOPHYSICS; "Analytic Signal vs. Reduction to Pole".

  * Ferramentas de referência (para o pipeline-alvo): Fatiando/Harmonica e
    SimPEG. Este módulo implementa o subconjunto essencial em numpy/scipy para
    manter o deploy leve; a migracao para Harmonica está no plano (Fase 2).

ATENÇÃO: a remoção rigorosa do IGRF exige um modelo IGRF (ex.: pacote `ppigrf`
ou `pyIGRF`). Quando esse modelo não está disponível, usamos um baseline
regional local (mediana/tendência sobre uma janela de pontos) como APROXIMAÇÃO,
e isso é sinalizado no resultado (`baseline_metodo`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# IGRF aproximado do campo total para a região de Carajás (~ -6 lat, -50 lon),
# época ~2020-2025, ao nível do solo. Usado APENAS como baseline padrão quando
# nenhum modelo IGRF está disponível. NÃO é um substituto do modelo real.
IGRF_CARAJAS_NT_APROX = 24800.0


@dataclass
class AnomaliaPontual:
    """Anomalia residual de um único ponto, com proveniência honesta."""
    campo_total_nt: float
    baseline_nt: float
    anomalia_nt: float            # campo_total - baseline (residual)
    baseline_metodo: str          # 'igrf', 'regional_local' ou 'igrf_aprox'
    confiavel: bool               # False se baseline é grosseiro (poucos pontos / aprox)

    def resumo(self) -> dict:
        return {
            "campo_total_nt": self.campo_total_nt,
            "anomalia_nt": self.anomalia_nt,
            "baseline_nt": self.baseline_nt,
            "baseline_metodo": self.baseline_metodo,
            "confiavel": self.confiavel,
        }


def anomalia_residual(
    campo_total_nt: float,
    baseline_nt: Optional[float] = None,
    janela_local_nt: Optional[Sequence[float]] = None,
) -> AnomaliaPontual:
    """Calcula a anomalia residual = campo total - baseline.

    Ordem de preferência do baseline:
      1. `baseline_nt` explícito (idealmente o IGRF do ponto via modelo).
      2. mediana de `janela_local_nt` (baseline regional local) — aproximação
         válida para realçar variações de curto comprimento de onda dentro de
         um levantamento pequeno, mas NÃO é o IGRF.
      3. IGRF aproximado de Carajás (grosseiro) — resultado marcado como NÃO
         confiável.

    A anomalia (e não o campo total) é a quantidade interpretável.
    """
    if baseline_nt is not None:
        return AnomaliaPontual(
            campo_total_nt=campo_total_nt,
            baseline_nt=baseline_nt,
            anomalia_nt=campo_total_nt - baseline_nt,
            baseline_metodo="igrf",
            confiavel=True,
        )

    if janela_local_nt is not None and len(janela_local_nt) >= 8:
        base = float(np.median(np.asarray(janela_local_nt, dtype=float)))
        return AnomaliaPontual(
            campo_total_nt=campo_total_nt,
            baseline_nt=base,
            anomalia_nt=campo_total_nt - base,
            baseline_metodo="regional_local",
            confiavel=True,
        )

    return AnomaliaPontual(
        campo_total_nt=campo_total_nt,
        baseline_nt=IGRF_CARAJAS_NT_APROX,
        anomalia_nt=campo_total_nt - IGRF_CARAJAS_NT_APROX,
        baseline_metodo="igrf_aprox",
        confiavel=False,
    )


# ---------------------------------------------------------------------------
# Produtos de realce em GRADE (quando há um conjunto de pontos / levantamento)
# ---------------------------------------------------------------------------

def gradiente_horizontal_total(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """THG = sqrt((dT/dx)^2 + (dT/dy)^2). Máximos delineiam contatos/bordas."""
    gy, gx = np.gradient(np.asarray(grid, dtype=float), dy, dx)
    return np.sqrt(gx**2 + gy**2)


def derivada_vertical(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Derivada vertical (dT/dz) via domínio de Fourier (continuação infinitesimal).

    Para campos potenciais, no domínio do número de onda, dT/dz <-> |k| * F(T).
    """
    g = np.asarray(grid, dtype=float)
    ny, nx = g.shape
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    k = np.sqrt(KX**2 + KY**2)
    F = np.fft.fft2(g)
    return np.real(np.fft.ifft2(F * k))


def sinal_analitico(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Amplitude do sinal analítico (gradiente total 3D):

        AS = sqrt((dT/dx)^2 + (dT/dy)^2 + (dT/dz)^2)

    Independente da direção de magnetização/campo -> apropriado para BAIXAS
    LATITUDES (Carajás), onde a RTP é instável. Picos sobre bordas das fontes.
    """
    g = np.asarray(grid, dtype=float)
    gy, gx = np.gradient(g, dy, dx)
    gz = derivada_vertical(g, dx, dy)
    return np.sqrt(gx**2 + gy**2 + gz**2)


def tilt_derivative(grid: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Tilt angle = atan2(dT/dz, THG). Limitado a [-pi/2, pi/2]; realça
    anomalias fracas/profundas; o contorno zero acompanha bordas das fontes.
    (Miller & Singh, 1994.)
    """
    g = np.asarray(grid, dtype=float)
    thg = gradiente_horizontal_total(g, dx, dy)
    gz = derivada_vertical(g, dx, dy)
    return np.arctan2(gz, thg + 1e-12)


def continuacao_para_cima(grid: np.ndarray, altura: float, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Continuação para cima de `altura` (mesma unidade de dx/dy). Atenua ruído
    raso de curto comprimento de onda; usada em separação regional-residual.
    Operador no domínio do número de onda: exp(-|k| * altura).
    """
    g = np.asarray(grid, dtype=float)
    ny, nx = g.shape
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    k = np.sqrt(KX**2 + KY**2)
    F = np.fft.fft2(g)
    return np.real(np.fft.ifft2(F * np.exp(-k * altura)))
