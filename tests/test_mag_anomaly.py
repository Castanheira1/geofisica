import math

import numpy as np

from mag_anomaly import (
    anomalia_residual,
    sinal_analitico,
    tilt_derivative,
    gradiente_horizontal_total,
    continuacao_para_cima,
)


def test_baseline_explicito_confiavel():
    a = anomalia_residual(25500.0, baseline_nt=24800.0)
    assert a.baseline_metodo == "igrf"
    assert a.confiavel is True
    assert a.anomalia_nt == 700.0


def test_baseline_local():
    janela = [24800, 24810, 24790, 24820, 24805, 24815, 24795, 24808, 25300]
    a = anomalia_residual(25300.0, janela_local_nt=janela)
    assert a.baseline_metodo == "regional_local"
    assert a.confiavel is True
    # anomalia = 25300 - mediana(janela)
    assert a.anomalia_nt == 25300.0 - float(np.median(janela))


def test_baseline_aproximado_nao_confiavel():
    a = anomalia_residual(26000.0)
    assert a.baseline_metodo == "igrf_aprox"
    assert a.confiavel is False  # honestidade: baseline grosseiro -> não confiável


def test_sinal_analitico_propriedades():
    y, x = np.mgrid[0:40, 0:40]
    grid = 500.0 * np.exp(-((x - 20) ** 2 + (y - 20) ** 2) / 30.0)
    AS = sinal_analitico(grid)
    assert AS.shape == grid.shape
    assert np.all(np.isfinite(AS))
    assert np.all(AS >= 0)  # amplitude é não-negativa


def test_tilt_limitado():
    y, x = np.mgrid[0:30, 0:30]
    grid = 100.0 * np.exp(-((x - 15) ** 2 + (y - 15) ** 2) / 20.0)
    t = tilt_derivative(grid)
    assert np.all(t >= -math.pi / 2 - 1e-6)
    assert np.all(t <= math.pi / 2 + 1e-6)


def test_thg_e_continuacao_finitos():
    grid = np.random.RandomState(0).normal(0, 10, (32, 32))
    assert np.all(np.isfinite(gradiente_horizontal_total(grid)))
    assert np.all(np.isfinite(continuacao_para_cima(grid, altura=2.0)))
