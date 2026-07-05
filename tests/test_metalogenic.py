from metalogenic_context import analisar_metalogenia, TipoDeposito


def test_iocg_magnetita():
    a = analisar_metalogenia(-6.0, -50.1, 1500.0, cu_ppm=800, au_ppb=150,
                             litologia_codigo=7, distancia_estrutura_m=300)
    assert a.tipo_mais_provavel == TipoDeposito.IOCG
    assert a.score_maximo > 0.7


def test_iocg_hematita_minimo_magnetico_nao_zera():
    # Correção-chave: IOCG hematítico pode ser MÍNIMO magnético (anomalia negativa).
    # Não deve ser descartado quando a geoquímica de Cu é forte.
    a = analisar_metalogenia(-6.0, -50.1, -400.0, cu_ppm=900, au_ppb=120,
                             litologia_codigo=7, distancia_estrutura_m=400)
    assert a.tipo_mais_provavel == TipoDeposito.IOCG
    assert a.score_maximo > 0.5


def test_bif_anomalia_forte():
    a = analisar_metalogenia(-6.0, -50.1, 12000.0, cu_ppm=5, au_ppb=1,
                             litologia_codigo=4, distancia_estrutura_m=2000)
    assert a.tipo_mais_provavel == TipoDeposito.BIF_HOSTED


def test_orogenic_gold():
    a = analisar_metalogenia(-6.0, -50.1, -50.0, cu_ppm=20, au_ppb=3000,
                             litologia_codigo=6, distancia_estrutura_m=200)
    assert a.tipo_mais_provavel == TipoDeposito.OROGENIC_GOLD


def test_usa_anomalia_nao_campo_total():
    # Mesmo "campo total" alto (28000) interpretado como anomalia ~0 (sem baseline
    # removido) NÃO deve cravar IOCG por causa do nT absoluto.
    a = analisar_metalogenia(-6.0, -50.1, 0.0, litologia_codigo=0,
                             distancia_estrutura_m=99999)
    # Sem geoquímica/estrutura/litologia e anomalia nula -> não é alvo de alta convicção.
    assert a.score_maximo < 0.75
