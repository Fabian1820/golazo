"""Guardián de fugas de datos.

El bug que invalidaba el modelo original era estructural: se alimentaban
estadísticas del propio partido a predecir. Estos tests hacen imposible que
vuelva a colarse sin que el CI lo detecte.
"""
from __future__ import annotations

import numpy as np
import pytest

from golazo.data import POST_MATCH_COLS, load_matches
from golazo.features import build_features, feature_columns


@pytest.fixture(scope="module")
def df():
    return build_features(load_matches())


def test_ninguna_columna_post_partido_es_feature(df):
    """Chequeo estructural: ninguna estadística de resultado entra como entrada."""
    cols = set(feature_columns(df))
    filtradas = cols & set(POST_MATCH_COLS)
    assert not filtradas, f"features post-partido filtradas: {sorted(filtradas)}"


def test_features_no_dependen_del_resultado_propio():
    """Chequeo de comportamiento, el que de verdad importa.

    Se altera el resultado de un partido y se comprueba que SUS PROPIAS
    features no cambian. Si cambiaran, el modelo estaría viendo el futuro.
    """
    base = load_matches()
    f0 = build_features(base)
    cols = [c for c in feature_columns(f0) if c != "league"]

    objetivo = 5000
    alterado = base.copy()
    for c in POST_MATCH_COLS:
        alterado.loc[objetivo, c] = float(alterado.loc[objetivo, c]) + 7.0
    f1 = build_features(alterado)

    fila0 = f0.loc[objetivo, cols].to_numpy(dtype=float)
    fila1 = f1.loc[objetivo, cols].to_numpy(dtype=float)
    np.testing.assert_allclose(fila0, fila1, rtol=0, atol=0,
                               err_msg="las features del partido cambian con su propio resultado: FUGA")


def test_el_resultado_si_afecta_a_partidos_posteriores():
    """Contraprueba: si nada cambiara nunca, el test anterior sería vacío."""
    base = load_matches()
    f0 = build_features(base)
    cols = [c for c in feature_columns(f0) if c != "league"]

    objetivo = 5000
    alterado = base.copy()
    for c in POST_MATCH_COLS:
        alterado.loc[objetivo, c] = float(alterado.loc[objetivo, c]) + 7.0
    f1 = build_features(alterado)

    post0 = f0.loc[objetivo + 1:, cols].to_numpy(dtype=float)
    post1 = f1.loc[objetivo + 1:, cols].to_numpy(dtype=float)
    difs = ~np.isclose(post0, post1, equal_nan=True)
    assert difs.any(), "alterar un partido no afectó a ninguno posterior: las medias móviles no funcionan"


def test_ultimo_partido_no_influye_en_nada(df):
    """Alterar el último partido del dataset no puede cambiar ninguna feature."""
    base = load_matches()
    f0 = build_features(base)
    cols = [c for c in feature_columns(f0) if c != "league"]

    ultimo = len(base) - 1
    alterado = base.copy()
    for c in POST_MATCH_COLS:
        alterado.loc[ultimo, c] = float(alterado.loc[ultimo, c]) + 7.0

    a = f0[cols].to_numpy(dtype=float)
    b = build_features(alterado)[cols].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg="el último partido influyó en features: FUGA")
