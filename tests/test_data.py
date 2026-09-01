"""Integridad de la carga y, sobre todo, temporalidad del Elo.

El Elo sólo es utilizable si es la valoración PREVIA al partido. Si fuera la
posterior, ya incorporaría el resultado y sería otra fuga.

Desde la Fase 2 el Elo no se lee de `ClubElo.csv` sino que se calcula en
`golazo.elo` sobre el propio historial; la comprobación de temporalidad sigue
siendo exactamente la misma y sigue siendo obligatoria.
"""
from __future__ import annotations

import numpy as np
import pytest

from golazo.data import POST_MATCH_COLS, PRE_MATCH_COLS, load_matches


@pytest.fixture(scope="module")
def df():
    return load_matches()


def test_todo_partido_tiene_elo(df):
    assert len(df) > 9000
    assert df["elo_h"].notna().all()
    assert df["elo_a"].notna().all()


def test_solo_se_cargan_partidos_jugados(df):
    """El almacén contiene también el calendario; `load_matches` no debe traerlo."""
    assert df["home_goals"].notna().all()
    assert df["away_goals"].notna().all()


def test_incluye_datos_de_ambas_fuentes(df):
    """El histórico de Kaggle y el refresco de Understat conviven en el almacén."""
    fuentes = set(df["match_id"].str.split(":").str[0])
    assert "kaggle" in fuentes


def test_columnas_esperadas(df):
    for c in PRE_MATCH_COLS + POST_MATCH_COLS + ["result"]:
        assert c in df.columns


def test_orden_cronologico(df):
    assert df["date"].is_monotonic_increasing


def test_resultado_coherente_con_los_goles(df):
    esperado = np.where(df["home_goals"] > df["away_goals"], "H",
                        np.where(df["home_goals"] < df["away_goals"], "A", "D"))
    assert (df["result"].to_numpy() == esperado).all()


def test_el_elo_es_previo_al_partido(df):
    """Prueba decisiva.

    Para cada equipo se ordena su serie de Elo. Si el Elo fuera POSTERIOR al
    partido, el salto que ENTRA a un partido estaría correlacionado con el
    resultado de ese mismo partido. Debe ser el salto que SALE el que lo esté.
    """
    from collections import defaultdict

    serie = defaultdict(list)
    for r in df.itertuples(index=False):
        gd = r.home_goals - r.away_goals
        serie[r.home].append((r.elo_h, gd))
        serie[r.away].append((r.elo_a, -gd))

    entra, sale = [], []
    for s in serie.values():
        for k in range(len(s)):
            if k > 0:
                entra.append((s[k][0] - s[k - 1][0], s[k][1]))
            if k < len(s) - 1:
                sale.append((s[k + 1][0] - s[k][0], s[k][1]))

    def corr(pares):
        a = np.array([p[0] for p in pares], dtype=float)
        b = np.array([p[1] for p in pares], dtype=float)
        return float(np.corrcoef(a, b)[0, 1])

    c_entra, c_sale = corr(entra), corr(sale)
    assert abs(c_entra) < 0.05, f"el Elo ya conoce el resultado del partido (corr={c_entra:+.3f}): FUGA"
    assert c_sale > 0.4, f"el Elo no reacciona al resultado (corr={c_sale:+.3f}): join sospechoso"


def test_tasas_base_realistas(df):
    tasas = df["result"].value_counts(normalize=True)
    assert 0.40 < tasas["H"] < 0.48   # ventaja de local
    assert 0.22 < tasas["D"] < 0.28
    assert tasas["H"] > tasas["A"]
