"""El Elo propio debe ser correcto y, sobre todo, estrictamente pre-partido."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from golazo.elo import BASE_RATING, EloRatings, attach_elo, expected_score, goal_difference_multiplier


def test_expectativa_simetrica():
    assert expected_score(1500, 1500, home_advantage=0) == pytest.approx(0.5)
    assert expected_score(1900, 1500, home_advantage=0) > 0.9
    assert expected_score(1500, 1900, home_advantage=0) < 0.1


def test_la_ventaja_de_campo_favorece_al_local():
    assert expected_score(1500, 1500, home_advantage=65) > 0.5


def test_multiplicador_por_diferencia_de_goles():
    assert goal_difference_multiplier(0) == 1.0
    assert goal_difference_multiplier(1) == 1.0
    assert goal_difference_multiplier(2) == 1.5
    assert goal_difference_multiplier(-2) == 1.5      # simétrico
    assert goal_difference_multiplier(5) > goal_difference_multiplier(3)


def test_el_elo_es_suma_cero():
    """Lo que gana un equipo lo pierde el otro."""
    elo = EloRatings()
    elo.update("A", "B", 3, 0)
    assert elo.rating("A") + elo.rating("B") == pytest.approx(2 * BASE_RATING)


def test_ganar_sube_y_perder_baja():
    elo = EloRatings()
    elo.update("A", "B", 2, 0)
    assert elo.rating("A") > BASE_RATING
    assert elo.rating("B") < BASE_RATING


def test_una_goleada_mueve_mas_que_un_ajustado():
    ajustado, goleada = EloRatings(), EloRatings()
    ajustado.update("A", "B", 1, 0)
    goleada.update("A", "B", 5, 0)
    assert goleada.rating("A") > ajustado.rating("A")


def test_ganar_al_favorito_da_mas_puntos():
    """Batir a un equipo fuerte debe valer más que batir a uno débil."""
    base = EloRatings()
    for _ in range(15):
        base.update("Fuerte", "Saco", 3, 0)

    contra_fuerte = EloRatings()
    contra_fuerte._ratings = dict(base._ratings)
    antes = contra_fuerte.rating("Saco")
    contra_fuerte.update("Saco", "Fuerte", 1, 0)
    ganancia_contra_fuerte = contra_fuerte.rating("Saco") - antes

    contra_igual = EloRatings()
    antes2 = contra_igual.rating("X")
    contra_igual.update("X", "Y", 1, 0)
    ganancia_contra_igual = contra_igual.rating("X") - antes2

    assert ganancia_contra_fuerte > ganancia_contra_igual


def test_equipo_nuevo_entra_por_debajo_de_la_media_de_su_liga():
    elo = EloRatings()
    for _ in range(10):
        elo.update("A", "B", 2, 1, league="L1")
    media = (elo.rating("A") + elo.rating("B")) / 2
    assert elo.rating("Ascendido", league="L1") < media
    assert not elo.known("Ascendido")


def test_la_primera_liga_arranca_en_la_linea_base():
    """Sin equipos previos no hay respecto de quién ascender."""
    assert EloRatings().rating("Cualquiera", league="L1") == BASE_RATING


# --- la propiedad que de verdad importa ------------------------------------


@pytest.fixture
def historial():
    rng = np.random.default_rng(11)
    n = 600
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="2D"),
        "league": "L",
        "home": [f"T{i % 12}" for i in range(n)],
        "away": [f"T{(i + 5) % 12}" for i in range(n)],
        "home_goals": rng.poisson(1.5, n),
        "away_goals": rng.poisson(1.2, n),
    })


def test_el_elo_asignado_es_previo_al_partido(historial):
    """Prueba decisiva, la misma que se aplicaba a ClubElo.

    Si el Elo fuese posterior, el salto que ENTRA a un partido estaría
    correlacionado con el resultado de ese mismo partido.
    """
    from collections import defaultdict

    df = attach_elo(historial)
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
        a = np.array([p[0] for p in pares])
        b = np.array([p[1] for p in pares])
        return float(np.corrcoef(a, b)[0, 1])

    assert abs(corr(entra)) < 0.05, "el Elo ya conoce el resultado del partido: FUGA"
    assert corr(sale) > 0.4, "el Elo no reacciona al resultado"


def test_alterar_un_resultado_no_cambia_su_propio_elo(historial):
    a = attach_elo(historial)
    modificado = historial.copy()
    modificado.loc[300, "home_goals"] = 9
    b = attach_elo(modificado)

    assert a.loc[300, "elo_h"] == pytest.approx(b.loc[300, "elo_h"])
    assert a.loc[300, "elo_a"] == pytest.approx(b.loc[300, "elo_a"])
    # Pero sí debe afectar a los siguientes.
    assert not np.allclose(a.loc[301:, "elo_h"], b.loc[301:, "elo_h"])


def test_attach_elo_no_pierde_filas(historial):
    out = attach_elo(historial)
    assert len(out) == len(historial)
    assert out["elo_h"].notna().all()
    assert out["elo_a"].notna().all()
