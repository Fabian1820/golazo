"""Los mercados derivados deben ser coherentes entre sí siempre.

Publicar un 1X2 que no cuadre con el over/under es la clase de incoherencia que
destruye la confianza en un producto de pronósticos. Aquí es imposible porque
todo sale de la misma matriz, y estos tests lo verifican.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from golazo import markets


def matriz(lam=1.5, mu=1.2, n=11):
    m = np.outer(poisson.pmf(np.arange(n), lam), poisson.pmf(np.arange(n), mu))
    return m / m.sum()


@pytest.fixture
def mat():
    return matriz()


def test_el_1x2_suma_uno(mat):
    o = markets.outcome_1x2(mat)
    assert sum(o.values()) == pytest.approx(1.0, abs=1e-9)


def test_la_ventaja_se_refleja_en_el_1x2():
    fuerte = markets.outcome_1x2(matriz(lam=2.5, mu=0.6))
    debil = markets.outcome_1x2(matriz(lam=0.6, mu=2.5))
    assert fuerte["H"] > fuerte["A"]
    assert debil["A"] > debil["H"]
    assert fuerte["H"] == pytest.approx(debil["A"], abs=1e-9)  # simetría


def test_distribucion_de_goles_totales(mat):
    dist = markets.total_goals_distribution(mat)
    assert dist.sum() == pytest.approx(1.0, abs=1e-9)
    assert (dist >= 0).all()
    # La media de la distribución total es la suma de las medias marginales.
    eg = markets.expected_goals(mat)
    media = float((dist * np.arange(len(dist))).sum())
    assert media == pytest.approx(eg["total"], abs=1e-6)


def test_over_under_complementarios(mat):
    for linea, v in markets.over_under(mat).items():
        assert v["over"] + v["under"] == pytest.approx(1.0, abs=1e-9), linea


def test_over_under_es_monotono(mat):
    """Superar una línea más alta siempre debe ser menos probable."""
    ou = markets.over_under(mat)
    valores = [ou[k]["over"] for k in sorted(ou, key=float)]
    assert all(a >= b for a, b in zip(valores, valores[1:]))


def test_over_05_coincide_con_no_haber_empate_a_cero(mat):
    assert markets.over_under(mat)["0.5"]["under"] == pytest.approx(float(mat[0, 0]), abs=1e-9)


def test_ambos_marcan(mat):
    b = markets.both_teams_score(mat)
    assert b["yes"] + b["no"] == pytest.approx(1.0, abs=1e-9)
    # 'no' es la probabilidad de que al menos uno se quede a cero.
    sin_goles = float(mat[0, :].sum() + mat[:, 0].sum() - mat[0, 0])
    assert b["no"] == pytest.approx(sin_goles, abs=1e-9)


def test_marcadores_ordenados_y_validos(mat):
    top = markets.top_scorelines(mat, n=6)
    assert len(top) == 6
    probs = [s["probability"] for s in top]
    assert probs == sorted(probs, reverse=True)
    assert all("-" in s["score"] for s in top)


def test_handicap_asiatico(mat):
    h = markets.asian_handicap(mat, line=-0.5)
    assert sum(h.values()) == pytest.approx(1.0, abs=1e-9)
    # Con línea de medio gol no puede haber empate técnico.
    assert h["push"] == pytest.approx(0.0, abs=1e-12)
    # Con hándicap -0.5 el local sólo gana si gana el partido.
    assert h["home"] == pytest.approx(markets.outcome_1x2(mat)["H"], abs=1e-9)


def test_handicap_entero_produce_push(mat):
    h = markets.asian_handicap(mat, line=0.0)
    assert h["push"] == pytest.approx(markets.outcome_1x2(mat)["D"], abs=1e-9)


def test_goles_esperados_coinciden_con_los_parametros():
    """Con Poisson puro y suficiente cola, la media recuperada es lambda."""
    eg = markets.expected_goals(matriz(lam=1.7, mu=1.1, n=25))
    assert eg["home"] == pytest.approx(1.7, abs=1e-4)
    assert eg["away"] == pytest.approx(1.1, abs=1e-4)


def test_verificacion_de_coherencia(mat):
    ok, motivo = markets.coherence_check(mat)
    assert ok, motivo

    rota = mat.copy()
    rota[0, 0] += 0.5
    assert not markets.coherence_check(rota)[0]

    negativa = mat.copy()
    negativa[2, 2] = -0.1
    assert not markets.coherence_check(negativa)[0]


def test_summarize_incluye_todos_los_mercados(mat):
    s = markets.summarize(mat)
    assert set(s) == {"outcome", "expected_goals", "over_under", "both_teams_score", "top_scorelines"}
