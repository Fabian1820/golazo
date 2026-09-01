"""El backtest debe ser incapaz de mirar al futuro."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from golazo.backtest import evaluate, walk_forward
from golazo.config import OUTCOMES
from golazo.models.base import Model


class _Espia(Model):
    """Modelo que registra los rangos de fechas que ve en cada reajuste."""

    name = "espia"
    llamadas: list = []

    def fit(self, train):
        self._train_max = train["date"].max()
        return self

    def predict_proba(self, test):
        _Espia.llamadas.append((self._train_max, test["date"].min()))
        return np.full((len(test), 3), 1 / 3)


@pytest.fixture
def datos():
    rng = np.random.default_rng(0)
    n = 1500
    fechas = pd.date_range("2018-01-01", periods=n, freq="12h")
    return pd.DataFrame({
        "match_id": np.arange(n),
        "date": fechas,
        "league": "L",
        "home": [f"H{i % 10}" for i in range(n)],
        "away": [f"A{i % 7}" for i in range(n)],
        "result": rng.choice(list(OUTCOMES), n),
        "home_goals": rng.poisson(1.5, n),
        "away_goals": rng.poisson(1.2, n),
        "elo_diff": rng.normal(0, 100, n),
    })


def test_el_entrenamiento_siempre_precede_a_la_prediccion(datos):
    _Espia.llamadas = []
    walk_forward(datos, {"espia": _Espia}, start="2018-06-01", refit_days=14, verbose=False)
    assert _Espia.llamadas
    for train_max, test_min in _Espia.llamadas:
        assert train_max < test_min, f"entrenó hasta {train_max} para predecir {test_min}: FUGA TEMPORAL"


def test_cada_partido_se_predice_una_sola_vez(datos):
    preds = walk_forward(datos, {"espia": _Espia}, start="2018-06-01", refit_days=14, verbose=False)
    assert preds["match_id"].is_unique


def test_solo_se_predicen_partidos_desde_la_fecha_de_inicio(datos):
    inicio = pd.Timestamp("2018-06-01")
    preds = walk_forward(datos, {"espia": _Espia}, start="2018-06-01", refit_days=14, verbose=False)
    assert preds["date"].min() >= inicio


def test_las_probabilidades_salen_normalizadas(datos):
    class Torcido(Model):
        def fit(self, train):
            return self

        def predict_proba(self, test):
            return np.full((len(test), 3), 5.0)  # sin normalizar a propósito

    preds = walk_forward(datos, {"m": Torcido}, start="2018-06-01", refit_days=30, verbose=False)
    suma = preds[[f"m_{o}" for o in OUTCOMES]].to_numpy().sum(axis=1)
    np.testing.assert_allclose(suma, 1.0, atol=1e-9)


def test_evaluate_calcula_skill_cero_contra_si_mismo(datos):
    preds = walk_forward(datos, {"espia": _Espia}, start="2018-06-01", refit_days=30, verbose=False)
    tabla = evaluate(preds, ["espia"], reference="espia")
    assert tabla.loc["espia", "rps_skill_%"] == pytest.approx(0.0)


def test_falla_si_no_hay_partidos_en_el_horizonte(datos):
    with pytest.raises(ValueError):
        walk_forward(datos, {"espia": _Espia}, start="2030-01-01", refit_days=14, verbose=False)
