"""Las métricas se verifican contra valores conocidos analíticamente."""
from __future__ import annotations

import numpy as np
import pytest

from golazo.metrics import (
    accuracy,
    brier,
    calibration_table,
    expected_calibration_error,
    log_loss,
    rps,
    skill_score,
)


def test_prediccion_perfecta_da_cero():
    p = np.array([[1.0, 0.0, 0.0]])
    assert rps(p, ["H"]) == pytest.approx(0.0)
    assert brier(p, ["H"]) == pytest.approx(0.0)
    assert log_loss(p, ["H"]) == pytest.approx(0.0, abs=1e-9)


def test_uniforme_coincide_con_los_valores_teoricos():
    p = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert log_loss(p, ["H"]) == pytest.approx(np.log(3))
    assert brier(p, ["H"]) == pytest.approx((2 / 3) ** 2 + 2 * (1 / 3) ** 2)
    # RPS = 0.5 * [(1/3-1)^2 + (2/3-1)^2] = 0.5 * [4/9 + 1/9]
    assert rps(p, ["H"]) == pytest.approx(0.5 * (4 / 9 + 1 / 9))


def test_rps_respeta_el_orden_de_los_resultados():
    """Con victoria local real, fallar hacia el empate debe costar menos que
    fallar hacia la victoria visitante. Es lo que distingue al RPS del Brier."""
    empate = rps(np.array([[0.0, 1.0, 0.0]]), ["H"])
    visitante = rps(np.array([[0.0, 0.0, 1.0]]), ["H"])
    assert empate < visitante
    assert empate == pytest.approx(0.5)
    assert visitante == pytest.approx(1.0)

    # El Brier, por contra, es ciego al orden: castiga ambos igual.
    assert brier(np.array([[0.0, 1.0, 0.0]]), ["H"]) == pytest.approx(
        brier(np.array([[0.0, 0.0, 1.0]]), ["H"]))


def test_log_loss_no_explota_con_probabilidad_cero():
    assert np.isfinite(log_loss(np.array([[0.0, 0.5, 0.5]]), ["H"]))


def test_accuracy_usa_el_argmax():
    p = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    assert accuracy(p, ["H", "A"]) == pytest.approx(1.0)
    assert accuracy(p, ["A", "H"]) == pytest.approx(0.0)


def test_skill_score():
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)   # mitad de error que la referencia
    assert skill_score(1.0, 1.0) == pytest.approx(0.0)   # igual que la referencia
    assert skill_score(2.0, 1.0) == pytest.approx(-1.0)  # el doble de error


def test_calibracion_perfecta_da_ece_cero():
    """Un modelo que dice 'H' con probabilidad 0.7 y acierta el 70% del tiempo."""
    rng = np.random.default_rng(0)
    n = 20000
    p_h = np.full(n, 0.7)
    y = np.where(rng.random(n) < 0.7, "H", "A")
    p = np.column_stack([p_h, np.zeros(n), 1 - p_h])
    assert expected_calibration_error(p, y) < 0.02


def test_tabla_de_calibracion_detecta_exceso_de_confianza():
    """Modelo que promete 0.9 pero acierta 0.5: la curva cae bajo la diagonal."""
    n = 2000
    p = np.column_stack([np.full(n, 0.9), np.zeros(n), np.full(n, 0.1)])
    y = np.array(["H"] * (n // 2) + ["A"] * (n // 2))
    tab = calibration_table(p, y, n_bins=10)
    fila = tab[tab["pred_mean"] > 0.8].iloc[0]
    assert fila["obs_freq"] < fila["pred_mean"]
    assert expected_calibration_error(p, y) > 0.2
