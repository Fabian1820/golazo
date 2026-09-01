"""Pronóstico de la próxima jornada.

Cubre las tres reglas que hacen publicable el resultado:

1. no se predice a un equipo sin historial suficiente;
2. no se firma un partido que ya empezó;
3. no se firma dos veces el mismo partido con el mismo modelo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from golazo import forecast
from golazo.artifacts import ModelArtifact, build_metadata
from golazo.ledger import PredictionLedger
from golazo.models.base import Model
from golazo.service import InsufficientHistoryError, PredictionService

AHORA = pd.Timestamp("2026-09-01 12:00:00")


class ModeloFijo(Model):
    name = "fijo"

    def fit(self, train):
        return self

    def predict_proba(self, test):
        return np.tile([0.5, 0.3, 0.2], (len(test), 1))


def _historial(n_por_equipo=30):
    """Cuatro equipos con historial de sobra y uno con casi ninguno."""
    filas, fecha = [], pd.Timestamp("2025-01-01")
    equipos = ["A", "B", "C", "D"]
    for i in range(n_por_equipo * 2):
        h, a = equipos[i % 4], equipos[(i + 1) % 4]
        filas.append({"match_id": f"m{i}", "date": fecha, "season": 2025, "league": "EPL",
                      "home": h, "away": a, "home_goals": 2, "away_goals": 1,
                      "home_xg": 1.7, "away_xg": 1.0, "home_deep": 8, "away_deep": 5,
                      "home_ppda": 10.0, "away_ppda": 12.0,
                      "home_shots": np.nan, "away_shots": np.nan,
                      "home_sot": np.nan, "away_sot": np.nan})
        fecha += pd.Timedelta(days=3)
    # 'Cantera' aparece sólo dos veces: el caso de copa.
    for i in range(2):
        filas.append({"match_id": f"c{i}", "date": fecha, "season": 2025, "league": "EPL",
                      "home": "A", "away": "Cantera", "home_goals": 3, "away_goals": 0,
                      "home_xg": 2.5, "away_xg": 0.3, "home_deep": 12, "away_deep": 2,
                      "home_ppda": 8.0, "away_ppda": 20.0,
                      "home_shots": np.nan, "away_shots": np.nan,
                      "home_sot": np.nan, "away_sot": np.nan})
        fecha += pd.Timedelta(days=3)
    df = pd.DataFrame(filas)
    df["result"] = np.where(df.home_goals > df.away_goals, "H",
                            np.where(df.home_goals < df.away_goals, "A", "D"))
    df["elo_h"] = 1600.0
    df["elo_a"] = 1550.0
    return df


@pytest.fixture
def servicio():
    hist = _historial()
    meta = build_metadata("fijo", hist)
    return PredictionService(ModelArtifact(ModeloFijo(), meta), hist)


@pytest.fixture
def ledger(tmp_path):
    return PredictionLedger(tmp_path / "p.jsonl")


@pytest.fixture
def calendario(monkeypatch):
    fx = pd.DataFrame({
        "match_id": ["f1", "f2", "f3", "f4"],
        "date": [AHORA + pd.Timedelta(days=2),      # normal
                 AHORA + pd.Timedelta(days=3),      # normal
                 AHORA - pd.Timedelta(hours=2),     # ya empezó
                 AHORA + pd.Timedelta(days=4)],     # equipo sin historial
        "season": 2026, "league": "EPL",
        "home": ["A", "B", "C", "Cantera"],
        "away": ["B", "C", "D", "A"],
        "home_goals": np.nan, "away_goals": np.nan,
    })
    monkeypatch.setattr(forecast, "upcoming", lambda df=None, **kw: fx)
    monkeypatch.setattr(forecast, "load_store", lambda *a, **kw: fx)
    return fx


# --- guardia de historial --------------------------------------------------


def test_se_niega_a_predecir_sin_historial_suficiente(servicio):
    with pytest.raises(InsufficientHistoryError, match="Cantera"):
        servicio.predict("Cantera", "A")


def test_predice_con_historial_suficiente(servicio):
    p = servicio.predict("A", "B")
    assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_el_umbral_es_configurable():
    hist = _historial()
    meta = build_metadata("fijo", hist)
    permisivo = PredictionService(ModelArtifact(ModeloFijo(), meta), hist, min_history=1)
    assert permisivo.predict("Cantera", "A").probabilities


# --- pronóstico ------------------------------------------------------------


def test_solo_predice_lo_que_esta_por_jugarse(servicio, ledger, calendario):
    r = forecast.run(horizon_days=10, service=servicio, ledger=ledger, now=AHORA)
    nombres = {(p["home"], p["away"]) for p in r["predicciones"]}
    assert ("C", "D") not in nombres          # ya había empezado
    assert ("Cantera", "A") not in nombres    # sin historial
    assert ("A", "B") in nombres


def test_omite_con_motivo_explicito(servicio, ledger, calendario):
    r = forecast.run(horizon_days=10, service=servicio, ledger=ledger, now=AHORA)
    assert any("Cantera" in o for o in r["omitidos"])


def test_firma_las_predicciones(servicio, ledger, calendario):
    r = forecast.run(horizon_days=10, service=servicio, ledger=ledger, now=AHORA)
    assert r["registrados"] == 2
    assert len(ledger) == 2
    assert ledger.verify()[0]
    assert all(p.get("ledger_hash") for p in r["predicciones"])


def test_dry_run_no_escribe(servicio, ledger, calendario):
    forecast.run(horizon_days=10, record=False, service=servicio, ledger=ledger, now=AHORA)
    assert len(ledger) == 0


def test_no_firma_dos_veces_el_mismo_partido(servicio, ledger, calendario):
    """Ejecutar el pipeline a diario no debe acumular predicciones repetidas."""
    forecast.run(horizon_days=10, service=servicio, ledger=ledger, now=AHORA)
    segunda = forecast.run(horizon_days=10, service=servicio, ledger=ledger, now=AHORA)
    assert segunda["registrados"] == 0
    assert segunda["repetidos"] == 2
    assert len(ledger) == 2


def test_el_horizonte_recorta(servicio, ledger, calendario):
    r = forecast.run(horizon_days=1, service=servicio, ledger=ledger, now=AHORA)
    assert r["fixtures"] == 0


def test_sin_calendario_no_falla(servicio, ledger, monkeypatch):
    vacio = pd.DataFrame(columns=["match_id", "date", "league", "home", "away",
                                  "home_goals", "away_goals"])
    monkeypatch.setattr(forecast, "upcoming", lambda df=None, **kw: vacio)
    monkeypatch.setattr(forecast, "load_store", lambda *a, **kw: vacio)
    r = forecast.run(service=servicio, ledger=ledger, now=AHORA)
    assert r["fixtures"] == 0
    assert "fetch" in forecast.describe(r)
