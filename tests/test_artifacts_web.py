"""Persistencia versionada y contrato de la API HTTP."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from golazo.artifacts import Metadata, ModelArtifact, build_metadata
from golazo.models.base import Model


class ModeloTonto(Model):
    """Modelo determinista, para no pagar entrenamientos en los tests."""

    name = "tonto"

    def __init__(self, sesgo=0.5):
        self.sesgo = sesgo

    def fit(self, train):
        return self

    def predict_proba(self, test):
        resto = (1.0 - self.sesgo) / 2
        return np.tile([self.sesgo, resto, resto], (len(test), 1))


@pytest.fixture
def train_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-03-01"]),
        "league": ["EPL", "EPL", "Serie A"],
        "home": ["A", "B", "C"],
        "away": ["B", "A", "D"],
    })


# --- metadatos -------------------------------------------------------------


def test_build_metadata_recoge_el_rango(train_df):
    m = build_metadata("tonto", train_df, feature_columns=["x", "y"])
    assert m.train_start == "2020-01-01"
    assert m.train_end == "2021-03-01"
    assert m.n_matches == 3
    assert m.leagues == ["EPL", "Serie A"]


def test_version_id_es_estable(train_df):
    """Mismos datos y mismo código -> mismo id, aunque cambie el reloj."""
    a = build_metadata("tonto", train_df, feature_columns=["x"])
    b = build_metadata("tonto", train_df, feature_columns=["x"])
    assert a.trained_at is not None
    assert a.version_id == b.version_id


def test_version_id_cambia_con_los_datos(train_df):
    a = build_metadata("tonto", train_df)
    b = build_metadata("tonto", train_df.iloc[:2])
    assert a.version_id != b.version_id


def test_version_id_cambia_con_el_modelo(train_df):
    assert build_metadata("tonto", train_df).version_id != build_metadata("otro", train_df).version_id


def test_version_id_empieza_por_la_fecha(train_df):
    assert build_metadata("tonto", train_df).version_id.startswith("20210301-")


# --- guardar y cargar ------------------------------------------------------


def test_ida_y_vuelta(tmp_path, train_df):
    meta = build_metadata("tonto", train_df, backtest={"rps": 0.2049})
    ruta = ModelArtifact(ModeloTonto(0.7), meta).save(tmp_path)
    assert (ruta / "model.joblib").exists()
    assert (ruta / "metadata.json").exists()

    cargado = ModelArtifact.load("latest", tmp_path)
    assert cargado.metadata.version_id == meta.version_id
    assert cargado.metadata.backtest["rps"] == 0.2049
    assert cargado.model.sesgo == 0.7


def test_latest_apunta_al_ultimo_guardado(tmp_path, train_df):
    m1 = build_metadata("tonto", train_df)
    ModelArtifact(ModeloTonto(0.6), m1).save(tmp_path)
    m2 = build_metadata("otro", train_df)
    ModelArtifact(ModeloTonto(0.8), m2).save(tmp_path)

    assert ModelArtifact.load("latest", tmp_path).metadata.version_id == m2.version_id
    # La versión anterior sigue accesible por su id.
    assert ModelArtifact.load(m1.version_id, tmp_path).model.sesgo == 0.6


def test_make_latest_false_no_mueve_el_puntero(tmp_path, train_df):
    m1 = build_metadata("tonto", train_df)
    ModelArtifact(ModeloTonto(0.6), m1).save(tmp_path)
    ModelArtifact(ModeloTonto(0.9), build_metadata("otro", train_df)).save(tmp_path, make_latest=False)
    assert ModelArtifact.load("latest", tmp_path).metadata.version_id == m1.version_id


def test_listar_versiones(tmp_path, train_df):
    ModelArtifact(ModeloTonto(), build_metadata("tonto", train_df)).save(tmp_path)
    ModelArtifact(ModeloTonto(), build_metadata("otro", train_df)).save(tmp_path)
    assert len(ModelArtifact.list_versions(tmp_path)) == 2


def test_errores_claros(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/train.py"):
        ModelArtifact.load("latest", tmp_path)
    with pytest.raises(FileNotFoundError, match="no-existe"):
        ModelArtifact.load("no-existe", tmp_path)


def test_metadata_json_es_legible(tmp_path, train_df):
    meta = build_metadata("tonto", train_df, backtest={"rps": 0.2})
    ruta = ModelArtifact(ModeloTonto(), meta).save(tmp_path)
    d = json.loads((ruta / "metadata.json").read_text(encoding="utf-8"))
    assert d["model_name"] == "tonto"
    assert d["backtest"]["rps"] == 0.2
    assert Metadata(**d).version_id == meta.version_id


# --- API HTTP --------------------------------------------------------------


@pytest.fixture
def cliente(tmp_path):
    from golazo.ledger import PredictionLedger
    from golazo.service import PredictionService
    from golazo.web import create_app

    historia = pd.DataFrame({
        "match_id": [1, 2, 3, 4],
        "date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-22"]),
        "season": 2019, "league": "EPL",
        "home": ["A", "B", "A", "C"], "away": ["B", "A", "C", "A"],
        "elo_h": [1700.0, 1650.0, 1710.0, 1600.0],
        "elo_a": [1650.0, 1700.0, 1600.0, 1710.0],
        "home_goals": [2, 1, 3, 0], "away_goals": [1, 1, 0, 2],
        "home_xg": [1.8, 1.1, 2.4, 0.6], "away_xg": [0.9, 1.2, 0.5, 1.9],
        "home_shots": [14, 10, 17, 7], "away_shots": [8, 11, 6, 15],
        "home_sot": [6, 3, 8, 2], "away_sot": [3, 4, 2, 6],
        "home_deep": [8, 5, 11, 3], "away_deep": [4, 6, 3, 9],
        "home_ppda": [9.0, 12.0, 8.0, 15.0], "away_ppda": [13.0, 10.0, 16.0, 8.0],
        "result": ["H", "D", "H", "A"],
    })
    meta = build_metadata("tonto", historia, backtest={"rps": 0.2049, "n_matches": 7203})
    # min_history=1: esta historia de juguete tiene 4 partidos y el umbral real
    # (10) la rechazaría. Aquí se prueba el contrato HTTP, no la guardia de
    # historial, que tiene sus propios tests en test_forecast.py.
    servicio = PredictionService(ModelArtifact(ModeloTonto(0.6), meta), historia, min_history=1)
    app = create_app(service=servicio, ledger=PredictionLedger(tmp_path / "p.jsonl"))
    app.config["TESTING"] = True
    return app.test_client()


def test_health(cliente):
    r = cliente.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_model_info_expone_la_procedencia(cliente):
    d = cliente.get("/api/model").get_json()
    assert d["name"] == "tonto"
    assert d["backtest"]["rps"] == 0.2049
    assert d["train_end"] == "2020-01-22"


def test_catalogo(cliente):
    assert cliente.get("/api/leagues").get_json() == ["EPL"]
    assert set(cliente.get("/api/teams/EPL").get_json()) == {"A", "B", "C"}
    assert cliente.get("/api/teams/Inexistente").status_code == 400


def test_predict_devuelve_probabilidades_y_procedencia(cliente):
    r = cliente.post("/api/predict", json={"home": "A", "away": "B"})
    assert r.status_code == 200
    d = r.get_json()
    assert sum(d["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert d["most_likely"] == "H"
    assert "model_version" in d and "trained_through" in d


def test_predict_valida_la_entrada(cliente):
    assert cliente.post("/api/predict", json={"home": "A"}).status_code == 400
    assert cliente.post("/api/predict", json={}).status_code == 400
    assert cliente.post("/api/predict", json={"home": "A", "away": "A"}).status_code == 400


def test_predict_equipo_desconocido_da_404(cliente):
    r = cliente.post("/api/predict", json={"home": "Inventado", "away": "B"})
    assert r.status_code == 404
    assert r.get_json()["error"] == "equipo_desconocido"


def test_las_predicciones_futuras_se_registran(cliente):
    d = cliente.post("/api/predict", json={"home": "A", "away": "B", "kickoff": "2030-01-01T15:00:00"}).get_json()
    assert "ledger_hash" in d
    track = cliente.get("/api/track-record").get_json()
    assert track["verified"] and track["n_predictions"] == 1


def test_las_predicciones_pasadas_no_se_registran(cliente):
    """El registro se niega, pero la predicción se sirve igual."""
    d = cliente.post("/api/predict", json={"home": "A", "away": "B", "kickoff": "2021-01-01T15:00:00"}).get_json()
    assert "ledger_hash" not in d
    assert "ledger_error" in d
    assert d["probabilities"]
    assert cliente.get("/api/track-record").get_json()["n_predictions"] == 0
