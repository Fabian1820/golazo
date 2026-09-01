"""Esquema canónico, almacén y transformación de la fuente viva.

Las pruebas de red van marcadas y se saltan por defecto: la suite no puede
depender de que un sitio de terceros esté levantado.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from golazo.sources.base import CANONICAL_COLUMNS, OPTIONAL_COLUMNS, merge_sources, normalize
from golazo.sources.understat import UnderstatSource
from golazo.store import load_store, played, upcoming, upsert


def fila(match_id="m1", date="2024-01-01 15:00:00", home="A", away="B",
         home_goals=2, away_goals=1, **extra):
    d = {"match_id": match_id, "date": date, "season": 2023, "league": "EPL",
         "home": home, "away": away, "home_goals": home_goals, "away_goals": away_goals}
    d.update(extra)
    return d


# --- esquema canónico ------------------------------------------------------


def test_normalize_rellena_las_opcionales():
    out = normalize(pd.DataFrame([fila()]))
    assert list(out.columns) == CANONICAL_COLUMNS
    for c in OPTIONAL_COLUMNS:
        assert out[c].isna().all()


def test_normalize_exige_las_obligatorias():
    with pytest.raises(ValueError, match="obligatorias"):
        normalize(pd.DataFrame([{"match_id": "x", "date": "2024-01-01"}]))


def test_normalize_permite_calendario_sin_goles():
    d = fila(home_goals=None, away_goals=None)
    del d["home_goals"], d["away_goals"]
    out = normalize(pd.DataFrame([d]), require_results=False)
    assert out["home_goals"].isna().all()


def test_normalize_ordena_por_fecha():
    out = normalize(pd.DataFrame([
        fila("b", "2024-03-01 15:00:00"), fila("a", "2024-01-01 15:00:00")]))
    assert out["match_id"].tolist() == ["a", "b"]


def test_normalize_convierte_tipos():
    out = normalize(pd.DataFrame([fila(match_id=99, home_goals="3")]))
    assert out["match_id"].iloc[0] == "99"
    assert out["home_goals"].iloc[0] == 3.0
    assert isinstance(out["date"].iloc[0], pd.Timestamp)


def test_merge_gana_la_ultima_aparicion():
    """Un partido anunciado que ya se jugó debe sustituir a su versión previa."""
    anunciado = normalize(pd.DataFrame([fila(home_goals=None, away_goals=None)]),
                          require_results=False)
    jugado = normalize(pd.DataFrame([fila(home_goals=3, away_goals=0)]))
    out = merge_sources(anunciado, jugado)
    assert len(out) == 1
    assert out["home_goals"].iloc[0] == 3.0


def test_merge_sin_entradas():
    assert merge_sources(None, pd.DataFrame()).empty


# --- almacén ---------------------------------------------------------------


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "matches.csv"


def test_almacen_vacio(store_path):
    assert load_store(store_path).empty


def test_upsert_inserta_y_actualiza(store_path):
    r1 = upsert(normalize(pd.DataFrame([fila("a"), fila("b", home="C", away="D")])), store_path)
    assert r1["nuevos"] == 2 and r1["despues"] == 2

    r2 = upsert(normalize(pd.DataFrame([fila("a", home_goals=5), fila("c", home="E", away="F")])),
                store_path)
    assert r2["nuevos"] == 1
    assert r2["actualizados"] == 1
    assert r2["despues"] == 3
    assert load_store(store_path).set_index("match_id").loc["a", "home_goals"] == 5.0


def test_upsert_es_idempotente(store_path):
    datos = normalize(pd.DataFrame([fila("a"), fila("b", home="C", away="D")]))
    upsert(datos, store_path)
    segundo = upsert(datos, store_path)
    assert segundo["despues"] == 2
    assert segundo["nuevos"] == 0


def test_upsert_vacio_no_rompe(store_path):
    upsert(normalize(pd.DataFrame([fila("a")])), store_path)
    r = upsert(pd.DataFrame(), store_path)
    assert r["despues"] == 1


def test_separacion_jugados_y_anunciados(store_path):
    anunciado = fila("f1", "2030-01-01 15:00:00")
    del anunciado["home_goals"], anunciado["away_goals"]
    upsert(merge_sources(normalize(pd.DataFrame([fila("a")])),
                         normalize(pd.DataFrame([anunciado]), require_results=False)), store_path)
    df = load_store(store_path)
    assert len(played(df)) == 1
    assert len(upcoming(df)) == 1


# --- transformación de Understat -------------------------------------------


RESPUESTA = {
    "teams": {
        "1": {"id": "1", "title": "Arsenal", "history": [
            {"date": "2024-08-17 14:00:00", "xG": 1.6, "deep": 14,
             "ppda": {"att": 220, "def": 20}},
        ]},
        "2": {"id": "2", "title": "Wolves", "history": [
            {"date": "2024-08-17 14:00:00", "xG": 0.5, "deep": 3,
             "ppda": {"att": 180, "def": 15}},
        ]},
    },
    "players": [],
    "dates": [
        {"id": "1", "isResult": True,
         "h": {"id": "1", "title": "Arsenal"}, "a": {"id": "2", "title": "Wolves"},
         "goals": {"h": "2", "a": "0"}, "xG": {"h": "1.6283", "a": "0.5"},
         "datetime": "2024-08-17 14:00:00"},
        {"id": "2", "isResult": False,
         "h": {"id": "2", "title": "Wolves"}, "a": {"id": "1", "title": "Arsenal"},
         "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None},
         "datetime": "2030-05-01 14:00:00"},
    ],
}


def test_transforma_partidos_jugados():
    df = UnderstatSource()._to_frame(RESPUESTA, "EPL", 2024, played=True)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["match_id"] == "understat:1"
    assert r["home"] == "Arsenal" and r["away"] == "Wolves"
    assert r["home_goals"] == 2.0
    assert r["home_xg"] == pytest.approx(1.6283)
    assert r["home_deep"] == 14
    assert r["home_ppda"] == pytest.approx(220 / 20)   # ppda = att / def
    # Understat no publica tiros: deben viajar como NaN, no inventados.
    assert pd.isna(r["home_shots"]) and pd.isna(r["home_sot"])


def test_transforma_calendario():
    df = UnderstatSource()._to_frame(RESPUESTA, "EPL", 2024, played=False)
    assert len(df) == 1
    assert df.iloc[0]["match_id"] == "understat:2"
    assert pd.isna(df.iloc[0]["home_goals"])


def test_los_ids_no_chocan_entre_fuentes():
    """El prefijo evita que un id de Understat pise uno de Kaggle."""
    df = UnderstatSource()._to_frame(RESPUESTA, "EPL", 2024, played=True)
    assert df["match_id"].iloc[0].startswith("understat:")


def test_usa_la_cache_si_existe(tmp_path):
    cache = tmp_path / "EPL_2024.json"
    cache.write_text(json.dumps(RESPUESTA), encoding="utf-8")
    src = UnderstatSource(cache_dir=tmp_path)
    # Si intentase salir a la red, este test fallaría o tardaría.
    assert src._get("EPL", 2024)["dates"][0]["id"] == "1"


@pytest.mark.skipif(not os.environ.get("GOLAZO_NETWORK_TESTS"),
                    reason="requiere red; exportar GOLAZO_NETWORK_TESTS=1")
def test_understat_en_vivo():
    df = UnderstatSource().fetch(leagues=["EPL"], seasons=[2024])
    assert len(df) > 300
    assert df["home_goals"].notna().all()
    assert df["home_xg"].notna().mean() > 0.95
