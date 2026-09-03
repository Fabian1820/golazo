"""Refresco, entrenamiento y CLI.

Eran los tres módulos sin tests propios. Ninguno de estos tests sale a la red:
las fuentes se sustituyen por dobles, porque lo que se prueba aquí es la
orquestación, no el transporte (eso está en test_sources_store.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from golazo import refresh as refresh_mod
from golazo.cli import build_parser
from golazo.refresh import current_season, describe
from golazo.sources.base import normalize

# --- temporada en curso ----------------------------------------------------


@pytest.mark.parametrize("fecha,esperado", [
    ("2026-08-15", 2026),   # agosto: temporada nueva
    ("2026-07-01", 2026),   # julio: ya cuenta como nueva
    ("2026-06-30", 2025),   # junio: todavía la anterior
    ("2026-01-15", 2025),   # enero: la que empezó en agosto
    ("2026-12-31", 2026),
])
def test_current_season(fecha, esperado):
    assert current_season(pd.Timestamp(fecha)) == esperado


# --- refresco --------------------------------------------------------------


def _partidos(n=6, prefijo="understat", inicio="2026-01-01"):
    fechas = pd.date_range(inicio, periods=n, freq="7D")
    return normalize(pd.DataFrame({
        "match_id": [f"{prefijo}:{i}" for i in range(n)],
        "date": fechas, "season": 2025, "league": "EPL",
        "home": [f"H{i}" for i in range(n)], "away": [f"A{i}" for i in range(n)],
        "home_goals": 2, "away_goals": 1,
    }))


class FuenteFalsa:
    name = "falsa"

    def __init__(self, jugados=None, calendario=None):
        self._jugados = jugados if jugados is not None else pd.DataFrame()
        self._calendario = calendario if calendario is not None else pd.DataFrame()
        self.llamadas = []

    def fetch(self, leagues=(), seasons=()):
        self.llamadas.append(("fetch", list(leagues), list(seasons)))
        return self._jugados

    def fixtures(self, leagues=(), seasons=()):
        self.llamadas.append(("fixtures", list(leagues), list(seasons)))
        return self._calendario


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "matches.csv"


def _parchear(monkeypatch, understat, footballdata=None):
    monkeypatch.setattr(refresh_mod, "UnderstatSource", lambda **kw: understat)
    fd = footballdata if footballdata is not None else FuenteFalsa()
    monkeypatch.setattr(refresh_mod, "FootballDataSource", lambda **kw: fd)
    return fd


def test_refresh_funde_en_el_almacen(monkeypatch, store_path):
    _parchear(monkeypatch, FuenteFalsa(jugados=_partidos(5)))
    r = refresh_mod.refresh(seasons=[2025], store_path=store_path, include_second_tier=False)
    assert r["nuevos"] == 5
    assert r["despues"] == 5


def test_refresh_es_idempotente(monkeypatch, store_path):
    _parchear(monkeypatch, FuenteFalsa(jugados=_partidos(5)))
    refresh_mod.refresh(seasons=[2025], store_path=store_path, include_second_tier=False)
    segundo = refresh_mod.refresh(seasons=[2025], store_path=store_path, include_second_tier=False)
    assert segundo["nuevos"] == 0
    assert segundo["despues"] == 5


def test_refresh_incluye_el_calendario(monkeypatch, store_path):
    calendario = _partidos(3, prefijo="fx", inicio="2030-01-01").assign(
        home_goals=np.nan, away_goals=np.nan)
    _parchear(monkeypatch, FuenteFalsa(jugados=_partidos(4), calendario=calendario))
    r = refresh_mod.refresh(seasons=[2025], store_path=store_path, include_second_tier=False)
    assert r["descargados_calendario"] == 3
    assert r["despues"] == 7


def test_refresh_puede_omitir_el_calendario(monkeypatch, store_path):
    fuente = FuenteFalsa(jugados=_partidos(4), calendario=_partidos(3, prefijo="fx"))
    _parchear(monkeypatch, fuente)
    refresh_mod.refresh(seasons=[2025], store_path=store_path,
                        include_fixtures=False, include_second_tier=False)
    assert not any(c[0] == "fixtures" for c in fuente.llamadas)


def test_refresh_añade_segunda_division(monkeypatch, store_path):
    segunda = _partidos(4, prefijo="fd", inicio="2026-02-01")
    _parchear(monkeypatch, FuenteFalsa(jugados=_partidos(3)), FuenteFalsa(jugados=segunda))
    r = refresh_mod.refresh(seasons=[2025], store_path=store_path, include_second_tier=True)
    assert r["descargados_segunda"] == 4
    assert r["despues"] == 7


def test_refresh_sin_datos_no_rompe(monkeypatch, store_path):
    _parchear(monkeypatch, FuenteFalsa())
    r = refresh_mod.refresh(seasons=[2025], store_path=store_path, include_second_tier=False)
    assert r["despues"] == 0


def test_refresh_usa_la_temporada_en_curso_por_defecto(monkeypatch, store_path):
    fuente = FuenteFalsa(jugados=_partidos(2))
    _parchear(monkeypatch, fuente)
    refresh_mod.refresh(store_path=store_path, include_second_tier=False)
    _, _, temporadas = fuente.llamadas[0]
    assert temporadas == [current_season()]


def test_describe_es_legible():
    texto = describe({
        "fuente": "understat", "temporadas": [2025], "ligas": ["EPL"],
        "descargados_jugados": 10, "descargados_calendario": 3, "descargados_segunda": 5,
        "antes": 0, "despues": 18, "nuevos": 18, "actualizados": 0, "hasta": "2026-05-01",
    })
    assert "understat" in texto and "18" in texto and "2026-05-01" in texto


# --- CLI -------------------------------------------------------------------


def test_el_parser_declara_todos_los_comandos():
    parser = build_parser()
    accion = next(a for a in parser._actions if getattr(a, "choices", None) and "train" in a.choices)
    esperados = {"train", "predict", "fetch", "odds", "forecast", "models",
                 "validate", "monitor", "score", "verify", "serve"}
    assert esperados <= set(accion.choices)


@pytest.mark.parametrize("argv,attr,valor", [
    (["forecast", "--horizon", "5"], "horizon", 5),
    (["forecast", "--dry-run"], "dry_run", True),
    (["fetch", "--seasons", "2024", "2025"], "seasons", [2024, 2025]),
    (["train", "--model", "elo_logistico"], "model", "elo_logistico"),
    (["odds", "--cache"], "cache", True),
])
def test_el_parser_lee_los_argumentos(argv, attr, valor):
    args = build_parser().parse_args(argv)
    assert getattr(args, attr) == valor


def test_cada_comando_tiene_funcion():
    parser = build_parser()
    accion = next(a for a in parser._actions if getattr(a, "choices", None) and "train" in a.choices)
    for nombre in accion.choices:
        args = build_parser().parse_args([nombre] if nombre != "predict" else ["predict", "A", "B"])
        assert callable(getattr(args, "func", None)), f"'{nombre}' no tiene función asociada"


def test_modelos_de_entrenamiento_disponibles():
    from golazo.training import MODELOS

    assert {"dixon_coles", "elo_logistico", "gradient_boosting"} <= set(MODELOS)
