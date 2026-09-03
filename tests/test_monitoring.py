"""Detección de degradación silenciosa.

Cada test degrada los datos de una forma concreta y exige que el hallazgo
aparezca. Un monitor que nunca salta no vigila nada.
"""
from __future__ import annotations

import pandas as pd
import pytest

from golazo.monitoring import check_freshness
from golazo.validation import has_errors

AHORA = pd.Timestamp("2026-03-15 12:00:00")


def _almacen(ultimo_jugado="2026-03-14", con_calendario=True, ligas=None, partidos_por_liga=400):
    ligas = ligas or ["EPL", "La liga", "Serie A", "Bundesliga", "Ligue 1",
                      "Championship", "La liga 2", "Serie B", "Bundesliga 2", "Ligue 2"]
    filas = []
    fin = pd.Timestamp(ultimo_jugado)
    for liga in ligas:
        # Temporada anterior, completa.
        for i in range(partidos_por_liga):
            filas.append({"match_id": f"{liga}-2024-{i}", "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i % 200),
                          "season": 2024, "league": liga, "home": f"{liga}A", "away": f"{liga}B",
                          "home_goals": 1.0, "away_goals": 0.0})
        # Temporada en curso, hasta `ultimo_jugado`.
        for i in range(30):
            filas.append({"match_id": f"{liga}-2025-{i}", "date": fin - pd.Timedelta(days=i),
                          "season": 2025, "league": liga, "home": f"{liga}A", "away": f"{liga}B",
                          "home_goals": 2.0, "away_goals": 1.0})
        if con_calendario:
            for i in range(1, 11):
                filas.append({"match_id": f"{liga}-fx-{i}", "date": AHORA + pd.Timedelta(days=i),
                              "season": 2025, "league": liga, "home": f"{liga}A", "away": f"{liga}B",
                              "home_goals": None, "away_goals": None})
    return pd.DataFrame(filas)


def _checks(hallazgos):
    return {f.check for f in hallazgos}


def test_datos_sanos_no_dan_error():
    hallazgos = check_freshness(_almacen(), now=AHORA)
    assert not has_errors(hallazgos)


# --- frescura --------------------------------------------------------------


def test_detecta_que_el_refresco_dejo_de_traer_resultados():
    """El fallo que importa: el pipeline en verde con datos congelados."""
    viejo = _almacen(ultimo_jugado="2026-01-01")
    hallazgos = check_freshness(viejo, now=AHORA)
    assert has_errors(hallazgos)
    assert "frescura" in _checks(hallazgos)


def test_un_parón_corto_es_sólo_aviso():
    hallazgos = check_freshness(_almacen(ultimo_jugado="2026-03-01"), now=AHORA)
    assert not has_errors(hallazgos)
    assert any(f.check == "frescura" and f.severity == "aviso" for f in hallazgos)


def test_almacen_sin_partidos_jugados():
    vacio = pd.DataFrame({"match_id": ["x"], "date": [AHORA + pd.Timedelta(days=1)],
                          "season": [2025], "league": ["EPL"], "home": ["A"], "away": ["B"],
                          "home_goals": [None], "away_goals": [None]})
    assert has_errors(check_freshness(vacio, now=AHORA))


# --- calendario ------------------------------------------------------------


def test_detecta_calendario_vacio():
    """Sin partidos por delante no se puede pronosticar."""
    hallazgos = check_freshness(_almacen(con_calendario=False), now=AHORA)
    assert has_errors(hallazgos)
    assert "calendario" in _checks(hallazgos)


# --- completitud -----------------------------------------------------------


def test_detecta_temporada_cerrada_incompleta():
    """Una descarga parcial deja temporadas pasadas a medias."""
    parcial = _almacen(partidos_por_liga=50)      # frente a los ~380 esperados
    hallazgos = check_freshness(parcial, now=AHORA)
    assert has_errors(hallazgos)
    assert "completitud" in _checks(hallazgos)


def test_la_temporada_en_curso_no_se_exige_completa():
    df = _almacen()
    df.loc[df["season"] == 2025, "season"] = 2025   # en curso respecto de AHORA
    hallazgos = check_freshness(df, now=AHORA)
    assert not any(f.check == "completitud" for f in hallazgos)


# --- cobertura -------------------------------------------------------------


def test_detecta_liga_que_dejo_de_publicarse():
    sin_una = _almacen(ligas=["EPL", "La liga", "Serie A", "Bundesliga", "Ligue 1",
                              "Championship", "La liga 2", "Serie B", "Bundesliga 2"])
    hallazgos = check_freshness(sin_una, now=AHORA)
    assert has_errors(hallazgos)
    assert any("Ligue 2" in f.message for f in hallazgos)


@pytest.mark.parametrize("degradacion", [
    {"ultimo_jugado": "2026-01-01"},
    {"con_calendario": False},
    {"partidos_por_liga": 40},
])
def test_toda_degradación_produce_error(degradacion):
    assert has_errors(check_freshness(_almacen(**degradacion), now=AHORA))
