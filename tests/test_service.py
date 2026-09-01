"""El servicio debe calcular exactamente las mismas features que el entrenamiento.

El modelo original fallaba por dos motivos: fuga de datos (cubierto en
`test_features_leakage.py`) y **desajuste entre entrenamiento y producción**:
entrenaba con unas variables y en el momento de predecir recibía otras.

Estos tests cierran esa segunda vía.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from golazo.data import load_matches
from golazo.features import FeatureBuilder, build_features, feature_columns


@pytest.fixture(scope="module")
def historia():
    return load_matches()


@pytest.fixture(scope="module")
def con_features(historia):
    return build_features(historia)


@pytest.mark.parametrize("k", [500, 3000, 6000, 9028])
def test_emit_coincide_con_el_entrenamiento(historia, con_features, k):
    """Reconstruir el estado hasta el partido k y emitir sus features debe dar
    exactamente la misma fila que produjo `build_features`."""
    builder = FeatureBuilder.from_history(historia.iloc[:k])
    fila = historia.iloc[k]
    emitido = builder.emit(fila["home"], fila["away"], fila["date"], fila["elo_h"], fila["elo_a"])

    cols = [c for c in feature_columns(con_features) if c in emitido]
    esperado = con_features.iloc[k][cols].to_numpy(dtype=float)
    obtenido = np.array([emitido[c] for c in cols], dtype=float)

    np.testing.assert_allclose(
        np.nan_to_num(esperado, nan=-9e9), np.nan_to_num(obtenido, nan=-9e9),
        rtol=0, atol=0,
        err_msg="las features de producción difieren de las de entrenamiento: DESAJUSTE")


def test_emit_no_muta_el_estado(historia):
    """`emit` debe ser puro: dos llamadas seguidas dan lo mismo."""
    builder = FeatureBuilder.from_history(historia.iloc[:1000])
    fila = historia.iloc[1000]
    args = (fila["home"], fila["away"], fila["date"], fila["elo_h"], fila["elo_a"])
    a, b = builder.emit(*args), builder.emit(*args)
    assert a == b or all(
        (np.isnan(a[k]) and np.isnan(b[k])) if isinstance(a[k], float) else a[k] == b[k]
        for k in a)


def test_ingest_avanza_el_estado(historia):
    builder = FeatureBuilder.from_history(historia.iloc[:1000])
    equipo = historia.iloc[999]["home"]
    antes = len(builder.hist[equipo])
    for fila in historia.iloc[1000:1010].itertuples(index=False):
        builder.ingest(fila)
    assert builder.last_date == historia.iloc[1009]["date"]
    assert len(builder.hist[equipo]) >= antes


def test_elo_disponible_para_todos_los_equipos(historia):
    builder = FeatureBuilder.from_history(historia)
    for equipo in set(historia["home"]) | set(historia["away"]):
        assert builder.elo_for(equipo) is not None
    assert builder.elo_for("Equipo Inexistente") is None


# --- servicio completo -----------------------------------------------------


@pytest.fixture(scope="module")
def servicio(historia):
    from golazo.artifacts import Metadata, ModelArtifact
    from golazo.models import DixonColes
    from golazo.service import PredictionService

    con_feat = build_features(historia)
    modelo = DixonColes().fit(con_feat)
    meta = Metadata(model_name="dixon_coles", trained_at="2023-06-05T00:00:00+00:00",
                    train_start="2018-08-10", train_end="2023-06-04",
                    n_matches=len(historia), leagues=sorted(historia["league"].unique()))
    return PredictionService(ModelArtifact(modelo, meta), historia)


def test_probabilidades_validas(servicio):
    p = servicio.predict("Liverpool", "Everton")
    assert set(p.probabilities) == {"H", "D", "A"}
    assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0 <= v <= 1 for v in p.probabilities.values())


def test_la_localia_importa(servicio):
    """El mismo enfrentamiento invertido debe favorecer al nuevo local."""
    a = servicio.predict("Liverpool", "Everton")
    b = servicio.predict("Everton", "Liverpool")
    assert a.probabilities["H"] > b.probabilities["A"]


def test_equipo_desconocido(servicio):
    from golazo.service import UnknownTeamError

    with pytest.raises(UnknownTeamError):
        servicio.predict("Real Bostezo CF", "Liverpool")


def test_no_se_juega_contra_si_mismo(servicio):
    with pytest.raises(ValueError):
        servicio.predict("Liverpool", "Liverpool")


def test_avisa_si_el_modelo_esta_desactualizado(servicio, historia):
    """Un partido muy posterior al último dato debe avisar del desfase."""
    lejano = historia["date"].max() + pd.Timedelta(days=400)
    p = servicio.predict("Liverpool", "Everton", kickoff=lejano)
    assert any("entrenó con datos hasta" in w for w in p.warnings)


def test_no_avisa_si_el_modelo_esta_al_dia(servicio, historia):
    cercano = historia["date"].max() + pd.Timedelta(days=3)
    p = servicio.predict("Liverpool", "Everton", kickoff=cercano)
    assert not any("entrenó con datos hasta" in w for w in p.warnings)


def test_los_mercados_son_coherentes_con_el_1x2(servicio):
    """El 1X2 publicado debe coincidir con el margen de la matriz de marcadores."""
    p = servicio.predict("Arsenal", "Manchester City")
    assert p.markets is not None
    for o in ("H", "D", "A"):
        assert p.markets["outcome"][o] == pytest.approx(p.probabilities[o], abs=1e-9)

    eg = p.markets["expected_goals"]
    assert eg["total"] == pytest.approx(eg["home"] + eg["away"], abs=1e-6)

    btts = p.markets["both_teams_score"]
    assert btts["yes"] + btts["no"] == pytest.approx(1.0, abs=1e-9)

    for v in p.markets["over_under"].values():
        assert v["over"] + v["under"] == pytest.approx(1.0, abs=1e-9)


def test_catalogo(servicio):
    assert "EPL" in servicio.leagues
    assert "Liverpool" in servicio.teams("EPL")
    assert servicio.league_of("Liverpool") == "EPL"
    with pytest.raises(ValueError):
        servicio.teams("Liga Inexistente")
