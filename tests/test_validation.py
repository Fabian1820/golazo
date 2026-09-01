"""El validador debe detectar cada clase de error que sabe buscar.

Un validador que nunca falla no valida nada: por eso cada test corrompe los
datos de una forma concreta y exige que el hallazgo aparezca.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from golazo.validation import has_errors, report, validate


@pytest.fixture
def limpio():
    n = 40
    fechas = pd.date_range("2021-08-01", periods=n, freq="3D")
    rng = np.random.default_rng(0)
    hg = rng.integers(0, 4, n)
    ag = rng.integers(0, 3, n)
    hg[:20] = 2   # asegura ventaja de local en la muestra
    ag[:20] = 0
    return pd.DataFrame({
        "match_id": np.arange(n),
        "date": fechas,
        "season": 2021,
        "league": "EPL",
        "home": [f"H{i % 8}" for i in range(n)],
        "away": [f"H{(i + 3) % 8}" for i in range(n)],
        "elo_h": 1700.0, "elo_a": 1650.0,
        "home_goals": hg, "away_goals": ag,
        "home_xg": 1.5, "away_xg": 1.0,
        "home_shots": 13, "away_shots": 10,
        "home_sot": 5, "away_sot": 3,
        "home_deep": 7, "away_deep": 5,
        "home_ppda": 10.0, "away_ppda": 12.0,
        "result": np.where(hg > ag, "H", np.where(hg < ag, "A", "D")),
    })


def _checks(findings):
    return {f.check for f in findings}


def test_datos_limpios_no_dan_errores(limpio):
    hallazgos = validate(limpio)
    assert not has_errors(hallazgos), report(hallazgos)


def test_detecta_nulos_en_columna_obligatoria(limpio):
    limpio.loc[3, "home_goals"] = np.nan
    hallazgos = validate(limpio)
    assert has_errors(hallazgos)
    assert "nulos" in _checks(hallazgos)


def test_los_nulos_en_columnas_opcionales_son_solo_aviso(limpio):
    """Understat no publica tiros. Tratarlo como error enseñaría a ignorar el
    validador, que es peor que no tenerlo."""
    limpio.loc[3, "home_shots"] = np.nan
    hallazgos = validate(limpio)
    assert not has_errors(hallazgos)
    assert any(f.check == "nulos" and f.severity == "aviso" for f in hallazgos)


def test_detecta_match_id_duplicado(limpio):
    limpio.loc[5, "match_id"] = limpio.loc[4, "match_id"]
    assert "duplicados" in _checks(validate(limpio))


def test_detecta_partido_repetido(limpio):
    limpio.loc[6, ["date", "home", "away"]] = limpio.loc[5, ["date", "home", "away"]].values
    assert "duplicados" in _checks(validate(limpio))


def test_detecta_valores_fuera_de_rango(limpio):
    limpio.loc[2, "home_goals"] = 99
    hallazgos = validate(limpio)
    assert has_errors(hallazgos)
    assert "rango" in _checks(hallazgos)


def test_detecta_elo_imposible(limpio):
    limpio.loc[1, "elo_h"] = 50.0
    assert "rango" in _checks(validate(limpio))


def test_detecta_mas_tiros_a_puerta_que_tiros(limpio):
    limpio.loc[7, "home_sot"] = 40
    hallazgos = validate(limpio)
    assert has_errors(hallazgos)
    assert "coherencia" in _checks(hallazgos)


def test_detecta_resultado_incoherente(limpio):
    limpio.loc[0, "result"] = "A"          # era victoria local
    hallazgos = validate(limpio)
    assert has_errors(hallazgos)
    assert any("no concuerda" in f.message for f in hallazgos)


def test_detecta_equipo_contra_si_mismo(limpio):
    limpio.loc[9, "away"] = limpio.loc[9, "home"]
    assert "coherencia" in _checks(validate(limpio))


def test_detecta_desorden_cronologico(limpio):
    desordenado = limpio.iloc[::-1].reset_index(drop=True)
    hallazgos = validate(desordenado)
    assert "orden" in _checks(hallazgos)
    assert has_errors(hallazgos)


def test_detecta_columnas_intercambiadas(limpio):
    """Si local y visitante se invierten, la ventaja de local desaparece.

    Es el error de datos más insidioso: todo parece válido y el modelo se
    entrena al revés.
    """
    invertido = limpio.copy()
    invertido["home_goals"], invertido["away_goals"] = limpio["away_goals"], limpio["home_goals"]
    invertido["result"] = np.where(invertido["home_goals"] > invertido["away_goals"], "H",
                                   np.where(invertido["home_goals"] < invertido["away_goals"], "A", "D"))
    hallazgos = validate(invertido)
    assert has_errors(hallazgos)
    assert any("intercambiadas" in f.message for f in hallazgos)


def test_avisa_de_datos_obsoletos(limpio):
    hallazgos = validate(limpio)
    assert any(f.check == "frescura" for f in hallazgos)
    assert not any(f.check == "frescura" and f.severity == "error" for f in hallazgos)


def test_el_informe_es_legible(limpio):
    texto = report(validate(limpio))
    assert "errores" in texto or "Sin problemas" in texto


def test_los_datos_reales_no_tienen_errores():
    """El dataset del repositorio debe pasar la validación."""
    from golazo.data import load_matches

    hallazgos = validate(load_matches())
    assert not has_errors(hallazgos), report(hallazgos)
