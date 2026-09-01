"""Dixon-Coles se valida recuperando parámetros conocidos de datos sintéticos.

Es la prueba más fuerte disponible para un modelo paramétrico: se generan
partidos con fuerzas de ataque/defensa y ventaja de local fijadas por nosotros
y se comprueba que el ajuste las reencuentra.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from golazo.models.dixon_coles import DixonColes, _LeagueFit, _tau


@pytest.fixture(scope="module")
def liga_sintetica():
    rng = np.random.default_rng(7)
    equipos = [f"T{i:02d}" for i in range(20)]
    atk = dict(zip(equipos, np.linspace(0.45, -0.45, len(equipos))))
    dfc = dict(zip(equipos, np.linspace(-0.35, 0.35, len(equipos))))
    gamma = 0.28

    filas, fecha = [], pd.Timestamp("2015-01-01")
    for _ in range(14):  # muchas vueltas: el ajuste necesita señal
        for h in equipos:
            for a in equipos:
                if h == a:
                    continue
                lam = np.exp(atk[h] + dfc[a] + gamma)
                mu = np.exp(atk[a] + dfc[h])
                filas.append({
                    "date": fecha, "league": "SYN", "home": h, "away": a,
                    "home_goals": rng.poisson(lam), "away_goals": rng.poisson(mu),
                })
                fecha += pd.Timedelta(hours=6)
    return pd.DataFrame(filas), atk, dfc, gamma


def test_recupera_los_parametros_verdaderos(liga_sintetica):
    df, atk, dfc, gamma = liga_sintetica
    # xi=0 desactiva el decaimiento temporal: aquí las fuerzas son constantes.
    fit = _LeagueFit(xi=0.0).fit(df, df["date"].max())
    assert fit.converged

    equipos = sorted(atk)
    atk_real = np.array([atk[t] for t in equipos]) - np.mean(list(atk.values()))
    atk_est = np.array([fit.atk[t] for t in equipos])
    dfc_real = np.array([dfc[t] for t in equipos])
    dfc_est = np.array([fit.dfc[t] for t in equipos])

    assert np.corrcoef(atk_real, atk_est)[0, 1] > 0.97
    assert np.abs(atk_real - atk_est).max() < 0.12
    # La defensa sólo se identifica salvo una constante; se compara centrada.
    assert np.corrcoef(dfc_real, dfc_est)[0, 1] > 0.97
    assert fit.gamma == pytest.approx(gamma, abs=0.06)


def test_probabilidades_validas(liga_sintetica):
    df, *_ = liga_sintetica
    fit = _LeagueFit(xi=0.0).fit(df, df["date"].max())
    p = fit.outcome_probs("T00", "T19")
    assert len(p) == 3
    assert all(0.0 <= v <= 1.0 for v in p)
    assert sum(p) == pytest.approx(1.0, abs=1e-6)


def test_la_ventaja_de_local_es_asimetrica(liga_sintetica):
    """El mismo emparejamiento invertido debe favorecer al nuevo local."""
    df, *_ = liga_sintetica
    fit = _LeagueFit(xi=0.0).fit(df, df["date"].max())
    h1, _, a1 = fit.outcome_probs("T05", "T06")
    h2, _, a2 = fit.outcome_probs("T06", "T05")
    assert h1 > a2, "jugar en casa debe aumentar la probabilidad de ganar"
    assert a1 < h2


def test_equipo_desconocido_cae_al_promedio_de_liga(liga_sintetica):
    """Un ascendido sin historial no debe romper la predicción."""
    df, *_ = liga_sintetica
    fit = _LeagueFit(xi=0.0).fit(df, df["date"].max())
    p = fit.outcome_probs("RECIEN_ASCENDIDO", "T00")
    assert sum(p) == pytest.approx(1.0, abs=1e-6)
    assert all(np.isfinite(p))


def test_tau_es_neutro_fuera_de_los_marcadores_bajos():
    lam = np.array([1.4])
    mu = np.array([1.1])
    rho = -0.08
    assert _tau(np.array(2), np.array(3), lam, mu, rho).item() == pytest.approx(1.0)
    assert _tau(np.array(1), np.array(1), lam, mu, rho).item() == pytest.approx(1.0 - rho)
    assert _tau(np.array(0), np.array(0), lam, mu, rho).item() == pytest.approx(1.0 - lam[0] * mu[0] * rho)


def test_decaimiento_temporal_pondera_lo_reciente():
    """Un equipo que empeora bruscamente debe verse peor con decaimiento fuerte."""
    filas, fecha = [], pd.Timestamp("2015-01-01")
    for i in range(400):
        fuerte = i < 200  # primera mitad: golea; segunda mitad: no marca
        filas.append({
            "date": fecha, "league": "SYN", "home": "A", "away": f"R{i % 8}",
            "home_goals": 4 if fuerte else 0, "away_goals": 0 if fuerte else 2,
        })
        fecha += pd.Timedelta(days=3)
    df = pd.DataFrame(filas)
    ref = df["date"].max()
    sin_decaimiento = _LeagueFit(xi=0.0).fit(df, ref)
    con_decaimiento = _LeagueFit(xi=0.02).fit(df, ref)
    assert con_decaimiento.atk["A"] < sin_decaimiento.atk["A"]


def test_ajusta_una_liga_por_separado():
    rng = np.random.default_rng(3)
    filas, fecha = [], pd.Timestamp("2016-01-01")
    for liga in ("L1", "L2"):
        for i in range(600):
            filas.append({
                "date": fecha, "league": liga,
                "home": f"{liga}_A{i % 6}", "away": f"{liga}_B{i % 5}",
                "home_goals": rng.poisson(1.6), "away_goals": rng.poisson(1.2),
            })
            fecha += pd.Timedelta(days=1)
    df = pd.DataFrame(filas)
    m = DixonColes(xi=0.0).fit(df)
    assert set(m.fits) == {"L1", "L2"}
    # Los equipos de una liga no aparecen en el ajuste de la otra.
    assert not any(t.startswith("L2") for t in m.fits["L1"].teams)
