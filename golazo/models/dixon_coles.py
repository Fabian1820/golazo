"""Dixon-Coles (1997): Poisson bivariado con corrección de marcadores bajos.

Es el baseline de referencia en la literatura de predicción de fútbol y es
notoriamente difícil de superar. Si un modelo de ML no le gana, no aporta.

    log λ = ataque[local] + defensa[visitante] + ventaja_local
    log μ = ataque[visitante] + defensa[local]

con la corrección τ sobre los marcadores 0-0, 0-1, 1-0 y 1-1, donde la
independencia de Poisson falla empíricamente, y un decaimiento exponencial que
pondera más los partidos recientes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .base import Model

MAX_GOALS = 10
TAU_FLOOR = 1e-10


def _tau(x, y, lam, mu, rho):
    """Corrección de dependencia en marcadores bajos."""
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return np.maximum(t, TAU_FLOOR)


class _LeagueFit:
    """Ajuste para una liga concreta."""

    def __init__(self, xi: float):
        self.xi = xi
        self.teams: list = []
        self.atk = {}
        self.dfc = {}
        self.gamma = 0.0
        self.rho = 0.0
        self.converged = False

    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp) -> "_LeagueFit":
        self.teams = sorted(set(df["home"]) | set(df["away"]))
        n = len(self.teams)
        idx = {t: i for i, t in enumerate(self.teams)}
        ih = df["home"].map(idx).to_numpy()
        ia = df["away"].map(idx).to_numpy()
        x = df["home_goals"].to_numpy(dtype=float)
        y = df["away_goals"].to_numpy(dtype=float)

        age = (ref_date - df["date"]).dt.total_seconds().to_numpy() / 86400.0
        w = np.exp(-self.xi * np.maximum(age, 0.0))

        def nll(p):
            atk = p[:n] - p[:n].mean()          # identificabilidad: media de ataque = 0
            dfc = p[n:2 * n]
            gamma, rho = p[2 * n], p[2 * n + 1]
            lam = np.exp(atk[ih] + dfc[ia] + gamma)
            mu = np.exp(atk[ia] + dfc[ih])
            ll = (x * np.log(lam) - lam) + (y * np.log(mu) - mu) + np.log(_tau(x, y, lam, mu, rho))
            return -float(np.sum(w * ll))

        p0 = np.zeros(2 * n + 2)
        p0[2 * n] = 0.25  # ventaja de local inicial
        bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 1.0), (-0.25, 0.25)]
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-9})

        p = res.x
        atk = p[:n] - p[:n].mean()
        self.atk = dict(zip(self.teams, atk))
        self.dfc = dict(zip(self.teams, p[n:2 * n]))
        self.gamma = float(p[2 * n])
        self.rho = float(p[2 * n + 1])
        self.converged = bool(res.success)
        return self

    def rates(self, home, away):
        """Goles esperados. Un equipo sin historial usa el promedio de liga (0)."""
        a_h, d_h = self.atk.get(home, 0.0), self.dfc.get(home, 0.0)
        a_a, d_a = self.atk.get(away, 0.0), self.dfc.get(away, 0.0)
        return np.exp(a_h + d_a + self.gamma), np.exp(a_a + d_h)

    def scoreline_matrix(self, home, away) -> np.ndarray:
        """Distribución conjunta P(goles_local=i, goles_visitante=j), normalizada.

        Es el objeto del que se derivan de forma coherente el 1X2, el over/under,
        el 'ambos marcan' y el marcador exacto. Un único modelo, sin que las
        distintas cifras puedan contradecirse entre sí.
        """
        lam, mu = self.rates(home, away)
        mat = np.outer(poisson.pmf(np.arange(MAX_GOALS + 1), lam),
                       poisson.pmf(np.arange(MAX_GOALS + 1), mu))
        for (i, j) in ((0, 0), (0, 1), (1, 0), (1, 1)):
            mat[i, j] *= _tau(np.array(i), np.array(j), np.array(lam), np.array(mu), self.rho).item()
        return mat / mat.sum()

    def outcome_probs(self, home, away):
        mat = self.scoreline_matrix(home, away)
        return (float(np.tril(mat, -1).sum()), float(np.trace(mat)), float(np.triu(mat, 1).sum()))


class DixonColes(Model):
    """Ajusta un modelo independiente por liga (los equipos no se cruzan)."""

    name = "dixon_coles"

    def __init__(self, xi: float = 0.0018):
        # xi=0.0018/día => vida media ~385 días, el rango habitual en la literatura
        self.xi = xi
        self.fits: dict = {}
        self.fallback = (0.4344, 0.2516, 0.3140)

    def fit(self, train: pd.DataFrame) -> "DixonColes":
        ref = train["date"].max()
        self.fits = {}
        for league, sub in train.groupby("league"):
            if len(sub) >= 50:
                self.fits[league] = _LeagueFit(self.xi).fit(sub, ref)
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(test), 3))
        for k, r in enumerate(test.itertuples(index=False)):
            fit = self.fits.get(r.league)
            out[k] = fit.outcome_probs(r.home, r.away) if fit else self.fallback
        return out
