"""Réplica del modelo original, para cuantificar su valor real.

Reproduce fielmente `src/predictor.py`:

* entrena RandomForestRegressor sobre estadísticas del PROPIO partido
  (tiros, tiros a puerta, deep, ppda) para predecir sus goles -> fuga de datos;
* al predecir, sustituye esas estadísticas por las del ÚLTIMO partido de cada
  equipo -> desajuste entre entrenamiento y producción;
* envuelve el punto estimado en una Poisson independiente.

Se incluye en el backtest para que la comparación sea explícita, no retórica.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import RandomForestRegressor

from .base import Model

MAX_GOALS = 10
_FEATS = ["home_shots", "away_shots", "home_sot", "away_sot", "home_deep", "away_deep", "home_ppda", "away_ppda"]


class LegacyLeakyModel(Model):
    name = "original_fugado"

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.kw = dict(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        self.rf_home = None
        self.rf_away = None
        self.last_home = {}
        self.last_away = {}
        self.league_mean = (1.53, 1.25)

    def fit(self, train: pd.DataFrame) -> LegacyLeakyModel:
        X = train[_FEATS].to_numpy(dtype=float)
        self.rf_home = RandomForestRegressor(**self.kw).fit(X, train["home_goals"])
        self.rf_away = RandomForestRegressor(**self.kw).fit(X, train["away_goals"])
        # Último partido como local / visitante de cada equipo (lo que hacía el original)
        h = train.sort_values("date").groupby("home").last()
        a = train.sort_values("date").groupby("away").last()
        self.last_home = h[["home_shots", "home_sot", "home_deep", "home_ppda"]].to_dict("index")
        self.last_away = a[["away_shots", "away_sot", "away_deep", "away_ppda"]].to_dict("index")
        self.league_mean = (float(train["home_goals"].mean()), float(train["away_goals"].mean()))
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        rows, known = [], []
        for r in test.itertuples(index=False):
            h, a = self.last_home.get(r.home), self.last_away.get(r.away)
            known.append(h is not None and a is not None)
            h = h or {"home_shots": 12, "home_sot": 4, "home_deep": 7, "home_ppda": 11}
            a = a or {"away_shots": 12, "away_sot": 4, "away_deep": 7, "away_ppda": 11}
            rows.append([h["home_shots"], a["away_shots"], h["home_sot"], a["away_sot"],
                         h["home_deep"], a["away_deep"], h["home_ppda"], a["away_ppda"]])

        X = np.asarray(rows, dtype=float)
        lam = np.clip(self.rf_home.predict(X), 0.05, None)
        mu = np.clip(self.rf_away.predict(X), 0.05, None)
        lam[~np.array(known)] = self.league_mean[0]
        mu[~np.array(known)] = self.league_mean[1]

        k = np.arange(MAX_GOALS + 1)
        out = np.empty((len(test), 3))
        for i in range(len(test)):
            mat = np.outer(poisson.pmf(k, lam[i]), poisson.pmf(k, mu[i]))
            mat /= mat.sum()
            out[i] = (np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum())
        return out
