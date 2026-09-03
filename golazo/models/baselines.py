"""Baselines mínimos. Todo modelo debe superarlos o no merece existir."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..config import OUTCOMES
from .base import Model


class UniformModel(Model):
    """1/3 a cada resultado. El suelo absoluto: log-loss = ln(3) = 1.0986."""

    name = "uniforme"

    def fit(self, train: pd.DataFrame) -> UniformModel:
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        return np.full((len(test), 3), 1.0 / 3.0)


class BaseRateModel(Model):
    """Frecuencia histórica de local/empate/visitante. Ignora quién juega."""

    name = "tasa_base"

    def __init__(self):
        self.rates = np.array([1 / 3, 1 / 3, 1 / 3])

    def fit(self, train: pd.DataFrame) -> BaseRateModel:
        counts = train["result"].value_counts()
        self.rates = np.array([counts.get(o, 0) for o in OUTCOMES], dtype=float)
        self.rates /= self.rates.sum()
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        return np.tile(self.rates, (len(test), 1))


class EloLogisticModel(Model):
    """Regresión logística multinomial sobre la diferencia de Elo.

    Referencia barata para aislar cuánto aporta el Elo por sí solo.
    """

    name = "elo_logistico"

    def __init__(self):
        self.clf = None
        self.classes_ = np.array(OUTCOMES)

    def fit(self, train: pd.DataFrame) -> EloLogisticModel:
        X = train[["elo_diff"]].to_numpy(dtype=float)
        self.clf = LogisticRegression(max_iter=1000, C=1.0).fit(X, train["result"])
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        p = self.clf.predict_proba(test[["elo_diff"]].to_numpy(dtype=float))
        order = [list(self.clf.classes_).index(o) for o in OUTCOMES]
        return p[:, order]
