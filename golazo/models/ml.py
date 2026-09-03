"""Gradient boosting sobre features pre-partido.

La pregunta que responde el backtest: ¿aporta algo sobre Dixon-Coles, que sólo
usa identidad de equipo y goles?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..config import OUTCOMES
from .base import Model


class GradientBoostingModel(Model):
    name = "gradient_boosting"

    # Configuración elegida en ventana interna (2019-08 a 2021-06), sin tocar
    # el periodo posterior. Ver scripts/tune_gb.py. La regularización fuerte no
    # es cosmética: con árboles de profundidad 4 y 300 iteraciones fijas el
    # modelo sobreajusta (RPS 0.2172, ECE 0.0787) y queda por detrás del Elo.
    DEFAULTS = dict(
        max_iter=1000, learning_rate=0.03, max_depth=2,
        l2_regularization=5.0, min_samples_leaf=60, random_state=42,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
    )

    def __init__(self, feature_cols=None, **kw):
        self.feature_cols = feature_cols
        params = dict(self.DEFAULTS)
        params.update(kw)
        self.params = params
        self.clf = None
        self.cat_cols = ["league"]

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].copy()
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X

    def fit(self, train: pd.DataFrame) -> GradientBoostingModel:
        if self.feature_cols is None:
            from ..features import feature_columns
            self.feature_cols = feature_columns(train)
        X = self._X(train)
        self.clf = HistGradientBoostingClassifier(
            categorical_features=[c for c in self.cat_cols if c in X.columns], **self.params
        ).fit(X, train["result"])
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        p = self.clf.predict_proba(self._X(test))
        order = [list(self.clf.classes_).index(o) for o in OUTCOMES]
        return p[:, order]
