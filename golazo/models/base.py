"""Interfaz común: todo modelo se evalúa exactamente igual."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Model:
    name = "model"

    def fit(self, train: pd.DataFrame) -> Model:
        raise NotImplementedError

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        """Matriz (n, 3) con probabilidades en el orden H, D, A."""
        raise NotImplementedError
