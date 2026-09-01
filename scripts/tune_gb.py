"""Selección de hiperparámetros del GB en ventana INTERNA (2019-08 a 2021-06).

El periodo posterior no se toca aquí, para que la tabla final del informe no
quede contaminada por esta búsqueda.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from golazo.backtest import evaluate, walk_forward
from golazo.data import load_matches
from golazo.features import build_features, feature_columns
from golazo.models import BaseRateModel, DixonColes, EloLogisticModel, GradientBoostingModel

df = build_features(load_matches())
cols = feature_columns(df)
inner = df[df.date < pd.Timestamp('2021-07-01')]

GRID = {
    "gb_actual": dict(max_iter=300, learning_rate=0.05, max_depth=4,
                      l2_regularization=1.0, min_samples_leaf=40),
    "gb_early": dict(max_iter=1000, learning_rate=0.05, max_depth=3,
                     l2_regularization=1.0, min_samples_leaf=40,
                     early_stopping=True, validation_fraction=0.15,
                     n_iter_no_change=25),
    "gb_shallow": dict(max_iter=1000, learning_rate=0.03, max_depth=2,
                       l2_regularization=5.0, min_samples_leaf=60,
                       early_stopping=True, validation_fraction=0.15,
                       n_iter_no_change=25),
}

fac = {"tasa_base": BaseRateModel, "elo_logistico": EloLogisticModel,
       "dixon_coles": DixonColes}
for k, v in GRID.items():
    fac[k] = (lambda v=v: GradientBoostingModel(feature_cols=cols, **v))

preds = walk_forward(inner, fac, start='2019-08-01', refit_days=60)
pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:8.4f}")
print("\n=== VENTANA INTERNA (2019-08 a 2021-06) ===")
print(evaluate(preds, list(fac), reference="tasa_base").to_string())
