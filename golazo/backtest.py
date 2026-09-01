"""Backtest walk-forward.

Se entrena sólo con el pasado y se predice el futuro inmediato, avanzando en
bloques. Es la única forma válida de evaluar series temporales: el
`train_test_split` aleatorio del modelo original entrenaba con partidos de
2023 para predecir 2019, e inflaba cualquier métrica.
"""
from __future__ import annotations

import sys
import time
from typing import Callable

import numpy as np
import pandas as pd

from .config import OUTCOMES
from .models.base import Model

ModelFactory = Callable[[], Model]


def walk_forward(
    df: pd.DataFrame,
    factories: dict[str, ModelFactory],
    start: str,
    refit_days: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    """Devuelve un DataFrame con una fila por partido predicho y las
    probabilidades de cada modelo (`<modelo>_H/_D/_A`).
    """
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    start_ts = pd.Timestamp(start)
    horizon = df.loc[df["date"] >= start_ts, "date"]
    if horizon.empty:
        raise ValueError(f"No hay partidos desde {start}")

    edges = pd.date_range(horizon.min().normalize(), horizon.max().normalize() + pd.Timedelta(days=refit_days),
                          freq=f"{refit_days}D")
    out, t0, n_blocks = [], time.time(), len(edges) - 1

    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        test = df[(df["date"] >= lo) & (df["date"] < hi)]
        if test.empty:
            continue
        train = df[df["date"] < lo]
        if len(train) < 200:
            continue

        block = test[["match_id", "date", "league", "home", "away", "result"]].copy()
        for name, factory in factories.items():
            p = factory().fit(train).predict_proba(test)
            p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
            p = p / p.sum(axis=1, keepdims=True)
            for k, o in enumerate(OUTCOMES):
                block[f"{name}_{o}"] = p[:, k]
        out.append(block)

        if verbose and (b % 20 == 0 or b == n_blocks - 1):
            done = sum(len(x) for x in out)
            sys.stderr.write(f"\r  bloque {b + 1}/{n_blocks} · {done} partidos · {time.time() - t0:.0f}s")
            sys.stderr.flush()

    if verbose:
        sys.stderr.write("\n")
    if not out:
        raise ValueError("El backtest no produjo predicciones")
    return pd.concat(out, ignore_index=True)


def evaluate(preds: pd.DataFrame, model_names, reference: str = "tasa_base") -> pd.DataFrame:
    """Tabla comparativa de métricas, con skill score contra un modelo de referencia."""
    from .metrics import expected_calibration_error, skill_score, summarize

    y = preds["result"].to_numpy()
    rows = {}
    for name in model_names:
        p = preds[[f"{name}_{o}" for o in OUTCOMES]].to_numpy()
        s = summarize(p, y)
        s["ece"] = expected_calibration_error(p, y)
        rows[name] = s

    ref = rows[reference]
    for s in rows.values():
        s["rps_skill_%"] = 100.0 * skill_score(s["rps"], ref["rps"])
        s["logloss_skill_%"] = 100.0 * skill_score(s["log_loss"], ref["log_loss"])

    cols = ["n", "rps", "log_loss", "brier", "ece", "accuracy", "rps_skill_%", "logloss_skill_%"]
    return pd.DataFrame(rows).T[cols].sort_values("rps")
