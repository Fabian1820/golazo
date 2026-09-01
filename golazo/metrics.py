"""Métricas propias de pronóstico probabilístico.

Un modelo de fútbol no se juzga por accuracy: acertar el ganador importa menos
que asignar probabilidades *bien calibradas*. Estas son las tres estándar más
la curva de fiabilidad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OUTCOMES

EPS = 1e-15


def _onehot(y) -> np.ndarray:
    y = np.asarray(y)
    idx = {o: i for i, o in enumerate(OUTCOMES)}
    out = np.zeros((len(y), len(OUTCOMES)))
    out[np.arange(len(y)), [idx[v] for v in y]] = 1.0
    return out


def log_loss(p: np.ndarray, y) -> float:
    """Log-loss multiclase. Penaliza con dureza la confianza mal puesta."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    return float(-np.mean(np.log((p * _onehot(y)).sum(axis=1))))


def brier(p: np.ndarray, y) -> float:
    """Brier multiclase: suma de errores cuadráticos sobre los 3 resultados (0-2)."""
    return float(np.mean(((np.asarray(p, dtype=float) - _onehot(y)) ** 2).sum(axis=1)))


def rps(p: np.ndarray, y) -> float:
    """Ranked Probability Score: la métrica estándar para 1X2.

    A diferencia de log-loss y Brier, respeta el orden H < D < A: equivocarse
    dando un empate cuesta menos que dando la victoria contraria.
    """
    p = np.asarray(p, dtype=float)
    e = _onehot(y)
    cp, ce = np.cumsum(p, axis=1), np.cumsum(e, axis=1)
    return float(np.mean(((cp[:, :-1] - ce[:, :-1]) ** 2).sum(axis=1) / (len(OUTCOMES) - 1)))


def accuracy(p: np.ndarray, y) -> float:
    """Se reporta sólo como referencia; no es el criterio de selección."""
    pred = np.asarray(OUTCOMES)[np.asarray(p, dtype=float).argmax(axis=1)]
    return float(np.mean(pred == np.asarray(y)))


def summarize(p: np.ndarray, y) -> dict:
    return {
        "n": int(len(y)),
        "rps": rps(p, y),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
        "accuracy": accuracy(p, y),
    }


def skill_score(value: float, reference: float) -> float:
    """Fracción de la métrica de referencia que se mejora. >0 es mejor que la referencia."""
    return float(1.0 - value / reference) if reference else float("nan")


def calibration_table(p: np.ndarray, y, n_bins: int = 10) -> pd.DataFrame:
    """Fiabilidad: agrupa TODAS las probabilidades emitidas por bin y compara
    la media predicha contra la frecuencia observada. Un modelo calibrado cae
    sobre la diagonal.
    """
    p = np.asarray(p, dtype=float).ravel()
    e = _onehot(y).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin_low": bins[b],
            "bin_high": bins[b + 1],
            "n": int(mask.sum()),
            "pred_mean": float(p[mask].mean()),
            "obs_freq": float(e[mask].mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(p: np.ndarray, y, n_bins: int = 10) -> float:
    """ECE: desviación media (ponderada) entre probabilidad predicha y observada."""
    tab = calibration_table(p, y, n_bins)
    if tab.empty:
        return float("nan")
    w = tab["n"] / tab["n"].sum()
    return float((w * (tab["pred_mean"] - tab["obs_freq"]).abs()).sum())
