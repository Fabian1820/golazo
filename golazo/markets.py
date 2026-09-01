"""Mercados derivados de la distribución conjunta de marcadores.

Todo lo que se publica sale de la MISMA matriz P(goles_local, goles_visitante).
Es lo que garantiza coherencia: la probabilidad de over 2.5 y la del 1X2 no
pueden contradecirse porque son márgenes del mismo objeto.
"""
from __future__ import annotations

import numpy as np


def outcome_1x2(mat: np.ndarray) -> dict[str, float]:
    return {
        "H": float(np.tril(mat, -1).sum()),
        "D": float(np.trace(mat)),
        "A": float(np.triu(mat, 1).sum()),
    }


def total_goals_distribution(mat: np.ndarray) -> np.ndarray:
    """P(goles totales = k) para k = 0, 1, 2, ..."""
    n = mat.shape[0]
    out = np.zeros(2 * n - 1)
    for i in range(n):
        out[i:i + n] += mat[i]
    return out


def over_under(mat: np.ndarray, lines=(0.5, 1.5, 2.5, 3.5, 4.5)) -> dict[str, dict[str, float]]:
    """Probabilidad de superar cada línea de goles totales."""
    dist = total_goals_distribution(mat)
    k = np.arange(len(dist))
    out = {}
    for line in lines:
        over = float(dist[k > line].sum())
        out[f"{line}"] = {"over": over, "under": 1.0 - over}
    return out


def both_teams_score(mat: np.ndarray) -> dict[str, float]:
    si = float(mat[1:, 1:].sum())
    return {"yes": si, "no": 1.0 - si}


def top_scorelines(mat: np.ndarray, n: int = 5) -> list[dict[str, float]]:
    """Los `n` marcadores exactos más probables."""
    flat = np.dstack(np.unravel_index(np.argsort(mat, axis=None)[::-1], mat.shape))[0]
    return [{"score": f"{int(i)}-{int(j)}", "probability": float(mat[i, j])} for i, j in flat[:n]]


def expected_goals(mat: np.ndarray) -> dict[str, float]:
    """Goles esperados según la distribución (no el parámetro crudo del ajuste)."""
    k = np.arange(mat.shape[0])
    home = float((mat.sum(axis=1) * k).sum())
    away = float((mat.sum(axis=0) * k).sum())
    return {"home": home, "away": away, "total": home + away}


def asian_handicap(mat: np.ndarray, line: float = -0.5) -> dict[str, float]:
    """Hándicap entero o de medio gol aplicado al equipo local."""
    n = mat.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    margen = i - j + line
    return {
        "home": float(mat[margen > 0].sum()),
        "push": float(mat[margen == 0].sum()),
        "away": float(mat[margen < 0].sum()),
    }


def summarize(mat: np.ndarray) -> dict:
    """Todos los mercados derivados de una sola matriz."""
    return {
        "outcome": outcome_1x2(mat),
        "expected_goals": expected_goals(mat),
        "over_under": over_under(mat),
        "both_teams_score": both_teams_score(mat),
        "top_scorelines": top_scorelines(mat),
    }


def coherence_check(mat: np.ndarray, tol: float = 1e-9) -> tuple[bool, str]:
    """Verifica que la matriz es una distribución de probabilidad válida."""
    if mat.min() < -tol:
        return False, f"probabilidad negativa: {mat.min()}"
    total = mat.sum()
    if abs(total - 1.0) > 1e-6:
        return False, f"la matriz no suma 1: {total}"
    o = outcome_1x2(mat)
    if abs(sum(o.values()) - 1.0) > 1e-6:
        return False, f"el 1X2 no suma 1: {o}"
    return True, "ok"
