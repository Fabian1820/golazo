"""Contrato común de las fuentes de datos.

El esquema canónico es la frontera del sistema: una fuente nueva sólo tiene que
producir estas columnas y todo lo demás —features, modelos, backtest, servicio—
funciona sin cambios.

El Elo NO forma parte del esquema. Se calcula en `golazo.elo` a partir de los
resultados, para que histórico y datos nuevos sigan el mismo camino.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import pandas as pd

# Identidad del partido y resultado. Obligatorias.
REQUIRED_COLUMNS = [
    "match_id", "date", "season", "league", "home", "away",
    "home_goals", "away_goals",
]

# Estadísticas avanzadas. Opcionales: no todas las fuentes las tienen, y las
# que falten viajan como NaN en lugar de inventarse.
OPTIONAL_COLUMNS = [
    "home_xg", "away_xg",
    "home_deep", "away_deep",
    "home_ppda", "away_ppda",
    "home_shots", "away_shots",
    "home_sot", "away_sot",
]

CANONICAL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


class MatchSource(Protocol):
    """Una fuente de partidos, histórica o viva."""

    name: str

    def fetch(self, leagues: Iterable[str], seasons: Iterable[int]) -> pd.DataFrame:
        """Partidos ya jugados, en el esquema canónico."""
        ...

    def fixtures(self, leagues: Iterable[str], seasons: Iterable[int]) -> pd.DataFrame:
        """Partidos anunciados y todavía no jugados (sin goles)."""
        ...


def normalize(df: pd.DataFrame, *, require_results: bool = True) -> pd.DataFrame:
    """Lleva un DataFrame al esquema canónico, validando lo imprescindible."""
    faltan = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if require_results:
        if faltan:
            raise ValueError(f"Faltan columnas obligatorias: {faltan}")
    else:
        # En un calendario aún no jugado los goles no existen.
        faltan = [c for c in faltan if c not in ("home_goals", "away_goals")]
        if faltan:
            raise ValueError(f"Faltan columnas obligatorias: {faltan}")

    out = df.copy()
    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out["date"] = pd.to_datetime(out["date"])
    out["match_id"] = out["match_id"].astype(str)
    for col in ("league", "home", "away"):
        out[col] = out[col].astype(str).str.strip()

    numericas = [c for c in CANONICAL_COLUMNS if c not in ("match_id", "date", "league", "home", "away")]
    for col in numericas:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out[CANONICAL_COLUMNS].sort_values("date", kind="mergesort").reset_index(drop=True)


def merge_sources(*frames: pd.DataFrame | None) -> pd.DataFrame:
    """Une varias fuentes resolviendo duplicados por `match_id`.

    Gana la última aparición: al refrescar, los datos nuevos sustituyen a los
    viejos del mismo partido (por ejemplo cuando un partido pasa de anunciado a
    jugado).
    """
    presentes = [f for f in frames if f is not None and not f.empty]
    if not presentes:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    todo = pd.concat(presentes, ignore_index=True)
    todo = todo.drop_duplicates(subset=["match_id"], keep="last")
    return todo.sort_values("date", kind="mergesort").reset_index(drop=True)
