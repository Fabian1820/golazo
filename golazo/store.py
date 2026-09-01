"""Almacén canónico de partidos.

Un único CSV con todos los partidos conocidos, vengan de donde vengan. Es el
sustituto de leer directamente los volcados de Kaggle: refrescar consiste en
pedir a una fuente los partidos nuevos y fundirlos aquí por `match_id`.

Es pequeño (unos 2 MB para cinco ligas y cinco temporadas), así que se versiona
en git: cualquiera clona el repositorio y todo funciona sin descargar nada.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import MATCHES_STORE
from .sources.base import CANONICAL_COLUMNS, merge_sources, normalize

log = logging.getLogger(__name__)


def load_store(path: Path = MATCHES_STORE) -> pd.DataFrame:
    """Lee el almacén. Devuelve un DataFrame vacío si aún no existe."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = pd.read_csv(path)
    return normalize(df)


def save_store(df: pd.DataFrame, path: Path = MATCHES_STORE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize(df).to_csv(path, index=False)
    return path


def upsert(new: pd.DataFrame, path: Path = MATCHES_STORE) -> dict:
    """Funde partidos nuevos en el almacén y describe qué cambió.

    Un partido que pasa de anunciado a jugado sustituye a su versión anterior:
    `merge_sources` se queda con la última aparición de cada `match_id`.
    """
    actual = load_store(path)
    if new is None or new.empty:
        return {"antes": len(actual), "despues": len(actual), "nuevos": 0, "actualizados": 0}

    new = normalize(new, require_results=False)
    ids_actuales = set(actual["match_id"]) if not actual.empty else set()
    nuevos = int((~new["match_id"].isin(ids_actuales)).sum())
    actualizados = int(new["match_id"].isin(ids_actuales).sum())

    fundido = merge_sources(actual, new)
    save_store(fundido, path)
    return {
        "antes": len(actual),
        "despues": len(fundido),
        "nuevos": nuevos,
        "actualizados": actualizados,
        "hasta": str(fundido["date"].max().date()) if not fundido.empty else None,
    }


def played(df: pd.DataFrame | None = None, path: Path = MATCHES_STORE) -> pd.DataFrame:
    """Sólo los partidos con resultado conocido."""
    df = load_store(path) if df is None else df
    if df.empty:
        return df
    return df[df["home_goals"].notna() & df["away_goals"].notna()].reset_index(drop=True)


def upcoming(df: pd.DataFrame | None = None, path: Path = MATCHES_STORE) -> pd.DataFrame:
    """Partidos anunciados sin resultado todavía."""
    df = load_store(path) if df is None else df
    if df.empty:
        return df
    return df[df["home_goals"].isna() | df["away_goals"].isna()].reset_index(drop=True)
