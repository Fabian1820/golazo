"""Contrato común de las fuentes de datos.

El esquema canónico es la frontera del sistema: una fuente nueva sólo tiene que
producir estas columnas y todo lo demás —features, modelos, backtest, servicio—
funciona sin cambios.

El Elo NO forma parte del esquema. Se calcula en `golazo.elo` a partir de los
resultados, para que histórico y datos nuevos sigan el mismo camino.
"""
from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)


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

    out = _sanear_tiros(out)
    return out[CANONICAL_COLUMNS].sort_values("date", kind="mergesort").reset_index(drop=True)


def _sanear_tiros(df: pd.DataFrame) -> pd.DataFrame:
    """Anula los pares de tiros imposibles en lugar de propagarlos.

    football-data.co.uk publica algún partido con más tiros a puerta que tiros
    totales (Burnley 2-7 contra el Swansea, por ejemplo). Es un error de
    transcripción del origen, no un dato raro.

    Se anulan ambos valores del lado afectado: no se puede saber cuál de los dos
    está mal, y quedarse con uno sería inventar. Al ser columnas opcionales, un
    NaN es una respuesta válida; un 2 y un 7 juntos no lo son.
    """
    for lado in ("home", "away"):
        tiros, puerta = f"{lado}_shots", f"{lado}_sot"
        if tiros not in df.columns or puerta not in df.columns:
            continue
        mal = df[puerta] > df[tiros]
        if mal.any():
            log.warning("%d partidos con %s > %s: se anulan ambos", int(mal.sum()), puerta, tiros)
            df.loc[mal, [tiros, puerta]] = pd.NA
    return df


def merge_sources(*frames: pd.DataFrame | None) -> pd.DataFrame:
    """Une varias fuentes resolviendo duplicados en dos pasos.

    1. Por `match_id`, quedándose con la última aparición: al refrescar, los
       datos nuevos sustituyen a los viejos del mismo partido (por ejemplo
       cuando un partido pasa de anunciado a jugado).

    2. Por la **clave natural** (fecha, local, visitante), quedándose con el
       registro más completo.

    El segundo paso no es redundante. El prefijo de fuente en `match_id` evita
    que dos fuentes se pisen, pero por eso mismo no detecta que describen el
    mismo partido: el tramo histórico de Kaggle y el refresco de Understat
    comparten origen, así que un partido aparecía a la vez como `kaggle:19342`
    y `understat:19342` y ambos sobrevivían. Se colaron 3.652 duplicados.

    Se conserva el registro con más campos informados —el tramo de Kaggle trae
    tiros, el de Understat no— y a igualdad, el último.
    """
    presentes = [f for f in frames if f is not None and not f.empty]
    if not presentes:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    todo = pd.concat(presentes, ignore_index=True)
    todo = todo.drop_duplicates(subset=["match_id"], keep="last")

    if len(todo) > 1:
        # Orden estable: primero los menos completos, para que `keep="last"`
        # conserve el más rico de cada partido.
        todo = todo.assign(_completitud=todo[OPTIONAL_COLUMNS].notna().sum(axis=1))
        todo = (todo.sort_values("_completitud", kind="mergesort")
                    .drop_duplicates(subset=["date", "home", "away"], keep="last")
                    .drop(columns="_completitud"))

    return todo.sort_values("date", kind="mergesort").reset_index(drop=True)
