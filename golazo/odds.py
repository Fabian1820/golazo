"""Cuotas de cierre del mercado, como referencia externa.

Un modelo que bate a la tasa base por un 12% suena bien hasta que se compara con
algo que de verdad intente predecir. Las casas de apuestas lo hacen con mucho más
dinero y mucha más información, así que su cuota de cierre —ya descontado el
margen— es el listón honesto.

No se guardan en el almacén de partidos porque no son un hecho del partido, sino
la predicción de otro. Viven aparte y se cruzan cuando hacen falta.

Una advertencia sobre referencias falsas
----------------------------------------
Understat publica un campo `forecast` que parece un pronóstico y no lo es:
correlaciona 0.942 con el xG del propio partido, o sea que se calcula DESPUÉS.
Da un RPS de 0.1621, mejor que cualquier casa de apuestas, precisamente porque
ya conoce el resultado. Usarlo como referencia sería repetir la fuga de datos
que este proyecto existe para corregir.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from .config import ODDS_STORE
from .sources.footballdata import LIGAS_PRIMERA, FootballDataSource

log = logging.getLogger(__name__)

CLAVE = ["date", "home", "away"]
COLUMNAS = CLAVE + ["league", "market_H", "market_D", "market_A", "odds_source"]


def load_odds(path: Path = ODDS_STORE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNAS)
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def save_odds(df: pd.DataFrame, path: Path = ODDS_STORE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(subset=CLAVE, keep="last").sort_values("date", kind="mergesort")
    df[COLUMNAS].to_csv(path, index=False)
    return path


def refresh_odds(seasons: Sequence[int], leagues: Iterable[str] = LIGAS_PRIMERA,
                 path: Path = ODDS_STORE, use_cache: bool = False) -> dict:
    """Descarga cuotas y las funde con las ya guardadas."""
    from .config import CACHE_DIR

    fuente = FootballDataSource(cache_dir=CACHE_DIR if use_cache else None)
    nuevas = fuente.odds(leagues=list(leagues), seasons=list(seasons))
    actuales = load_odds(path)

    if nuevas.empty:
        return {"antes": len(actuales), "despues": len(actuales), "nuevas": 0}

    fundidas = pd.concat([actuales, nuevas], ignore_index=True) if not actuales.empty else nuevas
    fundidas = fundidas.drop_duplicates(subset=CLAVE, keep="last")
    save_odds(fundidas, path)
    return {
        "antes": len(actuales),
        "despues": len(fundidas),
        "nuevas": len(fundidas) - len(actuales),
        "fuentes": nuevas["odds_source"].value_counts().to_dict(),
    }


def attach(preds: pd.DataFrame, path: Path = ODDS_STORE) -> pd.DataFrame:
    """Añade `market_H/D/A` a un DataFrame de predicciones del backtest.

    El cruce es por (día, local, visitante). Las filas sin cuota conocida
    quedan con NaN: es responsabilidad de quien compare restringirse al
    subconjunto común, no rellenar huecos.
    """
    cuotas = load_odds(path)
    if cuotas.empty or preds.empty:
        out = preds.copy()
        for c in ("market_H", "market_D", "market_A"):
            out[c] = float("nan")
        return out

    izq = preds.copy()
    izq["_dia"] = pd.to_datetime(izq["date"]).dt.normalize()
    der = cuotas.copy()
    der["_dia"] = pd.to_datetime(der["date"]).dt.normalize()

    return izq.merge(
        der[["_dia", "home", "away", "market_H", "market_D", "market_A", "odds_source"]],
        on=["_dia", "home", "away"], how="left",
    ).drop(columns="_dia")


def coverage(preds_con_cuotas: pd.DataFrame) -> dict:
    n = len(preds_con_cuotas)
    con = int(preds_con_cuotas["market_H"].notna().sum())
    return {"total": n, "con_cuota": con, "cobertura": con / n if n else 0.0}
