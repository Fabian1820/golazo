"""Carga del historial listo para modelar.

Regla central del proyecto: las columnas se dividen explícitamente en

* PRE-partido  -> conocidas antes del pitido inicial. Usables como feature.
* POST-partido -> sólo existen al terminar. Usables como *objetivo*, o como
  insumo de features de partidos ANTERIORES, nunca del partido en curso.

Mezclar ambas es la fuga de datos que invalidaba el modelo original.

El Elo se calcula aquí, sobre el historial ya ordenado, en vez de leerse de un
fichero: así los partidos nuevos que entren por una fuente viva reciben su
valoración por el mismo camino que los históricos.
"""
from __future__ import annotations

import pandas as pd

from .config import MATCHES_STORE
from .elo import attach_elo
from .store import load_store, played

# Estadísticas que sólo se conocen al final del partido. Se conservan para
# construir medias móviles de partidos ya jugados, jamás del actual.
POST_MATCH_COLS = [
    "home_goals", "away_goals",
    "home_xg", "away_xg",
    "home_shots", "away_shots",
    "home_sot", "away_sot",
    "home_deep", "away_deep",
    "home_ppda", "away_ppda",
]

# Conocidas antes del partido.
PRE_MATCH_COLS = ["match_id", "date", "season", "league", "home", "away", "elo_h", "elo_a"]


def load_matches(path=MATCHES_STORE) -> pd.DataFrame:
    """Historial de partidos jugados, ordenado y con Elo pre-partido.

    El Elo es previo por construcción: `attach_elo` recorre en orden y lee la
    valoración de cada equipo ANTES de incorporar el resultado del partido.
    Verificado en tests/test_data.py y tests/test_elo.py.
    """
    df = played(load_store(path))
    if df.empty:
        raise FileNotFoundError(
            f"El almacén de partidos está vacío o no existe ({path}). "
            "Ejecuta: python scripts/migrate_store.py  (o: golazo fetch)")

    df = attach_elo(df)

    df["result"] = pd.Series("D", index=df.index).where(
        df["home_goals"] == df["away_goals"],
        pd.Series("H", index=df.index).where(df["home_goals"] > df["away_goals"], "A"),
    )

    keep = PRE_MATCH_COLS + POST_MATCH_COLS + ["result"]
    return df[keep].sort_values("date", kind="mergesort").reset_index(drop=True)
