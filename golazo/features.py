"""Construcción de features estrictamente PRE-partido.

El recorrido es cronológico y cada partido sólo ve el historial *anterior*.
Esa es la diferencia con el modelo original, que alimentaba el resultado con
estadísticas del propio partido a predecir.

Una sola fuente de verdad
-------------------------
`FeatureBuilder` mantiene el estado (medias móviles, último Elo, descanso) y
emite features sin mutarlo. Tanto el entrenamiento como el servicio en
producción pasan por el MISMO código:

    entrenamiento -> build_features(df)          (bucle sobre emit + ingest)
    producción    -> builder.emit(fixture)        (mismo emit, sin ingest)

Tener dos implementaciones de las features es exactamente cómo aparece el
desajuste entre entrenamiento y producción que invalidaba el modelo original.
Aquí es imposible por construcción.

Los equipos sin historial suficiente (recién ascendidos, inicio del dataset)
salen con NaN a propósito: se prefiere que el modelo lo sepa a imputar un
valor inventado. HistGradientBoosting trata los NaN de forma nativa.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

WINDOWS = (5, 10)
MAXLEN = max(WINDOWS)

# Estadísticas que se acumulan por equipo, desde su propia perspectiva.
#
# Tiros y tiros a puerta se excluyeron a propósito: la fuente viva (Understat)
# no los publica, y una feature que existe al entrenar pero no al predecir es
# justo el desajuste que invalidaba el modelo original. El coste de quitarlas
# se midió en el backtest y es nulo (ver reports/README.md).
_STATS = ("goals_for", "goals_against", "xg_for", "xg_against", "deep_for", "ppda")


def _blank(prefix: str) -> dict:
    out = {}
    for w in WINDOWS:
        for s in _STATS:
            out[f"{prefix}_{s}_l{w}"] = np.nan
    for w in WINDOWS:
        out[f"{prefix}_venue_xg_for_l{w}"] = np.nan
        out[f"{prefix}_venue_xg_against_l{w}"] = np.nan
    out[f"{prefix}_rest_days"] = np.nan
    out[f"{prefix}_hist_depth"] = 0
    return out


def _rolling(hist: deque, prefix: str) -> dict:
    out = {}
    for w in WINDOWS:
        recent = list(hist)[-w:]
        for s in _STATS:
            out[f"{prefix}_{s}_l{w}"] = float(np.mean([r[s] for r in recent])) if recent else np.nan
    return out


def _rolling_venue(hist: deque, prefix: str) -> dict:
    out = {}
    for w in WINDOWS:
        recent = list(hist)[-w:]
        out[f"{prefix}_venue_xg_for_l{w}"] = float(np.mean([r["xg_for"] for r in recent])) if recent else np.nan
        out[f"{prefix}_venue_xg_against_l{w}"] = float(np.mean([r["xg_against"] for r in recent])) if recent else np.nan
    return out


class FeatureBuilder:
    """Estado incremental del historial de todos los equipos.

    `emit` es puro: no toca el estado. `ingest` lo avanza. El orden correcto es
    siempre emit-luego-ingest, que es lo que garantiza la causalidad.
    """

    def __init__(self):
        self.hist: dict = defaultdict(lambda: deque(maxlen=MAXLEN))
        self.hist_home: dict = defaultdict(lambda: deque(maxlen=MAXLEN))
        self.hist_away: dict = defaultdict(lambda: deque(maxlen=MAXLEN))
        self.last_played: dict = {}
        self.last_elo: dict = {}
        self.last_date: pd.Timestamp | None = None

    # -- emisión (pura) ---------------------------------------------------

    def emit(self, home: str, away: str, date, elo_h: float, elo_a: float) -> dict:
        """Features pre-partido para un enfrentamiento, jugado o no."""
        date = pd.Timestamp(date)
        feat = {}
        for prefix, team, venue_hist in (("h", home, self.hist_home), ("a", away, self.hist_away)):
            feat.update(_blank(prefix))
            h = self.hist[team]
            if h:
                feat.update(_rolling(h, prefix))
                feat[f"{prefix}_hist_depth"] = len(h)
            vh = venue_hist[team]
            if vh:
                feat.update(_rolling_venue(vh, prefix))
            if team in self.last_played:
                feat[f"{prefix}_rest_days"] = (date - self.last_played[team]).total_seconds() / 86400.0

        # elo_h / elo_a / league ya vienen del loader; aquí sólo los derivados.
        feat["elo_diff"] = elo_h - elo_a
        for w in WINDOWS:
            feat[f"xg_diff_l{w}"] = feat[f"h_xg_for_l{w}"] - feat[f"a_xg_for_l{w}"]
            feat[f"xga_diff_l{w}"] = feat[f"h_xg_against_l{w}"] - feat[f"a_xg_against_l{w}"]
        return feat

    # -- avance del estado ------------------------------------------------

    def ingest(self, row) -> None:
        """Incorpora un partido YA JUGADO al historial."""
        home_rec = {
            "goals_for": row.home_goals, "goals_against": row.away_goals,
            "xg_for": row.home_xg, "xg_against": row.away_xg,
            "deep_for": row.home_deep, "ppda": row.home_ppda,
        }
        away_rec = {
            "goals_for": row.away_goals, "goals_against": row.home_goals,
            "xg_for": row.away_xg, "xg_against": row.home_xg,
            "deep_for": row.away_deep, "ppda": row.away_ppda,
        }
        self.hist[row.home].append(home_rec)
        self.hist[row.away].append(away_rec)
        self.hist_home[row.home].append(home_rec)
        self.hist_away[row.away].append(away_rec)
        self.last_played[row.home] = row.date
        self.last_played[row.away] = row.date
        self.last_elo[row.home] = row.elo_h
        self.last_elo[row.away] = row.elo_a
        self.last_date = row.date

    # -- utilidades para servir -------------------------------------------

    def elo_for(self, team: str) -> float | None:
        """Último Elo conocido del equipo. None si nunca se ha visto."""
        return self.last_elo.get(team)

    def known_teams(self) -> list[str]:
        return sorted(self.last_elo)

    @classmethod
    def from_history(cls, df: pd.DataFrame) -> FeatureBuilder:
        """Estado tras consumir todo un historial de partidos jugados."""
        b = cls()
        for row in df.sort_values("date", kind="mergesort").itertuples(index=False):
            b.ingest(row)
        return b


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve `df` con las columnas de feature añadidas, mismo orden de filas."""
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    builder = FeatureBuilder()
    rows: list[dict] = []
    for r in df.itertuples(index=False):
        rows.append(builder.emit(r.home, r.away, r.date, r.elo_h, r.elo_a))
        # --- Recién ahora se incorpora el resultado, para el SIGUIENTE partido ---
        builder.ingest(r)

    feats = pd.DataFrame(rows, index=df.index)
    dupes = set(feats.columns) & set(df.columns)
    if dupes:
        raise ValueError(f"columnas de feature duplicadas con el loader: {sorted(dupes)}")
    return pd.concat([df, feats], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columnas de entrada del modelo. Excluye explícitamente todo lo post-partido."""
    from .data import POST_MATCH_COLS

    banned = set(POST_MATCH_COLS) | {"match_id", "date", "season", "home", "away", "result"}
    return [c for c in df.columns if c not in banned]
