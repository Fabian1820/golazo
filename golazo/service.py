"""Servicio de predicción para partidos no jugados.

Carga un artefacto entrenado una sola vez y reconstruye el estado del historial
en memoria. A partir de ahí cada predicción es O(1), sin reentrenar y sin releer
los CSV.

Usa `FeatureBuilder.emit`, el mismo método que se usa para entrenar. No hay una
segunda implementación de las features que pueda divergir.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from . import markets
from .artifacts import ModelArtifact
from .config import OUTCOMES
from .data import load_matches
from .features import FeatureBuilder

# Por debajo de este historial la predicción no es publicable.
#
# No es un umbral cosmético. Los feeds de liga de Understat incluyen partidos
# de copa, que arrastran equipos de divisiones inferiores con dos o tres
# encuentros registrados. Dixon-Coles asigna a un equipo desconocido el
# promedio de su liga, así que un rival de segunda pasaba a valorarse como un
# equipo medio de primera: en el primer pronóstico real, Hull salía al 75%
# frente al Aston Villa.
#
# No hay prior que arregle no tener datos. Lo correcto es no publicar.
MIN_HISTORY = 10

# Por encima del mínimo pero aún escaso: se predice, con aviso.
THIN_HISTORY = 25


class UnknownTeamError(ValueError):
    """El equipo no aparece en el historial con el que se entrenó el modelo."""


class InsufficientHistoryError(UnknownTeamError):
    """El equipo existe pero tiene demasiados pocos partidos para predecirlo."""


@dataclass
class Prediction:
    home: str
    away: str
    league: str
    kickoff: str
    probabilities: dict[str, float]
    model_version: str
    trained_through: str
    markets: dict | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def most_likely(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    def to_dict(self) -> dict:
        d = {
            "home": self.home,
            "away": self.away,
            "league": self.league,
            "kickoff": self.kickoff,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "most_likely": self.most_likely,
            "model_version": self.model_version,
            "trained_through": self.trained_through,
        }
        if self.markets:
            d["markets"] = _round(self.markets)
        if self.warnings:
            d["warnings"] = self.warnings
        return d


def _round(obj, nd: int = 4):
    """Redondea recursivamente para que el JSON no exponga ruido de coma flotante."""
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, float):
        return round(obj, nd)
    return obj


class PredictionService:
    """Punto de entrada único para predecir partidos futuros."""

    def __init__(self, artifact: ModelArtifact, history: pd.DataFrame,
                 min_history: int = MIN_HISTORY):
        self.artifact = artifact
        self.history = history
        self.min_history = min_history
        self.builder = FeatureBuilder.from_history(history)
        self._leagues = sorted(history["league"].unique().tolist())
        self._teams_by_league = {
            lg: sorted(set(sub["home"]) | set(sub["away"]))
            for lg, sub in history.groupby("league")
        }
        self._match_counts = pd.concat([history["home"], history["away"]]).value_counts().to_dict()

    @classmethod
    def load(cls, version: str = "latest") -> PredictionService:
        return cls(ModelArtifact.load(version), load_matches())

    def _history_size(self, team: str) -> int:
        """Partidos totales del equipo en el historial, sin tope de ventana."""
        return int(self._match_counts.get(team, 0))

    # -- catálogo ---------------------------------------------------------

    @property
    def leagues(self) -> list[str]:
        return self._leagues

    def teams(self, league: str) -> list[str]:
        if league not in self._teams_by_league:
            raise ValueError(f"Liga desconocida: {league}. Disponibles: {self._leagues}")
        return self._teams_by_league[league]

    def league_of(self, team: str) -> str | None:
        for lg, teams in self._teams_by_league.items():
            if team in teams:
                return lg
        return None

    # -- predicción -------------------------------------------------------

    def _fixture_row(self, home: str, away: str, league: str, kickoff: pd.Timestamp) -> pd.DataFrame:
        elo_h, elo_a = self.builder.elo_for(home), self.builder.elo_for(away)
        if elo_h is None:
            raise UnknownTeamError(f"Sin historial para el equipo local '{home}'")
        if elo_a is None:
            raise UnknownTeamError(f"Sin historial para el equipo visitante '{away}'")

        feat = self.builder.emit(home, away, kickoff, elo_h, elo_a)
        row = {"league": league, "home": home, "away": away, "date": kickoff,
               "elo_h": elo_h, "elo_a": elo_a, **feat}
        return pd.DataFrame([row])

    def predict(self, home: str, away: str, league: str | None = None,
                kickoff=None) -> Prediction:
        """Probabilidades 1X2 para un enfrentamiento que aún no se ha jugado."""
        if home == away:
            raise ValueError("Un equipo no puede jugar contra sí mismo")

        league = league or self.league_of(home)
        if league is None:
            raise UnknownTeamError(f"Equipo desconocido: '{home}'")

        # Por defecto, el día siguiente al último partido conocido.
        kickoff = pd.Timestamp(kickoff) if kickoff is not None else \
            self.builder.last_date + pd.Timedelta(days=1)

        # Guardia de historial ANTES de calcular nada: si no se puede respaldar,
        # no se publica.
        for etiqueta, team in (("local", home), ("visitante", away)):
            n = self._history_size(team)
            if n < self.min_history:
                raise InsufficientHistoryError(
                    f"El {etiqueta} '{team}' sólo tiene {n} partidos en el historial "
                    f"(mínimo {self.min_history}). Suele ser un rival de copa de otra "
                    f"división: sin datos suficientes no se emite pronóstico")

        row = self._fixture_row(home, away, league, kickoff)
        probs = self.artifact.model.predict_proba(row)[0]

        warnings = []
        for etiqueta, team in (("local", home), ("visitante", away)):
            n = self._history_size(team)
            if n < THIN_HISTORY:
                warnings.append(
                    f"El {etiqueta} '{team}' tiene sólo {n} partidos de historial; "
                    f"la predicción es menos fiable de lo habitual")
        rezago = (kickoff - self.builder.last_date).days
        if rezago > 60:
            warnings.append(
                f"El modelo se entrenó con datos hasta {self.builder.last_date:%Y-%m-%d}, "
                f"{rezago} días antes de este partido")

        # Si el modelo expone la distribución conjunta de marcadores, se derivan
        # de ella todos los mercados. Coherentes por construcción.
        mercados = None
        fits = getattr(self.artifact.model, "fits", None)
        if fits and league in fits:
            mat = fits[league].scoreline_matrix(home, away)
            ok, motivo = markets.coherence_check(mat)
            if not ok:
                raise RuntimeError(f"Distribución de marcadores inválida: {motivo}")
            mercados = markets.summarize(mat)

        return Prediction(
            home=home, away=away, league=league,
            kickoff=kickoff.isoformat(),
            probabilities={o: float(p) for o, p in zip(OUTCOMES, probs)},
            model_version=self.artifact.metadata.version_id,
            trained_through=self.artifact.metadata.train_end,
            markets=mercados,
            warnings=warnings,
        )

    def predict_many(self, fixtures: Sequence[dict]) -> list[Prediction]:
        """`fixtures` son dicts con claves home, away y opcionalmente league/kickoff."""
        return [self.predict(**f) for f in fixtures]
