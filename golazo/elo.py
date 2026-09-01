"""Elo calculado por nosotros, incrementalmente desde los resultados.

Por qué no usar `ClubElo.csv` directamente
------------------------------------------
Es un fichero estático que termina en 2023 y cuya API pública responde 502.
Un modelo que depende de él no puede predecir un partido de mañana.

Calcularlo aquí resuelve tres cosas a la vez: no hay dependencia externa que se
caiga, los equipos recién ascendidos reciben una valoración desde el primer
partido, y el histórico y los datos nuevos pasan por **el mismo código** — el
mismo principio que aplica `FeatureBuilder` a las features.

Formulación
-----------
Elo estándar de fútbol con dos ajustes habituales:

* ventaja de campo sumada al rating del local antes de calcular la expectativa;
* multiplicador por diferencia de goles (`G`), para que un 5-0 mueva más que
  un 1-0. Es la fórmula de World Football Elo Ratings.

    E_local = 1 / (1 + 10^(-(R_local + HFA - R_visitante) / 400))
    R' = R + K · G · (S - E)

Los parámetros son los valores convencionales de la literatura, no ajustados
sobre estos datos: ajustarlos contra el mismo backtest que después los evalúa
sería hacer trampa.
"""
from __future__ import annotations

import pandas as pd

BASE_RATING = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 65.0
# Un ascendido no vale lo mismo que la media de la categoría. La penalización
# es conservadora y se corrige sola en pocas jornadas.
PROMOTED_PENALTY = 40.0


def expected_score(rating_home: float, rating_away: float,
                   home_advantage: float = HOME_ADVANTAGE) -> float:
    """Probabilidad esperada (en puntos Elo) de que puntúe el local."""
    return 1.0 / (1.0 + 10 ** (-(rating_home + home_advantage - rating_away) / 400.0))


def goal_difference_multiplier(goal_diff: int) -> float:
    """`G`: cuánto más pesa una victoria según lo abultada que sea."""
    d = abs(int(goal_diff))
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


class EloRatings:
    """Valoraciones Elo que avanzan partido a partido.

    Igual que `FeatureBuilder`: `rating` consulta sin mutar, `update` avanza.
    El orden correcto es siempre consultar-antes-de-actualizar, que es lo que
    mantiene el Elo estrictamente pre-partido.
    """

    def __init__(self, k: float = K_FACTOR, home_advantage: float = HOME_ADVANTAGE,
                 base: float = BASE_RATING, promoted_penalty: float = PROMOTED_PENALTY):
        self.k = k
        self.home_advantage = home_advantage
        self.base = base
        self.promoted_penalty = promoted_penalty
        self._ratings: dict[str, float] = {}
        self._league_of: dict[str, str] = {}

    # -- consulta (pura) --------------------------------------------------

    def rating(self, team: str, league: str | None = None) -> float:
        """Valoración actual. Un equipo nunca visto entra con la línea base."""
        if team in self._ratings:
            return self._ratings[team]
        return self._initial_rating(league)

    def _initial_rating(self, league: str | None) -> float:
        """Un equipo nuevo entra por debajo de la media de su liga.

        Usar la media de la liga en vez de una constante evita que un ascendido
        a la Premier arranque igual que uno a la Ligue 1.
        """
        if league is None:
            return self.base
        pares = [r for t, r in self._ratings.items() if self._league_of.get(t) == league]
        if not pares:
            # Liga todavía sin equipos: no hay nadie respecto de quien ascender.
            return self.base
        return sum(pares) / len(pares) - self.promoted_penalty

    def known(self, team: str) -> bool:
        return team in self._ratings

    def as_dict(self) -> dict[str, float]:
        return dict(self._ratings)

    # -- avance del estado ------------------------------------------------

    def update(self, home: str, away: str, home_goals: int, away_goals: int,
               league: str | None = None) -> None:
        """Incorpora un resultado. Debe llamarse DESPUÉS de leer los ratings."""
        rh = self.rating(home, league)
        ra = self.rating(away, league)

        if home_goals > away_goals:
            score = 1.0
        elif home_goals < away_goals:
            score = 0.0
        else:
            score = 0.5

        esperado = expected_score(rh, ra, self.home_advantage)
        ajuste = self.k * goal_difference_multiplier(home_goals - away_goals) * (score - esperado)

        # Suma cero: lo que gana uno lo pierde el otro.
        self._ratings[home] = rh + ajuste
        self._ratings[away] = ra - ajuste
        if league is not None:
            self._league_of[home] = league
            self._league_of[away] = league


def attach_elo(df: pd.DataFrame, k: float = K_FACTOR,
               home_advantage: float = HOME_ADVANTAGE) -> pd.DataFrame:
    """Añade `elo_h` y `elo_a` PRE-partido a un historial ordenado.

    Recorre en orden cronológico leyendo antes de actualizar, así que la
    valoración de cada partido no puede contener su propio resultado.
    """
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    elo = EloRatings(k=k, home_advantage=home_advantage)

    eh, ea = [], []
    for r in df.itertuples(index=False):
        liga = getattr(r, "league", None)
        eh.append(elo.rating(r.home, liga))
        ea.append(elo.rating(r.away, liga))
        elo.update(r.home, r.away, r.home_goals, r.away_goals, liga)

    out = df.copy()
    out["elo_h"] = eh
    out["elo_a"] = ea
    return out


def ratings_after(df: pd.DataFrame) -> EloRatings:
    """Estado del Elo tras consumir todo un historial."""
    elo = EloRatings()
    for r in df.sort_values("date", kind="mergesort").itertuples(index=False):
        elo.update(r.home, r.away, r.home_goals, r.away_goals, getattr(r, "league", None))
    return elo
