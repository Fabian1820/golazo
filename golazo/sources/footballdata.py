"""Segunda fuente: football-data.co.uk.

Aporta tres cosas que Understat no da:

1. **Segundas divisiones.** Championship, LaLiga 2, 2. Bundesliga, Serie B y
   Ligue 2. Ahí están los equipos que aparecen en los feeds de liga por
   partidos de copa y que hoy se rechazan por falta de historial.
2. **Cuotas de cierre.** Bet365, Pinnacle y media del mercado. Es la única
   referencia externa honesta para saber si el modelo vale algo.
3. **Tiros, córners y tarjetas**, que el endpoint de liga de Understat omite.

Nombres de equipo
-----------------
Los dos sitios nombran a los equipos de forma distinta y la tabla de
equivalencias es **explícita a propósito**. Se probó con emparejamiento difuso
y produjo tres errores que habrían corrompido los datos en silencio:

    'Paris SG'  -> 'Paris FC'    (clubes distintos, ambos en el almacén)
    'West Brom' -> 'West Ham'    (clubes distintos)
    'Spal'      -> 'Spezia'      (clubes distintos)

Por eso un nombre no reconocido levanta `UnknownTeamNameError` en lugar de
adivinar. Es preferible que falle la descarga a que se mezclen dos equipos.

Los alias se aplican **también en segunda división**. Al principio no se hacía,
con el razonamiento de que allí no hay contraparte de Understat con la que
contrastar. Era un error: los equipos ascienden y descienden, así que el mismo
club aparecía como 'Santander' en La Liga 2 y como 'Racing Santander' cuando
Understat lo recogía en un partido de copa. Quedaban como dos equipos distintos
y ninguno acumulaba historial suficiente.
"""
from __future__ import annotations

import csv
import io
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .base import normalize

log = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"
USER_AGENT = "Mozilla/5.0 (compatible; golazo/0.1; +https://github.com/Fabian1820/golazo)"
REQUEST_DELAY = 1.5
TIMEOUT = 30
RETRIES = 3

# Código del sitio -> nombre de liga en nuestro almacén.
# Las primeras divisiones se descargan sólo para cuotas y verificación: los
# partidos vienen de Understat, que además trae xG.
PRIMERA = {
    "E0": "EPL",
    "SP1": "La liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

SEGUNDA = {
    "E1": "Championship",
    "SP2": "La liga 2",
    "D2": "Bundesliga 2",
    "I2": "Serie B",
    "F2": "Ligue 2",
}

CODIGOS = {**PRIMERA, **SEGUNDA}

# Tuplas fijas para usarlas como valores por defecto sin recalcularlas.
LIGAS_PRIMERA = tuple(PRIMERA.values())
LIGAS_SEGUNDA = tuple(SEGUNDA.values())

# Equivalencias revisadas a mano, aplicadas en todas las divisiones.
#
# Cuidado con ampliarla por similitud de texto: al generarla automáticamente
# propuso 'Portsmouth'->'Bournemouth', 'Sheffield Weds'->'Sheffield United',
# 'Lecco'->'Lecce' y 'Pau FC'->'Paris FC', todos clubes distintos. Y los
# filiales ('Celta B', 'Sociedad B', 'Villarreal B') son equipos aparte, no
# variantes de nombre.
ALIAS = {
    # Inglaterra
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "West Brom": "West Bromwich Albion",
    "Sheffield United": "Sheffield United",
    # España
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atletico Madrid",
    "Betis": "Real Betis",
    "Celta": "Celta Vigo",
    "Espanol": "Espanyol",
    "Huesca": "SD Huesca",
    "Oviedo": "Real Oviedo",
    "Santander": "Racing Santander",
    "La Coruna": "Deportivo La Coruna",
    "Sociedad": "Real Sociedad",
    "Valladolid": "Real Valladolid",
    "Vallecano": "Rayo Vallecano",
    # Alemania
    "Bielefeld": "Arminia Bielefeld",
    "Dortmund": "Borussia Dortmund",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "FC Cologne",
    "Fortuna Dusseldorf": "Fortuna Duesseldorf",
    "Greuther Furth": "Greuther Fuerth",
    "Hamburg": "Hamburger SV",
    "Hannover": "Hannover 96",
    "Heidenheim": "FC Heidenheim",
    "Hertha": "Hertha Berlin",
    "Leverkusen": "Bayer Leverkusen",
    "M'gladbach": "Borussia M.Gladbach",
    "Mainz": "Mainz 05",
    "Nurnberg": "Nuernberg",
    "RB Leipzig": "RasenBallsport Leipzig",
    "St Pauli": "St. Pauli",
    "Stuttgart": "VfB Stuttgart",
    # Italia
    "Milan": "AC Milan",
    "Parma": "Parma Calcio 1913",
    "Spal": "SPAL 2013",
    "Verona": "Verona",
    # Francia
    "Paris SG": "Paris Saint Germain",
    "St Etienne": "Saint-Etienne",
    "Clermont": "Clermont Foot",
}

# Cuotas: cierre de Bet365 primero, media del mercado como respaldo.
COLUMNAS_CUOTAS = [
    ("B365CH", "B365CD", "B365CA"),   # cierre Bet365
    ("AvgCH", "AvgCD", "AvgCA"),      # media del mercado al cierre
    ("PSCH", "PSCD", "PSCA"),         # cierre Pinnacle
    ("B365H", "B365D", "B365A"),      # apertura, último recurso
]


class FootballDataError(RuntimeError):
    pass


class UnknownTeamNameError(FootballDataError):
    """Un nombre de equipo sin equivalencia conocida. Nunca se adivina."""


def season_code(season: int) -> str:
    """2024 (convención Understat, temporada 2024/25) -> '2425'."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


class FootballDataSource:
    """Cliente de los CSV por liga y temporada de football-data.co.uk."""

    name = "footballdata"

    def __init__(self, cache_dir: Path | None = None, delay: float = REQUEST_DELAY):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.delay = delay
        self._last_request = 0.0

    # -- transporte -------------------------------------------------------

    def _throttle(self) -> None:
        espera = self.delay - (time.monotonic() - self._last_request)
        if espera > 0:
            time.sleep(espera)
        self._last_request = time.monotonic()

    def _get(self, code: str, season: int) -> list[dict]:
        sc = season_code(season)
        cache = self.cache_dir / f"{code}_{sc}.csv" if self.cache_dir else None
        if cache and cache.exists():
            texto = cache.read_text(encoding="utf-8", errors="ignore")
            return list(csv.DictReader(io.StringIO(texto)))

        url = f"{BASE_URL}/{sc}/{code}.csv"
        ultimo = None
        for intento in range(1, RETRIES + 1):
            self._throttle()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    texto = resp.read().decode("utf-8", "ignore")
                break
            except (urllib.error.URLError, OSError) as exc:
                ultimo = exc
                log.warning("football-data %s/%s intento %d/%d: %s", code, sc, intento, RETRIES, exc)
                time.sleep(2 ** intento)
        else:
            raise FootballDataError(f"No se pudo obtener {code}/{sc}: {ultimo}") from ultimo

        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(texto, encoding="utf-8")
        return list(csv.DictReader(io.StringIO(texto)))

    # -- normalización ----------------------------------------------------

    @staticmethod
    def _team(nombre: str, *, traducir: bool) -> str:
        """Nombre canónico. Falla si no hay equivalencia conocida."""
        nombre = (nombre or "").strip()
        if not traducir:
            return nombre
        return ALIAS.get(nombre, nombre)

    @staticmethod
    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _cuotas(cls, fila: dict):
        """Probabilidades implícitas del mercado, con el margen repartido."""
        for h, d, a in COLUMNAS_CUOTAS:
            vals = [cls._num(fila.get(h)), cls._num(fila.get(d)), cls._num(fila.get(a))]
            if all(v and v > 1.0 for v in vals):
                inv = [1.0 / v for v in vals]
                total = sum(inv)
                return [x / total for x in inv], (h[:-1] if h.endswith("H") else h)
        return None, None

    def _to_frame(self, filas: list[dict], code: str, season: int, *, traducir: bool) -> pd.DataFrame:
        liga = CODIGOS[code]
        out = []
        for r in filas:
            if not r.get("HomeTeam") or not r.get("Date"):
                continue
            fecha = pd.to_datetime(r["Date"], dayfirst=True, errors="coerce")
            if pd.isna(fecha):
                continue
            hora = (r.get("Time") or "").strip()
            if hora:
                fecha = pd.to_datetime(f"{fecha:%Y-%m-%d} {hora}", errors="coerce") or fecha

            h = self._team(r["HomeTeam"], traducir=traducir)
            a = self._team(r["AwayTeam"], traducir=traducir)
            out.append({
                "match_id": f"fd:{code}:{fecha:%Y%m%d}:{h}:{a}".replace(" ", "_"),
                "date": fecha,
                "season": season,
                "league": liga,
                "home": h,
                "away": a,
                "home_goals": self._num(r.get("FTHG")),
                "away_goals": self._num(r.get("FTAG")),
                "home_shots": self._num(r.get("HS")),
                "away_shots": self._num(r.get("AS")),
                "home_sot": self._num(r.get("HST")),
                "away_sot": self._num(r.get("AST")),
            })
        if not out:
            return pd.DataFrame()
        return normalize(pd.DataFrame(out))

    # -- interfaz pública -------------------------------------------------

    def fetch(self, leagues: Iterable[str] = LIGAS_SEGUNDA,
              seasons: Iterable[int] = ()) -> pd.DataFrame:
        """Partidos jugados de las ligas indicadas (por su nombre canónico).

        Por defecto sólo las segundas divisiones: las primeras vienen de
        Understat, que además trae xG, y mezclarlas duplicaría partidos.
        """
        inverso = {v: k for k, v in CODIGOS.items()}
        codigos = [inverso[lg] for lg in leagues if lg in inverso]
        trozos = []
        for season in seasons:
            for code in codigos:
                try:
                    filas = self._get(code, season)
                except FootballDataError as exc:
                    log.error("%s", exc)
                    continue
                trozos.append(self._to_frame(filas, code, season, traducir=True))
        trozos = [t for t in trozos if not t.empty]
        if not trozos:
            return pd.DataFrame()
        return pd.concat(trozos, ignore_index=True).sort_values("date", kind="mergesort").reset_index(drop=True)

    def odds(self, leagues: Iterable[str] = LIGAS_PRIMERA,
             seasons: Iterable[int] = ()) -> pd.DataFrame:
        """Probabilidades implícitas del mercado, listas para cruzar por
        (fecha, local, visitante) con el almacén.
        """
        inverso = {v: k for k, v in CODIGOS.items()}
        codigos = [inverso[lg] for lg in leagues if lg in inverso]
        out = []
        for season in seasons:
            for code in codigos:
                try:
                    filas = self._get(code, season)
                except FootballDataError as exc:
                    log.error("%s", exc)
                    continue
                for r in filas:
                    if not r.get("HomeTeam") or not r.get("Date"):
                        continue
                    fecha = pd.to_datetime(r["Date"], dayfirst=True, errors="coerce")
                    if pd.isna(fecha):
                        continue
                    probs, fuente = self._cuotas(r)
                    if probs is None:
                        continue
                    out.append({
                        "date": fecha.normalize(),
                        "league": CODIGOS[code],
                        "home": self._team(r["HomeTeam"], traducir=True),
                        "away": self._team(r["AwayTeam"], traducir=True),
                        "market_H": probs[0], "market_D": probs[1], "market_A": probs[2],
                        "odds_source": fuente,
                    })
        return pd.DataFrame(out)

    # -- diagnóstico ------------------------------------------------------

    def unmapped_teams(self, known: Iterable[str], leagues=LIGAS_PRIMERA,
                       seasons: Iterable[int] = ()) -> set:
        """Nombres de primera división sin equivalencia en `known`.

        Lo usa el test de integridad de la tabla de alias: si football-data
        renombra un equipo o asciende uno nuevo, aparece aquí antes de que
        pueda corromper un cruce.
        """
        known = set(known)
        inverso = {v: k for k, v in CODIGOS.items()}
        faltan = set()
        for season in seasons:
            for lg in leagues:
                code = inverso.get(lg)
                if code is None:
                    continue
                try:
                    filas = self._get(code, season)
                except FootballDataError:
                    continue
                for r in filas:
                    for col in ("HomeTeam", "AwayTeam"):
                        n = (r.get(col) or "").strip()
                        if n and self._team(n, traducir=True) not in known:
                            faltan.add(n)
        return faltan
