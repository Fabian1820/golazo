"""Fuente viva: Understat.

Understat expone los datos de liga en un endpoint JSON:

    https://understat.com/getLeagueData/{liga}/{temporada}

De ahí salen los partidos (jugados y anunciados) y las estadísticas avanzadas
por equipo y jornada: xG, PPDA y pases en zona profunda.

Buena ciudadanía
----------------
Es un sitio público que no cobra por esto. El cliente hace una petición por
liga y temporada, con una pausa entre ellas y una caché en disco, de modo que
refrescar a diario supone cinco peticiones. No hay motivo para pedir más.

Lo que NO trae
--------------
El endpoint de liga no incluye tiros ni tiros a puerta. Viajan como NaN. Ver
`docs/DATOS.md` sobre por qué las features no dependen de ellos.
"""
from __future__ import annotations

import gzip
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .base import normalize

log = logging.getLogger(__name__)

BASE_URL = "https://understat.com/getLeagueData"
LEAGUES = ("EPL", "La liga", "Bundesliga", "Serie A", "Ligue 1")

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
REQUEST_DELAY = 1.5      # segundos entre peticiones
TIMEOUT = 30
RETRIES = 3


class UnderstatError(RuntimeError):
    pass


class UnderstatSource:
    """Cliente del endpoint de liga de Understat."""

    name = "understat"

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

    def _get(self, league: str, season: int) -> dict:
        cache = self.cache_dir / f"{league.replace(' ', '_')}_{season}.json" if self.cache_dir else None
        if cache and cache.exists():
            log.debug("caché: %s", cache.name)
            return json.loads(cache.read_text(encoding="utf-8"))

        url = f"{BASE_URL}/{urllib.parse.quote(league)}/{season}"
        ultimo = None
        for intento in range(1, RETRIES + 1):
            self._throttle()
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": USER_AGENT,
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": f"https://understat.com/league/{league}/{season}",
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                })
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    raw = resp.read()
                if raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                data = json.loads(raw)
                break
            except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
                ultimo = exc
                log.warning("Understat %s/%s intento %d/%d: %s", league, season, intento, RETRIES, exc)
                time.sleep(2 ** intento)
        else:
            raise UnderstatError(f"No se pudo obtener {league}/{season}: {ultimo}") from ultimo

        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data), encoding="utf-8")
        return data

    # -- transformación ---------------------------------------------------

    @staticmethod
    def _team_stats(data: dict) -> dict:
        """Indexa las estadísticas por (equipo, día) para cruzarlas con los partidos."""
        out = {}
        for equipo in data.get("teams", {}).values():
            titulo = equipo["title"]
            for j in equipo.get("history", []):
                ppda = j.get("ppda") or {}
                att, dfn = ppda.get("att"), ppda.get("def")
                out[(titulo, str(j["date"])[:10])] = {
                    "xg": j.get("xG"),
                    "deep": j.get("deep"),
                    "ppda": (att / dfn) if att is not None and dfn else None,
                }
        return out

    def _to_frame(self, data: dict, league: str, season: int, *, played: bool) -> pd.DataFrame:
        stats = self._team_stats(data)
        filas = []
        for m in data.get("dates", []):
            if bool(m.get("isResult")) != played:
                continue
            dia = str(m["datetime"])[:10]
            h, a = m["h"]["title"], m["a"]["title"]
            sh, sa = stats.get((h, dia), {}), stats.get((a, dia), {})
            filas.append({
                "match_id": f"understat:{m['id']}",
                "date": m["datetime"],
                "season": season,
                "league": league,
                "home": h,
                "away": a,
                "home_goals": (m.get("goals") or {}).get("h") if played else None,
                "away_goals": (m.get("goals") or {}).get("a") if played else None,
                "home_xg": (m.get("xG") or {}).get("h") if played else None,
                "away_xg": (m.get("xG") or {}).get("a") if played else None,
                "home_deep": sh.get("deep"),
                "away_deep": sa.get("deep"),
                "home_ppda": sh.get("ppda"),
                "away_ppda": sa.get("ppda"),
            })
        if not filas:
            return pd.DataFrame()
        return normalize(pd.DataFrame(filas), require_results=played)

    # -- interfaz pública -------------------------------------------------

    def _collect(self, leagues, seasons, *, played: bool) -> pd.DataFrame:
        trozos = []
        for season in seasons:
            for league in leagues:
                try:
                    trozos.append(self._to_frame(self._get(league, season), league, season, played=played))
                except UnderstatError as exc:
                    log.error("%s", exc)
        trozos = [t for t in trozos if not t.empty]
        if not trozos:
            return pd.DataFrame()
        return pd.concat(trozos, ignore_index=True).sort_values("date", kind="mergesort").reset_index(drop=True)

    def fetch(self, leagues: Iterable[str] = LEAGUES, seasons: Iterable[int] = ()) -> pd.DataFrame:
        """Partidos ya jugados."""
        return self._collect(list(leagues), list(seasons), played=True)

    def fixtures(self, leagues: Iterable[str] = LEAGUES, seasons: Iterable[int] = ()) -> pd.DataFrame:
        """Partidos anunciados y aún no jugados."""
        return self._collect(list(leagues), list(seasons), played=False)
