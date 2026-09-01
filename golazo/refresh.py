"""Refresco de datos desde una fuente viva.

Este es el eslabón que separa un dataset congelado de un sistema que funciona:
pide a la fuente los partidos de las temporadas indicadas —jugados y
anunciados— y los funde en el almacén canónico.

Es idempotente: ejecutarlo dos veces seguidas no duplica nada, y un partido que
pasó de anunciado a jugado se sustituye por su versión con resultado.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR, MATCHES_STORE
from .sources.understat import LEAGUES, UnderstatSource
from .store import upsert

log = logging.getLogger(__name__)


def current_season(today: pd.Timestamp | None = None) -> int:
    """Temporada en curso según la convención de Understat (2024 = 2024/25).

    Las grandes ligas europeas arrancan en agosto; a partir de julio se
    considera que la temporada nueva es la del año en curso.
    """
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.now()
    return int(today.year if today.month >= 7 else today.year - 1)


def refresh(seasons: Sequence[int] | None = None,
            leagues: Iterable[str] = LEAGUES,
            store_path: Path = MATCHES_STORE,
            use_cache: bool = False,
            include_fixtures: bool = True) -> dict:
    """Descarga y funde. Devuelve un resumen de lo que cambió."""
    seasons = list(seasons) if seasons else [current_season()]
    leagues = list(leagues)

    fuente = UnderstatSource(cache_dir=CACHE_DIR if use_cache else None)

    log.info("Descargando %d liga(s) x %d temporada(s) de %s...",
             len(leagues), len(seasons), fuente.name)
    jugados = fuente.fetch(leagues=leagues, seasons=seasons)
    calendario = fuente.fixtures(leagues=leagues, seasons=seasons) if include_fixtures else pd.DataFrame()

    entrada = pd.concat([x for x in (jugados, calendario) if not x.empty], ignore_index=True) \
        if (not jugados.empty or not calendario.empty) else pd.DataFrame()

    resumen = upsert(entrada, store_path)
    resumen.update({
        "fuente": fuente.name,
        "temporadas": seasons,
        "ligas": leagues,
        "descargados_jugados": int(len(jugados)),
        "descargados_calendario": int(len(calendario)),
    })
    return resumen


def describe(resumen: dict) -> str:
    lineas = [
        f"Fuente: {resumen['fuente']} · temporadas {resumen['temporadas']} · {len(resumen['ligas'])} ligas",
        f"  descargados : {resumen['descargados_jugados']} jugados, "
        f"{resumen['descargados_calendario']} anunciados",
        f"  almacén     : {resumen['antes']} -> {resumen['despues']} partidos "
        f"({resumen['nuevos']} nuevos, {resumen['actualizados']} actualizados)",
    ]
    if resumen.get("hasta"):
        lineas.append(f"  datos hasta : {resumen['hasta']}")
    return "\n".join(lineas)
