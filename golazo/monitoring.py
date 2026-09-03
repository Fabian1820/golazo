"""Detección de degradación silenciosa de los datos.

El modo de fallo que importa no es que el refresco reviente: eso se ve. Es que
devuelva menos de lo que debería —porque el sitio cambió de formato, porque
respondió una página de error con código 200, porque una liga dejó de
publicarse— y el pipeline siga en verde reentrenando con datos viejos.

Ya pasó una vez: Understat movió los datos del HTML a un endpoint JSON y
cualquier scraper que leyera `datesData` habría empezado a devolver cero
partidos sin lanzar una sola excepción.

Estas comprobaciones se ejecutan tras cada refresco y comparan el estado del
almacén contra lo que cabe esperar. Devuelven `Finding` como `golazo.validation`,
para que el CLI las presente igual.
"""
from __future__ import annotations

import pandas as pd

from .validation import Finding

# Un refresco diario en temporada debería dejar el dato más reciente a pocos
# días. Más de esto y algo dejó de llegar.
DIAS_FRESCURA_AVISO = 10
DIAS_FRESCURA_ERROR = 30

# Partidos por temporada y liga, aproximados. Por debajo de esta fracción la
# temporada está incompleta de forma sospechosa.
FRACCION_MINIMA_TEMPORADA = 0.5

PARTIDOS_ESPERADOS = {
    "EPL": 380, "La liga": 380, "Serie A": 380,
    "Bundesliga": 306, "Ligue 1": 306,
    "Championship": 552, "La liga 2": 462, "Serie B": 380,
    "Bundesliga 2": 306, "Ligue 2": 306,
}


def _frescura(df: pd.DataFrame, ahora: pd.Timestamp) -> list[Finding]:
    jugados = df[df["home_goals"].notna()]
    if jugados.empty:
        return [Finding("error", "frescura", "No hay ningún partido jugado en el almacén")]

    dias = (ahora.normalize() - jugados["date"].max().normalize()).days
    if dias > DIAS_FRESCURA_ERROR:
        return [Finding("error", "frescura",
                        f"El último partido jugado tiene {dias} días. "
                        "El refresco no está trayendo resultados nuevos", dias)]
    if dias > DIAS_FRESCURA_AVISO:
        return [Finding("aviso", "frescura",
                        f"El último partido jugado tiene {dias} días. "
                        "Normal en parón de selecciones o fuera de temporada", dias)]
    return [Finding("info", "frescura", f"Último partido jugado hace {dias} días")]


def _calendario(df: pd.DataFrame, ahora: pd.Timestamp) -> list[Finding]:
    """Sin calendario por delante no se puede pronosticar nada."""
    futuros = df[(df["home_goals"].isna()) & (df["date"] > ahora)]
    if futuros.empty:
        return [Finding("error", "calendario",
                        "No hay ningún partido anunciado por delante. "
                        "El pronóstico no tiene sobre qué trabajar")]
    proximos = int((futuros["date"] <= ahora + pd.Timedelta(days=14)).sum())
    if proximos == 0:
        return [Finding("aviso", "calendario",
                        f"No hay partidos en los próximos 14 días ({len(futuros)} más adelante)")]
    return [Finding("info", "calendario",
                    f"{proximos} partidos en los próximos 14 días, {len(futuros)} en total")]


def _temporadas_completas(df: pd.DataFrame, ahora: pd.Timestamp) -> list[Finding]:
    """Una temporada cerrada muy por debajo de lo esperado indica descarga parcial."""
    out = []
    jugados = df[df["home_goals"].notna()]
    temporada_actual = ahora.year if ahora.month >= 7 else ahora.year - 1

    for (liga, temporada), sub in jugados.groupby(["league", "season"]):
        if temporada >= temporada_actual:
            continue  # en curso: es normal que esté incompleta
        esperados = PARTIDOS_ESPERADOS.get(liga)
        if not esperados:
            continue
        if len(sub) < esperados * FRACCION_MINIMA_TEMPORADA:
            out.append(Finding("error", "completitud",
                               f"{liga} {temporada} tiene {len(sub)} partidos, "
                               f"se esperaban ~{esperados}", len(sub)))
    return out


def _ligas_presentes(df: pd.DataFrame, esperadas: set | None = None) -> list[Finding]:
    presentes = set(df["league"].unique())
    esperadas = esperadas or set(PARTIDOS_ESPERADOS)
    faltan = esperadas - presentes
    if faltan:
        return [Finding("error", "cobertura",
                        f"Faltan ligas enteras en el almacén: {sorted(faltan)}", len(faltan))]
    return [Finding("info", "cobertura", f"{len(presentes)} ligas presentes")]


def check_freshness(df: pd.DataFrame, now: pd.Timestamp | None = None) -> list[Finding]:
    """Todas las comprobaciones de degradación, ordenadas por severidad."""
    from .validation import SEVERIDADES

    ahora = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
    hallazgos: list[Finding] = []
    for check in (_frescura, _calendario, _temporadas_completas):
        hallazgos.extend(check(df, ahora))
    hallazgos.extend(_ligas_presentes(df))
    return sorted(hallazgos, key=lambda f: SEVERIDADES.index(f.severity))
