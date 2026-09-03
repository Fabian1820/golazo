"""Validación de la integridad de los datos.

Un modelo no puede ser mejor que sus datos, y los errores de datos son
silenciosos: un duplicado, una fecha mal parseada o un partido con marcador
imposible no lanzan ninguna excepción, simplemente degradan el modelo sin que
nadie se entere.

Estas comprobaciones se ejecutan sobre el historial cargado y devuelven una
lista de hallazgos clasificados por severidad.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Rangos plausibles. Fuera de ellos casi siempre hay un error de datos, no un
# partido excepcional.
LIMITES = {
    "home_goals": (0, 15),
    "away_goals": (0, 15),
    "home_xg": (0.0, 10.0),
    "away_xg": (0.0, 10.0),
    "home_shots": (0, 60),
    "away_shots": (0, 60),
    "home_sot": (0, 30),
    "away_sot": (0, 30),
    "home_ppda": (0.0, 200.0),
    "away_ppda": (0.0, 200.0),
    # El suelo bajó de 1000 a 800 al incorporar las segundas divisiones. El Elo
    # es un sistema cerrado: con dos categorías en el mismo grupo de
    # valoraciones el rango se ensancha por abajo, y un equipo hundido en
    # segunda cae por debajo de lo que llegaba a bajar cualquiera cuando sólo
    # había primeras. No es un error de datos.
    "elo_h": (800.0, 2500.0),
    "elo_a": (800.0, 2500.0),
}

SEVERIDADES = ("error", "aviso", "info")


@dataclass
class Finding:
    severity: str
    check: str
    message: str
    n_affected: int = 0

    def __str__(self) -> str:
        marca = {"error": "✗", "aviso": "!", "info": "·"}[self.severity]
        sufijo = f" ({self.n_affected} filas)" if self.n_affected else ""
        return f" {marca} [{self.check}] {self.message}{sufijo}"


def _sin_nulos(df: pd.DataFrame) -> list[Finding]:
    """Un nulo en una columna obligatoria es un error; en una opcional, un aviso.

    El esquema canónico declara opcionales las estadísticas avanzadas porque no
    todas las fuentes las publican: Understat no da tiros ni tiros a puerta.
    Tratarlas como error haría fallar la validación de datos perfectamente
    válidos, y a la larga enseñaría a ignorar el validador.
    """
    from .sources.base import OPTIONAL_COLUMNS

    out = []
    nulos = df.isna().sum()
    for col, n in nulos[nulos > 0].items():
        opcional = col in OPTIONAL_COLUMNS
        out.append(Finding(
            "aviso" if opcional else "error", "nulos",
            f"La columna opcional '{col}' no está en todas las fuentes" if opcional
            else f"La columna obligatoria '{col}' tiene valores ausentes",
            int(n)))
    return out


def _sin_duplicados(df: pd.DataFrame) -> list[Finding]:
    out = []
    dup_id = int(df["match_id"].duplicated().sum())
    if dup_id:
        out.append(Finding("error", "duplicados", "Hay match_id repetidos", dup_id))

    dup_fix = int(df.duplicated(subset=["date", "home", "away"]).sum())
    if dup_fix:
        out.append(Finding("error", "duplicados",
                           "Hay partidos repetidos (misma fecha y mismos equipos)", dup_fix))
    return out


def _rangos_plausibles(df: pd.DataFrame) -> list[Finding]:
    out = []
    for col, (lo, hi) in LIMITES.items():
        if col not in df.columns:
            continue
        fuera = int(((df[col] < lo) | (df[col] > hi)).sum())
        if fuera:
            out.append(Finding("error", "rango",
                               f"'{col}' fuera del rango plausible [{lo}, {hi}]", fuera))
    return out


def _coherencia_interna(df: pd.DataFrame) -> list[Finding]:
    out = []
    for lado in ("home", "away"):
        mal = int((df[f"{lado}_sot"] > df[f"{lado}_shots"]).sum())
        if mal:
            out.append(Finding("error", "coherencia",
                               f"Más tiros a puerta que tiros totales ({lado})", mal))

    esperado = np.where(df["home_goals"] > df["away_goals"], "H",
                        np.where(df["home_goals"] < df["away_goals"], "A", "D"))
    mal = int((df["result"].to_numpy() != esperado).sum())
    if mal:
        out.append(Finding("error", "coherencia", "El resultado no concuerda con los goles", mal))

    mismo = int((df["home"] == df["away"]).sum())
    if mismo:
        out.append(Finding("error", "coherencia", "Partidos de un equipo contra sí mismo", mismo))
    return out


def _cobertura_temporal(df: pd.DataFrame) -> list[Finding]:
    out = [Finding("info", "cobertura",
                   f"{len(df)} partidos de {df['date'].min():%Y-%m-%d} a {df['date'].max():%Y-%m-%d}")]

    if not df["date"].is_monotonic_increasing:
        out.append(Finding("error", "orden", "El historial no está ordenado cronológicamente"))

    # Un hueco largo suele indicar una temporada que falta.
    huecos = df["date"].diff().dt.days
    grandes = int((huecos > 90).sum())
    if grandes:
        peor = df.loc[huecos.idxmax(), "date"]
        out.append(Finding("aviso", "huecos",
                           f"Hay pausas de más de 90 días (la mayor termina el {peor:%Y-%m-%d})",
                           grandes))

    antiguedad = (pd.Timestamp.now().normalize() - df["date"].max()).days
    if antiguedad > 365:
        out.append(Finding("aviso", "frescura",
                           f"El dato más reciente tiene {antiguedad} días. "
                           "El modelo no puede predecir partidos actuales"))
    return out


def _equilibrio_de_ligas(df: pd.DataFrame) -> list[Finding]:
    out = []
    for liga, sub in df.groupby("league"):
        equipos = set(sub["home"]) | set(sub["away"])
        locales = set(sub["home"])
        solo_visitante = equipos - locales
        if solo_visitante:
            out.append(Finding("aviso", "cobertura",
                               f"En {liga} hay equipos que nunca juegan en casa: "
                               f"{sorted(solo_visitante)[:3]}", len(solo_visitante)))
    return out


def _ventaja_de_local(df: pd.DataFrame) -> list[Finding]:
    """Comprobación de cordura futbolística: si el local no gana más, algo va mal."""
    tasas = df["result"].value_counts(normalize=True)
    h, a = float(tasas.get("H", 0)), float(tasas.get("A", 0))
    if h <= a:
        return [Finding("error", "cordura",
                        f"El local gana menos que el visitante ({h:.1%} vs {a:.1%}); "
                        "puede que las columnas estén intercambiadas")]
    return [Finding("info", "cordura",
                    f"Ventaja de local: {h:.1%} locales, {float(tasas.get('D', 0)):.1%} empates, {a:.1%} visitantes")]


CHECKS = (_sin_nulos, _sin_duplicados, _rangos_plausibles, _coherencia_interna,
          _cobertura_temporal, _equilibrio_de_ligas, _ventaja_de_local)


def validate(df: pd.DataFrame) -> list[Finding]:
    """Ejecuta todas las comprobaciones y devuelve los hallazgos ordenados."""
    hallazgos: list[Finding] = []
    for check in CHECKS:
        hallazgos.extend(check(df))
    return sorted(hallazgos, key=lambda f: SEVERIDADES.index(f.severity))


def has_errors(findings: list[Finding]) -> bool:
    return any(f.severity == "error" for f in findings)


def report(findings: list[Finding]) -> str:
    errores = sum(f.severity == "error" for f in findings)
    avisos = sum(f.severity == "aviso" for f in findings)
    lineas = [str(f) for f in findings]
    lineas.append("")
    lineas.append(f" {errores} errores · {avisos} avisos"
                  if errores or avisos else " Sin problemas.")
    return "\n".join(lineas)
