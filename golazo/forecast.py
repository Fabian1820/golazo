"""Pronóstico de la próxima jornada, registrado antes de que se juegue.

Éste es el cierre del círculo. Un backtest lo diseña uno mismo y siempre se
puede ajustar hasta que salga bien. Un pronóstico emitido y firmado antes del
saque inicial, y puntuado después, no admite retoque.

El flujo completo:

    golazo fetch      -> trae resultados nuevos y el calendario
    golazo train      -> reentrena con todo lo conocido
    golazo forecast   -> predice lo que viene y lo firma en el registro
    golazo score      -> puntúa lo que ya se jugó

El registro rechaza por diseño cualquier predicción de un partido ya empezado,
así que el historial sólo puede crecer hacia adelante.
"""
from __future__ import annotations

import logging

import pandas as pd

from .ledger import LedgerError, PredictionLedger
from .service import PredictionService, UnknownTeamError
from .store import load_store, upcoming

log = logging.getLogger(__name__)


def pending_fixtures(horizon_days: int = 10, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Partidos anunciados que se juegan dentro del horizonte y aún no han empezado."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
    fx = upcoming(load_store())
    if fx.empty:
        return fx
    limite = now + pd.Timedelta(days=horizon_days)
    return fx[(fx["date"] > now) & (fx["date"] <= limite)].sort_values("date").reset_index(drop=True)


def run(horizon_days: int = 10, record: bool = True,
        service: PredictionService | None = None,
        ledger: PredictionLedger | None = None,
        now: pd.Timestamp | None = None) -> dict:
    """Predice el calendario pendiente y lo firma en el registro."""
    fixtures = pending_fixtures(horizon_days, now=now)
    if fixtures.empty:
        return {"fixtures": 0, "predichos": 0, "registrados": 0, "repetidos": 0,
                "omitidos": [], "predicciones": []}

    service = service or PredictionService.load()
    ledger = ledger if ledger is not None else PredictionLedger()

    # Una predicción por partido y versión de modelo. Reentrenar y volver a
    # predecir es legítimo y queda registrado; repetir la misma predicción a
    # diario sólo serviría para acumular intentos del mismo partido.
    ya_firmados = {
        (r.home, r.away, r.kickoff[:10], r.model_version) for r in ledger.records()
    }
    version = service.artifact.metadata.version_id

    predicciones: list[dict] = []
    omitidos: list[str] = []
    repetidos = 0
    registrados = 0

    for f in fixtures.itertuples(index=False):
        if record and (f.home, f.away, f"{f.date:%Y-%m-%d}", version) in ya_firmados:
            repetidos += 1
            continue
        etiqueta = f"{f.home} vs {f.away} ({f.date:%Y-%m-%d})"
        try:
            pred = service.predict(home=f.home, away=f.away, league=f.league, kickoff=f.date)
        except (UnknownTeamError, ValueError) as exc:
            omitidos.append(f"{etiqueta}: {exc}")
            continue

        fila = pred.to_dict()
        if record:
            try:
                fila["ledger_hash"] = ledger.append(pred).hash
                registrados += 1
            except LedgerError as exc:
                # Ya empezó entre la consulta y el registro, o reloj desfasado.
                omitidos.append(f"{etiqueta}: {exc}")
        predicciones.append(fila)

    return {
        "fixtures": int(len(fixtures)),
        "predichos": len(predicciones),
        "registrados": registrados,
        "repetidos": repetidos,
        "omitidos": omitidos,
        "predicciones": predicciones,
    }


def describe(resultado: dict) -> str:
    if not resultado["fixtures"]:
        return ("No hay partidos anunciados en el horizonte.\n"
                "  Ejecuta `golazo fetch` para traer el calendario de la temporada en curso.")

    cabecera = (f"{resultado['fixtures']} partidos en el horizonte · "
                f"{resultado['predichos']} predichos · {resultado['registrados']} firmados")
    if resultado.get("repetidos"):
        cabecera += f" · {resultado['repetidos']} ya firmados con este modelo"
    lineas = [cabecera, ""]
    for p in resultado["predicciones"]:
        pr = p["probabilities"]
        marca = "✓" if p.get("ledger_hash") else " "
        lineas.append(f" {marca} {p['home']:>24}  {pr['H']:.0%} / {pr['D']:.0%} / {pr['A']:.0%}  "
                      f"{p['away']:<24} {p['kickoff'][:16]}")
    for o in resultado["omitidos"]:
        lineas.append(f"   omitido — {o}")
    return "\n".join(lineas)
