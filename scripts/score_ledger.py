#!/usr/bin/env python
"""Puntúa las predicciones registradas contra los resultados reales.

    python scripts/score_ledger.py

Nunca modifica el registro: lo lee, verifica la cadena, cruza con los
resultados conocidos y escribe el historial en reports/track_record.md.

Éste es el entregable que distingue un proyecto serio: no las métricas de un
backtest que uno mismo diseñó, sino un historial de predicciones fijadas antes
de cada partido y puntuadas después.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from golazo.config import OUTCOMES, REPORTS_DIR
from golazo.data import load_matches
from golazo.ledger import PredictionLedger
from golazo.metrics import expected_calibration_error, skill_score, summarize


def main() -> None:
    ledger = PredictionLedger()
    ok, mensaje = ledger.verify()
    print(f"Integridad del registro: {mensaje}")
    if not ok:
        raise SystemExit("El registro no supera la verificación. No se puntúa nada.")

    registros = ledger.records()
    if not registros:
        print("\nEl registro está vacío. Aún no hay nada que puntuar.")
        print("Las predicciones se añaden al emitirlas desde la API o el CLI.")
        return

    reg = pd.DataFrame([{
        "kickoff": pd.Timestamp(r.kickoff).tz_localize(None),
        "league": r.league, "home": r.home, "away": r.away,
        "model_version": r.model_version,
        **{f"p_{o}": r.probabilities[o] for o in OUTCOMES},
    } for r in registros])

    hechos = load_matches()[["date", "league", "home", "away", "result"]].copy()
    hechos["dia"] = hechos["date"].dt.normalize()
    reg["dia"] = reg["kickoff"].dt.normalize()

    unido = reg.merge(hechos[["dia", "home", "away", "result"]], on=["dia", "home", "away"], how="left")
    pendientes = int(unido["result"].isna().sum())
    jugados = unido[unido["result"].notna()]

    print(f"\nPredicciones registradas: {len(unido)}")
    print(f"  ya jugadas : {len(jugados)}")
    print(f"  pendientes : {pendientes}")

    if jugados.empty:
        print("\nNinguna se ha resuelto todavía. Vuelve cuando haya resultados.")
        return

    p = jugados[[f"p_{o}" for o in OUTCOMES]].to_numpy()
    y = jugados["result"].to_numpy()
    resumen = summarize(p, y)
    resumen["ece"] = expected_calibration_error(p, y)

    base = np.tile(np.array([0.4344, 0.2516, 0.3140]), (len(y), 1))
    resumen_base = summarize(base, y)

    lineas = [
        "# Historial de predicciones",
        "",
        "Predicciones fijadas **antes** de cada partido y puntuadas después.",
        f"Registro verificado: {mensaje.lower()}.",
        "",
        f"- Registradas: **{len(unido)}**  ·  resueltas: **{len(jugados)}**  ·  pendientes: {pendientes}",
        f"- Periodo: {jugados['kickoff'].min():%Y-%m-%d} a {jugados['kickoff'].max():%Y-%m-%d}",
        f"- Versiones de modelo: {', '.join(sorted(jugados['model_version'].unique()))}",
        "",
        "| métrica | modelo | tasa base | mejora |",
        "|---|---|---|---|",
        f"| RPS | {resumen['rps']:.4f} | {resumen_base['rps']:.4f} | "
        f"{100 * skill_score(resumen['rps'], resumen_base['rps']):+.1f}% |",
        f"| log-loss | {resumen['log_loss']:.4f} | {resumen_base['log_loss']:.4f} | "
        f"{100 * skill_score(resumen['log_loss'], resumen_base['log_loss']):+.1f}% |",
        f"| Brier | {resumen['brier']:.4f} | {resumen_base['brier']:.4f} | "
        f"{100 * skill_score(resumen['brier'], resumen_base['brier']):+.1f}% |",
        f"| ECE | {resumen['ece']:.4f} | — | — |",
        f"| acierto | {resumen['accuracy']:.1%} | {resumen_base['accuracy']:.1%} | — |",
        "",
    ]

    if len(jugados) < 100:
        lineas += [
            f"> Con {len(jugados)} predicciones resueltas estas cifras son orientativas. "
            "Hacen falta varios cientos para que sean estables.",
            "",
        ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    salida = REPORTS_DIR / "track_record.md"
    salida.write_text("\n".join(lineas), encoding="utf-8")

    print(f"\n  RPS {resumen['rps']:.4f} (tasa base {resumen_base['rps']:.4f}) · "
          f"ECE {resumen['ece']:.4f} · acierto {resumen['accuracy']:.1%}")
    print(f"\nEscrito en {salida}")


if __name__ == "__main__":
    main()
