#!/usr/bin/env python
"""Compara los modelos contra la cuota de cierre del mercado.

    python scripts/compare_market.py

La comparación se hace sobre el subconjunto de partidos con cuota conocida, y
todos los modelos se evalúan sobre exactamente esos mismos partidos. Comparar
un modelo sobre 12.000 partidos con el mercado sobre 1.700 no diría nada.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from golazo.config import OUTCOMES, REPORTS_DIR
from golazo.metrics import expected_calibration_error, skill_score, summarize
from golazo.odds import attach, coverage


def main() -> None:
    ruta = REPORTS_DIR / "backtest_predictions.csv"
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta}. Ejecuta antes: python scripts/run_backtest.py")

    preds = pd.read_csv(ruta, parse_dates=["date"])
    con = attach(preds)
    cob = coverage(con)
    print(f"Predicciones del backtest: {cob['total']}")
    print(f"  con cuota de cierre     : {cob['con_cuota']} ({cob['cobertura']:.1%})")
    if not cob["con_cuota"]:
        raise SystemExit("Sin cuotas. Ejecuta antes: golazo odds --seasons ...")

    sub = con[con["market_H"].notna()].copy()
    y = sub["result"].to_numpy()

    modelos = sorted({c.rsplit("_", 1)[0] for c in preds.columns
                      if c.endswith(("_H", "_D", "_A")) and c.rsplit("_", 1)[0] != "market"})
    modelos.append("market")

    filas = {}
    for m in modelos:
        p = sub[[f"{m}_{o}" for o in OUTCOMES]].to_numpy()
        s = summarize(p, y)
        s["ece"] = expected_calibration_error(p, y)
        filas[m] = s

    ref = filas["market"]
    for s in filas.values():
        s["rps_vs_mercado_%"] = -100.0 * skill_score(s["rps"], ref["rps"])

    tabla = pd.DataFrame(filas).T[
        ["n", "rps", "log_loss", "brier", "ece", "accuracy", "rps_vs_mercado_%"]
    ].sort_values("rps")

    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:9.4f}")
    print(f"\n{'=' * 88}")
    print(f"CONTRA EL MERCADO — {len(sub)} partidos con cuota de cierre")
    print("=" * 88)
    print(tabla.to_string())
    print("\nrps_vs_mercado_%: cuánto peor (positivo) o mejor (negativo) que la cuota de cierre.")
    print("La cuota lleva el margen de la casa ya repartido entre los tres resultados.")

    tabla.to_csv(REPORTS_DIR / "market_comparison.csv")
    print(f"\nEscrito en {REPORTS_DIR / 'market_comparison.csv'}")


if __name__ == "__main__":
    main()
