#!/usr/bin/env python
"""Valida el decaimiento temporal ξ de Dixon-Coles sobre estos datos.

    python scripts/tune_xi.py

ξ controla cuánto pesa el pasado: el peso de un partido de hace `t` días es
`exp(-ξ·t)`. Estaba fijado en 0.0018 porque es el rango habitual en la
literatura, no porque se hubiese comprobado aquí. Es el único parámetro libre
del modelo que se sirve.

Metodología
-----------
Se busca en una ventana INTERNA (hasta 2023-07) y se confirma en un periodo
RETENIDO (2023-07 en adelante) que no interviene en la elección. Ajustar ξ
contra el mismo tramo que después lo evalúa daría un número inflado, que es
justo el error que este proyecto existe para no repetir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from golazo.backtest import walk_forward
from golazo.config import OUTCOMES, REPORTS_DIR
from golazo.data import load_matches
from golazo.metrics import rps
from golazo.models import DixonColes

# ξ = 0 significa sin decaimiento: todo el historial pesa igual.
REJILLA = [0.0, 0.0005, 0.0010, 0.0015, 0.0018, 0.0022, 0.0030, 0.0040, 0.0060]

CORTE_INTERNO = "2023-07-01"


def vida_media(xi: float) -> str:
    return "∞" if xi == 0 else f"{np.log(2) / xi:.0f} d"


def evaluar(df: pd.DataFrame, rejilla, inicio: str, fin: str | None, refit_days: int) -> pd.DataFrame:
    sub = df if fin is None else df[df["date"] < pd.Timestamp(fin)]
    fabricas = {f"xi_{xi:.4f}": (lambda xi=xi: DixonColes(xi=xi)) for xi in rejilla}
    preds = walk_forward(sub, fabricas, start=inicio, refit_days=refit_days)

    y = preds["result"].to_numpy()
    filas = []
    for xi in rejilla:
        p = preds[[f"xi_{xi:.4f}_{o}" for o in OUTCOMES]].to_numpy()
        filas.append({"xi": xi, "vida_media": vida_media(xi), "n": len(y), "rps": rps(p, y)})
    return pd.DataFrame(filas).sort_values("rps").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-days", type=int, default=21,
                    help="más grueso que el backtest principal: aquí importa el orden, no el valor absoluto")
    ap.add_argument("--start", default="2019-08-01")
    args = ap.parse_args()

    df = load_matches()
    print(f"{len(df)} partidos · {df.date.min():%Y-%m-%d} a {df.date.max():%Y-%m-%d}")

    print(f"\n[1/2] Búsqueda en ventana INTERNA ({args.start} a {CORTE_INTERNO}), "
          f"{len(REJILLA)} valores de ξ:")
    interno = evaluar(df, REJILLA, args.start, CORTE_INTERNO, args.refit_days)
    print(interno.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    mejor = float(interno.iloc[0]["xi"])
    actual = 0.0018
    print(f"\n  mejor en interno: ξ = {mejor:.4f}  (vida media {vida_media(mejor)})")
    print(f"  valor actual    : ξ = {actual:.4f}  (vida media {vida_media(actual)})")

    if mejor == actual:
        print("\nEl valor actual ya es el mejor de la rejilla. No hay nada que cambiar.")

    print(f"\n[2/2] Confirmación en periodo RETENIDO ({CORTE_INTERNO} en adelante):")
    retenido = evaluar(df[df["date"] >= pd.Timestamp(CORTE_INTERNO)],
                       sorted({mejor, actual}), CORTE_INTERNO, None, args.refit_days)
    print(retenido.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    interno.to_csv(REPORTS_DIR / "xi_interno.csv", index=False)
    retenido.to_csv(REPORTS_DIR / "xi_retenido.csv", index=False)

    d = retenido.set_index("xi")["rps"]
    if mejor != actual and mejor in d.index and actual in d.index:
        delta = d[actual] - d[mejor]
        print(f"\nEn el periodo retenido, ξ={mejor:.4f} {'mejora' if delta > 0 else 'empeora'} "
              f"a ξ={actual:.4f} en {abs(delta):.4f} de RPS.")
        print("Cambiar el valor por defecto sólo se justifica si la mejora también aparece aquí.")

    print(f"\nEscrito en {REPORTS_DIR}/: xi_interno.csv, xi_retenido.csv")


if __name__ == "__main__":
    main()
