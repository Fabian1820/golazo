#!/usr/bin/env python
"""¿Sigue siendo defendible el modelo que se está sirviendo?

    python scripts/review_model.py

El backtest se ejecutó una vez y la decisión quedó congelada. Pero la respuesta
cambia con los datos: con 7.203 partidos de evaluación el gradient boosting
superaba a Dixon-Coles de forma significativa, y con 12.553 dejó de hacerlo.

Regla de decisión
-----------------
No se cambia de modelo porque otro encabece la tabla —empatan y el orden baila
con el ruido— sino sólo si el modelo servido es **significativamente peor** que
el mejor, con el intervalo de confianza del 95% excluyendo el cero.

Mientras estén empatados se mantiene Dixon-Coles, que a igualdad de precisión
entrega la distribución conjunta de marcadores.

Sale con código 1 si hay que revisar la decisión, para que el pipeline lo avise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from golazo.artifacts import ModelArtifact
from golazo.config import OUTCOMES, REPORTS_DIR

N_BOOT = 10_000
SEED = 42


def rps_por_partido(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    idx = {o: i for i, o in enumerate(OUTCOMES)}
    e = np.zeros_like(p)
    e[np.arange(len(y)), [idx[v] for v in y]] = 1.0
    cp, ce = np.cumsum(p, axis=1), np.cumsum(e, axis=1)
    return ((cp[:, :-1] - ce[:, :-1]) ** 2).sum(axis=1) / (len(OUTCOMES) - 1)


def intervalo(dif: np.ndarray, semanas: np.ndarray, n_boot: int = N_BOOT) -> tuple:
    """IC 95% de la diferencia media, remuestreando semanas enteras."""
    rng = np.random.default_rng(SEED)
    unicas = np.unique(semanas)
    por_semana = {s: dif[semanas == s] for s in unicas}
    medias = np.empty(n_boot)
    for b in range(n_boot):
        muestra = rng.choice(unicas, size=len(unicas), replace=True)
        medias[b] = np.concatenate([por_semana[s] for s in muestra]).mean()
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--servido", default=None,
                    help="modelo a auditar (por defecto, el del artefacto 'latest')")
    args = ap.parse_args()

    ruta = REPORTS_DIR / "backtest_predictions.csv"
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta}. Ejecuta antes: python scripts/run_backtest.py")

    servido = args.servido
    if servido is None:
        try:
            servido = ModelArtifact.load().metadata.model_name
        except FileNotFoundError:
            raise SystemExit("No hay modelo entrenado y no se indicó --servido") from None

    preds = pd.read_csv(ruta, parse_dates=["date"])
    y = preds["result"].to_numpy()
    semanas = preds["date"].dt.to_period("W").astype(str).to_numpy()

    modelos = sorted({c.rsplit("_", 1)[0] for c in preds.columns if c.endswith(("_H", "_D", "_A"))})
    if servido not in modelos:
        raise SystemExit(f"El modelo servido '{servido}' no está en el backtest ({modelos})")

    # Sólo compiten los modelos reales, no las referencias triviales.
    referencias = {"uniforme", "tasa_base", "original_fugado", "market"}
    candidatos = [m for m in modelos if m not in referencias]

    rps = {m: rps_por_partido(preds[[f"{m}_{o}" for o in OUTCOMES]].to_numpy(), y) for m in candidatos}
    medias = {m: float(v.mean()) for m, v in rps.items()}
    mejor = min(medias, key=medias.get)

    print(f"Partidos evaluados: {len(y)}")
    print(f"Modelo servido    : {servido}  (RPS {medias[servido]:.4f})")
    print(f"Mejor de la tabla : {mejor}  (RPS {medias[mejor]:.4f})")

    if mejor == servido:
        print("\nEl modelo servido encabeza la tabla. Nada que revisar.")
        return

    dif = rps[servido] - rps[mejor]
    lo, hi = intervalo(dif, semanas)
    print(f"\nDiferencia (servido - mejor): {dif.mean():+.4f}  IC 95% [{lo:+.4f}, {hi:+.4f}]")

    if lo > 0:
        print(f"\n  REVISAR: '{servido}' es significativamente peor que '{mejor}'.")
        print("  El intervalo excluye el cero, así que la diferencia es real.")
        print(f"  Considera: python scripts/train.py --model {mejor}")
        raise SystemExit(1)

    print(f"\n  Sin cambios. '{servido}' y '{mejor}' son indistinguibles "
          "(el intervalo cruza el cero).")
    print("  A igualdad de precisión se mantiene el que entrega marcadores completos.")


if __name__ == "__main__":
    main()
