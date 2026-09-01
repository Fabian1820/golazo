#!/usr/bin/env python
"""¿Las diferencias entre modelos son reales o ruido?

Compara modelos por pares sobre el MISMO conjunto de partidos usando el RPS
por partido, con bootstrap de bloques (bloques semanales, para no asumir
independencia entre partidos de la misma jornada).

Sin esto, un 0.5% de diferencia en RPS puede leerse como una mejora cuando es
indistinguible de cero.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from golazo.config import OUTCOMES, REPORTS_DIR

N_BOOT = 10_000
SEED = 42


def rps_por_partido(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    idx = {o: i for i, o in enumerate(OUTCOMES)}
    e = np.zeros_like(p)
    e[np.arange(len(y)), [idx[v] for v in y]] = 1.0
    cp, ce = np.cumsum(p, axis=1), np.cumsum(e, axis=1)
    return ((cp[:, :-1] - ce[:, :-1]) ** 2).sum(axis=1) / (len(OUTCOMES) - 1)


def main() -> None:
    preds = pd.read_csv(REPORTS_DIR / "backtest_predictions.csv", parse_dates=["date"])
    y = preds["result"].to_numpy()
    modelos = sorted({c.rsplit("_", 1)[0] for c in preds.columns if c.endswith(("_H", "_D", "_A"))})

    per_match = {m: rps_por_partido(preds[[f"{m}_{o}" for o in OUTCOMES]].to_numpy(), y) for m in modelos}

    # Bloques semanales: los partidos de una misma jornada están correlacionados.
    bloque = preds["date"].dt.to_period("W").astype(str).to_numpy()
    bloques = np.unique(bloque)
    idx_por_bloque = [np.where(bloque == b)[0] for b in bloques]
    rng = np.random.default_rng(SEED)
    sorteos = rng.integers(0, len(bloques), size=(N_BOOT, len(bloques)))

    orden = sorted(modelos, key=lambda m: per_match[m].mean())
    print(f"\nBootstrap de bloques ({N_BOOT} repeticiones, {len(bloques)} semanas, {len(preds)} partidos)")
    print("=" * 78)
    print(f"{'modelo':<20}{'RPS':>9}{'IC 95%':>22}")
    for m in orden:
        d = per_match[m]
        muestras = np.array([d[np.concatenate([idx_por_bloque[j] for j in s])].mean() for s in sorteos])
        lo, hi = np.percentile(muestras, [2.5, 97.5])
        print(f"{m:<20}{d.mean():>9.4f}{f'[{lo:.4f}, {hi:.4f}]':>22}")

    print("\nDiferencias por pares (RPS del primero menos el segundo; negativo = el primero es mejor)")
    print("=" * 78)
    print(f"{'comparación':<42}{'dif.':>9}{'IC 95%':>21}{'':>6}")
    for i, a in enumerate(orden):
        for b in orden[i + 1:]:
            d = per_match[a] - per_match[b]
            muestras = np.array([d[np.concatenate([idx_por_bloque[j] for j in s])].mean() for s in sorteos])
            lo, hi = np.percentile(muestras, [2.5, 97.5])
            sig = "*" if (lo > 0) or (hi < 0) else "ns"
            print(f"{a + ' vs ' + b:<42}{d.mean():>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>21}{sig:>6}")
    print("\n*  = el intervalo de confianza del 95% excluye el cero (diferencia real)")
    print("ns = indistinguible de cero con estos datos")


if __name__ == "__main__":
    main()
