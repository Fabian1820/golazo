#!/usr/bin/env python
"""Ejecuta el backtest walk-forward y escribe el informe en reports/.

    python scripts/run_backtest.py [--start 2019-08-01] [--refit-days 7]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from golazo.backtest import evaluate, walk_forward
from golazo.config import OUTCOMES, REPORTS_DIR, SERVED_LEAGUES
from golazo.data import load_matches
from golazo.features import build_features, feature_columns
from golazo.metrics import calibration_table
from golazo.models import (
    BaseRateModel,
    DixonColes,
    EloLogisticModel,
    GradientBoostingModel,
    LegacyLeakyModel,
    UniformModel,
)


def calibration_plot(preds: pd.DataFrame, names, path: Path) -> None:
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    y = preds["result"].to_numpy()
    for ax, name in zip(axes, names):
        p = preds[[f"{name}_{o}" for o in OUTCOMES]].to_numpy()
        tab = calibration_table(p, y, n_bins=10)
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="calibración perfecta")
        ax.plot(tab["pred_mean"], tab["obs_freq"], "o-", color="#2b7bba", lw=2, ms=5)
        for _, r in tab.iterrows():
            ax.annotate(int(r["n"]), (r["pred_mean"], r["obs_freq"]), fontsize=6,
                        xytext=(0, -11), textcoords="offset points", ha="center", color="0.45")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("probabilidad predicha")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("frecuencia observada")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Curvas de fiabilidad — backtest walk-forward", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-08-01", help="primera fecha predicha (2018 queda como burn-in)")
    ap.add_argument("--refit-days", type=int, default=7)
    ap.add_argument("--out", default=str(REPORTS_DIR))
    ap.add_argument("--all-leagues", action="store_true",
                    help="evaluar también sobre divisiones que no se pronostican")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Cargando datos y construyendo features pre-partido...")
    df = build_features(load_matches())
    cols = feature_columns(df)
    print(f"  {len(df)} partidos · {len(cols)} features · {df.date.min():%Y-%m-%d} a {df.date.max():%Y-%m-%d}")

    factories = {
        "uniforme": UniformModel,
        "tasa_base": BaseRateModel,
        "original_fugado": LegacyLeakyModel,
        "elo_logistico": EloLogisticModel,
        "dixon_coles": DixonColes,
        "gradient_boosting": lambda: GradientBoostingModel(feature_cols=cols),
    }

    print(f"\nBacktest walk-forward desde {args.start}, reajuste cada {args.refit_days} días:")
    servidas = None if args.all_leagues else list(SERVED_LEAGUES)
    if servidas:
        print(f"  evaluando sobre las ligas que se sirven: {', '.join(servidas)}")
        print("  (el entrenamiento usa TODAS las divisiones)")
    preds = walk_forward(df, factories, start=args.start, refit_days=args.refit_days,
                         eval_leagues=servidas)

    table = evaluate(preds, list(factories), reference="tasa_base")
    preds.to_csv(out / "backtest_predictions.csv", index=False)
    table.to_csv(out / "metrics.csv")
    calibration_plot(preds, ["original_fugado", "dixon_coles", "elo_logistico", "gradient_boosting"],
                     out / "calibration.png")

    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:8.4f}")
    print(f"\n{'=' * 92}\nRESULTADOS — {len(preds)} partidos predichos, ninguno visto en entrenamiento\n{'=' * 92}")
    print(table.to_string())
    print("\nRPS y log-loss: menor es mejor. skill_% : mejora sobre 'tasa_base' (>0 aporta).")
    print("ECE: error de calibración esperado; menor es mejor.")
    print(f"\nEscrito en {out}/: metrics.csv, backtest_predictions.csv, calibration.png")


if __name__ == "__main__":
    main()
