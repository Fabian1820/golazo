#!/usr/bin/env python
"""Migración única: del volcado de Kaggle al almacén canónico.

    python scripts/migrate_store.py

Lee `src/soccer/Matches.csv` (2018-2023) y lo escribe en `data/matches.csv` con
el esquema canónico. A partir de ahí el proyecto no vuelve a tocar los volcados
originales, y `golazo fetch` extiende el almacén con datos vivos.

`ClubElo.csv` deja de usarse: el Elo se calcula en `golazo.elo`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from golazo.config import LEGACY_MATCHES_CSV, MATCHES_STORE
from golazo.sources.base import normalize
from golazo.store import save_store

RENOMBRES = {
    "id": "match_id",
    "team_h": "home",
    "team_a": "away",
    "h_goals": "home_goals",
    "a_goals": "away_goals",
    "h_xg": "home_xg",
    "a_xg": "away_xg",
    "h_shot": "home_shots",
    "a_shot": "away_shots",
    "h_shotOnTarget": "home_sot",
    "a_shotOnTarget": "away_sot",
    "h_deep": "home_deep",
    "a_deep": "away_deep",
    "h_ppda": "home_ppda",
    "a_ppda": "away_ppda",
}


def main() -> None:
    if not LEGACY_MATCHES_CSV.exists():
        raise SystemExit(f"No se encuentra {LEGACY_MATCHES_CSV}. "
                         "Si ya migraste, el almacén está en data/matches.csv")

    df = pd.read_csv(LEGACY_MATCHES_CSV).rename(columns=RENOMBRES)
    df["match_id"] = "kaggle:" + df["match_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])

    canon = normalize(df)
    destino = save_store(canon, MATCHES_STORE)

    print(f"Migrados {len(canon)} partidos -> {destino}")
    print(f"  {canon['date'].min():%Y-%m-%d} a {canon['date'].max():%Y-%m-%d}")
    print(f"  ligas: {', '.join(sorted(canon['league'].unique()))}")
    print(f"  tamaño: {destino.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
