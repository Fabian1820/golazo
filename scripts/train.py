#!/usr/bin/env python
"""Entrena el modelo final y lo persiste versionado.

    python scripts/train.py                       # Dixon-Coles (por defecto)
    python scripts/train.py --model elo_logistico
    python scripts/train.py --model gradient_boosting

Envoltorio de `golazo.cli train`, que es donde vive la lógica. Equivalente a:

    python -m golazo.cli train --model dixon_coles

Por qué Dixon-Coles por defecto
-------------------------------
En el backtest, `gradient_boosting` (RPS 0.2030) y `elo_logistico` (0.2032)
empatan estadísticamente y ambos superan a Dixon-Coles (0.2049) por un margen
real pero pequeño: 0.0019, un 0.9%.

A cambio de ese 0.9%, Dixon-Coles entrega la distribución conjunta completa de
marcadores. De ella salen 1X2, over/under, ambos marcan, hándicap y marcador
exacto, todos coherentes entre sí por construcción. Los otros dos sólo producen
1X2. Es una decisión de producto explícita.

Para servir únicamente 1X2, `--model elo_logistico` es mejor y más simple: gana
en log-loss, Brier, ECE y acierto usando una sola variable frente a las 52 del
gradient boosting.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from golazo.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
