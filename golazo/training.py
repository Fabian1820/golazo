"""Entrenamiento del modelo final y su persistencia.

Vive en el paquete, no en `scripts/`, para que tanto el CLI como cualquier
automatización (cron, CI, un job de reentrenamiento) usen exactamente el mismo
camino.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .artifacts import ModelArtifact, build_metadata
from .config import MODELS_DIR, REPORTS_DIR
from .data import load_matches
from .features import build_features, feature_columns
from .models import DixonColes, EloLogisticModel, GradientBoostingModel

MODELOS = {
    "dixon_coles": DixonColes,
    "elo_logistico": EloLogisticModel,
    "gradient_boosting": GradientBoostingModel,
}

DEFAULT_MODEL = "dixon_coles"


def backtest_provenance(nombre: str, reports_dir: Path = REPORTS_DIR) -> dict:
    """Métricas del backtest que justifican la elección de este modelo.

    Adjuntarlas al artefacto permite responder «¿por qué este modelo?» sin
    depender de que alguien recuerde el informe.
    """
    ruta = Path(reports_dir) / "metrics.csv"
    if not ruta.exists():
        return {"warning": "sin backtest; ejecuta scripts/run_backtest.py"}
    tabla = pd.read_csv(ruta, index_col=0)
    if nombre not in tabla.index:
        return {"warning": f"'{nombre}' no aparece en {ruta.name}"}
    fila = tabla.loc[nombre]
    return {
        "source": ruta.name,
        "n_matches": int(fila["n"]),
        "rps": float(fila["rps"]),
        "log_loss": float(fila["log_loss"]),
        "brier": float(fila["brier"]),
        "ece": float(fila["ece"]),
        "accuracy": float(fila["accuracy"]),
        "rps_skill_vs_base_rate_pct": float(fila["rps_skill_%"]),
    }


def train(model: str = DEFAULT_MODEL, until: str | None = None,
          out: Path = MODELS_DIR, notes: str = "", verbose: bool = True) -> ModelArtifact:
    """Entrena sobre todo el historial disponible y persiste el artefacto."""
    if model not in MODELOS:
        raise ValueError(f"Modelo desconocido: {model}. Opciones: {sorted(MODELOS)}")

    df = build_features(load_matches())
    if until:
        df = df[df["date"] < pd.Timestamp(until)]
        if df.empty:
            raise ValueError(f"No hay partidos anteriores a {until}")

    cols = feature_columns(df)
    if verbose:
        print(f"  {len(df)} partidos · {df.date.min():%Y-%m-%d} a {df.date.max():%Y-%m-%d} "
              f"· {len(cols)} features")

    estimador = (GradientBoostingModel(feature_cols=cols) if model == "gradient_boosting"
                 else MODELOS[model]())
    estimador.fit(df)

    meta = build_metadata(
        model_name=model,
        train_df=df,
        feature_columns=cols if model == "gradient_boosting" else [],
        backtest=backtest_provenance(model),
        notes=notes,
    )
    artefacto = ModelArtifact(estimador, meta)
    destino = artefacto.save(Path(out))

    if verbose:
        print(f"\nArtefacto guardado: {destino}")
        print(f"  versión        {meta.version_id}")
        print(f"  entrenado con  {meta.n_matches} partidos hasta {meta.train_end}")
        print(f"  git            {meta.git_sha or 'desconocido'}")
        if "rps" in meta.backtest:
            print(f"  backtest       RPS {meta.backtest['rps']:.4f} · ECE {meta.backtest['ece']:.4f} "
                  f"· skill {meta.backtest['rps_skill_vs_base_rate_pct']:+.1f}%")
        else:
            print(f"  backtest       {meta.backtest.get('warning', '—')}")
    return artefacto
