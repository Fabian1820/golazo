"""Rutas y constantes del proyecto. Nada de rutas absolutas en el código."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LEGACY_DATA_DIR = ROOT / "src" / "soccer"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
PREDICTIONS_DB = ROOT / "predictions" / "predictions.jsonl"

# Almacén canónico: todos los partidos conocidos, de cualquier fuente.
MATCHES_STORE = DATA_DIR / "matches.csv"

# Cuotas de cierre del mercado. No son un hecho del partido sino la predicción
# de un tercero, así que viven aparte del almacén de partidos.
ODDS_STORE = DATA_DIR / "odds.csv"

# Volcados originales de Kaggle. Sólo los usa scripts/migrate_store.py, una vez.
LEGACY_MATCHES_CSV = LEGACY_DATA_DIR / "Matches.csv"
LEGACY_CLUBELO_CSV = LEGACY_DATA_DIR / "ClubElo.csv"

# Caché de respuestas de fuentes vivas.
CACHE_DIR = ROOT / ".cache" / "sources"

# Ligas que el producto realmente pronostica. Las segundas divisiones se
# incorporan al ENTRENAMIENTO —dan valoración real a los rivales de copa— pero
# no aparecen en el calendario: cero de 1.653 partidos anunciados. Evaluar
# sobre ellas mide algo que nunca se sirve.
#
# El backtest entrena con todo y evalúa aquí. Los informes publican también la
# cifra sin restringir, para que la diferencia quede a la vista.
SERVED_LEAGUES = ("EPL", "La liga", "Bundesliga", "Serie A", "Ligue 1")

# Resultados 1X2, en orden fijo. Este orden es un contrato: las métricas
# ordinales (RPS) dependen de que H < D < A.
OUTCOMES = ("H", "D", "A")
