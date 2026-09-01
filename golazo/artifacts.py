"""Persistencia versionada de modelos.

Una predicción sin saber qué modelo la produjo, entrenado con qué datos y hasta
qué fecha, no es auditable. Cada artefacto guarda el modelo junto a los metadatos
que permiten reconstruir esa cadena.

Estructura en disco:

    models/
      20230604-a3f19c2b/
        model.joblib
        metadata.json
      latest -> 20230604-a3f19c2b   (fichero de texto con el id)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .config import MODELS_DIR


def _git_sha() -> str | None:
    """SHA del commit actual, si el proyecto está en un repo limpio de errores."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return None


@dataclass
class Metadata:
    """Todo lo necesario para reproducir y auditar un modelo entrenado."""

    model_name: str
    trained_at: str
    train_start: str
    train_end: str
    n_matches: int
    leagues: list[str]
    feature_columns: list[str] = field(default_factory=list)
    git_sha: str | None = None
    backtest: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def version_id(self) -> str:
        """Identificador estable: fecha de los datos + hash del contenido.

        Depende de los datos y del código, no del reloj: reentrenar dos veces
        con lo mismo produce el mismo id.
        """
        payload = json.dumps({
            "model": self.model_name,
            "train_end": self.train_end,
            "n": self.n_matches,
            "features": sorted(self.feature_columns),
            "git": self.git_sha,
        }, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
        return f"{self.train_end.replace('-', '')[:8]}-{digest}"


class ModelArtifact:
    """Un modelo entrenado más sus metadatos."""

    def __init__(self, model, metadata: Metadata):
        self.model = model
        self.metadata = metadata

    # -- escritura --------------------------------------------------------

    def save(self, root: Path = MODELS_DIR, make_latest: bool = True) -> Path:
        root = Path(root)
        target = root / self.metadata.version_id
        target.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, target / "model.joblib", compress=3)
        (target / "metadata.json").write_text(
            json.dumps(asdict(self.metadata), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if make_latest:
            (root / "latest").write_text(self.metadata.version_id, encoding="utf-8")
        return target

    # -- lectura ----------------------------------------------------------

    @classmethod
    def load(cls, version: str = "latest", root: Path = MODELS_DIR) -> ModelArtifact:
        root = Path(root)
        if version == "latest":
            pointer = root / "latest"
            if not pointer.exists():
                raise FileNotFoundError(
                    f"No hay ningún modelo entrenado en {root}. Ejecuta: python scripts/train.py"
                )
            version = pointer.read_text(encoding="utf-8").strip()

        target = root / version
        if not target.is_dir():
            raise FileNotFoundError(f"No existe el modelo '{version}' en {root}")

        model = joblib.load(target / "model.joblib")
        meta = Metadata(**json.loads((target / "metadata.json").read_text(encoding="utf-8")))
        return cls(model, meta)

    @staticmethod
    def list_versions(root: Path = MODELS_DIR) -> list[str]:
        root = Path(root)
        if not root.is_dir():
            return []
        return sorted((d.name for d in root.iterdir() if d.is_dir() and (d / "metadata.json").exists()),
                      reverse=True)

    def __repr__(self) -> str:
        m = self.metadata
        return (f"<ModelArtifact {m.version_id} · {m.model_name} · "
                f"{m.n_matches} partidos hasta {m.train_end}>")


def build_metadata(model_name: str, train_df, feature_columns=None, backtest=None, notes="") -> Metadata:
    """Construye los metadatos a partir del conjunto de entrenamiento."""
    return Metadata(
        model_name=model_name,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        train_start=str(train_df["date"].min().date()),
        train_end=str(train_df["date"].max().date()),
        n_matches=int(len(train_df)),
        leagues=sorted(train_df["league"].unique().tolist()),
        feature_columns=list(feature_columns or []),
        git_sha=_git_sha(),
        backtest=backtest or {},
        notes=notes,
    )
