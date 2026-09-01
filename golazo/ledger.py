"""Registro append-only y a prueba de manipulación de predicciones emitidas.

Un modelo sólo es creíble si sus predicciones quedan fijadas **antes** de que se
juegue el partido. Cualquiera puede publicar aciertos elegidos a posteriori.

Dos garantías:

1. **Anterioridad.** Una predicción cuyo `created_at` no precede al saque
   inicial se rechaza. No hay forma de registrar un pronóstico de un partido ya
   jugado.

2. **Inmutabilidad.** Cada registro incluye el hash del anterior, formando una
   cadena. Alterar o borrar cualquier registro pasado rompe la verificación de
   todos los posteriores, y `verify()` lo detecta.

El fichero nunca se modifica: la puntuación posterior se calcula al vuelo
cruzando con los resultados reales, y se escribe aparte.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import PREDICTIONS_DB

GENESIS = "0" * 64


class LedgerError(RuntimeError):
    """Violación de una de las garantías del registro."""


@dataclass(frozen=True)
class Record:
    created_at: str
    kickoff: str
    league: str
    home: str
    away: str
    probabilities: dict
    model_version: str
    prev_hash: str
    hash: str

    @staticmethod
    def compute_hash(payload: dict) -> str:
        material = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


class PredictionLedger:
    """Registro en JSONL, una predicción por línea."""

    def __init__(self, path: Path = PREDICTIONS_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- lectura ----------------------------------------------------------

    def __iter__(self) -> Iterator[Record]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Record(**json.loads(line))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise LedgerError(f"Registro corrupto en la línea {n}: {exc}") from exc

    def records(self) -> list[Record]:
        return list(self)

    def last_hash(self) -> str:
        ultimo = GENESIS
        for r in self:
            ultimo = r.hash
        return ultimo

    def __len__(self) -> int:
        return sum(1 for _ in self)

    # -- escritura --------------------------------------------------------

    def append(self, prediction, created_at: datetime | None = None) -> Record:
        """Registra una predicción. Falla si el partido ya empezó."""
        ahora = created_at or datetime.now(timezone.utc)
        if ahora.tzinfo is None:
            ahora = ahora.replace(tzinfo=timezone.utc)

        kickoff = datetime.fromisoformat(prediction.kickoff)
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)

        if ahora >= kickoff:
            raise LedgerError(
                f"No se puede registrar una predicción de un partido ya comenzado "
                f"(saque {kickoff.isoformat()}, ahora {ahora.isoformat()})")

        prev = self.last_hash()
        payload = {
            "created_at": ahora.isoformat(timespec="seconds"),
            "kickoff": kickoff.isoformat(),
            "league": prediction.league,
            "home": prediction.home,
            "away": prediction.away,
            "probabilities": {k: round(float(v), 6) for k, v in prediction.probabilities.items()},
            "model_version": prediction.model_version,
            "prev_hash": prev,
        }
        record = Record(**payload, hash=Record.compute_hash(payload))

        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
        return record

    # -- verificación -----------------------------------------------------

    def verify(self) -> tuple:
        """Comprueba la cadena completa. Devuelve (ok, mensaje)."""
        prev = GENESIS
        n = 0
        for n, r in enumerate(self, 1):
            if r.prev_hash != prev:
                return False, (f"Cadena rota en el registro {n} ({r.home} vs {r.away}): "
                               f"esperaba prev_hash {prev[:12]}…, encontrado {r.prev_hash[:12]}…")
            payload = {k: v for k, v in asdict(r).items() if k != "hash"}
            if Record.compute_hash(payload) != r.hash:
                return False, f"Registro {n} alterado ({r.home} vs {r.away}): el hash no coincide"
            if r.created_at >= r.kickoff:
                return False, f"Registro {n} posterior al saque inicial ({r.home} vs {r.away})"
            prev = r.hash
        return True, f"Cadena íntegra: {n} predicciones verificadas"
