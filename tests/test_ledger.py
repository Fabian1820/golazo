"""El registro sólo vale si sus dos garantías son inviolables.

1. No se puede registrar una predicción de un partido ya comenzado.
2. No se puede alterar ni borrar un registro pasado sin que se detecte.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from golazo.ledger import GENESIS, LedgerError, PredictionLedger


class FakePrediction:
    def __init__(self, home="A", away="B", kickoff="2030-01-01T15:00:00",
                 probabilities=None, version="v1"):
        self.home = home
        self.away = away
        self.league = "EPL"
        self.kickoff = kickoff
        self.probabilities = probabilities or {"H": 0.5, "D": 0.3, "A": 0.2}
        self.model_version = version


@pytest.fixture
def ledger(tmp_path):
    return PredictionLedger(tmp_path / "pred.jsonl")


ANTES = datetime(2029, 1, 1, tzinfo=timezone.utc)


def test_registro_vacio_es_valido(ledger):
    ok, _ = ledger.verify()
    assert ok
    assert len(ledger) == 0
    assert ledger.last_hash() == GENESIS


def test_append_y_lectura(ledger):
    r = ledger.append(FakePrediction(), created_at=ANTES)
    assert len(ledger) == 1
    assert r.prev_hash == GENESIS
    assert len(r.hash) == 64
    guardado = ledger.records()[0]
    assert guardado.home == "A"
    assert guardado.probabilities["H"] == 0.5


def test_los_registros_se_encadenan(ledger):
    r1 = ledger.append(FakePrediction(home="A"), created_at=ANTES)
    r2 = ledger.append(FakePrediction(home="C"), created_at=ANTES)
    r3 = ledger.append(FakePrediction(home="E"), created_at=ANTES)
    assert r2.prev_hash == r1.hash
    assert r3.prev_hash == r2.hash
    ok, mensaje = ledger.verify()
    assert ok, mensaje
    assert "3" in mensaje


# --- garantía 1: anterioridad ---------------------------------------------


def test_rechaza_prediccion_de_partido_ya_jugado(ledger):
    p = FakePrediction(kickoff="2020-01-01T15:00:00")
    with pytest.raises(LedgerError, match="ya comenzado"):
        ledger.append(p, created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))


def test_rechaza_prediccion_justo_despues_del_saque(ledger):
    kickoff = datetime(2030, 5, 1, 15, 0, tzinfo=timezone.utc)
    p = FakePrediction(kickoff=kickoff.isoformat())
    with pytest.raises(LedgerError):
        ledger.append(p, created_at=kickoff + timedelta(seconds=1))
    # Un segundo antes sí es válido.
    assert ledger.append(p, created_at=kickoff - timedelta(seconds=1))


def test_nada_se_escribe_cuando_se_rechaza(ledger):
    ledger.append(FakePrediction(), created_at=ANTES)
    with pytest.raises(LedgerError):
        ledger.append(FakePrediction(kickoff="2020-01-01T15:00:00"),
                      created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert len(ledger) == 1


# --- garantía 2: inmutabilidad --------------------------------------------


def test_detecta_alteracion_de_un_registro(ledger):
    ledger.append(FakePrediction(home="A"), created_at=ANTES)
    ledger.append(FakePrediction(home="C"), created_at=ANTES)

    lineas = ledger.path.read_text().strip().split("\n")
    d = json.loads(lineas[0])
    d["probabilities"]["H"] = 0.99          # cambiar la predicción a posteriori
    lineas[0] = json.dumps(d, sort_keys=True)
    ledger.path.write_text("\n".join(lineas) + "\n")

    ok, motivo = ledger.verify()
    assert not ok
    assert "alterado" in motivo


def test_detecta_borrado_de_un_registro(ledger):
    for h in "ACE":
        ledger.append(FakePrediction(home=h), created_at=ANTES)

    lineas = ledger.path.read_text().strip().split("\n")
    del lineas[1]                            # borrar el del medio
    ledger.path.write_text("\n".join(lineas) + "\n")

    ok, motivo = ledger.verify()
    assert not ok
    assert "Cadena rota" in motivo


def test_detecta_reordenacion(ledger):
    for h in "ACE":
        ledger.append(FakePrediction(home=h), created_at=ANTES)

    lineas = ledger.path.read_text().strip().split("\n")
    lineas[0], lineas[1] = lineas[1], lineas[0]
    ledger.path.write_text("\n".join(lineas) + "\n")

    assert not ledger.verify()[0]


def test_detecta_registro_insertado(ledger):
    ledger.append(FakePrediction(home="A"), created_at=ANTES)
    lineas = ledger.path.read_text().strip().split("\n")

    falso = json.loads(lineas[0])
    falso["home"] = "INVENTADO"
    lineas.append(json.dumps(falso, sort_keys=True))
    ledger.path.write_text("\n".join(lineas) + "\n")

    assert not ledger.verify()[0]


def test_fichero_corrupto_da_error_claro(ledger):
    ledger.append(FakePrediction(), created_at=ANTES)
    ledger.path.write_text(ledger.path.read_text() + "esto no es json\n")
    with pytest.raises(LedgerError, match="línea 2"):
        ledger.records()
