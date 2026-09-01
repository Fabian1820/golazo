"""Interfaz de línea de comandos.

    golazo train [--model M] [--until FECHA]
    golazo predict LOCAL VISITANTE [--kickoff FECHA] [--record]
    golazo models
    golazo verify
    golazo serve [--port 8080]

Instalado con `pip install -e .`, o sin instalar con `python -m golazo.cli ...`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .config import MODELS_DIR
from .training import DEFAULT_MODEL, MODELOS


def _cmd_train(args) -> int:
    from .training import train

    print(f"Entrenando '{args.model}' sobre todo el historial...")
    art = train(model=args.model, until=args.until, notes=args.notes)
    print(f"\n  'latest' -> {art.metadata.version_id}")
    return 0


def _cmd_predict(args) -> int:
    from .ledger import LedgerError, PredictionLedger
    from .service import PredictionService, UnknownTeamError

    try:
        servicio = PredictionService.load(args.version)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        pred = servicio.predict(args.home, args.away, league=args.league, kickoff=args.kickoff)
    except (UnknownTeamError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    salida = pred.to_dict()
    if args.record:
        try:
            salida["ledger_hash"] = PredictionLedger().append(pred).hash
        except LedgerError as exc:
            print(f"aviso: no se registró — {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(salida, indent=2, ensure_ascii=False))
        return 0

    p = pred.probabilities
    print(f"\n{pred.home}  vs  {pred.away}   ({pred.league})")
    print(f"{'─' * 52}")
    for etiqueta, clave in ((f"Gana {pred.home}", "H"), ("Empate", "D"), (f"Gana {pred.away}", "A")):
        barra = "█" * round(p[clave] * 34)
        print(f"  {etiqueta:<26}{p[clave]:>6.1%}  {barra}")
    if pred.markets:
        eg = pred.markets["expected_goals"]
        print(f"\n  Goles esperados   {eg['home']:.2f} – {eg['away']:.2f}   (total {eg['total']:.2f})")
        print(f"  Más de 2.5        {pred.markets['over_under']['2.5']['over']:>6.1%}")
        print(f"  Ambos marcan      {pred.markets['both_teams_score']['yes']:>6.1%}")
        top = ", ".join(f"{s['score']} ({s['probability']:.1%})" for s in pred.markets["top_scorelines"][:3])
        print(f"  Marcadores        {top}")
    for w in pred.warnings:
        print(f"\n  aviso: {w}")
    print(f"\n  modelo {pred.model_version}, datos hasta {pred.trained_through}")
    return 0


def _cmd_models(args) -> int:
    from .artifacts import ModelArtifact

    versiones = ModelArtifact.list_versions()
    if not versiones:
        print(f"No hay modelos en {MODELS_DIR}. Ejecuta: golazo train")
        return 1

    puntero = (MODELS_DIR / "latest")
    actual = puntero.read_text(encoding="utf-8").strip() if puntero.exists() else None
    for v in versiones:
        art = ModelArtifact.load(v)
        m = art.metadata
        marca = "*" if v == actual else " "
        rps = f"RPS {m.backtest['rps']:.4f}" if "rps" in m.backtest else "sin backtest"
        print(f" {marca} {v}  {m.model_name:<18} {m.n_matches:>5} partidos hasta {m.train_end}  {rps}")
    print("\n  * = 'latest'")
    return 0


def _cmd_fetch(args) -> int:
    import logging

    from .refresh import current_season, describe, refresh

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    temporadas = args.seasons or [current_season()]
    resumen = refresh(seasons=temporadas, use_cache=args.cache,
                      include_fixtures=not args.no_fixtures)
    print(describe(resumen))
    return 0


def _cmd_forecast(args) -> int:
    from .forecast import describe, run

    resultado = run(horizon_days=args.horizon, record=not args.dry_run)
    print(describe(resultado))
    return 0


def _cmd_validate(args) -> int:
    from .data import load_matches
    from .validation import has_errors, report, validate

    hallazgos = validate(load_matches())
    print(report(hallazgos))
    return 1 if has_errors(hallazgos) else 0


def _cmd_verify(args) -> int:
    from .ledger import PredictionLedger

    ok, mensaje = PredictionLedger().verify()
    print(mensaje)
    return 0 if ok else 1


def _cmd_serve(args) -> int:
    import logging

    from .web import create_app

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    create_app().run(host=args.host, port=args.port, debug=args.debug)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="golazo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="entrena y persiste el modelo")
    t.add_argument("--model", choices=sorted(MODELOS), default=DEFAULT_MODEL)
    t.add_argument("--until", help="entrenar sólo con partidos anteriores a esta fecha")
    t.add_argument("--notes", default="")
    t.set_defaults(func=_cmd_train)

    p = sub.add_parser("predict", help="predice un enfrentamiento")
    p.add_argument("home")
    p.add_argument("away")
    p.add_argument("--league")
    p.add_argument("--kickoff")
    p.add_argument("--version", default="latest")
    p.add_argument("--record", action="store_true", help="anotar en el registro de predicciones")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_predict)

    f = sub.add_parser("fetch", help="refresca el almacén desde la fuente viva")
    f.add_argument("--seasons", type=int, nargs="+", help="por defecto, la temporada en curso")
    f.add_argument("--cache", action="store_true", help="reutilizar respuestas cacheadas")
    f.add_argument("--no-fixtures", action="store_true", help="no traer el calendario")
    f.set_defaults(func=_cmd_fetch)

    fc = sub.add_parser("forecast", help="predice la próxima jornada y la firma")
    fc.add_argument("--horizon", type=int, default=10, help="días hacia adelante (por defecto 10)")
    fc.add_argument("--dry-run", action="store_true", help="no escribir en el registro")
    fc.set_defaults(func=_cmd_forecast)

    m = sub.add_parser("models", help="lista los modelos entrenados")
    m.set_defaults(func=_cmd_models)

    d = sub.add_parser("validate", help="valida la integridad de los datos")
    d.set_defaults(func=_cmd_validate)

    sc = sub.add_parser("score", help="puntúa las predicciones ya resueltas")
    sc.set_defaults(func=_cmd_score)

    v = sub.add_parser("verify", help="verifica la integridad del registro")
    v.set_defaults(func=_cmd_verify)

    s = sub.add_parser("serve", help="levanta la API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--debug", action="store_true")
    s.set_defaults(func=_cmd_serve)

    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


# Puntos de entrada declarados en pyproject.toml
def train_main() -> int:
    return main(["train", *sys.argv[1:]])


def predict_main() -> int:
    return main(["predict", *sys.argv[1:]])


def _cmd_score(args) -> int:
    return score_main()


def score_main() -> int:
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).resolve().parent.parent / "scripts" / "score_ledger.py"),
                   run_name="__main__")
    return 0


def backtest_main() -> int:
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).resolve().parent.parent / "scripts" / "run_backtest.py"),
                   run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
