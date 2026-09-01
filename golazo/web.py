"""API HTTP.

Diferencias con el `main.py` original:

* no entrena nada al importar: carga un artefacto ya entrenado y versionado;
* sin rutas absolutas;
* toda respuesta lleva la versión del modelo y hasta qué fecha se entrenó;
* los errores devuelven JSON con código adecuado, no una traza de 500.
"""
from __future__ import annotations

import logging

from flask import Flask, jsonify, request, send_from_directory

from .config import ROOT
from .ledger import LedgerError, PredictionLedger
from .service import PredictionService, UnknownTeamError

log = logging.getLogger(__name__)

STATIC_DIR = ROOT / "web"


def create_app(service: PredictionService = None, ledger: PredictionLedger = None,
               record: bool = True) -> Flask:
    """Factory. Inyectar `service` permite testear sin tocar disco."""
    app = Flask(__name__, static_folder=None)
    app.config["RECORD_PREDICTIONS"] = record

    if service is None:
        log.info("Cargando artefacto de modelo...")
        service = PredictionService.load()
        log.info("Modelo %s listo", service.artifact.metadata.version_id)
    app.extensions["service"] = service
    app.extensions["ledger"] = ledger if ledger is not None else PredictionLedger()

    # -- errores ----------------------------------------------------------

    @app.errorhandler(UnknownTeamError)
    def _unknown_team(exc):
        return jsonify({"error": "equipo_desconocido", "detail": str(exc)}), 404

    @app.errorhandler(ValueError)
    def _bad_value(exc):
        return jsonify({"error": "peticion_invalida", "detail": str(exc)}), 400

    @app.errorhandler(404)
    def _not_found(_):
        return jsonify({"error": "no_encontrado"}), 404

    @app.errorhandler(500)
    def _server_error(exc):
        log.exception("Error no controlado")
        return jsonify({"error": "error_interno"}), 500

    # -- frontend ---------------------------------------------------------

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:filename>")
    def static_files(filename):
        return send_from_directory(STATIC_DIR, filename)

    # -- metadatos --------------------------------------------------------

    @app.get("/api/health")
    def health():
        meta = app.extensions["service"].artifact.metadata
        return jsonify({"status": "ok", "model_version": meta.version_id,
                        "trained_through": meta.train_end})

    @app.get("/api/model")
    def model_info():
        meta = app.extensions["service"].artifact.metadata
        return jsonify({
            "version": meta.version_id,
            "name": meta.model_name,
            "trained_at": meta.trained_at,
            "train_start": meta.train_start,
            "train_end": meta.train_end,
            "n_matches": meta.n_matches,
            "leagues": meta.leagues,
            "git_sha": meta.git_sha,
            "backtest": meta.backtest,
            "notes": meta.notes,
        })

    # -- catálogo ---------------------------------------------------------

    @app.get("/api/leagues")
    def leagues():
        return jsonify(app.extensions["service"].leagues)

    @app.get("/api/teams/<league>")
    def teams(league):
        return jsonify(app.extensions["service"].teams(league))

    # -- predicción -------------------------------------------------------

    @app.post("/api/predict")
    def predict():
        body = request.get_json(silent=True) or {}
        home, away = body.get("home"), body.get("away")
        if not home or not away:
            raise ValueError("Se requieren 'home' y 'away'")

        pred = app.extensions["service"].predict(
            home=home, away=away, league=body.get("league"), kickoff=body.get("kickoff"))

        payload = pred.to_dict()
        if app.config["RECORD_PREDICTIONS"]:
            try:
                registro = app.extensions["ledger"].append(pred)
                payload["ledger_hash"] = registro.hash
            except LedgerError as exc:
                # Registrar es opcional: no debe tumbar la predicción.
                payload["ledger_error"] = str(exc)
        return jsonify(payload)

    # -- historial --------------------------------------------------------

    @app.get("/api/track-record")
    def track_record():
        ledger = app.extensions["ledger"]
        ok, mensaje = ledger.verify()
        return jsonify({"verified": ok, "detail": mensaje, "n_predictions": len(ledger)})

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import os

    app = create_app()
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8080")),
            debug=os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"))


if __name__ == "__main__":
    main()
