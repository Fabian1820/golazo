.PHONY: help install test lint validate fetch backtest train forecast serve score verify pipeline docker clean

VENV := .venv
PY   := $(VENV)/bin/python

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(VENV):
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

install: $(VENV)  ## Instala el proyecto y sus dependencias de desarrollo
	$(PY) -m pip install -e ".[web,reports,dev]"

test: ## Ejecuta la suite completa
	$(PY) -m pytest

lint: ## Revisa el estilo
	$(PY) -m ruff check golazo scripts tests

validate: ## Valida la integridad de los datos
	$(PY) -m golazo.cli validate

fetch: ## Refresca el almacén desde Understat
	$(PY) -m golazo.cli fetch

backtest: ## Backtest walk-forward completo (~5 min) -> reports/
	$(PY) scripts/run_backtest.py
	$(PY) scripts/significance.py

train: ## Entrena y persiste el modelo -> models/
	$(PY) -m golazo.cli train

forecast: ## Predice la próxima jornada y la firma en el registro
	$(PY) -m golazo.cli forecast

pipeline: fetch train forecast score ## Ciclo completo: datos -> modelo -> pronóstico -> puntuación

serve: ## Levanta la API en http://127.0.0.1:8080
	$(PY) -m golazo.cli serve

score: ## Puntúa las predicciones registradas -> reports/track_record.md
	$(PY) scripts/score_ledger.py

verify: ## Verifica la integridad del registro de predicciones
	$(PY) -m golazo.cli verify

docker: ## Construye la imagen de servicio
	docker build -t golazo:latest .

clean: ## Borra artefactos generados (no toca el registro de predicciones)
	rm -rf models reports/backtest_predictions.csv .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
