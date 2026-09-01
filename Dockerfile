# Imagen de servicio. El modelo se entrena durante el build para que el
# contenedor arranque listo: sin entrenar en el import, sin descargas en runtime.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero: esta capa se cachea entre builds.
COPY pyproject.toml README.md ./
COPY golazo ./golazo
RUN pip install --upgrade pip && pip install ".[web]" gunicorn

# Datos y frontend. El almacén son ~2 MB: la imagen no descarga nada al arrancar.
COPY data ./data
COPY web ./web
COPY reports/metrics.csv ./reports/metrics.csv

# Validar los datos y entrenar. Si los datos están mal, el build falla aquí y
# no en producción.
RUN python -m golazo.cli validate && python -m golazo.cli train

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=4s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/api/health').status==200 else 1)"

# gunicorn, no el servidor de desarrollo de Flask.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "60", \
     "golazo.web:create_app()"]
