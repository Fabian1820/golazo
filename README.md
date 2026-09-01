# Golazo

Predicción calibrada de resultados de fútbol para las cinco grandes ligas
europeas, con backtest reproducible y un historial de pronósticos firmados
antes de cada partido.

```bash
make install
make serve      # http://127.0.0.1:8080
```

---

## El resultado que importa

Backtest walk-forward sobre **12.553 partidos** que ningún modelo vio al
entrenar, de 2019-08 a 2026-08:

| modelo | RPS | ECE | acierto | mejora sobre la tasa base |
|---|---|---|---|---|
| gradient_boosting | 0.2016 | 0.0095 | 52.3% | +12.7% |
| elo_logistico | 0.2019 | 0.0113 | 52.4% | +12.6% |
| **dixon_coles** *(el que se sirve)* | **0.2026** | **0.0085** | 52.2% | **+12.3%** |
| tasa base (43/25/31 histórico) | 0.2309 | — | 43.1% | 0.0% |
| 1/3 a cada resultado | 0.2357 | — | 43.1% | −2.1% |

Un ECE de 0.0085 significa que cuando el modelo dice 25%, ocurre el 26%; cuando
dice 64%, ocurre el 67%. Está calibrado en todo el rango.

Detalle completo, significancia estadística y curvas de fiabilidad en
[reports/README.md](reports/README.md).

### Por qué se reescribió el proyecto

El predictor original entrenaba con estadísticas **del mismo partido** que
intentaba predecir (tiros, tiros a puerta, ppda). Al predecir de verdad las
sustituía por las del partido anterior, así que aprendía de una señal 4,4 veces
más fuerte que la que después recibía.

Medido honestamente obtiene **RPS 0.2558: peor que responder 1/3 a cada
resultado**, y la diferencia es estadísticamente significativa. Sigue en el
repositorio como `golazo/models/legacy.py` y entra en el backtest, para que la
comparación sea reproducible y no una afirmación.

---

## Cómo funciona

```
golazo fetch      Understat -> data/matches.csv        (idempotente)
golazo train      historial -> models/<versión>/       (artefacto versionado)
golazo forecast   calendario -> predictions/*.jsonl    (firmado antes del saque)
golazo score      resultados -> reports/track_record.md
```

Ese ciclo corre solo a diario en [`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml).

### Las tres reglas de diseño

**1. Una sola definición de las features.** `FeatureBuilder.emit` es el mismo
método en entrenamiento y en producción. Tener dos implementaciones es
exactamente cómo aparece el desajuste que hundió al modelo original; aquí es
imposible por construcción, y hay tests que lo comprueban.

**2. Nada que no se conozca antes del saque inicial.** El recorrido es
cronológico y cada partido sólo ve el historial anterior. El Elo se calcula
igual: se lee antes de incorporar el resultado. `tests/test_features_leakage.py`
y `tests/test_elo.py` fallan si eso se rompe.

**3. No se publica lo que no se puede respaldar.** Los feeds de liga de
Understat incluyen partidos de copa, que traen equipos de segunda con dos o tres
encuentros. El servicio exige 10 partidos de historial mínimo y se niega a
predecir por debajo, diciendo por qué. Sin esa guardia el sistema daba *Hull al
75% frente al Aston Villa*.

### Un modelo, todos los mercados

Se sirve Dixon-Coles aunque no encabece la tabla: los tres primeros son
[estadísticamente indistinguibles](reports/README.md#significancia-estadística),
y sólo él produce la distribución conjunta de marcadores. 1X2, over/under, ambos
marcan, hándicap y marcador exacto salen todos de la misma matriz, así que no
pueden contradecirse entre sí.

---

## El historial

Un backtest lo diseña uno mismo y siempre se puede ajustar hasta que salga bien.
Por eso cada pronóstico se firma en un registro append-only y encadenado por
hash, con dos garantías:

- **Anterioridad**: un pronóstico cuyo `created_at` no precede al saque inicial
  se rechaza. No se puede registrar un partido ya jugado.
- **Inmutabilidad**: cada registro incluye el hash del anterior. Alterar o
  borrar cualquiera rompe la verificación de todos los posteriores.

```bash
golazo verify     # Cadena íntegra: 40 predicciones verificadas
```

Los errores tampoco se ocultan: las primeras 46 predicciones se descartaron
enteras por el fallo de los partidos de copa y quedan archivadas con su motivo
en [`predictions/descartadas/`](predictions/descartadas/README.md). Quedarse con
las que salieron bien sería justo la manipulación que el registro existe para
impedir.

El historial vivo estará en `reports/track_record.md` en cuanto se resuelvan los
primeros partidos.

---

## Uso

```bash
golazo predict Liverpool Everton              # un enfrentamiento
golazo forecast --horizon 7                   # la próxima jornada
golazo fetch --seasons 2026                   # refrescar datos
golazo validate                               # integridad de los datos
golazo models                                 # versiones entrenadas
```

API HTTP: `GET /api/health`, `/api/model`, `/api/leagues`, `/api/teams/<liga>`
y `POST /api/predict`. Toda respuesta lleva la versión del modelo y hasta qué
fecha se entrenó.

```bash
make test        # 144 tests
make backtest    # ~6 min -> reports/
make docker
```

---

## Datos

16.037 partidos en `data/matches.csv` (1,9 MB, versionado): Kaggle para
2018-2023, Understat en vivo desde ahí. Esquema, licencias y cómo refrescar en
[docs/DATOS.md](docs/DATOS.md).

Los volcados originales (~78 MB) ya no se versionan: sólo dos de sus siete
ficheros se usaban.

> ⚠️ **Este repositorio tuvo un token de la API de Kaggle en la historia de git
> desde el primer commit.** Si lo has clonado, el token sigue ahí. Ver
> [docs/SEGURIDAD.md](docs/SEGURIDAD.md) para revocarlo y purgar la historia.

---

## Limitaciones

- **No se ha comparado con las cuotas del mercado.** Batir a las casas de
  apuestas es un problema distinto y mucho más difícil; nada aquí sugiere que
  este modelo lo consiga. Haría falta un histórico de cuotas de cierre que el
  proyecto todavía no incorpora.
- **Sólo cinco ligas domésticas.** Sin competiciones europeas ni selecciones.
- **Sin información de plantilla.** No conoce lesiones, sanciones ni rotaciones,
  que son parte grande de lo que queda sin explicar.
- **Proyecto educativo.** No es consejo para apostar.

## Estructura

```
golazo/           paquete: datos, features, modelos, servicio, registro, API
  sources/        fuentes intercambiables tras un esquema canónico
  models/         dixon_coles · elo_logistico · gradient_boosting · legacy
scripts/          backtest, entrenamiento, significancia, migración, purga
tests/            144 tests
web/              frontend
reports/          resultados del backtest y curvas de calibración
docs/             DATOS.md · SEGURIDAD.md
```

Metodología y decisiones de diseño en [MODELADO.md](MODELADO.md).

## Licencia

MIT.
