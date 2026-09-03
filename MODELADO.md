# Arquitectura y metodología

Guía del paquete `golazo/`. Los resultados del backtest están en
[reports/README.md](reports/README.md).

## Por qué existe esto

El predictor original entrenaba con estadísticas del propio partido que
intentaba predecir. Medido con un backtest honesto resultó ser **peor que
asignar 1/3 a cada resultado**, y la diferencia es estadísticamente
significativa. Este paquete reconstruye el modelado sobre una base verificable,
lo conecta a datos vivos y firma cada pronóstico antes de que se juegue.

El código original se conserva como `golazo/models/legacy.py`, no por nostalgia
sino porque entra en el backtest como referencia: la comparación es explícita y
reproducible, no retórica.

## Uso

```bash
make install     # entorno + dependencias
make validate    # ¿los datos están bien?
make backtest    # ¿qué modelo es mejor, y es real la diferencia?
make train       # entrena y persiste el ganador
make serve       # API en http://127.0.0.1:8080
make test        # 144 tests
```

Todo lo anterior también está en el CLI: `golazo validate|train|predict|serve|models|verify`.

```bash
golazo predict Arsenal "Manchester City"
golazo predict Liverpool Everton --kickoff 2027-03-01 --record --json
```

## Estructura

```
golazo/
  config.py       Rutas. Ninguna absoluta, en ningún sitio.
  data.py         Carga el almacén. Separa columnas PRE y POST partido.
  store.py        Almacén canónico de partidos, con fusión idempotente.
  sources/        understat (1ª div + xG) · footballdata (2ª div + tiros + cuotas)
  elo.py          Elo propio, incremental y estrictamente pre-partido.
  odds.py         Cuotas de cierre del mercado, como referencia externa.
  validation.py   Integridad de datos: nulos, duplicados, rangos, coherencia.
  monitoring.py   Degradación silenciosa: datos congelados, ligas que desaparecen.
  features.py     FeatureBuilder: 44 features pre-partido, causales, una sola fuente.
  metrics.py      RPS, log-loss, Brier, ECE, curvas de calibración.
  backtest.py     Walk-forward con ventana expansiva.
  markets.py      1X2, over/under, BTTS, hándicap y marcadores desde una sola matriz.
  models/         baselines · dixon_coles · ml · legacy (réplica del original)
  training.py     Entrena sobre todo el historial y persiste.
  artifacts.py    Persistencia versionada con procedencia.
  service.py      Predicción de partidos no jugados. O(1) por consulta.
  ledger.py       Registro append-only encadenado por hash.
  web.py          API HTTP (factory de Flask).
  cli.py          Línea de comandos.

scripts/          run_backtest · significance · compare_market · review_model
                  tune_xi · tune_gb · train · score_ledger · migrate_store
web/              Frontend estático (sin build, sin dependencias)
tests/            180 tests
```

## Las cuatro reglas

### 1. Ninguna feature puede conocer el resultado de su propio partido

`data.py` separa `PRE_MATCH_COLS` de `POST_MATCH_COLS`. `FeatureBuilder` recorre
los partidos en orden y sólo incorpora un resultado *después* de emitir sus
features (`emit` es puro, `ingest` avanza el estado).

`tests/test_features_leakage.py` lo verifica **por comportamiento**, no por
inspección de nombres: altera el resultado de un partido y exige que sus propias
features no cambien, con contraprueba de que sí cambian las de los posteriores.

### 2. El entrenamiento y la producción usan el mismo código

La otra mitad del bug original era el desajuste: entrenaba con unas variables y
al predecir recibía otras. Aquí `build_features` (entrenamiento) y
`PredictionService` (producción) llaman al **mismo** `FeatureBuilder.emit`.

`tests/test_service.py` reconstruye el estado hasta cuatro puntos distintos del
historial y exige que las features emitidas coincidan **exactamente** con las que
produjo el entrenamiento.

### 3. El entrenamiento nunca ve el futuro

`tests/test_backtest.py` monta un modelo espía que registra los rangos de fechas
de cada reajuste y falla si el máximo del entrenamiento alcanza al mínimo de la
predicción.

### 4. Ninguna métrica se reporta sin comparación ni sin intervalo

Todo modelo se mide contra `tasa_base`, y las diferencias entre modelos pasan por
`scripts/significance.py` (bootstrap de bloques semanales) antes de llamarse
mejoras. Fue así como se estableció que el gradient boosting con 52 features **no
supera** a una regresión logística sobre una sola variable.

## Decisiones de diseño

### Qué modelo se sirve, y por qué

`gradient_boosting` (RPS 0.2030) y `elo_logistico` (0.2032) empatan
estadísticamente. Ambos superan a `dixon_coles` (0.2049) por 0.0019 — real pero
pequeño, un 0.9%.

Por defecto se sirve **Dixon-Coles**. A cambio de ese 0.9% entrega la
distribución conjunta de marcadores, y de ella salen 1X2, over/under, ambos
marcan, hándicap y marcador exacto **coherentes entre sí por construcción**: son
márgenes del mismo objeto, no cálculos independientes que puedan contradecirse.
Los otros dos modelos sólo producen 1X2.

Para servir únicamente 1X2, `--model elo_logistico` es mejor y más simple.

### El Elo

`ClubElo.csv` estaba en el repositorio sin usarse. Es la feature más informativa
del dataset (correlación 0.436 con la diferencia de goles) y es **previa** al
partido: el salto de Elo que entra a un partido no correlaciona con su resultado
(+0.004) mientras que el que sale sí (+0.668). El test
`test_el_elo_es_previo_al_partido` deja esa comprobación permanente.

### Artefactos versionados

Cada entrenamiento produce `models/<fecha>-<hash>/` con el modelo y sus
metadatos: rango de datos, número de partidos, SHA de git y **las métricas de
backtest que justifican esa elección**. El id depende de los datos y del código,
no del reloj: reentrenar con lo mismo produce el mismo id.

Toda predicción sale con su `model_version` y su `trained_through`.

### El registro de predicciones

`predictions/predictions.jsonl` es append-only y **encadenado por hash**. Dos
garantías:

- **Anterioridad**: una predicción cuyo `created_at` no precede al saque inicial
  se rechaza. No se puede registrar un pronóstico de un partido ya jugado.
- **Inmutabilidad**: cada registro incluye el hash del anterior. Alterar, borrar,
  reordenar o insertar rompe la cadena y `verify()` lo detecta.

Se versiona en git a propósito: su valor entero depende de ser auditable
públicamente. `scripts/score_ledger.py` lo puntúa contra los resultados reales
sin modificarlo nunca.

## Añadir un modelo

Hereda de `golazo.models.base.Model`, implementa `fit(train)` y
`predict_proba(test) -> (n, 3)` en el orden H, D, A, y añádelo al diccionario de
`scripts/run_backtest.py`. El backtest, las métricas y el test de significancia
funcionan sin más cambios. Si además expones `scoreline_matrix`, el servicio
publicará automáticamente todos los mercados derivados.

## Lo que falta

**El historial de predicciones apenas ha empezado.** Es el único resultado que
no se puede ajustar a posteriori, y necesita meses de partidos para significar
algo. No hay atajo: sólo lo llena el calendario.

**Sin alineaciones ni lesiones.** Se investigó la API pública de Fantasy Premier
League, que sí publica disponibilidad real (`status`, `chance_of_playing_next_round`).
No se integró por dos razones: sólo cubre la Premier, una de diez divisiones, y
**sólo da la foto actual, sin histórico**. Sin serie temporal no se puede
entrenar con ella, ni meterla en el backtest, ni demostrar que aporta. Añadirla
sería un ajuste a ojo disfrazado de feature.

**La fuerza entre divisiones es aproximada.** Ajustar Dixon-Coles por país
(`GRUPOS`) arregló los casos absurdos, pero la diferencia de nivel entre primera
y segunda se estima sobre todo desde ascensos y descensos, que el decaimiento
temporal degrada. Un término explícito de fuerza de división lo resolvería mejor.

**El ξ está validado pero no es el único parámetro a ojo.** Quedan la ventaja de
campo del Elo (65 puntos), su factor K (20) y la penalización al ascendido (40),
todos tomados de la convención.
