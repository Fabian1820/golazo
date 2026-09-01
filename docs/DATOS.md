# Datos: procedencia, esquema y cómo refrescarlos

## De dónde salen

| tramo | fuente | cómo |
|---|---|---|
| 2018-08 a 2023-06 | [Kaggle: `arkadiuszkaros/soccer-data-of-teams-players-and-shots-2018-2023`](https://www.kaggle.com/datasets/arkadiuszkaros/soccer-data-of-teams-players-and-shots-2018-2023) | volcado inicial, migrado una vez con `scripts/migrate_store.py` |
| 2023-07 en adelante | [Understat](https://understat.com) | `golazo fetch`, a diario |

Ambos tramos son en última instancia datos de Understat, así que las
definiciones de xG, PPDA y *deep* son homogéneas en toda la serie.

**Licencia.** El dataset de Kaggle y Understat tienen sus propias condiciones de
uso. Este repositorio es un proyecto educativo y no redistribuye los volcados
originales: sólo versiona el almacén derivado con los campos necesarios para
modelar. Si vas a darle uso comercial, revisa los términos de ambas fuentes.

## El almacén

Todo vive en un único fichero versionado en git:

```
data/matches.csv     ~2 MB, 16.000 partidos, cinco ligas
```

Es pequeño a propósito. Cualquiera clona el repositorio y todo funciona sin
descargar nada ni configurar credenciales.

### Esquema canónico

Definido en `golazo/sources/base.py`. Es la frontera del sistema: una fuente
nueva sólo tiene que producir estas columnas.

**Obligatorias** — `match_id`, `date`, `season`, `league`, `home`, `away`,
`home_goals`, `away_goals`.

**Opcionales** — `home_xg`, `away_xg`, `home_deep`, `away_deep`, `home_ppda`,
`away_ppda`, `home_shots`, `away_shots`, `home_sot`, `away_sot`.

Son opcionales porque no todas las fuentes las publican, y lo que falta viaja
como NaN en lugar de inventarse. `golazo validate` distingue las dos clases: un
nulo en una obligatoria es un error, en una opcional es un aviso.

El `match_id` lleva prefijo de fuente (`kaggle:10643`, `understat:26602`) para
que dos fuentes no puedan pisarse.

**El Elo no está en el esquema.** Se calcula en `golazo/elo.py` a partir de los
resultados, para que histórico y datos nuevos pasen por el mismo camino. Ver
[MODELADO.md](../MODELADO.md).

## Por qué las features no usan tiros

El endpoint de liga de Understat no publica tiros ni tiros a puerta. El volcado
de Kaggle sí los tiene, así que mantenerlos como feature significaría entrenar
con una variable que **no existe al predecir** — exactamente el desajuste que
invalidaba el modelo original.

Se quitaron del conjunto de features y se midió el coste en el backtest: **0.0007
de RPS**, menor que la diferencia entre modelos que ya sabemos que es ruido. A
cambio, entrenamiento y producción quedan sobre el mismo terreno.

Las columnas siguen en el esquema porque el tramo histórico las tiene y podrían
servir para otros análisis.

## Refrescar

```bash
golazo fetch                          # temporada en curso
golazo fetch --seasons 2024 2025      # temporadas concretas
golazo fetch --cache                  # reutiliza respuestas ya descargadas
```

Es idempotente: ejecutarlo dos veces no duplica nada, y un partido que pasa de
anunciado a jugado sustituye a su versión previa (`merge_sources` se queda con
la última aparición de cada `match_id`).

El cliente hace una petición por liga y temporada, con 1,5 s de pausa entre
ellas. Un refresco diario son cinco peticiones.

## Cuidado con los partidos de copa

Los feeds de liga de Understat **incluyen partidos de copa** etiquetados con la
liga doméstica. Eso mete equipos de divisiones inferiores con dos o tres
encuentros registrados.

No es una curiosidad: en el primer pronóstico real el sistema daba **Hull (2
partidos en el historial) al 75% frente al Aston Villa**, porque Dixon-Coles
valora a un equipo desconocido con el promedio de su liga.

`PredictionService` exige ahora un mínimo de 10 partidos por equipo y se niega a
emitir pronóstico por debajo (`InsufficientHistoryError`). Unos 236 de 1.658
partidos anunciados quedan fuera por este motivo, y el pronóstico dice
explícitamente cuáles y por qué.

## Recuperar los volcados originales

El proyecto sólo necesita `data/matches.csv`. Los volcados completos de Kaggle
(~78 MB) incluyen además datos por jugador y por disparo —`Rosters.csv`,
`Shots.csv`, `PlayersData.csv`— que no usa ningún modelo actual pero sirven para
trabajo a nivel de jugador o modelos de xG propios.

No se versionan. Para recuperarlos:

```bash
pip install kaggle
# Coloca tu token en ~/.kaggle/kaggle.json (nunca dentro del repositorio)
kaggle datasets download -d arkadiuszkaros/soccer-data-of-teams-players-and-shots-2018-2023 -p /tmp/golazo-bulk --unzip
```

> **Nunca guardes `kaggle.json` dentro del repositorio.** Este proyecto lo
> aprendió por las malas: el token estuvo en la historia de git desde el primer
> commit. Ver [`docs/SEGURIDAD.md`](SEGURIDAD.md).
