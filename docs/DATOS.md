# Datos: procedencia, esquema y cómo refrescarlos

## De dónde salen

| qué | fuente | cómo |
|---|---|---|
| Primeras divisiones 2018-08 a 2023-06 | [Kaggle: `arkadiuszkaros/soccer-data-of-teams-players-and-shots-2018-2023`](https://www.kaggle.com/datasets/arkadiuszkaros/soccer-data-of-teams-players-and-shots-2018-2023) | volcado inicial, migrado una vez con `scripts/migrate_store.py` |
| Primeras divisiones 2023-07 en adelante | [Understat](https://understat.com) | `golazo fetch`, a diario |
| Segundas divisiones | [football-data.co.uk](https://www.football-data.co.uk) | `golazo fetch`, a diario |
| Cuotas de cierre | [football-data.co.uk](https://www.football-data.co.uk) | `golazo odds` |

Los dos primeros tramos son en última instancia datos de Understat, así que las
definiciones de xG, PPDA y *deep* son homogéneas en toda la serie.

### Por qué dos fuentes

Understat cubre las cinco primeras divisiones con xG, pero no las segundas. Y
ahí están precisamente los equipos que aparecen en los feeds de liga por
partidos de copa. football-data.co.uk cubre Championship, LaLiga 2, 2.
Bundesliga, Serie B y Ligue 2, además de publicar tiros y cuotas.

Las **primeras divisiones no se descargan de football-data**: duplicarían
partidos que ya vienen de Understat, que además trae xG.

### Nombres de equipo

Cada sitio nombra a los equipos a su manera y la tabla de equivalencias
(`golazo/sources/footballdata.py`) es **explícita a propósito**. Al generarla por
similitud de texto propuso `Paris SG` → `Paris FC`, `West Brom` → `West Ham`,
`Spal` → `Spezia`, `Lecco` → `Lecce` y `Pau FC` → `Paris FC`: clubes distintos
en todos los casos, y varios existen por separado en el almacén. Un nombre sin
equivalencia conocida detiene la descarga en lugar de adivinarse.

Los filiales (`Celta B`, `Sociedad B`, `Villarreal B`) son equipos aparte, no
variantes de nombre.

**Licencia.** El dataset de Kaggle y Understat tienen sus propias condiciones de
uso. Este repositorio es un proyecto educativo y no redistribuye los volcados
originales: sólo versiona el almacén derivado con los campos necesarios para
modelar. Si vas a darle uso comercial, revisa los términos de ambas fuentes.

## El almacén

Todo vive en ficheros versionados en git:

```
data/matches.csv     ~3 MB, 22.000 partidos, diez divisiones
data/odds.csv        ~1 MB, 14.000 partidos con cuota de cierre
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

Se atacó por dos vías:

1. **Cubrir las segundas divisiones** con football-data.co.uk, para que esos
   equipos tengan historial y valoración Elo reales. Los rechazos por historial
   insuficiente bajaron del 14,2% al 0%.
2. **Ajustar Dixon-Coles por país**, no por liga (ver `GRUPOS` en
   `golazo/models/dixon_coles.py`). Con primera y segunda en el mismo grupo de
   valoraciones, un equipo de segunda ya no hereda el promedio de primera. Hull
   contra el Aston Villa pasó de 75% a 17%.

`PredictionService` mantiene el mínimo de 10 partidos por equipo
(`InsufficientHistoryError`) como red de seguridad para casos nuevos.

**Lo que sigue sin resolverse del todo:** la fuerza *relativa entre divisiones*
se estima sobre todo a partir de ascensos y descensos, y con decaimiento
temporal esa evidencia pesa poco. Los pronósticos entre categorías son
razonables pero probablemente conservadores. Corregirlo bien pide un término
explícito de fuerza de división en el modelo.

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
