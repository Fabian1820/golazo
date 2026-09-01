# Predicciones descartadas

Este directorio conserva registros retirados del historial vigente. Se guardan
en lugar de borrarse: un registro que presume de ser auditable no puede hacer
desaparecer sus errores.

## 2026-08-31-copa.jsonl — 46 predicciones

**Motivo.** Primer pronóstico real del sistema, emitido antes de detectar que
los feeds de liga de Understat incluyen partidos de copa. Esos partidos traen
equipos de divisiones inferiores con dos o tres encuentros registrados, y
Dixon-Coles valoraba a un equipo desconocido con el promedio de su liga. El
resultado eran pronósticos sin sentido, como Hull (segunda división, 2 partidos
en el historial) al 75% frente al Aston Villa.

**Corrección.** `PredictionService` ahora exige un mínimo de 10 partidos de
historial por equipo y se niega a emitir pronóstico por debajo de ese umbral
(`InsufficientHistoryError`). Ver `golazo/service.py`.

**Alcance.** Ninguna de estas predicciones llegó a publicarse. Se descartan
enteras, no selectivamente: quedarse con las que salieron bien sería
exactamente la manipulación que el registro existe para impedir.
