"""Fuentes de datos de partidos.

Todas producen el mismo esquema canónico, así que el resto del proyecto no
sabe —ni necesita saber— de dónde vienen los partidos.
"""
from .base import CANONICAL_COLUMNS, MatchSource, normalize
from .understat import UnderstatSource

__all__ = ["MatchSource", "CANONICAL_COLUMNS", "normalize", "UnderstatSource"]
