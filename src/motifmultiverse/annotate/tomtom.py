"""TomTom precomputed-result adapter."""
from __future__ import annotations

from .base import ConfiguredAnnotationBackend

__all__ = ["TomTomBackend"]


class TomTomBackend(ConfiguredAnnotationBackend):
    """Adapt a versioned ``tomtom`` database-result section."""

    def __init__(self, database_path: str):
        super().__init__("tomtom", database_path)
