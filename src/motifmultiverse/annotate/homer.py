"""HOMER precomputed-result adapter."""
from __future__ import annotations

from .base import ConfiguredAnnotationBackend

__all__ = ["HomerBackend"]


class HomerBackend(ConfiguredAnnotationBackend):
    """Adapt a versioned ``homer`` database-result section."""

    def __init__(self, database_path: str):
        super().__init__("homer", database_path)
