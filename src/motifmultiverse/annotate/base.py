"""Backend contracts and precomputed database-result adapter support."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from motifmultiverse.schema import MotifNode
from motifmultiverse.schema.annotation import AnnotationCandidate

__all__ = ["AnnotationBackend", "AnnotationBackendError", "ConfiguredAnnotationBackend"]


class AnnotationBackend(Protocol):
    """A source of non-adjudicating annotation candidates."""

    name: str
    version: str

    def annotate(self, nodes: Sequence[MotifNode]) -> Sequence[AnnotationCandidate]:
        raise NotImplementedError


class AnnotationBackendError(RuntimeError):
    """An optional backend could not provide a verifiable result."""


class ConfiguredAnnotationBackend:
    """Adapt a versioned, precomputed database-result section into candidates.

    The configuration is deliberately a result adapter rather than an invented
    score calculator: an annotation backend must preserve its source labels and
    cannot create occurrence-null quantities from database matches alone.
    """

    optional = True

    def __init__(self, name: str, database_path: str | Path):
        self.name = name
        self.database_path = Path(database_path)
        self.version = "UNVERIFIED"

    def _section(self) -> Mapping[str, Any]:
        try:
            text = self.database_path.read_text(encoding="utf-8")
            if self.database_path.suffix in {".yaml", ".yml"}:
                import yaml

                payload = yaml.safe_load(text)
            else:
                payload = json.loads(text)
        except (ImportError, OSError, ValueError) as exc:
            raise AnnotationBackendError(
                f"{self.name} database configuration could not be read: {exc}"
            ) from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get(self.name), Mapping):
            raise AnnotationBackendError(f"database configuration has no {self.name!r} section")
        section = payload[self.name]
        version = section.get("version")
        if not isinstance(version, str) or not version:
            raise AnnotationBackendError(f"{self.name} database configuration has no version")
        self.version = version
        return section

    def annotate(self, nodes: Sequence[MotifNode]) -> Sequence[AnnotationCandidate]:
        section = self._section()
        if section.get("error"):
            raise AnnotationBackendError(str(section["error"]))
        matches = section.get("matches", [])
        if not isinstance(matches, list):
            raise AnnotationBackendError(f"{self.name} matches must be a list")
        by_id = {node.node_id: node for node in nodes}
        candidates: list[AnnotationCandidate] = []
        for match in matches:
            if not isinstance(match, Mapping):
                raise AnnotationBackendError(f"{self.name} match must be a mapping")
            node_id = match.get("node_id")
            if node_id not in by_id:
                raise AnnotationBackendError(
                    f"{self.name} match names node {node_id!r}, absent from the registry"
                )
            node = by_id[node_id]
            if node.motif_length is None:
                raise AnnotationBackendError(
                    f"{self.name} cannot annotate {node_id!r}: motif_length is missing"
                )
            try:
                candidate = AnnotationCandidate.create(
                    node_id=node_id,
                    proposed_family_id=str(match["proposed_family_id"]), source=self.name,
                    source_version=self.version, matched_motif_id=str(match["matched_motif_id"]),
                    score=_optional_float(match.get("score")),
                    q_value=_optional_float(match.get("q_value")),
                    aligned_span=_optional_int(match.get("aligned_span")),
                    motif_length=node.motif_length,
                    trimmed_core_length=_declared_core_length(node),
                    seqlet_count=node.seqlet_count,
                    provenance={
                        "database_path": self.database_path.name,
                        "legacy_label": match.get("legacy_label", match["proposed_family_id"]),
                    },
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise AnnotationBackendError(f"invalid {self.name} match: {exc}") from exc
            candidates.append(candidate)
        return candidates


def _declared_core_length(node: MotifNode) -> int | None:
    """The width of the node's declared trimmed core, or None if it declares none.

    Derived here because this is where the node is in hand. The core is the only
    meaningful width of a tfmodisco-lite pattern -- ``motif_length`` is the fixed,
    background-padded window every pattern is emitted at -- so it, not the window,
    is what ``schema.annotation.low_confidence_annotation`` measures its short
    clause on. None is a declaration that no core was recorded, never a licence to
    substitute the window.
    """
    core = node.trimmed_core
    if not isinstance(core, (list, tuple)) or len(core) != 2:
        return None
    try:
        start, end = int(core[0]), int(core[1])
    except (TypeError, ValueError):
        return None
    return end - start if end >= start else None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
