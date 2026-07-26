"""Versioned, candidate-only annotation evidence.

Database matches propose labels.  They do not adjudicate a ``MotifNode`` family
assignment, because contradictory proposals are evidence that must survive to
the later adjudication stage.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from . import SchemaError

__all__ = [
    "ANNOTATION_SCHEMA_VERSION", "AnnotationCandidate", "AnnotationBackendLog",
    "BackendStatus", "stable_candidate_id", "low_confidence_annotation",
]

ANNOTATION_SCHEMA_VERSION = "1"


class BackendStatus(StrEnum):
    """A backend result is verified or explicitly unavailable, never absent."""

    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"


def stable_candidate_id(*, node_id: str, source: str, source_version: str,
                        matched_motif_id: str) -> str:
    """Return a row-order-independent identity for one proposed database match."""
    identity = {
        "matched_motif_id": matched_motif_id,
        "node_id": node_id,
        "source": source,
        "source_version": source_version,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"annotation:{hashlib.sha256(encoded).hexdigest()}"


def low_confidence_annotation(*, motif_length: int, source: str, q_value: float | None,
                              seqlet_count: int | None) -> bool:
    """Apply the documented short, weak-TomTom, and low-support flags."""
    return (
        motif_length <= 6
        or (source.casefold() == "tomtom" and q_value is not None and q_value > 0.05)
        or (seqlet_count is not None and seqlet_count < 100)
    )


@dataclass(frozen=True)
class AnnotationCandidate:
    """One backend proposal, deliberately separate from ``MotifNode`` fields."""

    candidate_id: str
    node_id: str
    proposed_family_id: str
    source: str
    source_version: str
    matched_motif_id: str
    score: float | None
    q_value: float | None
    aligned_span: int | None
    motif_length: int
    seqlet_count: int | None
    low_confidence_annotation: bool
    chance_occurrence_probability: float | None
    observed_to_null_ratio: float | None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ANNOTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        required = {
            "candidate_id": self.candidate_id,
            "node_id": self.node_id,
            "proposed_family_id": self.proposed_family_id,
            "source": self.source,
            "source_version": self.source_version,
            "matched_motif_id": self.matched_motif_id,
        }
        if any(not isinstance(value, str) or not value.strip() for value in required.values()):
            raise SchemaError("annotation candidates require non-empty identity fields")
        if self.schema_version != ANNOTATION_SCHEMA_VERSION:
            raise SchemaError(
                f"annotation candidate schema_version must be {ANNOTATION_SCHEMA_VERSION!r}"
            )
        if self.motif_length < 1:
            raise SchemaError("annotation candidate motif_length must be positive")
        if self.seqlet_count is not None and self.seqlet_count < 0:
            raise SchemaError("annotation candidate seqlet_count cannot be negative")
        if self.q_value is not None and not 0.0 <= self.q_value <= 1.0:
            raise SchemaError("annotation candidate q_value must be in [0, 1]")
        if self.chance_occurrence_probability is not None and not (
            0.0 <= self.chance_occurrence_probability <= 1.0
        ):
            raise SchemaError("chance_occurrence_probability must be in [0, 1]")
        if self.observed_to_null_ratio is not None and self.observed_to_null_ratio < 0.0:
            raise SchemaError("observed_to_null_ratio cannot be negative")
        expected_id = stable_candidate_id(
            node_id=self.node_id, source=self.source, source_version=self.source_version,
            matched_motif_id=self.matched_motif_id,
        )
        if self.candidate_id != expected_id:
            raise SchemaError("candidate_id does not match its stable annotation match identity")
        if low_confidence_annotation(
            motif_length=self.motif_length, source=self.source, q_value=self.q_value,
            seqlet_count=self.seqlet_count,
        ) and not self.low_confidence_annotation:
            raise SchemaError("a short, weak TomTom, or low-seqlet candidate must be flagged")

    @classmethod
    def create(cls, *, node_id: str, proposed_family_id: str, source: str,
               source_version: str, matched_motif_id: str, motif_length: int,
               seqlet_count: int | None, score: float | None = None,
               q_value: float | None = None, aligned_span: int | None = None,
               chance_occurrence_probability: float | None = None,
               observed_to_null_ratio: float | None = None,
               provenance: Mapping[str, Any] | None = None) -> AnnotationCandidate:
        """Build a candidate with its stable ID and mandatory confidence flag."""
        return cls(
            candidate_id=stable_candidate_id(
                node_id=node_id, source=source, source_version=source_version,
                matched_motif_id=matched_motif_id,
            ),
            node_id=node_id, proposed_family_id=proposed_family_id, source=source,
            source_version=source_version, matched_motif_id=matched_motif_id, score=score,
            q_value=q_value, aligned_span=aligned_span, motif_length=motif_length,
            seqlet_count=seqlet_count,
            low_confidence_annotation=low_confidence_annotation(
                motif_length=motif_length, source=source, q_value=q_value,
                seqlet_count=seqlet_count,
            ),
            chance_occurrence_probability=chance_occurrence_probability,
            observed_to_null_ratio=observed_to_null_ratio,
            provenance=dict(provenance or {}),
        )

    def with_occurrence_null(self, values: Mapping[str, Any]) -> AnnotationCandidate:
        """Attach only supplied precomputed null values; never estimate either one."""
        return replace(
            self,
            chance_occurrence_probability=values.get("chance_occurrence_probability"),
            observed_to_null_ratio=values.get("observed_to_null_ratio"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "node_id": self.node_id,
            "proposed_family_id": self.proposed_family_id,
            "source": self.source,
            "source_version": self.source_version,
            "matched_motif_id": self.matched_motif_id,
            "score": self.score,
            "q_value": self.q_value,
            "aligned_span": self.aligned_span,
            "motif_length": self.motif_length,
            "seqlet_count": self.seqlet_count,
            "low_confidence_annotation": self.low_confidence_annotation,
            "chance_occurrence_probability": self.chance_occurrence_probability,
            "observed_to_null_ratio": self.observed_to_null_ratio,
            "provenance": dict(self.provenance),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class AnnotationBackendLog:
    """One versioned backend outcome, including an explicit unverified state."""

    backend: str
    backend_version: str
    status: BackendStatus
    candidate_count: int
    detail: str | None = None
    schema_version: str = ANNOTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (not isinstance(self.backend, str) or not self.backend.strip()
                or not isinstance(self.backend_version, str) or not self.backend_version.strip()):
            raise SchemaError("backend logs require a backend name and version")
        try:
            status = BackendStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"backend log status {self.status!r} is not recognised") from exc
        object.__setattr__(self, "status", status)
        if self.candidate_count < 0:
            raise SchemaError("backend log candidate_count cannot be negative")
        if self.schema_version != ANNOTATION_SCHEMA_VERSION:
            raise SchemaError(
                f"annotation backend log schema_version must be {ANNOTATION_SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "status": self.status.value,
            "candidate_count": self.candidate_count,
            "detail": self.detail,
            "schema_version": self.schema_version,
        }
