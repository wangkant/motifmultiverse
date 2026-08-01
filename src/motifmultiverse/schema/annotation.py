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

#: Bumped to "2" when `AnnotationCandidate` gained the required
#: `trimmed_core_length`. The short-motif rule used to read `motif_length`, the
#: PADDED window, which is 50 on every real tfmodisco-lite node and so could
#: never cross a <=6bp threshold; the trimmed core can, and does on 40 of 139
#: real nodes.
ANNOTATION_SCHEMA_VERSION = "2"


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


def low_confidence_annotation(*, trimmed_core_length: int | None, source: str,
                              q_value: float | None, seqlet_count: int | None) -> bool:
    """Apply the documented short, weak-TomTom, and low-support flags.

    The short clause is measured on the **trimmed core**, never on the padded
    pattern window, for the reason ``align/__init__.py`` sets out at length:
    TF-MoDISco emits every pattern at one fixed window width and pads the flanks
    with near-uniform background, so ``motif_length`` is a property of the
    discovery run's window and not of the motif. Run end to end on thirteen real
    tfmodisco-lite analyses, ``motif_length`` was 50 for all 139 registry nodes,
    so ``motif_length <= 6`` could not fire once -- while 40 of those same 139
    nodes carry a contribution-bearing core of 6bp or less, which is exactly the
    population this clause exists to flag. A threshold that cannot be crossed by
    any value the upstream stage can produce is not protection, and it had been
    shipping as protection.

    ``trimmed_core_length`` is None when the node declares no core at all. That
    is "not measured", not "not short": the clause is then left unapplied rather
    than quietly re-evaluated against the padded window, because that fallback is
    what made the rule vacuous in the first place and nothing downstream could
    tell the two apart. ``guards.short_motif_flag`` counts those nodes in its own
    detail, so a run in which the clause could never fire says so out loud
    instead of reading as a pass.
    """
    return (
        (trimmed_core_length is not None and trimmed_core_length <= 6)
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
    #: The width of the annotated node's contribution-bearing core, or None when
    #: the node declares none. Required, and deliberately not defaulted: a field
    #: that could be left out would silently restore the padded-window reading
    #: this replaced. See ``low_confidence_annotation``.
    trimmed_core_length: int | None
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
        if self.trimmed_core_length is not None:
            if self.trimmed_core_length < 0:
                raise SchemaError("annotation candidate trimmed_core_length cannot be negative")
            if self.trimmed_core_length > self.motif_length:
                raise SchemaError(
                    "annotation candidate trimmed_core_length exceeds its motif_length; the "
                    "trimmed core is a span inside the pattern window, never wider than it"
                )
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
            trimmed_core_length=self.trimmed_core_length, source=self.source,
            q_value=self.q_value, seqlet_count=self.seqlet_count,
        ) and not self.low_confidence_annotation:
            raise SchemaError("a short, weak TomTom, or low-seqlet candidate must be flagged")

    @classmethod
    def create(cls, *, node_id: str, proposed_family_id: str, source: str,
               source_version: str, matched_motif_id: str, motif_length: int,
               trimmed_core_length: int | None,
               seqlet_count: int | None, score: float | None = None,
               q_value: float | None = None, aligned_span: int | None = None,
               chance_occurrence_probability: float | None = None,
               observed_to_null_ratio: float | None = None,
               provenance: Mapping[str, Any] | None = None) -> AnnotationCandidate:
        """Build a candidate with its stable ID and mandatory confidence flag.

        ``trimmed_core_length`` has no default on purpose. Every caller has the
        annotated node in hand and can read its declared core; a default would
        let one of them omit the only width the short clause can be measured on,
        which is the failure this argument exists to close.
        """
        return cls(
            candidate_id=stable_candidate_id(
                node_id=node_id, source=source, source_version=source_version,
                matched_motif_id=matched_motif_id,
            ),
            node_id=node_id, proposed_family_id=proposed_family_id, source=source,
            source_version=source_version, matched_motif_id=matched_motif_id, score=score,
            q_value=q_value, aligned_span=aligned_span, motif_length=motif_length,
            trimmed_core_length=trimmed_core_length,
            seqlet_count=seqlet_count,
            low_confidence_annotation=low_confidence_annotation(
                trimmed_core_length=trimmed_core_length, source=source, q_value=q_value,
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
            "trimmed_core_length": self.trimmed_core_length,
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
