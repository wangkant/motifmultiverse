"""Candidate-producing annotation orchestration; see README.md for the rule."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from motifmultiverse import guards
from motifmultiverse.guard_log import GuardLog
from motifmultiverse.provenance import record
from motifmultiverse.schema import MotifNode, SchemaError
from motifmultiverse.schema.annotation import (
    ANNOTATION_SCHEMA_VERSION,
    AnnotationBackendLog,
    AnnotationCandidate,
    BackendStatus,
)

from .base import AnnotationBackend

__all__ = [
    "AnnotationError", "AnnotationRun", "annotate_nodes", "annotate_registry",
    "write_annotation_artifacts", "run",
]


class AnnotationError(ValueError):
    """Annotation input or candidate evidence is structurally invalid."""


@dataclass(frozen=True)
class AnnotationRun:
    """Versioned result for a candidate-only annotation run."""

    candidates: tuple[AnnotationCandidate, ...]
    backend_logs: tuple[AnnotationBackendLog, ...]
    schema_version: str = ANNOTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ANNOTATION_SCHEMA_VERSION:
            raise AnnotationError(
                f"annotation run schema_version must be {ANNOTATION_SCHEMA_VERSION!r}"
            )


def annotate_nodes(nodes: Sequence[MotifNode], backends: Sequence[AnnotationBackend], *,
                   occurrence_nulls: Mapping[str, Mapping[str, Any]] | None = None,
                   guard_log: GuardLog | None = None) -> AnnotationRun:
    """Collect backend proposals without ever assigning a node's family fields.

    ``guard_log`` records what this stage's guard returned; bound to the run's
    output directory by :func:`annotate_registry`, the outcome is written before
    the guard's refusal propagates, so an annotation run refused for an unflagged
    low-confidence motif leaves the sentence that refused it in
    ``guard_outcomes.json`` rather than only on stderr.
    """
    guard_log = guard_log if guard_log is not None else GuardLog("annotate")
    node_ids = {node.node_id for node in nodes}
    candidates_by_id: dict[str, AnnotationCandidate] = {}
    logs: list[AnnotationBackendLog] = []
    for backend in backends:
        try:
            returned = tuple(backend.annotate(nodes))
            backend_candidates: dict[str, AnnotationCandidate] = {}
            for candidate in returned:
                if (candidate.source != backend.name
                        or candidate.source_version != backend.version):
                    raise AnnotationError(
                        f"{backend.name}/{backend.version} returned candidate "
                        f"source/version {candidate.source}/{candidate.source_version}"
                    )
                if candidate.node_id not in node_ids:
                    raise AnnotationError(
                        f"{backend.name} returned candidate for unknown node {candidate.node_id!r}"
                    )
                if occurrence_nulls is not None and candidate.candidate_id in occurrence_nulls:
                    candidate = candidate.with_occurrence_null(occurrence_nulls[candidate.candidate_id])
                previous = backend_candidates.get(candidate.candidate_id)
                if previous is not None and previous != candidate:
                    raise AnnotationError(
                        f"{backend.name} returned contradictory rows for {candidate.candidate_id}"
                    )
                backend_candidates[candidate.candidate_id] = candidate
            candidates_by_id.update(backend_candidates)
            logs.append(AnnotationBackendLog(
                backend=backend.name, backend_version=backend.version,
                status=BackendStatus.VERIFIED, candidate_count=len(returned),
            ))
        except Exception as exc:
            if not getattr(backend, "optional", True):
                raise
            logs.append(AnnotationBackendLog(
                backend=getattr(backend, "name", type(backend).__name__),
                backend_version=getattr(backend, "version", "UNVERIFIED"),
                status=BackendStatus.UNVERIFIED, candidate_count=0, detail=str(exc),
            ))
    retained = tuple(candidates_by_id[key] for key in sorted(candidates_by_id))
    # The stage's own executable check (annotate/README.md "How to check it").
    # schema.AnnotationCandidate applies the same rule when a candidate is built;
    # this re-applies it to what is about to be WRITTEN, from an independently
    # written threshold implementation in guards/. The two had already drifted --
    # the guard read a legitimate motif_length of 0 as "absent" and passed the
    # weakest possible motif -- which is the argument for keeping both.
    guard_log.record(
        guards.short_motif_flag([
            {
                "variant_id": candidate.candidate_id,
                "trimmed_core_length": candidate.trimmed_core_length,
                "seqlet_count": candidate.seqlet_count,
                "annotation_matches": (
                    {"tomtom_q": candidate.q_value}
                    if candidate.source.casefold() == "tomtom" else {}
                ),
                "low_confidence_annotation": candidate.low_confidence_annotation,
            }
            for candidate in retained
        ]),
        subject=(
            "the annotation candidates about to be written to "
            "annotation_candidates.parquet, from backend(s) "
            + (", ".join(f"{entry.backend}/{entry.backend_version}" for entry in logs)
               or "none")
        ),
    ).raise_if_failed()
    return AnnotationRun(candidates=retained, backend_logs=tuple(logs))


_CANDIDATE_COLUMNS = [
    "candidate_id", "node_id", "proposed_family_id", "source", "source_version",
    "matched_motif_id", "score", "q_value", "aligned_span", "motif_length",
    "trimmed_core_length", "seqlet_count",
    "low_confidence_annotation", "chance_occurrence_probability", "observed_to_null_ratio",
    "provenance", "schema_version",
]
_CANDIDATE_DTYPES = {
    "candidate_id": "string", "node_id": "string", "proposed_family_id": "string",
    "source": "string", "source_version": "string", "matched_motif_id": "string",
    "score": "float64", "q_value": "float64", "aligned_span": "Int64",
    "motif_length": "int64", "trimmed_core_length": "Int64",
    "seqlet_count": "Int64", "low_confidence_annotation": "bool",
    "chance_occurrence_probability": "float64", "observed_to_null_ratio": "float64",
    "provenance": "string", "schema_version": "string",
}


def write_annotation_artifacts(out_dir: str | Path, result: AnnotationRun) -> tuple[Path, Path]:
    """Write typed candidate rows and backend-specific verification logs."""
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in result.candidates:
        row = candidate.to_dict()
        row["provenance"] = json.dumps(row["provenance"], sort_keys=True, separators=(",", ":"))
        rows.append(row)
    candidates_path = out / "annotation_candidates.parquet"
    pd.DataFrame(rows, columns=_CANDIDATE_COLUMNS).astype(_CANDIDATE_DTYPES).to_parquet(
        candidates_path, index=False
    )
    logs_path = out / "annotation_backend_logs.json"
    logs_path.write_text(json.dumps({
        "schema_version": result.schema_version,
        "backends": [entry.to_dict() for entry in result.backend_logs],
    }, indent=2, sort_keys=True), encoding="utf-8")
    return candidates_path, logs_path


def _read_registry_nodes(registry_dir: str | Path) -> tuple[Path, list[MotifNode]]:
    registry_path = Path(registry_dir) / "registry.json"
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        raw_nodes = payload["nodes"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AnnotationError(f"{registry_path} is not a readable motif registry: {exc}") from exc
    try:
        return registry_path, [MotifNode(**node) for node in raw_nodes]
    except (SchemaError, TypeError) as exc:
        raise AnnotationError(f"{registry_path} contains invalid motif nodes: {exc}") from exc


def annotate_registry(registry_dir: str | Path, out_dir: str | Path, *,
                      backends: Sequence[AnnotationBackend],
                      occurrence_nulls: Mapping[str, Mapping[str, Any]] | None = None,
                      provenance_inputs: Sequence[str | Path] = ()) -> AnnotationRun:
    """Load registry nodes, retain candidate evidence, and write run provenance."""
    registry_path, nodes = _read_registry_nodes(registry_dir)
    provenance = record("annotate")
    provenance.add_input(registry_path)
    for path in provenance_inputs:
        try:
            provenance.add_input(path)
        except OSError:
            # An unavailable optional backend has no bytes to checksum. Its
            # backend-specific log records the UNVERIFIED state after the
            # adapter runs; inventing a checksum here would obscure that fact.
            continue
    provenance.write(out_dir)
    result = annotate_nodes(nodes, backends, occurrence_nulls=occurrence_nulls,
                            guard_log=GuardLog("annotate", out_dir))
    write_annotation_artifacts(out_dir, result)
    return result


run = annotate_registry
