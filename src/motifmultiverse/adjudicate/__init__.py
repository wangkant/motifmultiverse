"""adjudicate stage -- see README.md in this directory for rule / failure / check."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

from motifmultiverse.align import AlignmentEvidence
from motifmultiverse.schema import (
    DECISION_BUNDLE_PRODUCER,
    DECISION_BUNDLE_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
    Decision,
    DecisionBundle,
    DecisionRecord,
    MotifNode,
    RegistryMetadata,
    SchemaError,
    decision_bundle_artifact_id,
)
from motifmultiverse.schema.annotation import AnnotationCandidate
from motifmultiverse.schema.criteria import Criterion, evaluate_criterion

__all__ = [
    "ADJUDICATION_SCHEMA_VERSION",
    "AdjudicationError",
    "StabilityEvidence",
    "OntologyDecision",
    "stable_decision_id",
    "choose_medoid",
    "adjudicate_component",
    "adjudicate_all",
    "apply_manual_override",
    "write_adjudication_artifacts",
    "adjudicate_evidence",
    "run",
]

ADJUDICATION_SCHEMA_VERSION = "1"
_MEDOID_TIE_FIELDS = (
    "motif_completeness",
    "seqlet_count",
    "core_ic",
    "cross_context_recurrence",
)


class AdjudicationError(SchemaError):
    """Adjudication inputs or a produced decision violate the frozen contract."""


@runtime_checkable
class StabilityEvidence(Protocol):
    """Downstream stability evidence as adjudication reads it."""

    decision_id: str
    n_affected_peaks: int
    status: str


def stable_decision_id(
    node_ids: Sequence[str],
    relationship: str,
    criterion_id: str,
    criterion_version: str,
) -> str:
    """Return the stable identity of one considered cluster under one criterion."""
    payload = {
        "criterion_id": criterion_id,
        "criterion_version": criterion_version,
        "node_ids": sorted(node_ids),
        "relationship": relationship,
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"decision:{digest}"


@dataclass(frozen=True)
class OntologyDecision:
    """One first-class collapse, refusal, override, or deferred relationship."""

    decision_id: str
    node_ids: tuple[str, ...]
    relationship: str
    decision: Decision
    family_id: str | None
    representative_node_id: str | None
    criterion_id: str
    criterion_version: str
    evidence_ids: tuple[str, ...]
    evidence_for: tuple[str, ...]
    evidence_against: tuple[str, ...]
    rationale: str
    decided_by: str
    manual_override: bool
    automated_decision: Decision | None = None
    override_operator: str | None = None
    override_rationale: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ADJUDICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            decision = Decision(self.decision)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"ontology decision {self.decision!r} is not recognised") from exc
        object.__setattr__(self, "decision", decision)
        automated = self.automated_decision
        if automated is not None:
            try:
                automated = Decision(automated)
            except (TypeError, ValueError) as exc:
                raise SchemaError(
                    f"automated decision {self.automated_decision!r} is not recognised"
                ) from exc
            object.__setattr__(self, "automated_decision", automated)

        if self.schema_version != ADJUDICATION_SCHEMA_VERSION:
            raise SchemaError(
                f"ontology decision schema_version must be {ADJUDICATION_SCHEMA_VERSION!r}"
            )
        if len(self.node_ids) < 2 or len(set(self.node_ids)) != len(self.node_ids):
            raise SchemaError("ontology decision node_ids require at least two unique observed nodes")
        if tuple(sorted(self.node_ids)) != self.node_ids:
            raise SchemaError("ontology decision node_ids must be in lexical order")
        expected_id = stable_decision_id(
            self.node_ids, self.relationship, self.criterion_id, self.criterion_version
        )
        if self.decision_id != expected_id:
            raise SchemaError("decision_id does not match the considered cluster and criterion")
        if not self.relationship.strip() or not self.criterion_id.strip() or not self.criterion_version.strip():
            raise SchemaError("ontology relationship and criterion identity must be non-empty")
        if not self.rationale.strip():
            raise SchemaError("every ontology decision, including a refusal, requires a rationale")
        if not self.decided_by.strip():
            raise SchemaError("every ontology decision requires decided_by")
        if decision is Decision.COLLAPSE:
            if self.representative_node_id not in self.node_ids:
                raise SchemaError(
                    "a collapse representative must be an observed member of its own cluster"
                )
        elif self.representative_node_id is not None:
            raise SchemaError("a non-collapse decision cannot name a surviving representative")

        if self.manual_override:
            if decision is not Decision.KEEP_SEPARATE_CURATOR_OVERRIDE:
                raise SchemaError("a manual override must use KEEP_SEPARATE_CURATOR_OVERRIDE")
            if automated is None:
                raise SchemaError("a manual override must preserve the automated decision")
            if not self.override_operator or not self.override_operator.strip():
                raise SchemaError("a manual override requires an operator")
            if not self.override_rationale or not self.override_rationale.strip():
                raise SchemaError("a manual override requires an override rationale")
        else:
            if automated is None:
                object.__setattr__(self, "automated_decision", decision)
            elif automated is not decision:
                raise SchemaError(
                    "an automated ontology decision cannot disagree with automated_decision"
                )
            if self.override_operator is not None or self.override_rationale is not None:
                raise SchemaError("override fields require manual_override=True")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        payload["automated_decision"] = self.automated_decision.value
        return payload


def _pair_similarity(
    left: str,
    right: str,
    pairwise_similarity: Mapping[tuple[str, str], float],
) -> float:
    if (left, right) in pairwise_similarity:
        return float(pairwise_similarity[left, right])
    if (right, left) in pairwise_similarity:
        return float(pairwise_similarity[right, left])
    raise ValueError(f"missing pairwise similarity for {left!r} and {right!r}")


def choose_medoid(
    node_ids: Sequence[str],
    pairwise_similarity: Mapping[tuple[str, str], float],
    *,
    node_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Choose an observed member by mean similarity and declared deterministic ties."""
    members = tuple(node_ids)
    if not members:
        raise ValueError("cannot choose a medoid from an empty component")
    if len(set(members)) != len(members):
        raise ValueError("a medoid component cannot contain duplicate node IDs")
    if len(members) == 1:
        return members[0]

    metadata = node_metadata or {}

    def rank(node_id: str) -> tuple[float, float, float, float, float]:
        mean_similarity = sum(
            _pair_similarity(node_id, other, pairwise_similarity)
            for other in members
            if other != node_id
        ) / (len(members) - 1)
        values = metadata.get(node_id, {})

        def value(name: str) -> float:
            raw = values.get(name)
            return float(raw) if raw is not None else float("-inf")

        return (
            mean_similarity,
            value("motif_completeness"),
            value("seqlet_count"),
            value("core_ic"),
            value("cross_context_recurrence"),
        )

    # ``max`` supplies the descending scientific criteria and retains the first
    # item on an exact tie, so lexical order makes the smallest ID deterministic.
    return max(sorted(members), key=rank)


def _alignment_id(edge: AlignmentEvidence) -> str:
    payload = edge.to_dict()
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"alignment:{digest}"


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def _family_evidence(
    members: tuple[str, ...],
    candidates: Sequence[AnnotationCandidate],
) -> tuple[dict[str, set[str]], tuple[AnnotationCandidate, ...]]:
    member_set = set(members)
    usable = tuple(
        candidate
        for candidate in candidates
        if candidate.node_id in member_set and not candidate.low_confidence_annotation
    )
    by_node = {node_id: set() for node_id in members}
    for candidate in usable:
        by_node[candidate.node_id].add(candidate.proposed_family_id)
    return by_node, usable


def _medoid_metadata(
    members: tuple[str, ...],
    node_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for node_id in members:
        values = {
            key: node_metadata.get(node_id, {}).get(key)
            for key in _MEDOID_TIE_FIELDS
            if node_metadata.get(node_id, {}).get(key) is not None
        }
        if values:
            metadata[node_id] = values
    return metadata


def _missing_authoritative_tie_metadata(
    members: tuple[str, ...],
    pairwise_similarity: Mapping[tuple[str, str], float],
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Return missing persisted fields only when mean-similarity leaders tie."""
    if len(members) < 2:
        return {}
    means = {
        member: sum(
            _pair_similarity(member, other, pairwise_similarity)
            for other in members
            if other != member
        ) / (len(members) - 1)
        for member in members
    }
    best = max(means.values())
    leaders = tuple(member for member in members if means[member] == best)
    if len(leaders) < 2:
        return {}
    return {
        member: tuple(
            field_name
            for field_name in _MEDOID_TIE_FIELDS
            if metadata.get(member, {}).get(field_name) is None
        )
        for member in leaders
        if any(
            metadata.get(member, {}).get(field_name) is None
            for field_name in _MEDOID_TIE_FIELDS
        )
    }


def _criterion_for(
    relationship: str,
    criteria: Mapping[str, Criterion],
) -> Criterion:
    try:
        criterion = criteria[relationship]
    except KeyError as exc:
        raise AdjudicationError(
            f"no criterion is registered for relationship {relationship!r}"
        ) from exc
    if criterion.relationship != relationship:
        raise AdjudicationError(
            f"criterion {criterion.criterion_id!r} declares relationship "
            f"{criterion.relationship!r}, not {relationship!r}"
        )
    return criterion


def _stability_values(
    decision_id: str,
    stability_results: Sequence[StabilityEvidence],
    required_fields: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    for row in stability_results:
        if not isinstance(row, StabilityEvidence):
            raise AdjudicationError(
                "stability evidence must expose decision_id, n_affected_peaks, and status"
            )
        if (
            not isinstance(row.decision_id, str)
            or not row.decision_id.strip()
            or not isinstance(row.status, str)
            or not row.status.strip()
            or not isinstance(row.n_affected_peaks, int)
            or isinstance(row.n_affected_peaks, bool)
            or row.n_affected_peaks < 0
        ):
            raise AdjudicationError(
                "stability evidence fields have invalid types or values"
            )
    matching = [row for row in stability_results if row.decision_id == decision_id]
    if not matching:
        return {}, []
    if len(matching) > 1:
        raise AdjudicationError(
            f"multiple stability rows name decision_id {decision_id!r}; expected at most one"
        )
    row = matching[0]
    values = {
        "decision_id": row.decision_id,
        "n_affected_peaks": row.n_affected_peaks,
        "status": row.status,
        "downstream_stability_present": True,
    }
    for field_name in required_fields:
        value = getattr(row, field_name, None)
        if value is not None:
            values[field_name] = value
    return values, [f"stability:{row.decision_id}"]


def adjudicate_component(
    node_ids: Sequence[str],
    alignment_edges: Sequence[AlignmentEvidence],
    annotation_candidates: Sequence[AnnotationCandidate],
    stability_results: Sequence[StabilityEvidence],
    criteria: Mapping[str, Criterion],
    decided_by: str,
    *,
    node_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> OntologyDecision:
    """Adjudicate one proposed connected component without connectivity licensing a merge."""
    members = tuple(sorted(node_ids))
    if len(members) < 2 or len(set(members)) != len(members):
        raise AdjudicationError(
            "an adjudication component requires at least two unique node IDs"
        )
    if not decided_by.strip():
        raise AdjudicationError("adjudication requires a non-empty decided_by")
    member_set = set(members)
    node_metadata = node_metadata or {}

    pair_edges: dict[tuple[str, str], AlignmentEvidence] = {}
    for edge in alignment_edges:
        if edge.source_node_id not in member_set or edge.target_node_id not in member_set:
            continue
        key = _pair_key(edge.source_node_id, edge.target_node_id)
        if key in pair_edges:
            raise AdjudicationError(f"duplicate alignment evidence for pair {key}")
        pair_edges[key] = edge

    families_by_node, usable_candidates = _family_evidence(members, annotation_candidates)
    for node_id in members:
        declared_family = node_metadata.get(node_id, {}).get("family_id")
        if declared_family not in (None, "", "NA"):
            families_by_node[node_id].add(str(declared_family))
    proposed_families = set().union(*families_by_node.values())
    family_conflict = len(proposed_families) > 1
    variant_ids = [
        node_metadata.get(node_id, {}).get("variant_id")
        for node_id in members
    ]
    distinct_variant_ids = (
        all(isinstance(value, str) and value.strip() for value in variant_ids)
        and len(set(variant_ids)) > 1
    )
    if family_conflict:
        relationship = "AMBIGUOUS_CROSS_FAMILY"
    elif len(proposed_families) == 1 and distinct_variant_ids:
        relationship = "SAME_FAMILY_VARIANT"
    elif any(
        edge.overlap_frac_source < 1.0 or edge.overlap_frac_target < 1.0
        for edge in pair_edges.values()
    ):
        relationship = "FRAGMENT_MATCH"
    else:
        relationship = "TRUE_DUPLICATE"

    criterion = _criterion_for(relationship, criteria)
    decision_id = stable_decision_id(
        members, relationship, criterion.criterion_id, criterion.version
    )
    evidence_ids = [_alignment_id(edge) for edge in pair_edges.values()]
    evidence_ids.extend(candidate.candidate_id for candidate in usable_candidates)
    stability_values, stability_ids = _stability_values(
        decision_id, stability_results, criterion.required_evidence
    )
    evidence_ids.extend(stability_ids)
    provenance = {
        "criterion_id": criterion.criterion_id,
        "criterion_version": criterion.version,
        "registration_rule_versions": sorted(
            {edge.registration_rule_version for edge in pair_edges.values()}
        ),
    }
    family_id = next(iter(proposed_families)) if len(proposed_families) == 1 else None

    base_evidence: dict[str, Any] = {
        "alignment_registered": bool(pair_edges),
        "family_conflict": family_conflict,
        "same_family": family_id is not None and all(families_by_node.values()),
        "distinct_variant_ids": distinct_variant_ids if all(variant_ids) else None,
        **stability_values,
    }

    expected_pairs = {
        _pair_key(left, right)
        for index, left in enumerate(members)
        for right in members[index + 1 :]
    }
    missing_pairs = expected_pairs - set(pair_edges)
    if missing_pairs:
        return OntologyDecision(
            decision_id=decision_id,
            node_ids=members,
            relationship=relationship,
            decision=Decision.DEFERRED,
            family_id=family_id,
            representative_node_id=None,
            criterion_id=criterion.criterion_id,
            criterion_version=criterion.version,
            evidence_ids=tuple(sorted(set(evidence_ids))),
            evidence_for=tuple(f"registered pair {pair}" for pair in sorted(pair_edges)),
            evidence_against=(
                f"non-transitive connected component lacks independently gated pairs "
                f"{sorted(missing_pairs)}",
            ),
            rationale=(
                "non-transitive connected component: graph connectivity only proposed "
                "the cluster; absent pair gates prohibit unrestricted single-linkage collapse"
            ),
            decided_by=decided_by,
            manual_override=False,
            provenance=provenance,
        )

    if relationship in {"AMBIGUOUS_CROSS_FAMILY", "SAME_FAMILY_VARIANT"}:
        evaluation = evaluate_criterion(criterion, base_evidence)
        evidence_for = (
            f"registered matrix similarity up to "
            f"{max((edge.ppm_similarity for edge in pair_edges.values()), default=float('nan')):.3f}",
        )
        evidence_against = (
            (
                f"conflicting proposed families: {sorted(proposed_families)}"
                if family_conflict
                else f"same family retains distinct variant identities: {sorted(variant_ids)}"
            ),
        )
        structural_decision = (
            Decision.REFUSE_MERGE
            if relationship == "AMBIGUOUS_CROSS_FAMILY"
            else evaluation.decision
        )
        rationale = evaluation.rationale
        if relationship == "AMBIGUOUS_CROSS_FAMILY":
            rationale = (
                "conflicting authoritative family assignments always REFUSE_MERGE; "
                f"criterion context: {evaluation.rationale}"
            )
        return OntologyDecision(
            decision_id=decision_id,
            node_ids=members,
            relationship=relationship,
            decision=structural_decision,
            family_id=family_id,
            representative_node_id=None,
            criterion_id=criterion.criterion_id,
            criterion_version=criterion.version,
            evidence_ids=tuple(sorted(set(evidence_ids))),
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            rationale=rationale,
            decided_by=decided_by,
            manual_override=False,
            provenance=provenance,
        )

    similarities = {
        pair: edge.ppm_similarity
        for pair, edge in pair_edges.items()
    }
    medoid_metadata = _medoid_metadata(members, node_metadata)
    missing_tie_metadata = _missing_authoritative_tie_metadata(
        members, similarities, medoid_metadata
    )
    if missing_tie_metadata:
        return OntologyDecision(
            decision_id=decision_id,
            node_ids=members,
            relationship=relationship,
            decision=Decision.DEFERRED,
            family_id=family_id,
            representative_node_id=None,
            criterion_id=criterion.criterion_id,
            criterion_version=criterion.version,
            evidence_ids=tuple(sorted(set(evidence_ids))),
            evidence_for=tuple(
                f"registered pair {pair}" for pair in sorted(pair_edges)
            ),
            evidence_against=(
                f"missing authoritative medoid tie metadata {missing_tie_metadata}",
            ),
            rationale=(
                "mean-similarity medoid leaders tie, but authoritative medoid tie metadata "
                "is unavailable; adjudication defers rather than substituting identifier "
                "parsing, zeros, or lexical order as scientific evidence"
            ),
            decided_by=decided_by,
            manual_override=False,
            provenance=provenance,
        )
    medoid = choose_medoid(
        members,
        similarities,
        node_metadata=medoid_metadata,
    )
    evaluations = []
    for member in members:
        if member == medoid:
            continue
        edge = pair_edges[_pair_key(member, medoid)]
        pair_evidence = {
            **base_evidence,
            "ppm_similarity": edge.ppm_similarity,
            "signed_cwm_similarity": edge.signed_cwm_similarity,
            "empirical_p_value": edge.empirical_p_value,
            "overlap_frac_source": edge.overlap_frac_source,
            "overlap_frac_target": edge.overlap_frac_target,
        }
        evaluations.append(evaluate_criterion(criterion, pair_evidence))

    if all(result.decision is Decision.COLLAPSE for result in evaluations):
        final_decision = Decision.COLLAPSE
    elif any(result.decision is Decision.REFUSE_MERGE for result in evaluations):
        final_decision = Decision.REFUSE_MERGE
    else:
        final_decision = Decision.DEFERRED
    rationale = "; ".join(result.rationale for result in evaluations)
    passed = tuple(
        f"{member} independently satisfies the {criterion.criterion_id} gate to medoid {medoid}"
        for member, result in zip(
            (member for member in members if member != medoid), evaluations, strict=True
        )
        if result.decision is Decision.COLLAPSE
    )
    failed = tuple(
        f"{member} does not satisfy the {criterion.criterion_id} gate to medoid {medoid}"
        for member, result in zip(
            (member for member in members if member != medoid), evaluations, strict=True
        )
        if result.decision is not Decision.COLLAPSE
    )
    return OntologyDecision(
        decision_id=decision_id,
        node_ids=members,
        relationship=relationship,
        decision=final_decision,
        family_id=family_id,
        representative_node_id=medoid if final_decision is Decision.COLLAPSE else None,
        criterion_id=criterion.criterion_id,
        criterion_version=criterion.version,
        evidence_ids=tuple(sorted(set(evidence_ids))),
        evidence_for=passed,
        evidence_against=failed,
        rationale=rationale,
        decided_by=decided_by,
        manual_override=False,
        provenance=provenance,
    )


def apply_manual_override(
    automated: OntologyDecision,
    *,
    operator: str,
    rationale: str,
) -> OntologyDecision:
    """Record a curator separation without relabelling it as automated evidence."""
    return replace(
        automated,
        decision=Decision.KEEP_SEPARATE_CURATOR_OVERRIDE,
        representative_node_id=None,
        rationale=(
            f"Manual override by {operator}: {rationale}. Automated decision preserved "
            f"separately as {automated.decision.value}."
        ),
        manual_override=True,
        automated_decision=automated.decision,
        override_operator=operator,
        override_rationale=rationale,
    )


def adjudicate_all(
    alignment_edges: Sequence[AlignmentEvidence],
    annotation_candidates: Sequence[AnnotationCandidate],
    stability_results: Sequence[StabilityEvidence],
    criteria: Mapping[str, Criterion],
    decided_by: str,
    *,
    node_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[OntologyDecision, ...]:
    """Propose components by connectivity, then adjudicate every proposal."""
    adjacency: dict[str, set[str]] = {}
    for edge in alignment_edges:
        if edge.source_node_id == edge.target_node_id:
            raise AdjudicationError("an alignment edge cannot connect a node to itself")
        adjacency.setdefault(edge.source_node_id, set()).add(edge.target_node_id)
        adjacency.setdefault(edge.target_node_id, set()).add(edge.source_node_id)

    components: list[tuple[str, ...]] = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in component:
                continue
            component.add(node_id)
            stack.extend(adjacency.get(node_id, ()))
        remaining -= component
        if len(component) >= 2:
            components.append(tuple(sorted(component)))

    return tuple(
        adjudicate_component(
            component,
            alignment_edges,
            annotation_candidates,
            stability_results,
            criteria,
            decided_by,
            node_metadata=node_metadata,
        )
        for component in sorted(components)
    )


def _merge_record(decision: OntologyDecision) -> DecisionRecord:
    return DecisionRecord(
        cluster_id=decision.decision_id,
        decision=decision.decision,
        members=list(decision.node_ids),
        rationale=decision.rationale,
        decided_by=(
            decision.override_operator
            if decision.manual_override and decision.override_operator
            else decision.decided_by
        ),
        representative=decision.representative_node_id,
        family_ambiguity=decision.relationship == "AMBIGUOUS_CROSS_FAMILY",
        threshold_sensitive=decision.decision is Decision.DEFERRED,
    )


def _scientific_artifact_id(
    kind: str,
    decisions: Sequence[OntologyDecision],
    provenance: Mapping[str, Any],
) -> str:
    payload = {
        "kind": kind,
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "provenance": dict(provenance),
        "decisions": [decision.to_dict() for decision in decisions],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{kind}:{hashlib.sha256(encoded).hexdigest()}"


def write_adjudication_artifacts(
    out_dir: str | Path,
    decisions: Sequence[OntologyDecision],
    *,
    provenance: Mapping[str, Any],
    review_path: str | Path = "review.yaml",
) -> tuple[Path, Path, Path]:
    """Write the full audit table, validated compile handoff, and human review."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    if not provenance:
        raise AdjudicationError("adjudication artifacts require provenance")
    decision_list = tuple(decisions)
    if len({decision.decision_id for decision in decision_list}) != len(decision_list):
        raise AdjudicationError("adjudication output contains duplicate decision_id values")
    ontology_artifact_id = _scientific_artifact_id(
        "ontology-decisions", decision_list, provenance
    )
    review_artifact_id = _scientific_artifact_id("review", decision_list, provenance)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ontology_path = out / "ontology_decisions.parquet"
    merge_path = out / "merge_decisions.json"
    review_dest = Path(review_path)
    if not review_dest.is_absolute():
        review_dest = out / review_dest
    review_dest.parent.mkdir(parents=True, exist_ok=True)

    ontology_rows = []
    for decision in decision_list:
        row = decision.to_dict()
        row["artifact_id"] = ontology_artifact_id
        for name in ("node_ids", "evidence_ids", "evidence_for", "evidence_against"):
            row[name] = json.dumps(row[name], separators=(",", ":"))
        row["provenance"] = json.dumps(
            {**dict(provenance), **dict(decision.provenance)},
            sort_keys=True,
            separators=(",", ":"),
        )
        ontology_rows.append(row)
    ontology_columns = [
        "artifact_id", "decision_id", "node_ids", "relationship", "decision",
        "automated_decision",
        "family_id", "representative_node_id", "criterion_id", "criterion_version",
        "evidence_ids", "evidence_for", "evidence_against", "rationale", "decided_by",
        "manual_override", "override_operator", "override_rationale", "provenance",
        "schema_version",
    ]
    table = pa.Table.from_pandas(
        pd.DataFrame(ontology_rows, columns=ontology_columns),
        preserve_index=False,
    )
    file_metadata = dict(table.schema.metadata or {})
    file_metadata.update({
        b"motifmultiverse.artifact_kind": b"ontology-decisions",
        b"motifmultiverse.artifact_id": ontology_artifact_id.encode("utf-8"),
        b"motifmultiverse.schema_version": ADJUDICATION_SCHEMA_VERSION.encode("utf-8"),
        b"motifmultiverse.provenance": json.dumps(
            dict(provenance), sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
    })
    pq.write_table(table.replace_schema_metadata(file_metadata), ontology_path)

    records = [_merge_record(decision) for decision in decision_list]
    artifact_id = decision_bundle_artifact_id(records, {}, provenance)
    bundle = DecisionBundle(
        schema_version=DECISION_BUNDLE_SCHEMA_VERSION,
        artifact_id=artifact_id,
        producer=DECISION_BUNDLE_PRODUCER,
        provenance=dict(provenance),
        decisions=records,
        tiers={},
    )
    merge_path.write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    review_dest.write_text(
        yaml.safe_dump(
            {
                "schema_version": ADJUDICATION_SCHEMA_VERSION,
                "artifact_id": review_artifact_id,
                "provenance": dict(provenance),
                "decisions": [decision.to_dict() for decision in decision_list],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return ontology_path, merge_path, review_dest


def _none_if_missing(value: Any) -> Any:
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _read_alignment_edges(path: Path) -> list[AlignmentEvidence]:
    import pandas as pd

    try:
        rows = pd.read_parquet(path).to_dict("records")
        edges = [
            AlignmentEvidence(**{
                key: _none_if_missing(value)
                for key, value in row.items()
            })
            for row in rows
        ]
        if any(not edge.is_calibrated for edge in edges):
            raise ValueError(
                "persisted alignment evidence must carry a calibrated p-value "
                "and positive null_shuffles"
            )
        return edges
    except (OSError, TypeError, ValueError) as exc:
        raise AdjudicationError(f"{path} is not valid alignment evidence: {exc}") from exc


def _read_annotation_candidates(path: Path) -> list[AnnotationCandidate]:
    import pandas as pd

    try:
        rows = pd.read_parquet(path).to_dict("records")
        candidates = []
        for row in rows:
            payload = {
                key: _none_if_missing(value)
                for key, value in row.items()
            }
            provenance = payload.get("provenance")
            if isinstance(provenance, str):
                payload["provenance"] = json.loads(provenance)
            candidates.append(AnnotationCandidate(**payload))
        return candidates
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AdjudicationError(f"{path} is not valid annotation evidence: {exc}") from exc


def _read_stability_results(path: Path) -> list[StabilityEvidence]:
    import pandas as pd

    if not path.exists():
        return []
    try:
        return [
            SimpleNamespace(**{
                key: _none_if_missing(value)
                for key, value in row.items()
            })
            for row in pd.read_parquet(path).to_dict("records")
        ]
    except (OSError, TypeError, ValueError) as exc:
        raise AdjudicationError(f"{path} is not valid stability evidence: {exc}") from exc


def _read_node_metadata(registry_dir: str | Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    registry_path = Path(registry_dir) / "registry.json"
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        raw_metadata = payload["registry_metadata"]
        raw_nodes = payload["nodes"]
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("registry_metadata must be a mapping")
        if "schema_version" not in raw_metadata:
            raise TypeError("registry_metadata requires schema_version")
        registry_metadata = RegistryMetadata(**raw_metadata)
        if registry_metadata.schema_version != REGISTRY_SCHEMA_VERSION:
            raise TypeError("registry schema version is not supported")
        if not isinstance(raw_nodes, list):
            raise TypeError("nodes must be a list")
        nodes = [MotifNode(**node) for node in raw_nodes]
        from motifmultiverse.guards import variant_id_unique

        identity_guard = variant_id_unique(nodes)
        if not identity_guard.passed:
            raise SchemaError(identity_guard.detail)
    except (OSError, KeyError, TypeError, SchemaError, json.JSONDecodeError) as exc:
        raise AdjudicationError(
            f"{registry_path} is not a readable motif registry: {exc}"
        ) from exc

    metadata: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if node.node_id in metadata:
            raise AdjudicationError(
                f"registry contains duplicate node_id {node.node_id!r}"
            )
        values = {
            "family_id": node.family_id,
            "variant_id": node.variant_id,
            "motif_completeness": node.motif_completeness,
            "seqlet_count": node.seqlet_count,
            "core_ic": node.core_ic,
            "cross_context_recurrence": node.cross_context_recurrence,
        }
        metadata[node.node_id] = {
            key: value for key, value in values.items() if value is not None
        }
    return registry_path, metadata


def adjudicate_evidence(
    evidence_dir: str | Path,
    out_dir: str | Path,
    *,
    registry_dir: str | Path,
    criteria_path: str | Path | None = None,
    review_path: str | Path = "review.yaml",
    policy: str = "conservative",
    decided_by: str = "motifmultiverse.adjudicate",
) -> tuple[OntologyDecision, ...]:
    """Read prior-stage evidence, adjudicate every component, and write artifacts."""
    from motifmultiverse.provenance import record, sha256_file
    from motifmultiverse.schema.criteria import load_criteria

    if policy != "conservative":
        raise AdjudicationError(
            "only conservative adjudication is defined; permissive policy is refused "
            "until criterion-safe semantics are frozen"
        )
    evidence = Path(evidence_dir)
    alignment_path = evidence / "alignment_edges.parquet"
    annotation_path = evidence / "annotation_candidates.parquet"
    stability_path = evidence / "stability_results.parquet"
    if criteria_path is None:
        criteria_path = Path(__file__).resolve().parents[3] / "config" / "criteria.v1.yaml"
    criteria_path = Path(criteria_path)
    registry_path, node_metadata = _read_node_metadata(registry_dir)

    provenance_record = record("adjudicate")
    try:
        input_paths = [alignment_path, annotation_path, criteria_path]
        input_paths.append(registry_path)
        for path in input_paths:
            provenance_record.add_input(path)
        if stability_path.exists():
            provenance_record.add_input(stability_path)
    except OSError:
        # A missing input cannot be checksummed, but the refused attempt still
        # receives exactly one provenance row.
        provenance_record.write(out_dir)
        raise
    provenance_record.write(out_dir)

    criteria = load_criteria(criteria_path)
    decisions = adjudicate_all(
        _read_alignment_edges(alignment_path),
        _read_annotation_candidates(annotation_path),
        _read_stability_results(stability_path),
        criteria,
        decided_by,
        node_metadata=node_metadata,
    )
    artifact_provenance = {
        "stage": "adjudicate",
        "policy": policy,
        "criteria_sha256": sha256_file(criteria_path),
        "inputs": dict(provenance_record.inputs),
        "software": dict(provenance_record.software),
    }
    write_adjudication_artifacts(
        out_dir,
        decisions,
        provenance=artifact_provenance,
        review_path=review_path,
    )
    return decisions


run = adjudicate_evidence
