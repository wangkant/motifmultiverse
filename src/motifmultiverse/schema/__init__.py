"""Node / edge / config schema, with the four rules that came from real failures.

See ``docs/DATA_MODEL.md``. The rules encoded here (T-12) are not stylistic:

1. ``variant_id`` is the only stable semantic identity, and is *marked* as such.
2. No semantics may be parsed out of an identifier string.
3. Missingness is four-state and never collapses to 0.
4. A decision must be able to express a REFUSAL, and confidence must be a measure.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from .identity import (
    DEFAULT_ATTRIBUTION_DERIVED_FEATURE_NAMES,
    IDENTITY_SCHEMA_VERSION,
    ClaimScope,
    RepresentationId,
    StatisticalLicense,
    VariantId,
    resolve_query_permissions,
)

__all__ = [
    "Missingness", "Decision", "Tier", "IdentityError", "SchemaError",
    "MotifNode", "EvidenceEdge", "DecisionRecord", "DecisionBundle", "OntologyDecision",
    "AnalysisConfig",
    "NamespacedId", "translate", "assert_no_key_parsing", "MISSING_SENTINEL",
    "SelectionProvenance", "OutputMode", "MOST_CONSERVATIVE_OUTPUT_MODE",
    "OUTPUT_MODE_BY_PROVENANCE", "output_mode_for", "HitRecord",
    "HIT_TABLE_COLUMNS", "PeakSetQuery", "HealthFloors",
    "MetaclusterState", "RegistryMetadata", "REGISTRY_SCHEMA_VERSION", "LexiconManifest",
    "LEXICON_MANIFEST_SCHEMA_VERSION", "UNION_ID_RE",
    "Estimator", "IMPLEMENTED_ESTIMATORS", "InferenceCapability", "ESTIMATOR_CAPABILITY",
    "MergeConfidence",
    "MERGE_CONFIDENCE_CRITERIA", "CRITERION_NOT_YET_DEFINED",
    "SensitivityTrigger", "sensitivity_triggers", "DECISION_BUNDLE_SCHEMA_VERSION",
    "DECISION_BUNDLE_PRODUCER", "decision_bundle_artifact_id",
    "StatisticalLicense", "ClaimScope", "RepresentationId", "VariantId",
    "resolve_query_permissions", "DEFAULT_ATTRIBUTION_DERIVED_FEATURE_NAMES",
    "IDENTITY_SCHEMA_VERSION",
    "SUBSTRATE_SCHEMA_VERSION", "JsonValue", "CallerSpecification", "HitSubstrateManifest",
    "SPLIT_MANIFEST_SCHEMA_VERSION", "SplitRole", "PeakSplitManifest",
    "build_peak_split_manifest", "peak_split_manifest_checksum",
]


def __getattr__(name: str) -> Any:
    """Lazily expose adjudication's schema object without an import cycle."""
    if name == "OntologyDecision":
        from motifmultiverse.adjudicate import OntologyDecision

        return OntologyDecision
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# An explicit sentinel. Never 0, never NaN, never "".
MISSING_SENTINEL = "NA"

_VARIANT_ID_RE = re.compile(r"^[A-Za-z0-9]+_[A-Za-z0-9]+_\d{2,}$")


class SchemaError(ValueError):
    """A structural violation of the data model."""


class IdentityError(SchemaError):
    """An identifier was used across a namespace boundary without translation."""


SPLIT_MANIFEST_SCHEMA_VERSION = "1"


class SplitRole(StrEnum):
    """One mutually exclusive primary-analysis role for a peak."""

    DISCOVERY = "DISCOVERY"
    ADJUDICATION = "ADJUDICATION"
    VALIDATION = "VALIDATION"
    INFERENCE = "INFERENCE"


def _require_sha256_digest(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError(f"{name} must be a lowercase SHA-256 digest")


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def peak_split_manifest_checksum(
    schema_version: str,
    assignments: Mapping[str, SplitRole],
) -> str:
    """Hash the complete, canonically ordered primary split declaration."""
    payload = {
        "assignments": [
            {"peak_id": peak_id, "role": role.value}
            for peak_id, role in sorted(assignments.items())
        ],
        "schema_version": schema_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PeakSplitManifest:
    """A sealed, content-addressed primary-analysis assignment of peak IDs."""

    schema_version: str
    assignments: Mapping[str, SplitRole]
    checksum: str

    def __post_init__(self) -> None:
        if self.schema_version != SPLIT_MANIFEST_SCHEMA_VERSION:
            raise SchemaError(
                "peak split manifest schema_version must be "
                f"{SPLIT_MANIFEST_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.assignments, Mapping) or not self.assignments:
            raise SchemaError("peak split manifest assignments must be a non-empty mapping")

        frozen_assignments: dict[str, SplitRole] = {}
        for peak_id, role in self.assignments.items():
            if not _is_nonempty_string(peak_id):
                raise SchemaError("peak split manifest peak IDs must be non-empty strings")
            if not isinstance(role, SplitRole):
                raise SchemaError("peak split manifest assignments must use SplitRole values")
            frozen_assignments[peak_id] = role
        expected = peak_split_manifest_checksum(self.schema_version, frozen_assignments)
        _require_sha256_digest("peak split manifest checksum", self.checksum)
        if self.checksum != expected:
            raise SchemaError("peak split manifest checksum does not match canonical assignments")
        object.__setattr__(self, "assignments", MappingProxyType(dict(sorted(frozen_assignments.items()))))


def build_peak_split_manifest(assignments: Mapping[str, SplitRole | str]) -> PeakSplitManifest:
    """Create a canonical manifest, accepting only exact enum spellings at I/O."""
    if not isinstance(assignments, Mapping):
        raise SchemaError("peak split manifest assignments must be a mapping")
    normalized: dict[str, SplitRole] = {}
    for peak_id, role in assignments.items():
        try:
            normalized[peak_id] = SplitRole(role)
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                f"peak split manifest role for {peak_id!r} is not a SplitRole"
            ) from exc
    checksum = peak_split_manifest_checksum(SPLIT_MANIFEST_SCHEMA_VERSION, normalized)
    return PeakSplitManifest(
        schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
        assignments=normalized,
        checksum=checksum,
    )


from .substrate import (  # noqa: E402  (SchemaError is the substrate validation base)
    SUBSTRATE_SCHEMA_VERSION,
    CallerSpecification,
    HitSubstrateManifest,
    JsonValue,
)


class Missingness(StrEnum):
    """Rule 3. Four distinguishable states; ``0`` is none of them.

    Collapsing these to 0 at table-build time is the exact defect that put
    undefined family shares into arm means as zeros in the reference
    implementation, where a coverage figure computed *after* the coercion then
    reported perfect coverage and so corroborated the error.
    """

    NOT_SEARCHED = "not_searched"
    NO_SEQUENCE_MATCH = "no_sequence_match"
    HIT_BELOW_FLOOR = "hit_below_floor"
    USED = "used"


class Decision(StrEnum):
    """Rule 4. ``REFUSE_MERGE`` exists so a refusal is a record, not an absence.

    In the reference implementation every row of the merge table was
    ``collapse``; refusals existed only as missing rows and were therefore
    indistinguishable from "never considered".
    """

    COLLAPSE = "collapse"
    REFUSE_MERGE = "refuse_merge"
    KEEP_SEPARATE_CURATOR_OVERRIDE = "keep_separate_curator_override"
    DEFERRED = "deferred"


class Tier(StrEnum):
    """T-13. Discovery support and analysis admission are different questions."""

    CORE = "core"
    EXPANDED = "expanded"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class NamespacedId:
    """Rule 2. An identifier that carries its own namespace.

    Bare strings are rejected at boundaries. ``motif_name`` from a hit caller and
    ``pattern_key`` from a discovery manifest are different namespaces that have
    collided before: a hit-caller row number was once matched against a manifest
    pattern id, filing one factor's evidence under another's name.
    """

    namespace: str
    value: str

    def __post_init__(self) -> None:
        if not self.namespace or not self.value:
            raise IdentityError("both namespace and value are required")
        if "/" in self.namespace:
            raise IdentityError(f"namespace must not contain '/': {self.namespace!r}")

    def __str__(self) -> str:
        return f"{self.namespace}:{self.value}"


def translate(ident: NamespacedId, table: dict[str, str], to_namespace: str) -> NamespacedId:
    """Explicit cross-namespace translation, raising on unknown keys.

    Rule 2: no implicit coercion and no positional fallback. An unknown key is an
    error, never a silently dropped row.
    """
    if not isinstance(ident, NamespacedId):
        raise IdentityError("refusing to translate a bare string; wrap it in NamespacedId")
    if ident.value not in table:
        raise IdentityError(
            f"{ident} has no entry in the {ident.namespace}->{to_namespace} table; "
            "refusing to guess"
        )
    return NamespacedId(to_namespace, table[ident.value])


def assert_no_key_parsing(expression: str) -> None:
    """Rule 2, runtime twin of ``guards.no_key_parsing``."""
    banned = (".split(", "[:", "startswith(", "endswith(", "slice(")
    if any(b in expression for b in banned):
        raise IdentityError(
            f"semantics parsed out of an identifier string: {expression!r}; "
            "use an explicit translation table instead"
        )


@dataclass
class MotifNode:
    """Six field groups; see ``docs/DATA_MODEL.md``."""

    node_id: str
    model: str
    readout: str
    context: str
    metacluster: str
    denovo_pattern_id: str
    variant_id: str
    family_id: str
    cwm: Any = None
    hypothetical_cwm: Any = None
    ppm: Any = None
    trimmed_core: Any = None
    motif_length: int | None = None
    seqlet_count: int | None = None
    core_ic: float | None = None
    motif_completeness: float | None = None
    cross_context_recurrence: int | None = None
    source_peak_count: int | None = None
    putative_tf_label: str = MISSING_SENTINEL
    annotation_matches: dict[str, Any] = field(default_factory=dict)
    low_confidence_annotation: bool = False
    family_assignment_source: str = MISSING_SENTINEL
    family_assignment_confidence: float | None = None
    discovery_tier: Tier | None = None
    analysis_tier: Tier | None = None
    tier_reason: str = MISSING_SENTINEL
    exclusion_reason: str = MISSING_SENTINEL
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _VARIANT_ID_RE.match(self.variant_id):
            raise SchemaError(
                f"variant_id {self.variant_id!r} is not of the form <UNION>_<FAMILY>_<NN>. "
                "It is the only stable semantic identity in the model and must be explicit."
            )
        if self.discovery_tier is not None and self.analysis_tier is not None:
            if self.discovery_tier != self.analysis_tier and self.tier_reason == MISSING_SENTINEL:
                raise SchemaError(
                    "discovery_tier != analysis_tier requires an explicit tier_reason"
                )
        if self.family_assignment_confidence is not None:
            if not 0.0 <= self.family_assignment_confidence <= 1.0:
                raise SchemaError("family_assignment_confidence must be a measure in [0, 1]")
        if self.motif_completeness is not None:
            if (
                isinstance(self.motif_completeness, bool)
                or not isinstance(self.motif_completeness, (int, float))
                or not 0.0 <= self.motif_completeness <= 1.0
            ):
                raise SchemaError("motif_completeness must be a measure in [0, 1]")
        if self.seqlet_count is not None and (
            isinstance(self.seqlet_count, bool)
            or not isinstance(self.seqlet_count, int)
            or self.seqlet_count < 0
        ):
            raise SchemaError("seqlet_count must be a non-negative integer")
        if self.core_ic is not None and (
            isinstance(self.core_ic, bool)
            or not isinstance(self.core_ic, (int, float))
            or not math.isfinite(self.core_ic)
            or self.core_ic < 0
        ):
            raise SchemaError("core_ic must be a non-negative measure")
        if self.cross_context_recurrence is not None and (
            isinstance(self.cross_context_recurrence, bool)
            or not isinstance(self.cross_context_recurrence, int)
            or self.cross_context_recurrence < 1
        ):
            raise SchemaError("cross_context_recurrence must be a positive integer")

    @property
    def identity(self) -> NamespacedId:
        return NamespacedId("variant_id", self.variant_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceEdge:
    """One of six evidence classes; see ``docs/DATA_MODEL.md``."""

    VALID_TYPES = (
        "alignment", "sequence_hit", "attribution_hit",
        "downstream_sensitivity", "external_biology", "decision",
    )

    edge_type: str
    source: str
    target: str
    statistic: float | None = None
    uncertainty: tuple[float, float] | None = None
    missingness: Missingness = Missingness.USED
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.edge_type not in self.VALID_TYPES:
            raise SchemaError(f"unknown edge_type {self.edge_type!r}; expected {self.VALID_TYPES}")
        if self.missingness is not Missingness.USED and self.statistic == 0:
            raise SchemaError(
                f"missingness={self.missingness.value} recorded with statistic=0; "
                "undefined values take an explicit sentinel, never 0"
            )


def _reject_unknown_keys(payload: Mapping[str, Any], allowed: set[str], what: str) -> None:
    """Shared by every ``from_dict`` in this module: an unrecognised key is refused,
    not silently ignored -- that is how a renamed field becomes a no-op.
    """
    unknown = set(payload) - allowed
    if unknown:
        raise SchemaError(
            f"{what} has unknown key(s) {sorted(unknown)}; expected a subset of "
            f"{sorted(allowed)}."
        )


@dataclass
class DecisionRecord:
    """Rule 4. Refusals are rows. Confidence is a measure, not a name lookup.

    Note the two separate fields, which is the whole point. ``confidence`` is a
    **measure** in [0, 1] and may be compared with a number. ``merge_confidence``
    is a **grade** and may not: it is dispatched on by name. The reference
    implementation's failure was one value doing both jobs -- a two-label lookup
    that a downstream gate read as though it were a measure.
    """

    cluster_id: str
    decision: Decision
    members: list[str]
    rationale: str
    decided_by: str
    confidence: float | None = None
    merge_confidence: MergeConfidence | None = None
    #: The observed medoid a collapse cluster survives as. Required, and must be
    #: one of ``members``, whenever ``decision`` is ``COLLAPSE`` -- see
    #: ``__post_init__``. Never set from a decision that does not collapse.
    representative: str | None = None
    family_ambiguity: bool = False
    threshold_sensitive: bool = False

    def __post_init__(self) -> None:
        # Coerce/validate unconditionally, not just for `str`: `Decision` is a
        # `StrEnum`, so a member is *also* `isinstance(..., str)` and this call
        # is idempotent for one. A mistyped or wrong-case value (``"COLLAPSE"``)
        # would otherwise compare unequal to every real `Decision` value
        # downstream and be silently treated as a no-op decision (round-1 review).
        try:
            self.decision = Decision(self.decision)
        except ValueError as exc:
            raise SchemaError(
                f"decision {self.decision!r} is not one of "
                f"{[d.value for d in Decision]}"
            ) from exc
        if not self.rationale.strip():
            raise SchemaError("every decision, including a refusal, requires a rationale")
        if not self.decided_by.strip():
            raise SchemaError("every decision requires decided_by")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise SchemaError(
                "confidence must be a measure in [0, 1]. A per-family name lookup is not a "
                "measure, and downstream gates must not read one."
            )
        if isinstance(self.merge_confidence, str):
            try:
                self.merge_confidence = MergeConfidence(self.merge_confidence)
            except ValueError as exc:
                raise SchemaError(
                    f"merge_confidence {self.merge_confidence!r} is not one of "
                    f"{[g.value for g in MergeConfidence]}. It is a grade, not a number: "
                    "there is no scalar to threshold here (see MERGE_CONFIDENCE_CRITERIA)."
                ) from exc
        if self.decision == Decision.COLLAPSE:
            if self.representative is None:
                raise SchemaError(
                    f"collapse {self.cluster_id!r} names no representative; a collapse "
                    "decision must record which observed medoid the surviving motif is "
                    "(FP-05)."
                )
            if self.representative not in self.members:
                raise SchemaError(
                    f"collapse {self.cluster_id!r} names representative "
                    f"{self.representative!r}, which is not one of its members. The "
                    "representative is an observed medoid, never a constructed average "
                    "(FP-05)."
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DecisionRecord:
        """Parse one decision, refusing any key this record does not define.

        A renamed or misspelled field must fail loudly here rather than being
        silently dropped and read back as "not decided".
        """
        allowed = {f.name for f in fields(cls)}
        _reject_unknown_keys(payload, allowed, f"decision {payload.get('cluster_id')!r}")
        return cls(**payload)


#: ``DecisionBundle`` schema revision. Bumped only if the payload shape changes.
DECISION_BUNDLE_SCHEMA_VERSION = "3"
DECISION_BUNDLE_PRODUCER = "motifmultiverse.adjudicate"

#: The only keys a decisions payload may declare. Anything else is refused rather
#: than ignored, per the same rule as ``DecisionRecord.from_dict``.
_DECISION_BUNDLE_KEYS = frozenset({
    "schema_version", "artifact_id", "producer", "provenance", "decisions", "tiers",
})


def decision_bundle_artifact_id(
    decisions: Sequence[Mapping[str, Any] | DecisionRecord],
    tiers: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    """Full content identity for an adjudication bundle, including provenance."""
    normalized_decisions = [
        asdict(decision) if isinstance(decision, DecisionRecord) else dict(decision)
        for decision in decisions
    ]
    payload = {
        "schema_version": DECISION_BUNDLE_SCHEMA_VERSION,
        "producer": DECISION_BUNDLE_PRODUCER,
        "provenance": dict(provenance),
        "decisions": normalized_decisions,
        "tiers": dict(tiers),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"merge-decisions:{hashlib.sha256(encoded).hexdigest()}"

#: The only keys one node's `tiers` override may declare.
_TIER_OVERRIDE_KEYS = frozenset({"discovery_tier", "analysis_tier", "tier_reason"})

#: What `discovery_tier` / `analysis_tier` may legally say.
_VALID_TIER_VALUES = frozenset(t.value for t in Tier)


def _validate_tier_overrides(tiers: Mapping[str, Any]) -> None:
    """Every ``tiers`` override must be a recognised key naming a real Tier value.

    ``compile._apply_tiers`` copies an override's ``analysis_tier`` /
    ``discovery_tier`` onto the node it names and then matches it by exact
    string equality against ``Tier.CORE.value`` / ``Tier.EXPANDED.value`` in
    ``_members_for_tier``. A typo'd value (``"coree"`` for ``"core"``) matches
    neither branch, so the node it names silently disappears from every tier
    at once with no error -- the same silent-content-change failure this
    schema exists to refuse for decisions, just reached through the sibling
    field (round-1 review finding 1b).
    """
    for node_id, override in tiers.items():
        if not isinstance(override, Mapping):
            raise SchemaError(
                f"tier override for node {node_id!r} must be a mapping, got "
                f"{type(override).__name__}"
            )
        _reject_unknown_keys(
            override, set(_TIER_OVERRIDE_KEYS), f"tier override for node {node_id!r}"
        )
        for key in ("discovery_tier", "analysis_tier"):
            if key in override and override[key] not in _VALID_TIER_VALUES:
                raise SchemaError(
                    f"tier override for node {node_id!r}: {key} {override[key]!r} "
                    f"is not a valid Tier ({sorted(_VALID_TIER_VALUES)})"
                )


@dataclass
class DecisionBundle:
    """A decisions JSON payload, parsed and validated once before ``compile`` applies it.

    ``compile`` used to hand ``payload.get("decisions")`` -- a bare list of dicts --
    straight to its tiering logic, which tolerated a decision naming a node that no
    longer exists in the registry, and tolerated two collapse decisions both
    claiming the same node. Both are silent content changes to the compiled
    lexicon, and both are refused here instead: two decisions cannot claim the same
    ``cluster_id``, and a node cannot be claimed by two different collapse clusters
    at once, because a node's fate is decided exactly once.

    What this class does **not** know is the registry: a decision naming a node
    that has never existed at all is ``compile``'s check (it is the only place
    that has the registry's node ids), and so is whether a tier-specific
    representative actually survives into the tier being compiled.
    """

    schema_version: str
    decisions: list[DecisionRecord]
    tiers: dict[str, dict[str, str]]
    artifact_id: str | None = None
    producer: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> DecisionBundle:
        payload = payload or {}
        _reject_unknown_keys(payload, set(_DECISION_BUNDLE_KEYS), "decisions payload")
        schema_version = str(payload.get("schema_version") or DECISION_BUNDLE_SCHEMA_VERSION)
        if schema_version != DECISION_BUNDLE_SCHEMA_VERSION:
            raise SchemaError(
                f"decision bundle schema_version must be {DECISION_BUNDLE_SCHEMA_VERSION!r}; "
                f"got {schema_version!r}"
            )
        decisions = [d if isinstance(d, DecisionRecord) else DecisionRecord.from_dict(d)
                     for d in (payload.get("decisions") or [])]

        seen_clusters: set[str] = set()
        member_owner: dict[str, str] = {}
        for d in decisions:
            if d.cluster_id in seen_clusters:
                raise SchemaError(
                    f"duplicate cluster_id {d.cluster_id!r} in decisions; each cluster "
                    "is decided exactly once"
                )
            seen_clusters.add(d.cluster_id)
            if d.decision != Decision.COLLAPSE:
                continue
            for member in d.members:
                owner = member_owner.get(member)
                if owner is not None and owner != d.cluster_id:
                    raise SchemaError(
                        f"node {member!r} belongs to multiple collapse clusters "
                        f"({owner!r} and {d.cluster_id!r}); a node's fate is decided once"
                    )
                member_owner[member] = d.cluster_id

        tiers = dict(payload.get("tiers") or {})
        _validate_tier_overrides(tiers)

        return cls(
            schema_version=schema_version,
            decisions=decisions,
            tiers=tiers,
            artifact_id=payload.get("artifact_id"),
            producer=payload.get("producer"),
            provenance=dict(payload.get("provenance") or {}),
        )

    @classmethod
    def from_adjudication_artifact(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> DecisionBundle:
        """Validate the exact identity-bearing handoff emitted by adjudicate."""
        if not isinstance(payload, Mapping):
            raise SchemaError("merge decisions artifact must be a JSON object")
        required = {"schema_version", "artifact_id", "producer", "provenance", "decisions", "tiers"}
        missing = required - set(payload)
        if missing:
            raise SchemaError(
                f"merge decisions artifact missing required key(s) {sorted(missing)}"
            )
        schema_version = str(payload["schema_version"])
        if schema_version != DECISION_BUNDLE_SCHEMA_VERSION:
            raise SchemaError(
                f"decision bundle schema_version must be {DECISION_BUNDLE_SCHEMA_VERSION!r}; "
                f"got {schema_version!r}"
            )
        if payload["producer"] != DECISION_BUNDLE_PRODUCER:
            raise SchemaError(
                f"merge decisions producer must be {DECISION_BUNDLE_PRODUCER!r}; "
                f"got {payload['producer']!r}"
            )
        if not payload["provenance"]:
            raise SchemaError("merge decisions artifact requires provenance")
        expected_id = decision_bundle_artifact_id(
            payload["decisions"], payload["tiers"], payload["provenance"]
        )
        if payload["artifact_id"] != expected_id:
            raise SchemaError(
                "merge decisions artifact_id does not match its decisions, tiers, and provenance"
            )
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "producer": self.producer,
            "provenance": dict(self.provenance),
            "decisions": [
                {
                    **asdict(decision),
                    "decision": decision.decision.value,
                    "merge_confidence": (
                        decision.merge_confidence.value
                        if decision.merge_confidence is not None
                        else None
                    ),
                }
                for decision in self.decisions
            ],
            "tiers": self.tiers,
        }


#: A union id names one model/readout's shared lexicon namespace. It is declared in
#: the project config, never derived from a filename or an analysis id -- deriving it
#: would be parsing semantics out of an identifier, which is exactly ``BA-11``.
UNION_ID_RE = re.compile(r"^[A-Za-z0-9]+$")


class MetaclusterState(StrEnum):
    """Why a metacluster group is not contributing patterns. Three distinct absences.

    Collapsing these is the discovery-stage form of ``BA-01``. In the reference
    implementation four discovery leaves had **no** negative group in the HDF5 at
    all, which is not the same claim as an empty one:

    - ``GROUP_ABSENT`` -- discovery ran and the group never formed. The seqlet
      count did not clear the metacluster admission gate. Evidence about the
      *gate*, not about the sequence.
    - ``GROUP_EMPTY`` -- the group exists and contains no patterns. Discovery
      looked and found nothing.
    - ``NOT_SEARCHED`` -- this run never looked. No evidence either way.

    Read as "no negative motifs", all three become the same false statement.
    """

    PRESENT = "present"
    GROUP_ABSENT = "group_absent"
    GROUP_EMPTY = "group_empty"
    NOT_SEARCHED = "not_searched"


#: Persisted motif-registry schema revision. The metadata object carries this
#: value and registry readers require the key, so tie-breaking fields cannot be
#: silently absent under an old, unversioned shape.
REGISTRY_SCHEMA_VERSION = "1"


@dataclass
class RegistryMetadata:
    """What ``ingest`` emits alongside the motif nodes.

    ``cross_model_claims_restricted`` is the runtime form of the N >= 3 rule: it
    travels **with the data** so a downstream step does not have to remember to
    ask. It is derived, not supplied, and the constructor refuses a value that
    disagrees with ``n_models``.
    """

    project: str
    peak_universe_id: str
    analyses: list[dict[str, Any]]
    n_models: int
    cross_model_claims_restricted: bool
    metacluster_states: dict[str, dict[str, str]]
    trim_threshold: float
    schema_version: str = REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REGISTRY_SCHEMA_VERSION:
            raise SchemaError(
                f"registry schema_version must be {REGISTRY_SCHEMA_VERSION!r}; "
                f"got {self.schema_version!r}"
            )
        if self.cross_model_claims_restricted != (self.n_models < 3):
            raise SchemaError(
                f"cross_model_claims_restricted={self.cross_model_claims_restricted} "
                f"contradicts n_models={self.n_models}; it is derived, not declared"
            )
        for analysis_id, states in self.metacluster_states.items():
            for group, state in states.items():
                if state not in {s.value for s in MetaclusterState}:
                    raise SchemaError(
                        f"{analysis_id}/{group}: {state!r} is not a MetaclusterState. "
                        "An absent group, an empty group and an unsearched group are "
                        "three different claims and none of them is 'no motifs'."
                    )


#: ``LexiconManifest`` schema revision. Bumped only if the manifest shape changes.
#: Also the ``schema_version`` folded into ``lexicon_content_hash``'s metadata
#: blob, so a future shape change changes identity too, not just the on-disk JSON.
LEXICON_MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass
class LexiconManifest:
    """What ``compile`` emits beside each lexicon H5.

    ``lexicon_content_hash`` exists so ``FP-11`` can eventually be enforced: every
    family-level number must state the lexicon version it was computed under, and
    without a content hash there is nothing for it to state. The hash is
    **semantic**, not just an array checksum: two lexicons built from identical
    motif arrays but read back under different loader settings
    (``trim_threshold``, ``motif_type``, ``include_rc``, ``loader_parameters``)
    load differently and must not share an identity. Those settings are therefore
    recorded on the manifest -- not only folded into the hash -- so *what* was
    hashed is legible without recomputing it.

    ``comparisons`` exists because a tier contrast that changes nothing must say
    so. In the reference implementation ``core`` and ``expanded`` had **identical
    positive sets**, so the sensitivity analysis that looked like it tested lexicon
    width tested only the negative half -- and nothing in the artifact said that.
    """

    tier: str
    lexicon_content_hash: str
    n_motifs: int
    pattern_order: list[str]
    node_ids: list[str]
    schema_version: str
    trim_threshold: float
    motif_type: str
    include_rc: bool
    loader_backend: str
    loader_parameters: dict[str, Any]
    comparisons: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_registry: str = MISSING_SENTINEL
    #: cluster_id -> the named triggers that keep it split in the sensitivity
    #: lexicon. Recorded so "why is this tier different" is answered by the
    #: artifact rather than by re-deriving it from a threshold that does not exist.
    sensitivity_triggers: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.pattern_order) != self.n_motifs:
            raise SchemaError(
                f"{self.tier}: n_motifs={self.n_motifs} but {len(self.pattern_order)} "
                "pattern names; the manifest must describe the file it accompanies"
            )
        if len(set(self.pattern_order)) != len(self.pattern_order):
            raise SchemaError(f"{self.tier}: duplicate pattern names in the index")
        if len(self.node_ids) != self.n_motifs:
            raise SchemaError(f"{self.tier}: node_ids and pattern_order disagree in length")


#: Recorded where a criterion is genuinely absent from the design, rather than
#: filled with something plausible. A written-down gap can be closed by whoever
#: owns the design; an invented threshold looks closed and is not.
CRITERION_NOT_YET_DEFINED = "CRITERION_NOT_YET_DEFINED"


class MergeConfidence(StrEnum):
    """How confident a merge decision is -- **categorical**, not a scalar.

    There is deliberately no numeric threshold anywhere in this package for this
    field. The design lists "moderate-confidence merge" as one trigger for a
    sensitivity lexicon but never defines it and never says it is a scalar, and in
    the reference implementation the value was produced by
    ``"moderate" if family == "ZNF76" else "high"`` -- a lookup by name over a
    hard-coded singleton. There is no continuous quantity here to compare against,
    so any cut-off (0.8, 0.7, anything) would smuggle in a premise the design does
    not contain.

    Being a :class:`StrEnum` gives that some teeth: ``grade < 0.8`` raises
    ``TypeError`` rather than quietly comparing. Downstream dispatches **by name**.

    See :data:`MERGE_CONFIDENCE_CRITERIA` for what is still missing.
    """

    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


#: What evidence combination earns each grade. **Nothing here is decided yet.**
#: This is the first recorded design gap in this repository, and it is recorded
#: rather than filled: assigning the grade belongs to ``adjudicate``, whose
#: criteria are a preregistration item (``FP-13``). ``compile`` only *dispatches*
#: on a declared grade; it never assigns one.
MERGE_CONFIDENCE_CRITERIA: dict[str, str] = {
    MergeConfidence.HIGH.value: CRITERION_NOT_YET_DEFINED,
    MergeConfidence.MODERATE.value: CRITERION_NOT_YET_DEFINED,
    MergeConfidence.LOW.value: CRITERION_NOT_YET_DEFINED,
}


class SensitivityTrigger(StrEnum):
    """Why a merge is left split in the sensitivity lexicon.

    Three named conditions, taken from the design's own list, evaluated
    independently. Any one of them is enough. None of them is a number.
    """

    MERGE_CONFIDENCE_NOT_HIGH = "merge_confidence_not_high"
    FAMILY_AMBIGUITY = "family_ambiguity"
    THRESHOLD_SENSITIVE = "threshold_sensitive"


def sensitivity_triggers(decision: Mapping[str, Any]) -> list[str]:
    """Which triggers fire for one decision. Empty means the merge is safe to apply.

    An **undeclared** ``merge_confidence`` fires the first trigger. That is the
    conservative branch and it is deliberate: "not stated" is not "high", and the
    cost of being wrong is one extra sensitivity lexicon rather than a merge
    nobody checked.
    """
    fired = []
    grade = decision.get("merge_confidence")
    if grade != MergeConfidence.HIGH.value:
        fired.append(SensitivityTrigger.MERGE_CONFIDENCE_NOT_HIGH.value)
    if decision.get("family_ambiguity"):
        fired.append(SensitivityTrigger.FAMILY_AMBIGUITY.value)
    if decision.get("threshold_sensitive"):
        fired.append(SensitivityTrigger.THRESHOLD_SENSITIVE.value)
    return fired


class Estimator(StrEnum):
    """Every interval estimator this project recognises, implemented or not.

    Enumerated rather than left as a free string so that a caller can branch on
    the value today and keep working when ``FP-15``'s estimators arrive. A result
    names the one that actually ran; ``IMPLEMENTED_ESTIMATORS`` says which of these
    exist in this release, and it is deliberately smaller than the enum.
    """

    PERCENTILE_BLOCK_BOOTSTRAP = "percentile_block_bootstrap"
    #: FP-15's specified interval. Not implemented; see docs/ROADMAP.md.
    BCA_PAIRED_BLOCK_BOOTSTRAP = "bca_paired_block_bootstrap"
    #: FP-15's specified p value. Not implemented; see docs/ROADMAP.md.
    WILD_CLUSTER_BOOTSTRAP_T = "wild_cluster_bootstrap_t"


#: Label permutation is absent on purpose: it is not unimplemented, it is
#: **abandoned**. Under block-correlated structure it understates the variance,
#: so adding it back would be a regression rather than a feature.
IMPLEMENTED_ESTIMATORS = frozenset({Estimator.PERCENTILE_BLOCK_BOOTSTRAP})


class InferenceCapability(StrEnum):
    """What a result's uncertainty numbers license a reader to do with them.

    A percentile bootstrap's replicate tail is not a calibrated hypothesis
    test: the proportion of replicates crossing zero looks like a *p* value
    but has none of the guarantees one requires. ``ESTIMATION_ONLY`` results
    may report a point estimate and an interval; only ``INTERVAL_AND_TEST``
    results may additionally report ``p_value`` / ``q_value``. This is
    strictly narrower than "an estimator was implemented" -- a result must be
    *licensed* for a test, not merely produced by code capable of a division.
    """

    ESTIMATION_ONLY = "ESTIMATION_ONLY"
    INTERVAL_AND_TEST = "INTERVAL_AND_TEST"


#: Which capability each recognised estimator is licensed for (`FP-15`). The
#: percentile and BCa bootstraps are interval estimators only; the block-level
#: wild cluster bootstrap-*t* is `FP-15`'s specified hypothesis test. Declared
#: for every value in :class:`Estimator`, implemented or not, so a caller does
#: not have to wait for an estimator to ship to know what it will be licensed
#: to emit.
ESTIMATOR_CAPABILITY: dict[Estimator, InferenceCapability] = {
    Estimator.PERCENTILE_BLOCK_BOOTSTRAP: InferenceCapability.ESTIMATION_ONLY,
    Estimator.BCA_PAIRED_BLOCK_BOOTSTRAP: InferenceCapability.ESTIMATION_ONLY,
    Estimator.WILD_CLUSTER_BOOTSTRAP_T: InferenceCapability.INTERVAL_AND_TEST,
}


class SelectionProvenance(StrEnum):
    """How the peak set under analysis was chosen (``FP-20``, ``BA-16``).

    This is not metadata. It decides the output mode: if the criterion that
    selected a peak set came from the same signal being measured, an inference
    over that set can be statistically valid and semantically circular.

    ``DECLARATION_MISSING`` is not one of the report's grades. It is the recorded
    state of a query that declared nothing, and it exists so that "undeclared" is
    a value in the record rather than an absence -- the same reason
    :class:`Decision` has ``REFUSE_MERGE``.
    """

    EXTERNAL = "EXTERNAL"
    PROGRAMMATIC_RULE = "PROGRAMMATIC_RULE"
    CLUSTERED_WITH_SPLIT = "CLUSTERED_WITH_SPLIT"
    CLUSTERED_NO_SPLIT = "CLUSTERED_NO_SPLIT"
    EYEBALLED = "EYEBALLED"
    MODEL_SELECTED_NO_TRANSCRIPT = "MODEL_SELECTED_NO_TRANSCRIPT"
    DECLARATION_MISSING = "DECLARATION_MISSING"


class OutputMode(StrEnum):
    """What a query is allowed to emit, given its selection provenance."""

    FULL_INFERENCE = "FULL_INFERENCE"
    FULL_INFERENCE_HELD_OUT = "FULL_INFERENCE_HELD_OUT"
    DESCRIPTIVE_ONLY = "DESCRIPTIVE_ONLY"
    DESCRIPTIVE_ONLY_UNVERIFIABLE_CONDITIONING = "DESCRIPTIVE_ONLY_UNVERIFIABLE_CONDITIONING"


#: The floor. An undeclared or unrecognised grade lands here, never in FULL_INFERENCE.
MOST_CONSERVATIVE_OUTPUT_MODE = OutputMode.DESCRIPTIVE_ONLY_UNVERIFIABLE_CONDITIONING

#: The dispatch table. ``MODEL_SELECTED_NO_TRANSCRIPT`` is stricter than ``EYEBALLED``:
#: a human selector can at least testify afterwards to what they looked at, whereas an
#: agent's conditioning set cannot be reconstructed and in particular cannot be shown
#: not to have already included downstream information.
OUTPUT_MODE_BY_PROVENANCE: dict[SelectionProvenance, OutputMode] = {
    SelectionProvenance.EXTERNAL: OutputMode.FULL_INFERENCE,
    SelectionProvenance.PROGRAMMATIC_RULE: OutputMode.FULL_INFERENCE,
    SelectionProvenance.CLUSTERED_WITH_SPLIT: OutputMode.FULL_INFERENCE_HELD_OUT,
    SelectionProvenance.CLUSTERED_NO_SPLIT: OutputMode.DESCRIPTIVE_ONLY,
    SelectionProvenance.EYEBALLED: OutputMode.DESCRIPTIVE_ONLY,
    SelectionProvenance.MODEL_SELECTED_NO_TRANSCRIPT: MOST_CONSERVATIVE_OUTPUT_MODE,
    SelectionProvenance.DECLARATION_MISSING: MOST_CONSERVATIVE_OUTPUT_MODE,
}


def output_mode_for(grade: Any) -> OutputMode:
    """Resolve an output mode, failing safe.

    Anything unrecognised -- ``None``, an empty string, a grade from a future
    revision of the ledger -- resolves to the most conservative mode. It must
    never resolve to the most permissive one, which is what a plain
    ``dict.get(grade, EXTERNAL)`` would eventually do.
    """
    try:
        return OUTPUT_MODE_BY_PROVENANCE[SelectionProvenance(grade)]
    except (ValueError, KeyError):
        return MOST_CONSERVATIVE_OUTPUT_MODE


def _output_mode_from_permissions(
    statistical_license: StatisticalLicense, claim_scope: ClaimScope,
) -> OutputMode:
    """The old four-state grade, as a *read* of the two independent axes.

    Retained for one release so an existing reader keyed on ``output_mode``
    (:class:`PeakSetQuery` and :class:`~motifmultiverse.interpret.Interpretation`
    both still emit it) does not break. This function computes nothing of its
    own: it only maps ``(statistical_license, claim_scope)`` onto the value
    ``output_mode_for(selection_provenance)`` already produced, so it is a
    compatibility view, not a second source of truth.

    ``ClaimScope`` has no representation in the old four states beyond
    ``CONDITIONING_UNVERIFIABLE`` (the old ``*_UNVERIFIABLE_CONDITIONING``
    mode); a claim scope of ``SUBSTRATE_CIRCULAR`` has no legacy equivalent and
    is intentionally invisible here -- that lost distinction is exactly why
    ``claim_scope`` is emitted as its own field now instead of being folded back
    into this one.
    """
    if claim_scope is ClaimScope.CONDITIONING_UNVERIFIABLE:
        return MOST_CONSERVATIVE_OUTPUT_MODE
    if statistical_license is StatisticalLicense.FULL_INFERENCE:
        return OutputMode.FULL_INFERENCE
    if statistical_license is StatisticalLicense.HELD_OUT_INFERENCE:
        return OutputMode.FULL_INFERENCE_HELD_OUT
    return OutputMode.DESCRIPTIVE_ONLY


#: The hit-table contract. A row is either a called instance (``USED``) or the
#: record of a peak that was searched and produced nothing. The second kind is not
#: optional: without it the peak universe silently loses its zero-hit peaks and
#: every coverage figure computed from the table is inflated.
HIT_TABLE_COLUMNS = (
    "region_id", "chrom", "start", "end",
    "variant_id", "family_id", "hit_coefficient",
    "missingness", "input_scale", "lexicon_id", "substrate_id",
)


@dataclass
class HitRecord:
    """One row of a frozen hit table.

    The four-state rule is enforced here rather than left to the caller: an
    undefined coefficient is ``None``, never ``0.0``. A zero is a measurement.
    """

    region_id: str
    chrom: str
    start: int
    end: int
    missingness: Missingness
    input_scale: int
    lexicon_id: str
    substrate_id: str = MISSING_SENTINEL
    variant_id: str = MISSING_SENTINEL
    family_id: str = MISSING_SENTINEL
    hit_coefficient: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.missingness, str):
            self.missingness = Missingness(self.missingness)
        if self.start >= self.end:
            raise SchemaError(f"{self.region_id}: start {self.start} >= end {self.end}")
        if self.substrate_id != MISSING_SENTINEL:
            if (len(self.substrate_id) != 64
                    or any(c not in "0123456789abcdef" for c in self.substrate_id)):
                raise SchemaError(f"{self.region_id}: substrate_id must be a lowercase SHA-256 digest")
        if self.missingness is Missingness.USED:
            if self.variant_id == MISSING_SENTINEL:
                raise SchemaError(f"{self.region_id}: a used hit must name a variant_id")
            if self.hit_coefficient is None:
                raise SchemaError(f"{self.region_id}/{self.variant_id}: used hit with no coefficient")
        elif self.hit_coefficient is not None:
            raise SchemaError(
                f"{self.region_id}: missingness={self.missingness.value} carries "
                f"hit_coefficient={self.hit_coefficient!r}. An undefined value takes no "
                "number at all -- writing 0.0 here is the coercion this encoding exists to stop."
            )

    def block(self, block_size: int) -> tuple[str, int]:
        """Whole-block assignment: peaks are never split across blocks (``FP-20``)."""
        if block_size <= 0:
            raise SchemaError(f"block_size must be positive, got {block_size}")
        return (self.chrom, self.start // block_size)


@dataclass
class PeakSetQuery:
    """A peak set submitted for interpretation, with how it was chosen.

    ``selection_provenance`` has no default. Leaving it out is a legal state, but
    it is a *recorded* state (``DECLARATION_MISSING``) that costs the query its
    inference, not a field that quietly inherits the permissive grade.
    """

    query_id: str
    region_ids: list[str]
    selection_provenance: SelectionProvenance = SelectionProvenance.DECLARATION_MISSING
    selection_rule: str = MISSING_SENTINEL
    comparator_id: str = MISSING_SENTINEL
    comparator_region_ids: list[str] = field(default_factory=list)
    held_out_region_ids: list[str] = field(default_factory=list)
    #: Names of the feature(s) used to select this peak set, e.g.
    #: ``["attribution_pc1"]``. Defaults to empty, which preserves the behaviour
    #: of every caller that predates this field: with no declared feature,
    #: :func:`resolve_query_permissions` can never find a match in an
    #: attribution-derived registry, so ``claim_scope`` falls out of
    #: ``selection_provenance`` exactly as ``output_mode`` used to.
    selection_feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise SchemaError("a query needs an id; it is what a comparator refers to")
        if not isinstance(self.selection_provenance, SelectionProvenance):
            try:
                self.selection_provenance = SelectionProvenance(self.selection_provenance)
            except (TypeError, ValueError):
                self.selection_provenance = SelectionProvenance.DECLARATION_MISSING
        if (self.selection_provenance is SelectionProvenance.PROGRAMMATIC_RULE
                and self.selection_rule == MISSING_SENTINEL):
            raise SchemaError(
                "PROGRAMMATIC_RULE requires selection_rule: the grade's whole claim is that "
                "an executable rule chose the set without seeing results, so the rule is the evidence"
            )

    @property
    def statistical_license(self) -> StatisticalLicense:
        """May this query's numbers support inference at all (T-07, Task 7)."""
        statistical_license, _ = resolve_query_permissions(
            self.selection_provenance, self.selection_feature_names,
            DEFAULT_ATTRIBUTION_DERIVED_FEATURE_NAMES)
        return statistical_license

    @property
    def claim_scope(self) -> ClaimScope:
        """What the resulting number can be a claim about (T-07, Task 7).

        Independent of :attr:`statistical_license`: see
        ``schema/identity.py`` for the held-out-yet-circular case that is the
        entire reason the two are separate properties instead of one grade.
        """
        _, claim_scope = resolve_query_permissions(
            self.selection_provenance, self.selection_feature_names,
            DEFAULT_ATTRIBUTION_DERIVED_FEATURE_NAMES)
        return claim_scope

    @property
    def output_mode(self) -> OutputMode:
        """Deprecated compatibility view, derived from the two properties above.

        Not a second source of truth: it never inspects ``selection_provenance``
        directly, only the already-resolved ``statistical_license`` /
        ``claim_scope``. See :func:`_output_mode_from_permissions`.
        """
        return _output_mode_from_permissions(self.statistical_license, self.claim_scope)


@dataclass(frozen=True)
class HealthFloors:
    """Pre-registered floors. Declared before the query runs, recorded with it.

    They are arguments, not thresholds discovered afterwards: a floor chosen once
    the numbers are visible is not a floor. ``min_blocks`` matches the estimability
    floor used elsewhere (N >= 30) because for a clustered peak set the effective
    sample size is the number of blocks, not the number of peaks.
    """

    min_intersection_coverage: float = 0.90
    min_blocks: int = 30
    min_explained_fraction: float = 0.50

    def __post_init__(self) -> None:
        for name in ("min_intersection_coverage", "min_explained_fraction"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise SchemaError(f"{name} must be a proportion in [0, 1], got {v}")
        if self.min_blocks < 1:
            raise SchemaError(f"min_blocks must be >= 1, got {self.min_blocks}")


@dataclass
class AnalysisConfig:
    """T-01. ``analyses`` is unbounded, and N<3 is a runtime constraint."""

    project: str
    analyses: list[dict[str, Any]]
    peak_universe_id: str = MISSING_SENTINEL
    input_scale: int | None = None

    def __post_init__(self) -> None:
        if len(self.analyses) < 1:
            raise SchemaError("at least one analysis (model x readout) is required")
        ids = [a.get("id") for a in self.analyses]
        if len(set(ids)) != len(ids):
            raise SchemaError(f"analysis ids must be unique: {ids}")

    @property
    def n_models(self) -> int:
        return len({a.get("model") for a in self.analyses})

    @property
    def cross_model_claims_restricted(self) -> bool:
        """T-01 as a data field rather than a thing a later step must remember."""
        return self.n_models < 3

    def assert_between_model_heterogeneity_estimable(self) -> None:
        """T-01, as a runtime assertion rather than a documentation promise.

        With fewer than three models there is no between-model variance to
        estimate. Only sign consistency and leave-one-model-out are reportable.
        """
        if self.n_models < 3:
            raise SchemaError(
                f"between-model heterogeneity is not estimable with n_models={self.n_models}. "
                "Report sign consistency and leave-one-model-out instead. "
                "See docs/CONSTRAINTS.md (N>=3)."
            )
