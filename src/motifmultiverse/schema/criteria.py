"""Executable, versioned adjudication criteria.

See ``docs/DATA_MODEL.md`` (Criterion registry), ``docs/CONSTRAINTS.md``
(``FP-04``, ``FP-08``, ``FP-13``) and ``docs/CONCEPT.md`` ("A merge is validated
downstream, not by similarity").

The reference implementation's failure here was not a missing threshold -- it was
a threshold nobody wrote down until asked, at which point one gets invented on the
spot and looks exactly as principled as one that was actually derived (``FP-13``).
This module makes the distinction machine-checkable: a :class:`Criterion` is
either ``FROZEN`` (its predicates are declared, in named operators, over named
evidence fields) or ``CRITERION_NOT_YET_DEFINED`` (the frozen design states no
magnitude for it). :func:`evaluate_criterion` on the second kind always returns
``Decision.DEFERRED`` -- never a guess, and never silently reuses a plausible
number from somewhere else.

The evaluator is a small named-operator interpreter (``ge``, ``le``, ``eq``,
``is_true``, ``present``) over a flat evidence mapping. It never calls ``eval``,
``exec``, or any user-supplied callable: a predicate is data, not code, which is
the whole point of a *registry* rather than a rule buried in ``adjudicate``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from motifmultiverse.schema import Decision, SchemaError

__all__ = [
    "CRITERIA_SCHEMA_VERSION",
    "CriteriaError",
    "CriterionStatus",
    "Predicate",
    "Criterion",
    "CriterionEvaluation",
    "load_criteria",
    "evaluate_criterion",
]

#: The criteria-*file* format version (distinct from a single criterion's own
#: ``version``, which scopes one relationship's rule). Bumped only if the YAML
#: shape changes, mirroring ``schema.LEXICON_MANIFEST_SCHEMA_VERSION`` /
#: ``schema.DECISION_BUNDLE_SCHEMA_VERSION``. A file declaring any other value is
#: refused rather than tolerated: ``FP-13`` requires a merge/split rule parameter
#: set to be exactly what was checksummed, not a file the loader was lenient
#: about.
CRITERIA_SCHEMA_VERSION = "1"


class CriteriaError(SchemaError):
    """A criterion registry file, or a criterion built in-process, is invalid."""


class CriterionStatus(StrEnum):
    """Whether a criterion's predicates may be evaluated or must be deferred."""

    FROZEN = "FROZEN"
    #: Recorded rather than filled. See module docstring and ``FP-13``.
    CRITERION_NOT_YET_DEFINED = "CRITERION_NOT_YET_DEFINED"


#: Named operators the evaluator understands. Nothing else -- in particular, no
#: string is ever handed to ``eval``/``exec``, and a predicate cannot name a
#: Python callable. An operator outside this set is refused at construction, not
#: at evaluation time, so a bad registry file fails to load rather than fails
#: (or worse, silently no-ops) the first time adjudication runs.
_NUMERIC_OPERATORS = frozenset({"ge", "le"})
_OTHER_OPERATORS = frozenset({"eq", "is_true", "present"})
VALID_OPERATORS = frozenset(_NUMERIC_OPERATORS | _OTHER_OPERATORS)

#: Operators that compare ``field`` against ``value`` and therefore need a real
#: value: ``is_true``/``present`` read only the field and must NOT require one.
#: ``eq`` is included alongside ``ge``/``le`` -- an ``eq`` predicate with no value
#: would compare every evidence value against ``None``, which is unreachable by
#: the time a predicate runs (the missing-evidence gate already refused any
#: ``None`` required-evidence value), so such a predicate could never match and
#: would silently and permanently refuse its criterion.
_VALUE_REQUIRED_OPERATORS = frozenset(_NUMERIC_OPERATORS | {"eq"})

#: Evidence fields whose value is a grade/category dispatched **by name**, never
#: a magnitude to be thresholded -- the same distinction ``schema.MergeConfidence``
#: draws for merge confidence. A numeric operator (``ge``/``le``) against one of
#: these would smuggle in a scalar comparison the schema has already ruled these
#: fields do not have; ``eq`` remains legal, since dispatch-by-name is exactly how
#: these fields are meant to be read.
GRADE_VALUED_FIELDS = frozenset({
    "merge_confidence", "discovery_tier", "analysis_tier", "missingness",
    "decision", "selection_provenance", "status", "orientation",
})


def _reject_unknown_keys(payload: Mapping[str, Any], allowed: set[str], what: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise CriteriaError(
            f"{what} has unknown key(s) {sorted(unknown)}; expected a subset of {sorted(allowed)}"
        )


@dataclass(frozen=True)
class Predicate:
    """One named-operator check of a single evidence field against a value.

    ``value`` is unused (and should be omitted) for ``is_true``/``present``; it is
    the comparison value for ``ge``/``le``/``eq``.
    """

    field: str
    operator: str
    value: Any = None

    def __post_init__(self) -> None:
        if self.operator not in VALID_OPERATORS:
            raise CriteriaError(
                f"predicate operator {self.operator!r} is not one of "
                f"{sorted(VALID_OPERATORS)}; the evaluator supports only named operators "
                "and never executes an arbitrary expression"
            )
        if self.operator in _NUMERIC_OPERATORS and self.field in GRADE_VALUED_FIELDS:
            raise CriteriaError(
                f"{self.field!r} is a grade-valued field (dispatched by name, not "
                f"thresholded); operator {self.operator!r} compares a magnitude and "
                "may not be applied to it"
            )
        if self.operator in _VALUE_REQUIRED_OPERATORS and self.value is None:
            raise CriteriaError(
                f"operator {self.operator!r} on field {self.field!r} requires a "
                "comparison value; omitting it would make this predicate silently "
                "unsatisfiable forever (evidence is never None once it reaches a "
                "predicate -- see evaluate_criterion's missing-evidence gate)"
            )


@dataclass(frozen=True)
class Criterion:
    """One versioned, executable rule for one candidate relationship.

    ``decision_if_matched`` is not part of the brief's literal field list but is
    required here so the registry stays fully data-driven: nothing in
    ``evaluate_criterion`` hard-codes "if relationship == TRUE_DUPLICATE then
    COLLAPSE" -- that mapping lives in the YAML alongside the predicates it
    licenses. It must be set whenever ``status`` is ``FROZEN`` (enforced below);
    a ``CRITERION_NOT_YET_DEFINED`` criterion never reaches the branch that would
    read it, since :func:`evaluate_criterion` defers before touching predicates.
    """

    criterion_id: str
    version: str
    status: CriterionStatus
    relationship: str
    required_evidence: tuple[str, ...]
    predicates: tuple[Predicate, ...]
    insufficient_evidence_action: Decision
    decision_if_matched: Decision | None = None

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            try:
                object.__setattr__(self, "status", CriterionStatus(self.status))
            except ValueError as exc:
                raise CriteriaError(
                    f"{self.criterion_id}: status {self.status!r} is not a CriterionStatus"
                ) from exc
        if isinstance(self.insufficient_evidence_action, str):
            try:
                object.__setattr__(
                    self, "insufficient_evidence_action", Decision(self.insufficient_evidence_action)
                )
            except ValueError as exc:
                raise CriteriaError(
                    f"{self.criterion_id}: insufficient_evidence_action "
                    f"{self.insufficient_evidence_action!r} is not a Decision"
                ) from exc
        if isinstance(self.decision_if_matched, str):
            try:
                object.__setattr__(self, "decision_if_matched", Decision(self.decision_if_matched))
            except ValueError as exc:
                raise CriteriaError(
                    f"{self.criterion_id}: decision_if_matched {self.decision_if_matched!r} "
                    "is not a Decision"
                ) from exc

        if not self.criterion_id.strip():
            raise CriteriaError("a criterion requires a non-empty criterion_id")
        if self.insufficient_evidence_action is not Decision.DEFERRED:
            raise CriteriaError(
                f"{self.criterion_id}: insufficient_evidence_action must be deferred; "
                "missing evidence can never license collapse or assert a refusal"
            )
        if not self.required_evidence:
            raise CriteriaError(f"{self.criterion_id}: required_evidence must name >=1 field")

        undeclared = {p.field for p in self.predicates} - set(self.required_evidence)
        if undeclared:
            raise CriteriaError(
                f"{self.criterion_id}: predicate field(s) {sorted(undeclared)} are not "
                f"declared in required_evidence {list(self.required_evidence)}; a predicate "
                "may not read a field the criterion does not admit needing"
            )

        if self.status is CriterionStatus.FROZEN:
            if not self.predicates:
                raise CriteriaError(
                    f"{self.criterion_id}: FROZEN status with zero predicates would match "
                    "vacuously (all([]) is True); a frozen criterion must declare at least "
                    "one predicate"
                )
            if self.decision_if_matched is None:
                raise CriteriaError(
                    f"{self.criterion_id}: FROZEN status requires decision_if_matched; a "
                    "criterion capable of matching must say what it licenses"
                )


@dataclass(frozen=True)
class CriterionEvaluation:
    """The result of evaluating one criterion against one evidence mapping.

    Always carries a ``rationale`` -- a refusal or a deferral is a first-class
    recorded state, not a bare enum value (``docs/DATA_MODEL.md`` rule 4).
    """

    criterion_id: str
    criterion_version: str
    decision: Decision
    matched: bool
    rationale: str

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise CriteriaError("a criterion evaluation must record a rationale")


#: The fail-safe decision when a FROZEN criterion's evidence is complete but its
#: predicates do not hold: this criterion's relationship simply is not established,
#: which is not the same claim as "collapse is licensed". Never resolves to
#: ``COLLAPSE`` here, mirroring ``schema.MOST_CONSERVATIVE_OUTPUT_MODE``'s rule
#: that an unlicensed case must fail toward the conservative branch, not the
#: permissive one.
_UNMATCHED_FAIL_SAFE_DECISION = Decision.REFUSE_MERGE


def _evaluate_predicate(predicate: Predicate, evidence: Mapping[str, Any]) -> bool:
    actual = evidence.get(predicate.field)
    if predicate.operator == "ge":
        return actual is not None and actual >= predicate.value
    if predicate.operator == "le":
        return actual is not None and actual <= predicate.value
    if predicate.operator == "eq":
        return actual == predicate.value
    if predicate.operator == "is_true":
        return bool(actual)
    if predicate.operator == "present":
        return actual is not None
    # Unreachable: Predicate.__post_init__ already rejects any other operator.
    raise CriteriaError(f"unhandled operator {predicate.operator!r}")


def evaluate_criterion(criterion: Criterion, evidence: Mapping[str, Any]) -> CriterionEvaluation:
    """Evaluate one criterion against one evidence mapping.

    Three outcomes, in order:

    1. ``criterion.status is CRITERION_NOT_YET_DEFINED`` -> ``Decision.DEFERRED``,
       unconditionally. The frozen design states no magnitude for this criterion,
       so no evidence, however complete, can make it evaluable.
    2. Evidence missing one of ``criterion.required_evidence`` (absent, or
       explicitly ``None`` -- never coerced to a false-y placeholder) ->
       ``Decision.DEFERRED``. Construction also requires the persisted
       ``insufficient_evidence_action`` field to state that exact invariant.
    3. Otherwise, every predicate is evaluated. All satisfied ->
       ``criterion.decision_if_matched``. Any unsatisfied -> the fail-safe
       ``REFUSE_MERGE``, never a guessed collapse.
    """
    if criterion.status is CriterionStatus.CRITERION_NOT_YET_DEFINED:
        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            criterion_version=criterion.version,
            decision=Decision.DEFERRED,
            matched=False,
            rationale=(
                f"{criterion.criterion_id} v{criterion.version} is "
                "CRITERION_NOT_YET_DEFINED: the frozen design states no magnitude "
                "threshold for it, so adjudication defers rather than inventing one"
            ),
        )

    missing = [field for field in criterion.required_evidence if evidence.get(field) is None]
    if missing:
        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            criterion_version=criterion.version,
            decision=Decision.DEFERRED,
            matched=False,
            rationale=(
                f"{criterion.criterion_id} v{criterion.version}: missing required "
                f"evidence {sorted(missing)}"
            ),
        )

    matched = all(_evaluate_predicate(p, evidence) for p in criterion.predicates)
    if matched:
        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            criterion_version=criterion.version,
            decision=criterion.decision_if_matched,
            matched=True,
            rationale=(
                f"{criterion.criterion_id} v{criterion.version}: all "
                f"{len(criterion.predicates)} predicate(s) satisfied"
            ),
        )
    return CriterionEvaluation(
        criterion_id=criterion.criterion_id,
        criterion_version=criterion.version,
        decision=_UNMATCHED_FAIL_SAFE_DECISION,
        matched=False,
        rationale=(
            f"{criterion.criterion_id} v{criterion.version}: evidence complete but not "
            "every predicate held; failing safe to REFUSE_MERGE rather than defaulting "
            "toward collapse"
        ),
    )


def _predicate_from_dict(payload: Mapping[str, Any]) -> Predicate:
    if not isinstance(payload, Mapping):
        raise CriteriaError(f"a predicate entry must be a mapping, got {type(payload).__name__}")
    _reject_unknown_keys(payload, {"field", "operator", "value"}, "predicate")
    missing = {"field", "operator"} - set(payload)
    if missing:
        raise CriteriaError(f"predicate entry missing required key(s) {sorted(missing)}: {payload!r}")
    return Predicate(field=payload["field"], operator=payload["operator"], value=payload.get("value"))


_CRITERION_KEYS = frozenset({
    "criterion_id", "version", "status", "relationship", "required_evidence",
    "predicates", "insufficient_evidence_action", "decision_if_matched",
})
_REQUIRED_CRITERION_KEYS = frozenset({
    "criterion_id", "version", "status", "relationship",
    "required_evidence", "insufficient_evidence_action",
})


def _criterion_from_dict(payload: Mapping[str, Any]) -> Criterion:
    if not isinstance(payload, Mapping):
        raise CriteriaError(f"a criterion entry must be a mapping, got {type(payload).__name__}")
    _reject_unknown_keys(payload, set(_CRITERION_KEYS), "criterion")
    missing = _REQUIRED_CRITERION_KEYS - set(payload)
    if missing:
        raise CriteriaError(f"criterion entry missing required key(s) {sorted(missing)}: {payload!r}")

    required_evidence_raw = payload["required_evidence"]
    if isinstance(required_evidence_raw, str) or not isinstance(required_evidence_raw, (list, tuple)):
        raise CriteriaError(
            f"criterion {payload.get('criterion_id')!r}: required_evidence must be a list "
            f"of evidence field names, got {required_evidence_raw!r} "
            f"({type(required_evidence_raw).__name__}); a bare string would silently "
            "iterate into one field name per character"
        )

    predicates = tuple(_predicate_from_dict(p) for p in (payload.get("predicates") or []))
    # status / insufficient_evidence_action / decision_if_matched are passed through as
    # the raw YAML strings (or None): Criterion.__post_init__ owns coercing str ->
    # CriterionStatus/Decision, wrapping an invalid value in CriteriaError. Converting
    # them here too would call CriterionStatus(...)/Decision(...) directly and let a
    # bad value's bare ValueError escape uncaught, bypassing that guard entirely.
    return Criterion(
        criterion_id=str(payload["criterion_id"]),
        version=str(payload["version"]),
        status=payload["status"],
        relationship=str(payload["relationship"]),
        required_evidence=tuple(required_evidence_raw),
        predicates=predicates,
        insufficient_evidence_action=payload["insufficient_evidence_action"],
        decision_if_matched=payload.get("decision_if_matched"),
    )


def load_criteria(path: str | Path) -> dict[str, Criterion]:
    """Load and validate a criterion registry file, keyed by ``criterion_id``.

    Refuses (raising :class:`CriteriaError`): an unrecognised top-level or
    per-entry key, a missing or mismatched ``schema_version``, an unknown
    predicate operator, a numeric predicate on a grade-valued field, a predicate
    field not declared in its criterion's ``required_evidence``, and a duplicate
    ``criterion_id``.
    """
    text = Path(path).read_text()
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, Mapping):
        raise CriteriaError(f"{path}: a criteria file must be a mapping at the top level")
    _reject_unknown_keys(payload, {"schema_version", "criteria"}, f"{path}")

    schema_version = payload.get("schema_version")
    if schema_version is None:
        raise CriteriaError(f"{path}: missing required top-level 'schema_version'")
    if str(schema_version) != CRITERIA_SCHEMA_VERSION:
        raise CriteriaError(
            f"{path}: schema_version {schema_version!r} does not match this loader's "
            f"{CRITERIA_SCHEMA_VERSION!r}. A criteria file's shape is frozen the same way "
            "FP-13 requires a merge/split rule parameter set to be: bump "
            "CRITERIA_SCHEMA_VERSION deliberately, do not silently accept a drifted shape"
        )

    registry: dict[str, Criterion] = {}
    for entry in payload.get("criteria") or []:
        criterion = _criterion_from_dict(entry)
        if criterion.criterion_id in registry:
            raise CriteriaError(f"{path}: duplicate criterion_id {criterion.criterion_id!r}")
        registry[criterion.criterion_id] = criterion
    return registry
