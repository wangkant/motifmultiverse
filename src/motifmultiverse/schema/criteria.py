"""Executable, versioned adjudication criteria.

See ``docs/DATA_MODEL.md`` (Criterion registry), ``docs/CONSTRAINTS.md``
(``FP-04``, ``FP-08``, ``FP-13``) and ``docs/CONCEPT.md`` ("A merge is validated
downstream, not by similarity").

The reference implementation's failure here was not a missing threshold -- it was
a threshold nobody wrote down until asked, at which point one gets invented on the
spot and looks exactly as principled as one that was actually derived (``FP-13``).
This module makes the distinction machine-checkable. A :class:`Criterion` is one
of three things:

``FROZEN``
    Its predicates are declared, in named operators, over named evidence fields,
    and every magnitude in them *follows from something* -- a document, a
    structural definition, or an instrument's own resolution.
``FROZEN_DECLARED_HEURISTIC``
    Evaluable exactly like ``FROZEN``, but at least one magnitude was **chosen by
    a maintainer**. Added in criteria schema version 2 so that "we froze a number"
    and "the number follows from something" can never again look identical in the
    artifact -- which is the precise failure ``FP-13`` names.
``CRITERION_NOT_YET_DEFINED``
    The frozen design states no magnitude for it. :func:`evaluate_criterion` on
    this kind always returns ``Decision.DEFERRED`` -- never a guess, and never
    silently reuses a plausible number from somewhere else.

Two *file* format versions are supported. A ``schema_version: "1"`` file loads
with exactly its historical meaning: predicate provenance is optional there,
because no v1 file ever carried it and an existing pinned run must not change
meaning (or start failing) because a later release grew a field. A
``schema_version: "2"`` file is held to the stricter contract -- every ordered
comparison (``ge``/``le``) must state ``provenance`` and ``basis``, so an
unattributed threshold cannot be written down at all.

The evaluator is a small named-operator interpreter (``ge``, ``le``, ``eq``,
``is_true``, ``present``) over a flat evidence mapping. It never calls ``eval``,
``exec``, or any user-supplied callable: a predicate is data, not code, which is
the whole point of a *registry* rather than a rule buried in ``adjudicate``.

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

from motifmultiverse.schema import EVIDENCE_FIELD_DOMAINS, Decision, SchemaError

__all__ = [
    "CRITERIA_SCHEMA_VERSION",
    "SUPPORTED_CRITERIA_SCHEMA_VERSIONS",
    "PROVENANCE_REQUIRED_FROM_SCHEMA_VERSION",
    "CriteriaError",
    "CriterionStatus",
    "EVALUABLE_STATUSES",
    "VALID_PREDICATE_PROVENANCE",
    "DERIVED_DOMAIN_ENDPOINTS",
    "Predicate",
    "Criterion",
    "CriterionEvaluation",
    "load_criteria",
    "evaluate_criterion",
]

#: The criteria-*file* format version this release WRITES (distinct from a single
#: criterion's own ``version``, which scopes one relationship's rule). Bumped only
#: if the YAML shape changes, mirroring ``schema.LEXICON_MANIFEST_SCHEMA_VERSION``
#: / ``schema.DECISION_BUNDLE_SCHEMA_VERSION``. Version 2 adds predicate-level
#: ``provenance``/``basis`` and criterion-level
#: ``declared_rationale``/``replacement_evidence``.
CRITERIA_SCHEMA_VERSION = "2"

#: The file format versions this loader READS. ``"1"`` is retained deliberately,
#: not tolerated: a run pinned to a v1 registry must keep meaning exactly what it
#: meant, and must not start failing because a later release grew a field. A
#: version outside this set is still refused rather than guessed at -- ``FP-13``
#: requires a merge/split rule parameter set to be exactly what was checksummed,
#: not a file the loader was lenient about.
SUPPORTED_CRITERIA_SCHEMA_VERSIONS = ("1", "2")

#: From this file version on, every ordered comparison must state where its
#: magnitude came from. Scoped to a file version rather than applied globally so
#: that adding the requirement cannot retroactively invalidate a v1 registry
#: somebody already ran and checksummed -- the v1 files have no provenance because
#: the field did not exist, which is a different fact from a v2 file omitting it.
PROVENANCE_REQUIRED_FROM_SCHEMA_VERSION = "2"


class CriteriaError(SchemaError):
    """A criterion registry file, or a criterion built in-process, is invalid."""


class CriterionStatus(StrEnum):
    """Whether a criterion's predicates may be evaluated or must be deferred."""

    FROZEN = "FROZEN"
    #: Evaluable exactly like ``FROZEN``, but at least one of its magnitudes was
    #: *chosen by a maintainer*, not read off a document, a calibrated null, or a
    #: measurement. It exists so a reader can tell a derived rule from a declared
    #: one without opening the YAML -- which is the whole distinction ``FP-13``
    #: says gets lost. A criterion in this state must additionally carry
    #: ``declared_rationale`` (why this value, in prose) and
    #: ``replacement_evidence`` (what a future release must measure to promote it
    #: to ``FROZEN``), and each declared magnitude must be marked
    #: ``provenance: declared`` at the predicate level. Every evaluation it
    #: produces stamps ``DECLARED-NOT-DERIVED`` into its rationale, so the string
    #: that lands in ``ontology_decisions.parquet`` carries the caveat too.
    FROZEN_DECLARED_HEURISTIC = "FROZEN_DECLARED_HEURISTIC"
    #: Recorded rather than filled. See module docstring and ``FP-13``.
    CRITERION_NOT_YET_DEFINED = "CRITERION_NOT_YET_DEFINED"


#: The statuses whose predicates are actually executed. Grouped once, so that a
#: new evaluable status cannot be added without every ``is FROZEN`` test in the
#: package being reconsidered.
EVALUABLE_STATUSES = frozenset({
    CriterionStatus.FROZEN, CriterionStatus.FROZEN_DECLARED_HEURISTIC
})

#: How a predicate's magnitude came to be. ``derived`` covers a value that follows
#: from the frozen design, from an instrument's own resolution (e.g. the
#: ``1/(null_shuffles+1)`` floor of the alignment null), or from a structural
#: definition (e.g. full bilateral containment). ``declared`` covers a value a
#: maintainer chose. ``None`` is legal only for predicates carrying no magnitude
#: at all (``is_true``/``present``), and -- for backward compatibility only -- for
#: thresholds in a ``schema_version: "1"`` file.
VALID_PREDICATE_PROVENANCE = frozenset({"declared", "derived"})

#: The endpoints of an evidence field's validated range that a ``derived``
#: threshold may name as its source. Each is RESOLVED against
#: :data:`~motifmultiverse.schema.EVIDENCE_FIELD_DOMAINS` and compared against the
#: predicate's own ``value``; a mismatch refuses the predicate.
#:
#: ``sign_boundary`` is a distinct name from ``min`` on purpose. On a field
#: validated into ``[0.0, 1.0]`` the number 0.0 is the bottom of the range, and
#: gating there is a floor; on one validated into ``[-1.0, 1.0]`` the same number
#: separates two opposite meanings, and gating there is a sign test. Same value,
#: different claim, so ``sign_boundary`` is legal only where the range actually
#: straddles zero.
DERIVED_DOMAIN_ENDPOINTS = frozenset({"min", "max", "sign_boundary"})

#: The keys a ``derived_from`` mapping must carry -- exactly these, no others.
_DERIVED_FROM_KEYS = frozenset({"evidence_domain", "endpoint"})


def _resolve_derived_from(field: str, derived_from: Mapping[str, Any]) -> float:
    """Recompute what a ``derived_from`` reference points at, or refuse it.

    This is the entire machine check behind ``provenance: derived``. It answers one
    question -- *is this number a structural landmark of this field's own validated
    range?* -- and nothing else. It cannot and does not check that gating at that
    landmark is the right rule; that argument lives in ``basis`` and is read by a
    person. A magnitude that is not expressible as such a landmark is not derived,
    and must be written ``provenance: declared``, which claims nothing.
    """
    if not isinstance(derived_from, Mapping):
        raise CriteriaError(
            f"predicate {field!r}: derived_from must be a mapping with keys "
            f"{sorted(_DERIVED_FROM_KEYS)}, got {type(derived_from).__name__}"
        )
    unknown = set(derived_from) - _DERIVED_FROM_KEYS
    missing = _DERIVED_FROM_KEYS - set(derived_from)
    if unknown or missing:
        raise CriteriaError(
            f"predicate {field!r}: derived_from must carry exactly "
            f"{sorted(_DERIVED_FROM_KEYS)}; unknown {sorted(unknown)}, "
            f"missing {sorted(missing)}"
        )

    named_field = derived_from["evidence_domain"]
    if named_field != field:
        raise CriteriaError(
            f"predicate {field!r}: derived_from names the domain of "
            f"{named_field!r}. A threshold may only be derived from its own "
            "domain -- otherwise every field with a [0, 1] range lends its "
            "endpoints to every other field, and the check degrades into 'is this "
            "number 0.0 or 1.0 somewhere in the package', which is not a "
            "provenance claim"
        )
    if named_field not in EVIDENCE_FIELD_DOMAINS:
        raise CriteriaError(
            f"predicate {field!r}: no validated domain is recorded for "
            f"{named_field!r}. Fields with a recorded domain are "
            f"{sorted(EVIDENCE_FIELD_DOMAINS)}; a field whose range nothing "
            "enforces has no landmark to derive a threshold from"
        )

    endpoint = derived_from["endpoint"]
    if endpoint not in DERIVED_DOMAIN_ENDPOINTS:
        raise CriteriaError(
            f"predicate {field!r}: derived_from endpoint {endpoint!r} is not one "
            f"of {sorted(DERIVED_DOMAIN_ENDPOINTS)}"
        )

    low, high = EVIDENCE_FIELD_DOMAINS[named_field]
    if endpoint == "min":
        return float(low)
    if endpoint == "max":
        return float(high)
    if not low < 0.0 < high:
        raise CriteriaError(
            f"predicate {field!r}: endpoint 'sign_boundary' is meaningless on a "
            f"field validated into [{low}, {high}], which never changes sign. "
            "Zero there is the bottom of the range, not a boundary between two "
            "opposite meanings, and writing it this way would make a floor read "
            "as a driver/repressor distinction"
        )
    return 0.0


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
    #: ``"declared"`` / ``"derived"`` / ``None``. Permitted only on a predicate
    #: that carries a magnitude (``ge``/``le``); refused on ones that do not, so a
    #: boolean check cannot be dressed up as a measured quantity. *Required* on
    #: every ``ge``/``le`` in a ``schema_version: "2"`` file -- see
    #: :func:`load_criteria`, which owns that check because it is a property of
    #: the file format, not of a criterion built in-process.
    provenance: str | None = None
    #: Free prose recording *where the magnitude came from* -- the calculation for
    #: ``derived``, the reasoning and the caveat for ``declared``. Required
    #: alongside ``provenance``; a provenance label with no accompanying account is
    #: the thing this schema exists to prevent.
    basis: str | None = None
    #: Where a ``derived`` magnitude comes from, in a form a loader can RESOLVE:
    #: ``{"evidence_domain": <this predicate's field>, "endpoint": min|max|
    #: sign_boundary}``. Refused on a ``declared`` predicate -- a chosen number does
    #: not become derived by naming a landmark next to it -- and *required* on a
    #: ``derived`` one from criteria schema version 2 on (enforced in
    #: :func:`load_criteria`, which owns file-format requirements).
    #:
    #: ``basis`` is prose and cannot be checked. This can: the value is recomputed
    #: from :data:`~motifmultiverse.schema.EVIDENCE_FIELD_DOMAINS` and a mismatch
    #: refuses the predicate. That makes ``derived`` mean exactly one checkable
    #: thing and no more -- see :func:`_resolve_derived_from`.
    derived_from: Mapping[str, Any] | None = None

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
        if self.provenance is not None and self.provenance not in VALID_PREDICATE_PROVENANCE:
            raise CriteriaError(
                f"predicate provenance {self.provenance!r} on field {self.field!r} is not "
                f"one of {sorted(VALID_PREDICATE_PROVENANCE)}"
            )
        # Scoped to ``ge``/``le`` deliberately. Those are the ordered comparisons
        # -- the thresholds -- and a threshold is the thing FP-13 is about. ``eq``
        # is dispatch-by-name (see GRADE_VALUED_FIELDS) or exact equality, neither
        # of which is a magnitude someone had to pick a value for, and ``is_true``
        # carries no value at all.
        if self.operator not in _NUMERIC_OPERATORS and self.provenance is not None:
            raise CriteriaError(
                f"predicate {self.field!r} {self.operator!r} is not a threshold, so "
                "provenance may not be set; an equality or boolean check must not be "
                "presented as a chosen or derived magnitude"
            )
        if (self.provenance is None) != (self.basis is None) or (
            self.basis is not None and not str(self.basis).strip()
        ):
            raise CriteriaError(
                f"predicate {self.field!r}: provenance and basis must be set together and "
                "basis must be non-empty; a provenance label with no stated basis is an "
                "unaudited number wearing a label"
            )
        # The resolution check binds to the predicate, not to the file, because
        # `adjudicate` and callers construct Predicates directly: a check that
        # lived only in `load_criteria` would leave the in-process path exactly as
        # unchecked as the YAML path was.
        if self.derived_from is not None:
            if self.provenance != "derived":
                raise CriteriaError(
                    f"predicate {self.field!r}: derived_from is set on a predicate whose "
                    f"provenance is {self.provenance!r}. A chosen magnitude does not "
                    "become derived by naming a landmark beside it; either the value "
                    "IS that landmark, in which case say provenance 'derived', or it "
                    "is declared and claims nothing"
                )
            resolved = _resolve_derived_from(self.field, self.derived_from)
            if float(self.value) != resolved:
                raise CriteriaError(
                    f"predicate {self.field!r}: value {self.value!r} is not what its "
                    f"derived_from resolves to ({resolved!r}). Naming a source is not "
                    "enough -- an unresolved reference is a second self-report, a "
                    "longer way of writing the same unchecked claim"
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
    #: Required when ``status`` is ``FROZEN_DECLARED_HEURISTIC``: prose saying why
    #: the declared magnitudes were set where they were, and (honestly) what the
    #: alternative would have been.
    declared_rationale: str | None = None
    #: Required when ``status`` is ``FROZEN_DECLARED_HEURISTIC``: the named
    #: measurements a future release must produce to promote this criterion to
    #: ``FROZEN``. Written down now, while the gap is fresh, rather than
    #: reconstructed later by whoever inherits the number.
    replacement_evidence: tuple[str, ...] = ()

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

        if self.status in EVALUABLE_STATUSES:
            if not self.predicates:
                raise CriteriaError(
                    f"{self.criterion_id}: {self.status.value} status with zero predicates "
                    "would match vacuously (all([]) is True); an evaluable criterion must "
                    "declare at least one predicate"
                )
            if self.decision_if_matched is None:
                raise CriteriaError(
                    f"{self.criterion_id}: {self.status.value} status requires "
                    "decision_if_matched; a criterion capable of matching must say what it "
                    "licenses"
                )

        declared = tuple(p for p in self.predicates if p.provenance == "declared")
        if self.status is CriterionStatus.FROZEN and declared:
            raise CriteriaError(
                f"{self.criterion_id}: status FROZEN but predicate field(s) "
                f"{sorted(p.field for p in declared)} are marked provenance 'declared'. A "
                "criterion containing a chosen magnitude must use "
                "FROZEN_DECLARED_HEURISTIC, so a reader can never mistake it for one whose "
                "numbers follow from something"
            )
        if self.status is CriterionStatus.FROZEN_DECLARED_HEURISTIC:
            if not declared:
                raise CriteriaError(
                    f"{self.criterion_id}: status FROZEN_DECLARED_HEURISTIC but no predicate "
                    "is marked provenance 'declared'; if every magnitude is derived the "
                    "criterion should be FROZEN and claim the stronger status it has earned"
                )
            if not (self.declared_rationale or "").strip():
                raise CriteriaError(
                    f"{self.criterion_id}: FROZEN_DECLARED_HEURISTIC requires "
                    "declared_rationale; freezing a chosen number without saying why is the "
                    "state this status was introduced to replace"
                )
            if not self.replacement_evidence:
                raise CriteriaError(
                    f"{self.criterion_id}: FROZEN_DECLARED_HEURISTIC requires "
                    "replacement_evidence naming >=1 measurement that would promote it to "
                    "FROZEN; a heuristic with no stated exit is a permanent one"
                )
        elif self.status is CriterionStatus.FROZEN and (
            self.declared_rationale is not None or self.replacement_evidence
        ):
            raise CriteriaError(
                f"{self.criterion_id}: a plain FROZEN criterion may not carry "
                "declared_rationale/replacement_evidence -- those fields announce a chosen "
                "magnitude, which requires FROZEN_DECLARED_HEURISTIC. "
                "(CRITERION_NOT_YET_DEFINED may retain them: downgrading a heuristic back "
                "to undefined must not require deleting the account of why it was ever "
                "frozen.)"
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


def _declared_caveat(criterion: Criterion) -> str:
    """Stamp the declared magnitudes into any rationale a heuristic criterion emits.

    The caveat travels *with the decision*, into ``ontology_decisions.parquet`` and
    ``review.yaml``, rather than living only in the registry file that a reader of
    those artifacts may never open. Emitted on refusal as well as on match: "this
    pair missed a threshold" means something different when the threshold was
    chosen rather than derived, and the reader of a refusal deserves to know which.
    """
    if criterion.status is not CriterionStatus.FROZEN_DECLARED_HEURISTIC:
        return ""
    declared = ", ".join(
        f"{p.field} {p.operator} {p.value}"
        for p in criterion.predicates
        if p.provenance == "declared"
    )
    return (
        f" [DECLARED-NOT-DERIVED: {criterion.criterion_id} is a "
        f"FROZEN_DECLARED_HEURISTIC; the magnitude(s) {declared} were chosen by a "
        "maintainer, not derived from a document, a calibrated null, or a "
        "measurement. Replacement evidence required: "
        f"{', '.join(criterion.replacement_evidence)}]"
    )


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
                + _declared_caveat(criterion)
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
            + _declared_caveat(criterion)
        ),
    )


def _predicate_from_dict(payload: Mapping[str, Any]) -> Predicate:
    if not isinstance(payload, Mapping):
        raise CriteriaError(f"a predicate entry must be a mapping, got {type(payload).__name__}")
    _reject_unknown_keys(
        payload,
        {"field", "operator", "value", "provenance", "basis", "derived_from"},
        "predicate",
    )
    missing = {"field", "operator"} - set(payload)
    if missing:
        raise CriteriaError(f"predicate entry missing required key(s) {sorted(missing)}: {payload!r}")
    return Predicate(
        field=payload["field"],
        operator=payload["operator"],
        value=payload.get("value"),
        provenance=payload.get("provenance"),
        basis=payload.get("basis"),
        derived_from=payload.get("derived_from"),
    )


_CRITERION_KEYS = frozenset({
    "criterion_id", "version", "status", "relationship", "required_evidence",
    "predicates", "insufficient_evidence_action", "decision_if_matched",
    "declared_rationale", "replacement_evidence",
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
    replacement_raw = payload.get("replacement_evidence") or ()
    if isinstance(replacement_raw, str) or not isinstance(replacement_raw, (list, tuple)):
        raise CriteriaError(
            f"criterion {payload.get('criterion_id')!r}: replacement_evidence must be a "
            f"list, got {replacement_raw!r} ({type(replacement_raw).__name__}); a bare "
            "string would silently iterate into one entry per character"
        )
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
        declared_rationale=payload.get("declared_rationale"),
        replacement_evidence=tuple(replacement_raw),
    )


def load_criteria(path: str | Path) -> dict[str, Criterion]:
    """Load and validate a criterion registry file, keyed by ``criterion_id``.

    Refuses (raising :class:`CriteriaError`): an unrecognised top-level or
    per-entry key, a missing or unsupported ``schema_version``, an unknown
    predicate operator, a numeric predicate on a grade-valued field, a predicate
    field not declared in its criterion's ``required_evidence``, a duplicate
    ``criterion_id``, and -- from schema version 2 on -- a ``ge``/``le`` predicate
    that does not say where its magnitude came from.

    Both supported file versions are read with their own meaning. A v1 file's
    thresholds carry no ``provenance`` because the key did not exist when it was
    written, and it keeps loading unchanged: a run pinned to a v1 registry that
    was already checksummed must not change meaning, or start failing, because a
    later release grew a field. The stricter contract applies only to files that
    declare themselves v2 and therefore opted into it.
    """
    text = Path(path).read_text()
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, Mapping):
        raise CriteriaError(f"{path}: a criteria file must be a mapping at the top level")
    _reject_unknown_keys(payload, {"schema_version", "criteria"}, f"{path}")

    schema_version = payload.get("schema_version")
    if schema_version is None:
        raise CriteriaError(f"{path}: missing required top-level 'schema_version'")
    schema_version = str(schema_version)
    if schema_version not in SUPPORTED_CRITERIA_SCHEMA_VERSIONS:
        raise CriteriaError(
            f"{path}: schema_version {schema_version!r} is not one of this loader's "
            f"supported versions {list(SUPPORTED_CRITERIA_SCHEMA_VERSIONS)}. A criteria "
            "file's shape is frozen the same way FP-13 requires a merge/split rule "
            "parameter set to be: add the version to SUPPORTED_CRITERIA_SCHEMA_VERSIONS "
            "deliberately, do not silently accept a drifted shape"
        )
    # Ordered by position in SUPPORTED_CRITERIA_SCHEMA_VERSIONS, not by comparing
    # the version strings: `"10" >= "2"` is False lexicographically, which would
    # silently drop the requirement again at exactly the version where nobody is
    # looking. The tuple is declared oldest-first and is the ordering.
    requires_provenance = SUPPORTED_CRITERIA_SCHEMA_VERSIONS.index(
        schema_version
    ) >= SUPPORTED_CRITERIA_SCHEMA_VERSIONS.index(PROVENANCE_REQUIRED_FROM_SCHEMA_VERSION)

    registry: dict[str, Criterion] = {}
    for entry in payload.get("criteria") or []:
        criterion = _criterion_from_dict(entry)
        if requires_provenance:
            unattributed = sorted(
                f"{p.field} {p.operator} {p.value}"
                for p in criterion.predicates
                if p.operator in _NUMERIC_OPERATORS and p.provenance is None
            )
            if unattributed:
                raise CriteriaError(
                    f"{path}: {criterion.criterion_id}: threshold predicate(s) "
                    f"{unattributed} do not state provenance ('declared' or 'derived') and "
                    f"a basis. From schema_version "
                    f"{PROVENANCE_REQUIRED_FROM_SCHEMA_VERSION} on, an unattributed "
                    "threshold cannot be written down: it is exactly the FP-13 failure -- a "
                    "number nobody wrote the origin of, indistinguishable afterwards from "
                    "one that was derived"
                )
            # `basis` is prose and a reader has to take it on trust. `derived_from`
            # is the part a loader can check, so `derived` may not be claimed
            # without it. Scoped to the file version for the same reason
            # `provenance` itself is: a v1 file carries no provenance at all, so
            # there is nothing here for it to attach to, and a pinned v1 run must
            # not start failing because a later release grew a field.
            unresolvable = sorted(
                f"{p.field} {p.operator} {p.value}"
                for p in criterion.predicates
                if p.provenance == "derived" and p.derived_from is None
            )
            if unresolvable:
                raise CriteriaError(
                    f"{path}: {criterion.criterion_id}: threshold predicate(s) "
                    f"{unresolvable} claim provenance 'derived' without a "
                    "derived_from a loader can resolve. 'derived' asserts that a "
                    "number follows from something; an unchecked assertion about "
                    "provenance is worth less than none, because it READS as "
                    "evidence. Either name the domain landmark this value is -- "
                    f"{{evidence_domain: <field>, endpoint: "
                    f"{sorted(DERIVED_DOMAIN_ENDPOINTS)}}} -- or write "
                    "provenance 'declared', which claims nothing"
                )
        if criterion.criterion_id in registry:
            raise CriteriaError(f"{path}: duplicate criterion_id {criterion.criterion_id!r}")
        registry[criterion.criterion_id] = criterion
    return registry
