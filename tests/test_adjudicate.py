"""Adjudicate stage tests.

Task 9 (this file's creator) owns the first behavior the file covers: the
criterion registry loader, the evaluator, and ``DEFERRED`` semantics
(``schema.criteria``). Task 12 later extends this same file with
``adjudicate_component`` / medoid-selection tests; it does not replace anything
below.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import yaml

from motifmultiverse.adjudicate import packaged_criteria_path
from motifmultiverse.schema import (
    MISSING_SENTINEL,
    REGISTRY_SCHEMA_VERSION,
    Decision,
    MotifNode,
    RegistryMetadata,
)
from motifmultiverse.schema.criteria import (
    CRITERIA_SCHEMA_VERSION,
    CriteriaError,
    Criterion,
    CriterionEvaluation,
    CriterionStatus,
    Predicate,
    evaluate_criterion,
    load_criteria,
)

CRITERIA_PATH = packaged_criteria_path()


def _complete_duplicate_evidence():
    return {
        "paired_delta_reconstruction_affected": 0.0,
        "family_coefficient_share": 0.95,
    }


def _criterion(**overrides):
    """An ad hoc criterion for evaluator unit tests -- not one of the four v1
    relationships, which are exercised separately against the real registry file.
    """
    base = dict(
        criterion_id="TEST_CRITERION",
        version="1",
        status=CriterionStatus.FROZEN,
        relationship="TEST_CRITERION",
        required_evidence=("score",),
        predicates=(Predicate(field="score", operator="ge", value=0.5),),
        insufficient_evidence_action=Decision.DEFERRED,
        decision_if_matched=Decision.COLLAPSE,
    )
    base.update(overrides)
    return Criterion(**base)


def _entry(**overrides):
    """A minimal, valid criterion-file entry dict, for synthesising bad registries."""
    base = {
        "criterion_id": "X",
        "version": "1",
        "status": "FROZEN",
        "relationship": "X",
        "required_evidence": ["a"],
        "predicates": [{"field": "a", "operator": "is_true"}],
        "insufficient_evidence_action": "deferred",
        "decision_if_matched": "refuse_merge",
    }
    base.update(overrides)
    return base


def _write_registry(tmp_path, entries, schema_version=CRITERIA_SCHEMA_VERSION):
    payload: dict = {"criteria": entries}
    if schema_version is not False:  # False = omit the key entirely
        payload["schema_version"] = schema_version
    path = tmp_path / "criteria.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


# ---------------------------------------------------------------------------
# Loader / parser tests -- select with `-k criterion`.
# ---------------------------------------------------------------------------

def test_criterion_loader_loads_the_four_v1_relationships():
    criteria = load_criteria(CRITERIA_PATH)
    assert set(criteria) == {
        "TRUE_DUPLICATE", "FRAGMENT_MATCH", "SAME_FAMILY_VARIANT", "AMBIGUOUS_CROSS_FAMILY",
    }


def test_criterion_loader_marks_undefined_thresholds_correctly():
    """TRUE_DUPLICATE / FRAGMENT_MATCH need a magnitude threshold the frozen
    design does not state and must stay CRITERION_NOT_YET_DEFINED; the two
    structural, categorical criteria may be FROZEN.
    """
    criteria = load_criteria(CRITERIA_PATH)
    assert criteria["TRUE_DUPLICATE"].status is CriterionStatus.CRITERION_NOT_YET_DEFINED
    assert criteria["FRAGMENT_MATCH"].status is CriterionStatus.CRITERION_NOT_YET_DEFINED
    assert criteria["SAME_FAMILY_VARIANT"].status is CriterionStatus.FROZEN
    assert criteria["AMBIGUOUS_CROSS_FAMILY"].status is CriterionStatus.FROZEN


def test_criterion_loader_rejects_unknown_predicate_operator(tmp_path):
    path = _write_registry(tmp_path, [
        _entry(predicates=[{"field": "a", "operator": "smells_like"}]),
    ])
    with pytest.raises(CriteriaError, match="operator"):
        load_criteria(path)


def test_criterion_loader_rejects_an_eval_operator_outright(tmp_path):
    """Not merely unfamiliar -- an operator that looks like an escape hatch into
    arbitrary code must be refused exactly like any other unknown operator.
    """
    path = _write_registry(tmp_path, [
        _entry(predicates=[{"field": "a", "operator": "eval"}]),
    ])
    with pytest.raises(CriteriaError, match="operator"):
        load_criteria(path)


def test_criterion_loader_rejects_duplicate_criterion_ids(tmp_path):
    path = _write_registry(tmp_path, [_entry(), _entry(version="2")])
    with pytest.raises(CriteriaError, match="duplicate criterion_id"):
        load_criteria(path)


def test_criterion_loader_rejects_predicate_referencing_an_undeclared_evidence_field(tmp_path):
    path = _write_registry(tmp_path, [
        _entry(required_evidence=["a"], predicates=[{"field": "b", "operator": "is_true"}]),
    ])
    with pytest.raises(CriteriaError, match="required_evidence"):
        load_criteria(path)


def test_criterion_loader_rejects_numeric_predicate_on_grade_valued_field(tmp_path):
    path = _write_registry(tmp_path, [
        _entry(
            required_evidence=["merge_confidence"],
            predicates=[{"field": "merge_confidence", "operator": "ge", "value": 0.8}],
        ),
    ])
    with pytest.raises(CriteriaError, match="grade-valued"):
        load_criteria(path)


def test_criterion_loader_rejects_missing_schema_version(tmp_path):
    path = _write_registry(tmp_path, [_entry()], schema_version=False)
    with pytest.raises(CriteriaError, match="schema_version"):
        load_criteria(path)


def test_criterion_loader_rejects_mismatched_schema_version(tmp_path):
    path = _write_registry(tmp_path, [_entry()], schema_version="999")
    with pytest.raises(CriteriaError, match="schema_version"):
        load_criteria(path)


def test_criterion_loader_rejects_an_unknown_top_level_key(tmp_path):
    path = tmp_path / "criteria.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": CRITERIA_SCHEMA_VERSION, "criteria": [], "notes": "typo'd key",
    }))
    with pytest.raises(CriteriaError, match="unknown key"):
        load_criteria(path)


def test_criterion_loader_rejects_an_unknown_criterion_key(tmp_path):
    path = _write_registry(tmp_path, [_entry(reationship="X")])  # typo'd key alongside the real one
    with pytest.raises(CriteriaError, match="unknown key"):
        load_criteria(path)


def test_criterion_loader_rejects_an_invalid_status_value_as_criteriaerror(tmp_path):
    """A bad `status` (wrong case, typo) must raise CriteriaError -- the documented
    exception every caller catches -- never a bare ValueError from CriterionStatus().
    """
    path = _write_registry(tmp_path, [_entry(status="Frozen")])
    with pytest.raises(CriteriaError):
        load_criteria(path)


def test_criterion_loader_rejects_an_invalid_insufficient_evidence_action(tmp_path):
    """Same guarantee for `insufficient_evidence_action`: a typo'd Decision value
    must raise CriteriaError, not a bare ValueError from Decision().
    """
    path = _write_registry(tmp_path, [_entry(insufficient_evidence_action="deffered")])
    with pytest.raises(CriteriaError):
        load_criteria(path)


def test_criterion_loader_rejects_collapse_as_missing_evidence_action(tmp_path):
    """A valid Decision enum is still corrupt when it licenses missing evidence."""
    path = _write_registry(
        tmp_path,
        [_entry(insufficient_evidence_action="collapse")],
    )
    with pytest.raises(CriteriaError, match="must be deferred"):
        load_criteria(path)


def test_criterion_loader_rejects_an_invalid_decision_if_matched(tmp_path):
    path = _write_registry(tmp_path, [_entry(decision_if_matched="colapse")])
    with pytest.raises(CriteriaError):
        load_criteria(path)


def test_criterion_loader_rejects_required_evidence_that_is_not_a_list(tmp_path):
    """`required_evidence: score` (a bare string, missing the YAML list brackets)
    must be refused, not silently iterated into one field name per character
    (`('s', 'c', 'o', 'r', 'e')`).

    Regression note: an earlier version of this test used
    `required_evidence="score"` with the default single predicate on field
    `"a"`. That accidentally passed for the wrong reason even with the guard
    (criteria.py:354-361) deleted: exploding `"score"` yields
    `('s','c','o','r','e')`, which does not contain `"a"`, so the *unrelated*
    undeclared-predicate-field check (`Criterion.__post_init__`,
    ~line 193-199) fires instead and happens to mention "required_evidence"
    too -- masking the guard's absence.

    Use `required_evidence="a"` with a predicate on field `"a"` instead: a
    single-character string explodes to the tuple `("a",)`, which *does*
    declare the "a" predicate field, so the undeclared-field check stays
    silent. This is the exact case the audit demonstrated loads with zero
    error once the guard is removed, so only the guard under test can raise
    here -- and we match on wording unique to its message, not merely the
    substring "required_evidence".
    """
    path = _write_registry(tmp_path, [
        _entry(required_evidence="a", predicates=[{"field": "a", "operator": "is_true"}]),
    ])
    with pytest.raises(CriteriaError, match="must be a list of evidence field names"):
        load_criteria(path)


def test_criterion_not_yet_defined_never_reads_decision_if_matched():
    """`decision_if_matched` is this module's own addition beyond the brief
    (see `Criterion`'s docstring) and Task 12 will consume it, so the
    guarantee that a `CRITERION_NOT_YET_DEFINED` criterion can never let it
    override `DEFERRED` deserves its own pin -- even though the status branch
    in `evaluate_criterion` is checked before predicates or
    `decision_if_matched` are ever read, so this cannot structurally fire
    today.
    """
    criterion = _criterion(status=CriterionStatus.CRITERION_NOT_YET_DEFINED)
    assert criterion.decision_if_matched is Decision.COLLAPSE  # sanity: it IS set

    # Evidence that would satisfy the criterion's only predicate (score >= 0.5)
    # were it FROZEN, to prove DEFERRED wins even when a match "would" occur.
    result = evaluate_criterion(criterion, {"score": 1.0})

    assert result.decision is Decision.DEFERRED
    assert result.matched is False


def test_criterion_loader_rejects_entry_missing_a_required_top_level_key(tmp_path):
    """A criterion entry missing one of the required top-level keys (e.g. a
    typo'd or omitted `relationship`) must be refused at the loader level, not
    merely at `Criterion` construction.
    """
    entry = _entry()
    del entry["relationship"]
    path = _write_registry(tmp_path, [entry])
    with pytest.raises(CriteriaError, match="missing required key"):
        load_criteria(path)


def test_criterion_loader_rejects_frozen_criterion_with_no_predicates():
    with pytest.raises(CriteriaError, match="predicate"):
        Criterion(
            criterion_id="X", version="1", status=CriterionStatus.FROZEN,
            relationship="X", required_evidence=("a",), predicates=(),
            insufficient_evidence_action=Decision.DEFERRED,
            decision_if_matched=Decision.REFUSE_MERGE,
        )


def test_criterion_loader_rejects_frozen_criterion_with_no_matched_decision():
    with pytest.raises(CriteriaError, match="decision_if_matched"):
        Criterion(
            criterion_id="X", version="1", status=CriterionStatus.FROZEN,
            relationship="X", required_evidence=("a",),
            predicates=(Predicate(field="a", operator="is_true"),),
            insufficient_evidence_action=Decision.DEFERRED,
        )


def test_criterion_loader_present_operator_is_recognised():
    """`present` is one of the five named operators even though none of the v1
    relationships happen to need it.
    """
    criterion = Criterion(
        criterion_id="X", version="1", status=CriterionStatus.FROZEN,
        relationship="X", required_evidence=("a",),
        predicates=(Predicate(field="a", operator="present"),),
        insufficient_evidence_action=Decision.DEFERRED,
        decision_if_matched=Decision.REFUSE_MERGE,
    )
    result = evaluate_criterion(criterion, {"a": "anything, even falsy-looking"})
    assert result.matched is True


def test_criterion_loader_every_v1_criterion_carries_its_own_version():
    for criterion in load_criteria(CRITERIA_PATH).values():
        assert criterion.version


# ---------------------------------------------------------------------------
# Predicate construction rejects bad operators directly (not just via the file
# loader) -- the guard is on the dataclass, so it holds for any caller.
# ---------------------------------------------------------------------------

def test_predicate_rejects_unknown_operator():
    with pytest.raises(CriteriaError, match="operator"):
        Predicate(field="score", operator="smells_like")


def test_predicate_never_accepts_an_arbitrary_expression_as_an_operator():
    with pytest.raises(CriteriaError, match="operator"):
        Predicate(field="score", operator="__import__('os').system('true')")


def test_predicate_rejects_numeric_operator_on_grade_valued_field():
    with pytest.raises(CriteriaError, match="grade-valued"):
        Predicate(field="merge_confidence", operator="ge", value=0.8)


def test_predicate_allows_eq_on_a_grade_valued_field():
    """`eq` is legitimate on a grade: MergeConfidence is meant to be dispatched by
    name, and `eq` is a name comparison, not a magnitude one.
    """
    p = Predicate(field="merge_confidence", operator="eq", value="HIGH")
    assert p.operator == "eq"


def test_predicate_numeric_operator_requires_a_comparison_value():
    with pytest.raises(CriteriaError):
        Predicate(field="score", operator="ge", value=None)


def test_predicate_eq_operator_also_requires_a_comparison_value():
    """An `eq` predicate with no value would compare every evidence value against
    `None` -- unreachable once evaluate_criterion's missing-evidence gate has run
    -- so it could never match and would silently, permanently refuse its
    criterion. Reject it at construction instead.
    """
    with pytest.raises(CriteriaError, match="comparison value"):
        Predicate(field="family_id", operator="eq", value=None)


# ---------------------------------------------------------------------------
# DEFERRED semantics -- the scientific point of this task.
# ---------------------------------------------------------------------------

def test_undefined_criterion_returns_deferred_not_a_guessed_decision():
    criterion = load_criteria(CRITERIA_PATH)["TRUE_DUPLICATE"]
    criterion = replace(criterion, status=CriterionStatus.CRITERION_NOT_YET_DEFINED)
    decision = evaluate_criterion(criterion, _complete_duplicate_evidence())
    assert decision.decision is Decision.DEFERRED


def test_true_duplicate_is_already_criterion_not_yet_defined_in_the_shipped_registry():
    """The registry itself must not have quietly filled this in already: loading
    the real file and evaluating TRUE_DUPLICATE with complete evidence must still
    defer, with no `replace()` needed.
    """
    criterion = load_criteria(CRITERIA_PATH)["TRUE_DUPLICATE"]
    assert criterion.status is CriterionStatus.CRITERION_NOT_YET_DEFINED
    decision = evaluate_criterion(criterion, _complete_duplicate_evidence())
    assert decision.decision is Decision.DEFERRED
    assert decision.matched is False


def test_fragment_match_also_defers_with_complete_evidence():
    criterion = load_criteria(CRITERIA_PATH)["FRAGMENT_MATCH"]
    decision = evaluate_criterion(criterion, {
        "overlap_frac_source": 1.0,
        "overlap_frac_target": 0.4,
        "paired_delta_reconstruction_affected": 0.0,
    })
    assert decision.decision is Decision.DEFERRED
    assert decision.matched is False


# ---------------------------------------------------------------------------
# Evaluator behavior against ad hoc (non-registry) criteria.
# ---------------------------------------------------------------------------

def test_evaluator_returns_insufficient_evidence_action_when_evidence_missing():
    criterion = _criterion()
    result = evaluate_criterion(criterion, {})
    assert result.decision is Decision.DEFERRED
    assert result.matched is False
    assert "score" in result.rationale


def test_evaluator_treats_an_explicit_none_as_missing_not_as_a_falsey_zero():
    criterion = _criterion()
    result = evaluate_criterion(criterion, {"score": None})
    assert result.decision is Decision.DEFERRED
    assert result.matched is False


def test_evaluator_returns_matched_decision_when_predicates_satisfied():
    criterion = _criterion()
    result = evaluate_criterion(criterion, {"score": 0.9})
    assert result.matched is True
    assert result.decision is Decision.COLLAPSE


def test_evaluator_fails_safe_to_refuse_merge_when_predicates_not_satisfied():
    """Evidence is complete, but the criterion does not hold: fail toward
    REFUSE_MERGE, never toward COLLAPSE.
    """
    criterion = _criterion()
    result = evaluate_criterion(criterion, {"score": 0.1})
    assert result.matched is False
    assert result.decision is Decision.REFUSE_MERGE


def test_criterion_evaluation_requires_a_rationale():
    with pytest.raises(CriteriaError):
        CriterionEvaluation(
            criterion_id="X", criterion_version="1", decision=Decision.DEFERRED,
            matched=False, rationale="   ",
        )


# ---------------------------------------------------------------------------
# The two structural (FROZEN) v1 criteria, exercised against the real registry.
# ---------------------------------------------------------------------------

def test_same_family_variant_refuses_merge_when_matched():
    criterion = load_criteria(CRITERIA_PATH)["SAME_FAMILY_VARIANT"]
    result = evaluate_criterion(criterion, {"same_family": True, "distinct_variant_ids": True})
    assert result.decision is Decision.REFUSE_MERGE
    assert result.matched is True


def test_same_family_variant_defers_when_evidence_absent():
    criterion = load_criteria(CRITERIA_PATH)["SAME_FAMILY_VARIANT"]
    result = evaluate_criterion(criterion, {"same_family": True})
    assert result.decision is Decision.DEFERRED
    assert result.matched is False


def test_ambiguous_cross_family_refuses_merge_when_matched():
    criterion = load_criteria(CRITERIA_PATH)["AMBIGUOUS_CROSS_FAMILY"]
    result = evaluate_criterion(
        criterion, {"family_conflict": True, "alignment_registered": True}
    )
    assert result.decision is Decision.REFUSE_MERGE
    assert result.matched is True


def test_ambiguous_cross_family_defers_when_evidence_absent():
    criterion = load_criteria(CRITERIA_PATH)["AMBIGUOUS_CROSS_FAMILY"]
    result = evaluate_criterion(criterion, {})
    assert result.decision is Decision.DEFERRED
    assert result.matched is False


# ---------------------------------------------------------------------------
# Task 12: observed medoids with a complete, deterministic tie break.
# ---------------------------------------------------------------------------

def test_choose_medoid_returns_the_observed_member_with_highest_mean_similarity():
    """Returning an average/consensus or ignoring the similarity objective fails."""
    from motifmultiverse import adjudicate

    similarities = {
        ("node-a", "node-b"): 0.9,
        ("node-a", "node-c"): 0.8,
        ("node-b", "node-c"): 0.2,
    }

    chosen = adjudicate.choose_medoid(["node-a", "node-b", "node-c"], similarities)

    assert chosen == "node-a"
    assert chosen in {"node-a", "node-b", "node-c"}


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            {
                "node-a": {"motif_completeness": 0.8, "seqlet_count": 500, "core_ic": 20.0,
                           "cross_context_recurrence": 8},
                "node-z": {"motif_completeness": 0.9, "seqlet_count": 1, "core_ic": 1.0,
                           "cross_context_recurrence": 1},
            },
            "node-z",
        ),
        (
            {
                "node-a": {"motif_completeness": 0.9, "seqlet_count": 499, "core_ic": 20.0,
                           "cross_context_recurrence": 8},
                "node-z": {"motif_completeness": 0.9, "seqlet_count": 500, "core_ic": 1.0,
                           "cross_context_recurrence": 1},
            },
            "node-z",
        ),
        (
            {
                "node-a": {"motif_completeness": 0.9, "seqlet_count": 500, "core_ic": 19.0,
                           "cross_context_recurrence": 8},
                "node-z": {"motif_completeness": 0.9, "seqlet_count": 500, "core_ic": 20.0,
                           "cross_context_recurrence": 1},
            },
            "node-z",
        ),
        (
            {
                "node-a": {"motif_completeness": 0.9, "seqlet_count": 500, "core_ic": 20.0,
                           "cross_context_recurrence": 7},
                "node-z": {"motif_completeness": 0.9, "seqlet_count": 500, "core_ic": 20.0,
                           "cross_context_recurrence": 8},
            },
            "node-z",
        ),
        ({}, "node-a"),
    ],
)
def test_choose_medoid_breaks_similarity_ties_in_the_declared_order(metadata, expected):
    """Deleting or reordering any declared tie dimension fails its own row."""
    from motifmultiverse import adjudicate

    assert adjudicate.choose_medoid(
        ["node-z", "node-a"],
        {("node-a", "node-z"): 0.9},
        node_metadata=metadata,
    ) == expected


# ---------------------------------------------------------------------------
# Task 12: adjudication schema, evidence protocol, and executable gates.
# ---------------------------------------------------------------------------

def _alignment(left="node-a", right="node-b", *, similarity=0.99,
               overlap_source=1.0, overlap_target=1.0):
    from motifmultiverse.align import AlignmentEvidence

    return AlignmentEvidence(
        source_node_id=left,
        target_node_id=right,
        orientation="+",
        offset=0,
        overlap_bp=10,
        overlap_frac_source=overlap_source,
        overlap_frac_target=overlap_target,
        ppm_similarity=similarity,
        signed_cwm_similarity=similarity,
        empirical_p_value=0.001,
        null_shuffles=1000,
        seed=7,
    )


def _annotation(node_id, family_id, *, source="tomtom", match=None):
    from motifmultiverse.schema.annotation import AnnotationCandidate

    return AnnotationCandidate.create(
        node_id=node_id,
        proposed_family_id=family_id,
        source=source,
        source_version="1",
        matched_motif_id=match or f"database:{node_id}:{family_id}",
        motif_length=10,
        seqlet_count=150,
    )


def _collapse_criterion(*, require_stability=True):
    required = ("ppm_similarity", "status") if require_stability else ("ppm_similarity",)
    predicates = [Predicate(field="ppm_similarity", operator="ge", value=0.9)]
    if require_stability:
        predicates.append(Predicate(field="status", operator="eq", value="STABLE"))
    return Criterion(
        criterion_id="TRUE_DUPLICATE",
        version="test-1",
        status=CriterionStatus.FROZEN,
        relationship="TRUE_DUPLICATE",
        required_evidence=required,
        predicates=tuple(predicates),
        insufficient_evidence_action=Decision.DEFERRED,
        decision_if_matched=Decision.COLLAPSE,
    )


def _complete_medoid_metadata(*node_ids):
    return {
        node_id: {
            "motif_completeness": 1.0,
            "seqlet_count": 150,
            "core_ic": 10.0,
            "cross_context_recurrence": 1,
        }
        for node_id in node_ids
    }


@dataclass(frozen=True)
class _StabilityStub:
    decision_id: str
    n_affected_peaks: int
    status: str


def test_stability_evidence_is_runtime_checkable_and_structural():
    """Removing runtime_checkable or any of the three declared fields fails."""
    from motifmultiverse.adjudicate import StabilityEvidence

    assert isinstance(_StabilityStub("decision:x", 12, "STABLE"), StabilityEvidence)
    assert not isinstance(object(), StabilityEvidence)


def _ontology_decision(**changes):
    from motifmultiverse.adjudicate import OntologyDecision, stable_decision_id

    payload = {
        "decision_id": stable_decision_id(
            ("node-a", "node-b"), "TRUE_DUPLICATE", "TRUE_DUPLICATE", "test-1"
        ),
        "node_ids": ("node-a", "node-b"),
        "relationship": "TRUE_DUPLICATE",
        "decision": Decision.COLLAPSE,
        "family_id": "FAM_ALPHA",
        "representative_node_id": "node-a",
        "criterion_id": "TRUE_DUPLICATE",
        "criterion_version": "test-1",
        "evidence_ids": ("alignment:one",),
        "evidence_for": ("registered pair satisfies criterion",),
        "evidence_against": (),
        "rationale": "all declared evidence gates passed",
        "decided_by": "automated:test",
        "manual_override": False,
        "provenance": {"criteria_sha256": "a" * 64},
    }
    payload.update(changes)
    return OntologyDecision(**payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": "999"}, "schema_version"),
        ({"decision_id": "decision:corrupted"}, "decision_id"),
        ({"rationale": " "}, "rationale"),
        ({"decided_by": " "}, "decided_by"),
        ({"representative_node_id": "constructed-average"}, "observed member"),
    ],
)
def test_ontology_decision_refuses_each_corrupted_guarded_value(changes, message):
    """Removing the named OntologyDecision validation branch fails its row."""
    from motifmultiverse.schema import SchemaError

    with pytest.raises(SchemaError, match=message):
        _ontology_decision(**changes)


def test_ontology_decision_is_exported_as_a_public_schema_object():
    """Dropping the schema-level public contract while keeping an internal class fails."""
    import motifmultiverse.schema as schema
    from motifmultiverse.adjudicate import OntologyDecision

    assert schema.OntologyDecision is OntologyDecision


def test_family_conflict_refuses_merge_even_when_matrix_similarity_is_high():
    """A similarity-first collapse in place of conflict-first adjudication fails."""
    from motifmultiverse.adjudicate import adjudicate_component

    result = adjudicate_component(
        ["node-a", "node-b"],
        [_alignment(similarity=0.999)],
        [_annotation("node-a", "FAM_ALPHA"), _annotation("node-b", "FAM_BETA")],
        [],
        load_criteria(CRITERIA_PATH),
        "automated:test",
    )

    assert result.relationship == "AMBIGUOUS_CROSS_FAMILY"
    assert result.decision is Decision.REFUSE_MERGE
    assert result.representative_node_id is None
    assert set(result.node_ids) == {"node-a", "node-b"}
    assert result.evidence_against


@pytest.mark.parametrize(
    "configured_decision",
    [Decision.DEFERRED, Decision.COLLAPSE],
)
def test_family_conflict_always_refuses_merge_despite_configured_output(configured_decision):
    """Family conflict is a hard ontology invariant, not a configurable outcome."""
    from motifmultiverse.adjudicate import adjudicate_component

    adversarial = Criterion(
        criterion_id="AMBIGUOUS_CROSS_FAMILY",
        version="adversarial",
        status=CriterionStatus.FROZEN,
        relationship="AMBIGUOUS_CROSS_FAMILY",
        required_evidence=("family_conflict",),
        predicates=(Predicate(field="family_conflict", operator="is_true"),),
        insufficient_evidence_action=Decision.DEFERRED,
        decision_if_matched=configured_decision,
    )
    result = adjudicate_component(
        ["node-a", "node-b"],
        [_alignment(similarity=0.999)],
        [_annotation("node-a", "FAM_ALPHA"), _annotation("node-b", "FAM_BETA")],
        [],
        {"AMBIGUOUS_CROSS_FAMILY": adversarial},
        "automated:test",
    )

    assert result.relationship == "AMBIGUOUS_CROSS_FAMILY"
    assert result.decision is Decision.REFUSE_MERGE


def test_missing_required_downstream_evidence_is_deferred_not_guessed():
    """Filling an absent status or treating high similarity as sufficient fails."""
    from motifmultiverse.adjudicate import adjudicate_component

    result = adjudicate_component(
        ["node-a", "node-b"],
        [_alignment()],
        [_annotation("node-a", "FAM_ALPHA"), _annotation("node-b", "FAM_ALPHA")],
        [],
        {"TRUE_DUPLICATE": _collapse_criterion(require_stability=True)},
        "automated:test",
        node_metadata=_complete_medoid_metadata("node-a", "node-b"),
    )

    assert result.decision is Decision.DEFERRED
    assert result.representative_node_id is None
    assert "status" in result.rationale


def test_present_downstream_evidence_can_satisfy_a_frozen_test_criterion():
    """Ignoring a supplied structural stability row fails this companion control."""
    from motifmultiverse.adjudicate import adjudicate_component, stable_decision_id

    decision_id = stable_decision_id(
        ("node-a", "node-b"), "TRUE_DUPLICATE", "TRUE_DUPLICATE", "test-1"
    )
    result = adjudicate_component(
        ["node-a", "node-b"],
        [_alignment()],
        [_annotation("node-a", "FAM_ALPHA"), _annotation("node-b", "FAM_ALPHA")],
        [_StabilityStub(decision_id, 12, "STABLE")],
        {"TRUE_DUPLICATE": _collapse_criterion(require_stability=True)},
        "automated:test",
        node_metadata=_complete_medoid_metadata("node-a", "node-b"),
    )

    assert result.decision is Decision.COLLAPSE
    assert result.representative_node_id in result.node_ids


def test_manual_override_preserves_automated_decision_and_names_operator_and_reason():
    """Overwriting or relabelling the automated gate result fails this separation."""
    from motifmultiverse.adjudicate import apply_manual_override

    automated = _ontology_decision()
    overridden = apply_manual_override(
        automated,
        operator="curator@example",
        rationale="known paralog-specific binding modes must remain separate",
    )

    assert overridden.decision is Decision.KEEP_SEPARATE_CURATOR_OVERRIDE
    assert overridden.automated_decision is Decision.COLLAPSE
    assert overridden.manual_override is True
    assert overridden.override_operator == "curator@example"
    assert overridden.override_rationale == (
        "known paralog-specific binding modes must remain separate"
    )
    assert "manual override" in overridden.rationale.lower()
    assert "multi-evidence gate" not in overridden.rationale.lower()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"manual_override": True, "decision": Decision.KEEP_SEPARATE_CURATOR_OVERRIDE,
          "automated_decision": Decision.COLLAPSE, "override_operator": "",
          "override_rationale": "reason", "representative_node_id": None}, "operator"),
        ({"manual_override": True, "decision": Decision.KEEP_SEPARATE_CURATOR_OVERRIDE,
          "automated_decision": Decision.COLLAPSE, "override_operator": "curator",
          "override_rationale": "", "representative_node_id": None}, "rationale"),
        ({"manual_override": True, "decision": Decision.KEEP_SEPARATE_CURATOR_OVERRIDE,
          "automated_decision": None, "override_operator": "curator",
          "override_rationale": "reason", "representative_node_id": None},
         "automated decision"),
    ],
)
def test_manual_override_schema_refuses_erased_separation(changes, message):
    """Each missing separation field is a corrupted override, not a gate output."""
    from motifmultiverse.schema import SchemaError

    with pytest.raises(SchemaError, match=message):
        _ontology_decision(**changes)


def test_nontransitive_connected_component_is_deferred_not_single_linkage_collapsed():
    """Deleting the all-pairs/medoid gate turns this A-B-C chain into a false merge."""
    from motifmultiverse.adjudicate import adjudicate_component

    result = adjudicate_component(
        ["node-a", "node-b", "node-c"],
        [_alignment("node-a", "node-b"), _alignment("node-b", "node-c")],
        [],
        [],
        {"TRUE_DUPLICATE": _collapse_criterion(require_stability=False)},
        "automated:test",
    )

    assert result.decision is Decision.DEFERRED
    assert result.representative_node_id is None
    assert "non-transitive" in result.rationale


def test_nontransitive_family_conflict_is_deferred_before_structural_refusal():
    """A structural label cannot bypass the all-pairs evidence requirement."""
    from motifmultiverse.adjudicate import adjudicate_component

    result = adjudicate_component(
        ["node-a", "node-b", "node-c"],
        [_alignment("node-a", "node-b"), _alignment("node-b", "node-c")],
        [
            _annotation("node-a", "FAM_ALPHA"),
            _annotation("node-b", "FAM_BETA"),
            _annotation("node-c", "FAM_BETA"),
        ],
        [],
        load_criteria(CRITERIA_PATH),
        "automated:test",
    )

    assert result.relationship == "AMBIGUOUS_CROSS_FAMILY"
    assert result.decision is Decision.DEFERRED
    assert "non-transitive" in result.rationale


def test_nontransitive_same_family_variants_are_deferred_before_structural_refusal():
    """Distinct variants still need independent evidence for every proposed pair."""
    from motifmultiverse.adjudicate import adjudicate_component

    result = adjudicate_component(
        ["node-a", "node-b", "node-c"],
        [_alignment("node-a", "node-b"), _alignment("node-b", "node-c")],
        [_annotation(node_id, "FAM_ALPHA") for node_id in ("node-a", "node-b", "node-c")],
        [],
        load_criteria(CRITERIA_PATH),
        "automated:test",
        node_metadata={
            "node-a": {"variant_id": "UA_ALPHA_01"},
            "node-b": {"variant_id": "UA_ALPHA_02"},
            "node-c": {"variant_id": "UA_ALPHA_03"},
        },
    )

    assert result.relationship == "SAME_FAMILY_VARIANT"
    assert result.decision is Decision.DEFERRED
    assert "non-transitive" in result.rationale


def test_same_family_distinct_variants_reach_the_structural_refusal_criterion():
    """Routing explicit variant identity into TRUE_DUPLICATE makes this unreachable."""
    from motifmultiverse.adjudicate import adjudicate_component

    result = adjudicate_component(
        ["node-a", "node-b"],
        [_alignment()],
        [_annotation("node-a", "FAM_ALPHA"), _annotation("node-b", "FAM_ALPHA")],
        [],
        load_criteria(CRITERIA_PATH),
        "automated:test",
        node_metadata={
            "node-a": {"variant_id": "UA_ALPHA_01"},
            "node-b": {"variant_id": "UA_ALPHA_02"},
        },
    )

    assert result.relationship == "SAME_FAMILY_VARIANT"
    assert result.decision is Decision.REFUSE_MERGE


def test_component_medoid_uses_authoritative_metadata_without_relabelling_motif_length():
    """Ignoring explicit completeness or substituting candidate length fails this."""
    from motifmultiverse.adjudicate import adjudicate_component

    result = adjudicate_component(
        ["node-a", "node-b", "node-z"],
        [
            _alignment("node-a", "node-b"),
            _alignment("node-a", "node-z"),
            _alignment("node-b", "node-z"),
        ],
        [
            _annotation("node-a", "FAM_ALPHA"),
            _annotation("node-b", "FAM_ALPHA"),
            _annotation("node-z", "FAM_ALPHA"),
        ],
        [],
        {"TRUE_DUPLICATE": _collapse_criterion(require_stability=False)},
        "automated:test",
        node_metadata={
            "node-a": {"motif_completeness": 0.8, "seqlet_count": 1000, "core_ic": 20.0,
                       "cross_context_recurrence": 10},
            "node-b": {"motif_completeness": 0.8, "seqlet_count": 900, "core_ic": 19.0,
                       "cross_context_recurrence": 9},
            "node-z": {"motif_completeness": 0.9, "seqlet_count": 1, "core_ic": 1.0,
                       "cross_context_recurrence": 1},
        },
    )

    assert result.decision is Decision.COLLAPSE
    assert result.representative_node_id == "node-z"


def test_component_defers_when_similarity_tie_lacks_authoritative_medoid_metadata():
    """Lexical fallback is deterministic but is not scientific evidence."""
    from motifmultiverse.adjudicate import adjudicate_component

    result = adjudicate_component(
        ["node-a", "node-b"],
        [_alignment()],
        [],
        [],
        {"TRUE_DUPLICATE": _collapse_criterion(require_stability=False)},
        "automated:test",
        node_metadata={
            "node-a": {
                "motif_completeness": 1.0,
                "seqlet_count": 150,
                "core_ic": 10.0,
            },
            "node-b": {
                "motif_completeness": 1.0,
                "seqlet_count": 150,
                "core_ic": 10.0,
            },
        },
    )

    assert result.decision is Decision.DEFERRED
    assert result.representative_node_id is None
    assert "authoritative medoid tie metadata" in result.rationale
    assert "cross_context_recurrence" in result.rationale


def test_direct_component_refuses_complete_ghost_component_outside_registry():
    """A complete high-scoring graph cannot invent an unregistered representative."""
    from motifmultiverse.adjudicate import AdjudicationError, adjudicate_component

    with pytest.raises(
        AdjudicationError,
        match=r"unknown registry node_id\(s\).*ghost-a.*ghost-b.*ghost-c",
    ):
        adjudicate_component(
            ["ghost-a", "ghost-b", "ghost-c"],
            [
                _alignment("ghost-a", "ghost-b", similarity=0.99),
                _alignment("ghost-a", "ghost-c", similarity=0.98),
                _alignment("ghost-b", "ghost-c", similarity=0.1),
            ],
            [],
            [],
            {"TRUE_DUPLICATE": _collapse_criterion(require_stability=False)},
            "automated:test",
            node_metadata=_complete_medoid_metadata("node-a", "node-b"),
        )


def test_malformed_stability_evidence_is_a_controlled_refusal():
    """Indexing an unchecked object raises AttributeError instead of this domain error."""
    from motifmultiverse.adjudicate import AdjudicationError, adjudicate_component

    with pytest.raises(AdjudicationError, match="stability evidence"):
        adjudicate_component(
            ["node-a", "node-b"],
            [_alignment()],
            [],
            [object()],
            {"TRUE_DUPLICATE": _collapse_criterion(require_stability=True)},
            "automated:test",
        )


# ---------------------------------------------------------------------------
# Task 12: component orchestration, artifacts, identity, and CLI.
# ---------------------------------------------------------------------------

def test_adjudicate_all_emits_every_connected_cluster_including_refusal_and_deferred():
    """Dropping a negative/non-collapse component from the returned audit fails."""
    from motifmultiverse.adjudicate import adjudicate_all

    decisions = adjudicate_all(
        [
            _alignment("node-a", "node-b"),
            _alignment("node-c", "node-d"),
        ],
        [
            _annotation("node-a", "FAM_ALPHA"),
            _annotation("node-b", "FAM_BETA"),
            _annotation("node-c", "FAM_GAMMA"),
            _annotation("node-d", "FAM_GAMMA"),
        ],
        [],
        load_criteria(CRITERIA_PATH),
        "automated:test",
    )

    assert [decision.node_ids for decision in decisions] == [
        ("node-a", "node-b"),
        ("node-c", "node-d"),
    ]
    assert [decision.decision for decision in decisions] == [
        Decision.REFUSE_MERGE,
        Decision.DEFERRED,
    ]


def test_adjudication_artifacts_include_all_states_identity_schema_and_provenance(tmp_path):
    """Omitting a refusal row, an output, identity, schema, or provenance fails."""
    import json

    import pandas as pd
    import yaml

    from motifmultiverse.adjudicate import (
        adjudicate_all,
        write_adjudication_artifacts,
    )
    from motifmultiverse.schema import DecisionBundle

    decisions = adjudicate_all(
        [_alignment("node-a", "node-b"), _alignment("node-c", "node-d")],
        [
            _annotation("node-a", "FAM_ALPHA"),
            _annotation("node-b", "FAM_BETA"),
            _annotation("node-c", "FAM_GAMMA"),
            _annotation("node-d", "FAM_GAMMA"),
        ],
        [],
        load_criteria(CRITERIA_PATH),
        "automated:test",
    )
    paths = write_adjudication_artifacts(
        tmp_path,
        decisions,
        provenance={"criteria_sha256": "b" * 64, "inputs": {"alignment": "c" * 64}},
        review_path="human-review.yaml",
    )

    assert {path.name for path in paths} == {
        "ontology_decisions.parquet", "merge_decisions.json", "human-review.yaml",
    }
    table = pd.read_parquet(tmp_path / "ontology_decisions.parquet")
    bundle_payload = json.loads((tmp_path / "merge_decisions.json").read_text())
    review = yaml.safe_load((tmp_path / "human-review.yaml").read_text())
    bundle = DecisionBundle.from_adjudication_artifact(bundle_payload)
    assert set(table["decision"]) == {"refuse_merge", "deferred"}
    assert set(table["decision_id"]) == {decision.decision_id for decision in decisions}
    assert set(table["schema_version"]) == {"1"}
    assert table["provenance"].str.len().gt(2).all()
    assert table["artifact_id"].str.startswith("ontology-decisions:").all()
    assert len(bundle.decisions) == len(review["decisions"]) == 2
    assert bundle.artifact_id.startswith("merge-decisions:")
    assert bundle.producer == "motifmultiverse.adjudicate"
    assert bundle.provenance["criteria_sha256"] == "b" * 64
    assert review["artifact_id"].startswith("review:")
    assert review["artifact_id"] != bundle.artifact_id


def test_full_ontology_and_review_identity_changes_when_a_scientific_field_changes(tmp_path):
    """Reusing cluster/merge identity after a rationale change fails this tamper control."""
    import json
    from dataclasses import replace

    import pandas as pd
    import yaml

    from motifmultiverse.adjudicate import write_adjudication_artifacts

    original = _ontology_decision()
    revised = replace(original, rationale="a materially revised scientific rationale")
    write_adjudication_artifacts(
        tmp_path / "original", [original], provenance={"input": "a" * 64}
    )
    write_adjudication_artifacts(
        tmp_path / "revised", [revised], provenance={"input": "a" * 64}
    )
    original_table = pd.read_parquet(
        tmp_path / "original" / "ontology_decisions.parquet"
    )
    revised_table = pd.read_parquet(
        tmp_path / "revised" / "ontology_decisions.parquet"
    )
    original_review = yaml.safe_load((tmp_path / "original" / "review.yaml").read_text())
    revised_review = yaml.safe_load((tmp_path / "revised" / "review.yaml").read_text())

    assert original.decision_id == revised.decision_id  # stable cluster identity
    assert original_table["artifact_id"].iat[0] != revised_table["artifact_id"].iat[0]
    assert original_review["artifact_id"] != revised_review["artifact_id"]
    assert json.loads(original_table["provenance"].iat[0])["input"] == "a" * 64


def test_merge_bundle_identity_changes_when_provenance_changes(tmp_path):
    """The compile handoff identity covers provenance, not only decision rows."""
    import json

    from motifmultiverse.adjudicate import write_adjudication_artifacts

    write_adjudication_artifacts(
        tmp_path / "a",
        [_ontology_decision()],
        provenance={"criteria_sha256": "a" * 64},
    )
    write_adjudication_artifacts(
        tmp_path / "b",
        [_ontology_decision()],
        provenance={"criteria_sha256": "b" * 64},
    )
    first = json.loads((tmp_path / "a" / "merge_decisions.json").read_text())
    second = json.loads((tmp_path / "b" / "merge_decisions.json").read_text())

    assert first["decisions"] == second["decisions"]
    assert first["artifact_id"] != second["artifact_id"]


def test_empty_ontology_parquet_carries_identity_and_provenance_in_file_metadata(tmp_path):
    """A zero-row artifact still has an independently inspectable identity."""
    import json

    import pyarrow.parquet as pq

    from motifmultiverse.adjudicate import write_adjudication_artifacts

    provenance = {"criteria_sha256": "c" * 64}
    write_adjudication_artifacts(tmp_path, [], provenance=provenance)
    metadata = pq.read_metadata(tmp_path / "ontology_decisions.parquet").metadata

    assert metadata[b"motifmultiverse.artifact_id"].decode().startswith(
        "ontology-decisions:"
    )
    assert json.loads(metadata[b"motifmultiverse.provenance"]) == provenance
    assert metadata[b"motifmultiverse.schema_version"].decode() == "1"


def test_permissive_policy_is_refused_until_criterion_safe_semantics_are_frozen(tmp_path):
    """Accepting a policy label that changes no behavior creates false provenance."""
    from motifmultiverse.adjudicate import AdjudicationError, adjudicate_evidence

    with pytest.raises(AdjudicationError, match="only conservative"):
        adjudicate_evidence(
            tmp_path,
            tmp_path / "out",
            policy="permissive",
            registry_dir=tmp_path / "registry",
        )


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda payload: payload.update(schema_version="999"), "schema_version"),
        (lambda payload: payload.update(artifact_id="merge-decisions:corrupted"), "artifact_id"),
        (lambda payload: payload.pop("provenance"), "provenance"),
        (
            lambda payload: payload["provenance"].update(criteria_sha256="c" * 64),
            "artifact_id",
        ),
        (lambda payload: payload.update(producer="handwritten"), "producer"),
    ],
)
def test_decision_bundle_strict_handoff_refuses_each_corrupted_artifact(
    tmp_path, corrupt, message,
):
    """Removing any strict adjudication-artifact guard fails its corrupted row."""
    import json

    from motifmultiverse.adjudicate import write_adjudication_artifacts
    from motifmultiverse.schema import DecisionBundle, SchemaError

    write_adjudication_artifacts(
        tmp_path,
        [_ontology_decision()],
        provenance={"criteria_sha256": "b" * 64},
    )
    payload = json.loads((tmp_path / "merge_decisions.json").read_text())
    corrupt(payload)

    with pytest.raises(SchemaError, match=message):
        DecisionBundle.from_adjudication_artifact(payload)


def _write_adjudication_registry(
    tmp_path,
    *,
    completeness_a=0.8,
    completeness_b=0.9,
    collide_variant_ids=False,
):
    registry = tmp_path / "registry"
    registry.mkdir()
    metadata = RegistryMetadata(
        project="adjudication-test",
        peak_universe_id="peaks",
        analyses=[{"model": "model-a"}],
        n_models=1,
        cross_model_claims_restricted=True,
        metacluster_states={},
        trim_threshold=0.3,
        schema_version=REGISTRY_SCHEMA_VERSION,
    )
    nodes = []
    for index, (node_id, completeness, seqlets) in enumerate((
        ("node-a", completeness_a, 500),
        ("node-b", completeness_b, 1),
    )):
        variant_index = 1 if collide_variant_ids else index + 1
        nodes.append(
            MotifNode(
                node_id=node_id,
                model="model-a",
                readout="readout",
                context="promoter",
                metacluster="pos",
                denovo_pattern_id=f"opaque-{node_id}",
                variant_id=f"UA_ALPHA_{variant_index:02d}",
                family_id=MISSING_SENTINEL,
                motif_length=10,
                trimmed_core=[0, 10],
                motif_completeness=completeness,
                seqlet_count=seqlets,
                core_ic=10.0,
                cross_context_recurrence=1,
            )
        )
    payload = {
        "registry_metadata": metadata.__dict__,
        "nodes": [node.to_dict() for node in nodes],
    }
    (registry / "registry.json").write_text(json.dumps(payload, default=str))
    return registry


def test_cli_adjudicate_reads_evidence_and_honors_review_path(tmp_path, capsys):
    """Routing to the skeleton or ignoring --review fails this end-to-end path."""
    import json

    import pandas as pd

    from motifmultiverse.cli import main

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    pd.DataFrame([_alignment().to_dict()]).to_parquet(
        evidence / "alignment_edges.parquet", index=False
    )
    candidate_rows = []
    for candidate in (
        _annotation("node-a", "FAM_ALPHA"),
        _annotation("node-b", "FAM_ALPHA"),
    ):
        row = candidate.to_dict()
        row["provenance"] = json.dumps(row["provenance"])
        candidate_rows.append(row)
    pd.DataFrame(candidate_rows).to_parquet(
        evidence / "annotation_candidates.parquet", index=False
    )
    out = tmp_path / "adjudication"
    registry = _write_adjudication_registry(tmp_path)

    assert main([
        "adjudicate",
        str(evidence),
        "--registry",
        str(registry),
        "--review",
        "human-review.yaml",
        "--out",
        str(out),
    ]) == 0

    assert (out / "ontology_decisions.parquet").exists()
    assert (out / "merge_decisions.json").exists()
    assert (out / "human-review.yaml").exists()
    provenance = json.loads((out / "provenance.json").read_text())
    assert len(provenance) == 1
    assert provenance[0]["subcommand"] == "adjudicate"
    assert "refuse_merge" in capsys.readouterr().out.lower()


def test_cli_adjudicate_requires_an_authoritative_registry():
    """The real CLI cannot silently run without representative metadata."""
    from motifmultiverse.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["adjudicate", "evidence/"])
    assert exc.value.code == 2


def test_adjudicate_refuses_a_registry_with_colliding_semantic_identities(tmp_path):
    from motifmultiverse.adjudicate import AdjudicationError, adjudicate_evidence

    registry = _write_adjudication_registry(tmp_path, collide_variant_ids=True)
    with pytest.raises(AdjudicationError, match="variant_id"):
        adjudicate_evidence(
            tmp_path / "evidence",
            tmp_path / "out",
            registry_dir=registry,
        )


def test_adjudicate_refuses_uncalibrated_persisted_alignment_evidence(tmp_path):
    import pandas as pd

    from motifmultiverse.adjudicate import AdjudicationError, adjudicate_evidence

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    row = _alignment().to_dict()
    row["empirical_p_value"] = None
    row["null_shuffles"] = 0
    pd.DataFrame([row]).to_parquet(evidence / "alignment_edges.parquet", index=False)
    pd.DataFrame(columns=[
        "candidate_id", "node_id", "proposed_family_id", "source", "source_version",
        "matched_motif_id", "match_score", "occurrence_null_value", "motif_length",
        "seqlet_count", "low_confidence_annotation", "provenance",
    ]).to_parquet(evidence / "annotation_candidates.parquet", index=False)
    registry = _write_adjudication_registry(tmp_path)

    with pytest.raises(AdjudicationError, match="calibrated"):
        adjudicate_evidence(evidence, tmp_path / "out", registry_dir=registry)


def test_adjudicate_refuses_forged_alignment_registration_provenance(tmp_path):
    import pandas as pd

    from motifmultiverse.adjudicate import AdjudicationError, adjudicate_evidence

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    row = _alignment().to_dict()
    row["registered_on"] = "signed_cwm"
    pd.DataFrame([row]).to_parquet(evidence / "alignment_edges.parquet", index=False)
    pd.DataFrame(columns=[
        "candidate_id", "node_id", "proposed_family_id", "source", "source_version",
        "matched_motif_id", "match_score", "occurrence_null_value", "motif_length",
        "seqlet_count", "low_confidence_annotation", "provenance",
    ]).to_parquet(evidence / "annotation_candidates.parquet", index=False)
    registry = _write_adjudication_registry(tmp_path)

    with pytest.raises(AdjudicationError, match="registered_on"):
        adjudicate_evidence(evidence, tmp_path / "out", registry_dir=registry)


def test_cli_adjudicate_refuses_all_unknown_evidence_node_ids(tmp_path, capsys):
    import json

    import pandas as pd

    from motifmultiverse.cli import main

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    pd.DataFrame([
        _alignment("ghost-a", "ghost-b", similarity=0.99).to_dict(),
        _alignment("ghost-a", "ghost-c", similarity=0.98).to_dict(),
        _alignment("ghost-b", "ghost-c", similarity=0.1).to_dict(),
    ]).to_parquet(evidence / "alignment_edges.parquet", index=False)
    candidate = _annotation("annotation-ghost", "FAM_ALPHA").to_dict()
    candidate["provenance"] = json.dumps(candidate["provenance"])
    pd.DataFrame([candidate]).to_parquet(
        evidence / "annotation_candidates.parquet", index=False
    )
    criteria = _write_registry(
        tmp_path,
        [{
            "criterion_id": "TRUE_DUPLICATE",
            "version": "ghost-test",
            "status": "FROZEN",
            "relationship": "TRUE_DUPLICATE",
            "required_evidence": ["ppm_similarity"],
            "predicates": [{"field": "ppm_similarity", "operator": "ge", "value": 0.0}],
            "insufficient_evidence_action": "deferred",
            "decision_if_matched": "collapse",
        }],
    )
    registry = _write_adjudication_registry(tmp_path)

    assert main([
        "adjudicate",
        str(evidence),
        "--registry",
        str(registry),
        "--criteria",
        str(criteria),
        "--out",
        str(tmp_path / "out"),
    ]) == 4
    error = capsys.readouterr().err
    assert "unknown registry node_id(s)" in error
    for node_id in ("annotation-ghost", "ghost-a", "ghost-b", "ghost-c"):
        assert node_id in error


def test_real_ingest_registry_completeness_resolves_before_missing_recurrence(
    tmp_path,
):
    import json

    import h5py
    import numpy as np
    import pandas as pd

    from motifmultiverse import ingest
    from motifmultiverse.cli import main

    modisco = tmp_path / "modisco.h5"
    with h5py.File(modisco, "w") as h5:
        patterns = h5.create_group("pos_patterns")
        for index in range(2):
            pattern = patterns.create_group(f"pattern_{index}")
            cwm = np.ones((10, 4))
            if index == 1:
                cwm[:] = 0.0
                cwm[4:6] = 1.0
            pattern.create_dataset("contrib_scores", data=cwm)
            pattern.create_dataset("sequence", data=np.full((10, 4), 0.25))
            pattern.create_group("seqlets").create_dataset("n_seqlets", data=150)
    project = tmp_path / "project.json"
    project.write_text(json.dumps({
        "project": "tie-resolution",
        "peak_universe_id": "peaks",
        "analyses": [{
            "id": "analysis",
            "model": "model",
            "readout": "readout",
            "union_id": "UA",
            "context": "promoter",
            "modisco_h5": str(modisco),
        }],
    }))
    registry = tmp_path / "registry"
    _, nodes = ingest.ingest_project(project, registry)
    assert [node.motif_completeness for node in nodes] == [1.0, 0.2]
    assert all(node.cross_context_recurrence is None for node in nodes)

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    pd.DataFrame([
        _alignment(nodes[0].node_id, nodes[1].node_id).to_dict()
    ]).to_parquet(evidence / "alignment_edges.parquet", index=False)
    pd.DataFrame(columns=[
        "candidate_id", "node_id", "proposed_family_id", "source", "source_version",
        "matched_motif_id", "match_score", "occurrence_null_value", "motif_length",
        "seqlet_count", "low_confidence_annotation", "provenance",
    ]).to_parquet(evidence / "annotation_candidates.parquet", index=False)
    criteria = _write_registry(
        tmp_path,
        [{
            "criterion_id": "TRUE_DUPLICATE",
            "version": "real-ingest-tie",
            "status": "FROZEN",
            "relationship": "TRUE_DUPLICATE",
            "required_evidence": ["ppm_similarity"],
            "predicates": [{"field": "ppm_similarity", "operator": "ge", "value": 0.9}],
            "insufficient_evidence_action": "deferred",
            "decision_if_matched": "collapse",
        }],
    )
    out = tmp_path / "out"

    assert main([
        "adjudicate",
        str(evidence),
        "--registry",
        str(registry),
        "--criteria",
        str(criteria),
        "--out",
        str(out),
    ]) == 0
    decision = pd.read_parquet(out / "ontology_decisions.parquet").iloc[0]
    assert decision["decision"] == "collapse"
    assert decision["representative_node_id"] == nodes[0].node_id


def test_cli_adjudicate_registry_metadata_changes_the_representative(tmp_path):
    """The end-to-end path uses persisted completeness, not lexical ID or zeros."""
    import pandas as pd

    from motifmultiverse.cli import main

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    pd.DataFrame([_alignment().to_dict()]).to_parquet(
        evidence / "alignment_edges.parquet", index=False
    )
    pd.DataFrame(columns=[
        "candidate_id", "node_id", "proposed_family_id", "source", "source_version",
        "matched_motif_id", "match_score", "occurrence_null_value", "motif_length",
        "seqlet_count", "low_confidence_annotation", "provenance",
    ]).to_parquet(evidence / "annotation_candidates.parquet", index=False)
    criteria = _write_registry(
        tmp_path,
        [{
            "criterion_id": "TRUE_DUPLICATE",
            "version": "tie-test",
            "status": "FROZEN",
            "relationship": "TRUE_DUPLICATE",
            "required_evidence": ["ppm_similarity"],
            "predicates": [{"field": "ppm_similarity", "operator": "ge", "value": 0.9}],
            "insufficient_evidence_action": "deferred",
            "decision_if_matched": "collapse",
        }],
    )

    representatives = []
    for name, values in (
        ("prefer-a", (0.9, 0.8)),
        ("prefer-b", (0.8, 0.9)),
    ):
        run_root = tmp_path / name
        run_root.mkdir()
        registry = _write_adjudication_registry(
            run_root,
            completeness_a=values[0],
            completeness_b=values[1],
        )
        out = run_root / "out"
        assert main([
            "adjudicate",
            str(evidence),
            "--registry",
            str(registry),
            "--criteria",
            str(criteria),
            "--out",
            str(out),
        ]) == 0
        table = pd.read_parquet(out / "ontology_decisions.parquet")
        representatives.append(table["representative_node_id"].iat[0])

    assert representatives == ["node-a", "node-b"]


# --- regression: the default criterion registry must survive `pip install` -----
def test_packaged_criteria_resolve_without_the_repository_tree():
    """`--help` called the default "packaged" while it was not in the wheel.

    The default was `Path(__file__).parents[3] / "config" / "criteria.v1.yaml"`,
    which is the repository root only from a source checkout. Installed, it
    pointed above site-packages -- and `config/` was never listed in
    `package-data`, so no wheel could have carried it. A plain `pip install`
    left `adjudicate` unusable at its default.
    """
    from motifmultiverse.adjudicate import packaged_criteria_path
    from motifmultiverse.schema.criteria import load_criteria

    path = packaged_criteria_path()
    assert path.exists(), f"packaged criterion registry missing at {path}"
    # It must live inside the package, not be reached by walking out of it.
    import motifmultiverse
    package_root = Path(motifmultiverse.__file__).resolve().parent
    assert package_root in path.resolve().parents, (
        f"{path} is outside the installed package; it cannot ship in a wheel"
    )
    assert load_criteria(path), "packaged criterion registry is empty"


def test_criteria_resource_is_declared_as_package_data():
    """A resource that is loaded but not declared ships only by accident."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("source tree not present in this installation")
    data = tomllib.loads(pyproject.read_text())
    declared = data["tool"]["setuptools"]["package-data"]["motifmultiverse"]
    assert any("criteria.v1.yaml" in entry for entry in declared), (
        f"criteria.v1.yaml is loaded at runtime but not in package-data: {declared}"
    )
