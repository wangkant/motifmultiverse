"""Adjudicate stage tests.

Task 9 (this file's creator) owns the first behavior the file covers: the
criterion registry loader, the evaluator, and ``DEFERRED`` semantics
(``schema.criteria``). Task 12 later extends this same file with
``adjudicate_component`` / medoid-selection tests; it does not replace anything
below.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from motifmultiverse.schema import Decision
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

CRITERIA_PATH = Path(__file__).resolve().parents[1] / "config" / "criteria.v1.yaml"


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


def test_criterion_loader_rejects_an_invalid_decision_if_matched(tmp_path):
    path = _write_registry(tmp_path, [_entry(decision_if_matched="colapse")])
    with pytest.raises(CriteriaError):
        load_criteria(path)


def test_criterion_loader_rejects_required_evidence_that_is_not_a_list(tmp_path):
    """`required_evidence: score` (a bare string, missing the YAML list brackets)
    must be refused, not silently iterated into one field name per character
    (`('s', 'c', 'o', 'r', 'e')`).
    """
    path = _write_registry(tmp_path, [_entry(required_evidence="score")])
    with pytest.raises(CriteriaError, match="required_evidence"):
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
