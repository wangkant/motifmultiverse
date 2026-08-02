"""Family assignment: the consensus rule, and the refusals it must not skip.

The rule replaces "take the top database hit". Every test here pins a way that
substitution could quietly become the old rule again.
"""
from __future__ import annotations

import pytest

from motifmultiverse.schema import SchemaError
from motifmultiverse.schema.family import (
    FAMILY_ASSIGNMENT_SCHEMA_VERSION,
    FamilyAssignment,
    FamilyAssignmentState,
    FamilyVocabulary,
    assign_family_by_consensus,
    stable_family_assignment_id,
)

VOCAB = FamilyVocabulary(
    vocabulary_id="test_archetypes", version="v1.0", content_sha256="0" * 64,
    mapping={
        "SP1_HUMAN.H11MO.0.A": "T:109", "KLF12_HUMAN.H11MO.0.C": "T:109",
        "SP3_HUMAN.H11MO.0.B": "T:109",
        "ATF4_HUMAN.H11MO.0.A": "T:51", "CREB1_MA0018.3": "T:51",
        "CTCF_MA0139.1": "T:265",
        "GATA1_MOUSE.H11MO.0.A": "T:242",
        # Two clusters, one display name -- the collision the id must survive.
        "ZNF143_C2H2_1": "T:172", "IKZF1_HUMAN.H11MO.0.C": "T:173",
    },
    labels={"T:109": "KLF/SP/2", "T:51": "CREB/ATF/3", "T:265": "CTCF",
            "T:242": "GATA", "T:172": "ZNF143", "T:173": "ZNF143"},
)


def _assign(ids, **kw):
    return assign_family_by_consensus(node_id="n", matched_motif_ids=ids, vocabulary=VOCAB, **kw)


# --------------------------------------------------------------- the rule
def test_three_candidates_naming_one_family_are_unanimous():
    a = _assign(["SP1_HUMAN.H11MO.0.A", "KLF12_HUMAN.H11MO.0.C", "SP3_HUMAN.H11MO.0.B"])
    assert a.state is FamilyAssignmentState.ASSIGNED_UNANIMOUS
    assert (a.family_id, a.family_label) == ("T:109", "KLF/SP/2")
    assert (a.n_candidates, a.n_resolved, a.n_agreeing) == (3, 3, 3)


def test_two_of_three_is_assigned_but_not_recorded_as_unanimous():
    a = _assign(["SP1_HUMAN.H11MO.0.A", "CTCF_MA0139.1", "SP3_HUMAN.H11MO.0.B"])
    assert a.state is FamilyAssignmentState.ASSIGNED_MAJORITY
    assert a.family_id == "T:109" and a.n_agreeing == 2


def test_a_chance_top_hit_cannot_name_a_family_on_its_own():
    """The failure the rule exists for.

    A pattern matched ATF4 at q=1.000 -- chance -- and under a top-hit rule
    inherited that matrix's family, which then dominated its layer. Here the
    chance hit is outvoted by the two that agree, and the family it proposed is
    not the one assigned.
    """
    a = _assign(["ATF4_HUMAN.H11MO.0.A", "SP1_HUMAN.H11MO.0.A", "SP3_HUMAN.H11MO.0.B"])
    assert a.state is FamilyAssignmentState.ASSIGNED_MAJORITY
    assert a.family_id == "T:109", "the rank-0 candidate must not win by being rank 0"


def test_three_candidates_naming_three_families_is_a_typed_hole():
    a = _assign(["SP1_HUMAN.H11MO.0.A", "CTCF_MA0139.1", "GATA1_MOUSE.H11MO.0.A"])
    assert a.state is FamilyAssignmentState.NOT_ASSIGNED_SPLIT
    assert a.family_id is None and a.n_agreeing == 0


def test_a_tie_is_a_hole_and_is_never_broken_by_rank():
    """Rank is not evidence. Falling back to rank-0 on a tie IS the top-hit rule."""
    a = _assign(["SP1_HUMAN.H11MO.0.A", "SP3_HUMAN.H11MO.0.B",
                 "ATF4_HUMAN.H11MO.0.A", "CREB1_MA0018.3"])
    assert a.state is FamilyAssignmentState.NOT_ASSIGNED_SPLIT
    assert a.family_id is None


def test_one_candidate_alone_cannot_agree_with_itself():
    a = _assign(["SP1_HUMAN.H11MO.0.A"])
    assert a.state is FamilyAssignmentState.NOT_ASSIGNED_TOO_FEW
    assert a.family_id is None


def test_no_candidate_and_no_declared_candidate_are_different_refusals():
    assert _assign([]).state is FamilyAssignmentState.NOT_ASSIGNED_NO_CANDIDATE
    assert _assign(["NOT_IN_VOCAB_1", "NOT_IN_VOCAB_2"]).state is \
        FamilyAssignmentState.NOT_ASSIGNED_UNDECLARED


def test_undeclared_candidates_are_reported_but_never_lower_the_bar():
    a = _assign(["NOT_IN_VOCAB", "SP1_HUMAN.H11MO.0.A"])
    assert a.state is FamilyAssignmentState.NOT_ASSIGNED_TOO_FEW
    assert (a.n_candidates, a.n_resolved) == (2, 1)
    assert a.evidence == (("NOT_IN_VOCAB", None), ("SP1_HUMAN.H11MO.0.A", "T:109"))


def test_min_agreeing_of_one_is_refused_because_it_is_the_top_hit_rule():
    with pytest.raises(SchemaError, match="top-hit rule"):
        _assign(["SP1_HUMAN.H11MO.0.A"], min_agreeing=1)


# ------------------------------------------------------- identity and shape
def test_two_clusters_sharing_a_display_name_stay_distinct():
    """Vierstra v1.0 has 286 clusters and 282 names; the id must carry the difference."""
    a = _assign(["ZNF143_C2H2_1", "ZNF143_C2H2_1", "IKZF1_HUMAN.H11MO.0.C"])
    assert a.family_id == "T:172" and a.family_label == "ZNF143"
    b = _assign(["IKZF1_HUMAN.H11MO.0.C", "IKZF1_HUMAN.H11MO.0.C", "ZNF143_C2H2_1"])
    assert b.family_id == "T:173" and b.family_label == "ZNF143"
    assert a.family_id != b.family_id, "same label, different family: ids must differ"


def test_the_assignment_id_admits_a_second_proposer_instead_of_overwriting():
    one = stable_family_assignment_id(node_id="n", rule_id="a", rule_version="1",
                                      vocabulary_id="v", vocabulary_version="1")
    other = stable_family_assignment_id(node_id="n", rule_id="b", rule_version="1",
                                        vocabulary_id="v", vocabulary_version="1")
    assert one != other
    assert one == stable_family_assignment_id(
        node_id="n", rule_id="a", rule_version="1", vocabulary_id="v", vocabulary_version="1")


def test_a_refusal_that_names_a_family_is_refused_by_the_type():
    with pytest.raises(SchemaError, match="must not carry"):
        FamilyAssignment(
            assignment_id="x", node_id="n", state=FamilyAssignmentState.NOT_ASSIGNED_SPLIT,
            family_id="T:109", family_label="KLF/SP/2", n_candidates=3, n_resolved=3,
            n_agreeing=0, rule_id="r", rule_version="1", vocabulary_id="v",
            vocabulary_version="1", vocabulary_sha256="0" * 64)


def test_an_assignment_that_names_no_family_is_refused_by_the_type():
    with pytest.raises(SchemaError, match="names none"):
        FamilyAssignment(
            assignment_id="x", node_id="n", state=FamilyAssignmentState.ASSIGNED_UNANIMOUS,
            family_id=None, family_label=None, n_candidates=3, n_resolved=3, n_agreeing=3,
            rule_id="r", rule_version="1", vocabulary_id="v", vocabulary_version="1",
            vocabulary_sha256="0" * 64)


def test_unanimous_with_a_dissenter_is_refused_by_the_type():
    with pytest.raises(SchemaError, match="UNANIMOUS requires"):
        FamilyAssignment(
            assignment_id="x", node_id="n", state=FamilyAssignmentState.ASSIGNED_UNANIMOUS,
            family_id="T:109", family_label="KLF/SP/2", n_candidates=3, n_resolved=3,
            n_agreeing=2, rule_id="r", rule_version="1", vocabulary_id="v",
            vocabulary_version="1", vocabulary_sha256="0" * 64)


# ----------------------------------------------------------- the vocabulary
@pytest.mark.parametrize("field", ["vocabulary_id", "version", "content_sha256"])
def test_an_uncitable_vocabulary_is_refused_not_defaulted(field):
    kw = dict(vocabulary_id="v", version="1", content_sha256="0" * 64, mapping={"a": "b"})
    kw[field] = ""
    with pytest.raises(SchemaError, match=field):
        FamilyVocabulary(**kw)


def test_an_empty_vocabulary_is_refused_because_it_looks_like_a_stage_that_did_not_run():
    with pytest.raises(SchemaError, match="empty"):
        FamilyVocabulary(vocabulary_id="v", version="1", content_sha256="0" * 64, mapping={})


def test_every_assignment_carries_the_vocabulary_it_was_made_under():
    a = _assign(["SP1_HUMAN.H11MO.0.A", "SP3_HUMAN.H11MO.0.B"])
    assert (a.vocabulary_id, a.vocabulary_version, a.vocabulary_sha256) == \
        ("test_archetypes", "v1.0", "0" * 64)
    assert FAMILY_ASSIGNMENT_SCHEMA_VERSION == "1"
