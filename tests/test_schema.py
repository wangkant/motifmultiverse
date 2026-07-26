"""Schema tests: the four rules from real failures must be enforced, not documented."""
from __future__ import annotations

import pytest

from motifmultiverse.schema import (
    AnalysisConfig,
    Decision,
    DecisionRecord,
    EvidenceEdge,
    IdentityError,
    Missingness,
    MotifNode,
    NamespacedId,
    SchemaError,
    Tier,
    translate,
)


def _node(**kw):
    base = dict(
        node_id="n0", model="modelA", readout="r1", context="promoter",
        metacluster="pos", denovo_pattern_id="pattern_0",
        variant_id="UA_FAM_00", family_id="FAM",
    )
    base.update(kw)
    return MotifNode(**base)


# ---- rule 1: variant_id is the stable semantic identity, and is marked -------
def test_valid_node_builds():
    assert _node().identity == NamespacedId("variant_id", "UA_FAM_00")


@pytest.mark.parametrize("bad", ["", "nofmt", "UA-FAM-00", "UA_FAM", "UA_FAM_X"])
def test_malformed_variant_id_rejected(bad):
    with pytest.raises(SchemaError):
        _node(variant_id=bad)


# ---- rule 2: no semantics parsed out of identifier strings -------------------
def test_translate_requires_namespaced_id():
    with pytest.raises(IdentityError):
        translate("UA_FAM_00", {"UA_FAM_00": "x"}, "motif_name")


def test_translate_raises_on_unknown_key():
    with pytest.raises(IdentityError):
        translate(NamespacedId("variant_id", "UA_FAM_99"), {"UA_FAM_00": "x"}, "motif_name")


def test_translate_succeeds_with_explicit_table():
    out = translate(NamespacedId("variant_id", "UA_FAM_00"), {"UA_FAM_00": "m1"}, "motif_name")
    assert out.namespace == "motif_name" and out.value == "m1"


def test_namespace_with_slash_rejected():
    with pytest.raises(IdentityError):
        NamespacedId("a/b", "v")


# ---- rule 3: four-state missingness, never collapsed to 0 -------------------
def test_all_four_missingness_states_exist():
    assert {m.value for m in Missingness} == {
        "not_searched", "no_sequence_match", "hit_below_floor", "used"}


def test_undefined_edge_may_not_carry_zero():
    with pytest.raises(SchemaError):
        EvidenceEdge("sequence_hit", "a", "b", statistic=0,
                     missingness=Missingness.NO_SEQUENCE_MATCH)


def test_used_edge_may_carry_zero():
    assert EvidenceEdge("sequence_hit", "a", "b", statistic=0).statistic == 0


def test_unknown_edge_type_rejected():
    with pytest.raises(SchemaError):
        EvidenceEdge("not_a_type", "a", "b")


# ---- rule 4: refusals are expressible; confidence is a measure --------------
def test_refusal_is_a_first_class_decision():
    d = DecisionRecord("c0", Decision.REFUSE_MERGE, ["a", "b"],
                       rationale="geometry passed but TF identity differs",
                       decided_by="curator")
    assert d.decision is Decision.REFUSE_MERGE


def test_decision_requires_rationale_and_decider():
    with pytest.raises(SchemaError):
        DecisionRecord("c0", Decision.COLLAPSE, ["a"], rationale="  ", decided_by="x")
    with pytest.raises(SchemaError):
        DecisionRecord("c0", Decision.COLLAPSE, ["a"], rationale="ok", decided_by="")


@pytest.mark.parametrize("bad", [-0.1, 1.5, 42])
def test_confidence_must_be_a_measure(bad):
    with pytest.raises(SchemaError):
        DecisionRecord("c0", Decision.COLLAPSE, ["a"], rationale="ok",
                       decided_by="x", confidence=bad)


# ---- T-13: two tier fields, and a reason when they disagree -----------------
def test_diverging_tiers_require_a_reason():
    with pytest.raises(SchemaError):
        _node(discovery_tier=Tier.CORE, analysis_tier=Tier.EXCLUDED)
    ok = _node(discovery_tier=Tier.CORE, analysis_tier=Tier.EXCLUDED,
               tier_reason="not an independent detector")
    assert ok.analysis_tier is Tier.EXCLUDED


# ---- T-01: N>=3 for between-model heterogeneity -----------------------------
def _cfg(models):
    return AnalysisConfig("p", [{"id": f"a{i}", "model": m} for i, m in enumerate(models)])


@pytest.mark.parametrize("models", [["A"], ["A", "B"], ["A", "A", "B"]])
def test_between_model_heterogeneity_refused_below_three_models(models):
    with pytest.raises(SchemaError, match="not estimable"):
        _cfg(models).assert_between_model_heterogeneity_estimable()


def test_between_model_heterogeneity_allowed_at_three_models():
    _cfg(["A", "B", "C"]).assert_between_model_heterogeneity_estimable()


def test_analyses_list_is_unbounded():
    cfg = _cfg([f"m{i}" for i in range(25)])
    assert cfg.n_models == 25
    cfg.assert_between_model_heterogeneity_estimable()


def test_duplicate_analysis_ids_rejected():
    with pytest.raises(SchemaError):
        AnalysisConfig("p", [{"id": "a", "model": "A"}, {"id": "a", "model": "B"}])
