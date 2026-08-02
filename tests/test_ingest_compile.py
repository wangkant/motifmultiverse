"""ingest + compile: the registry, the three absences, and the loader round trip.

The round-trip test is behavioural. Asserting that the written HDF5 contains the
groups we just wrote would prove only that this package can read its own output;
the question is whether the *hit caller* can, and in which order it hands the
motifs back.

It needs the finemo backend, which `pip install -e ".[finemo]"` supplies. Where
the backend is absent it still skips, and the skip must be read as "unverified
here", not as "verified" -- `MOTIFMULTIVERSE_REQUIRE_FINEMO=1` turns it into a
failure for runs that are not allowed to make that trade. The tests asserting the
*no-backend* path no longer skip at all: they blank the import (see
`conftest.no_finemo_backend`) rather than waiting for a machine that lacks it,
because a pair of tests that skip under opposite conditions is a pair that is
never both verified.
"""
from __future__ import annotations

import dataclasses
import json
import re
import sys

import pytest

from conftest import require_finemo_backend
from motifmultiverse import compile as compile_mod
from motifmultiverse import guards, ingest
from motifmultiverse.schema import (
    DECISION_BUNDLE_PRODUCER,
    DECISION_BUNDLE_SCHEMA_VERSION,
    IDENTITY_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
    Decision,
    MetaclusterState,
    RegistryMetadata,
    RepresentationId,
    SchemaError,
    VariantId,
    decision_bundle_artifact_id,
)

h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

MOTIF_LEN = 12


def _pattern(h5, group, name, seed):
    rng = np.random.default_rng(seed)
    grp = h5.require_group(group).create_group(name)
    cwm = rng.normal(size=(MOTIF_LEN, 4))
    grp.create_dataset("contrib_scores", data=cwm)
    grp.create_dataset("hypothetical_contribs", data=cwm * 0.5)
    ppm = np.abs(rng.normal(size=(MOTIF_LEN, 4)))
    grp.create_dataset("sequence", data=ppm / ppm.sum(axis=1, keepdims=True))
    grp.create_group("seqlets").create_dataset("n_seqlets", data=np.array(250 + seed))


def _modisco(path, n_pos=3, n_neg=2, neg_group="present"):
    with h5py.File(path, "w") as h5:
        for i in range(n_pos):
            _pattern(h5, "pos_patterns", f"pattern_{i}", seed=i)
        if neg_group == "present":
            for i in range(n_neg):
                _pattern(h5, "neg_patterns", f"pattern_{i}", seed=100 + i)
        elif neg_group == "empty":
            h5.require_group("neg_patterns")
        # neg_group == "absent": write nothing at all
    return path


def _project(tmp_path, **over):
    analyses = over.pop("analyses", None)
    if analyses is None:
        analyses = [{
            "id": "modelA_r1", "model": "modelA", "readout": "r1", "union_id": "MA",
            "context": "promoter", "modisco_h5": str(_modisco(tmp_path / "a.h5")),
        }]
    cfg = {"project": "test-project", "peak_universe_id": "u1", "analyses": analyses}
    cfg.update(over)
    path = tmp_path / "project.json"
    path.write_text(json.dumps(cfg))
    return path


# ----------------------------------------------------------------------- ingest
def test_ingest_builds_a_registry_with_the_six_field_groups(tmp_path):
    meta, nodes = ingest.ingest_project(_project(tmp_path), tmp_path / "registry")
    assert len(nodes) == 5                      # 3 pos + 2 neg
    n = nodes[0]
    assert n.model == "modelA" and n.readout == "r1" and n.context == "promoter"
    assert (
        n.metacluster == "pos"
        and n.denovo_pattern_id == "modelA_r1::pos_patterns.pattern_0"
    )
    assert n.motif_length == MOTIF_LEN and n.seqlet_count == 250
    assert n.core_ic is not None and n.trimmed_core is not None
    assert n.motif_completeness == (n.trimmed_core[1] - n.trimmed_core[0]) / MOTIF_LEN
    assert n.cross_context_recurrence is None
    assert n.provenance["modisco_h5_sha256"] and n.provenance["analysis_id"] == "modelA_r1"
    assert meta.project == "test-project"
    assert meta.schema_version == REGISTRY_SCHEMA_VERSION


def test_ingest_writes_arrays_and_provenance(tmp_path):
    out = tmp_path / "registry"
    ingest.ingest_project(_project(tmp_path), out)
    assert (out / "registry.json").exists() and (out / "arrays.h5").exists()
    prov = json.loads((out / "provenance.json").read_text())[0]
    assert prov["subcommand"] == "ingest"
    # h5 inputs are keyed by the config's analysis_id, not by basename -- see
    # test_ingest_records_one_checksum_per_analysis_even_when_the_files_share_a_name
    assert set(prov["inputs"]) == {"project.json", "modelA_r1:a.h5"}
    assert prov["redaction_policy"] == "basenames_only_except_command"
    meta, records, arrays = ingest.load_registry(out)
    try:
        assert len(records) == 5
        assert set(arrays[records[0]["node_id"]].keys()) == {"cwm", "hypothetical_cwm", "ppm"}
    finally:
        arrays.close()


def test_registry_loader_refuses_an_unversioned_persisted_registry(tmp_path):
    out = tmp_path / "registry"
    ingest.ingest_project(_project(tmp_path), out)
    payload = json.loads((out / "registry.json").read_text())
    payload["registry_metadata"].pop("schema_version")
    (out / "registry.json").write_text(json.dumps(payload))

    with pytest.raises(SchemaError, match="schema_version"):
        ingest.load_registry(out)


@pytest.mark.parametrize("neg_group,search,expected", [
    ("present", True, MetaclusterState.PRESENT),
    ("empty", True, MetaclusterState.GROUP_EMPTY),
    ("absent", True, MetaclusterState.GROUP_ABSENT),
    ("present", False, MetaclusterState.NOT_SEARCHED),
])
def test_the_three_absences_are_three_states(tmp_path, neg_group, search, expected):
    """group_absent, group_empty and not_searched are different claims (V-08).

    Absent means the metacluster never formed -- evidence about the admission gate.
    Empty means discovery looked and found nothing. not_searched means no evidence
    either way. Collapsed into "no negative motifs", all three become one false
    statement.
    """
    h5_path = _modisco(tmp_path / "a.h5", neg_group=neg_group)
    analyses = [{"id": "a1", "model": "m", "readout": "r", "union_id": "MA",
                 "context": "promoter", "modisco_h5": str(h5_path),
                 "search_metaclusters": {"pos_patterns": True, "neg_patterns": search}}]
    meta, nodes = ingest.ingest_project(_project(tmp_path, analyses=analyses),
                                        tmp_path / "registry")
    assert meta.metacluster_states["a1"]["neg_patterns"] == expected.value
    assert len(nodes) == (5 if expected is MetaclusterState.PRESENT else 3)


def test_registry_metadata_refuses_a_state_it_does_not_recognise():
    with pytest.raises(SchemaError, match="three different claims"):
        RegistryMetadata(project="p", peak_universe_id="u", analyses=[{"model": "m"}],
                         n_models=1, cross_model_claims_restricted=True,
                         metacluster_states={"a1": {"neg_patterns": "no_motifs"}},
                         trim_threshold=0.3)


def test_cross_model_restriction_is_derived_not_declared(tmp_path):
    _, _ = ingest.ingest_project(_project(tmp_path), tmp_path / "registry")
    meta, _, arrays = ingest.load_registry(tmp_path / "registry")
    arrays.close()
    assert meta.n_models == 1 and meta.cross_model_claims_restricted is True
    with pytest.raises(SchemaError, match="derived, not declared"):
        RegistryMetadata(project="p", peak_universe_id="u",
                         analyses=[{"model": "a"}, {"model": "b"}, {"model": "c"}],
                         n_models=3, cross_model_claims_restricted=True,
                         metacluster_states={}, trim_threshold=0.3)


def test_three_models_lift_the_restriction(tmp_path):
    analyses = [{"id": f"a{i}", "model": f"m{i}", "readout": "r", "union_id": f"U{i}",
                 "context": "promoter", "modisco_h5": str(_modisco(tmp_path / f"{i}.h5"))}
                for i in range(3)]
    meta, _ = ingest.ingest_project(_project(tmp_path, analyses=analyses),
                                    tmp_path / "registry")
    assert meta.n_models == 3 and meta.cross_model_claims_restricted is False


def test_ingest_keeps_registry_identities_unique_across_shared_union_contexts(tmp_path):
    analyses = [
        {
            "id": f"a{i}",
            "model": "model",
            "readout": "readout",
            "union_id": "SHARED",
            "context": context,
            "modisco_h5": str(_modisco(tmp_path / f"{i}.h5", n_pos=1, n_neg=0)),
        }
        for i, context in enumerate(("promoter", "enhancer"))
    ]
    _, nodes = ingest.ingest_project(
        _project(tmp_path, analyses=analyses),
        tmp_path / "registry",
    )

    assert len({node.variant_id for node in nodes}) == len(nodes)
    assert len({node.denovo_pattern_id for node in nodes}) == len(nodes)
    assert all(node.cross_context_recurrence is None for node in nodes)


# -------------------------------------------- analyses that share one attribution
#
# `shared_attribution_groups` was documented in `config/project.example.yaml`,
# with a paragraph on why a corroboration count must use the number of distinct
# sources -- and `grep` found the identifier nowhere else. A user who set it got
# nothing, silently. The case study is the case that needed it: thirteen analyses,
# one ChromBPNet model, one counts-head DeepSHAP readout.
def _two_analyses(tmp_path):
    return [{"id": f"a{i}", "model": "m", "readout": "r", "union_id": "U",
             "context": f"c{i}", "modisco_h5": str(_modisco(tmp_path / f"{i}.h5",
                                                            n_pos=1, n_neg=0))}
            for i in range(2)]


def test_shared_attribution_groups_reach_the_registry(tmp_path):
    """Declared sharing survives ingest and the registry.json round trip."""
    project = _project(tmp_path, analyses=_two_analyses(tmp_path),
                       shared_attribution_groups=[["a0", "a1"]])
    meta, _ = ingest.ingest_project(project, tmp_path / "registry")
    assert meta.shared_attribution_groups == [["a0", "a1"]]
    # Two analyses, one attribution array: one source, not two.
    assert meta.n_attribution_sources == 1

    reloaded, _, arrays = ingest.load_registry(tmp_path / "registry")
    arrays.close()
    assert reloaded.shared_attribution_groups == [["a0", "a1"]]
    assert reloaded.n_attribution_sources == 1
    on_disk = json.loads((tmp_path / "registry" / "registry.json").read_text())
    assert on_disk["registry_metadata"]["n_attribution_sources"] == 1


def test_declaring_nothing_shared_is_not_the_same_as_declaring_nothing(tmp_path):
    """`[]` is a claim; an absent key is not. Silence must not become a count."""
    declared = _project(tmp_path, analyses=_two_analyses(tmp_path),
                        shared_attribution_groups=[])
    meta, _ = ingest.ingest_project(declared, tmp_path / "declared")
    assert meta.shared_attribution_groups == [] and meta.n_attribution_sources == 2

    silent = _project(tmp_path, analyses=_two_analyses(tmp_path))
    meta, _ = ingest.ingest_project(silent, tmp_path / "silent")
    assert meta.shared_attribution_groups is None
    # NOT 2. Nothing was declared, so the number of distinct sources is unknown,
    # and len(analyses) would be an invented claim of independence.
    assert meta.n_attribution_sources is None


def test_a_shared_attribution_group_of_one_is_refused(tmp_path):
    """The shape the shipped example carried: a declaration that changes nothing."""
    project = _project(tmp_path, analyses=_two_analyses(tmp_path),
                       shared_attribution_groups=[["a0"]])
    with pytest.raises(SchemaError, match="declares nothing"):
        ingest.ingest_project(project, tmp_path / "registry")


def test_a_shared_attribution_group_naming_an_unknown_analysis_is_refused(tmp_path):
    project = _project(tmp_path, analyses=_two_analyses(tmp_path),
                       shared_attribution_groups=[["a0", "a7"]])
    with pytest.raises(SchemaError, match="not a declared analysis id"):
        ingest.ingest_project(project, tmp_path / "registry")


def test_an_analysis_cannot_share_two_attribution_arrays(tmp_path):
    analyses = _two_analyses(tmp_path)
    analyses.append({"id": "a2", "model": "m", "readout": "r", "union_id": "U",
                     "context": "c2",
                     "modisco_h5": str(_modisco(tmp_path / "2.h5", n_pos=1, n_neg=0))})
    project = _project(tmp_path, analyses=analyses,
                       shared_attribution_groups=[["a0", "a1"], ["a1", "a2"]])
    with pytest.raises(SchemaError, match="at most one shared group"):
        ingest.ingest_project(project, tmp_path / "registry")


def test_a_flat_list_of_analysis_ids_is_refused(tmp_path):
    """`[a, b]` (one group, flattened) must not be read as two groups of one."""
    project = _project(tmp_path, analyses=_two_analyses(tmp_path),
                       shared_attribution_groups=["a0", "a1"])
    with pytest.raises(SchemaError, match=r"\[\[a, b\]\], not \[a, b\]"):
        ingest.ingest_project(project, tmp_path / "registry")


def test_the_source_count_is_derived_not_declared():
    """Same rule as cross_model_claims_restricted: the constructor checks it."""
    with pytest.raises(SchemaError, match="derived from the declared groups"):
        RegistryMetadata(project="p", peak_universe_id="u",
                         analyses=[{"id": "a0", "model": "m"}, {"id": "a1", "model": "m"}],
                         n_models=1, cross_model_claims_restricted=True,
                         metacluster_states={}, trim_threshold=0.3,
                         shared_attribution_groups=[["a0", "a1"]],
                         n_attribution_sources=2)


def test_thirteen_analyses_over_one_attribution_source_count_as_one(tmp_path):
    """The case study's shape: 13 discovery contexts, one model, one readout.

    `registry: 139 motif nodes from 13 analyses` is what the run printed. The
    thirteen are thirteen Leiden slices of ONE DeepSHAP attribution over one peak
    universe, so a corroboration count over them has one source to divide by.
    """
    analyses = [{"id": f"cl{i}", "model": "cbp", "readout": "counts", "union_id": "CBPK562",
                 "context": f"leiden_cl{i}",
                 "modisco_h5": str(_modisco(tmp_path / f"cl{i}.h5", n_pos=1, n_neg=0))}
                for i in range(13)]
    project = _project(tmp_path, analyses=analyses,
                       shared_attribution_groups=[[f"cl{i}" for i in range(13)]])
    meta, nodes = ingest.ingest_project(project, tmp_path / "registry")
    assert len(meta.analyses) == 13
    assert meta.n_attribution_sources == 1
    # And the count-of-corroboration field is still unpopulated -- the thing the
    # denominator is being recorded for has not arrived yet.
    assert all(node.cross_context_recurrence is None for node in nodes)


def test_the_shipped_example_project_config_is_read_not_only_documented():
    """`config/project.example.yaml` must not document a field nothing reads."""
    from pathlib import Path

    import motifmultiverse

    root = Path(motifmultiverse.__file__).resolve().parents[2]
    example = root / "config" / "project.example.yaml"
    if not example.exists():
        pytest.skip("config/ not present in this installation")
    raw = ingest.read_project(example)
    assert "shared_attribution_groups" in raw, "the example still documents the field"
    cfg = ingest.validate_project(raw)
    # The example's own value must survive validation -- the version that shipped
    # (`- [modelB_readout1]`) does not, which is how a never-read field stays wrong.
    assert cfg.shared_attribution_groups == raw["shared_attribution_groups"]
    assert cfg.n_attribution_sources == len(cfg.analyses)


def test_union_id_must_be_declared_not_derived(tmp_path):
    analyses = [{"id": "a1", "model": "m", "readout": "r", "context": "promoter",
                 "modisco_h5": str(_modisco(tmp_path / "a.h5"))}]
    with pytest.raises(ingest.IngestError, match="union_id"):
        ingest.ingest_project(_project(tmp_path, analyses=analyses), tmp_path / "registry")


def test_a_union_id_that_would_need_parsing_is_refused(tmp_path):
    analyses = [{"id": "a1", "model": "m", "readout": "r", "union_id": "CBP_2048",
                 "context": "promoter", "modisco_h5": str(_modisco(tmp_path / "a.h5"))}]
    with pytest.raises(ingest.IngestError, match="alphanumeric"):
        ingest.ingest_project(_project(tmp_path, analyses=analyses), tmp_path / "registry")


def test_pattern_ids_are_opaque_join_tokens(tmp_path):
    """V-09: a digit inside an identifier is never read by the package.

    A reference-implementation key said 2048 while the real input width was 2114.
    It was harmless only because nothing parsed it. Here that is checked rather
    than hoped: the guard runs over ingest's and compile's own source.
    """
    import pathlib
    _, nodes = ingest.ingest_project(_project(tmp_path), tmp_path / "registry")
    assert {n.denovo_pattern_id for n in nodes} == {
        "modelA_r1::pos_patterns.pattern_0",
        "modelA_r1::pos_patterns.pattern_1",
        "modelA_r1::pos_patterns.pattern_2",
        "modelA_r1::neg_patterns.pattern_0",
        "modelA_r1::neg_patterns.pattern_1",
    }
    for mod in (ingest, compile_mod):
        src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        assert guards.no_key_parsing(src).passed, mod.__name__


# ---------------------------------------------------------------------- compile
def _registry(tmp_path):
    ingest.ingest_project(_project(tmp_path), tmp_path / "registry")
    return tmp_path / "registry"


def test_compile_writes_one_lexicon_per_tier_with_a_manifest(tmp_path):
    manifests = compile_mod.compile_lexicons(_registry(tmp_path), tmp_path / "lex")
    assert set(manifests) == set(compile_mod.TIERS)
    for tier, manifest in manifests.items():
        assert (tmp_path / "lex" / f"{tier}.h5").exists()
        assert manifest.n_motifs == 5
        assert len(manifest.lexicon_content_hash) == 64
    assert (tmp_path / "lex" / "manifest.tsv").exists()


def test_validation_accepts_the_compiler_public_output_contract(tmp_path):
    """The strict validator must consume real compile output, not a parallel fixture format."""
    from motifmultiverse.validate import load_lexicon_binding

    lexicons = tmp_path / "lex"
    compile_mod.compile_lexicons(_registry(tmp_path), lexicons, verify="skip")

    binding = load_lexicon_binding(lexicons)

    assert {entry[0] for entry in binding.entries} == set(compile_mod.TIERS)


def test_index_is_written_in_loader_order_positives_first(tmp_path):
    manifests = compile_mod.compile_lexicons(_registry(tmp_path), tmp_path / "lex")
    assert manifests["core"].pattern_order == [
        "pos_patterns.pattern_0", "pos_patterns.pattern_1", "pos_patterns.pattern_2",
        "neg_patterns.pattern_0", "neg_patterns.pattern_1",
    ]
    assert [r["metacluster"] for r in _index_rows(tmp_path / "lex", "core")] == \
        ["pos", "pos", "pos", "neg", "neg"]


def _index_rows(lex_dir, tier):
    return json.loads((lex_dir / f"{tier}.manifest.json").read_text())["index"]


def test_manifest_declares_when_a_tier_contrast_varies_nothing(tmp_path):
    """The reference implementation's core and expanded had identical positive sets."""
    manifests = compile_mod.compile_lexicons(_registry(tmp_path), tmp_path / "lex")
    cmp = manifests["core"].comparisons["expanded"]
    assert cmp["positive_sets_identical"] is True
    assert "does not vary the positive lexicon" in cmp["warning"]


def _decisions(tmp_path, registry, **fields):
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    members = [nodes[0]["node_id"], nodes[1]["node_id"]]
    path = tmp_path / "d.json"
    decision = {"cluster_id": "c1", "decision": "collapse", "members": members,
                "representative": members[0], "rationale": "r", "decided_by": "test"}
    decision.update(fields)
    path.write_text(json.dumps(_adjudication_payload(decisions=[decision])))
    return path


def _adjudication_payload(*, decisions=(), tiers=None):
    """Identity-bearing Task 12 handoff, allowing one test to corrupt inner data."""
    tiers = dict(tiers or {})
    decisions = list(decisions)
    provenance = {"test_fixture": "tests/test_ingest_compile.py"}
    return {
        "schema_version": DECISION_BUNDLE_SCHEMA_VERSION,
        "artifact_id": decision_bundle_artifact_id(decisions, tiers, provenance),
        "producer": DECISION_BUNDLE_PRODUCER,
        "provenance": provenance,
        "decisions": decisions,
        "tiers": tiers,
    }


def _write_decisions(tmp_path, members, representative, cluster_id="c1",
                     decision="collapse", **fields):
    """A single decision, with caller-chosen members/representative (may be stale)."""
    payload = {"cluster_id": cluster_id, "decision": decision, "members": members,
               "representative": representative, "rationale": "r", "decided_by": "test"}
    payload.update(fields)
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_adjudication_payload(decisions=[payload])))
    return path


def _overlapping_decisions(tmp_path, registry):
    """Two collapse decisions that both claim the same node (V-08's compile-stage twin)."""
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    shared = nodes[1]["node_id"]
    path = tmp_path / "d.json"
    decisions = [
        {"cluster_id": "c1", "decision": "collapse",
         "members": [nodes[0]["node_id"], shared], "representative": nodes[0]["node_id"],
         "rationale": "r1", "decided_by": "test"},
        {"cluster_id": "c2", "decision": "collapse",
         "members": [shared, nodes[2]["node_id"]], "representative": shared,
         "rationale": "r2", "decided_by": "test"},
    ]
    path.write_text(json.dumps(_adjudication_payload(decisions=decisions)))
    return path


def _decision_whose_representative_is_expanded_only(tmp_path, registry):
    """A collapse whose representative is demoted out of `core` after the decision was made."""
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    rep = nodes[0]["node_id"]
    other = nodes[1]["node_id"]
    path = tmp_path / "d.json"
    decisions = [{
            "cluster_id": "c1", "decision": "collapse",
            "members": [rep, other], "representative": rep,
            "merge_confidence": "HIGH",
            "rationale": "representative demoted to expanded after the decision was made",
            "decided_by": "test",
        }]
    tier_overrides = {
        rep: {"analysis_tier": "expanded", "tier_reason": "demoted post hoc"}
    }
    path.write_text(json.dumps(
        _adjudication_payload(decisions=decisions, tiers=tier_overrides)
    ))
    return path


@pytest.mark.parametrize("fields,trigger", [
    ({"merge_confidence": "MODERATE"}, "merge_confidence_not_high"),
    ({"merge_confidence": "LOW"}, "merge_confidence_not_high"),
    ({}, "merge_confidence_not_high"),                    # undeclared: conservative
    ({"merge_confidence": "HIGH", "family_ambiguity": True}, "family_ambiguity"),
    ({"merge_confidence": "HIGH", "threshold_sensitive": True}, "threshold_sensitive"),
])
def test_each_named_trigger_leaves_the_merge_split_in_sensitivity(tmp_path, fields, trigger):
    """Membership is decided by named conditions, never by a number (X-02).

    There is no scalar merge confidence to threshold: the design lists
    "moderate-confidence merge" as a trigger without defining it, and the
    reference implementation produced the value by looking up a family name.
    """
    registry = _registry(tmp_path)
    manifests = compile_mod.compile_lexicons(
        registry, tmp_path / "lex", decisions_path=_decisions(tmp_path, registry, **fields))
    assert manifests["core"].n_motifs == 4          # collapsed
    assert manifests["sensitivity"].n_motifs == 5   # kept split
    assert manifests["core"].comparisons["sensitivity"]["positive_sets_identical"] is False
    assert trigger in manifests["sensitivity"].sensitivity_triggers["c1"]


def test_a_high_confidence_unambiguous_merge_is_applied_everywhere(tmp_path):
    registry = _registry(tmp_path)
    manifests = compile_mod.compile_lexicons(
        registry, tmp_path / "lex",
        decisions_path=_decisions(tmp_path, registry, merge_confidence="HIGH"))
    assert manifests["core"].n_motifs == manifests["sensitivity"].n_motifs == 4
    assert manifests["sensitivity"].sensitivity_triggers == {}


def test_merge_confidence_is_a_grade_and_cannot_be_thresholded():
    """The failure this replaces: a two-label lookup read as though it were a measure."""
    from motifmultiverse.schema import (
        CRITERION_NOT_YET_DEFINED,
        MERGE_CONFIDENCE_CRITERIA,
        DecisionRecord,
        MergeConfidence,
    )
    with pytest.raises(TypeError):
        _ = MergeConfidence.MODERATE < 0.8          # noqa: B015 - that is the assertion
    assert not hasattr(compile_mod, "MODERATE_MERGE_CONFIDENCE")
    # The criteria are recorded as undecided rather than filled with a number.
    assert set(MERGE_CONFIDENCE_CRITERIA.values()) == {CRITERION_NOT_YET_DEFINED}
    with pytest.raises(SchemaError, match="not a number"):
        DecisionRecord(cluster_id="c", decision="collapse", members=["a"],
                       rationale="r", decided_by="t", merge_confidence="0.8")


def test_a_representative_that_is_not_a_member_is_refused(tmp_path):
    registry = _registry(tmp_path)
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    decisions = tmp_path / "d.json"
    decision_rows = [{
        "cluster_id": "c1", "decision": "collapse",
        "members": [nodes[0]["node_id"], nodes[1]["node_id"]],
        "representative": "a_constructed_average", "confidence": 0.9,
        "rationale": "x", "decided_by": "test"}]
    decisions.write_text(json.dumps(_adjudication_payload(decisions=decision_rows)))
    with pytest.raises(compile_mod.CompileError, match="observed medoid"):
        compile_mod.compile_lexicons(registry, tmp_path / "lex", decisions_path=decisions)


def test_unknown_decision_member_is_refused(tmp_path):
    registry = _registry(tmp_path)
    decisions = _write_decisions(tmp_path, members=["missing-node"], representative="missing-node")
    with pytest.raises(compile_mod.CompileError, match="unknown decision member"):
        compile_mod.compile_lexicons(registry, tmp_path / "lex", decisions_path=decisions)


def test_node_cannot_belong_to_two_collapse_clusters(tmp_path):
    registry = _registry(tmp_path)
    decisions = _overlapping_decisions(tmp_path, registry)
    with pytest.raises(compile_mod.CompileError, match="multiple collapse clusters"):
        compile_mod.compile_lexicons(registry, tmp_path / "lex", decisions_path=decisions)


def test_representative_missing_from_tier_is_refused(tmp_path):
    registry = _registry(tmp_path)
    decisions = _decision_whose_representative_is_expanded_only(tmp_path, registry)
    with pytest.raises(compile_mod.CompileError, match="representative is absent from tier core"):
        compile_mod.compile_lexicons(registry, tmp_path / "lex", decisions_path=decisions)


def test_tier_override_naming_an_unknown_node_is_refused(tmp_path):
    """A `tiers` override is a decisions-payload member too: a stale one is refused,
    not a silent no-op (round-1 review finding 1a).
    """
    registry = _registry(tmp_path)
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_adjudication_payload(
        tiers={"missing-node": {"analysis_tier": "expanded"}}
    )))
    with pytest.raises(compile_mod.CompileError, match="unknown node"):
        compile_mod.compile_lexicons(registry, tmp_path / "lex", decisions_path=path)


def test_tier_override_with_an_invalid_analysis_tier_value_is_refused(tmp_path):
    """A typo'd tier value (round-1 review finding 1b) must not silently drop a node
    out of every tier at once. Before the fix this dropped `nodes[0]` from core,
    expanded AND sensitivity with `n_motifs` going 5 -> 4 and no error raised.
    """
    registry = _registry(tmp_path)
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    node_id = nodes[0]["node_id"]
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_adjudication_payload(
        tiers={node_id: {"analysis_tier": "coree"}}
    )))
    with pytest.raises(compile_mod.CompileError, match="not a valid Tier"):
        compile_mod.compile_lexicons(registry, tmp_path / "lex", decisions_path=path)


def test_tier_override_diverging_tiers_without_a_reason_is_refused(tmp_path):
    """The divergence invariant must hold after a tiers override too, not just at
    ingest time.

    `MotifNode.__post_init__` refuses `discovery_tier != analysis_tier` unless
    `tier_reason` is given explicitly. Ingest always writes
    `discovery_tier == analysis_tier == CORE`, so that invariant is trivially
    satisfied there; the only place a divergence can be *introduced* is a
    `tiers` override applied by `_apply_tiers` at compile time -- and
    `_apply_tiers` mutates a plain dict, never re-running the check that would
    have rejected this exact state had it been constructed as a `MotifNode`
    from scratch. An override that sets only `analysis_tier` (no
    `tier_reason`) must be refused, not silently applied.
    """
    registry = _registry(tmp_path)
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    node_id = nodes[0]["node_id"]
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_adjudication_payload(
        tiers={node_id: {"analysis_tier": "expanded"}}
    )))
    with pytest.raises(compile_mod.CompileError, match="tier_reason"):
        compile_mod.compile_lexicons(registry, tmp_path / "lex", decisions_path=path)


def test_tier_override_diverging_tiers_with_a_reason_is_applied(tmp_path):
    """The companion case: the same divergence WITH an explicit tier_reason is a
    legitimate, licensed override and must go through unchanged.
    """
    registry = _registry(tmp_path)
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    node_id = nodes[0]["node_id"]
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_adjudication_payload(tiers={
        node_id: {"analysis_tier": "expanded", "tier_reason": "demoted post hoc"}
    })))
    manifests = compile_mod.compile_lexicons(registry, tmp_path / "lex", decisions_path=path)
    assert manifests["core"].n_motifs == 4          # node_id demoted out of core
    assert manifests["expanded"].n_motifs == 5


def test_a_mistyped_decision_value_is_refused(tmp_path):
    """A mistyped `decision` (round-1 review finding 2) must not silently become a
    no-op collapse: before the fix this decision was compared against
    `Decision.COLLAPSE.value` byte-for-byte, matched nothing, and both members
    survived uncollapsed with no error raised.
    """
    registry = _registry(tmp_path)
    decisions = _decisions(tmp_path, registry, decision="COLLAPSE")   # wrong case
    with pytest.raises(compile_mod.CompileError, match="is not one of"):
        compile_mod.compile_lexicons(registry, tmp_path / "lex", decisions_path=decisions)


def test_a_refused_compile_still_writes_provenance(tmp_path):
    """T-09: every subcommand writes provenance, including the ones that refuse.

    A decisions payload rejected for referencing a stale node must not also
    lose the record of what was attempted.
    """
    registry = _registry(tmp_path)
    decisions = _write_decisions(tmp_path, members=["missing-node"], representative="missing-node")
    out = tmp_path / "lex"
    with pytest.raises(compile_mod.CompileError, match="unknown decision member"):
        compile_mod.compile_lexicons(registry, out, decisions_path=decisions)
    assert (out / "provenance.json").exists()


def test_the_content_hash_is_deterministic_and_tracks_membership(tmp_path):
    registry = _registry(tmp_path)
    a = compile_mod.compile_lexicons(registry, tmp_path / "lex1")
    b = compile_mod.compile_lexicons(registry, tmp_path / "lex2")
    assert a["core"].lexicon_content_hash == b["core"].lexicon_content_hash

    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    decisions = tmp_path / "d.json"
    decision_rows = [{
        "cluster_id": "c1", "decision": "collapse",
        "members": [nodes[0]["node_id"], nodes[1]["node_id"]],
        "representative": nodes[0]["node_id"], "merge_confidence": "MODERATE",
        "rationale": "moderate", "decided_by": "test"}]
    decisions.write_text(json.dumps(_adjudication_payload(decisions=decision_rows)))
    c = compile_mod.compile_lexicons(registry, tmp_path / "lex3", decisions_path=decisions)
    assert c["core"].lexicon_content_hash != a["core"].lexicon_content_hash
    assert c["sensitivity"].lexicon_content_hash == a["core"].lexicon_content_hash


def test_manifest_tsv_binds_each_row_to_the_lexicon_hash_of_its_OWN_tier(tmp_path):
    """One table, three tiers, one hash column -- and nothing checked which hash.

    `manifest.tsv` is the join table: a hit table records a
    `lexicon_content_hash`, and this is where a reader resolves it to the motif
    rows that hash stands for. Every existing test reads the per-tier
    `*.manifest.json`, so writing the loop variable's hash into every row --
    binding all three tiers to whichever tier the loop ended on -- left the suite
    green.

    The three tiers exist so a conclusion can be tested against a wider or
    narrower lexicon, which means their hashes differ exactly when the tiers do.
    A reader joining a core hit table's hash to this table then finds no rows at
    all, while `sensitivity`'s hash pulls out `core`'s motifs under
    `sensitivity`'s identity -- the citation resolving to the wrong lexicon, in
    the one file whose job is to make that citation resolvable.

    The MODERATE-confidence collapse is the fixture
    `test_the_content_hash_is_deterministic_and_tracks_membership` already uses to
    force `core` and `sensitivity` apart; without it all three tiers would share a
    hash and this test would pass on the defect.
    """
    registry = _registry(tmp_path)
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    decisions = tmp_path / "d.json"
    decisions.write_text(json.dumps(_adjudication_payload(decisions=[{
        "cluster_id": "c1", "decision": "collapse",
        "members": [nodes[0]["node_id"], nodes[1]["node_id"]],
        "representative": nodes[0]["node_id"], "merge_confidence": "MODERATE",
        "rationale": "moderate", "decided_by": "test"}])))

    out = tmp_path / "lex"
    manifests = compile_mod.compile_lexicons(registry, out, decisions_path=decisions)

    distinct = {m.lexicon_content_hash for m in manifests.values()}
    assert len(distinct) > 1, "the fixture must make at least two tiers differ"

    header, *body = [line.split("\t")
                     for line in (out / "manifest.tsv").read_text().splitlines() if line]
    tier_at, hash_at = header.index("tier"), header.index("lexicon_content_hash")
    for row in body:
        tier = row[tier_at]
        assert row[hash_at] == manifests[tier].lexicon_content_hash, (
            f"a {tier} row cites the hash of another tier's lexicon"
        )


def test_trim_threshold_changes_lexicon_identity(tmp_path):
    registry = _registry(tmp_path)
    a = compile_mod.compile_lexicons(registry, tmp_path / "a", trim_threshold=0.2)
    b = compile_mod.compile_lexicons(registry, tmp_path / "b", trim_threshold=0.3)
    assert a["core"].lexicon_content_hash != b["core"].lexicon_content_hash


def test_loader_parameters_propagate_to_roundtrip_verification(tmp_path, monkeypatch):
    """`verify_roundtrip` must read back with the manifest's own loader configuration.

    A `load_back(h5_path)` call that ignores its arguments would silently verify
    against `trim_threshold=0.3, motif_type="cwm", include_rc=False` no matter what
    the lexicon was actually compiled with -- the same non-semantic-identity bug
    this task fixes, just at read time instead of hash time.
    """
    registry = _registry(tmp_path)
    manifests = compile_mod.compile_lexicons(
        registry, tmp_path / "lex", trim_threshold=0.21, motif_type="pfm",
        include_rc=True, verify="skip",
    )
    manifest = manifests["core"]

    captured: dict[str, object] = {}

    def fake_load_back(h5_path, **kwargs):
        captured.update(kwargs)
        return list(manifest.pattern_order)

    monkeypatch.setattr(compile_mod, "load_back", fake_load_back)
    result = compile_mod.verify_roundtrip(tmp_path / "lex" / "core.h5", manifest)
    assert result.passed
    assert captured["trim_threshold"] == 0.21
    assert captured["motif_type"] == "pfm"
    assert captured["include_rc"] is True


def test_loader_parameters_hash_by_effective_value_not_spelling(tmp_path):
    """The hash must track what the loader actually does, not how the caller spelled it.

    `loader_parameters=None` and `loader_parameters={}` both resolve to the same
    effective configuration at read time -- `load_back` falls back to
    `motif_lambda_default=0.7` in both cases -- so they must content-address
    identically. A spelling that changes the *effective* value
    (`motif_lambda_default=0.5` instead of the 0.7 default) must still change the
    hash: this is not a return to "the hash ignores loader_parameters", it is "the
    hash tracks the resolved value, not the unresolved spelling" (round-1 review
    finding 1).
    """
    registry = _registry(tmp_path)
    none_spelling = compile_mod.compile_lexicons(
        registry, tmp_path / "none", loader_parameters=None)
    empty_spelling = compile_mod.compile_lexicons(
        registry, tmp_path / "empty", loader_parameters={})
    assert (none_spelling["core"].lexicon_content_hash
            == empty_spelling["core"].lexicon_content_hash)
    assert none_spelling["core"].loader_parameters == empty_spelling["core"].loader_parameters

    different = compile_mod.compile_lexicons(
        registry, tmp_path / "different",
        loader_parameters={"motif_lambda_default": 0.5})
    assert (different["core"].lexicon_content_hash
            != none_spelling["core"].lexicon_content_hash)


def test_each_tier_manifest_owns_its_own_loader_parameters_dict(tmp_path):
    """The three tier manifests must not share one `loader_parameters` dict object.

    `compile_lexicons` resolves `loader_parameters` once and currently hands the
    *same* dict to every tier's `LexiconManifest`. Nothing in this package mutates
    it in place today, but a future caller that does (e.g. `manifest.loader_parameters
    ["x"] = 1`) would silently corrupt all three tiers' manifests at once. Each
    tier must own an independent copy.
    """
    registry = _registry(tmp_path)
    manifests = compile_mod.compile_lexicons(registry, tmp_path / "lex")
    assert manifests["core"].loader_parameters is not manifests["expanded"].loader_parameters
    assert manifests["core"].loader_parameters is not manifests["sensitivity"].loader_parameters
    manifests["core"].loader_parameters["motif_lambda_default"] = 999.0
    assert manifests["expanded"].loader_parameters["motif_lambda_default"] == 0.7
    assert manifests["sensitivity"].loader_parameters["motif_lambda_default"] == 0.7


def test_a_lexicon_of_mixed_motif_lengths_is_refused(tmp_path):
    """The loader stacks every motif into one array, so lengths must agree."""
    registry = _registry(tmp_path)
    with h5py.File(registry / "arrays.h5", "a") as h5:
        node_id = sorted(h5.keys())[0]
        del h5[node_id]["cwm"]
        h5[node_id].create_dataset("cwm", data=np.zeros((MOTIF_LEN + 3, 4)))
    with pytest.raises(compile_mod.CompileError, match="mixes motif lengths"):
        compile_mod.compile_lexicons(registry, tmp_path / "lex")


def test_compile_refuses_a_motif_type_whose_loader_dataset_is_missing(tmp_path):
    registry = _registry(tmp_path)
    with h5py.File(registry / "arrays.h5", "a") as h5:
        node_id = sorted(h5.keys())[0]
        del h5[node_id]["hypothetical_cwm"]

    with pytest.raises(
        compile_mod.CompileError,
        match="motif_type.*hcwm|hypothetical",
    ):
        compile_mod.compile_lexicons(
            registry,
            tmp_path / "lex",
            motif_type="hcwm",
            verify="skip",
        )


# ------------------------------------------------- the real loader, or a skip
def test_roundtrip_against_the_real_loader(tmp_path):
    """Behavioural, not structural: the hit caller reads it, in this order."""
    require_finemo_backend()
    manifests = compile_mod.compile_lexicons(_registry(tmp_path), tmp_path / "lex",
                                             verify="require")
    names = compile_mod.load_back(tmp_path / "lex" / "core.h5")
    assert names == manifests["core"].pattern_order
    assert guards.index_order_matches_loader(manifests["core"].pattern_order, names).passed


def test_verify_require_fails_loudly_when_the_backend_is_absent(tmp_path, no_finemo_backend):
    """The absent-backend refusal, verified on a machine that HAS the backend.

    This used to skip wherever finemo was installed, which made it the exact
    mirror of the skip it was written to compensate for: the round trip was
    unverified without the backend, the refusal was unverified with it, and no
    environment ran both. `no_finemo_backend` blanks the import instead, so the
    two halves are no longer in competition for the same machine.
    """
    with pytest.raises(compile_mod.BackendMissing, match="finemo"):
        compile_mod.compile_lexicons(_registry(tmp_path), tmp_path / "lex", verify="require")


# --- regression: the loader's argument names are not ours to assume -----------
# `load_back` passed the loader's settings by keyword inside a `try` that caught
# only ImportError. finemo 0.40 renamed `trim_threshold` to
# `trim_threshold_default` and added `trim_coords` / `trim_thresholds`, so on any
# machine with a current backend every call raised
#   TypeError: load_modisco_motifs() got an unexpected keyword argument 'trim_threshold'
# from inside `compile_lexicons`. Because `--verify-roundtrip auto` is the default
# and only catches BackendMissing, that aborted the compile outright: 31 tests in
# this suite failed, and no lexicon was written at all. The skip hid it -- nothing
# in CI had the backend, so nothing ever made the call.
#
# These signatures are copied from the two real releases. `_loader_call_kwargs`
# must bind to whichever one is installed, and refuse -- loudly, by name -- rather
# than guess when it recognises neither.
def _loader_signature_0_30(modisco_h5_path, trim_threshold, motif_type, motifs_include,
                           motif_name_map, motif_lambdas, motif_lambda_default,
                           include_rc):        # pragma: no cover - inspected, not called
    ...


def _loader_signature_0_41(modisco_h5_path, trim_coords, trim_thresholds,
                           trim_threshold_default, motif_type, motifs_include,
                           motif_name_map, motif_lambdas, motif_lambda_default,
                           include_rc):        # pragma: no cover - inspected, not called
    ...


@pytest.mark.parametrize("loader,threshold_param", [
    (_loader_signature_0_30, "trim_threshold"),
    (_loader_signature_0_41, "trim_threshold_default"),
])
def test_loader_settings_bind_to_whichever_backend_release_is_installed(loader,
                                                                       threshold_param):
    kwargs = compile_mod._loader_call_kwargs(
        loader, trim_threshold=0.3, motif_type="cwm", include_rc=False,
        extra={"motif_lambda_default": 0.7})
    assert kwargs[threshold_param] == 0.3
    assert kwargs["motif_type"] == "cwm" and kwargs["include_rc"] is False
    assert kwargs["motif_lambda_default"] == 0.7
    # The call must be complete: nothing without a default may be left to the
    # backend, because those are the arguments that change what comes back.
    import inspect as _inspect
    required = [name for name, p in _inspect.signature(loader).parameters.items()
                if p.default is _inspect.Parameter.empty][1:]
    assert sorted(kwargs) == sorted(required)


def test_a_backend_whose_trim_threshold_is_renamed_again_is_refused_not_guessed():
    """A third spelling must stop the round trip, not be silently dropped.

    Dropping it would let the backend apply its own trimming default while the
    manifest still recorded ours -- a round trip that passes against a lexicon
    nobody compiled.
    """
    def renamed(modisco_h5_path, trimming_cutoff, motif_type, motifs_include,
                motif_name_map, motif_lambdas, motif_lambda_default, include_rc):
        ...  # pragma: no cover - inspected, not called

    with pytest.raises(compile_mod.BackendIncompatible, match="trim"):
        compile_mod._loader_call_kwargs(renamed, trim_threshold=0.3, motif_type="cwm",
                                        include_rc=False, extra={})


def test_a_backend_requiring_an_argument_this_package_does_not_know_is_refused():
    def with_a_new_required_knob(modisco_h5_path, trim_threshold_default, motif_type,
                                 motifs_include, motif_name_map, motif_lambdas,
                                 motif_lambda_default, include_rc, score_transform):
        ...  # pragma: no cover - inspected, not called

    with pytest.raises(compile_mod.BackendIncompatible, match="score_transform"):
        compile_mod._loader_call_kwargs(with_a_new_required_knob, trim_threshold=0.3,
                                        motif_type="cwm", include_rc=False, extra={})


def test_a_manifest_setting_the_backend_no_longer_accepts_is_refused():
    """`loader_parameters` is manifest content; a backend that dropped one of them
    cannot reproduce the lexicon's declared configuration."""
    def without_lambda(modisco_h5_path, trim_threshold_default, motif_type,
                       motifs_include, motif_name_map, motif_lambdas, include_rc):
        ...  # pragma: no cover - inspected, not called

    with pytest.raises(compile_mod.BackendIncompatible, match="motif_lambda_default"):
        compile_mod._loader_call_kwargs(without_lambda, trim_threshold=0.3,
                                        motif_type="cwm", include_rc=False,
                                        extra={"motif_lambda_default": 0.7})


def _install_fake_backend(monkeypatch, loader):
    """Put a `finemo.data_io.load_modisco_motifs` on the import path for one test."""
    import types
    package = types.ModuleType("finemo")
    data_io = types.ModuleType("finemo.data_io")
    data_io.load_modisco_motifs = loader
    package.data_io = data_io
    monkeypatch.setitem(sys.modules, "finemo", package)
    monkeypatch.setitem(sys.modules, "finemo.data_io", data_io)


def test_an_uncallable_backend_leaves_auto_writing_and_require_refusing(tmp_path,
                                                                       monkeypatch):
    """An installed-but-uncallable backend must behave like an absent one.

    Not like a crash: `auto` promises to verify if it can and carry on if it
    cannot, and a TypeError escaping from the middle of the compile broke that
    promise in the worst direction -- the lexicon was not written at all, for a
    reason whose message never mentioned the backend.
    """
    def wrong_signature(modisco_h5_path, trimming_cutoff, motif_type, motifs_include,
                        motif_name_map, motif_lambdas, motif_lambda_default, include_rc):
        ...  # pragma: no cover - never reached; the binding refuses first

    _install_fake_backend(monkeypatch, wrong_signature)
    registry = _registry(tmp_path)

    compile_mod.compile_lexicons(registry, tmp_path / "auto", verify="auto")
    assert (tmp_path / "auto" / "core.h5").exists()

    with pytest.raises(compile_mod.BackendMissing, match="trim"):
        compile_mod.compile_lexicons(registry, tmp_path / "require", verify="require")


def test_the_declared_finemo_extra_names_the_distribution_that_provides_the_backend():
    """`pip install -e ".[finemo]"` has to actually install the backend.

    The extra named `finemo-gpu`, which is the GitHub project's name and is not a
    distribution on PyPI at all -- `pip install finemo-gpu` fails with "No
    matching distribution found". So the documented way to make the round trip
    runnable could not be followed, and the test that needed it skipped
    everywhere, including CI. Read from pyproject rather than trusted, because
    the failure is invisible to anyone who already has the backend.
    """
    import tomllib
    from importlib.metadata import PackageNotFoundError, distribution
    from pathlib import Path

    # Only the machines that HAVE the backend can say which distribution supplied
    # it, so this rides the same gate as the round trip: skipped where the backend
    # is absent, failed where the run declared it must be present.
    require_finemo_backend()
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("source tree not present in this installation")
    declared = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]
    requirements = declared["finemo"]
    assert requirements, "the finemo extra installs nothing"
    names = [re.split(r"[<>=!~\[; ]", req, maxsplit=1)[0] for req in requirements]
    for name in names:
        try:
            distribution(name)
        except PackageNotFoundError:
            pytest.fail(
                f"the finemo extra requires the distribution {name!r}, which is not "
                f"installed here even though the backend is: declared {requirements}"
            )


def test_auto_verification_still_writes_but_claims_nothing(tmp_path, no_finemo_backend):
    """Without a backend the lexicon is written and simply not verified.

    `auto` must neither fail nor pretend: the file exists, and nothing in the
    manifest asserts a round trip that did not happen. The backend is blanked
    rather than assumed absent, so this states the no-backend behaviour on every
    machine instead of only on the ones that happen to lack finemo -- where it
    was passing for the wrong reason once the backend became installable.
    """
    manifests = compile_mod.compile_lexicons(_registry(tmp_path), tmp_path / "lex",
                                             verify="auto")
    assert (tmp_path / "lex" / "core.h5").exists()
    assert "verified" not in json.dumps(manifests["core"].pattern_order)


# ------------------------------------------------------ representation identity
# `RepresentationId` / `VariantId` (Task 7, `schema/identity.py`) give the same
# five fields `MotifNode` already carries (model/readout/context/metacluster/
# pattern id) a structured, hashable identity, so a later task can key a table on
# a node's *representation* rather than parsing one out of `node_id`. Nothing in
# `ingest`/`compile` constructs them yet -- that wiring belongs to a later task --
# but their shape is load-bearing now, so it is exercised here.
def test_representation_id_is_frozen_and_hashable():
    rep = RepresentationId(model="modelA", readout="r1", context="promoter",
                           metacluster="pos_patterns", local_pattern_id="pattern_0")
    assert rep.schema_version == IDENTITY_SCHEMA_VERSION
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.model = "modelB"                          # type: ignore[misc]
    same = RepresentationId(model="modelA", readout="r1", context="promoter",
                            metacluster="pos_patterns", local_pattern_id="pattern_0")
    different = dataclasses.replace(same, local_pattern_id="pattern_1")
    assert rep == same and hash(rep) == hash(same)
    assert rep != different
    assert {rep, same, different} == {rep, different}  # usable as a set/dict key


def test_variant_id_is_frozen_and_hashable():
    vid = VariantId(family_id="FAMA", namespace="union_id", value="MA_FAMA_01")
    assert vid.schema_version == IDENTITY_SCHEMA_VERSION
    with pytest.raises(dataclasses.FrozenInstanceError):
        vid.value = "MA_FAMA_02"                        # type: ignore[misc]
    same = VariantId(family_id="FAMA", namespace="union_id", value="MA_FAMA_01")
    different = VariantId(family_id="FAMA", namespace="union_id", value="MA_FAMA_02")
    assert vid == same and hash(vid) == hash(same)
    assert vid != different


def test_identity_schema_version_is_declared():
    """T-07's default version is defined once and carried by each identity."""
    assert isinstance(IDENTITY_SCHEMA_VERSION, str) and IDENTITY_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Task 12: compile accepts only the validated adjudicate handoff.
# ---------------------------------------------------------------------------

def test_compile_refuses_a_legacy_or_handwritten_decision_payload(tmp_path):
    """Replacing the strict handoff parser with permissive from_dict fails this."""
    registry = _registry(tmp_path)
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"decisions": [{
        "cluster_id": "legacy",
        "decision": "collapse",
        "members": [nodes[0]["node_id"], nodes[1]["node_id"]],
        "representative": nodes[0]["node_id"],
        "rationale": "handwritten legacy row",
        "decided_by": "test",
    }]}))

    with pytest.raises(compile_mod.CompileError, match="artifact"):
        compile_mod.compile_lexicons(
            registry,
            tmp_path / "lex",
            decisions_path=legacy,
            verify="skip",
        )


def test_compile_consumes_the_schema_validated_merge_decisions_emitted_by_adjudicate(tmp_path):
    """Rejecting the producer's own identity-bearing bundle fails this companion."""
    from motifmultiverse.adjudicate import (
        OntologyDecision,
        stable_decision_id,
        write_adjudication_artifacts,
    )

    registry = _registry(tmp_path)
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    members = tuple(sorted((nodes[0]["node_id"], nodes[1]["node_id"])))
    decision = OntologyDecision(
        decision_id=stable_decision_id(
            members, "TRUE_DUPLICATE", "TRUE_DUPLICATE", "test-1"
        ),
        node_ids=members,
        relationship="TRUE_DUPLICATE",
        decision=Decision.COLLAPSE,
        family_id=None,
        representative_node_id=members[0],
        criterion_id="TRUE_DUPLICATE",
        criterion_version="test-1",
        evidence_ids=("alignment:test",),
        evidence_for=("all gates passed",),
        evidence_against=(),
        rationale="all declared evidence gates passed",
        decided_by="automated:test",
        manual_override=False,
        provenance={"criteria_sha256": "a" * 64},
    )
    adjudication = tmp_path / "adjudication"
    write_adjudication_artifacts(
        adjudication,
        [decision],
        provenance={"criteria_sha256": "a" * 64},
    )

    manifests = compile_mod.compile_lexicons(
        registry,
        tmp_path / "lex",
        decisions_path=adjudication / "merge_decisions.json",
        verify="skip",
    )

    assert manifests["core"].n_motifs == 4


# --- regression: several discovery runs, all named modisco.h5 ------------------
# The real layout is <cluster>/modisco.h5 -- the filename is a TF-MoDISco
# convention, not a choice. Keying provenance by basename recorded ONE checksum
# for the whole project and attributed it to whichever file was read last, so a
# record that looked complete described a different run than the one that ran.
def test_ingest_records_one_checksum_per_analysis_even_when_the_files_share_a_name(tmp_path):
    analyses = []
    for cluster in ("promoter_cl5", "distal_cl8"):
        d = tmp_path / cluster
        d.mkdir()
        # distinct pattern counts -> distinct bytes -> distinct checksums
        n_pos = 3 if cluster == "promoter_cl5" else 2
        analyses.append({
            "id": f"cbp_{cluster}", "model": "cbp", "readout": "r1", "union_id": "CBP",
            "context": "promoter",
            "modisco_h5": str(_modisco(d / "modisco.h5", n_pos=n_pos)),
        })
    out = tmp_path / "registry"
    ingest.ingest_project(_project(tmp_path, analyses=analyses), out)

    prov = json.loads((out / "provenance.json").read_text())[0]
    h5_keys = {k for k in prov["inputs"] if k.endswith("modisco.h5")}
    assert h5_keys == {"cbp_promoter_cl5:modisco.h5", "cbp_distal_cl8:modisco.h5"}, (
        f"one checksum per discovery run expected, got {sorted(prov['inputs'])}"
    )
    assert len({prov["inputs"][k] for k in h5_keys}) == 2, "two files, two checksums"
    assert not any(k.startswith("/") for k in prov["inputs"]), "keys must stay relative"


# --- regression: a lexicon's identity must cover what downstream binds to ------
# The hit caller hands back `pattern_tag`; the manifest index is the only table
# that turns a tag into the `variant_id` every family-level number is grouped by
# (interpret/README.md step 3), and `schema` calls variant_id the only stable
# semantic identity. A content hash blind to that column is a citation that
# cannot name which lexicon a number came from -- the thing FP-11 asks for.
def test_lexicon_identity_covers_the_variant_id_downstream_binds_to(tmp_path):
    modisco = _modisco(tmp_path / "shared.h5")
    seen: dict[str, dict[str, object]] = {}
    for union_id in ("MA", "MB"):
        home = tmp_path / union_id
        home.mkdir()
        project = home / "project.json"
        project.write_text(json.dumps({
            "project": "test-project", "peak_universe_id": "u1",
            "analyses": [{"id": "modelA_r1", "model": "modelA", "readout": "r1",
                          "union_id": union_id, "context": "promoter",
                          "modisco_h5": str(modisco)}]}))
        ingest.ingest_project(project, home / "registry")
        manifests = compile_mod.compile_lexicons(
            home / "registry", home / "lex", tiers=("core",), verify="skip")
        index = json.loads((home / "lex" / "core.manifest.json").read_text())["index"]
        seen[union_id] = {
            "hash": manifests["core"].lexicon_content_hash,
            "node_ids": [row["node_id"] for row in index],
            "variant_ids": [row["variant_id"] for row in index],
            "pattern_order": manifests["core"].pattern_order,
        }

    # Same discovery file, same nodes, same tags, same arrays: variant_id is the
    # only thing that differs, which is exactly the case the hash used to miss.
    assert seen["MA"]["node_ids"] == seen["MB"]["node_ids"]
    assert seen["MA"]["pattern_order"] == seen["MB"]["pattern_order"]
    assert seen["MA"]["variant_ids"] != seen["MB"]["variant_ids"]
    assert seen["MA"]["hash"] != seen["MB"]["hash"]


@pytest.mark.parametrize("field, other_value", [
    ("trim_threshold", 0.5),
    ("motif_type", "pfm"),
    ("include_rc", True),
    ("loader_backend", "other-caller"),
    ("loader_parameters", {"motif_lambda_default": 0.9}),
])
def test_every_loader_setting_the_identity_names_actually_changes_the_identity(field, other_value):
    """The docstring names the settings that must not collide; nothing checked them.

    `lexicon_semantic_hash` states the rule it exists to enforce: two lexicons
    built from byte-identical motif arrays but compiled to be read back under
    different loader settings -- a different `trim_threshold`, `motif_type`,
    `include_rc` or `loader_parameters` -- load differently and must not collide
    on identity. The variant_id half of that promise has the test above. The
    loader-configuration half had none: dropping any one key from the hashed
    metadata blob left the whole suite green.

    `include_rc` is the sharpest. Compile one registry twice, reverse-complement
    matching off and on: the real loader calls hits on the minus strand in one and
    not the other, so they are different instruments. With the key absent from the
    blob both manifests carry the same `lexicon_content_hash`, so
    `compute_substrate_id` gives both frozen runs the same substrate identity, a
    hit table citing that hash under FP-11 no longer names which lexicon produced
    it, and `_load_and_verify` still passes -- it recomputes through this same
    function using `manifest.include_rc`, so the artifact verifies itself as
    intact while its identity has stopped distinguishing what it exists to
    distinguish.

    Parametrised over every setting the docstring names, so a key dropped from the
    blob fails on the row that names it rather than on none of them.
    """
    ordered = [("pos_patterns", "pattern_0",
                {"node_id": "node-0", "variant_id": "UA_FAM_A_0"})]
    arrays = {"node-0": {"cwm": np.zeros((MOTIF_LEN, 4))}}
    settings = dict(schema_version="1.0", trim_threshold=0.3, motif_type="cwm",
                    include_rc=False, loader_backend="finemo",
                    loader_parameters={"motif_lambda_default": 0.7})

    baseline = compile_mod.lexicon_semantic_hash(ordered, arrays, **settings)
    varied = compile_mod.lexicon_semantic_hash(
        ordered, arrays, **{**settings, field: other_value})

    assert baseline != varied, (
        f"{field} never reaches lexicon_content_hash: two lexicons the loader reads "
        f"differently share one identity"
    )


def test_lexicon_identity_refuses_a_pattern_that_names_no_variant_id():
    """Silently hashing the tag alone is how two variant assignments collided."""
    with pytest.raises(compile_mod.CompileError, match="variant_id"):
        compile_mod.lexicon_semantic_hash(
            [("pos_patterns", "pattern_0", {"node_id": "node-0"})],
            {"node-0": {"cwm": np.zeros((MOTIF_LEN, 4))}},
            schema_version="1.0", trim_threshold=0.3, motif_type="cwm",
            include_rc=False, loader_backend="finemo",
            loader_parameters={"motif_lambda_default": 0.7},
        )


# --- regression: motif_type is the loader's vocabulary, not ours ---------------
# `motif_type` is handed verbatim to the backend named by `loader_backend`, which
# dispatches on cwm / hcwm / pfm / pfm_softmax. How it rejects anything else has
# changed across releases -- finemo 0.30 had no else-branch, so an unknown value
# left its motif locals unbound and raised `UnboundLocalError: motif_norm` from
# inside the backend; 0.41 raises `ValueError: Invalid motif_type` -- which is
# the reason `compile` refuses the value itself rather than relying on how the
# backend happens to complain this year. Verified against both loaders on a
# compiled lexicon: cwm -> 5 motifs; pfm -> 5 motifs; ppm -> refused.
def test_compile_refuses_a_motif_type_the_declared_loader_cannot_dispatch(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(compile_mod.CompileError, match="motif_type"):
        compile_mod.compile_lexicons(
            registry, tmp_path / "unreadable", motif_type="ppm", verify="skip")
    assert not (tmp_path / "unreadable" / "core.h5").exists()

    manifests = compile_mod.compile_lexicons(
        registry, tmp_path / "readable", motif_type="pfm", verify="skip")
    assert manifests["core"].motif_type == "pfm"
    with h5py.File(tmp_path / "readable" / "core.h5", "r") as h5:
        assert "sequence" in h5["pos_patterns"]["pattern_0"]


# --- the operations log, and the guard that reads it --------------------------
# `guards.no_cross_model_cwm_avg` had no call site because nothing recorded what
# any stage did to a CWM. The obvious repair -- have the collapse write down
# "medoid" beside itself -- would have produced a log that cannot contain a
# violation, which is the vacuity the pending registry named. `operations_log`
# instead opens the lexicon that was just written and classifies every motif in
# it against the registry arrays it stands for, so what the guard reads is a
# property of the bytes. These tests fix both halves: that it says "selected" on
# a real compile, and that it says "mean" -- and the guard then refuses -- when
# the bytes are an average, without the classifier being told.
def _two_model_registry(tmp_path):
    """Two analyses, two models, and deliberately different motif content.

    `_modisco` seeds its patterns by position, so two files built with it hold
    bit-identical matrices -- and the mean of two identical matrices IS one of
    them, the one case `operations_log` documents that it cannot see. Seeding the
    second file apart makes an average of one motif from each model a matrix that
    is neither of its inputs.
    """
    first, second = tmp_path / "model_a.h5", tmp_path / "model_b.h5"
    for path, offset in ((first, 0), (second, 200)):
        with h5py.File(path, "w") as h5:
            for i in range(2):
                _pattern(h5, "pos_patterns", f"pattern_{i}", seed=offset + i)
    project = _project(tmp_path, analyses=[
        {"id": "modelA_r1", "model": "modelA", "readout": "r1", "union_id": "MA",
         "context": "promoter", "modisco_h5": str(first)},
        {"id": "modelB_r1", "model": "modelB", "readout": "r1", "union_id": "MB",
         "context": "promoter", "modisco_h5": str(second)},
    ])
    registry = tmp_path / "two_model_registry"
    ingest.ingest_project(project, registry)
    return registry


def _average_on_write(monkeypatch, representative, absorbed):
    """Make the writer construct `representative` as the mean of it and `absorbed`.

    This is the future edit the guard exists to catch, applied without touching
    the classifier or the guard: `_write_h5` is the one place a motif's matrices
    reach the file, and everything downstream of it -- the log, the axes, the
    refusal -- has to come out of the bytes on its own.
    """
    real = compile_mod._write_h5

    def averaging(path, ordered, arrays):
        payload = {
            node["node_id"]: {name: np.asarray(arrays[node["node_id"]][name][:], dtype=float)
                              for name in arrays[node["node_id"]]}
            for _, _, node in ordered
        }
        if representative in payload:
            other = {name: np.asarray(arrays[absorbed][name][:], dtype=float)
                     for name in arrays[absorbed]}
            payload[representative] = {
                name: (values + other[name]) / 2.0
                for name, values in payload[representative].items() if name in other
            }
        real(path, ordered, payload)

    monkeypatch.setattr(compile_mod, "_write_h5", averaging)


def _operations(lex_dir):
    return json.loads((lex_dir / "combination_operations.json").read_text())


def test_the_operations_log_reports_a_collapse_as_a_selection_read_off_the_bytes(tmp_path):
    """A real collapse, and what the log says about it.

    The representative's emitted matrices are the registry's own, so the entry is
    `select_representative` and names both members as its inputs. Nothing here
    asked the collapse what it did.
    """
    registry = _registry(tmp_path)
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    representative, absorbed = nodes[0]["node_id"], nodes[1]["node_id"]
    decisions = _write_decisions(
        tmp_path, [representative, absorbed], representative, merge_confidence="HIGH")

    lex = tmp_path / "lex"
    compile_mod.compile_lexicons(registry, lex, decisions_path=decisions, verify="skip")

    operations = _operations(lex)
    assert {entry["op"] for entry in operations} == {"copy", "select_representative"}
    collapsed = [entry for entry in operations
                 if entry["op"] == "select_representative" and entry["tier"] == "core"]
    assert len(collapsed) == 1
    assert collapsed[0]["inputs"] == [representative, absorbed]
    assert collapsed[0]["group_by"] == list(guards.CROSS_MODEL_AXES)
    assert guards.no_cross_model_cwm_avg(operations).passed


def test_a_representative_averaged_across_two_models_is_refused_before_publication(
        tmp_path, monkeypatch):
    """The falsification the pending entry said this guard could not have.

    Nothing in the classifier or the guard is told that an average happened: the
    writer produces one, and the emitted matrices stop matching either input's
    registry arrays and start matching their mean. `model` is then not among the
    axes the operation held fixed, and the compile is refused with nothing
    published.
    """
    registry = _two_model_registry(tmp_path)
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    by_analysis = {node["model"]: node["node_id"] for node in reversed(nodes)}
    representative, absorbed = by_analysis["modelA"], by_analysis["modelB"]
    decisions = _write_decisions(
        tmp_path, [representative, absorbed], representative, merge_confidence="HIGH")
    _average_on_write(monkeypatch, representative, absorbed)

    lex = tmp_path / "lex"
    with pytest.raises(guards.GuardError, match="does not hold model fixed"):
        compile_mod.compile_lexicons(registry, lex, decisions_path=decisions, verify="skip")
    assert sorted(path.name for path in lex.iterdir()) == [
        "guard_outcomes.json", "provenance.json"]
    recorded = json.loads((lex / "guard_outcomes.json").read_text())
    assert [(row["guard_id"], row["passed"]) for row in recorded] == [
        ("no_cross_model_cwm_avg", False)]


def test_a_representative_averaged_within_one_model_is_recorded_and_still_passes(
        tmp_path, monkeypatch):
    """FP-05's remaining hole, made visible rather than closed.

    A mean over two motifs from the same analysis holds model, readout and
    metacluster fixed, so this guard passes it -- deliberately; it is a
    cross-model rule. What changes is that the operation is no longer invisible:
    the shipped log says `mean` over two named inputs, which is the evidence a
    reader needs to notice a constructed representative at all.
    """
    registry = _registry(tmp_path)
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    representative, absorbed = nodes[0]["node_id"], nodes[1]["node_id"]
    assert nodes[0]["model"] == nodes[1]["model"]
    decisions = _write_decisions(
        tmp_path, [representative, absorbed], representative, merge_confidence="HIGH")
    _average_on_write(monkeypatch, representative, absorbed)

    lex = tmp_path / "lex"
    compile_mod.compile_lexicons(registry, lex, decisions_path=decisions, verify="skip")

    averaged = [entry for entry in _operations(lex) if entry["op"] == "mean"]
    assert averaged and all(
        entry["inputs"] == [representative, absorbed] for entry in averaged)
    assert all(entry["group_by"] == list(guards.CROSS_MODEL_AXES) for entry in averaged)
    assert guards.no_cross_model_cwm_avg(_operations(lex)).passed


def test_an_emitted_motif_that_is_neither_its_input_nor_their_mean_is_refused(
        tmp_path, monkeypatch):
    """An operation this package cannot name, it does not file under one it can.

    Labelling an unrecognised construction `mean` would be the classifier
    inventing the finding; labelling it `copy` would let it past the guard. It is
    refused instead, naming what it was compared against.
    """
    registry = _registry(tmp_path)
    real = compile_mod._write_h5

    def doctored(path, ordered, arrays):
        payload = {
            node["node_id"]: {name: np.asarray(arrays[node["node_id"]][name][:], dtype=float)
                              for name in arrays[node["node_id"]]}
            for _, _, node in ordered
        }
        first = ordered[0][2]["node_id"]
        payload[first] = {name: values * 2.0 for name, values in payload[first].items()}
        real(path, ordered, payload)

    monkeypatch.setattr(compile_mod, "_write_h5", doctored)
    with pytest.raises(compile_mod.CompileError, match="neither the registry's own matrices"):
        compile_mod.compile_lexicons(registry, tmp_path / "lex", verify="skip")


# --- regression: a refused tier must not leave a publishable fragment ----------
# `validate.load_lexicon_binding` binds a lexicon set by globbing
# `*.manifest.json`, so a compile that wrote `core` and then refused `expanded`
# read downstream as a perfectly valid one-tier lexicon, with a `lexicon_identity`
# computed over the surviving fragment and nothing saying two tiers are missing.
def test_a_tier_refused_after_another_was_written_publishes_nothing(tmp_path):
    registry = _registry(tmp_path)
    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    odd = sorted(node["node_id"] for node in nodes)[0]
    with h5py.File(registry / "arrays.h5", "a") as h5:
        del h5[odd]["cwm"]
        h5[odd].create_dataset("cwm", data=np.zeros((MOTIF_LEN + 3, 4)))
    decisions = tmp_path / "d.json"
    decisions.write_text(json.dumps(_adjudication_payload(tiers={
        odd: {"discovery_tier": "core", "analysis_tier": "expanded",
              "tier_reason": "held out of core so only expanded mixes lengths"}})))

    out = tmp_path / "lex"
    with pytest.raises(compile_mod.CompileError, match="mixes motif lengths"):
        compile_mod.compile_lexicons(
            registry, out, decisions_path=decisions, verify="skip")

    # provenance.json stays: a rejected compile still records what was attempted
    # (T-09). Everything a consumer would read as a lexicon does not.
    assert sorted(path.name for path in out.iterdir()) == ["provenance.json"]


def test_a_successful_compile_leaves_no_staging_directory_behind(tmp_path):
    out = tmp_path / "lex"
    compile_mod.compile_lexicons(_registry(tmp_path), out, verify="skip")
    assert not [path for path in out.iterdir() if path.is_dir()]
    assert (out / "manifest.tsv").exists()
    assert {f"{tier}.h5" for tier in compile_mod.TIERS} <= {
        path.name for path in out.iterdir()}


# --- regression: an unreadable layout is not three-absence evidence ------------
# `group_absent` claims discovery ran and the group never formed (DATA_MODEL.md):
# evidence about the admission gate. An original (pre-lite) TF-MoDISco file keeps
# its patterns under `metacluster_idx_to_submetacluster_results`, so this reader
# saw neither group and recorded that claim twice, with exit 0 and an empty
# registry, for a file that may hold dozens of patterns.
def test_an_original_tfmodisco_output_is_refused_not_recorded_as_two_absences(tmp_path):
    prelite = tmp_path / "prelite.h5"
    with h5py.File(prelite, "w") as h5:
        patterns = (h5.create_group(ingest.PRE_LITE_ROOT_GROUP)
                    .create_group("metacluster_0")
                    .create_group("seqlets_to_patterns_result")
                    .create_group("patterns"))
        for name in ("pattern_0", "pattern_1"):
            motif = patterns.create_group(name)
            motif.create_dataset("contrib_scores", data=np.ones((MOTIF_LEN, 4)))
            motif.create_dataset("sequence", data=np.full((MOTIF_LEN, 4), 0.25))

    project = _project(tmp_path, analyses=[{
        "id": "legacy_run", "model": "m", "readout": "r", "union_id": "MA",
        "context": "promoter", "modisco_h5": str(prelite)}])
    with pytest.raises(ingest.IngestError, match="group_absent"):
        ingest.ingest_project(project, tmp_path / "registry")
    assert not (tmp_path / "registry" / "registry.json").exists()


# --- pinned: ingest order is lexical, and compile renumbers to the loader's ----
# This is the documented design, not a defect: recovering the loader's order at
# ingest would mean parsing the digits out of a pattern name (`no_key_parsing`).
# compile ASSIGNS names so the loader's own numeric sort reproduces the manifest,
# and the manifest carries both tag and node_id so the mapping is a table. The
# 9 -> 10 boundary is where lexicographic and numeric sorts diverge, and
# compile/README.md claims it holds there; nothing pinned that claim until here.
def test_ingest_reads_lexically_and_compile_renumbers_across_the_nine_ten_boundary(tmp_path):
    many = tmp_path / "many.h5"
    with h5py.File(many, "w") as h5:
        for i in range(12):
            _pattern(h5, "pos_patterns", f"pattern_{i}", seed=i)
    project = _project(tmp_path, analyses=[{
        "id": "wide", "model": "m", "readout": "r", "union_id": "MA",
        "context": "promoter", "modisco_h5": str(many)}])
    _meta, nodes = ingest.ingest_project(project, tmp_path / "registry")

    source_names = [node.node_id.split(".")[-1] for node in nodes]
    assert source_names.index("pattern_10") < source_names.index("pattern_2")

    manifests = compile_mod.compile_lexicons(
        tmp_path / "registry", tmp_path / "lex", tiers=("core",), verify="skip")
    index = json.loads((tmp_path / "lex" / "core.manifest.json").read_text())["index"]
    by_tag = {row["pattern_tag"]: row["node_id"] for row in index}
    assert by_tag["pos_patterns.pattern_2"].endswith("pattern_10")

    # What the loader would hand back: fixed group order, numeric suffix within.
    with h5py.File(tmp_path / "lex" / "core.h5", "r") as h5:
        emitted = [
            f"{group}.{name}"
            for group in ingest.MODISCO_GROUPS if group in h5
            for name in sorted(h5[group].keys(), key=lambda n: int(n.rsplit("_", 1)[1]))
        ]
    assert guards.index_order_matches_loader(manifests["core"].pattern_order,
                                             emitted).passed


def test_the_real_loader_returns_manifest_order_across_the_nine_ten_boundary(tmp_path):
    """The 9-10 boundary, asked of the REAL loader instead of of our model of it.

    The test above compiles with ``verify="skip"`` and then recomputes, inside the test,
    "what the loader would hand back". That is the structural assertion this module's
    docstring says proves nothing: it shows the package agrees with its own model of the
    loader, which is exactly what held while the backend renamed the argument underneath it.

    Every round trip that does reach the real loader has run on a small fixture -- five
    motifs for the recorded-outcome test, and ``probe_backend`` uses ONE -- so no test has
    ever handed the loader a group in which numeric and lexicographic order disagree. A
    union lexicon compiled from real discovery has dozens of motifs per group and crosses
    that boundary on the first one. Ten patterns is the smallest case that does.
    """
    require_finemo_backend()

    many = tmp_path / "many.h5"
    with h5py.File(many, "w") as h5:
        for i in range(12):
            _pattern(h5, "pos_patterns", f"pattern_{i}", seed=i)
        for i in range(11):
            _pattern(h5, "neg_patterns", f"pattern_{i}", seed=100 + i)
    project = _project(tmp_path, analyses=[{
        "id": "wide", "model": "m", "readout": "r", "union_id": "MA",
        "context": "promoter", "modisco_h5": str(many)}])
    ingest.ingest_project(project, tmp_path / "registry")

    # verify="require" makes compile itself do the round trip; it raises rather than write
    # a lexicon the loader disagrees with, so reaching the next line is already the claim.
    manifests = compile_mod.compile_lexicons(
        tmp_path / "registry", tmp_path / "lex", tiers=("core",), verify="require")
    manifest = manifests["core"]
    assert manifest.n_motifs == 23

    names = compile_mod.load_back(
        tmp_path / "lex" / "core.h5", trim_threshold=manifest.trim_threshold,
        motif_type=manifest.motif_type, include_rc=manifest.include_rc,
        loader_parameters=manifest.loader_parameters)
    assert names == manifest.pattern_order

    # Both groups cross the boundary. Under a lexicographic sort anywhere in the chain,
    # pattern_10 would sit between pattern_1 and pattern_2 in whichever group did it.
    for group, n in (("pos_patterns", 12), ("neg_patterns", 11)):
        suffixes = [int(name.rsplit("_", 1)[1])
                    for name in names if name.startswith(f"{group}.")]
        assert suffixes == list(range(n)), group


# --- regression: one discovery file, one checksum ------------------------------
# The digest was recomputed inside the per-pattern loop, so a 17 MB real
# modisco.h5 with 33 patterns was read 33 times to re-derive a value that cannot
# change mid-loop (~23 ms per pass on the measured file).
def test_the_discovery_file_is_hashed_once_per_analysis_not_once_per_pattern(
        tmp_path, monkeypatch):
    calls: list[str] = []
    real = ingest.sha256_file
    monkeypatch.setattr(
        ingest, "sha256_file", lambda path: (calls.append(str(path)), real(path))[1])

    _meta, nodes = ingest.ingest_project(_project(tmp_path), tmp_path / "registry")

    assert len(nodes) == 5
    digests = {node.provenance["modisco_h5_sha256"] for node in nodes}
    assert len(digests) == 1
    assert len([c for c in calls if c.endswith("a.h5")]) == 1


def test_compile_records_the_round_trip_outcome_beside_the_lexicons(tmp_path):
    """The round trip is what `compile` exists to guarantee; the directory now says so.

    `index_order_matches_loader` is run per tier against the REAL loader, and the
    lexicon directory used to carry no statement that it had been. Requires the
    backend: with none installed no round trip happens and there is, correctly,
    nothing to record.
    """
    require_finemo_backend()

    from motifmultiverse import guard_log

    out = tmp_path / "lex"
    manifests = compile_mod.compile_lexicons(_registry(tmp_path), out)

    recorded = json.loads((out / guard_log.GUARD_OUTCOMES_FILENAME).read_text())
    assert {row["guard_id"] for row in recorded} == {
        "index_order_matches_loader", "no_cross_model_cwm_avg"}
    assert all(row["stage"] == "compile" and row["passed"] for row in recorded)
    recorded = [row for row in recorded
                if row["guard_id"] == "index_order_matches_loader"]
    assert len(recorded) == len(manifests)
    for tier, manifest in manifests.items():
        assert any(f"tier {tier!r}" in row["subject"]
                   and manifest.lexicon_content_hash in row["subject"]
                   for row in recorded), tier


def test_compile_records_nothing_for_a_tier_no_backend_could_verify(tmp_path,
                                                                    no_finemo_backend):
    """No round trip happened, so no outcome exists -- and none is invented.

    Writing a "not verified" entry would put a guard's name in the record of a run
    where that guard never saw the lexicon, which is an absence rendered as a
    result. The backend's absence is `status`'s UNVERIFIED and the CLI's own line.
    """
    from motifmultiverse import guard_log

    out = tmp_path / "lex"
    compile_mod.compile_lexicons(_registry(tmp_path), out, verify="auto")

    assert [row["guard_id"] for row in guard_log.read_guard_outcomes(out)] == [
        "no_cross_model_cwm_avg"]


def test_a_cross_model_mean_made_upstream_of_the_registry_passes(tmp_path):
    """The blind spot in `no_cross_model_cwm_avg`, pinned so it cannot be forgotten.

    `compile.operations_log` classifies the emitted lexicon against the registry
    arrays, so what it can see is combination performed BETWEEN those two points.
    A cross-model mean performed one step earlier -- where a real meta-analysed-CWM
    stage would live -- arrives as an ordinary registry motif, is classified
    `copy`, and the guard passes on a lexicon that contains exactly the operation
    the guard prohibits.

    This test asserts the gap rather than the fix, which needs saying. The gap is
    not closable at compile: nothing downstream of `ingest` can tell a motif that
    was averaged upstream from one that was discovered. What made it worth a test
    is the consequence -- the pass sentence is persisted verbatim by `guard_log`
    and printed verbatim by `report`, so an unscoped one would become a durable
    published claim of something false. So this also pins the SENTENCE: if a future
    edit widens it back to "no CWM averaging crosses model, readout or
    metacluster", this fails and says why.

    When a combining stage does appear, the honest close is a record written by
    that stage's inputs, not a better classifier here.
    """
    reg, lex = tmp_path / "registry", tmp_path / "lex"
    a, b, c = tmp_path / "a.h5", tmp_path / "b.h5", tmp_path / "c.h5"
    with h5py.File(a, "w") as h5:
        _pattern(h5, "pos_patterns", "pattern_0", seed=1)
    with h5py.File(b, "w") as h5:
        _pattern(h5, "pos_patterns", "pattern_0", seed=900)
    # The prohibited operation, performed upstream of the registry.
    with h5py.File(a) as ha, h5py.File(b) as hb, h5py.File(c, "w") as hc:
        grp = hc.require_group("pos_patterns").create_group("pattern_0")
        for ds in ("contrib_scores", "hypothetical_contribs", "sequence"):
            grp.create_dataset(ds, data=(np.asarray(ha[f"pos_patterns/pattern_0/{ds}"][:], float)
                                         + np.asarray(hb[f"pos_patterns/pattern_0/{ds}"][:], float)) / 2.0)
        grp.create_group("seqlets").create_dataset("n_seqlets", data=np.array(500))

    project = tmp_path / "project.json"
    project.write_text(json.dumps({
        "project": "p", "peak_universe_id": "u1",
        "analyses": [
            {"id": i, "model": f"model{i}", "readout": "r1", "union_id": f"M{i}",
             "context": "promoter", "modisco_h5": str(path)}
            for i, path in (("A", a), ("B", b), ("C", c))
        ],
    }))
    ingest.ingest_project(project, reg)
    compile_mod.compile_lexicons(reg, lex, verify="skip")

    ops = json.loads((lex / "combination_operations.json").read_text())
    assert {op["op"] for op in ops} == {"copy"}, (
        "the averaged motif was classified as something other than a copy; if the "
        "classifier gained upstream reach, this test is the one to rewrite")

    # It really is the prohibited operation, in the published bytes.
    _, nodes, arrays = ingest.load_registry(reg)
    by_model = {n["model"]: n["node_id"] for n in nodes}
    mean_ab = (np.asarray(arrays[by_model["modelA"]]["cwm"][:], float)
               + np.asarray(arrays[by_model["modelB"]]["cwm"][:], float)) / 2.0
    arrays.close()
    with h5py.File(lex / "core.h5") as h5:
        published = [name for name in h5["pos_patterns"]
                     if np.allclose(h5[f"pos_patterns/{name}/contrib_scores"][:], mean_ab)]
    assert published, "the probe did not actually publish a cross-model mean"

    result = guards.no_cross_model_cwm_avg(ops)
    assert result.passed, "the guard is expected to pass here -- that IS the gap"
    assert "outside what this checked" in result.detail, (
        "the pass sentence no longer states its scope; it is persisted verbatim by "
        "guard_log and printed verbatim by report, so an unscoped sentence becomes a "
        f"published claim that this very lexicon refutes. Got: {result.detail!r}")
