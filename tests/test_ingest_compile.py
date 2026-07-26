"""ingest + compile: the registry, the three absences, and the loader round trip.

The round-trip test is behavioural. Asserting that the written HDF5 contains the
groups we just wrote would prove only that this package can read its own output;
the question is whether the *hit caller* can, and in which order it hands the
motifs back. That test needs the finemo backend and skips without it -- a skip
that must be read as "unverified here", not as "verified".
"""
from __future__ import annotations

import json

import pytest

from motifmultiverse import compile as compile_mod
from motifmultiverse import guards, ingest
from motifmultiverse.schema import MetaclusterState, RegistryMetadata, SchemaError

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
    assert n.metacluster == "pos" and n.denovo_pattern_id == "pos_patterns.pattern_0"
    assert n.motif_length == MOTIF_LEN and n.seqlet_count == 250
    assert n.core_ic is not None and n.trimmed_core is not None
    assert n.provenance["modisco_h5_sha256"] and n.provenance["analysis_id"] == "modelA_r1"
    assert meta.project == "test-project"


def test_ingest_writes_arrays_and_provenance(tmp_path):
    out = tmp_path / "registry"
    ingest.ingest_project(_project(tmp_path), out)
    assert (out / "registry.json").exists() and (out / "arrays.h5").exists()
    prov = json.loads((out / "provenance.json").read_text())[0]
    assert prov["subcommand"] == "ingest"
    assert set(prov["inputs"]) == {"project.json", "a.h5"}
    assert prov["redaction_policy"] == "basenames_only_except_command"
    meta, records, arrays = ingest.load_registry(out)
    try:
        assert len(records) == 5
        assert set(arrays[records[0]["node_id"]].keys()) == {"cwm", "hypothetical_cwm", "ppm"}
    finally:
        arrays.close()


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
        "pos_patterns.pattern_0", "pos_patterns.pattern_1", "pos_patterns.pattern_2",
        "neg_patterns.pattern_0", "neg_patterns.pattern_1"}
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
    path.write_text(json.dumps({"decisions": [decision]}))
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
    decisions.write_text(json.dumps({"decisions": [{
        "cluster_id": "c1", "decision": "collapse",
        "members": [nodes[0]["node_id"], nodes[1]["node_id"]],
        "representative": "a_constructed_average", "confidence": 0.9,
        "rationale": "x", "decided_by": "test"}]}))
    with pytest.raises(compile_mod.CompileError, match="observed medoid"):
        compile_mod.compile_lexicons(registry, tmp_path / "lex", decisions_path=decisions)


def test_the_content_hash_is_deterministic_and_tracks_membership(tmp_path):
    registry = _registry(tmp_path)
    a = compile_mod.compile_lexicons(registry, tmp_path / "lex1")
    b = compile_mod.compile_lexicons(registry, tmp_path / "lex2")
    assert a["core"].lexicon_content_hash == b["core"].lexicon_content_hash

    _, nodes, arrays = ingest.load_registry(registry)
    arrays.close()
    decisions = tmp_path / "d.json"
    decisions.write_text(json.dumps({"decisions": [{
        "cluster_id": "c1", "decision": "collapse",
        "members": [nodes[0]["node_id"], nodes[1]["node_id"]],
        "representative": nodes[0]["node_id"], "merge_confidence": "MODERATE",
        "rationale": "moderate", "decided_by": "test"}]}))
    c = compile_mod.compile_lexicons(registry, tmp_path / "lex3", decisions_path=decisions)
    assert c["core"].lexicon_content_hash != a["core"].lexicon_content_hash
    assert c["sensitivity"].lexicon_content_hash == a["core"].lexicon_content_hash


def test_a_lexicon_of_mixed_motif_lengths_is_refused(tmp_path):
    """The loader stacks every motif into one array, so lengths must agree."""
    registry = _registry(tmp_path)
    with h5py.File(registry / "arrays.h5", "a") as h5:
        node_id = sorted(h5.keys())[0]
        del h5[node_id]["cwm"]
        h5[node_id].create_dataset("cwm", data=np.zeros((MOTIF_LEN + 3, 4)))
    with pytest.raises(compile_mod.CompileError, match="mixes motif lengths"):
        compile_mod.compile_lexicons(registry, tmp_path / "lex")


# ------------------------------------------------- the real loader, or a skip
def test_roundtrip_against_the_real_loader(tmp_path):
    """Behavioural, not structural: the hit caller reads it, in this order."""
    pytest.importorskip("finemo", reason="round-trip needs the finemo backend")
    manifests = compile_mod.compile_lexicons(_registry(tmp_path), tmp_path / "lex",
                                             verify="require")
    names = compile_mod.load_back(tmp_path / "lex" / "core.h5")
    assert names == manifests["core"].pattern_order
    assert guards.index_order_matches_loader(manifests["core"].pattern_order, names).passed


def test_verify_require_fails_loudly_when_the_backend_is_absent(tmp_path):
    import importlib.util
    if importlib.util.find_spec("finemo") is not None:
        pytest.skip("finemo is installed; this asserts the no-backend path")
    with pytest.raises(compile_mod.BackendMissing, match="finemo"):
        compile_mod.compile_lexicons(_registry(tmp_path), tmp_path / "lex", verify="require")


def test_auto_verification_still_writes_but_claims_nothing(tmp_path):
    """Without a backend the lexicon is written and simply not verified.

    `auto` must neither fail nor pretend: the file exists, and nothing in the
    manifest asserts a round trip that did not happen.
    """
    manifests = compile_mod.compile_lexicons(_registry(tmp_path), tmp_path / "lex",
                                             verify="auto")
    assert (tmp_path / "lex" / "core.h5").exists()
    assert "verified" not in json.dumps(manifests["core"].pattern_order)
