"""The reference path, run end to end on synthetic inputs (Task 18).

Every other test file checks one stage against fixtures shaped by hand. This one
checks that the stages actually compose: that what `ingest` writes is what
`align` can read, that what `adjudicate` decides is what `compile` will consume,
and that the artifacts a reader is promised at the end are all present and
parseable rather than merely reachable in principle.

Two things this file deliberately does NOT do:

* It does not count a skipped optional-backend check as a pass. The `finemo`
  round trip is the only way to prove a compiled lexicon loads in the tool that
  will consume it, and it is not installed here; the test asserts the *structure*
  of the H5 and separately asserts that the round trip is reported UNVERIFIED.
  A green run of this file is not evidence that a lexicon loads.
* It does not use the shipped `config/criteria.v1.yaml` to demonstrate a
  collapse. Two of that file's four criteria are `CRITERION_NOT_YET_DEFINED` on
  purpose, so the shipped pipeline defers every duplicate and every fragment.
  The collapse path is exercised with a criteria file written *inside this test*
  and named as such, and a companion assertion pins that the shipped file still
  defers -- otherwise this test would quietly become the thing that made a
  threshold look decided.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motifmultiverse.cli import main
from motifmultiverse.schema import HIT_TABLE_COLUMNS, build_peak_split_manifest
from motifmultiverse.validate import DecisionSplitArtifact, ValidationSplitArtifact

h5py = pytest.importorskip("h5py")


# --------------------------------------------------------------------------- #
# Fixtures: a synthetic TF-MoDISco output with designed relationships
# --------------------------------------------------------------------------- #
def _ppm(length: int, seed: int) -> np.ndarray:
    """A deterministic informative PPM: one dominant base per position."""
    rng = np.random.default_rng(seed)
    matrix = np.full((length, 4), 0.05)
    for i in range(length):
        matrix[i, rng.integers(0, 4)] = 0.85
    return matrix / matrix.sum(axis=1, keepdims=True)


def _contrib(ppm: np.ndarray, strong_upto: int) -> np.ndarray:
    """Contribution mass concentrated in the first `strong_upto` positions.

    `ingest` derives `motif_completeness` from the trimmed core, so two patterns
    sharing a PPM but differing here are near-identical motifs that a medoid
    tie-break can still order -- which is what lets a duplicate pair resolve a
    representative instead of deferring for want of tie metadata.
    """
    out = (ppm - 0.25) * 0.05
    out[:strong_upto] = (ppm[:strong_upto] - 0.25) * 2.0
    return out


def _write_modisco(path: Path, patterns: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    with h5py.File(path, "w") as h5:
        group = h5.require_group("pos_patterns")
        for name, (sequence, contrib) in patterns.items():
            pattern = group.create_group(name)
            pattern.create_dataset("sequence", data=sequence)
            pattern.create_dataset("contrib_scores", data=contrib)
            pattern.create_dataset("hypothetical_contribs", data=contrib)
    return path


def _project(tmp_path: Path, modisco: Path) -> Path:
    project = tmp_path / "project.json"
    project.write_text(json.dumps({
        "project": "end-to-end", "peak_universe_id": "u1",
        "analyses": [{"id": "a1", "model": "m1", "readout": "counts", "union_id": "MA",
                      "context": "promoter", "modisco_h5": str(modisco)}],
    }), encoding="utf-8")
    return project


def _uniform_length_patterns() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Three 20bp patterns: a complete one, a less complete near-duplicate, and a
    sign flip of the first.

    All one length on purpose. `compile` refuses a lexicon whose patterns differ
    in length (the loader stacks them into one array), so the path that ends in a
    compiled lexicon has to be uniform -- which in turn means `align`'s bilateral
    overlap rule registers every pair and proposes ONE component. That is the
    shape of a real single-width TF-MoDISco lexicon, not a limitation of the
    fixture.
    """
    base = _ppm(20, 11)
    return {
        "pattern_0": (base, _contrib(base, 18)),
        "pattern_1": (base.copy(), _contrib(base, 12)),
        "pattern_2": (base.copy(), -_contrib(base, 18)),
    }


def _length_separated_patterns() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Six patterns in three pairs that `align` cannot join to each other.

    The bilateral overlap floor (`overlap_bp >= 6` AND `>= 0.5` of *each* motif's
    own length) is the only thing separating components, so the three pairs use
    lengths 7 / (16,30) / 70: within a pair the registration clears the floor,
    across pairs the shorter motif covers under half the longer one.

      A  7,  7   identical PPM, different core spans   -> TRUE_DUPLICATE
      B 30, 16   a fragment of its parent              -> FRAGMENT_MATCH
      C 70, 70   identical PPM, negated CWM            -> TRUE_DUPLICATE, sign-flipped
    """
    a = _ppm(7, 11)
    parent = _ppm(30, 22)
    c = _ppm(70, 33)
    return {
        "pattern_0": (a, _contrib(a, 6)),
        "pattern_1": (a.copy(), _contrib(a, 4)),
        "pattern_2": (parent, _contrib(parent, 24)),
        "pattern_3": (parent[:16].copy(), _contrib(parent[:16].copy(), 12)),
        # Pair C's two members differ in core span so the medoid tie resolves on
        # motif_completeness. Without that they tie on every dimension the
        # registry can supply and the component defers for want of tie metadata
        # -- a real behaviour, but not the deferral this fixture is here to show.
        "pattern_4": (c, _contrib(c, 60)),
        "pattern_5": (c.copy(), -_contrib(c, 50)),
    }


NODE = "a1::pos_patterns.pattern_{}"


# --------------------------------------------------------------------------- #
# A frozen hit substrate for the inference tail of the path
# --------------------------------------------------------------------------- #
SUBSTRATE_ID = "e" * 64


def _hit_substrate(tmp_path: Path, n_blocks: int = 36) -> tuple[Path, Path, Path]:
    """A frozen hit table with a planted +0.9 effect, above every floor."""
    lines = ["\t".join(HIT_TABLE_COLUMNS)]
    query, comparator = [], []
    for block in range(n_blocks):
        for side in (0, 1):
            region = f"r{block}_{side}"
            (query if side == 0 else comparator).append(region)
            start = block * 1_000_000 + side * 1000
            coefficient = 1.0 + (block % 3) * 0.3 if side == 0 else 0.4
            lines.append("\t".join([
                region, "chr1", str(start), str(start + 500),
                f"UA_FAMA_{side}", "FAM_A", str(coefficient), "used",
                "9999", "lex_v1", SUBSTRATE_ID,
            ]))
    hits = tmp_path / "hits.tsv"
    hits.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "query.txt").write_text("\n".join(query) + "\n", encoding="utf-8")
    (tmp_path / "comparator.txt").write_text("\n".join(comparator) + "\n", encoding="utf-8")
    return hits, tmp_path / "query.txt", tmp_path / "comparator.txt"


def _validation_inputs(tmp_path: Path, lexicons: Path, decision_id: str,
                       n_peaks: int = 70, n_affected: int = 32):
    """Everything `validate` needs, bound to a real compiled lexicon.

    `n_affected` of `n_peaks` change between the before and after tables, chosen
    so BOTH of Task 14's properties are live on the real path: at least 30
    affected peaks, so the result is estimable rather than
    `LOW_RISK_RARE_NOT_VALIDATED`, and fewer than half of them, so the all-peak
    delta dilutes to zero while the affected-subset delta does not. A fixture
    where the two agree would let a regression that silently reported the
    all-peak number pass unnoticed.
    """
    from motifmultiverse.schema.substrate import CallerSpecification
    from motifmultiverse.substrate import build_manifest, write_manifest

    manifest_blob = json.loads((lexicons / "core.manifest.json").read_text(encoding="utf-8"))
    substrate = build_manifest(
        peak_universe_hash="a" * 64,
        n_regions=n_peaks,
        caller_specification=CallerSpecification(
            caller_name="finemo", caller_version="0.test",
            lexicon_content_hash=manifest_blob["lexicon_content_hash"],
            parameters={"motif_type": manifest_blob["motif_type"]},
            preprocessing_contract_hash="b" * 64,
        ),
        input_files={"peaks.bed": "c" * 64},
        created_at="2026-07-26T12:00:00Z",
    )
    substrate_path = write_manifest(substrate, tmp_path / "substrate.manifest.json")

    peaks = [f"v{i}" for i in range(n_peaks)]
    before_rows, after_rows = [], []
    for i, peak in enumerate(peaks):
        before_rows.append({"peak_id": peak, "hit_id": f"old-{i}", "coefficient": 1.0,
                            "reconstruction": 0.5, "substrate_id": substrate.substrate_id})
        changed = i < n_affected
        after_rows.append({
            "peak_id": peak,
            "hit_id": f"new-{i}" if changed else f"old-{i}",
            "coefficient": 1.4 if changed else 1.0,
            "reconstruction": 0.8 if changed else 0.5,
            "substrate_id": substrate.substrate_id,
        })
    before = tmp_path / "before.parquet"
    after = tmp_path / "after.parquet"
    pd.DataFrame(before_rows).to_parquet(before, index=False)
    pd.DataFrame(after_rows).to_parquet(after, index=False)

    split = build_peak_split_manifest({
        **{peak: "VALIDATION" for peak in peaks},
        "d0": "DISCOVERY",
    })
    split_path = tmp_path / "split-manifest.json"
    split_path.write_text(json.dumps({
        "schema_version": split.schema_version,
        "assignments": {key: value.value for key, value in split.assignments.items()},
        "checksum": split.checksum,
    }), encoding="utf-8")
    decision_artifact = DecisionSplitArtifact.create(
        manifest=split, decision_id=decision_id,
        decision_peak_ids=frozenset({"d0"}), validation_peak_ids=frozenset(peaks))
    validation_artifact = ValidationSplitArtifact.create(
        manifest=split, decision_id=decision_id, result_id="pending",
        decision_peak_ids=frozenset({"d0"}), validation_peak_ids=frozenset(peaks))
    decision_path = tmp_path / "decision-split.json"
    validation_path = tmp_path / "validation-split.json"
    decision_path.write_text(json.dumps(decision_artifact.to_dict()), encoding="utf-8")
    validation_path.write_text(json.dumps(validation_artifact.to_dict()), encoding="utf-8")
    return dict(before=before, after=after, substrate=substrate_path,
                split=split_path, decision=decision_path, validation=validation_path)


# --------------------------------------------------------------------------- #
# The reference path
# --------------------------------------------------------------------------- #
#: Every artifact the reference path promises a reader, relative to the run root.
DECLARED_ARTIFACTS = (
    "registry/registry.json",
    "evidence/alignment_edges.parquet",
    "evidence/annotation_candidates.parquet",
    "adjudication/ontology_decisions.parquet",
    "lexicons/core.h5",
    "lexicons/core.manifest.json",
    "validation/stability_results.parquet",
    "inference/effect_estimates.tsv",
)


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory) -> Path:
    """Run the whole declared path once, through the real CLI, and hand back its root.

    Module-scoped because it is the same run every assertion below is about: a
    per-test re-run would let the artifacts drift apart between assertions, which
    is precisely the failure this file exists to catch.
    """
    root = tmp_path_factory.mktemp("reference")
    modisco = _write_modisco(root / "modisco.h5", _uniform_length_patterns())
    project = _project(root, modisco)

    assert main(["ingest", str(project), "--out", str(root / "registry")]) == 0
    assert main(["align", str(root / "registry"), "--out", str(root / "evidence"),
                 "--null-shuffles", "10", "--seed", "3"]) == 0
    assert main(["annotate", str(root / "evidence"), "--registry", str(root / "registry"),
                 "--out", str(root / "evidence")]) == 0
    assert main(["adjudicate", str(root / "evidence"), "--registry", str(root / "registry"),
                 "--out", str(root / "adjudication"),
                 "--review", str(root / "adjudication" / "review.yaml")]) == 0
    assert main(["compile", str(root / "registry"),
                 "--decisions", str(root / "adjudication" / "merge_decisions.json"),
                 "--out", str(root / "lexicons")]) == 0

    decisions = pd.read_parquet(root / "adjudication" / "ontology_decisions.parquet")
    inputs = _validation_inputs(root, root / "lexicons", decisions["decision_id"].iloc[0])
    assert main(["validate", str(root / "lexicons"),
                 "--before-hits", str(inputs["before"]), "--after-hits", str(inputs["after"]),
                 "--substrate-manifest", str(inputs["substrate"]),
                 "--split-manifest", str(inputs["split"]),
                 "--decision-artifact", str(inputs["decision"]),
                 "--validation-artifact", str(inputs["validation"]),
                 "--out", str(root / "validation")]) == 0

    hits, query, comparator = _hit_substrate(root)
    assert main(["infer", str(hits), "--peaks", str(query), "--comparator", str(comparator),
                 "--comparator-id", "even_vs_odd", "--selection-provenance", "EXTERNAL",
                 "--estimator", "bca-wild-cluster", "--bootstrap", "200", "--seed", "1",
                 "--floor-blocks", "36", "--out", str(root / "inference")]) == 0
    # `interpret` is not a stage of the path -- it is a separate consumer of the
    # same frozen substrate -- but it must still work against what the path built.
    assert main(["interpret", str(hits), "--peaks", str(query), "--comparator", str(comparator),
                 "--comparator-id", "even_vs_odd", "--selection-provenance", "EXTERNAL",
                 "--bootstrap", "50", "--seed", "1", "--floor-blocks", "36",
                 "--out", str(root / "interpretation")]) == 0
    return root


@pytest.mark.parametrize("relative", DECLARED_ARTIFACTS)
def test_every_declared_artifact_exists_and_is_not_empty(reference_run, relative):
    path = reference_run / relative
    assert path.exists(), f"{relative} was never written"
    assert path.stat().st_size > 0, f"{relative} is a zero-byte file"


def test_every_declared_artifact_parses_as_its_own_format(reference_run):
    """Existence is not content. Each artifact is opened with the reader its
    consumer would use, and asked for something only a valid file can answer."""
    registry = json.loads((reference_run / "registry" / "registry.json").read_text())
    assert len(registry["nodes"]) == 3
    assert registry["registry_metadata"]["cross_model_claims_restricted"] is True

    edges = pd.read_parquet(reference_run / "evidence" / "alignment_edges.parquet")
    assert len(edges) == 3                      # every pair of the three patterns
    assert set(edges["registered_on"]) == {"unsigned_ppm"}
    assert (edges["null_shuffles"] == 10).all() and (edges["seed"] == 3).all()

    # Zero rows is the CORRECT content here: no optional annotation backend is
    # installed, so no candidate was proposed. The file still has to carry its
    # schema, or a later reader cannot tell an empty result from a broken one.
    candidates = pd.read_parquet(reference_run / "evidence" / "annotation_candidates.parquet")
    assert {"candidate_id", "node_id", "proposed_family_id", "source"} <= set(candidates.columns)

    decisions = pd.read_parquet(reference_run / "adjudication" / "ontology_decisions.parquet")
    assert len(decisions) == 1
    assert decisions["relationship"].iloc[0] == "TRUE_DUPLICATE"
    assert decisions["criterion_id"].iloc[0] == "TRUE_DUPLICATE"

    manifest = json.loads((reference_run / "lexicons" / "core.manifest.json").read_text())
    assert len(manifest["lexicon_content_hash"]) == 64
    assert manifest["n_motifs"] == len(manifest["index"]) == 3

    stability = pd.read_parquet(reference_run / "validation" / "stability_results.parquet")
    assert len(stability) == 1
    assert stability["n_affected_peaks"].iloc[0] == 32
    # Task 14's dilution property, on the real path: the change is invisible in
    # the all-peak number and visible in the affected subset.
    assert stability["paired_delta_reconstruction_all"].iloc[0] == pytest.approx(0.0)
    assert stability["paired_delta_reconstruction_affected"].iloc[0] > 0.1
    assert stability["status"].iloc[0] != "LOW_RISK_RARE_NOT_VALIDATED"

    tsv = (reference_run / "inference" / "effect_estimates.tsv").read_text().strip().split("\n")
    assert len(tsv) == 2                        # header + one family
    assert "p_value" in tsv[0].split("\t")


def test_the_compiled_lexicon_is_structurally_loadable(reference_run):
    """The H5 and its manifest agree, position by position.

    This is NOT the real round trip: that needs the `finemo` backend, which is
    not installed here (see the companion test below). What it proves is that the
    file the loader would open has the shape the manifest promises -- the
    necessary half of the claim, not the sufficient one.
    """
    manifest = json.loads((reference_run / "lexicons" / "core.manifest.json").read_text())
    with h5py.File(reference_run / "lexicons" / "core.h5", "r") as h5:
        tags = [f"{group}.{name}"
                for group in ("pos_patterns", "neg_patterns") if group in h5
                for name in h5[group]]
        assert tags == manifest["pattern_order"], "H5 order diverges from the manifest index"
        lengths = set()
        for entry in manifest["index"]:
            group, name = entry["pattern_tag"].split(".")
            array = h5[group][name]["contrib_scores"][:]
            assert array.ndim == 2 and array.shape[1] == 4
            lengths.add(array.shape[0])
    # One stacked array means one width; a mixed-width lexicon cannot be read at all.
    assert len(lengths) == 1


def test_the_real_loader_round_trip_is_reported_unverified_not_passed(reference_run, capsys):
    """A skipped backend check is unverified. It is never a pass, and this run
    must say which of the two it was rather than leaving a green suite to imply
    the stronger claim."""
    from motifmultiverse import compile as compile_mod

    try:
        compile_mod.load_back(reference_run / "lexicons" / "core.h5")
    except compile_mod.BackendMissing as exc:
        assert "finemo" in str(exc)
        return
    pytest.skip("the finemo backend IS installed here; the round trip is verified elsewhere")


def test_the_sign_flipped_representation_aligns_at_the_same_registration(reference_run):
    """`FP-06`: registration is chosen on unsigned sequence content, so a motif
    and its sign flip must select the SAME offset and orientation, and the sign
    must then show up in the signed CWM similarity rather than in the alignment.

    If registration were chosen on signed CWM, this pair would either fail to
    register or register at some offset that maximised a negative correlation --
    and the opposition, which is the scientific finding, would be invisible.
    """
    edges = pd.read_parquet(reference_run / "evidence" / "alignment_edges.parquet")
    flip = edges[
        (edges["source_node_id"] == NODE.format(0))
        & (edges["target_node_id"] == NODE.format(2))
    ]
    assert len(flip) == 1
    row = flip.iloc[0]
    assert row["orientation"] == "+" and row["offset"] == 0
    assert row["overlap_frac_source"] == row["overlap_frac_target"] == 1.0
    assert row["ppm_similarity"] == pytest.approx(1.0)          # identical sequence content
    assert row["signed_cwm_similarity"] <= -0.9                 # opposite contribution


def test_the_inference_tail_carries_its_licence_onto_every_row(reference_run):
    from motifmultiverse.cli import EFFECT_ESTIMATE_COLUMNS

    lines = (reference_run / "inference" / "effect_estimates.tsv").read_text().strip().split("\n")
    row = dict(zip(EFFECT_ESTIMATE_COLUMNS, lines[1].split("\t"), strict=True))
    assert row["family_id"] == "FAM_A"
    assert float(row["effect"]) == pytest.approx(0.9, abs=0.05)
    assert row["inference_capability"] == "INTERVAL_AND_TEST"
    assert row["estimator"] == "wild_cluster_bootstrap_t"
    assert float(row["p_value"]) < 0.05
    assert row["substrate_id"] == SUBSTRATE_ID
    assert row["statistical_license"] == "FULL_INFERENCE"

    # interpret consumed the same substrate and, on its conservative default,
    # withheld what it is not licensed to emit.
    blob = json.loads((reference_run / "interpretation" / "interpretation.json").read_text())
    assert blob["substrate_id"] == SUBSTRATE_ID
    assert blob["effects"][0]["p_value"] is None
    assert blob["effects"][0]["inference_capability"] == "ESTIMATION_ONLY"


def test_every_stage_left_a_provenance_record(reference_run):
    """T-09: a stage that cannot name its inputs describes nothing."""
    for directory, stage in (("registry", "ingest"), ("evidence", "align"),
                             ("adjudication", "adjudicate"), ("lexicons", "compile"),
                             ("validation", "validate"), ("inference", "infer"),
                             ("interpretation", "interpret")):
        records = json.loads((reference_run / directory / "provenance.json").read_text())
        stages = {record["subcommand"] for record in records}
        assert stage in stages, f"{directory}/provenance.json never records {stage}"
        for record in records:
            assert record["command"] and record["timestamp_utc"] and record["software"]


# --------------------------------------------------------------------------- #
# Adjudication: collapse, refusal and deferral in one run
# --------------------------------------------------------------------------- #
#: A criteria registry written FOR THIS TEST, not the project's.
#:
#: `config/criteria.v1.yaml` leaves TRUE_DUPLICATE and FRAGMENT_MATCH
#: `CRITERION_NOT_YET_DEFINED`, because no frozen design document states how much
#: reconstruction loss a collapse may cost, and inventing that number is the
#: thing the criterion registry exists to prevent. That is a statement about the
#: science, not about the code -- but it does mean the shipped pipeline can never
#: collapse anything, so the collapse path needs a criteria file of its own to be
#: exercised at all. These thresholds are fixture values with no scientific
#: standing, and `test_the_shipped_criteria_still_refuse_to_guess_a_threshold`
#: pins that the shipped file is unchanged.
TEST_CRITERIA = """\
schema_version: "1"
criteria:
  - criterion_id: TRUE_DUPLICATE
    version: "1"
    status: FROZEN
    relationship: TRUE_DUPLICATE
    required_evidence:
      - paired_delta_reconstruction_affected
      - family_coefficient_share
      - ppm_similarity
    predicates:
      - field: paired_delta_reconstruction_affected
        operator: le
        value: 0.5
      - field: family_coefficient_share
        operator: ge
        value: 0.5
      - field: ppm_similarity
        operator: ge
        value: 0.99
    insufficient_evidence_action: deferred
    decision_if_matched: collapse

  - criterion_id: FRAGMENT_MATCH
    version: "1"
    status: FROZEN
    relationship: FRAGMENT_MATCH
    required_evidence:
      - overlap_frac_source
      - overlap_frac_target
    predicates:
      - field: overlap_frac_source
        operator: ge
        value: 1.0
      - field: overlap_frac_target
        operator: ge
        value: 1.0
    insufficient_evidence_action: deferred
    decision_if_matched: collapse
"""


def _decision_by_members(decisions: pd.DataFrame, *suffixes: str) -> pd.Series:
    """Find the row whose component is exactly these registry nodes."""
    wanted = {NODE.format(suffix) for suffix in suffixes}
    for row in decisions.itertuples():
        if set(json.loads(row.node_ids)) == wanted:
            return row
    raise AssertionError(f"no decision covers exactly {sorted(wanted)}")


def _standalone_lexicon(tmp_path: Path) -> Path:
    """A one-motif lexicon directory, so `validate` has an identity to bind to.

    The length-separated registry below cannot be compiled -- `compile` refuses a
    lexicon whose motifs differ in width -- but `validate` binds to a lexicon
    MANIFEST, not to the registry that adjudication used. Supplying the manifest
    directly is what lets the downstream-evidence loop be exercised on a registry
    whose whole point is that its motifs are different lengths.
    """
    from motifmultiverse.compile import lexicon_semantic_hash

    lexicons = tmp_path / "standalone-lexicons"
    lexicons.mkdir()
    array = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    with h5py.File(lexicons / "core.h5", "w") as h5:
        h5.create_group("pos_patterns").create_group("pattern_0").create_dataset(
            "contrib_scores", data=array)
    content_hash = lexicon_semantic_hash(
        [("pos_patterns", "pattern_0", {"node_id": "node-0"})],
        {"node-0": {"cwm": array}},
        schema_version="1.0", trim_threshold=0.3, motif_type="cwm", include_rc=False,
        loader_backend="finemo", loader_parameters={"motif_lambda_default": 0.7})
    (lexicons / "core.manifest.json").write_text(json.dumps({
        "tier": "core", "lexicon_content_hash": content_hash, "n_motifs": 1,
        "pattern_order": ["pos_patterns.pattern_0"], "node_ids": ["node-0"],
        "index": [{"index": 0, "pattern_tag": "pos_patterns.pattern_0",
                   "node_id": "node-0", "variant_id": "MA_FAM_01", "metacluster": "pos"}],
        "schema_version": "1.0", "trim_threshold": 0.3, "motif_type": "cwm",
        "include_rc": False, "loader_backend": "finemo",
        "loader_parameters": {"motif_lambda_default": 0.7},
        "comparisons": {}, "source_registry": "registry", "sensitivity_triggers": {},
        "project": "end-to-end", "cross_model_claims_restricted": True,
    }), encoding="utf-8")
    return lexicons


@pytest.fixture(scope="module")
def adjudication_run(tmp_path_factory) -> dict:
    """align -> adjudicate -> validate -> adjudicate again, on one registry.

    The second adjudication pass is the point. `TRUE_DUPLICATE` requires
    downstream stability evidence, and downstream stability cannot exist before a
    provisional merge has been evaluated -- so a first pass necessarily defers,
    `validate` measures what the merge would cost, and only then can a criterion
    decide. "A merge is validated downstream, not by similarity", made operational.
    """
    root = tmp_path_factory.mktemp("adjudication")
    modisco = _write_modisco(root / "modisco.h5", _length_separated_patterns())
    project = _project(root, modisco)

    assert main(["ingest", str(project), "--out", str(root / "registry")]) == 0
    assert main(["align", str(root / "registry"), "--out", str(root / "evidence"),
                 "--null-shuffles", "10", "--seed", "3"]) == 0
    assert main(["annotate", str(root / "evidence"), "--registry", str(root / "registry"),
                 "--out", str(root / "evidence")]) == 0
    assert main(["adjudicate", str(root / "evidence"), "--registry", str(root / "registry"),
                 "--out", str(root / "pass1"),
                 "--review", str(root / "pass1" / "review.yaml")]) == 0
    first = pd.read_parquet(root / "pass1" / "ontology_decisions.parquet")

    # Downstream evidence for the duplicate pair only, keyed by ITS decision id.
    # The criterion ids and versions are identical between passes, so the id a
    # stability row names is the id the second pass recomputes.
    duplicate_id = _decision_by_members(first, "0", "1").decision_id
    lexicons = _standalone_lexicon(root)
    inputs = _validation_inputs(root, lexicons, duplicate_id)
    assert main(["validate", str(lexicons),
                 "--before-hits", str(inputs["before"]), "--after-hits", str(inputs["after"]),
                 "--substrate-manifest", str(inputs["substrate"]),
                 "--split-manifest", str(inputs["split"]),
                 "--decision-artifact", str(inputs["decision"]),
                 "--validation-artifact", str(inputs["validation"]),
                 "--out", str(root / "validation")]) == 0
    (root / "evidence" / "stability_results.parquet").write_bytes(
        (root / "validation" / "stability_results.parquet").read_bytes())

    criteria = root / "test-criteria.yaml"
    criteria.write_text(TEST_CRITERIA, encoding="utf-8")
    assert main(["adjudicate", str(root / "evidence"), "--registry", str(root / "registry"),
                 "--criteria", str(criteria), "--out", str(root / "pass2"),
                 "--review", str(root / "pass2" / "review.yaml")]) == 0
    return {
        "root": root,
        "first": first,
        "second": pd.read_parquet(root / "pass2" / "ontology_decisions.parquet"),
    }


def test_the_first_pass_defers_everything_for_want_of_downstream_evidence(adjudication_run):
    """Under the SHIPPED criteria, nothing collapses -- and that is the design."""
    first = adjudication_run["first"]
    assert len(first) == 3
    assert set(first["decision"]) == {"deferred"}
    assert set(first["relationship"]) == {"TRUE_DUPLICATE", "FRAGMENT_MATCH"}
    assert first["representative_node_id"].isna().all()


def test_adjudication_emits_collapse_refusal_and_deferral_in_one_run(adjudication_run):
    """All three first-class outcomes, each for its own stated reason."""
    second = adjudication_run["second"]
    assert len(second) == 3

    duplicate = _decision_by_members(second, "0", "1")
    assert duplicate.relationship == "TRUE_DUPLICATE"
    assert duplicate.decision == "collapse"
    # The representative is an OBSERVED member, never a synthesised average.
    assert duplicate.representative_node_id in set(json.loads(duplicate.node_ids))
    assert duplicate.representative_node_id == NODE.format(0)   # the more complete motif
    assert any("stability:" in evidence for evidence in json.loads(duplicate.evidence_ids))

    fragment = _decision_by_members(second, "2", "3")
    assert fragment.relationship == "FRAGMENT_MATCH"
    assert fragment.decision == "refuse_merge"
    # `None` survives the Parquet round trip as NaN; either way it must not name
    # a representative -- a refused cluster has no surviving member by definition.
    assert pd.isna(fragment.representative_node_id)
    assert json.loads(fragment.evidence_against)

    deferred = _decision_by_members(second, "4", "5")
    assert deferred.decision == "deferred"
    assert pd.isna(deferred.representative_node_id)
    # Deferred for the RIGHT reason: no downstream evidence names this decision.
    assert "missing required evidence" in deferred.rationale


def test_every_considered_cluster_appears_including_the_ones_that_did_not_collapse(
        adjudication_run):
    """A refusal that is not written down is indistinguishable from a cluster
    nobody looked at."""
    root = adjudication_run["root"]
    bundle = json.loads((root / "pass2" / "merge_decisions.json").read_text())
    review = (root / "pass2" / "review.yaml").read_text()
    considered = {tuple(sorted(json.loads(row.node_ids)))
                  for row in adjudication_run["second"].itertuples()}
    assert len(considered) == 3
    for members in considered:
        for node_id in members:
            assert node_id in review, f"{node_id} is missing from the human review file"
    # Only the collapse reaches compile; the other two are recorded, not applied.
    collapsing = [d for d in bundle["decisions"] if d["decision"] == "collapse"]
    assert len(collapsing) == 1


def test_the_shipped_criteria_still_refuse_to_guess_a_threshold():
    """The guard on the test above: its criteria file must not become the project's.

    If someone ever fills a magnitude into `config/criteria.v1.yaml` to make a
    pipeline collapse duplicates, this fails and says why.
    """
    from motifmultiverse.adjudicate import packaged_criteria_path
    from motifmultiverse.schema.criteria import CriterionStatus, load_criteria

    shipped = load_criteria(packaged_criteria_path())
    for criterion_id in ("TRUE_DUPLICATE", "FRAGMENT_MATCH"):
        criterion = shipped[criterion_id]
        assert criterion.status is CriterionStatus.CRITERION_NOT_YET_DEFINED, (
            f"{criterion_id} became FROZEN: the frozen design states no magnitude for it, "
            "so a threshold here was invented rather than derived"
        )
        assert criterion.predicates == ()
