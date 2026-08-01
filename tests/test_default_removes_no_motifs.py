"""The one property a default run must have: **it removes no motifs.**

This file pins the property, not the file name. `CRITERIA_RESOURCE` may be
whatever a maintainer decides it should be; what a maintainer may not do is
change it to a criterion that deletes without noticing that is what they did.
Nothing here asserts `criteria.v1.yaml` -- swapping in a different non-deleting
registry leaves every assertion below green, and swapping in a deleting one
fails all of them.

WHY THIS DIRECTION AND NOT THE OTHER. The two failure directions are not
symmetric for a compiled lexicon. An under-deduplicated lexicon carries a
duplicate: the reader can see both entries, compare them, and merge them
downstream. An over-deduplicated one has lost a motif and does not record which
one -- the surviving medoid does not say what it stands for in any way the reader
can invert. Deletion is therefore the direction that must be opted into, and a
tool whose *default* deletes has made that choice on the reader's behalf.

The v2 `TRUE_DUPLICATE` heuristic is exactly such a criterion. It still ships,
still loads, and is still reachable through `--criteria` and
`packaged_v2_criteria_path()`; on the K562 run one of the four collapses it
licenses merges CTCF with its paralog CTCFL, and on its own preregistered
held-out set it fired two of its own falsifiers
(`docs/MERGE_CRITERION_PREREGISTRATION.md`). Available and asked for -- not
administered by default.

Four tests, in increasing strength:

1. a TOTAL property of the default registry, over every possible input;
2. that the deleting criterion is still shipped and still reachable, so
   "removes no motifs" cannot be satisfied by dropping the capability;
3. a POSITIVE CONTROL that the fixture below is one a deleting criterion really
   does delete from, so the default's restraint cannot be mistaken for a fixture
   that never had anything to collapse;
4. the same property through the real CLI, end to end -- with the compiled
   lexicon checked to still name every motif the registry ingested, by node_id
   and not merely by count.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from motifmultiverse.adjudicate import (
    packaged_criteria_path,
    packaged_v2_criteria_path,
)
from motifmultiverse.cli import main
from motifmultiverse.schema import Decision
from motifmultiverse.schema.criteria import EVALUABLE_STATUSES, load_criteria

TIERS = ("core", "expanded", "sensitivity")


# --------------------------------------------------------------------------- #
# 1. The total property: no input can make the default delete.
# --------------------------------------------------------------------------- #

def test_no_criterion_the_default_registry_can_evaluate_decides_collapse():
    """Over every possible evidence bundle, not just the fixture below.

    `evaluate_criterion` can only return `decision_if_matched` for a criterion
    whose status is evaluable; a `CRITERION_NOT_YET_DEFINED` one always DEFERS.
    So "no evaluable criterion in the default registry decides COLLAPSE" is the
    whole invariant, quantified over all inputs, and it is one line to check.

    If this fails, someone made a deleting criterion the default. That may be the
    right call one day -- it is a scientific decision, not a code-quality one --
    and the way to make it is to change this test deliberately, with the reasons
    written down, rather than to discover the change from a shrunken lexicon.
    """
    default = load_criteria(packaged_criteria_path())
    deleting = {
        criterion_id: criterion.status.value
        for criterion_id, criterion in default.items()
        if criterion.status in EVALUABLE_STATUSES
        and criterion.decision_if_matched is Decision.COLLAPSE
    }
    assert not deleting, (
        f"the DEFAULT criterion registry can collapse: {deleting}. A run with no "
        "--criteria would delete motifs, and an over-deduplicated lexicon does not "
        "record what it lost. Ship the deleting criterion behind --criteria "
        "instead, the way criteria.v2.yaml is shipped."
    )


def test_the_deleting_criterion_is_still_shipped_and_still_reachable():
    """The other half: refusing to delete by default is not refusing to delete.

    A reader who wants the preregistered heuristic must be able to get it from an
    installed wheel, and get exactly it -- same file, same digest, same meaning.
    """
    v2 = load_criteria(packaged_v2_criteria_path())
    assert v2["TRUE_DUPLICATE"].decision_if_matched is Decision.COLLAPSE
    assert v2["TRUE_DUPLICATE"].status in EVALUABLE_STATUSES
    assert packaged_v2_criteria_path().is_file()
    assert packaged_v2_criteria_path() != packaged_criteria_path(), (
        "the deleting criterion is the default again"
    )


# --------------------------------------------------------------------------- #
# 2 and 3. The same property through the real CLI, on a fixture built to be
#          deleted from.
# --------------------------------------------------------------------------- #

def _ppm(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = np.full((length, 4), 0.05)
    for position in range(length):
        matrix[position, rng.integers(0, 4)] = 0.85
    return matrix / matrix.sum(axis=1, keepdims=True)


def _write_project(root: Path) -> Path:
    """Two 20bp patterns with an IDENTICAL core and different seqlet counts.

    Every gate a deleting criterion could apply is cleared on purpose: bilateral
    overlap 1.0, `overlap_bp` 20, `ppm_similarity` 1.0, `signed_cwm_similarity`
    1.0, and (at `--null-shuffles 10`) an `empirical_p_value` of 1/11, which is
    the null floor. This is a duplicate by any reading.

    The seqlet counts differ (240 vs 120) for a mechanical reason: medoid
    selection needs one authoritative field to order the two members by, and with
    identical cores in identical windows `motif_completeness` and `core_ic` tie.
    Without the tie-break the component defers for want of tie metadata, which
    would make the fixture defer under EVERY criterion and this whole file
    vacuous.
    """
    base = _ppm(20, 11)
    contrib = (base - 0.25) * 2.0
    modisco = root / "modisco.h5"
    with h5py.File(modisco, "w") as h5:
        group = h5.require_group("pos_patterns")
        for name, n_seqlets in (("pattern_0", 240), ("pattern_1", 120)):
            pattern = group.create_group(name)
            pattern.create_dataset("sequence", data=base)
            pattern.create_dataset("contrib_scores", data=contrib)
            pattern.create_dataset("hypothetical_contribs", data=contrib)
            pattern.create_group("seqlets").create_dataset("n_seqlets", data=n_seqlets)

    project = root / "project.json"
    project.write_text(json.dumps({
        "project": "default-removes-no-motifs", "peak_universe_id": "u1",
        "analyses": [{"id": "a1", "model": "m1", "readout": "counts", "union_id": "MA",
                      "context": "promoter", "modisco_h5": str(modisco)}],
    }), encoding="utf-8")
    return project


@pytest.fixture(scope="module")
def duplicate_run(tmp_path_factory) -> dict:
    """ingest -> align -> annotate, then adjudicate TWICE off the same evidence.

    Once with no `--criteria` (the default) and once pinned to v2, so the two
    differ in exactly one thing: which criterion registry was loaded. Each is
    compiled separately.
    """
    root = tmp_path_factory.mktemp("default-removes-no-motifs")
    project = _write_project(root)

    assert main(["ingest", str(project), "--out", str(root / "registry")]) == 0
    assert main(["align", str(root / "registry"), "--out", str(root / "evidence"),
                 "--null-shuffles", "10", "--seed", "3"]) == 0
    assert main(["annotate", str(root / "evidence"), "--registry", str(root / "registry"),
                 "--out", str(root / "evidence")]) == 0

    for label, criteria_args in (
        ("default", []),
        ("v2", ["--criteria", str(packaged_v2_criteria_path())]),
    ):
        assert main(["adjudicate", str(root / "evidence"),
                     "--registry", str(root / "registry"),
                     *criteria_args,
                     "--out", str(root / label),
                     "--review", str(root / label / "review.yaml")]) == 0
        assert main(["compile", str(root / "registry"),
                     "--decisions", str(root / label / "merge_decisions.json"),
                     "--out", str(root / f"lexicons-{label}")]) == 0

    registry = json.loads((root / "registry" / "registry.json").read_text())
    return {
        "root": root,
        "registry_node_ids": {node["node_id"] for node in registry["nodes"]},
        "default": pd.read_parquet(root / "default" / "ontology_decisions.parquet"),
        "v2": pd.read_parquet(root / "v2" / "ontology_decisions.parquet"),
    }


def test_the_fixture_is_one_a_deleting_criterion_really_does_delete_from(duplicate_run):
    """POSITIVE CONTROL. Without it the test below passes on any fixture at all.

    A pair no criterion would ever collapse proves nothing about restraint, and
    "the default removed no motifs" would be true of an empty registry. Pinned to
    v2 the same evidence collapses and the compiled lexicon comes back one motif
    short -- so the restraint asserted next is restraint, not absence of an
    opportunity.
    """
    assert len(duplicate_run["registry_node_ids"]) == 2

    v2 = duplicate_run["v2"]
    assert list(v2["decision"]) == ["collapse"]
    assert v2["representative_node_id"].iat[0] in duplicate_run["registry_node_ids"]

    manifest = json.loads(
        (duplicate_run["root"] / "lexicons-v2" / "core.manifest.json").read_text())
    assert manifest["n_motifs"] == 1, (
        "the fixture stopped being a duplicate under v2; rebuild it before "
        "reading anything into the default's behaviour"
    )


def test_a_default_run_removes_no_motifs(duplicate_run):
    """THE INVARIANT. Same registry, same evidence, no `--criteria`: nothing goes.

    The pair above is a textbook duplicate and the default says so by DEFERRING
    it -- recorded, visible in `ontology_decisions.parquet`, and available to a
    curator -- rather than by deleting one of the two. Every tier of the compiled
    lexicon still carries both motifs, by node_id and not merely by count.
    """
    default = duplicate_run["default"]
    assert Decision.COLLAPSE.value not in set(default["decision"]), (
        "a run with no --criteria collapsed a component; deletion is the one "
        "error a reader of the lexicon cannot undo, so it must be opted into"
    )
    assert default["representative_node_id"].isna().all(), (
        "a default run elected a survivor, which only a collapse needs"
    )

    bundle = json.loads(
        (duplicate_run["root"] / "default" / "merge_decisions.json").read_text())
    assert not [d for d in bundle["decisions"] if d["decision"] == "collapse"], (
        "a collapse reached the compile handoff"
    )

    for tier in TIERS:
        manifest = json.loads(
            (duplicate_run["root"] / "lexicons-default" / f"{tier}.manifest.json").read_text())
        assert set(manifest["node_ids"]) == duplicate_run["registry_node_ids"], (
            f"the {tier} lexicon does not carry every ingested motif; a default "
            f"run lost {duplicate_run['registry_node_ids'] - set(manifest['node_ids'])}"
        )
        assert manifest["n_motifs"] == len(duplicate_run["registry_node_ids"])
