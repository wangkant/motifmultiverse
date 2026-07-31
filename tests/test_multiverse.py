"""The specification multiverse: what it must refuse, and what it must not lose.

These are falsification tests before they are unit tests. A grid that runs is easy;
what makes one worth reading is that certain things are impossible in it, and each
of those has to be demonstrated failing rather than asserted in a docstring:

* two baseline populations are never averaged into one number;
* a planned cell that refused is still in the output;
* a hit table that is not the declared frozen run is refused, and so is a lexicon
  content hash that disagrees with the compiled lexicon's own manifest;
* a family that was not estimable in a cell never appears as a measured zero;
* the same design produces the same ids and the same numbers twice.

The fixtures are small and synthetic on purpose: the real-data audit lives in
`docs/K562_MULTIVERSE_AUDIT.md` and cannot be run here, but nothing below depends
on a scratch directory or on which machine it is.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from motifmultiverse import guards, multiverse
from motifmultiverse.multiverse import CellStatus

# --------------------------------------------------------------------------- #
# A frozen dataset small enough to reason about, in the two shapes the module
# reads: a hit table and peak sets.
# --------------------------------------------------------------------------- #
SUBSTRATE = "a" * 64
LEXICON = "lex_core"
FAMILIES = ("CTCF", "AP-1")


def _hit_rows(n_peaks: int = 90, substrate: str = SUBSTRATE, lexicon: str = LEXICON):
    """One row per (peak, family), with a deterministic coefficient.

    `used` throughout except for a block of `hit_below_floor` rows, which exist so
    that a family can be genuinely absent from a peak without that absence being
    representable as a zero anywhere in the pipeline.
    """
    rows = []
    for i in range(n_peaks):
        for family in FAMILIES:
            used = not (family == "CTCF" and i % 5 == 0)
            rows.append({
                "region_id": f"peak{i}",
                "chrom": "chr1" if i < n_peaks // 2 else "chr2",
                "start": 1000 + i * 10_000,
                "end": 1000 + i * 10_000 + 200,
                "missingness": "used" if used else "hit_below_floor",
                "input_scale": n_peaks,
                "lexicon_id": lexicon,
                "substrate_id": substrate,
                "variant_id": f"{family}_v1",
                "family_id": family,
                "hit_coefficient": (0.5 + 0.01 * i) if used else "",
            })
    return rows


def _write_hits(path: Path, rows) -> Path:
    columns = list(rows[0])
    lines = ["\t".join(columns)]
    lines += ["\t".join(str(row[c]) for c in columns) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path):
    """One frozen dataset, two peak sets, and a second lexicon over the same peaks."""
    _write_hits(tmp_path / "hits_core.tsv", _hit_rows())
    _write_hits(tmp_path / "hits_expanded.tsv",
                _hit_rows(substrate="b" * 64, lexicon="lex_expanded"))
    peaks = [f"peak{i}" for i in range(90)]
    (tmp_path / "query.txt").write_text("\n".join(peaks[:40]) + "\n")
    (tmp_path / "complement.txt").write_text("\n".join(peaks[40:]) + "\n")
    # A second baseline over the same universe: a genuinely different question.
    (tmp_path / "matched.txt").write_text("\n".join(peaks[40:70]) + "\n")
    return tmp_path


def _estimand(baseline: str, **over):
    base = {
        "query_id": "island",
        "query_regions": "query.txt",
        "baseline_id": baseline,
        "baseline_population_type": "COMPLEMENT_WITHIN_UNIVERSE",
        "baseline_construction_rule": "every universe peak not in the query",
        "baseline_regions": f"{baseline}.txt",
        "selection_provenance": "PROGRAMMATIC_RULE",
        "selection_rule": "leiden res0.5 == 5",
        "selection_feature_names": ("leiden_res0.5",),
    }
    return multiverse.Estimand(**{**base, **over})


def _measurement(mid="core", lexicon=LEXICON, substrate=SUBSTRATE, table="hits_core.tsv",
                 **over):
    base = {
        "measurement_id": mid, "lexicon_id": lexicon, "substrate_id": substrate,
        "hit_table": table, "lexicon_content_hash": f"hash_of_{lexicon}",
    }
    return multiverse.Measurement(**{**base, **over})


def _statistical(sid="pct", **over):
    # Blocks are genomic bins, so the block size has to be small relative to the
    # 900 kb the fixture peaks span or every peak lands in one block and the
    # health floor refuses before any effect -- which is a real refusal, and not
    # the one most of these tests are about.
    base = {"statistical_id": sid, "block_size": 50_000, "n_bootstrap": 40,
            "seed": 0, "floor_blocks": 2.0, "floor_coverage": 0.5,
            "floor_explained": 0.1}
    return multiverse.StatisticalChoice(**{**base, **over})


def _design(root, estimands=None, measurements=None, statistical=None):
    return multiverse.MultiverseDesign(
        multiverse_id="test_grid",
        estimands=tuple(estimands or [_estimand("complement")]),
        measurements=tuple(measurements or [_measurement()]),
        statistical_choices=tuple(statistical or [_statistical()]),
        root=Path(root),
    )


# --------------------------------------------------------------------------- #
# 1. Different baselines are not pooled.
# --------------------------------------------------------------------------- #
def test_two_baselines_produce_two_estimands_and_are_never_summarised_together(dataset,
                                                                              tmp_path):
    """The requirement this module exists for.

    Same query, same lexicon, same estimator; only the baseline population moves.
    A generic robustness score would report one number per family. Here the two
    baselines are two estimands, so each family gets two summaries that a reader
    can see disagree — or agree, which is then a statement about two questions
    rather than a wider interval on one.
    """
    design = _design(dataset, estimands=[
        _estimand("complement"),
        _estimand("matched", baseline_population_type="MATCHED_SUBSET",
                  baseline_construction_rule="30 peaks drawn from the complement"),
    ])
    result = multiverse.run_multiverse(design, tmp_path / "out")

    estimands = {s["estimand_id"] for s in result.summaries}
    assert len(estimands) == 2, "two baseline populations collapsed into one estimand"
    for summary in result.summaries:
        assert len({summary["estimand_id"]}) == 1
        # And the cells inside each summary all belong to that estimand.
        planned = {s["cell_id"]: s["estimand_id"]
                   for s in result.manifest["specifications"]}
        assert {planned[c] for c in summary["cell_ids"]} == {summary["estimand_id"]}

    families = [s["family_id"] for s in result.summaries]
    assert families.count("CTCF") == 2, (
        "one summary per family across both baselines is the pooled score this "
        "module refuses to produce"
    )


def test_a_summary_that_pools_two_baselines_is_refused_not_rendered(dataset, tmp_path):
    """Falsification: reach past the grouping and pool deliberately.

    `stability_within_estimand` is where the grouping is decided, so a test that
    only called it could never observe the guard doing anything. This one builds
    the pooled summary the guard exists to catch and asserts the guard refuses it
    against the real manifest.
    """
    design = _design(dataset, estimands=[_estimand("complement"), _estimand("matched")])
    manifest = multiverse.plan(design)
    specs = {s["cell_id"]: s for s in manifest["specifications"]}
    pooled = [{"group_key": "CTCF across all specifications",
               "cell_ids": [s["cell_id"] for s in manifest["specifications"]]}]

    result = guards.no_cross_estimand_pooling(pooled, specs)
    assert not result.passed
    assert "do not average" in result.detail


# --------------------------------------------------------------------------- #
# 2. No cell disappears.
# --------------------------------------------------------------------------- #
def test_every_planned_cell_appears_in_the_output_exactly_once(dataset, tmp_path):
    design = _design(
        dataset,
        estimands=[_estimand("complement"), _estimand("matched")],
        measurements=[_measurement(),
                      _measurement("expanded", "lex_expanded", "b" * 64,
                                   "hits_expanded.tsv")],
        statistical=[_statistical("pct"), _statistical("pct_small", block_size=25_000)],
    )
    out = tmp_path / "out"
    result = multiverse.run_multiverse(design, out)

    planned = [s["cell_id"] for s in result.manifest["specifications"]]
    assert len(planned) == 2 * 2 * 2
    assert sorted(c.cell_id for c in result.cells) == sorted(planned)

    rows = (out / "cells.tsv").read_text().strip().split("\n")
    assert len(rows) - 1 == len(planned), "cells.tsv lost a planned cell"
    assert sum(result.by_status.values()) == len(planned)


def test_a_refused_cell_keeps_its_row_and_its_reason(dataset, tmp_path):
    """A cell that cannot be estimated is a finding, not an absence.

    The floor is raised past what the fixture can meet, so the cell refuses on
    health. It must still be a row, with the reason, in both `cells.tsv` and
    `dropped_cells.tsv`.
    """
    design = _design(dataset, statistical=[
        _statistical("ok"),
        _statistical("impossible", floor_blocks=10_000.0),
    ])
    out = tmp_path / "out"
    result = multiverse.run_multiverse(design, out)

    statuses = {c.status for c in result.cells}
    assert CellStatus.SUCCESS in statuses
    dropped = [c for c in result.cells if c.status != CellStatus.SUCCESS]
    assert dropped, "the impossible floor produced no dropped cell"
    assert all(c.reason for c in dropped), "a dropped cell with no reason"

    text = (out / "dropped_cells.tsv").read_text()
    for cell in dropped:
        assert cell.cell_id in text


def test_a_cell_that_raises_unexpectedly_is_recorded_not_fatal(dataset, tmp_path,
                                                              monkeypatch):
    """A grid that stops at its first surprise reports a non-random subset of itself."""
    from motifmultiverse import interpret as interpret_mod

    real = interpret_mod.interpret_query
    calls = {"n": 0}

    def exploding(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("something nobody predicted")
        return real(*args, **kwargs)

    monkeypatch.setattr(multiverse.interpret, "interpret_query", exploding)
    design = _design(dataset, statistical=[_statistical("a"), _statistical("b", seed=1)])
    result = multiverse.run_multiverse(design, tmp_path / "out")

    assert len(result.cells) == 2
    assert result.by_status[CellStatus.ERROR] == 1
    assert result.by_status[CellStatus.SUCCESS] == 1
    errored = next(c for c in result.cells if c.status == CellStatus.ERROR)
    assert "something nobody predicted" in errored.reason
    assert (tmp_path / "out" / f"error_{errored.cell_id}.txt").exists()


# --------------------------------------------------------------------------- #
# 3. Mixed substrates and wrong lexicon hashes are refused.
# --------------------------------------------------------------------------- #
def test_a_hit_table_spanning_two_substrates_is_refused(dataset, tmp_path):
    """Rows from two frozen hit callers do not make one measurement.

    The refusal comes from `interpret.read_hit_table`, which the cell reaches
    before any declaration is compared -- this asserts the outcome the requirement
    asks for (the cell refuses, and says why) rather than which layer produced it.
    A second mixed-substrate check inside `multiverse` would be unreachable.
    """
    mixed = _hit_rows(20) + _hit_rows(20, substrate="c" * 64)
    _write_hits(dataset / "mixed.tsv", mixed)
    design = _design(dataset, measurements=[_measurement(table="mixed.tsv")])

    result = multiverse.run_multiverse(design, tmp_path / "out")
    assert [c.status for c in result.cells] == [CellStatus.REFUSED_SCHEMA]
    assert "mixes substrates" in result.cells[0].reason
    assert result.cells[0].cell_id in (tmp_path / "out" / "dropped_cells.tsv").read_text()


def test_a_hit_table_that_is_not_the_declared_frozen_run_is_refused(dataset, tmp_path):
    design = _design(dataset, measurements=[
        _measurement(substrate="d" * 64)])           # declared != what the table holds
    result = multiverse.run_multiverse(design, tmp_path / "out")
    assert [c.status for c in result.cells] == [CellStatus.REFUSED_SCHEMA]
    assert "declared substrate_id" in result.cells[0].reason


def test_a_lexicon_content_hash_that_disagrees_with_its_manifest_is_refused(dataset,
                                                                           tmp_path):
    """The declared hash is checked against the compiled lexicon's own record.

    Without the manifest there is nothing for the declaration to be wrong against;
    with it, pointing a design at a lexicon that was recompiled under the same name
    is caught before any effect is computed.
    """
    (dataset / "core.manifest.json").write_text(
        json.dumps({"lexicon_content_hash": "the_real_hash"}), encoding="utf-8")
    design = _design(dataset, measurements=[
        _measurement(lexicon_manifest="core.manifest.json")])

    result = multiverse.run_multiverse(design, tmp_path / "out")
    assert [c.status for c in result.cells] == [CellStatus.REFUSED_SCHEMA]
    assert "the_real_hash" in result.cells[0].reason


def test_a_matching_lexicon_content_hash_passes(dataset, tmp_path):
    (dataset / "core.manifest.json").write_text(
        json.dumps({"lexicon_content_hash": f"hash_of_{LEXICON}"}), encoding="utf-8")
    design = _design(dataset, measurements=[
        _measurement(lexicon_manifest="core.manifest.json")])
    result = multiverse.run_multiverse(design, tmp_path / "out")
    assert [c.status for c in result.cells] == [CellStatus.SUCCESS]


# --------------------------------------------------------------------------- #
# 4. Missing never becomes zero.
# --------------------------------------------------------------------------- #
def test_a_family_with_no_estimate_is_marked_not_measured_as_zero(dataset, tmp_path):
    """The founding failure, in the shape this module could commit it.

    A family present in one cell of an estimand and absent from another must not
    be counted as an effect of 0.0 in the summary: the counts are over cells that
    produced an estimate, and the range fields say `NOT_ESTIMABLE` rather than
    reporting a min of 0.
    """
    summaries = multiverse.stability_within_estimand(
        [multiverse.CellResult(cell_id="c1", estimand_id="e1",
                               status=CellStatus.SUCCESS,
                               effects=[{"family_id": "CTCF", "effect": 0.4}]),
         multiverse.CellResult(cell_id="c2", estimand_id="e1",
                               status=CellStatus.NOT_ESTIMABLE, reason="floor")],
        {"specifications": [{"cell_id": "c1", "estimand_id": "e1"},
                            {"cell_id": "c2", "estimand_id": "e1"}]},
    )
    ctcf = next(s for s in summaries if s["family_id"] == "CTCF")
    assert ctcf["n_cells_with_estimate"] == 1
    assert ctcf["n_cells_planned_in_estimand"] == 2
    assert ctcf["effect_min"] == ctcf["effect_max"] == 0.4, (
        "the non-estimable cell was folded in as a zero"
    )


def test_a_family_estimable_nowhere_reports_not_estimable_not_zero():
    summaries = multiverse.stability_within_estimand(
        [multiverse.CellResult(cell_id="c1", estimand_id="e1",
                               status=CellStatus.SUCCESS,
                               effects=[{"family_id": "CTCF", "effect": None}])],
        {"specifications": [{"cell_id": "c1", "estimand_id": "e1"}]},
    )
    ctcf = next(s for s in summaries if s["family_id"] == "CTCF")
    assert ctcf["effect_min"] == multiverse.NOT_ESTIMABLE_MARKER
    assert ctcf["effect_median"] == multiverse.NOT_ESTIMABLE_MARKER
    assert ctcf["n_cells_with_estimate"] == 0
    assert ctcf["sign_agreement"] == multiverse.NOT_ESTIMABLE_MARKER


# --------------------------------------------------------------------------- #
# 5. Determinism.
# --------------------------------------------------------------------------- #
def test_the_same_design_produces_the_same_ids_and_the_same_numbers(dataset, tmp_path):
    design = _design(dataset, estimands=[_estimand("complement"), _estimand("matched")])
    first = multiverse.run_multiverse(design, tmp_path / "a")
    second = multiverse.run_multiverse(design, tmp_path / "b")

    assert [c.cell_id for c in first.cells] == [c.cell_id for c in second.cells]
    assert [c.status for c in first.cells] == [c.status for c in second.cells]
    assert first.summaries == second.summaries
    assert ((tmp_path / "a" / "family_effects.tsv").read_text()
            == (tmp_path / "b" / "family_effects.tsv").read_text())


def test_cell_ids_do_not_depend_on_dict_order(dataset):
    """The id is over canonical JSON, so a reordered declaration is the same cell."""
    a = multiverse.Measurement(measurement_id="m", lexicon_id="l", substrate_id="s",
                               hit_table="h.tsv", lexicon_content_hash="x")
    b = multiverse.Measurement(lexicon_content_hash="x", hit_table="h.tsv",
                               substrate_id="s", lexicon_id="l", measurement_id="m")
    spec_a = multiverse.Specification(_estimand("complement"), a, _statistical())
    spec_b = multiverse.Specification(_estimand("complement"), b, _statistical())
    assert spec_a.cell_id == spec_b.cell_id


def test_a_changed_axis_changes_the_cell_id_but_not_the_estimand_id():
    """Statistical choices are not the question. The ids have to say so.

    If changing the estimator changed the estimand id, every statistical
    robustness check would look like a different scientific question and the
    within-estimand summaries would each hold one cell — which is how a
    multiverse silently stops comparing anything.
    """
    estimand = _estimand("complement")
    base = multiverse.Specification(estimand, _measurement(), _statistical())
    other = multiverse.Specification(
        estimand, _measurement(),
        _statistical(estimator=multiverse.interpret.ESTIMATOR_BCA_WILD))
    assert base.cell_id != other.cell_id
    assert base.estimand_id == other.estimand_id


# --------------------------------------------------------------------------- #
# The declared grid, and what a design may not silently omit.
# --------------------------------------------------------------------------- #
def test_an_estimand_without_a_baseline_construction_rule_is_refused():
    with pytest.raises(multiverse.MultiverseError, match="baseline_construction_rule"):
        _estimand("complement", baseline_construction_rule="")


def test_a_design_with_an_unknown_field_is_refused_not_partly_applied(tmp_path):
    """A silently ignored axis is a specification nobody ran."""
    design = tmp_path / "design.json"
    design.write_text(json.dumps({
        "multiverse_id": "x",
        "estimands": [{**{k: "v" for k in (
            "query_id", "query_regions", "baseline_id", "baseline_population_type",
            "baseline_construction_rule", "baseline_regions", "selection_rule")},
            "selection_provenance": "PROGRAMMATIC_RULE",
            "stratify_by": "gc_content"}],
        "measurements": [], "statistical_choices": [],
    }))
    with pytest.raises(multiverse.MultiverseError, match="stratify_by"):
        multiverse.read_design(design)


def test_the_manifest_is_written_before_the_grid_runs(dataset, tmp_path, monkeypatch):
    """A manifest written afterwards can only hold the cells that survived.

    Which is the question a reader of a multiverse most needs answered: whether
    the reported cells are the planned cells or a selection from them.
    """
    out = tmp_path / "out"
    seen = {}

    real = multiverse.interpret.interpret_query

    def note(*args, **kwargs):
        seen["manifest_existed"] = (out / "specification_manifest.json").exists()
        raise RuntimeError("stop here")

    monkeypatch.setattr(multiverse.interpret, "interpret_query", note)
    multiverse.run_multiverse(_design(dataset), out)
    assert seen["manifest_existed"], "the manifest was not on disk before the first cell"
    assert real is not None


def test_the_module_computes_no_statistics_of_its_own():
    """Requirement 6, checked structurally rather than promised.

    A second implementation of a block bootstrap that agrees with the first 99% of
    the time is worse than none: the 1% arrives as a number nobody can trace. The
    only arithmetic this module is allowed is over already-estimated effects —
    min, max, median, and counting signs — so the names below must not appear.
    """
    import ast
    import inspect

    source = inspect.getsource(multiverse)
    tree = ast.parse(source)
    forbidden = {"bootstrap", "resample", "percentile", "quantile", "bca",
                 "confidence_interval", "standard_error"}
    offenders = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(word in node.name.lower() for word in forbidden)
    ]
    assert not offenders, f"multiverse defines its own statistics: {offenders}"
    assert "interpret_query" in source, "the module must call interpret, not reimplement it"


def test_the_design_documented_here_parses(tmp_path):
    """The example in `docs/MULTIVERSE_DESIGN.md` is built, not just displayed.

    A documented JSON block that nothing reads goes stale the first time a field
    is renamed, and it goes stale invisibly: the document still looks right. This
    reads the block out of the document and builds a design from it, so a renamed
    or removed field fails here and names the file to fix.
    """
    doc = Path(__file__).resolve().parent.parent / "docs" / "MULTIVERSE_DESIGN.md"
    text = doc.read_text(encoding="utf-8")
    block = text.split("<!-- example-design:begin -->")[1].split("<!-- example-design:end -->")[0]
    payload = block.split("```json")[1].split("```")[0]
    path = tmp_path / "design.json"
    path.write_text(payload, encoding="utf-8")

    design = multiverse.read_design(path)
    assert len(design.specifications()) == 2 * 1 * 2, (
        "the documented grid is not the product of its axes"
    )
    assert design.preregistered_threshold == multiverse.NO_PREREGISTERED_THRESHOLD
    # Two baselines means two estimands, which is the point of the example.
    assert len({e.estimand_id for e in design.estimands}) == 2
