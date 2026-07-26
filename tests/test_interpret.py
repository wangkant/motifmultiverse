"""interpret tests: the dispatch, the three health numbers, and the suppression rule.

Much of the behaviour under test is *refusal*, so most of these assert that a
number was NOT produced. That is the point of the module: what a peak set is
allowed to say depends on how it was chosen, and a query that cannot support an
interval must not emit one with a caveat attached.
"""
from __future__ import annotations

import json

import pytest

from motifmultiverse import interpret
from motifmultiverse.schema import (
    HIT_TABLE_COLUMNS,
    HealthFloors,
    HitRecord,
    Missingness,
    OutputMode,
    PeakSetQuery,
    SchemaError,
    SelectionProvenance,
)

BLOCK = 1_000_000
LEXICON = "lex_core_v1"
SCALE = 33917

N_BLOCKS = 42          # divisible by 3 and by 2, so the planted effects are exact
PER_BLOCK = 4

NOT_SEARCHED_PEAK = "r000_notsearched"
NO_MATCH_PEAK = "r001_nomatch"


def _rows() -> list[HitRecord]:
    """A deterministic substrate: 42 blocks x 4 peaks, two families, all four states.

    Query peaks are the even index within a block, comparator peaks the odd one.
    ``FAM_A`` carries a planted effect of exactly +0.9 and ``FAM_B`` a planted null
    of exactly 0. Both vary **between blocks**, which is what gives the block
    bootstrap something to resample: a fixture that is identical in every block
    produces a degenerate interval and would not exercise the estimator at all.
    Two extra peaks exercise the non-``USED`` states and belong to neither side.
    """
    rows: list[HitRecord] = []
    for b in range(N_BLOCKS):
        for i in range(PER_BLOCK):
            start = b * BLOCK + i * 1000
            is_query = i % 2 == 0
            common = dict(region_id=f"r{b:03d}_{i}", chrom="chr1", start=start, end=start + 500,
                          input_scale=SCALE, lexicon_id=LEXICON)
            rows.append(HitRecord(
                missingness=Missingness.USED, variant_id=f"UA_FAMA_{i:02d}", family_id="FAM_A",
                hit_coefficient=1.0 + (b % 3) * 0.3 if is_query else 0.4, **common))
            rows.append(HitRecord(
                missingness=Missingness.USED, variant_id=f"UA_FAMB_{i:02d}", family_id="FAM_B",
                hit_coefficient=0.5 + ((b if is_query else b + 1) % 3) * 0.2, **common))
    rows.append(HitRecord(region_id=NOT_SEARCHED_PEAK, chrom="chr1", start=900_000, end=900_500,
                          missingness=Missingness.NOT_SEARCHED,
                          input_scale=SCALE, lexicon_id=LEXICON))
    rows.append(HitRecord(region_id=NO_MATCH_PEAK, chrom="chr1", start=1_900_000, end=1_900_500,
                          missingness=Missingness.NO_SEQUENCE_MATCH,
                          input_scale=SCALE, lexicon_id=LEXICON))
    return rows


def _ids(parity: int, blocks: range | None = None) -> list[str]:
    blocks = blocks if blocks is not None else range(N_BLOCKS)
    return [f"r{b:03d}_{i}" for b in blocks for i in range(PER_BLOCK) if i % 2 == parity]


def _query(**over) -> PeakSetQuery:
    kw = dict(query_id="q1", region_ids=_ids(0),
              selection_provenance=SelectionProvenance.EXTERNAL,
              comparator_id="odd_peaks", comparator_region_ids=_ids(1))
    kw.update(over)
    return PeakSetQuery(**kw)


# ------------------------------------------------------------------ substrate
def test_universe_includes_peaks_that_produced_no_hit():
    """The peak whose only row is NO_SEQUENCE_MATCH is in the universe, at 0 families.

    Deriving the universe from called hits alone would drop it and inflate every
    ratio computed against it.
    """
    peaks = interpret.peak_universe(_rows(), BLOCK)
    assert NO_MATCH_PEAK in peaks
    assert peaks[NO_MATCH_PEAK].searched is True
    assert peaks[NO_MATCH_PEAK].family_hit_count == {}
    assert peaks[NO_MATCH_PEAK].family_coefficient_sum == {}
    assert peaks[NO_MATCH_PEAK].family_abs_coefficient_sum == {}


def test_not_searched_peak_is_excluded_from_denominators_rather_than_counted_as_zero():
    peaks = interpret.peak_universe(_rows(), BLOCK)
    assert peaks[NOT_SEARCHED_PEAK].searched is False
    health = interpret.health_report(
        peaks, [NOT_SEARCHED_PEAK, NO_MATCH_PEAK], HealthFloors(), BLOCK)
    assert health.n_in_universe == 2
    assert health.n_searched == 1          # the NOT_SEARCHED peak is not a zero
    assert health.n_with_used_hit == 0
    assert health.explained_fraction == 0.0


def test_hit_record_refuses_a_zero_for_an_undefined_value():
    with pytest.raises(SchemaError, match="coercion"):
        HitRecord(region_id="r", chrom="chr1", start=0, end=10,
                  missingness=Missingness.NO_SEQUENCE_MATCH,
                  input_scale=SCALE, lexicon_id=LEXICON, hit_coefficient=0.0)


def test_one_region_cannot_mix_used_and_not_searched():
    rows = [
        HitRecord(region_id="r", chrom="chr1", start=0, end=10,
                  missingness=Missingness.NOT_SEARCHED,
                  input_scale=SCALE, lexicon_id=LEXICON),
        HitRecord(region_id="r", chrom="chr1", start=0, end=10,
                  missingness=Missingness.USED, variant_id="UA_FAMA_01",
                  family_id="FAM_A", hit_coefficient=1.0,
                  input_scale=SCALE, lexicon_id=LEXICON),
    ]
    with pytest.raises(interpret.InterpretError, match="mixes NOT_SEARCHED and measured rows"):
        interpret.peak_universe(rows, BLOCK)


def test_one_region_cannot_change_coordinates():
    rows = [
        HitRecord(region_id="r", chrom="chr1", start=0, end=10,
                  missingness=Missingness.NO_SEQUENCE_MATCH,
                  input_scale=SCALE, lexicon_id=LEXICON),
        HitRecord(region_id="r", chrom="chr1", start=1, end=11,
                  missingness=Missingness.NO_SEQUENCE_MATCH,
                  input_scale=SCALE, lexicon_id=LEXICON),
    ]
    with pytest.raises(interpret.InterpretError, match="inconsistent coordinates"):
        interpret.peak_universe(rows, BLOCK)


def test_opposite_hits_cancel_mass_but_not_occupancy():
    rows = [
        HitRecord(region_id="r", chrom="chr1", start=0, end=10,
                  missingness=Missingness.USED, variant_id="UA_FAMA_01",
                  family_id="FAM_A", hit_coefficient=1.0,
                  input_scale=SCALE, lexicon_id=LEXICON),
        HitRecord(region_id="r", chrom="chr1", start=0, end=10,
                  missingness=Missingness.USED, variant_id="UA_FAMA_02",
                  family_id="FAM_A", hit_coefficient=-1.0,
                  input_scale=SCALE, lexicon_id=LEXICON),
    ]
    peak = interpret.peak_universe(rows, BLOCK)["r"]
    assert peak.family_hit_count["FAM_A"] == 2
    assert peak.family_coefficient_sum["FAM_A"] == 0.0
    assert peak.family_abs_coefficient_sum["FAM_A"] == 2.0
    assert peak.has_used_hit


def test_compose_counts_occupancy_not_cancelled_mass():
    """A regression guard on ``compose()``'s call path, not just ``Peak`` itself.

    ``compose()`` used to derive ``n_peaks_with_family`` from a nonzero coefficient
    sum, which undercounts exactly when a family's coefficients cancel at a peak.
    It must read occupancy (``family_hit_count``) instead, independently of the
    mean signed coefficient, which is still allowed to be 0.0.
    """
    rows = [
        HitRecord(region_id="r", chrom="chr1", start=0, end=10,
                  missingness=Missingness.USED, variant_id="UA_FAMA_01",
                  family_id="FAM_A", hit_coefficient=1.0,
                  input_scale=SCALE, lexicon_id=LEXICON),
        HitRecord(region_id="r", chrom="chr1", start=0, end=10,
                  missingness=Missingness.USED, variant_id="UA_FAMA_02",
                  family_id="FAM_A", hit_coefficient=-1.0,
                  input_scale=SCALE, lexicon_id=LEXICON),
    ]
    peaks = interpret.peak_universe(rows, BLOCK)
    composition = interpret.compose(peaks, ["r"])
    fam_a = next(c for c in composition if c.family_id == "FAM_A")
    assert fam_a.n_peaks_with_family == 1
    assert fam_a.peak_share == 1.0
    assert fam_a.mean_coefficient_per_peak == 0.0


# --------------------------------------------------------------------- health
def test_health_numbers_carry_their_denominators():
    peaks = interpret.peak_universe(_rows(), BLOCK)
    ids = _ids(0) + ["not_in_the_universe_1", "not_in_the_universe_2"]
    h = interpret.health_report(peaks, ids, HealthFloors(), BLOCK)
    assert h.n_submitted == len(ids)
    assert h.n_in_universe == len(_ids(0))
    assert h.intersection_coverage == pytest.approx(len(_ids(0)) / len(ids))
    assert h.n_blocks == N_BLOCKS
    assert h.explained_fraction == pytest.approx(h.n_with_used_hit / h.n_searched)


def test_not_searched_blocks_do_not_satisfy_min_blocks():
    peaks = interpret.peak_universe(_rows(), BLOCK)
    ids = [NOT_SEARCHED_PEAK, NO_MATCH_PEAK]
    health = interpret.health_report(
        peaks, ids, HealthFloors(min_blocks=2, min_intersection_coverage=0.0,
                                 min_explained_fraction=0.0), BLOCK)
    assert health.n_blocks == 1
    assert any("n_blocks=1" in failure for failure in health.floor_failures)


def test_a_floor_failure_suppresses_the_reading_instead_of_annotating_it():
    result = interpret.interpret_query(_rows(), _query(region_ids=_ids(0)[:4]), n_bootstrap=50)
    assert result.floor_failures                      # too few blocks
    assert result.composition is None and result.effects is None
    assert result.interpretation_emitted is False
    assert "suppressed" in (result.suppression_reason or "")


def test_bad_comparator_health_suppresses_effects():
    q = _query(comparator_region_ids=_ids(1)[:2])
    result = interpret.interpret_query(_rows(), q, n_bootstrap=20)
    assert result.effects is None
    assert any("comparator" in failure for failure in result.floor_failures)


def test_health_is_emitted_first_even_when_everything_passes():
    result = interpret.interpret_query(_rows(), _query(), n_bootstrap=50)
    assert result.emitted_order == ["health", "composition", "effects"]


# ------------------------------------------------------------------- dispatch
@pytest.mark.parametrize("grade,mode", [
    (SelectionProvenance.EXTERNAL, OutputMode.FULL_INFERENCE),
    (SelectionProvenance.PROGRAMMATIC_RULE, OutputMode.FULL_INFERENCE),
    (SelectionProvenance.CLUSTERED_WITH_SPLIT, OutputMode.FULL_INFERENCE_HELD_OUT),
    (SelectionProvenance.CLUSTERED_NO_SPLIT, OutputMode.DESCRIPTIVE_ONLY),
    (SelectionProvenance.EYEBALLED, OutputMode.DESCRIPTIVE_ONLY),
    (SelectionProvenance.MODEL_SELECTED_NO_TRANSCRIPT,
     OutputMode.DESCRIPTIVE_ONLY_UNVERIFIABLE_CONDITIONING),
])
def test_every_grade_dispatches_to_its_documented_mode(grade, mode):
    extra = ({"selection_rule": "distance_to_tss < 2000"}
             if grade is SelectionProvenance.PROGRAMMATIC_RULE else {})
    assert _query(selection_provenance=grade, **extra).output_mode is mode


def test_an_undeclared_grade_takes_the_most_conservative_mode_not_the_most_permissive():
    q = PeakSetQuery(query_id="q", region_ids=_ids(0))
    assert q.selection_provenance is SelectionProvenance.DECLARATION_MISSING
    assert q.output_mode is OutputMode.DESCRIPTIVE_ONLY_UNVERIFIABLE_CONDITIONING
    assert q.output_mode is not OutputMode.FULL_INFERENCE


def test_an_unrecognised_grade_degrades_rather_than_raising_or_upgrading():
    q = PeakSetQuery(query_id="q", region_ids=_ids(0), selection_provenance="FROM_A_LATER_LEDGER")
    assert q.selection_provenance is SelectionProvenance.DECLARATION_MISSING
    assert q.output_mode is OutputMode.DESCRIPTIVE_ONLY_UNVERIFIABLE_CONDITIONING


def test_programmatic_rule_without_the_rule_is_refused():
    with pytest.raises(SchemaError, match="selection_rule"):
        PeakSetQuery(query_id="q", region_ids=_ids(0),
                     selection_provenance=SelectionProvenance.PROGRAMMATIC_RULE)


@pytest.mark.parametrize("grade", [
    SelectionProvenance.CLUSTERED_NO_SPLIT,
    SelectionProvenance.EYEBALLED,
    SelectionProvenance.MODEL_SELECTED_NO_TRANSCRIPT,
])
def test_descriptive_modes_emit_no_interval_and_no_p_value(grade):
    result = interpret.interpret_query(_rows(), _query(selection_provenance=grade), n_bootstrap=50)
    assert result.composition and result.effects is None
    assert "effects" not in result.emitted_order


def test_model_selected_says_its_conditioning_cannot_be_verified():
    result = interpret.interpret_query(
        _rows(), _query(selection_provenance=SelectionProvenance.MODEL_SELECTED_NO_TRANSCRIPT),
        n_bootstrap=50)
    assert any("cannot be verified" in n for n in result.notes)


def test_undeclared_query_is_told_apart_from_external_in_its_own_record():
    result = interpret.interpret_query(
        _rows(), PeakSetQuery(query_id="q", region_ids=_ids(0)), n_bootstrap=50)
    assert result.selection_provenance == "DECLARATION_MISSING"
    assert any("not the same as EXTERNAL" in n for n in result.notes)


# ------------------------------------------------------------------ inference
def test_full_inference_recovers_the_planted_effect_and_the_planted_null():
    result = interpret.interpret_query(_rows(), _query(), n_bootstrap=200, seed=7)
    by_family = {e["family_id"]: e for e in result.effects}
    assert by_family["FAM_A"]["effect"] == pytest.approx(0.9, abs=1e-9)
    assert by_family["FAM_B"]["effect"] == pytest.approx(0.0, abs=1e-9)
    lo, hi = by_family["FAM_A"]["ci"]
    assert lo > 0.0 and hi > lo                       # planted effect, interval excludes zero
    lo_b, hi_b = by_family["FAM_B"]["ci"]
    assert lo_b < 0.0 < hi_b                          # planted null, interval straddles zero


def test_every_effect_carries_block_size_replicates_and_seed():
    """FP-15: block size, B and the seed are saved beside the interval, not in a log."""
    result = interpret.interpret_query(_rows(), _query(), n_bootstrap=64, seed=3)
    for e in result.effects:
        assert e["block_size"] == BLOCK and e["n_bootstrap"] == 64 and e["random_seed"] == 3
        assert e["estimator"] == interpret.ESTIMATOR
        assert e["comparator_id"] == "odd_peaks"


def test_percentile_bootstrap_does_not_emit_p_or_q_values():
    result = interpret.interpret_query(_rows(), _query(), n_bootstrap=50, seed=1)
    for effect in result.effects:
        assert effect["estimator"] == "percentile_block_bootstrap"
        assert effect["inference_capability"] == "ESTIMATION_ONLY"
        assert effect["p_value"] is None
        assert effect["q_value"] is None


def test_estimation_only_note_appears_exactly_once_per_interpretation():
    """The withheld-p/q note is a property of the interpretation, not of each effect.

    It must show up once in ``notes`` no matter how many families are tested, and
    it must not be duplicated across the record's other serialization surfaces.
    """
    note = (
        "The implemented percentile block bootstrap supports estimation only. "
        "Hypothesis-test p and q values are withheld until the preregistered "
        "wild cluster bootstrap-t estimator is used."
    )
    result = interpret.interpret_query(_rows(), _query(), n_bootstrap=50, seed=1)
    assert len(result.effects) > 1                    # more than one family in the fixture
    assert result.notes.count(note) == 1


def test_no_withheld_pq_note_when_there_are_no_effects_to_withhold_them_from():
    """When query and comparator share no family at all, `estimate_effects` returns
    `effects=[]` (there is nothing to compute a per-family difference over). The
    withheld-p/q note describes a limitation of effects that exist; with none, it
    must not fire -- unlike `test_estimation_only_note_appears_exactly_once_per_interpretation`,
    which requires it fire exactly once when effects DO exist.
    """
    rows = []
    for i in range(4):
        common = dict(chrom="chr1", input_scale=SCALE, lexicon_id=LEXICON)
        rows.append(HitRecord(region_id=f"q{i}", start=i * 1000, end=i * 1000 + 500,
                              missingness=Missingness.NO_SEQUENCE_MATCH, **common))
        rows.append(HitRecord(region_id=f"c{i}", start=500_000 + i * 1000,
                              end=500_000 + i * 1000 + 500,
                              missingness=Missingness.HIT_BELOW_FLOOR, **common))
    query = PeakSetQuery(
        query_id="q_nofam", region_ids=[f"q{i}" for i in range(4)],
        selection_provenance=SelectionProvenance.EXTERNAL,
        comparator_id="c_nofam", comparator_region_ids=[f"c{i}" for i in range(4)])
    floors = HealthFloors(min_intersection_coverage=0.0, min_blocks=1, min_explained_fraction=0.0)
    result = interpret.interpret_query(rows, query, floors=floors, n_bootstrap=20, seed=1)
    assert result.effects == []
    assert not any("withheld" in n for n in result.notes)


def test_same_seed_reproduces_the_interval_exactly():
    a = interpret.interpret_query(_rows(), _query(), n_bootstrap=64, seed=5)
    b = interpret.interpret_query(_rows(), _query(), n_bootstrap=64, seed=5)
    assert [e["ci"] for e in a.effects] == [e["ci"] for e in b.effects]


def test_inference_without_a_named_baseline_is_refused():
    q = PeakSetQuery(query_id="q1", region_ids=_ids(0),
                     selection_provenance=SelectionProvenance.EXTERNAL)
    with pytest.raises(interpret.InterpretError, match="baseline"):
        interpret.interpret_query(_rows(), q, n_bootstrap=20)


def test_clustered_with_split_without_a_split_is_refused():
    q = _query(selection_provenance=SelectionProvenance.CLUSTERED_WITH_SPLIT)
    with pytest.raises(interpret.InterpretError, match="held-out"):
        interpret.interpret_query(_rows(), q, n_bootstrap=20)


def test_clustered_with_split_infers_on_the_held_out_half_only():
    half = range(N_BLOCKS // 2, N_BLOCKS)
    q = _query(selection_provenance=SelectionProvenance.CLUSTERED_WITH_SPLIT,
               held_out_region_ids=_ids(0, half) + _ids(1, half))
    result = interpret.interpret_query(_rows(), q, n_bootstrap=100,
                                       floors=HealthFloors(min_blocks=N_BLOCKS // 2))
    assert result.output_mode == "FULL_INFERENCE_HELD_OUT"
    assert result.health["n_blocks"] == N_BLOCKS // 2
    assert result.effects and all(e["n_query_peaks"] == len(_ids(0, half)) for e in result.effects)


def test_an_empty_query_is_refused_rather_than_producing_a_ratio_without_a_denominator():
    with pytest.raises(interpret.InterpretError, match="denominator"):
        interpret.interpret_query(_rows(), _query(region_ids=[]), n_bootstrap=20)


# ------------------------------------------------------------------------- io
def _write_table(tmp_path):
    path = tmp_path / "hits.tsv"
    lines = ["\t".join(HIT_TABLE_COLUMNS)]
    for r in _rows():
        lines.append("\t".join([
            r.region_id, r.chrom, str(r.start), str(r.end), r.variant_id, r.family_id,
            "" if r.hit_coefficient is None else str(r.hit_coefficient),
            r.missingness.value, str(r.input_scale), r.lexicon_id,
        ]))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_hit_table_round_trips_through_tsv(tmp_path):
    read_back = interpret.read_hit_table(_write_table(tmp_path))
    assert len(read_back) == len(_rows())
    assert {r.missingness for r in read_back} == {r.missingness for r in _rows()}
    assert all(r.hit_coefficient is None for r in read_back
               if r.missingness is not Missingness.USED)


def test_a_table_spanning_two_input_scales_is_refused(tmp_path):
    lines = _write_table(tmp_path).read_text().splitlines()
    lines.append(lines[-1].replace(f"\t{SCALE}\t", "\t13277\t"))
    path = tmp_path / "mixed.tsv"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(interpret.InterpretError, match="input scale"):
        interpret.read_hit_table(path)


def test_peak_set_reads_bed_name_column_and_plain_ids(tmp_path):
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t0\t500\tr000_0\nchr1\t2000\t2500\tr000_2\n")
    assert interpret.read_peak_set(bed) == ["r000_0", "r000_2"]
    ids = tmp_path / "peaks.txt"
    ids.write_text("# comment\nr000_0\nr000_2\n")
    assert interpret.read_peak_set(ids) == ["r000_0", "r000_2"]


def test_every_result_enumerates_the_estimators_it_recognises():
    """A caller branches on the recognised set, not on a string literal (FP-15)."""
    from motifmultiverse.schema import IMPLEMENTED_ESTIMATORS, Estimator
    result = interpret.interpret_query(_rows(), _query(), n_bootstrap=50)
    assert result.estimator in result.estimators_implemented
    assert set(result.estimators_defined) == {e.value for e in Estimator}
    assert set(result.estimators_implemented) == {e.value for e in IMPLEMENTED_ESTIMATORS}
    # The specified estimators are recognised but absent, and say so by their absence.
    assert Estimator.BCA_PAIRED_BLOCK_BOOTSTRAP.value in result.estimators_defined
    assert Estimator.BCA_PAIRED_BLOCK_BOOTSTRAP.value not in result.estimators_implemented
    # Label permutation is abandoned, not pending: it is not even recognised.
    assert not any("permutation" in e for e in result.estimators_defined)


def test_interpretation_writes_json(tmp_path):
    result = interpret.interpret_query(_rows(), _query(), n_bootstrap=50)
    dest = result.write(tmp_path / "o")
    blob = json.loads(dest.read_text())
    assert blob["output_mode"] == "FULL_INFERENCE"
    assert blob["health"]["n_blocks"] == N_BLOCKS
    assert blob["estimator"] == interpret.ESTIMATOR
