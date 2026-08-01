"""interpret tests: the dispatch, the three health numbers, and the suppression rule.

Much of the behaviour under test is *refusal*, so most of these assert that a
number was NOT produced. That is the point of the module: what a peak set is
allowed to say depends on how it was chosen, and a query that cannot support an
interval must not emit one with a caveat attached.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from motifmultiverse import guards as guards_mod
from motifmultiverse import infer as infer_mod
from motifmultiverse import interpret
from motifmultiverse import schema as schema_mod
from motifmultiverse.guard_log import GuardLog
from motifmultiverse.schema import (
    DEFAULT_ATTRIBUTION_DERIVED_FEATURE_NAMES,
    HIT_TABLE_COLUMNS,
    MISSING_SENTINEL,
    ClaimScope,
    HealthFloors,
    HitRecord,
    Missingness,
    OutputMode,
    PeakSetQuery,
    SchemaError,
    SelectionProvenance,
    StatisticalLicense,
    resolve_query_permissions,
)

BLOCK = 1_000_000
LEXICON = "lex_core_v1"
SCALE = 33917

N_BLOCKS = 42          # divisible by 3 and by 2, so the planted effects are exact
PER_BLOCK = 4

NOT_SEARCHED_PEAK = "r000_notsearched"
NO_MATCH_PEAK = "r001_nomatch"
SUBSTRATE_ID = "e" * 64


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
                          input_scale=SCALE, lexicon_id=LEXICON, substrate_id=SUBSTRATE_ID)
            rows.append(HitRecord(
                missingness=Missingness.USED, variant_id=f"UA_FAMA_{i:02d}", family_id="FAM_A",
                hit_coefficient=1.0 + (b % 3) * 0.3 if is_query else 0.4, **common))
            rows.append(HitRecord(
                missingness=Missingness.USED, variant_id=f"UA_FAMB_{i:02d}", family_id="FAM_B",
                hit_coefficient=0.5 + ((b if is_query else b + 1) % 3) * 0.2, **common))
    rows.append(HitRecord(region_id=NOT_SEARCHED_PEAK, chrom="chr1", start=900_000, end=900_500,
                          missingness=Missingness.NOT_SEARCHED,
                          input_scale=SCALE, lexicon_id=LEXICON, substrate_id=SUBSTRATE_ID))
    rows.append(HitRecord(region_id=NO_MATCH_PEAK, chrom="chr1", start=1_900_000, end=1_900_500,
                          missingness=Missingness.NO_SEQUENCE_MATCH,
                          input_scale=SCALE, lexicon_id=LEXICON, substrate_id=SUBSTRATE_ID))
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


def test_interpret_query_refuses_records_without_a_substrate_id():
    """The direct API must not emit an artifact with the identity sentinel."""
    rows = [replace(row, substrate_id=MISSING_SENTINEL) for row in _rows()]
    with pytest.raises(interpret.InterpretError, match="without a substrate_id"):
        interpret.interpret_query(rows, _query(), n_bootstrap=20)


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


def test_a_region_may_legally_mix_three_measured_states_at_once():
    """Task 1 covers NOT_SEARCHED mixed with measured rows (refused). This is the
    companion case Task 1 does not reach: a region carrying three *measured*
    states at once -- USED, NO_SEQUENCE_MATCH, HIT_BELOW_FLOOR, with no
    NOT_SEARCHED row at all -- is legal (a region can be probed by more than one
    variant/model and get different measured outcomes) and must aggregate
    correctly: `searched=True`, and only the USED row contributes to
    `family_hit_count` / `family_coefficient_sum`.
    """
    rows = [
        HitRecord(region_id="r", chrom="chr1", start=0, end=10,
                  missingness=Missingness.USED, variant_id="UA_FAMA_01",
                  family_id="FAM_A", hit_coefficient=1.5,
                  input_scale=SCALE, lexicon_id=LEXICON),
        HitRecord(region_id="r", chrom="chr1", start=0, end=10,
                  missingness=Missingness.NO_SEQUENCE_MATCH,
                  input_scale=SCALE, lexicon_id=LEXICON),
        HitRecord(region_id="r", chrom="chr1", start=0, end=10,
                  missingness=Missingness.HIT_BELOW_FLOOR,
                  input_scale=SCALE, lexicon_id=LEXICON),
    ]
    peak = interpret.peak_universe(rows, BLOCK)["r"]
    assert peak.searched is True
    assert peak.family_hit_count == {"FAM_A": 1}
    assert peak.family_coefficient_sum == {"FAM_A": 1.5}
    assert peak.has_used_hit is True


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


def test_contrast_health_shared_and_union_blocks_count_the_overlap():
    """`shared_blocks`/`union_blocks` are required `ContrastHealth` interface fields
    with no prior test coverage at all. Query spans blocks 0-9, comparator spans
    blocks 5-14: five blocks in common (5..9), fifteen in the union (0..14).
    """
    peaks = interpret.peak_universe(_rows(), BLOCK)
    query_ids = _ids(0, range(0, 10))
    comparator_ids = _ids(1, range(5, 15))
    contrast = interpret.contrast_health_report(
        peaks, query_ids, comparator_ids, HealthFloors(), BLOCK)
    assert contrast.shared_blocks == 5
    assert contrast.union_blocks == 15


def test_effects_gate_reads_the_single_contrast_passed_field(monkeypatch):
    """The effects gate must read the one computed `ContrastHealth.passed`, not
    re-derive the same AND from `.query.passed` / `.comparator.passed` inline.

    This forces `.passed` to disagree with what those two sub-fields would
    recompute (both real sides pass, but `.passed` is forced to False). If the
    gate still recomputes its own condition instead of reading the field, the
    force has no effect and effects would be emitted anyway.
    """
    real = interpret.contrast_health_report

    def forced_false(*args, **kwargs):
        ch = real(*args, **kwargs)
        assert ch.passed is True and ch.query.passed and ch.comparator.passed
        ch.passed = False
        return ch

    monkeypatch.setattr(interpret, "contrast_health_report", forced_false)
    result = interpret.interpret_query(_rows(), _query(), n_bootstrap=20, seed=1)
    assert result.effects is None


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


# ------------------------------------------------------- two independent axes
def test_held_out_attribution_cluster_is_inferential_but_substrate_circular():
    """The brief's own dispatch case: a held-out split earns full statistical
    license, and an attribution-derived selection feature is substrate-circular,
    *in the same query*. Collapsing these into one grade would either lose the
    license or lose the circularity warning; this is the reason the two exist.
    """
    query = PeakSetQuery(
        query_id="q", region_ids=_ids(0),
        selection_provenance=SelectionProvenance.CLUSTERED_WITH_SPLIT,
        selection_feature_names=["attribution_pc1"],
        held_out_region_ids=_ids(0),
    )
    assert query.statistical_license is StatisticalLicense.HELD_OUT_INFERENCE
    assert query.claim_scope is ClaimScope.SUBSTRATE_CIRCULAR


@pytest.mark.parametrize("feature_names,expected_scope", [
    ([], ClaimScope.INTERNAL_DECOMPOSITION),
    (["tss_distance"], ClaimScope.INTERNAL_DECOMPOSITION),
    (["attribution_pc1"], ClaimScope.SUBSTRATE_CIRCULAR),
    (["tss_distance", "hit_coefficient"], ClaimScope.SUBSTRATE_CIRCULAR),
])
def test_claim_scope_tracks_selection_features_while_license_holds_fixed(
        feature_names, expected_scope):
    """Fixing the provenance (and so the license) and varying only the selection
    features must move `claim_scope` alone. If `claim_scope` were derived from
    `statistical_license` rather than independently from the feature names, this
    would not vary -- it would be constant across all four rows.
    """
    query = PeakSetQuery(
        query_id="q", region_ids=_ids(0),
        selection_provenance=SelectionProvenance.CLUSTERED_WITH_SPLIT,
        selection_feature_names=feature_names,
        held_out_region_ids=_ids(0),
    )
    assert query.statistical_license is StatisticalLicense.HELD_OUT_INFERENCE
    assert query.claim_scope is expected_scope


@pytest.mark.parametrize("grade,expected_license", [
    (SelectionProvenance.PROGRAMMATIC_RULE, StatisticalLicense.FULL_INFERENCE),
    (SelectionProvenance.CLUSTERED_WITH_SPLIT, StatisticalLicense.HELD_OUT_INFERENCE),
    (SelectionProvenance.CLUSTERED_NO_SPLIT, StatisticalLicense.DESCRIPTIVE_ONLY),
    (SelectionProvenance.EYEBALLED, StatisticalLicense.DESCRIPTIVE_ONLY),
])
def test_statistical_license_varies_while_claim_scope_holds_fixed(grade, expected_license):
    """The mirror of the previous test: fixing "no attribution feature, not
    EXTERNAL, not conditioning-unverifiable" pins `claim_scope` at
    INTERNAL_DECOMPOSITION across four different provenances whose licenses are
    all different -- proving `statistical_license` is not read off `claim_scope`.
    """
    extra = {"selection_rule": "distance_to_tss < 2000"} if grade is SelectionProvenance.PROGRAMMATIC_RULE else {}
    query = PeakSetQuery(
        query_id="q", region_ids=_ids(0), selection_provenance=grade,
        held_out_region_ids=_ids(0), **extra)
    assert query.claim_scope is ClaimScope.INTERNAL_DECOMPOSITION
    assert query.statistical_license is expected_license


@pytest.mark.parametrize("grade", [
    SelectionProvenance.MODEL_SELECTED_NO_TRANSCRIPT,
    SelectionProvenance.DECLARATION_MISSING,
])
def test_conditioning_unverifiable_provenance_stays_unverifiable_even_with_a_clean_feature(grade):
    """MODEL_SELECTED_NO_TRANSCRIPT / DECLARATION_MISSING cannot be rescued into
    EXTERNAL_STRUCTURE or INTERNAL_DECOMPOSITION by declaring an innocuous
    selection feature: what actually happened cannot be verified either way, so
    the scope stays at its floor regardless of what the feature list claims.
    """
    query = PeakSetQuery(
        query_id="q", region_ids=_ids(0), selection_provenance=grade,
        selection_feature_names=["tss_distance"])
    assert query.statistical_license is StatisticalLicense.DESCRIPTIVE_ONLY
    assert query.claim_scope is ClaimScope.CONDITIONING_UNVERIFIABLE


@pytest.mark.parametrize("bad_provenance", [None, "", "FROM_A_LATER_LEDGER", 42])
def test_resolve_query_permissions_refuses_unknown_provenance_to_the_floor_of_both_axes(
        bad_provenance):
    """Direct call, bypassing `PeakSetQuery.__post_init__`'s own coercion, since a
    future caller (Task 8/12/13/18) may call the resolver directly. Unknown or
    missing provenance must land on the floor of *both* axes, never the
    permissive value of either -- an undeclared selection is not a safe one.
    """
    license_, scope = resolve_query_permissions(
        bad_provenance, ["attribution_pc1"], DEFAULT_ATTRIBUTION_DERIVED_FEATURE_NAMES)
    assert license_ is StatisticalLicense.DESCRIPTIVE_ONLY
    assert scope is ClaimScope.CONDITIONING_UNVERIFIABLE


def test_permission_resolver_uses_the_caller_supplied_attribution_registry():
    """The resolver's registry parameter is semantic input, not decoration.
    A caller with a differently named attribution feature must be able to mark
    it substrate-circular without modifying the package-wide default registry.
    """
    license_, scope = resolve_query_permissions(
        SelectionProvenance.EXTERNAL, ["custom_attribution_signal"],
        {"custom_attribution_signal"})
    assert license_ is StatisticalLicense.FULL_INFERENCE
    assert scope is ClaimScope.SUBSTRATE_CIRCULAR


def test_declaration_missing_never_resolves_to_the_permissive_grade_of_either_axis():
    """DECLARATION_MISSING is the recorded state of a query that declared
    nothing. It must not land on FULL_INFERENCE / HELD_OUT_INFERENCE nor on
    EXTERNAL_STRUCTURE / INTERNAL_DECOMPOSITION -- both would treat silence as
    if it were a stated, safe selection.
    """
    q = PeakSetQuery(query_id="q", region_ids=_ids(0))
    assert q.selection_provenance is SelectionProvenance.DECLARATION_MISSING
    assert q.statistical_license is StatisticalLicense.DESCRIPTIVE_ONLY
    assert q.claim_scope is ClaimScope.CONDITIONING_UNVERIFIABLE


def test_missing_provenance_stays_conservative_through_interpretation_serialization():
    """A missing provenance must become the recorded declaration-missing state,
    rather than resolving safely at first and then crashing when the result is
    serialized. This protects the end-to-end unknown-provenance contract.
    """
    query = PeakSetQuery(query_id="q", region_ids=_ids(0), selection_provenance=None)
    result = interpret.interpret_query(_rows(), query, n_bootstrap=20)
    assert result.selection_provenance == "DECLARATION_MISSING"
    assert result.statistical_license == "DESCRIPTIVE_ONLY"
    assert result.claim_scope == "CONDITIONING_UNVERIFIABLE"


def test_selection_feature_names_defaults_to_empty_and_does_not_disturb_existing_callers():
    q = PeakSetQuery(query_id="q", region_ids=_ids(0),
                     selection_provenance=SelectionProvenance.EXTERNAL)
    assert q.selection_feature_names == []
    assert q.claim_scope is ClaimScope.EXTERNAL_STRUCTURE


def test_legacy_output_mode_is_read_from_the_two_axes_not_a_second_independent_source(
        monkeypatch):
    """`output_mode` is documented as a compatibility field DERIVED from
    `statistical_license` / `claim_scope`, not a second thing computed from
    `selection_provenance` on its own. Patch the resolver to return a
    combination that disagrees with what EXTERNAL would normally produce
    (FULL_INFERENCE): if `output_mode` still tracked `selection_provenance`
    directly (the pre-Task-7 shape), it would report FULL_INFERENCE here and
    this assertion would fail.
    """
    def fake_resolve(provenance, selection_feature_names, attribution_derived_registry):
        return StatisticalLicense.HELD_OUT_INFERENCE, ClaimScope.SUBSTRATE_CIRCULAR

    monkeypatch.setattr(schema_mod, "resolve_query_permissions", fake_resolve)
    q = PeakSetQuery(query_id="q", region_ids=_ids(0),
                     selection_provenance=SelectionProvenance.EXTERNAL)
    assert q.statistical_license is StatisticalLicense.HELD_OUT_INFERENCE
    assert q.claim_scope is ClaimScope.SUBSTRATE_CIRCULAR
    assert q.output_mode is OutputMode.FULL_INFERENCE_HELD_OUT


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
        common = dict(chrom="chr1", input_scale=SCALE, lexicon_id=LEXICON,
                      substrate_id=SUBSTRATE_ID)
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


def test_interpretation_json_emits_the_two_independent_permission_axes(tmp_path):
    """Task 7's serialized record must preserve both permissions, not only
    legacy ``output_mode``. Removing either emitted field, or serializing scope
    from the legacy mode (which cannot encode SUBSTRATE_CIRCULAR), must fail.
    """
    half = range(N_BLOCKS // 2, N_BLOCKS)
    query = _query(
        selection_provenance=SelectionProvenance.CLUSTERED_WITH_SPLIT,
        selection_feature_names=["attribution_pc1"],
        held_out_region_ids=_ids(0, half) + _ids(1, half),
    )
    result = interpret.interpret_query(
        _rows(), query, n_bootstrap=50,
        floors=HealthFloors(min_blocks=N_BLOCKS // 2))
    payload = json.loads(result.write(tmp_path / "interpretation").read_text())
    assert payload["statistical_license"] == "HELD_OUT_INFERENCE"
    assert payload["claim_scope"] == "SUBSTRATE_CIRCULAR"
    assert payload["output_mode"] == "FULL_INFERENCE_HELD_OUT"


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
            r.missingness.value, str(r.input_scale), r.lexicon_id, "e" * 64,
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
    # Task 16 closed the gap this test was written around: FP-15's specified pair
    # now exists, so every recognised estimator is implemented. The invariant that
    # outlives the gap is the one asserted above and re-asserted below -- a result
    # may never name an estimator its own enumeration calls unavailable, which is
    # what a stale IMPLEMENTED_ESTIMATORS would have produced.
    assert Estimator.BCA_PAIRED_BLOCK_BOOTSTRAP.value in result.estimators_defined
    assert set(result.estimators_implemented) == set(result.estimators_defined)
    # Label permutation is abandoned, not pending: it is not even recognised.
    assert not any("permutation" in e for e in result.estimators_defined)


def test_a_result_never_names_an_estimator_its_own_enumeration_calls_unavailable():
    """The self-consistency `estimators_implemented` exists to provide.

    Checked on the path that actually changed: a `bca-wild-cluster` run records
    `wild_cluster_bootstrap_t`, and a reader who trusts the enumeration beside it
    must not conclude that estimator does not exist in this release.
    """
    result = interpret.interpret_query(
        _rows(), _query(), n_bootstrap=100, seed=3, estimator=interpret.ESTIMATOR_BCA_WILD)
    assert result.estimator == interpret.ESTIMATOR_BCA_WILD
    assert result.estimator in result.estimators_implemented
    for effect in result.effects:
        assert effect["estimator"] in result.estimators_implemented


def test_interpretation_writes_json(tmp_path):
    result = interpret.interpret_query(_rows(), _query(), n_bootstrap=50)
    dest = result.write(tmp_path / "o")
    blob = json.loads(dest.read_text())
    assert blob["output_mode"] == "FULL_INFERENCE"
    assert blob["health"]["n_blocks"] == N_BLOCKS
    assert blob["estimator"] == interpret.ESTIMATOR


# --------------------------------------------------------------------------- #
# Task 3 carried obligation: direct `_bh` coverage BEFORE it is reactivated    #
# (Task 16 wires it behind INTERVAL_AND_TEST). All expected values below are   #
# hand-computed from the BH rule q_(i) = min_(j>=i) p_(j)*m/j, not from the    #
# implementation under test.                                                   #
# --------------------------------------------------------------------------- #
def test_bh_matches_a_hand_computed_known_vector():
    # m=5. Raw: .005*5/1=.025, .01*5/2=.025, .02*5/3=1/30, .04*5/4=.05, .5*5/5=.5;
    # running min from the largest is already monotone here.
    q = interpret._bh([0.005, 0.01, 0.02, 0.04, 0.5])
    assert q == pytest.approx([0.025, 0.025, 1 / 30, 0.05, 0.5])


def test_bh_returns_q_values_in_input_positions_not_sorted_order():
    # Same multiset as the known vector, permuted: each q must follow its p.
    q = interpret._bh([0.5, 0.02, 0.005, 0.04, 0.01])
    assert q == pytest.approx([0.5, 1 / 30, 0.025, 0.05, 0.025])


def test_bh_applies_the_running_min_from_the_largest_p():
    # Without the running-min step the second entry would be 0.01*3/2 = 0.015 and
    # the first 0.005*3/1 = 0.015; the raw middle value 0.04*3/2 = 0.06 must be
    # pulled down by nothing (it is the max) -- use a vector where an interior
    # raw value exceeds a later one: raw .04*3/2=.06 > .05*3/3=.05.
    q = interpret._bh([0.005, 0.04, 0.05])
    assert q == pytest.approx([0.015, 0.05, 0.05])


def test_bh_ties_share_one_q_value():
    # m=3, two tied p: q for both is min(.01*3/1, .01*3/2) carried across the tie.
    q = interpret._bh([0.01, 0.01, 0.05])
    assert q[0] == q[1] == pytest.approx(0.015)
    assert q[2] == pytest.approx(0.05)


def test_bh_is_monotone_in_the_p_values():
    p = [0.6, 0.01, 0.04, 0.04, 0.9, 0.001]
    q = interpret._bh(p)
    for i in range(len(p)):
        for j in range(len(p)):
            if p[i] < p[j]:
                assert q[i] <= q[j]


def test_bh_never_exceeds_one_and_never_undercuts_p():
    # Raw BH values here are 0.9*3/1 = 2.7 and 0.95*3/2 = 1.425 -- both ABOVE 1 --
    # with 1.0*3/3 = 1.0 at the top; the running min plus the cap at 1 pulls all
    # three to exactly 1.0 (matches R p.adjust(c(.9,.95,1), "BH")).
    p = [0.9, 0.95, 1.0]
    q = interpret._bh(p)
    assert all(qi <= 1.0 for qi in q)
    assert all(qi >= pi for pi, qi in zip(p, q, strict=True))
    assert q == [1.0, 1.0, 1.0]


def test_bh_single_p_value_is_its_own_q():
    assert interpret._bh([0.03]) == [pytest.approx(0.03)]


def test_bh_empty_input_gives_empty_output():
    assert interpret._bh([]) == []


# --------------------------------------------------------------------------- #
# Task 16: the bca-wild-cluster estimator path (`FP-15`, INTERVAL_AND_TEST)    #
# --------------------------------------------------------------------------- #
def test_bca_wild_cluster_recovers_the_planted_effect_and_the_planted_null():
    """The Task 15 fixture's planted truth, now with the licensed test attached.

    FAM_A's per-block effects cycle 0.6/0.9/1.2 (mean exactly 0.9, t_obs ~= 19):
    no null-world replicate reaches that, so p sits at the resolution floor and
    the BCa interval excludes zero. FAM_B's per-block effects cycle
    -0.2/-0.2/+0.4 (mean exactly 0 up to float dust): the planted null yields a
    large p and an interval straddling zero. With m=2 families, BH gives
    q_A = 2*p_A and q_B = p_B by hand-computation.
    """
    n_bootstrap = 200
    result = interpret.interpret_query(
        _rows(), _query(), n_bootstrap=n_bootstrap, seed=7,
        estimator=interpret.ESTIMATOR_BCA_WILD)
    by_family = {e["family_id"]: e for e in result.effects}
    a, b = by_family["FAM_A"], by_family["FAM_B"]

    assert a["estimator"] == "wild_cluster_bootstrap_t"
    assert a["inference_capability"] == "INTERVAL_AND_TEST"
    assert a["effect"] == pytest.approx(0.9, abs=1e-9)
    lo, hi = a["ci"]
    assert lo > 0.0 and hi > lo
    floor = 1.0 / (n_bootstrap + 1)
    assert a["p_value"] == floor
    assert a["n_bootstrap_valid"] == n_bootstrap

    assert b["effect"] == pytest.approx(0.0, abs=1e-9)
    lo_b, hi_b = b["ci"]
    assert lo_b < 0.0 < hi_b
    assert b["p_value"] > 0.5

    assert a["q_value"] == pytest.approx(min(1.0, 2 * a["p_value"]))
    assert b["q_value"] == pytest.approx(b["p_value"])


def _unbalanced_rows() -> list[HitRecord]:
    """Two query peaks and ONE comparator peak per block, so per-block *sums* and
    per-block *contributions to the mean difference* are not the same quantity.

    ``FAM_D``'s query coefficient cycles 0.8/1.0/1.2 and its comparator
    coefficient cycles the reverse, 1.2/1.0/0.8, both with mean exactly 1.0: the
    reported effect (mean per-peak query minus mean per-peak comparator) is
    exactly 0, while the per-block query sum is twice the query mean because the
    query side has twice the peaks.
    """
    rows: list[HitRecord] = []
    cycle = (0.8, 1.0, 1.2)
    for b in range(N_BLOCKS):
        start = b * BLOCK
        for i in (0, 1):     # two query peaks, same coefficient
            rows.append(HitRecord(
                region_id=f"u{b:03d}_q{i}", chrom="chr1", start=start + i * 1000,
                end=start + i * 1000 + 500, missingness=Missingness.USED,
                variant_id=f"UA_FAMD_q{i}", family_id="FAM_D",
                hit_coefficient=cycle[b % 3],
                input_scale=SCALE, lexicon_id=LEXICON, substrate_id=SUBSTRATE_ID))
        rows.append(HitRecord(
            region_id=f"u{b:03d}_c0", chrom="chr1", start=start + 5000, end=start + 5500,
            missingness=Missingness.USED, variant_id="UA_FAMD_c0", family_id="FAM_D",
            hit_coefficient=cycle[2 - (b % 3)],
            input_scale=SCALE, lexicon_id=LEXICON, substrate_id=SUBSTRATE_ID))
    return rows


def test_the_p_value_tests_a_hypothesis_about_the_effect_reported_beside_it():
    """The per-block effects must be scaled to the reported point estimate.

    This is what `(G/N_q) * sum_q(g) - (G/N_c) * sum_c(g)` buys, and the fixture
    is built so the two candidate quantities disagree in *sign of conclusion*
    rather than in a digit:

    * scaled (correct): per-block effects cycle -0.4 / 0.0 / +0.4, mean exactly
      0 -- the planted truth -- so the test finds no evidence against the null
      and p is large;
    * unscaled per-block sums: 0.4 / 1.0 / 1.6, mean 1.0 with t ~= 13, so p
      collapses to the resolution floor.

    An unscaled implementation would therefore report `effect == 0.0` beside
    `p == 1/(B+1)`: a p value that is a true statement about a quantity nobody
    is shown, printed next to the quantity everybody reads. Every other
    bca-wild-cluster test in this file passes under both implementations, which
    is why this one exists.
    """
    n_bootstrap = 200
    query = _query(query_id="unbalanced",
                   region_ids=[f"u{b:03d}_q{i}" for b in range(N_BLOCKS) for i in (0, 1)],
                   comparator_region_ids=[f"u{b:03d}_c0" for b in range(N_BLOCKS)])
    result = interpret.interpret_query(
        _unbalanced_rows(), query, n_bootstrap=n_bootstrap, seed=13,
        estimator=interpret.ESTIMATOR_BCA_WILD)
    effect = result.effects[0]
    assert effect["family_id"] == "FAM_D"
    assert effect["n_query_peaks"] == 2 * N_BLOCKS
    assert effect["n_comparator_peaks"] == N_BLOCKS
    assert effect["effect"] == pytest.approx(0.0, abs=1e-12)
    assert effect["p_value"] > 0.5
    assert effect["p_value"] > 10.0 / (n_bootstrap + 1)   # nowhere near the floor
    lo, hi = effect["ci"]
    assert lo < 0.0 < hi


def test_bca_wild_cluster_is_deterministic_given_the_seed():
    kw = dict(n_bootstrap=100, seed=5, estimator=interpret.ESTIMATOR_BCA_WILD)
    a = interpret.interpret_query(_rows(), _query(), **kw)
    b = interpret.interpret_query(_rows(), _query(), **kw)
    assert [e["ci"] for e in a.effects] == [e["ci"] for e in b.effects]
    assert [e["p_value"] for e in a.effects] == [e["p_value"] for e in b.effects]
    assert [e["q_value"] for e in a.effects] == [e["q_value"] for e in b.effects]


def test_bca_wild_cluster_names_both_estimator_halves_once_and_withholds_nothing():
    result = interpret.interpret_query(
        _rows(), _query(), n_bootstrap=100, seed=1,
        estimator=interpret.ESTIMATOR_BCA_WILD)
    # The withheld-p/q note belongs to the estimation-only path; it must not
    # appear where p/q are actually emitted.
    assert not any("withheld" in n for n in result.notes)
    # The effect-level `estimator` field names the capability-licensing member
    # (the test); the BCa interval half is named once per interpretation.
    naming = [n for n in result.notes if "BCa" in n and "wild cluster bootstrap-t" in n]
    assert len(naming) == 1
    assert result.estimator == "wild_cluster_bootstrap_t"
    for e in result.effects:
        assert e["p_value"] is not None and e["q_value"] is not None


def test_percentile_estimator_explicit_still_withholds_p_and_q():
    result = interpret.interpret_query(
        _rows(), _query(), n_bootstrap=50, seed=1,
        estimator=interpret.ESTIMATOR_PERCENTILE)
    for effect in result.effects:
        assert effect["estimator"] == "percentile_block_bootstrap"
        assert effect["inference_capability"] == "ESTIMATION_ONLY"
        assert effect["p_value"] is None
        assert effect["q_value"] is None
    assert any("withheld" in n for n in result.notes)


def test_an_unknown_estimator_is_refused_not_silently_mapped():
    with pytest.raises(interpret.InterpretError, match="estimator"):
        interpret.interpret_query(_rows(), _query(), n_bootstrap=20, estimator="bootstrap")


def test_bca_wild_cluster_refuses_when_the_block_frame_is_below_the_infer_floor():
    """Health floors are caller-adjustable; infer's own estimability floor is
    not. A 12-block frame clears a lowered --floor-blocks 10 health gate but is
    far below MIN_ESTIMABLE_BLOCKS, so the estimator itself must refuse.
    """
    blocks = range(12)
    q = _query(region_ids=_ids(0, blocks), comparator_region_ids=_ids(1, blocks))
    with pytest.raises(infer_mod.InferError, match="below the preregistered floor"):
        interpret.interpret_query(
            _rows(), q, floors=HealthFloors(min_blocks=10), n_bootstrap=100,
            estimator=interpret.ESTIMATOR_BCA_WILD)


def _constant_family_rows() -> list[HitRecord]:
    """Same shape as `_rows` but every FAM_C coefficient is constant per side:
    per-block effects are then identical in every block, the wild bootstrap-t's
    observed SE is exactly 0, and no replicate is estimable.
    """
    rows: list[HitRecord] = []
    for b in range(N_BLOCKS):
        for i in range(PER_BLOCK):
            start = b * BLOCK + i * 1000
            rows.append(HitRecord(
                region_id=f"r{b:03d}_{i}", chrom="chr1", start=start, end=start + 500,
                missingness=Missingness.USED, variant_id=f"UA_FAMC_{i:02d}",
                family_id="FAM_C", hit_coefficient=1.0 if i % 2 == 0 else 0.4,
                input_scale=SCALE, lexicon_id=LEXICON, substrate_id=SUBSTRATE_ID))
    return rows


def test_bca_wild_cluster_refuses_a_family_with_a_degenerate_bootstrap_reference():
    """Constant block effects leave zero estimable replicates. The doctrine is
    refusal, not annotation: a p value computed from nothing must not travel
    beside the other families' valid ones.
    """
    with pytest.raises(interpret.InterpretError, match="degenerate"):
        interpret.interpret_query(
            _constant_family_rows(), _query(), n_bootstrap=100, seed=1,
            estimator=interpret.ESTIMATOR_BCA_WILD)


# --------------------------------------------------------------------------- #
# Task 17: two-part usage summaries wired into an interpretation               #
# --------------------------------------------------------------------------- #
def _cancelling_rows() -> list[HitRecord]:
    """A family whose one-part effect is ~0 for two opposite reasons.

    ``FAM_E`` is used in every query peak at 0.5, and in half the comparator
    peaks at 1.0: occupancy is twice as high on the query side, intensity is
    half. The mean per-peak coefficient is 0.5 on both sides, so a single
    difference reports 0.0 -- and reports it for a family whose two margins are
    each large.
    """
    rows: list[HitRecord] = []
    for b in range(N_BLOCKS):
        for i in range(PER_BLOCK):
            start = b * BLOCK + i * 1000
            is_query = i % 2 == 0
            used = is_query or (i == 1)          # every query peak, half the comparator
            if not used:
                rows.append(HitRecord(
                    region_id=f"e{b:03d}_{i}", chrom="chr1", start=start, end=start + 500,
                    missingness=Missingness.NO_SEQUENCE_MATCH,
                    input_scale=SCALE, lexicon_id=LEXICON, substrate_id=SUBSTRATE_ID))
                continue
            rows.append(HitRecord(
                region_id=f"e{b:03d}_{i}", chrom="chr1", start=start, end=start + 500,
                missingness=Missingness.USED, variant_id=f"UA_FAME_{i:02d}",
                family_id="FAM_E", hit_coefficient=0.5 if is_query else 1.0,
                input_scale=SCALE, lexicon_id=LEXICON, substrate_id=SUBSTRATE_ID))
    return rows


def _cancelling_query(**over) -> PeakSetQuery:
    kw = dict(query_id="cancel",
              region_ids=[f"e{b:03d}_{i}" for b in range(N_BLOCKS)
                          for i in range(PER_BLOCK) if i % 2 == 0],
              selection_provenance=SelectionProvenance.EXTERNAL,
              comparator_id="odd_peaks",
              comparator_region_ids=[f"e{b:03d}_{i}" for b in range(N_BLOCKS)
                                     for i in range(PER_BLOCK) if i % 2 == 1])
    kw.update(over)
    return PeakSetQuery(**kw)


def test_two_part_summary_reveals_what_the_one_part_effect_hides():
    """The one-part effect and the two-part summary, in one interpretation."""
    result = interpret.interpret_query(
        _cancelling_rows(), _cancelling_query(), n_bootstrap=50, seed=1,
        usage_definition=infer_mod.UsageDefinition.ANY_HIT)

    effect = next(e for e in result.effects if e["family_id"] == "FAM_E")
    assert effect["effect"] == pytest.approx(0.0, abs=1e-12)   # "nothing here"

    part = next(t for t in result.two_part_effects if t["family_id"] == "FAM_E")
    assert part["usage_definition"] == "ANY_HIT"
    assert part["probability_effect"] == pytest.approx(0.5)    # ...but this,
    assert part["conditional_intensity_effect"] == pytest.approx(-0.5)   # ...and this
    assert part["total_effect"] == pytest.approx(0.0, abs=1e-12)
    assert part["n_used_query"] == 2 * N_BLOCKS
    assert part["n_used_comparator"] == N_BLOCKS
    assert part["n_measured_query"] == part["n_measured_comparator"] == 2 * N_BLOCKS
    assert "two_part_effects" in result.emitted_order


def test_no_usage_definition_means_no_two_part_section_not_a_default_one():
    """Absence records that nobody chose a definition of "used"."""
    result = interpret.interpret_query(_cancelling_rows(), _cancelling_query(), n_bootstrap=50)
    assert result.two_part_effects is None
    assert "two_part_effects" not in result.emitted_order
    assert not any("usage is defined" in n for n in result.notes)
    # and the one-part effect is still there, still reading 0.0 -- which is the
    # state this whole task exists to make visible rather than to forbid.
    assert next(e for e in result.effects
                if e["family_id"] == "FAM_E")["effect"] == pytest.approx(0.0, abs=1e-12)


def test_two_part_names_its_usage_definition_in_the_notes_and_on_every_row():
    result = interpret.interpret_query(
        _cancelling_rows(), _cancelling_query(), n_bootstrap=50,
        usage_definition=infer_mod.UsageDefinition.ANY_HIT)
    assert sum("usage is defined as ANY_HIT" in n for n in result.notes) == 1
    assert all(t["usage_definition"] == "ANY_HIT" for t in result.two_part_effects)


def test_two_part_contribution_floor_carries_its_threshold_provenance_to_disk(tmp_path):
    floor = infer_mod.UsageThreshold(value=0.75, null_source="dinucleotide_shuffle_v1")
    result = interpret.interpret_query(
        _cancelling_rows(), _cancelling_query(), n_bootstrap=50,
        usage_definition=infer_mod.UsageDefinition.CONTRIBUTION_FLOOR,
        usage_threshold=floor)
    blob = json.loads(result.write(tmp_path / "o").read_text())
    part = next(t for t in blob["two_part_effects"] if t["family_id"] == "FAM_E")
    assert part["usage_definition"] == "CONTRIBUTION_FLOOR"
    assert part["usage_threshold"] == 0.75
    assert part["usage_threshold_source"] == "dinucleotide_shuffle_v1"
    # The 0.5 query hits are below the floor: measured non-use, still in the
    # denominator, so the query side's probability of use drops to 0.
    assert part["n_used_query"] == 0
    assert part["n_measured_query"] == 2 * N_BLOCKS
    assert part["conditional_intensity_effect"] is None      # undefined, not 0.0


def test_an_unusable_usage_definition_is_refused_at_the_interpretation_boundary():
    with pytest.raises(infer_mod.InferError, match="no default"):
        interpret.interpret_query(_cancelling_rows(), _cancelling_query(),
                                  n_bootstrap=50, usage_definition="ANY_HIT")


def test_two_part_is_suppressed_with_the_effects_it_sits_beside():
    """It is a contrast, so an unhealthy comparator withholds it too."""
    result = interpret.interpret_query(
        _cancelling_rows(),
        _cancelling_query(comparator_region_ids=[f"e000_{i}" for i in (1, 3)]),
        n_bootstrap=50, usage_definition=infer_mod.UsageDefinition.ANY_HIT)
    assert result.effects is None
    assert result.two_part_effects is None
    assert result.composition is not None            # query-only reading stands
    assert any("comparator" in f for f in result.floor_failures)


# --- regression: a hit table must carry every declared column ------------------
# `interpret/README.md` promises "columns from schema.HIT_TABLE_COLUMNS" but the
# reader never checked. A table without `family_id` was accepted and every row
# took the sentinel, so every family share/effect/CI was computed for one
# fabricated family named "NA"; a renamed column raised a bare KeyError, i.e. a
# traceback out of the module whose principle is that bad input gets a sentence.
def _hit_row_fields():
    return dict(
        region_id="peak_1", chrom="chr1", start="100", end="200",
        variant_id="v1", family_id="F1", hit_coefficient="0.5",
        missingness="used", input_scale="10", lexicon_id="lex",
        substrate_id="c" * 64,
    )


def _write_hit_table(path, drop=None, rename=None):
    from motifmultiverse.schema import HIT_TABLE_COLUMNS

    full = _hit_row_fields()
    cols = [c for c in HIT_TABLE_COLUMNS if c != drop]
    header = [(rename or {}).get(c, c) for c in cols]
    path.write_text("\t".join(header) + "\n" + "\t".join(full[c] for c in cols) + "\n")
    return path


@pytest.mark.parametrize("dropped", ["family_id", "hit_coefficient", "substrate_id"])
def test_read_hit_table_refuses_a_missing_declared_column(tmp_path, dropped):
    from motifmultiverse.interpret import InterpretError, read_hit_table

    p = _write_hit_table(tmp_path / "hits.tsv", drop=dropped)
    with pytest.raises(InterpretError, match="missing required hit-table column"):
        read_hit_table(p)


def test_read_hit_table_names_the_missing_column(tmp_path):
    from motifmultiverse.interpret import InterpretError, read_hit_table

    p = _write_hit_table(tmp_path / "hits.tsv", drop="family_id")
    with pytest.raises(InterpretError) as excinfo:
        read_hit_table(p)
    assert "family_id" in str(excinfo.value)


def test_read_hit_table_refuses_a_renamed_column_with_a_sentence_not_a_keyerror(tmp_path):
    from motifmultiverse.interpret import InterpretError, read_hit_table

    p = _write_hit_table(tmp_path / "hits.tsv", rename={"chrom": "chromosome"})
    with pytest.raises(InterpretError, match="missing required hit-table column"):
        read_hit_table(p)


def test_read_hit_table_accepts_a_complete_table(tmp_path):
    from motifmultiverse.interpret import read_hit_table

    p = _write_hit_table(tmp_path / "hits.tsv")
    records = read_hit_table(p)
    assert len(records) == 1
    assert records[0].family_id == "F1"


def test_a_used_hit_must_name_a_family_not_the_sentinel(tmp_path):
    """A sentinel family is reported AS a family, not as a missing assignment.

    Verified against the pre-fix code: a table whose family_id column was entirely
    MISSING_SENTINEL produced a composition entry with family_id "NA",
    peak_share 1.0 and an effect. That is the table this package's own `compile`
    leads to -- it emits variant ids and no family, because family assignment is
    the slot annotate/README.md records as never specified.
    """
    from motifmultiverse.interpret import read_hit_table
    from motifmultiverse.schema import MISSING_SENTINEL, SchemaError

    p = _write_hit_table(tmp_path / "hits.tsv")
    text = p.read_text().replace("\tF1\t", f"\t{MISSING_SENTINEL}\t")
    p.write_text(text)
    with pytest.raises(SchemaError, match="must name a family_id"):
        read_hit_table(p)


def test_a_not_searched_row_may_leave_family_unnamed(tmp_path):
    """The refusal is about USED rows: absence still has no family to name."""
    from motifmultiverse.interpret import read_hit_table
    from motifmultiverse.schema import MISSING_SENTINEL

    p = _write_hit_table(tmp_path / "hits.tsv")
    text = (p.read_text()
            .replace("\tF1\t", f"\t{MISSING_SENTINEL}\t")
            .replace("\tused\t", "\tno_sequence_match\t")
            .replace("\t0.5\t", "\t\t"))
    p.write_text(text)
    assert len(read_hit_table(p)) == 1


# --- the two encodings of one table must be one table -------------------------
# The parquet branch of read_hit_table had NO test. That is why a null identifier
# becoming the literal string "nan" shipped green: pandas returns a null in an
# object column as float NaN, NaN is truthy, so `x or SENTINEL` was skipped and
# str(nan) produced an identifier that no sentinel check could see. The TSV
# encoding of the same table was refused; the parquet one reported a family
# called "nan".
def _frame(**overrides):
    import pandas as pd

    from motifmultiverse.schema import HIT_TABLE_COLUMNS

    rows = []
    for b in range(8):
        for variant, family in (("V_AP1", "AP-1/bZIP"), ("V_CTCF", "CTCF")):
            row = dict(
                region_id=f"p{b}", chrom="chr1", start=b * 1000, end=b * 1000 + 50,
                variant_id=variant, family_id=family, hit_coefficient=1.0,
                missingness="used", input_scale=8, lexicon_id="L",
                substrate_id="a" * 64,
            )
            rows.append(row)
    frame = pd.DataFrame(rows)[list(HIT_TABLE_COLUMNS)]
    for column, (mask_variant, value) in overrides.items():
        frame.loc[frame.variant_id == mask_variant, column] = value
    return frame


def _both_encodings(tmp_path, frame):
    tsv, parquet = tmp_path / "h.tsv", tmp_path / "h.parquet"
    frame.to_csv(tsv, sep="\t", index=False)
    frame.to_parquet(parquet, index=False)
    return tsv, parquet


def test_parquet_and_tsv_encodings_of_one_table_read_identically(tmp_path):
    pytest.importorskip("pyarrow")
    from motifmultiverse.interpret import read_hit_table

    tsv, parquet = _both_encodings(tmp_path, _frame())
    from_tsv, from_parquet = read_hit_table(tsv), read_hit_table(parquet)
    assert len(from_tsv) == len(from_parquet)
    assert from_tsv == from_parquet


@pytest.mark.parametrize("column", ["family_id", "variant_id", "chrom", "region_id"])
def test_a_null_identifier_is_never_read_as_the_string_nan(tmp_path, column):
    """Both encodings must reach the same verdict, and neither may invent a value."""
    pytest.importorskip("pyarrow")
    from motifmultiverse.interpret import InterpretError, read_hit_table
    from motifmultiverse.schema import SchemaError

    tsv, parquet = _both_encodings(tmp_path, _frame(**{column: ("V_CTCF", None)}))
    verdicts = {}
    for name, path in (("tsv", tsv), ("parquet", parquet)):
        try:
            records = read_hit_table(path)
            verdicts[name] = sorted({getattr(r, column) for r in records})
        except (SchemaError, InterpretError) as exc:
            verdicts[name] = f"refused: {type(exc).__name__}"
    assert "nan" not in str(verdicts["parquet"]), f"fabricated identifier: {verdicts}"
    assert verdicts["tsv"] == verdicts["parquet"], f"encodings disagree: {verdicts}"


def test_a_null_coefficient_is_absence_in_both_encodings(tmp_path):
    pytest.importorskip("pyarrow")
    from motifmultiverse.interpret import read_hit_table

    frame = _frame(hit_coefficient=("V_CTCF", None), missingness=("V_CTCF", "no_sequence_match"))
    tsv, parquet = _both_encodings(tmp_path, frame)
    from_tsv, from_parquet = read_hit_table(tsv), read_hit_table(parquet)
    assert from_tsv == from_parquet
    absent = [r for r in from_parquet if r.variant_id == "V_CTCF"]
    assert absent and all(r.hit_coefficient is None for r in absent)


# --- regression: a comparator that overlaps the query ------------------------
# `estimate_effects` differenced the two sides without ever asking whether they
# were two sides. Measured on the real K562 CBP-2114 substrate before this
# existed: island cluster 5 (8,277 peaks) against *all* 33,917 peaks reported all
# twelve families at exactly 0.7560 of their value against the same 33,917 minus
# the island -- 25,640/33,917, the comparator's disjoint fraction -- with the
# interval shifted to match, no note and no health field.
def test_a_comparator_that_overlaps_the_query_is_refused_rather_than_attenuated():
    """A peak on both sides of the difference is subtracted from itself."""
    peaks = interpret.peak_universe(_rows(), BLOCK)
    everything = _ids(0) + _ids(1)
    with pytest.raises(interpret.InterpretError, match="share 84 peak"):
        interpret.estimate_effects(peaks, _ids(0), everything, "all_peaks",
                                   n_bootstrap=50, seed=1, block_size=BLOCK)


def test_a_comparator_identical_to_the_query_is_refused_rather_than_reported_as_zero():
    """The degenerate end of the same defect, which produced the worst artifact.

    comparator == query returned effect 0.0 with the interval [0.0, 0.0] and
    ``is_cross_condition`` True on every family: a zero-width interval around an
    exact zero, which reads as "measured, and there is no difference".
    """
    peaks = interpret.peak_universe(_rows(), BLOCK)
    with pytest.raises(interpret.InterpretError, match="subtracted from itself"):
        interpret.estimate_effects(peaks, _ids(0), _ids(0), "itself",
                                   n_bootstrap=50, seed=1, block_size=BLOCK)


def test_contrast_health_counts_shared_peaks_because_shared_blocks_cannot_show_them():
    """`shared_blocks` was the only overlap number and it is not a proxy for this.

    On the real substrate the overlapping and disjoint contrasts spanned 283 and
    282 shared blocks -- one apart, while one had 8,277 peaks on both sides and
    the other none. The fixture reproduces the same insensitivity exactly.
    """
    peaks = interpret.peak_universe(_rows(), BLOCK)
    disjoint = interpret.contrast_health_report(peaks, _ids(0), _ids(1), HealthFloors(), BLOCK)
    overlapping = interpret.contrast_health_report(
        peaks, _ids(0), _ids(0) + _ids(1), HealthFloors(), BLOCK)

    assert disjoint.n_shared_peaks == 0
    assert overlapping.n_shared_peaks == len(_ids(0))
    assert overlapping.shared_blocks == disjoint.shared_blocks, (
        "the block counts are equal, which is why they cannot stand in for the peak overlap")


# --- regression: a family measured as zero must keep its composition row ------
def _rows_with_a_family_searched_and_never_retained() -> list[HitRecord]:
    """`_rows()` plus FAM_C, searched in every peak and retained in none.

    ``NO_SEQUENCE_MATCH`` rows name their family -- the real K562 substrate
    carries 11 of its 12 families that way -- so this is a measured zero, not an
    absence of measurement.
    """
    rows = _rows()
    for row in list(rows):
        if row.missingness is Missingness.USED and row.family_id == "FAM_A":
            rows.append(HitRecord(
                region_id=row.region_id, chrom=row.chrom, start=row.start, end=row.end,
                missingness=Missingness.NO_SEQUENCE_MATCH, input_scale=SCALE,
                lexicon_id=LEXICON, substrate_id=SUBSTRATE_ID,
                variant_id="UA_FAMC_00", family_id="FAM_C"))
    return rows


def test_a_family_every_query_peak_measured_as_zero_keeps_its_composition_row():
    """An omitted row is indistinguishable from "this family was never searched".

    `compose` took its family list from `Peak.family_hit_count`, which only gains
    a key through `add_used_hit`, so a family searched everywhere and retained
    nowhere vanished. On the real substrate a 2,176-peak query with no CTCF and
    no GATA hit emitted a ten-row composition beside a twelve-row effects table.
    """
    peaks = interpret.peak_universe(_rows_with_a_family_searched_and_never_retained(), BLOCK)
    rows = {c.family_id: c for c in interpret.compose(peaks, _ids(0))}

    assert set(rows) == {"FAM_A", "FAM_B", "FAM_C"}
    assert rows["FAM_C"].n_peaks_with_family == 0
    assert rows["FAM_C"].peak_share == 0.0
    assert rows["FAM_C"].mean_coefficient_per_peak == 0.0
    assert rows["FAM_C"].n_peaks_searched == len(_ids(0))


def test_every_effect_records_the_block_floor_its_own_estimator_enforced():
    """The two estimator paths treat the resampling-unit count differently, and
    until now nothing in the artifact said so.

    `bca-wild-cluster` refuses below `infer.MIN_ESTIMABLE_BLOCKS` whatever the run
    declared. The percentile path floors REPLICATES twice -- requested and
    estimable -- and never floors blocks, so a run that lowers `--floor-blocks`
    past the health gate gets a 95% percentile interval resampled from that many
    units, on data the other estimator would have refused. Whether it *should*
    floor blocks is a question about what a percentile block bootstrap needs, and
    it is not answered here; what is recorded is which floor, if any, the number
    in front of the reader cleared.
    """
    blocks = range(12)
    query = _query(region_ids=_ids(0, blocks), comparator_region_ids=_ids(1, blocks))

    percentile = interpret.interpret_query(
        _rows(), query, floors=HealthFloors(min_blocks=6), n_bootstrap=100, seed=1)
    assert percentile.effects
    for effect in percentile.effects:
        assert effect["n_blocks"] == 12
        assert effect["estimator_min_blocks"] is None, (
            "the percentile path applies no floor to the block count; None says that, "
            "and 0 would read as a floor that was met"
        )
        assert effect["n_blocks"] < infer_mod.MIN_ESTIMABLE_BLOCKS, (
            "the fixture must actually be a frame the other estimator would refuse, "
            "or this test is not about the asymmetry it names"
        )

    licensed = interpret.interpret_query(
        _rows(), _query(), floors=HealthFloors(min_blocks=6), n_bootstrap=100, seed=1,
        estimator=interpret.ESTIMATOR_BCA_WILD)
    assert licensed.effects
    for effect in licensed.effects:
        assert effect["estimator_min_blocks"] == infer_mod.MIN_ESTIMABLE_BLOCKS
        assert effect["n_blocks"] >= effect["estimator_min_blocks"]


def test_the_emitted_interpretation_says_which_schema_it_is(tmp_path):
    """A widened record that does not say it widened is two files a reader cannot tell apart.

    `estimator_min_blocks` was added to the effect record while `interpretation.json`
    was the one artifact here carrying no schema version -- so an old file and a new
    one differed by a missing key and nothing else, and "absent" would have had to be
    read as both "written before the field existed" and "the estimator applied no
    floor". Those are different facts and one of them is a finding.
    """
    written = interpret.interpret_query(
        _rows(), _query(), floors=HealthFloors(min_blocks=6), n_bootstrap=100, seed=1)
    payload = json.loads(written.write(tmp_path).read_text())

    assert payload["schema_version"] == interpret.INTERPRETATION_SCHEMA_VERSION
    assert interpret.INTERPRETATION_SCHEMA_VERSION != "1", (
        "version 1 is the unversioned files already written; a wider record reusing "
        "it would be indistinguishable from them"
    )
    assert "estimator_min_blocks" in payload["effects"][0], (
        "the field whose arrival motivated the version is not in the versioned record"
    )


def test_an_unnamed_family_is_not_promoted_to_a_measured_zero():
    """The sentinel is an absent assignment, not a family that measured zero.

    Admitting it would put `NA` in a composition table as a family, which is the
    failure `HitRecord.__post_init__` already refuses for USED rows.
    """
    peaks = interpret.peak_universe(_rows(), BLOCK)
    assert peaks[NO_MATCH_PEAK].families_measured == set()
    assert interpret.compose(peaks, [NO_MATCH_PEAK]) == []


# --- regression: the shipped README said FP-15's estimators were absent -------
def test_the_interpret_readme_does_not_call_an_implemented_estimator_unimplemented():
    """`interpret/README.md` ships in the wheel and said of the BCa interval and
    the wild cluster bootstrap-t "Neither is implemented", while both run and are
    selectable as `--estimator bca-wild-cluster`. A reader who pip-installed the
    package was told the licensed path did not exist.
    """
    from pathlib import Path

    readme = (Path(interpret.__file__).parent / "README.md").read_text(encoding="utf-8")
    assert "Neither is implemented" not in readme
    for spelling, recorded in interpret.ESTIMATOR_CHOICES.items():
        assert spelling in readme, f"--estimator {spelling} runs but the README never names it"
        assert recorded in readme, f"results record {recorded} but the README never names it"


# --- regression: the percentile path had no replicate floor -------------------
def test_a_percentile_interval_is_refused_below_the_replicate_floor():
    """`--bootstrap 1` emitted `[x, x]`: a zero-width 95% interval.

    B replicates resolve a tail no finer than 1/(B+1), so a 2.5% tail needs
    B >= 39; below it both endpoints are the extreme replicates and at B = 1 they
    are the same one. The interval was printed beside its point estimate with
    nothing saying it came from a single draw.
    """
    peaks = interpret.peak_universe(_rows(), BLOCK)
    with pytest.raises(interpret.InterpretError, match="below the preregistered floor of 39"):
        interpret.estimate_effects(peaks, _ids(0), _ids(1), "odd",
                                   n_bootstrap=1, seed=1, block_size=BLOCK)


def test_the_replicate_floor_itself_still_produces_a_real_interval():
    """The floor is the smallest count that works, not one past it."""
    peaks = interpret.peak_universe(_rows(), BLOCK)
    effects = interpret.estimate_effects(
        peaks, _ids(0), _ids(1), "odd",
        n_bootstrap=interpret.MIN_PERCENTILE_REPLICATES, seed=1, block_size=BLOCK)
    low, high = effects[0].ci
    assert low < high


# --- regression: --held-out silently emptied the comparator -------------------
def test_a_held_out_set_that_empties_the_comparator_names_the_split_not_the_baseline():
    """The refusal blamed the caller for something the caller had supplied.

    `CLUSTERED_WITH_SPLIT` intersects BOTH sides with the held-out set. A
    held-out list naming only query peaks therefore left `comparator_ids` empty,
    and the run failed with "a cross-condition effect needs a named baseline peak
    set" at a caller who had named one, with the split never mentioned.
    """
    half = range(N_BLOCKS // 2, N_BLOCKS)
    query = _query(
        selection_provenance=SelectionProvenance.CLUSTERED_WITH_SPLIT,
        held_out_region_ids=_ids(0, half),
    )
    with pytest.raises(interpret.InterpretError, match="retains none of the .* comparator peaks"):
        interpret.interpret_query(_rows(), query, n_bootstrap=50,
                                  floors=HealthFloors(min_blocks=N_BLOCKS // 4))


def test_the_held_out_note_reports_both_sides_it_restricted():
    """It announced a query-side restriction only, so the comparator's loss was
    invisible even when the run went on to succeed."""
    half = range(N_BLOCKS // 2, N_BLOCKS)
    query = _query(
        selection_provenance=SelectionProvenance.CLUSTERED_WITH_SPLIT,
        held_out_region_ids=_ids(0, half) + _ids(1, half),
    )
    result = interpret.interpret_query(_rows(), query, n_bootstrap=50,
                                       floors=HealthFloors(min_blocks=N_BLOCKS // 4))
    assert any("comparator peaks" in note for note in result.notes), result.notes


# --- regression: SUBSTRATE_CIRCULAR was silent --------------------------------
def test_substrate_circular_says_so_in_the_notes():
    """The flagship two-axis outcome produced no note at all.

    `output_mode` -- the field the CLI printed -- has no representation for it,
    so a FULL_INFERENCE run selected on `hit_coefficient` was indistinguishable
    in every emitted sentence from one selected on genomic position. The scope
    existed only as a JSON field beside a number people quote.
    """
    result = interpret.interpret_query(
        _rows(), _query(selection_feature_names=["hit_coefficient"]), n_bootstrap=50)

    assert result.claim_scope == ClaimScope.SUBSTRATE_CIRCULAR.value
    assert result.statistical_license == StatisticalLicense.FULL_INFERENCE.value
    circular = [n for n in result.notes if "SUBSTRATE_CIRCULAR" in n]
    assert circular, result.notes
    assert "hit_coefficient" in circular[0], "the note must name the feature that made it circular"


def test_an_external_selection_gets_no_circularity_note():
    """The falsification twin: the note must not fire on every query."""
    result = interpret.interpret_query(
        _rows(), _query(selection_feature_names=["gc_content"]), n_bootstrap=50)
    assert result.claim_scope == ClaimScope.EXTERNAL_STRUCTURE.value
    assert not [n for n in result.notes if "SUBSTRATE_CIRCULAR" in n]


# --- regression: zero overlap is a key mismatch, not thin coverage ------------
def test_zero_intersection_is_reported_as_a_key_mismatch_rather_than_thin_coverage():
    """`--peaks` accepts a BED, but matching is exact string equality on region_id.

    A 3-column BED is read as `chrom:start-end`, so against a table that spells
    the same peak `peak_000001` every submitted id misses. The only thing said
    was "intersection_coverage=0.0 < floor 0.9", which sends the reader looking
    for missing peaks rather than for the key the two sides disagree on.
    """
    peaks = interpret.peak_universe(_rows(), BLOCK)
    health = interpret.health_report(
        peaks, ["chr1:0-500", "chr1:1000-1500"], HealthFloors(), BLOCK)

    assert health.intersection_coverage == 0.0
    assert any("key mismatch" in f for f in health.floor_failures), health.floor_failures


def test_thin_but_nonzero_coverage_is_not_called_a_key_mismatch():
    """The falsification twin: a genuinely thin query keeps the plain message."""
    peaks = interpret.peak_universe(_rows(), BLOCK)
    health = interpret.health_report(
        peaks, [*_ids(0)[:2], "chr1:0-500", "chr1:1000-1500"], HealthFloors(), BLOCK)
    assert 0.0 < health.intersection_coverage < 0.9
    assert not any("key mismatch" in f for f in health.floor_failures)


# --------------------------------------------------------------------------- #
# The opportunity ledger, and the guard it finally gives a claim to check.
# --------------------------------------------------------------------------- #
def _ledger(rows, **over):
    from motifmultiverse.substrate import OpportunityLedger

    base = dict(
        substrate_id=SUBSTRATE_ID,
        n_opportunities=len(rows),
        n_retained=sum(1 for r in rows if str(r.missingness) == "used"),
        n_searched=sum(1 for r in rows if str(r.missingness) != "not_searched"),
        producer="test-freezer 0.0",
    )
    return OpportunityLedger(**{**base, **over})


def test_a_truthful_ledger_passes_and_records_what_it_checked():
    rows = _rows()
    log = GuardLog("interpret")
    interpret.verify_missingness_against_ledger(rows, _ledger(rows), guard_log=log)

    assert [o.guard_id for o in log.outcomes] == ["four_state_missingness"]
    assert log.outcomes[0].passed
    assert "test-freezer 0.0" in log.outcomes[0].subject, (
        "the subject must name who wrote the claim; that is why the claim is evidence"
    )


def test_the_ledger_catches_a_fill_into_an_undefined_row():
    """The founding failure, caught end to end rather than in a guard unit test.

    A value is written into a `no_sequence_match` row -- the thing a fill does --
    and the ledger, written before that happened, still says how many rows were
    retained. The recomputation moves; the claim does not; the run refuses.

    Note what is NOT relied on: the planted value is 0.7, not 0. A check for
    literal zeros passes this, which is exactly why the reference implementation's
    coverage figure corroborated its own error.
    """
    rows = _rows()
    ledger = _ledger(rows)
    filled = [
        replace(r, missingness=Missingness.USED, hit_coefficient=0.7,
                variant_id="UA_FILLED_00", family_id="FAM_A")
        if str(r.missingness) == "no_sequence_match" else r
        for r in rows
    ]
    assert any(str(r.missingness) == "used" and r.hit_coefficient == 0.7 for r in filled), (
        "the fixture must actually contain a filled row or this test proves nothing"
    )
    assert len(filled) == len(rows), "a fill promotes rows, it does not add them"

    with pytest.raises(guards_mod.GuardError, match="four_state_missingness"):
        interpret.verify_missingness_against_ledger(filled, ledger)


def test_a_ledger_for_another_substrate_is_refused():
    """A ledger that can be read beside any substrate is a claim about nothing."""
    import tempfile

    from motifmultiverse.substrate import (
        SubstrateError,
        read_opportunity_ledger,
        write_opportunity_ledger,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ledger.json"
        write_opportunity_ledger(_ledger(_rows(), substrate_id="b" * 64), path)
        with pytest.raises(SubstrateError, match="refusing to check one frozen run"):
            read_opportunity_ledger(path, substrate_id=SUBSTRATE_ID)


def test_a_ledger_whose_counts_are_impossible_is_refused():
    """Retained beyond searched is not a disagreement to be checked, it is nonsense."""
    from motifmultiverse.substrate import SubstrateError

    with pytest.raises(SubstrateError, match="cannot be retained without having been searched"):
        _ledger(_rows(), n_retained=10, n_searched=5)


def test_the_ledger_records_both_denominators_and_conflates_neither():
    """`retained/opportunities` and `searched/opportunities` are different questions.

    Redefining the guard's `defined` to match `peak_universe` -- which treats a
    searched-but-unretained opportunity as a measurement contributing zero -- was
    the tempting fix and would have destroyed the guard: a `defined` that counts
    `no_sequence_match` rows can no longer detect a fill that wrote a value into
    one. So both counts are recorded, under names that cannot be mistaken for each
    other, and the guard keeps its own arithmetic.
    """
    rows = _rows()
    ledger = _ledger(rows)
    assert ledger.n_searched > ledger.n_retained, (
        "the fixture must distinguish the two denominators"
    )
    assert ledger.retained_coverage == ledger.n_retained / ledger.n_opportunities
    assert ledger.retained_coverage != ledger.n_retained / ledger.n_searched


def test_a_reported_mean_does_not_depend_on_the_order_the_values_arrive_in():
    """Every reported effect, CI endpoint and composition mean moved with the interpreter.

    `validate.evaluate_stability` was fixed in fc461d4 for exactly this: builtin
    `sum` accumulates left to right, Python 3.12 switched it to Neumaier
    compensated summation and 3.11 did not, so a value summed with `sum` is not
    the same value on the two interpreters the classifiers claim. That fix
    covered one function; the accumulators feeding `interpret`'s reported numbers
    were still naive, and it was measured rather than argued -- one `interpret`
    run over identical input and seed reported `effect` and both `ci` endpoints
    as 0.7999999999999999 on 3.11 and 0.8 on 3.12. CI runs both interpreters and
    compares nothing, so nothing saw it.

    `math.fsum` is correctly rounded, which makes it order-independent, which is
    what makes it interpreter-independent. This pins the property in-process: the
    same values in two orders must give the same mean, and it must be the exactly
    rounded one. Under the naive sum the two orders below differ.
    """
    from motifmultiverse.interpret import _mean

    values = [1e16, 1.0, -1e16, 1.0]

    assert _mean(values) == _mean(list(reversed(values)))
    assert _mean(values) == 0.5, "the exact mean of [1e16, 1, -1e16, 1] is 2/4"
