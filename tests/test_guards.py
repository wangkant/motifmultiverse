"""Guard tests: a positive case AND a falsification case for every guard (T-15).

The falsification half is the point. A guard that has never been shown to fail is
not evidence -- in the reference implementation 5 framework guards all passed, yet
2 of them still passed under a row-shifted and a permuted lexicon index, and none
of the 5 could detect a reordered index. A vacuous guard and a correct guard are
indistinguishable in a report unless someone tries to break them.

Each test below corrupts the guard's OWN input by shifting, permuting or
reordering it, and asserts the guard rejects it.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

import motifmultiverse
from motifmultiverse import guards
from motifmultiverse.schema import MotifNode

SEED = 20260725


def _nodes(n: int = 6) -> list[MotifNode]:
    return [
        MotifNode(
            node_id=f"n{i}", model="modelA", readout="r1", context="promoter",
            metacluster="pos", denovo_pattern_id=f"pattern_{i}",
            variant_id=f"UA_FAM_{i:02d}", family_id="FAM",
            motif_length=12, seqlet_count=250,
            annotation_matches={"tomtom_q": 0.001},
        )
        for i in range(n)
    ]


# --------------------------------------------------------------- single_scale
def test_single_scale_passes_on_one_scale():
    recs = [{"input_scale": 33917} for _ in range(5)]
    assert guards.single_scale(recs).passed


def test_single_scale_FALSIFIED_by_mixed_scales():
    recs = [{"input_scale": 33917} for _ in range(4)] + [{"input_scale": 13277}]
    assert not guards.single_scale(recs).passed


def test_single_scale_FALSIFIED_by_missing_field():
    assert not guards.single_scale([{"input_scale": 1}, {}]).passed


# ---------------------------------------------------------- variant_id_unique
def test_variant_id_unique_passes():
    assert guards.variant_id_unique(_nodes()).passed


def test_variant_id_unique_FALSIFIED_by_permutation_collision():
    ns = _nodes()
    ns[3].variant_id = ns[0].variant_id          # a collision, as a permutation would create
    assert not guards.variant_id_unique(ns).passed


def test_variant_id_unique_FALSIFIED_by_row_shift():
    """Shift denovo_pattern_id by one row: identity is no longer 1:1."""
    ns = _nodes()
    keys = [n.denovo_pattern_id for n in ns]
    for n, k in zip(ns, keys[-1:] + keys[:-1], strict=True):
        n.denovo_pattern_id = k
    ns[0].denovo_pattern_id = ns[1].denovo_pattern_id
    assert not guards.variant_id_unique(ns).passed


# ------------------------------------------------------------- no_key_parsing
def test_no_key_parsing_passes_on_clean_source():
    src = "def f(table, ident):\n    return table[ident.value]\n"
    assert guards.no_key_parsing(src).passed


def test_no_key_parsing_FALSIFIED_by_slicing_an_identifier():
    src = "def f(variant_id):\n    return variant_id.split('_')[1]\n"
    assert not guards.no_key_parsing(src).passed


def test_no_key_parsing_FALSIFIED_by_prefix_test():
    src = "def f(pattern_key):\n    return pattern_key.startswith('AG_')\n"
    assert not guards.no_key_parsing(src).passed


def test_no_key_parsing_success_detail_is_labelled_heuristic():
    """The guard's own success message must not overclaim what an AST scan proves."""
    src = "def f(table, ident):\n    return table[ident.value]\n"
    assert guards.no_key_parsing(src).detail == "heuristic scan passed"


def test_no_key_parsing_does_not_catch_aliasing_or_dataflow():
    """Documents a real gap, it does not just assert one.

    This is a syntactic AST scan over literal identifier names, not a dataflow
    analysis. If the value is renamed before being sliced, or reaches the slice
    through a parameter or attribute the scan does not track back to one of the
    watched names, the same defect the guard exists to catch passes silently. A
    passing result here certifies only "no watched identifier is sliced or
    prefix-tested directly by its own name" -- it does not certify BA-11 for the
    file as a whole, which is why the corresponding frozen principle is not
    classified ENFORCED on the strength of this guard alone.
    """
    src = (
        "def f(variant_id):\n"
        "    alias = variant_id\n"
        "    return alias.split('_')[1]\n"
    )
    result = guards.no_key_parsing(src)
    assert result.passed  # the aliased slice is invisible to this heuristic


# ------------------------------------------------------ four_state_missingness
def test_four_state_missingness_passes():
    rows = [{"statistic": 1.0, "missingness": "used"},
            {"statistic": None, "missingness": "not_searched"}]
    result = guards.four_state_missingness(
        rows, claimed_coverage=0.5, claimed_defined=1, claimed_total=2)
    assert result.passed


def test_four_state_missingness_FALSIFIED_by_zero_collapse():
    rows = [{"statistic": 1.0, "missingness": "used"},
            {"statistic": 0, "missingness": "no_sequence_match"}]
    result = guards.four_state_missingness(
        rows, claimed_coverage=0.5, claimed_defined=1, claimed_total=2)
    assert not result.passed


def test_four_state_missingness_FALSIFIED_by_absent_state():
    result = guards.four_state_missingness(
        [{"statistic": 1.0}], claimed_coverage=1.0, claimed_defined=1, claimed_total=1)
    assert not result.passed


def test_four_state_guard_rejects_coverage_computed_after_fill():
    """The project's founding failure: a coverage figure computed AFTER a fill.

    Neither row contains a numeric zero, so the old zero-collapse check would have
    passed this. The claimed coverage of 1.0 (and claimed_defined=2) can only be
    caught by recomputing `defined`/`total`/coverage from the raw rows and
    comparing -- one row is genuinely `not_searched` with no value at all, so the
    true coverage is 0.5, not 1.0.
    """
    rows = [
        {"missingness": "used", "statistic": 1.0},
        {"missingness": "not_searched", "statistic": None},
    ]
    result = guards.four_state_missingness(
        rows, claimed_coverage=1.0, claimed_defined=2, claimed_total=2)
    assert not result.passed
    assert "claimed coverage" in result.detail


def test_four_state_missingness_FALSIFIED_by_wrong_claimed_total():
    rows = [{"statistic": 1.0, "missingness": "used"},
            {"statistic": None, "missingness": "not_searched"}]
    result = guards.four_state_missingness(
        rows, claimed_coverage=0.5, claimed_defined=1, claimed_total=3)
    assert not result.passed
    assert "claimed coverage" in result.detail


# --------------------------------------------------- no_cross_model_cwm_avg
def test_no_cross_model_cwm_avg_passes():
    ops = [{"op": "mean", "group_by": ["model", "readout", "metacluster", "family_id"]}]
    assert guards.no_cross_model_cwm_avg(ops).passed


def test_no_cross_model_cwm_avg_FALSIFIED_when_model_not_held_fixed():
    ops = [{"op": "mean", "group_by": ["readout", "metacluster"]}]
    assert not guards.no_cross_model_cwm_avg(ops).passed


# ------------------------------------------------------------ sign_alignment
def test_sign_alignment_passes_on_unsigned_ppm():
    assert guards.sign_alignment([{"registered_on": "unsigned_ppm"}]).passed


def test_sign_alignment_FALSIFIED_by_signed_registration():
    assert not guards.sign_alignment([{"registered_on": "signed_cwm"}]).passed


def test_sign_alignment_FALSIFIED_when_signed_similarity_picks_offset():
    a = [{"registered_on": "unsigned_ppm", "signed_similarity_used_for_registration": True}]
    assert not guards.sign_alignment(a).passed


# ------------------------------------------------------- interaction_required
def test_interaction_required_passes():
    claims = [{"id": "c1", "is_specificity_claim": True, "interaction_ci": (0.2, 0.6)}]
    assert guards.interaction_required(claims).passed


def test_interaction_required_FALSIFIED_by_ci_containing_zero():
    claims = [{"id": "c1", "is_specificity_claim": True, "interaction_ci": (-0.1, 0.6)}]
    assert not guards.interaction_required(claims).passed


def test_interaction_required_FALSIFIED_by_difference_of_significance():
    claims = [{"id": "c1", "is_specificity_claim": True, "interaction_ci": (0.2, 0.6),
               "derived_from": "difference_of_significance"}]
    assert not guards.interaction_required(claims).passed


# -------------------------------------------------------- estimability_floor
def test_estimability_floor_passes():
    assert guards.estimability_floor([{"id": "a", "n": 500, "ci": (0.1, 0.3), "reference": 0.2}]).passed


def test_estimability_floor_FALSIFIED_by_small_n():
    assert not guards.estimability_floor([{"id": "a", "n": 12}]).passed


def test_estimability_floor_FALSIFIED_when_ci_holds_zero_and_reference():
    cells = [{"id": "a", "n": 100, "ci": (-0.5, 0.9), "reference": 0.4}]
    assert not guards.estimability_floor(cells).passed


# ------------------------------------------------------------ stratum_parity
def test_stratum_parity_passes():
    cells = [{"stratum_rules": {"tss_context": "unified_tss_distance"}} for _ in range(3)]
    assert guards.stratum_parity(cells).passed


def test_stratum_parity_FALSIFIED_by_two_rules_for_one_variable():
    cells = [{"stratum_rules": {"tss_context": "unified_tss_distance"}},
             {"stratum_rules": {"tss_context": "per_arm_recomputed"}}]
    assert not guards.stratum_parity(cells).passed


# ---------------------------------------------------------- short_motif_flag
def test_short_motif_flag_passes():
    assert guards.short_motif_flag(_nodes()).passed


@pytest.mark.parametrize("field,value", [
    ("motif_length", 5), ("seqlet_count", 40),
])
def test_short_motif_flag_FALSIFIED_by_unflagged_weak_motif(field, value):
    ns = _nodes()
    setattr(ns[2], field, value)
    assert not guards.short_motif_flag(ns).passed


def test_short_motif_flag_FALSIFIED_by_unflagged_weak_tomtom_q():
    ns = _nodes()
    ns[1].annotation_matches = {"tomtom_q": 0.4}
    assert not guards.short_motif_flag(ns).passed


# -------------------------------------------------------- single_family_layer
def test_single_family_layer_passes():
    strata = [{"id": "s", "families": ["FAM"], "status": "NOT_ESTIMABLE"}]
    assert guards.single_family_layer(strata).passed


def test_single_family_layer_FALSIFIED_by_share_of_one():
    strata = [{"id": "s", "families": ["FAM"], "within_peak_share": 1.0}]
    assert not guards.single_family_layer(strata).passed


# ------------------------------------------------- selection_provenance_declared
def test_selection_provenance_declared_passes():
    qs = [{"query_id": "q1", "selection_provenance": "EXTERNAL", "output_mode": "FULL_INFERENCE"}]
    assert guards.selection_provenance_declared(qs).passed


def test_selection_provenance_declared_passes_when_undeclared_is_most_conservative():
    qs = [{"query_id": "q1", "selection_provenance": "DECLARATION_MISSING",
           "output_mode": "DESCRIPTIVE_ONLY_UNVERIFIABLE_CONDITIONING"}]
    assert guards.selection_provenance_declared(qs).passed


def test_selection_provenance_declared_FALSIFIED_by_missing_field():
    assert not guards.selection_provenance_declared([{"query_id": "q1",
                                                      "output_mode": "FULL_INFERENCE"}]).passed


def test_selection_provenance_declared_FALSIFIED_by_permissive_default():
    """The failure that actually happens: undeclared, silently treated as EXTERNAL."""
    qs = [{"query_id": "q1", "selection_provenance": None, "output_mode": "FULL_INFERENCE"}]
    assert not guards.selection_provenance_declared(qs).passed


def test_selection_provenance_declared_FALSIFIED_by_unknown_grade_running_full():
    qs = [{"query_id": "q1", "selection_provenance": "SOME_FUTURE_GRADE",
           "output_mode": "FULL_INFERENCE"}]
    assert not guards.selection_provenance_declared(qs).passed


def test_selection_provenance_declared_FALSIFIED_by_upgraded_mode():
    """A declared grade may not be dispatched to a stronger mode than its own."""
    qs = [{"query_id": "q1", "selection_provenance": "CLUSTERED_NO_SPLIT",
           "output_mode": "FULL_INFERENCE"}]
    assert not guards.selection_provenance_declared(qs).passed


# ------------------------------------------------------------ health_before_effect
def _report(**over):
    base = {
        "health": {"intersection_coverage": 0.98, "n_blocks": 120, "explained_fraction": 0.71},
        "emitted_order": ["health", "composition", "effects"],
        "floor_failures": [],
        "effects": [{"id": "e1"}],
        "interpretation_emitted": True,
    }
    base.update(over)
    return base


def test_health_before_effect_passes():
    assert guards.health_before_effect(_report()).passed


def test_health_before_effect_FALSIFIED_by_effect_after_failed_floor():
    r = _report(floor_failures=["n_blocks=4 < floor 30"])
    assert not guards.health_before_effect(r).passed


def test_health_before_effect_FALSIFIED_by_effect_before_health():
    r = _report(emitted_order=["effects", "health"])
    assert not guards.health_before_effect(r).passed


def test_health_before_effect_FALSIFIED_by_missing_health_number():
    r = _report(health={"intersection_coverage": 0.98, "n_blocks": 120})
    assert not guards.health_before_effect(r).passed


def test_health_before_effect_FALSIFIED_by_silent_null_health_number():
    r = _report(health={"intersection_coverage": None, "n_blocks": 120,
                        "explained_fraction": 0.71})
    assert not guards.health_before_effect(r).passed


# --------------------------------------------------------------- comparator_declared
def test_comparator_declared_passes():
    claims = [{"id": "GATA_vs_gc_matched", "comparator_id": "gc_matched_negatives"}]
    assert guards.comparator_declared(claims).passed


def test_comparator_declared_FALSIFIED_by_missing_baseline():
    assert not guards.comparator_declared([{"id": "GATA", "comparator_id": None}]).passed


def test_comparator_declared_FALSIFIED_by_one_effect_two_baselines():
    """U-13: the same effect value carrying two different baselines.

    This is the shape of the real failure. The measurements were identical; one
    baseline was the unselected universe and the other a residual subset with the
    relevant peaks already removed, and the same numbers read as 'replicates
    exactly' and 'four times stronger, prediction falsified'. The tool must refuse
    to present that as one number.
    """
    effect = 0.0315
    claims = [
        {"id": "CTCF_island_effect", "effect": effect, "comparator_id": "unselected_universe"},
        {"id": "CTCF_island_effect", "effect": effect, "comparator_id": "residual_after_removal"},
    ]
    result = guards.comparator_declared(claims)
    assert not result.passed
    assert "more than one baseline" in result.detail


def test_comparator_declared_FALSIFIED_by_a_list_of_baselines_in_one_claim():
    claims = [{"id": "CTCF", "comparator_id": ["unselected_universe", "residual_after_removal"]}]
    assert not guards.comparator_declared(claims).passed


# ------------------------------------------------------ index_order_matches_loader
LOADER_ORDER = ["pos_patterns.pattern_0", "pos_patterns.pattern_1",
                "neg_patterns.pattern_0", "neg_patterns.pattern_1"]


def test_index_order_matches_loader_passes():
    assert guards.index_order_matches_loader(list(LOADER_ORDER), LOADER_ORDER).passed


def test_index_order_matches_loader_FALSIFIED_by_metacluster_ascending_index():
    """The real defect: an index sorted neg-before-pos while the loader emits pos first.

    In the reference implementation this was invisible because one model had no
    negative motifs, so both orders coincided -- and every positional read against
    the other model's index would have been off by the size of its negative set.
    """
    index = LOADER_ORDER[2:] + LOADER_ORDER[:2]        # neg < pos, as a sort would give
    result = guards.index_order_matches_loader(index, LOADER_ORDER)
    assert not result.passed
    assert "different order" in result.detail


def test_index_order_matches_loader_FALSIFIED_by_a_missing_motif():
    assert not guards.index_order_matches_loader(LOADER_ORDER[:-1], LOADER_ORDER).passed


def test_index_order_matches_loader_FALSIFIED_by_a_duplicate_name():
    dupe = [LOADER_ORDER[0]] + LOADER_ORDER[:3]
    assert not guards.index_order_matches_loader(dupe, LOADER_ORDER).passed


def test_index_order_matches_loader_FALSIFIED_by_an_empty_side():
    assert not guards.index_order_matches_loader([], LOADER_ORDER).passed


# ------------------------------------------------------------------ coverage
def test_every_guard_has_a_falsification_test():
    """Meta-test: no guard may ship without a test that makes it fail."""
    import pathlib
    src = pathlib.Path(__file__).read_text()
    missing = [g for g in guards.ALL_GUARDS if f"def test_{g}_FALSIFIED" not in src]
    assert not missing, f"guards with no falsification test: {missing}"


def test_the_package_passes_its_own_no_key_parsing_guard():
    """The rule binds this repository's source, not only its users'.

    A constraint the tool enforces on other people's code and not on its own is
    the prose-only failure wearing a different hat.
    """
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders = [f"{p}: {guards.no_key_parsing(p.read_text()).detail}"
                 for p in sorted(src.rglob("*.py"))
                 if not guards.no_key_parsing(p.read_text()).passed]
    assert not offenders, offenders


def test_random_reorder_does_not_silently_change_a_verdict():
    """Order-independent guards are declared as such IN ADVANCE (G-E2d analogue)."""
    rng = random.Random(SEED)
    ns = _nodes(8)
    before = guards.variant_id_unique(ns).passed
    shuffled = ns[:]
    rng.shuffle(shuffled)
    assert guards.variant_id_unique(shuffled).passed == before


# ------------------------------------------------------------ package shape
def test_guards_is_a_package_and_no_shadowing_module_exists():
    """A `guards.py` beside the `guards/` package would win the import race silently.

    Whichever of the two loads first (an artifact of `sys.path` order, not of
    anything declared) becomes `motifmultiverse.guards`; new code added to the
    other one would never run and no import error would announce it. This test
    fails if `guards.py` and `guards/` ever coexist, and separately fails if the
    package is ever demoted to a bare module -- it checks the filesystem
    directly, not what happened to get imported this run.
    """
    package_root = Path(motifmultiverse.__file__).parent
    assert (package_root / "guards" / "__init__.py").is_file(), (
        "guards must remain a package; guard modules live under guards/ and are "
        "re-exported from guards/__init__.py"
    )
    assert not (package_root / "guards.py").exists(), (
        "a sibling guards.py alongside the guards/ package makes which module is "
        "imported depend on path order; only one of the two may ever exist"
    )
