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
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

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
        rows, claimed_coverage=0.5, claimed_defined=1, claimed_total=2, value_key="statistic")
    assert result.passed


def test_four_state_missingness_FALSIFIED_by_zero_collapse():
    rows = [{"statistic": 1.0, "missingness": "used"},
            {"statistic": 0, "missingness": "no_sequence_match"}]
    result = guards.four_state_missingness(
        rows, claimed_coverage=0.5, claimed_defined=1, claimed_total=2, value_key="statistic")
    assert not result.passed


def test_four_state_missingness_FALSIFIED_by_absent_state():
    result = guards.four_state_missingness(
        [{"statistic": 1.0}], claimed_coverage=1.0, claimed_defined=1, claimed_total=1, value_key="statistic")
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
        rows, claimed_coverage=1.0, claimed_defined=2, claimed_total=2, value_key="statistic")
    assert not result.passed
    assert "claimed coverage" in result.detail


def test_four_state_missingness_FALSIFIED_by_wrong_claimed_total():
    rows = [{"statistic": 1.0, "missingness": "used"},
            {"statistic": None, "missingness": "not_searched"}]
    result = guards.four_state_missingness(
        rows, claimed_coverage=0.5, claimed_defined=1, claimed_total=3, value_key="statistic")
    assert not result.passed
    assert "claimed coverage" in result.detail


def test_four_state_missingness_accepts_a_coverage_rounded_for_display():
    """A coverage legitimately rounded to 4 decimal places for display must not
    false-fail against a full-precision recomputation.

    2/3 recomputes to 0.6666666666666666; a report that displays "0.6667" is not
    lying, and an exact-match tolerance (the old rel_tol=1e-9) would reject it. A
    guard that cries wolf on correct input gets disabled by the next maintainer,
    which is worse than a slightly weaker guard.
    """
    rows = [{"statistic": 1.0, "missingness": "used"},
            {"statistic": 1.0, "missingness": "used"},
            {"statistic": None, "missingness": "no_sequence_match"}]
    result = guards.four_state_missingness(
        rows, claimed_coverage=0.6667, claimed_defined=2, claimed_total=3, value_key="statistic")
    assert result.passed


def test_four_state_missingness_still_rejects_a_mismatch_beyond_rounding():
    """The tolerance must not be widened so far that a genuine mismatch slips
    through. 0.67 is a coarser (2-decimal) rounding of 2/3 than this guard's
    display convention (4 decimals) admits, and must still be caught -- alongside
    the project's founding failure (coverage 1.0 claimed when rows say 0.5),
    which the pre-existing
    `test_four_state_guard_rejects_coverage_computed_after_fill` already covers.
    """
    rows = [{"statistic": 1.0, "missingness": "used"},
            {"statistic": 1.0, "missingness": "used"},
            {"statistic": None, "missingness": "no_sequence_match"}]
    result = guards.four_state_missingness(
        rows, claimed_coverage=0.67, claimed_defined=2, claimed_total=3, value_key="statistic")
    assert not result.passed
    assert "claimed coverage" in result.detail


# --------------------------------------------------- no_cross_model_cwm_avg
def test_no_cross_model_cwm_avg_passes():
    ops = [{"op": "mean", "group_by": ["model", "readout", "metacluster", "family_id"]}]
    assert guards.no_cross_model_cwm_avg(ops).passed


def test_no_cross_model_cwm_avg_FALSIFIED_when_model_not_held_fixed():
    ops = [{"op": "mean", "group_by": ["readout", "metacluster"]}]
    assert not guards.no_cross_model_cwm_avg(ops).passed


# --------------------------------------------------- no_cross_estimand_pooling
def test_no_cross_estimand_pooling_passes_within_one_estimand():
    summaries = [{"group_key": "est_a|CTCF", "cell_ids": ["c1", "c2"]}]
    specs = {"c1": {"estimand_id": "est_a"}, "c2": {"estimand_id": "est_a"}}
    assert guards.no_cross_estimand_pooling(summaries, specs).passed


def test_no_cross_estimand_pooling_FALSIFIED_when_two_baselines_are_averaged():
    """The failure a specification multiverse is most likely to commit.

    The two cells differ only in their baseline population -- the same family, the
    same lexicon, the same estimator -- which is exactly the case where the spread
    looks like robustness and is actually two answers to two questions.
    """
    summaries = [{"group_key": "CTCF across specifications", "cell_ids": ["c1", "c2"]}]
    specs = {"c1": {"estimand_id": "est_complement"},
             "c2": {"estimand_id": "est_gc_matched"}}
    result = guards.no_cross_estimand_pooling(summaries, specs)
    assert not result.passed
    assert "est_complement" in result.detail and "est_gc_matched" in result.detail


def test_no_cross_estimand_pooling_FALSIFIED_when_a_cell_is_not_in_the_manifest():
    """A summary is checked against the manifest, so an unplanned cell is a refusal.

    Skipping it would let a summariser evade the guard by naming cells the
    manifest never planned: whatever those cells' estimands were, nothing here can
    establish that they were one.
    """
    summaries = [{"group_key": "est_a|CTCF", "cell_ids": ["c1", "ghost"]}]
    assert not guards.no_cross_estimand_pooling(
        summaries, {"c1": {"estimand_id": "est_a"}}).passed


def test_no_cross_estimand_pooling_FALSIFIED_by_a_summary_naming_no_cells():
    """A group that names no members cannot be shown to stay within one estimand."""
    assert not guards.no_cross_estimand_pooling(
        [{"group_key": "empty", "cell_ids": []}], {}).passed


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
def _guard_id_from_test_name(name: str) -> str | None:
    """Extract the guard id a ``test_<gid>_FALSIFIED...`` function names, or None."""
    marker = "_FALSIFIED"
    if not name.startswith("test_") or marker not in name:
        return None
    candidate = name[len("test_"):name.index(marker)]
    return candidate if candidate in guards.ALL_GUARDS else None


def _expand_parametrized_calls(test_fn: Any) -> list[Callable[[], None]]:
    """Return zero-argument callables that run ``test_fn`` for every parameter set.

    A plain test function needs no expansion. A ``@pytest.mark.parametrize``’d
    one is called directly here (bypassing pytest's own collection), once per
    row of its argvalues, using the marker's own argnames/argvalues -- so this
    works without pytest fixtures or a live test session.
    """
    marks = [m for m in getattr(test_fn, "pytestmark", ()) if m.name == "parametrize"]
    if not marks:
        return [test_fn]
    argnames, argvalues = marks[0].args[0], marks[0].args[1]
    names = [n.strip() for n in argnames.split(",")] if isinstance(argnames, str) else list(argnames)
    calls = []
    for row in argvalues:
        values = row if isinstance(row, (tuple, list)) else (row,)
        kwargs = dict(zip(names, values, strict=True))
        calls.append(lambda test_fn=test_fn, kwargs=kwargs: test_fn(**kwargs))
    return calls


def _guards_confirmed_failing_by(namespace: Mapping[str, Any], guard_ids: Iterable[str]) -> set[str]:
    """Run every ``test_<gid>_FALSIFIED*`` callable in ``namespace`` with its guard spied on.

    Returns the subset of ``guard_ids`` for which some such test actually
    produced a ``GuardResult`` with ``passed=False`` *from that guard's own
    function* while running. This is the check that distinguishes a real
    falsification test from a same-named stub: T-15's own standard (a guard
    that has never been shown to fail is not evidence) applies to the tests
    that are supposed to prove it, not only to the guards themselves. A test
    whose body is ``pass``, or that asserts something unrelated and never
    calls the guard, cannot land its guard id in the returned set no matter
    what its name promises.
    """
    guard_ids = set(guard_ids)
    confirmed: set[str] = set()
    for name, obj in namespace.items():
        gid = _guard_id_from_test_name(name)
        if gid is None or gid not in guard_ids or gid in confirmed or not callable(obj):
            continue
        original = getattr(guards, gid)
        observed = False

        def _spy(*args, __original=original, **kwargs):
            nonlocal observed
            result = __original(*args, **kwargs)
            if not result.passed:
                observed = True
            return result

        setattr(guards, gid, _spy)
        try:
            for call in _expand_parametrized_calls(obj):
                call()
        finally:
            setattr(guards, gid, original)
        if observed:
            confirmed.add(gid)
    return confirmed


def test_every_guard_has_a_falsification_test():
    """Meta-test: no guard may ship without a test that DEMONSTRATES a failure.

    A prior version of this check was `f"def test_{g}_FALSIFIED" not in src` --
    a pure substring search over the file's own text. A reviewer fooled it: they
    deleted the real body of a falsification test and replaced it with a
    same-named no-op, and the check still reported `missing=[]`, because the
    string `"def test_single_scale_FALSIFIED..."` was still on disk. That is
    exactly the defect class this task exists to close in the guards
    themselves (a coverage figure that was never compared against anything
    supplies its own evidence of correctness) -- so it cannot be left standing
    in the test that is supposed to prove the guards are covered.

    This version actually CALLS every `test_<gid>_FALSIFIED*` function with its
    guard spied on, and only counts a guard as covered if some call produced a
    real `GuardResult(passed=False)` from that guard. See
    `test_meta_test_rejects_a_correctly_named_but_vacuous_falsification_test`
    for the proof that this version resists the same attack.
    """
    confirmed = _guards_confirmed_failing_by(globals(), guards.ALL_GUARDS)
    missing = sorted(set(guards.ALL_GUARDS) - confirmed)
    assert not missing, (
        f"guards with no test that demonstrates a failing GuardResult: {missing}"
    )


def test_meta_test_rejects_a_correctly_named_but_vacuous_falsification_test():
    """Reproduces the reviewer's attack and proves the CURRENT check rejects it.

    Empirically, before this fix, temporarily replacing the body of
    `test_single_scale_FALSIFIED_by_mixed_scales` with a bare `pass` left
    `test_every_guard_has_a_falsification_test` passing -- the old substring
    check could not tell a real falsification test from a stub with the right
    name. This test reproduces that exact attack shape (right name, empty
    body, the guard is never called) as a synthetic namespace entry and
    asserts the shared verification helper -- the one the real meta-test now
    calls -- correctly refuses to count it.
    """
    def test_single_scale_FALSIFIED_by_mixed_scales():
        pass  # right name; asserts nothing; never calls guards.single_scale

    attacked_namespace = {
        "test_single_scale_FALSIFIED_by_mixed_scales": test_single_scale_FALSIFIED_by_mixed_scales,
    }
    confirmed = _guards_confirmed_failing_by(attacked_namespace, {"single_scale"})
    assert "single_scale" not in confirmed, (
        "a same-named stub that never calls the guard must not count as a "
        "falsification test"
    )


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


# --- the guard named for four states must check that there are four -----------
def test_four_state_missingness_rejects_a_state_outside_the_four():
    """`missingness: "banana"` used to count as not-used and pass.

    A recomputation that agrees with a claim derived from the same nonsense is
    self-corroborating, which is the failure mode this whole module exists for.
    """
    rows = [{"statistic": 1.0, "missingness": "used"},
            {"statistic": None, "missingness": "banana"}]
    result = guards.four_state_missingness(
        rows, claimed_coverage=0.5, claimed_defined=1, claimed_total=2,
        value_key="statistic")
    assert not result.passed
    assert "outside the four states" in result.detail


def test_four_state_missingness_fails_when_the_value_column_is_absent():
    """A guard that no-ops on its own project's data certifies rather than checks.

    `value_key` defaulted to "statistic" -- the name used by this file's fixtures,
    not by any artifact the project produces. On real hit rows (`hit_coefficient`)
    `.get` returned None, `None == 0` was False, and the zero-collapse check passed
    on everything. It is now required, and a column no row carries fails.
    """
    rows = [{"hit_coefficient": 1.0, "missingness": "used"},
            {"hit_coefficient": None, "missingness": "not_searched"}]
    result = guards.four_state_missingness(
        rows, claimed_coverage=0.5, claimed_defined=1, claimed_total=2,
        value_key="statistic")
    assert not result.passed
    assert "no row carries the value column" in result.detail


def test_four_state_missingness_catches_a_fill_on_the_real_column_name():
    """The check the default was silently skipping, on the project's own column."""
    rows = [{"hit_coefficient": 1.0, "missingness": "used"},
            {"hit_coefficient": 0, "missingness": "no_sequence_match"}]
    result = guards.four_state_missingness(
        rows, claimed_coverage=0.5, claimed_defined=1, claimed_total=2,
        value_key="hit_coefficient")
    assert not result.passed
    assert "collapsed to 0" in result.detail


def test_every_guard_is_called_or_declared_pending():
    """A guard that is defined, exported and never invoked reads as protection.

    Seven of the fifteen were in that position, `four_state_missingness` -- the
    guard for this project's founding failure -- among them, while
    docs/CONSTRAINTS.md labelled principles ENFORCED partly on their strength.
    Being silent about that is the same shape as the failure. Each guard must now
    either have a call site in src/ or say, in guards.GUARDS_AWAITING_INPUT, which
    input it is waiting for.
    """
    import re
    from pathlib import Path

    import motifmultiverse
    from motifmultiverse.guards import GUARDS_AWAITING_INPUT

    src = Path(motifmultiverse.__file__).resolve().parent
    import inspect
    names = [n for n in guards.__all__
             if inspect.isfunction(getattr(guards, n, None)) and n != "run_all"]
    body = "\n".join(
        p.read_text() for p in src.rglob("*.py") if p.name != "__init__.py" or "guards" not in p.parts
    )
    orphans, stale = [], []
    for name in names:
        called = bool(re.search(rf"\bguards\.{name}\b|\b{name}\(", body))
        declared = name in GUARDS_AWAITING_INPUT
        if not called and not declared:
            orphans.append(name)
        if called and declared:
            stale.append(name)
    assert not orphans, (
        "guards with no call site and no declared pending input: " + ", ".join(orphans)
    )
    assert not stale, (
        "guards listed as awaiting input but actually called: " + ", ".join(stale)
    )


def test_guards_awaiting_input_each_say_what_they_wait_for():
    """Every pending entry says all three things, not just that it is pending.

    The previous version of this test accepted any string over forty characters,
    which "waiting for the right input from a future stage" satisfies while telling
    a reader nothing. An entry now has to name the artifact that comes closest, the
    failure that wiring it there would cause, and the thing whose existence closes
    it -- the three parts of `PendingGuardInput` -- and each has to be a sentence.
    """
    from motifmultiverse.guards import GUARDS_AWAITING_INPUT, PendingGuardInput

    assert GUARDS_AWAITING_INPUT, "the pending registry should not be silently emptied"
    for name, entry in GUARDS_AWAITING_INPUT.items():
        assert name in guards.__all__, f"{name} is not a guard"
        assert isinstance(entry, PendingGuardInput), (
            f"{name}: a pending entry is a PendingGuardInput, not a bare string; the "
            "three parts exist so the entry can be checked rather than graded"
        )
        assert "." in entry.nearest_artifact, (
            f"{name}: nearest_artifact must be a dotted name of something this release "
            f"emits, got {entry.nearest_artifact!r}"
        )
        for field in ("why_not_a_call_site", "closes_when"):
            assert len(getattr(entry, field)) > 40, f"{name}.{field}: a phrase is not a reason"
        assert entry.nearest_artifact in str(entry), (
            f"{name}: str() must still render one readable sentence for a reader that "
            "printed the old string value"
        )


def test_run_all_can_actually_run_every_guard():
    """`run_all` called every guard with one positional argument.

    Two guards take more than one: `four_state_missingness` needs the claimed
    coverage/defined/total it exists to recompute against, and
    `index_order_matches_loader` needs both orders it compares. Handing each a
    single value raised `TypeError` out of the entry point whose whole promise is
    to run everything, so the "run every guard" call could not run the guard for
    this project's founding failure.

    The payload table is checked against `ALL_GUARDS` rather than listed loosely,
    so a new guard cannot be added without deciding what `run_all` must feed it.
    """
    from motifmultiverse.guards import ALL_GUARDS, run_all

    payloads = {
        "single_scale": [{"input_scale": 12}],
        "variant_id_unique": [{"variant_id": "v1", "denovo_pattern_id": "p1"}],
        "no_key_parsing": "value = 1\n",
        "four_state_missingness": {
            "rows": [{"missingness": "used", "coefficient": 1.0}],
            "claimed_coverage": 1.0, "claimed_defined": 1, "claimed_total": 1,
            "value_key": "coefficient",
        },
        "no_cross_model_cwm_avg": [],
        "no_cross_estimand_pooling": {
            "summaries": [{"group_key": "est_a|CTCF", "cell_ids": ["cell_1"]}],
            "specifications": {"cell_1": {"estimand_id": "est_a"}},
        },
        "sign_alignment": [{"registered_on": "unsigned_ppm"}],
        "interaction_required": [],
        "estimability_floor": [],
        "stratum_parity": [],
        "short_motif_flag": [],
        "single_family_layer": [],
        "selection_provenance_declared": [],
        "health_before_effect": {
            "health": {"intersection_coverage": 1.0, "n_blocks": 40,
                       "explained_fraction": 0.9},
            "emitted_order": ["health"],
        },
        "comparator_declared": [],
        "index_order_matches_loader": {"index_names": ["a", "b"],
                                       "loader_names": ["a", "b"]},
    }
    assert set(payloads) == set(ALL_GUARDS), "the payload table drifted from ALL_GUARDS"

    results = run_all(payloads)
    assert [r.guard_id for r in results] == list(ALL_GUARDS)
    assert all(r.passed for r in results), [r for r in results if not r.passed]


def test_run_all_still_reports_a_guard_that_fails_on_its_multi_argument_input():
    """Reaching the guard is not the point unless the guard can still say no.

    A repair that swallowed the argument mismatch -- returning a passing result,
    or skipping the guard -- would make `run_all` green for the same reason the
    old one was red, so the multi-argument path is exercised with input the guard
    must reject.
    """
    from motifmultiverse.guards import run_all

    results = run_all({
        # The claim is 1.0 coverage over rows that are half undefined: the
        # founding failure, in the shape the guard was written to catch.
        "four_state_missingness": {
            "rows": [{"missingness": "used", "coefficient": 1.0},
                     {"missingness": "no_sequence_match", "coefficient": 0.4}],
            "claimed_coverage": 1.0, "claimed_defined": 2, "claimed_total": 2,
            "value_key": "coefficient",
        },
        "index_order_matches_loader": {"index_names": ["neg-1", "pos-1"],
                                       "loader_names": ["pos-1", "neg-1"]},
    })
    assert [(r.guard_id, r.passed) for r in results] == [
        ("four_state_missingness", False), ("index_order_matches_loader", False),
    ]


def test_run_all_refuses_a_multi_argument_guard_given_a_bare_value():
    """A wrong-shaped input is a caller error, not a data violation.

    Returning a failed `GuardResult` would let "you passed the wrong thing" and
    "the artifact broke the rule" arrive at a report looking identical, so the
    shape error is raised and names the arguments the guard wanted.
    """
    import pytest

    from motifmultiverse.guards import run_all

    with pytest.raises(TypeError, match="index_names, loader_names"):
        run_all({"index_order_matches_loader": ["pos-1", "neg-1"]})


def test_every_pending_entry_names_a_real_artifact():
    """`nearest_artifact` has to resolve, or the entry is pointing at a memory.

    A pending entry outlives the code it describes -- that is what makes it a
    pending entry -- so the one part of it that CAN be checked mechanically is
    checked: every dotted name resolves from the package. When a rename or a
    deletion makes an entry stale, this fails instead of the entry quietly
    becoming fiction that nobody can act on.
    """
    import importlib

    from motifmultiverse.guards import GUARDS_AWAITING_INPUT

    unresolved = []
    for name, entry in GUARDS_AWAITING_INPUT.items():
        target: Any = importlib.import_module("motifmultiverse")
        for part in entry.nearest_artifact.split("."):
            try:
                target = getattr(target, part)
            except AttributeError:
                try:
                    target = importlib.import_module(f"{target.__name__}.{part}")
                except (ImportError, AttributeError):
                    unresolved.append(f"{name} -> {entry.nearest_artifact}")
                    break
    assert not unresolved, (
        "pending entries naming an artifact that does not exist: " + ", ".join(unresolved)
    )


# --- the executable half of each pending entry --------------------------------
# Each test below asserts the fact its registry entry rests on, and FAILS when the
# awaited input arrives. That direction is deliberate: a pending entry that only a
# human re-reads stays pending forever, and the one thing worse than an unwired
# guard is an unwired guard whose reason stopped being true without anyone noticing.

def test_four_state_missingness_still_has_no_independently_claimed_coverage():
    """The manifest states a total and nothing else; health_report states all three.

    So the only artifact that could put a CLAIM in front of a recomputation carries
    one third of the claim, and the artifact that carries all three computes them in
    the same expression it would be checked against.
    """
    from motifmultiverse.guards import GUARDS_AWAITING_INPUT
    from motifmultiverse.schema.substrate import HitSubstrateManifest

    entry = GUARDS_AWAITING_INPUT["four_state_missingness"]
    assert entry.nearest_artifact == "schema.substrate.HitSubstrateManifest"
    fields = set(HitSubstrateManifest.__dataclass_fields__)
    assert "n_regions" in fields, "the one independently claimed count is gone"
    claimed = {f for f in fields
               if any(word in f for word in ("defined", "coverage", "searched", "used"))}
    assert not claimed, (
        f"the substrate manifest now claims {sorted(claimed)} independently of the hit "
        "rows: four_state_missingness has the input it was waiting for. Wire it into "
        "interpret.read_hit_table / verify_against_manifest and delete its "
        "GUARDS_AWAITING_INPUT entry."
    )


def test_estimability_floor_still_has_no_interval_and_no_reference_to_read_it_against():
    """`affected_interval` has a schema slot, a validator, and no writer.

    `evaluate_stability` is its only producer. Until it fills the slot, the guard's
    CI clause has nothing to read; the N clause is meanwhile an invariant of
    `StabilityResult` itself, so a guard there could only fail on an object the
    constructor already refuses to build.
    """
    import pandas as pd

    from motifmultiverse.guards import GUARDS_AWAITING_INPUT
    from motifmultiverse.validate import (
        MIN_AFFECTED_PEAKS,
        StabilityResult,
        ValidationError,
        evaluate_stability,
    )

    entry = GUARDS_AWAITING_INPUT["estimability_floor"]
    assert entry.nearest_artifact == "validate.StabilityResult"

    def table(*, affected: int, total: int, merged: bool) -> pd.DataFrame:
        return pd.DataFrame([
            {"peak_id": f"peak-{i:03d}",
             "hit_id": "family-merged" if (merged and i < affected) else "family-original",
             "coefficient": 2.0 if (merged and i < affected) else 1.0,
             "reconstruction": 1.0 if (merged and i < affected) else 0.0}
            for i in range(total)
        ])

    result = evaluate_stability(
        "decision:estimability",
        table(affected=40, total=200, merged=False),
        table(affected=40, total=200, merged=True),
    )
    assert result.n_affected_peaks == 40 > MIN_AFFECTED_PEAKS
    assert result.paired_delta_reconstruction_all is not None    # the reference exists
    assert result.affected_interval is None, (
        "a stability result now carries the interval estimability_floor's CI clause "
        "needs, and the reference it is read against is already there. Wire the guard "
        "at validate.evaluate_stability and delete its GUARDS_AWAITING_INPUT entry."
    )

    # The N clause, meanwhile, is the record's own invariant: below the floor the
    # constructor refuses before any guard could look.
    with pytest.raises(ValidationError):
        StabilityResult(
            decision_id="d", n_affected_peaks=MIN_AFFECTED_PEAKS - 1, n_affected_hits=1,
            family_coefficient_share=0.5, paired_delta_reconstruction_affected=1.0,
            paired_delta_reconstruction_all=0.0, hit_jaccard=None,
            coefficient_conservation=None, status="CHANGED_AFFECTED_SUBSET",
            power_statement="descriptive",
        )


def test_estimability_floor_would_be_vacuous_on_the_effects_table():
    """The other candidate call site, and why the number there cannot be under the floor.

    `health_report` floors the query's block count against the run's pre-registered
    floor before any effect is computed, and the effect frame's blocks are the union
    of the two sides' -- so an emitted effect's N is at least the declared floor, by
    construction rather than by luck. A guard wired at that floor cannot fail; one
    wired at a hard-coded 30 would override a floor the run itself declared.
    """
    from motifmultiverse import interpret
    from motifmultiverse.guards import GUARDS_AWAITING_INPUT
    from motifmultiverse.schema import HealthFloors, HitRecord, Missingness, PeakSetQuery

    assert "interpret.FamilyEffect" in GUARDS_AWAITING_INPUT["estimability_floor"].why_not_a_call_site

    assert not {"reference", "status", "estimability"} & set(
        interpret.FamilyEffect.__dataclass_fields__), (
        "FamilyEffect now carries a reference or an estimability state; re-read "
        "GUARDS_AWAITING_INPUT['estimability_floor'] -- its second clause may now have "
        "something to read."
    )

    hits = []
    for block in range(8):
        for side in (0, 1):
            start = block * 1_000_000 + side * 1_000
            hits.append(HitRecord(
                region_id=f"r{block}_{side}", chrom="chr1", start=start, end=start + 500,
                missingness=Missingness.USED, input_scale=16, lexicon_id="lex",
                substrate_id="e" * 64, variant_id=f"UA_FAM_{side}", family_id="FAM",
                hit_coefficient=1.0 if side == 0 else 0.2,
            ))
    floors = HealthFloors(min_intersection_coverage=0.9, min_blocks=8,
                          min_explained_fraction=0.9)
    result = interpret.interpret_query(
        hits,
        PeakSetQuery(query_id="q", region_ids=[h.region_id for h in hits if h.region_id.endswith("_0")],
                     comparator_region_ids=[h.region_id for h in hits if h.region_id.endswith("_1")],
                     comparator_id="odd", selection_provenance="EXTERNAL"),
        floors=floors, n_bootstrap=50, seed=1,
    )
    assert result.effects, "the fixture must actually reach the effects stage"
    for effect in result.effects:
        assert effect["n_blocks"] >= floors.min_blocks


def test_a_single_family_composition_share_is_measured_not_forced_by_being_alone():
    """The question the entry has to answer: undefined, or defined and out of scope.

    Those are different answers and only one of them puts this guard on
    `FamilyComposition`. It is the second, and the difference is measurable rather
    than a matter of reading. The share this guard rules on has the stratum's
    other families in its denominator, so one family forces it to 1.0
    IDENTICALLY -- the number cannot come out otherwise, which is why it carries
    no information. `FamilyComposition.peak_share` counts searched PEAKS, and
    being the only family constrains it to nothing at all: the same single-family
    query below reports 0.5 or 1.0 depending on what was measured.

    Verified on the real K562 substrate too (576,589 rows, 33,917 peaks):
    restricted to one family the composition comes out 0.310670 for
    CTCF/CTCFL-like and 0.998143 for AP-1/bZIP -- two single-family compositions,
    neither of them 1.0.

    So an estimability status on `FamilyComposition` would not be wiring this
    guard, it would be inventing an estimability semantics for an estimable
    quantity -- and a 1.0 marked NOT_ESTIMABLE would suppress the finding that
    every searched peak carries the family. This fails if that field is ever
    added, or if the entry stops recording which of the two answers it reached.
    """
    from motifmultiverse import interpret
    from motifmultiverse.guards import GUARDS_AWAITING_INPUT
    from motifmultiverse.schema import HitRecord, Missingness

    entry = GUARDS_AWAITING_INPUT["single_family_layer"]
    assert "DEFINED AND OUT OF SCOPE" in entry.why_not_a_call_site, (
        "the entry must say WHICH answer it reached about FamilyComposition.peak_share, "
        "not only that the guard is not wired there"
    )
    assert not {"status", "estimability", "estimable"} & set(
        interpret.FamilyComposition.__dataclass_fields__), (
        "FamilyComposition now carries an estimability state; re-read "
        "GUARDS_AWAITING_INPUT['single_family_layer'] -- the entry records that the "
        "share there is defined, so a status field on it is a decision someone made "
        "on other grounds and this guard is still not its reader"
    )

    def rows(n_carrying: int, n_total: int) -> list[HitRecord]:
        out = []
        for i in range(n_total):
            start = i * 1_000_000
            carries = i < n_carrying
            out.append(HitRecord(
                region_id=f"r{i:03d}", chrom="chr1", start=start, end=start + 500,
                missingness=Missingness.USED if carries else Missingness.NO_SEQUENCE_MATCH,
                input_scale=17, lexicon_id="lex", substrate_id="e" * 64,
                variant_id="UA_ONLY_00", family_id="ONLY",
                hit_coefficient=1.0 if carries else None))
        return out

    for n_carrying, expected in ((4, 0.5), (8, 1.0)):
        peaks = interpret.peak_universe(rows(n_carrying, 8), 1_000_000)
        composition = interpret.compose(peaks, sorted(peaks))
        assert [c.family_id for c in composition] == ["ONLY"], "a single-family composition"
        assert composition[0].peak_share == expected
        assert composition[0].n_peaks_searched == 8


def test_single_family_layer_names_a_share_that_really_does_come_out_one():
    """The failure the guard describes is live under BUDGET_FRACTION.

    A peak whose only family with mass is this one has a budget share of exactly
    1.0, so it clears any threshold in (0, 1] and BUDGET_FRACTION collapses into
    ANY_HIT there -- the same family with the same absolute mass in a peak that has
    other families is correctly not used. That is why the entry names this artifact
    and not `FamilyComposition.peak_share`, whose denominator is peaks.
    """
    from motifmultiverse import infer
    from motifmultiverse.guards import GUARDS_AWAITING_INPUT

    entry = GUARDS_AWAITING_INPUT["single_family_layer"]
    assert entry.nearest_artifact == "infer.UsageDefinition.BUDGET_FRACTION"
    assert "families" not in " ".join(infer.PeakUsage.__dataclass_fields__), (
        "PeakUsage now carries a family count, so 'this peak has one family' is "
        "observable where the share is computed; re-read "
        "GUARDS_AWAITING_INPUT['single_family_layer'] -- only the scientific decision "
        "it names is still outstanding."
    )

    sole = infer.PeakUsage(searched=True, hit_count=1, coefficient_sum=0.01,
                           abs_coefficient_sum=0.01, peak_abs_coefficient_sum=0.01)
    crowded = infer.PeakUsage(searched=True, hit_count=1, coefficient_sum=0.01,
                              abs_coefficient_sum=0.01, peak_abs_coefficient_sum=1.0)
    threshold = infer.UsageThreshold(value=0.90, null_source="test: a deliberately steep cut")
    effect = infer.two_part_summary(
        [sole], [crowded], family_id="FAM",
        usage_definition=infer.UsageDefinition.BUDGET_FRACTION,
        usage_threshold=threshold,
    )
    assert effect.n_used_query == 1 and effect.n_used_comparator == 0
    assert effect.probability_effect == 1.0, (
        "the single-family peak counts as fully used at a 0.90 budget threshold on a "
        "share of 1.0 that nothing else could have diluted"
    )


def test_the_two_guards_with_no_producing_stage_have_none():
    """`interaction_required`, `stratum_parity`.

    Each waits on a stage that does not exist, and "does not exist" is checkable:
    no record type carries the field the guard reads. When one appears, this fails
    and says which entry to re-read -- which beats discovering it by grepping, the
    way the seven unwired guards were found in the first place.

    `no_cross_model_cwm_avg` used to be the third, and the way it left is the
    reason this test is worth keeping. Its entry said an operations log would hold
    "only medoid-shaped entries" and so could not contain a violation. That was
    true of a log the combining stage writes about itself; it stopped being true
    when `compile.operations_log` began classifying the emitted lexicon against the
    registry arrays instead -- the same code, handed a file that averages, produces
    the violating entry. What the entry was really waiting for was a log with an
    author it is not.
    """
    from motifmultiverse import infer, interpret
    from motifmultiverse.guards import GUARDS_AWAITING_INPUT

    assert "no_cross_model_cwm_avg" not in GUARDS_AWAITING_INPUT

    records = (interpret.FamilyEffect, interpret.FamilyComposition, infer.TwoPartEffect)
    fields = {f for record in records for f in record.__dataclass_fields__}
    assert not {"is_specificity_claim", "interaction_ci"} & fields, (
        "an emitted record now claims specificity; wire interaction_required"
    )
    assert not {"stratum_rules", "stratum_id"} & fields, (
        "an emitted record now names a stratum; wire stratum_parity"
    )
