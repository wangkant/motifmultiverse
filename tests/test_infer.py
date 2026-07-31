"""Tests for the BCa paired block bootstrap interval (`FP-15`, Task 15/Increment D)
and the block-level wild cluster bootstrap-*t* p-value (`FP-15`, Task 16).

Task 3 removed the percentile block bootstrap's p/q values because a percentile
bootstrap cannot license a hypothesis test. Task 15 built the interval that
legitimately can: the BCa (bias-corrected and accelerated) paired block bootstrap.
Task 16 adds the wild cluster bootstrap-*t* p-value; only a result produced by both
may carry `InferenceCapability.INTERVAL_AND_TEST`.

The single most important correctness property under test: **the resampling AND
jackknife unit is the genomic block, never the peak.** Peaks within a block are
correlated (shared local coverage, shared regulatory context); resampling at the
peak level would silently understate the interval. `test_bca_block_level_resampling_is_much_wider_than_naive_peak_level`
makes this an executable, falsifiable claim rather than a comment.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from motifmultiverse import infer
from motifmultiverse.infer import (
    MIN_ESTIMABLE_BLOCKS,
    InferError,
    bca_paired_block_interval,
    wild_cluster_bootstrap_t,
)


def _mean_diff(q, c):
    return sum(q) / len(q) - sum(c) / len(c)


def _blocks(n, per_block=1):
    """`n` blocks, each holding `per_block` peaks, keyed (chrom, block_index)."""
    return [("chr1", i) for i in range(n)]


# --------------------------------------------------------------------------- #
# Step 1: exact reproducibility (-k bca selects every test in this module)
# --------------------------------------------------------------------------- #
def test_bca_same_input_and_seed_gives_byte_identical_endpoints():
    n = 40
    query_values = {("chr1", i): [float(i) + 0.5] for i in range(n)}
    comparator_values = {("chr1", i): [float(i) * 0.8] for i in range(n)}

    lo1, hi1 = bca_paired_block_interval(
        query_values, comparator_values, statistic=_mean_diff, n_bootstrap=500, seed=7,
    )
    lo2, hi2 = bca_paired_block_interval(
        query_values, comparator_values, statistic=_mean_diff, n_bootstrap=500, seed=7,
    )
    # Byte-identical, not merely close: same seed must reproduce the same floats.
    assert lo1 == lo2
    assert hi1 == hi2


def test_bca_different_seed_can_change_the_result():
    """Sanity twin of the identity test: seed must actually drive the resampling."""
    n = 40
    query_values = {("chr1", i): [float(i) + 0.5] for i in range(n)}
    comparator_values = {("chr1", i): [float(i) * 0.8] for i in range(n)}

    ci_a = bca_paired_block_interval(
        query_values, comparator_values, statistic=_mean_diff, n_bootstrap=200, seed=1,
    )
    ci_b = bca_paired_block_interval(
        query_values, comparator_values, statistic=_mean_diff, n_bootstrap=200, seed=2,
    )
    assert ci_a != ci_b


def test_bca_different_top_level_dict_order_same_block_membership_gives_same_result():
    """Dict-insertion order of the block keys themselves must never leak in."""
    n = 40
    query_values_a = {("chr1", i): [float(i) + 0.5] for i in range(n)}
    comparator_values_a = {("chr1", i): [float(i) * 0.8] for i in range(n)}

    query_values_b = {("chr1", i): [float(i) + 0.5] for i in reversed(range(n))}
    comparator_values_b = {("chr1", i): [float(i) * 0.8] for i in reversed(range(n))}

    assert list(query_values_a) != list(query_values_b)  # actually different row order

    ci_a = bca_paired_block_interval(
        query_values_a, comparator_values_a, statistic=_mean_diff, n_bootstrap=500, seed=11,
    )
    ci_b = bca_paired_block_interval(
        query_values_b, comparator_values_b, statistic=_mean_diff, n_bootstrap=500, seed=11,
    )
    assert ci_a == ci_b


def test_bca_within_block_row_order_invariance_holds_across_many_seeds_and_bootstrap_sizes():
    """`ci_a == ci_b` at one (seed, n_bootstrap) is not evidence the property holds.

    A round-1 audit found that reordering peaks *within* a block (same block
    membership, same content, different original list order) changed the final
    interval in 61% of 180 (seed, n_bootstrap) trials against the first cut of
    this function, by up to ~2e-3 absolute -- and the single-seed version of this
    test that shipped in round 1 happened to land in the ~39% where the two
    constructions agreed by luck. `sum()` over floats is not associative, so a
    different accumulation order for the *same multiset* can produce a
    bit-different replicate value; because a bootstrap replicate's neighbours in
    the sorted array are not infinitesimally close, that bit-level difference can
    flip which two order statistics the BCa percentile interpolates between,
    producing a real (not epsilon) shift in the reported endpoint.

    This sweeps the same scale the audit used (60 seeds x 3 n_bootstrap sizes =
    180 trials), with 5 peaks per block and only `MIN_ESTIMABLE_BLOCKS` (30)
    blocks -- few enough relative to `n_bootstrap` that a block is drawn more
    than once within a replicate on effectively every trial, which is exactly
    the condition that exposed the defect. A fix that merely reorders the
    unlucky seed cannot pass this; it has to make within-block order genuinely
    not matter.
    """
    n_blocks = MIN_ESTIMABLE_BLOCKS
    peaks_per_block = 5

    base_query = {
        ("chr1", i): [float(i) + 0.1 * k for k in range(peaks_per_block)]
        for i in range(n_blocks)
    }
    base_comparator = {
        ("chr1", i): [float(i) * 0.5 + 0.2 * k for k in range(peaks_per_block)]
        for i in range(n_blocks)
    }
    # Same block membership, same per-block multiset of values, reversed order
    # within every block.
    reordered_query = {b: list(reversed(v)) for b, v in base_query.items()}
    reordered_comparator = {b: list(reversed(v)) for b, v in base_comparator.items()}

    mismatches = []
    for seed in range(60):
        for n_bootstrap in (50, 137, 500):
            ci_a = bca_paired_block_interval(
                base_query, base_comparator, statistic=_mean_diff,
                n_bootstrap=n_bootstrap, seed=seed,
            )
            ci_b = bca_paired_block_interval(
                reordered_query, reordered_comparator, statistic=_mean_diff,
                n_bootstrap=n_bootstrap, seed=seed,
            )
            if ci_a != ci_b:
                mismatches.append((seed, n_bootstrap, ci_a, ci_b))

    assert not mismatches, (
        f"{len(mismatches)}/180 (seed, n_bootstrap) trials disagreed under pure "
        f"within-block reordering; first few: {mismatches[:5]}"
    )


# --------------------------------------------------------------------------- #
# Step 2: skewed distribution, checked against an independent reference value
# --------------------------------------------------------------------------- #
def test_bca_skewed_distribution_matches_checked_scipy_reference():
    """The reference values below are NOT derived from this implementation.

    They come from an independent, one-sample reformulation of the same data run
    through scipy's own BCa machinery (`scipy==1.17.1`, `scipy.stats.bootstrap`).

    The data: 50 blocks, one query and one comparator peak each (a "paired block
    bootstrap" with one peak per side per block reduces exactly to resampling
    matched pairs). The query side is heavily right-skewed (a linear ramp plus 5
    large outliers); the comparator side is a plain linear ramp with no skew. Under
    that reduction, `mean(query) - mean(comparator)` computed over resampled block
    pairs is identical in distribution to `mean(delta)` where
    `delta = query - comparator`, so scipy's own one-sample BCa bootstrap over
    `delta` is a legitimate, independently-implemented check on this function's
    formula (bias correction z0 + jackknife acceleration a), not a restatement of
    it. The exact command used to produce the two reference numbers stored here:

        import numpy as np
        from scipy import stats
        query = np.array([0.05 * (i + 1) for i in range(50)])
        query[-5:] = [8.0, 10.0, 13.0, 18.0, 25.0]
        comparator = np.array([0.03 * (i + 1) for i in range(50)])
        delta = query - comparator
        stats.bootstrap(
            (delta,), np.mean, method="BCa", n_resamples=50000,
            random_state=np.random.default_rng(20260726), confidence_level=0.95,
        ).confidence_interval
        # -> ConfidenceInterval(low=0.8956274710149755, high=3.5815440258598628)
        stats.bootstrap(
            (delta,), np.mean, method="percentile", n_resamples=50000,
            random_state=np.random.default_rng(20260726), confidence_level=0.95,
        ).confidence_interval
        # -> ConfidenceInterval(low=0.7204000000000002, high=3.072810000000001)

    Because this implementation resamples whole blocks with its own RNG (Python's
    `random.Random`, not numpy's `Generator`), the two bootstrap distributions are
    not bit-identical, so the comparison uses a tolerance sized from the observed
    Monte Carlo spread of the reference itself (rerunning scipy's own BCa call with
    a different seed/resample count moved the low bound by <=0.004 and the high
    bound -- the skewed tail -- by <=0.05 at n_resamples=50000-200000). The
    tolerances below are several times that spread.
    """
    n = 50
    query_list = [0.05 * (i + 1) for i in range(n)]
    query_list[-5:] = [8.0, 10.0, 13.0, 18.0, 25.0]
    comparator_list = [0.03 * (i + 1) for i in range(n)]

    query_values = {("chr1", i): [query_list[i]] for i in range(n)}
    comparator_values = {("chr1", i): [comparator_list[i]] for i in range(n)}

    lo, hi = bca_paired_block_interval(
        query_values, comparator_values, statistic=_mean_diff, n_bootstrap=50_000, seed=42,
    )

    scipy_bca_lo, scipy_bca_hi = 0.8956274710149755, 3.5815440258598628
    scipy_percentile_lo, scipy_percentile_hi = 0.7204000000000002, 3.072810000000001

    assert lo == pytest.approx(scipy_bca_lo, abs=0.05)
    assert hi == pytest.approx(scipy_bca_hi, abs=0.15)

    # And the whole point of BCa: it must differ materially from the percentile
    # interval on this skewed fixture, in the direction scipy's own reference
    # differs (both bounds shifted up, correcting for the positive skew/bias).
    assert lo - scipy_percentile_lo > 0.1
    assert hi - scipy_percentile_hi > 0.3


# --------------------------------------------------------------------------- #
# Step 3: block-level jackknife/resampling, and the estimability floor
# --------------------------------------------------------------------------- #
def test_bca_rejects_fewer_than_the_preregistered_minimum_blocks():
    assert MIN_ESTIMABLE_BLOCKS == 30  # schema.HealthFloors().min_blocks, pinned
    n = MIN_ESTIMABLE_BLOCKS - 1
    query_values = {("chr1", i): [float(i)] for i in range(n)}
    comparator_values = {("chr1", i): [float(i) * 2] for i in range(n)}

    with pytest.raises(InferError):
        bca_paired_block_interval(
            query_values, comparator_values, statistic=_mean_diff, n_bootstrap=200, seed=0,
        )


def test_bca_accepts_exactly_the_preregistered_minimum_blocks():
    n = MIN_ESTIMABLE_BLOCKS
    query_values = {("chr1", i): [float(i)] for i in range(n)}
    comparator_values = {("chr1", i): [float(i) * 2] for i in range(n)}

    lo, hi = bca_paired_block_interval(
        query_values, comparator_values, statistic=_mean_diff, n_bootstrap=500, seed=0,
    )
    assert lo <= hi


def test_bca_block_level_resampling_is_much_wider_than_naive_peak_level():
    """The correctness property the brief calls out as most important.

    40 blocks x 15 peaks; peaks within a block are IDENTICAL (perfectly
    correlated), only block-to-block values vary. The true independent sample
    size is 40 (blocks), never 600 (peaks). A bootstrap that resampled individual
    peaks (ignoring block membership) would treat 600 highly-correlated
    observations as independent and report an interval far narrower than the data
    actually support. This function must not do that.
    """
    n_blocks = 40
    peaks_per_block = 15
    query_values = {}
    comparator_values = {}
    for i in range(n_blocks):
        b = ("chr1", i)
        query_values[b] = [1.0 + 0.3 * i] * peaks_per_block
        comparator_values[b] = [1.0] * peaks_per_block

    lo, hi = bca_paired_block_interval(
        query_values, comparator_values, statistic=_mean_diff, n_bootstrap=5_000, seed=3,
    )
    block_width = hi - lo

    # Foil: a peak-level bootstrap that flattens away block membership entirely.
    all_q = [v for i in range(n_blocks) for v in query_values[("chr1", i)]]
    all_c = [v for i in range(n_blocks) for v in comparator_values[("chr1", i)]]
    rng = random.Random(3)
    reps = []
    for _ in range(5_000):
        qd = [all_q[rng.randrange(len(all_q))] for _ in range(len(all_q))]
        cd = [all_c[rng.randrange(len(all_c))] for _ in range(len(all_c))]
        reps.append(_mean_diff(qd, cd))
    reps.sort()
    naive_lo = reps[int(0.025 * len(reps))]
    naive_hi = reps[int(0.975 * len(reps))]
    naive_width = naive_hi - naive_lo

    assert block_width > 2 * naive_width


def test_bca_jackknife_and_resampling_survive_asymmetric_block_membership():
    """Not every block need carry both a query and a comparator peak.

    Mirrors interpret's percentile bootstrap, whose resampling frame is the union
    of blocks touched by either side (`interpret.estimate_effects`); some blocks
    are query-only or comparator-only. This must still produce a finite interval
    when there are enough doubly-populated blocks to satisfy the estimability
    floor.
    """
    n = 40
    query_values = {("chr1", i): [float(i) + 1.0] for i in range(n)}
    comparator_values = {("chr1", i): [float(i) + 0.5] for i in range(n)}
    # A handful of query-only and comparator-only blocks layered on top.
    for i in range(n, n + 5):
        query_values[("chr1", i)] = [100.0]
    for i in range(n + 5, n + 10):
        comparator_values[("chr1", i)] = [-100.0]

    lo, hi = bca_paired_block_interval(
        query_values, comparator_values, statistic=_mean_diff, n_bootstrap=1_000, seed=5,
    )
    assert lo <= hi
    assert lo > float("-inf") and hi < float("inf")


def test_bca_refuses_when_the_statistic_is_never_finite():
    """NaN/Inf must never propagate into a reported interval, only a refusal.

    Every jackknife and bootstrap replicate a non-finite `statistic` would
    produce is excluded (`math.isfinite` guards on both loops); with zero
    survivors that is far below `MIN_ESTIMABLE_BLOCKS`, so the function refuses
    outright rather than reporting a NaN- or Inf-contaminated interval.
    """
    n = MIN_ESTIMABLE_BLOCKS
    query_values = {("chr1", i): [float(i)] for i in range(n)}
    comparator_values = {("chr1", i): [float(i) * 2] for i in range(n)}

    with pytest.raises(InferError):
        bca_paired_block_interval(
            query_values, comparator_values, statistic=lambda q, c: float("nan"),
            n_bootstrap=200, seed=0,
        )
    with pytest.raises(InferError):
        bca_paired_block_interval(
            query_values, comparator_values, statistic=lambda q, c: float("inf"),
            n_bootstrap=200, seed=0,
        )


def test_bca_zero_variance_returns_a_finite_degenerate_interval_never_nan():
    """Constant data (every block identical) means constant jackknife replicates:
    the acceleration denominator is exactly 0. That must fall back to a_hat=0
    (see the `denominator > 0 else 0.0` guard), not divide by zero into a NaN
    that then contaminates the whole interval.
    """
    n = MIN_ESTIMABLE_BLOCKS
    query_values = {("chr1", i): [3.0] for i in range(n)}
    comparator_values = {("chr1", i): [1.0] for i in range(n)}

    lo, hi = bca_paired_block_interval(
        query_values, comparator_values, statistic=_mean_diff, n_bootstrap=500, seed=0,
    )
    assert math.isfinite(lo)
    assert math.isfinite(hi)
    assert lo == hi == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Task 16: block-level wild cluster bootstrap-t (`FP-15`'s specified p value)  #
# --------------------------------------------------------------------------- #
def _effects_dict(values, offset=0):
    """A block_effects mapping keyed (chrom, block_index) from a flat value list."""
    return {("chr1", offset + i): float(v) for i, v in enumerate(values)}


def _loop_reference_wct(values, null_value, n_bootstrap, seed):
    """An INDEPENDENT re-implementation of the specified estimator, written as
    plain-Python per-replicate loops (no numpy vectorisation, no chunking) so that
    a defect in the shipped vectorised path cannot be mirrored here by
    construction. It shares only the RNG protocol (numpy default_rng consuming
    `integers(0, 2, size=G, dtype=int8)` per replicate, which the implementation's
    chunked matrix draws reproduce exactly by sequential bit-stream consumption --
    dtype matters: int8 and the int64 default consume the stream differently) and
    the preregistered formula `p = (extreme + 1) / (B + 1)`.
    """
    g = len(values)
    mean = sum(values) / g
    var = sum((v - mean) ** 2 for v in values) / (g - 1)
    se = math.sqrt(var / g)
    t_obs = (mean - null_value) / se
    centered = [v - mean for v in values]
    rng = np.random.default_rng(seed)
    extreme = 0
    n_valid = 0
    for _ in range(n_bootstrap):
        w = rng.integers(0, 2, size=g, dtype=np.int8) * 2 - 1
        y = [int(wi) * ei for wi, ei in zip(w, centered, strict=True)]
        if all(v == y[0] for v in y):
            continue  # degenerate: structurally zero-variance replicate
        n_valid += 1
        m = sum(y) / g
        s2 = sum((v - m) ** 2 for v in y) / (g - 1)
        t = m / math.sqrt(s2 / g)
        if abs(t) >= abs(t_obs):
            extreme += 1
    return (extreme + 1) / (n_bootstrap + 1), n_valid


def test_wct_same_input_and_seed_gives_byte_identical_p_value():
    values = [0.13 * (i % 7) - 0.02 * i for i in range(40)]
    effects = _effects_dict(values)
    p1, n1 = wild_cluster_bootstrap_t(effects, n_bootstrap=999, seed=11)
    p2, n2 = wild_cluster_bootstrap_t(effects, n_bootstrap=999, seed=11)
    assert p1 == p2  # byte-identical, not merely close
    assert n1 == n2


def test_wct_seed_actually_drives_the_resampling():
    """Sanity twin of the identity test, made non-vacuous: a single seed pair
    could coincide legitimately (the p-value's resolution is 1/(B+1)), but ten
    seeds all returning the same p-value means the seed is not driving anything.
    """
    values = [0.13 * (i % 7) - 0.02 * i for i in range(40)]
    effects = _effects_dict(values)
    ps = {wild_cluster_bootstrap_t(effects, n_bootstrap=999, seed=s)[0] for s in range(10)}
    assert len(ps) >= 2


def test_wct_block_key_insertion_order_never_changes_the_result():
    """ROW-ORDER INVARIANCE (the Task 15 lesson): dict insertion order must not be
    observable. IEEE-754 summation is non-associative, so the implementation must
    sort by block key before any summation or weight assignment.
    """
    n = 40
    values = [0.13 * (i % 7) - 0.02 * i for i in range(n)]
    forward = {("chr1", i): values[i] for i in range(n)}
    reversed_order = {("chr1", i): values[i] for i in reversed(range(n))}
    shuffled = dict(random.Random(5).sample(list(forward.items()), n))
    assert list(forward) != list(reversed_order) != list(shuffled)  # really permuted

    p_fwd = wild_cluster_bootstrap_t(forward, n_bootstrap=999, seed=3)
    assert wild_cluster_bootstrap_t(reversed_order, n_bootstrap=999, seed=3) == p_fwd
    assert wild_cluster_bootstrap_t(shuffled, n_bootstrap=999, seed=3) == p_fwd


def test_wct_null_p_values_do_not_collapse_to_the_resolution_floor():
    """Step 1 calibration: across five FIXED null datasets (mean exactly 0 by
    symmetric construction), the bootstrap-t p-value must not sit at the
    resolution floor 1/(B+1) systematically -- a collapse there is the signature
    of a test that rejects everything (or of a degenerate reference
    distribution), not of a calibrated one.
    """
    n_bootstrap, seed = 999, 777
    floor = 1.0 / (n_bootstrap + 1)
    ps = []
    for data_seed in (11, 22, 33, 44, 55):
        half = np.random.default_rng(data_seed).normal(0.0, 1.0, size=30)
        # symmetric pairs -> the dataset mean is exactly 0.0, a true null
        values = np.concatenate([half, -half])
        p, n_valid = wild_cluster_bootstrap_t(
            _effects_dict(values), n_bootstrap=n_bootstrap, seed=seed)
        assert n_valid == n_bootstrap  # non-degenerate data: nothing skipped
        ps.append(p)
    assert sum(p > floor for p in ps) >= 4, ps
    assert sum(p > 0.05 for p in ps) >= 4, ps


def test_wct_planted_effect_yields_the_resolution_floor():
    """A planted effect this large (t_obs ~= 38) is structurally incapable of
    producing a non-floor p-value under a correct implementation: no null-world
    replicate can reach it. Equally, under the null this p == floor event has
    probability ~1/(B+1), so the assertion discriminates the two worlds.
    """
    rng = np.random.default_rng(4)
    values = 3.0 + 0.5 * rng.normal(0.0, 1.0, size=40)
    n_bootstrap = 999
    p, n_valid = wild_cluster_bootstrap_t(
        _effects_dict(values), n_bootstrap=n_bootstrap, seed=9)
    assert p == 1.0 / (n_bootstrap + 1)
    assert n_valid == n_bootstrap


def test_wct_planted_null_yields_a_non_small_p():
    """The mirror fixture: a symmetric null dataset whose mean is exactly 0, so
    t_obs = 0 and every valid replicate counts as extreme -> p == 1.0 exactly.
    """
    half = np.random.default_rng(8).normal(0.0, 1.0, size=25)
    values = np.concatenate([half, -half])
    p, n_valid = wild_cluster_bootstrap_t(_effects_dict(values), n_bootstrap=999, seed=13)
    assert p == 1.0
    assert n_valid == 999


def test_wct_null_imposition_makes_p_reflection_invariant_about_the_null():
    """THE executable claim of null-imposition. With the null correctly imposed
    (bootstrap world centred at the null), reflecting the data about
    `null_value` negates every centred effect; Rademacher weights are
    sign-symmetric, so the |t*| reference set is bit-identical and the two-sided
    p-value is byte-identical. An implementation that forgets to centre the
    block effects (uses raw x in the replicates) breaks this reflection
    invariance: `2*null - x` is not `-(x)` unless null == 0. The fixture is
    placed at t_obs ~= 1.5 so the p-value is mid-range and the equality cannot
    be satisfied trivially by both sides landing on the floor or on 1.0.
    """
    null_value = 0.7
    rng = np.random.default_rng(21)
    values = rng.normal(0.0, 1.0, size=40)
    values = values - values.mean()
    se = values.std(ddof=1) / math.sqrt(len(values))
    values = values + null_value + 1.5 * se  # t_obs ~= 1.5, mid-range p

    p_fwd, _ = wild_cluster_bootstrap_t(
        _effects_dict(values), null_value=null_value, n_bootstrap=1999, seed=17)
    p_ref, _ = wild_cluster_bootstrap_t(
        _effects_dict(2 * null_value - values), null_value=null_value,
        n_bootstrap=1999, seed=17)
    assert 0.01 < p_fwd < 0.9  # mid-range: the equality below is not vacuous
    assert p_fwd == p_ref


def test_wct_studentised_statistic_matches_independent_loop_reference():
    """The implementation divides by a standard error computed WITHIN each
    replicate. On this heterogeneous fixture (39 small blocks plus one dominant
    outlier block) replicate SEs vary enormously, so a non-studentised wild
    bootstrap (extreme count over the raw replicate means) lands far away --
    the in-test gap assertion proves the fixture actually separates the two
    estimators, and the reference-agreement assertion proves the implementation
    is the studentised one.
    """
    rng = np.random.default_rng(6)
    values = rng.normal(0.0, 0.02, size=40)
    values[17] = 8.0
    n_bootstrap, seed = 999, 23

    p_impl, n_valid_impl = wild_cluster_bootstrap_t(
        _effects_dict(values), n_bootstrap=n_bootstrap, seed=seed)
    p_ref, n_valid_ref = _loop_reference_wct(
        [float(v) for v in values], 0.0, n_bootstrap, seed)
    assert n_valid_impl == n_valid_ref
    # ULP-level slack only: the two implementations sum in different orders.
    assert abs(p_impl - p_ref) <= 2.0 / (n_bootstrap + 1)

    # Foil: the NON-studentised variant (same null-imposed replicates, but the
    # extreme count compares raw replicate means, never divided by a replicate
    # SE). On this fixture it must sit far from the studentised answer -- this
    # is what makes the agreement assertion above a test of studentisation
    # rather than of "some bootstrap".
    g = len(values)
    mean = sum(values) / g
    centered = [float(v) - mean for v in values]
    rng_foil = np.random.default_rng(seed)
    extreme_raw = 0
    for _ in range(n_bootstrap):
        w = rng_foil.integers(0, 2, size=g, dtype=np.int8) * 2 - 1
        y = [int(wi) * ei for wi, ei in zip(w, centered, strict=True)]
        if abs(sum(y) / g) >= abs(mean):
            extreme_raw += 1
    p_raw = (extreme_raw + 1) / (n_bootstrap + 1)
    assert abs(p_impl - p_raw) > 10.0 / (n_bootstrap + 1), (p_impl, p_raw)


def test_wct_agrees_with_scipy_t_test_on_a_gaussian_null_within_a_wide_band():
    """Independent-machinery calibration anchor: for iid Gaussian block effects
    at G=200, the bootstrap-t p-value must agree with the exact Student-t test
    (scipy's t CDF -- machinery this implementation does not share) well inside
    a wide band. The observed statistic is pinned at t_obs = 1.0 by shifting the
    data (which leaves the SE untouched), so the comparison sits mid-range where
    a disagreement is visible, not at the floor.
    """
    from scipy import stats

    g = 200
    rng = np.random.default_rng(31)
    values = rng.normal(0.0, 1.0, size=g)
    values = values - values.mean()
    se = values.std(ddof=1) / math.sqrt(g)
    values = values + 1.0 * se  # t_obs == 1.0 up to float dust

    p, n_valid = wild_cluster_bootstrap_t(
        _effects_dict(values), n_bootstrap=9999, seed=29)
    assert n_valid == 9999
    p_t = 2.0 * stats.t.sf(1.0, df=g - 1)
    assert abs(p - p_t) <= 0.05


def test_wct_chunk_size_cannot_change_the_result(monkeypatch):
    """The weight matrix is drawn in chunks for memory reasons; chunk boundaries
    must be invisible because the bit stream is consumed sequentially. A future
    'optimisation' that reseeds per chunk breaks this immediately.
    """
    values = list(np.random.default_rng(2).normal(0.0, 1.0, size=60))
    effects = _effects_dict(values)
    p_big = wild_cluster_bootstrap_t(effects, n_bootstrap=500, seed=41)
    monkeypatch.setattr(infer, "_MAX_WEIGHTS_PER_CHUNK", 7)
    p_small = wild_cluster_bootstrap_t(effects, n_bootstrap=500, seed=41)
    assert p_big == p_small


def test_wct_rejects_fewer_than_the_preregistered_minimum_blocks():
    n = MIN_ESTIMABLE_BLOCKS - 1
    with pytest.raises(InferError, match="below the preregistered floor"):
        wild_cluster_bootstrap_t(_effects_dict(range(n)), n_bootstrap=200, seed=0)


def test_wct_accepts_exactly_the_preregistered_minimum_blocks():
    values = [0.05 * (i % 4) for i in range(MIN_ESTIMABLE_BLOCKS)]
    p, n_valid = wild_cluster_bootstrap_t(_effects_dict(values), n_bootstrap=200, seed=0)
    assert 0.0 < p <= 1.0


def test_wct_refuses_non_finite_inputs():
    base = _effects_dict([0.1 * i for i in range(MIN_ESTIMABLE_BLOCKS)])
    for bad in (float("nan"), float("inf"), float("-inf")):
        corrupted = dict(base)
        corrupted[("chr1", 3)] = bad
        with pytest.raises(InferError, match="non-finite"):
            wild_cluster_bootstrap_t(corrupted, n_bootstrap=200, seed=0)
    for bad_null in (float("nan"), float("inf")):
        with pytest.raises(InferError, match="non-finite"):
            wild_cluster_bootstrap_t(base, null_value=bad_null, n_bootstrap=200, seed=0)


def test_wct_refuses_a_nonpositive_bootstrap_count():
    with pytest.raises(InferError, match="n_bootstrap"):
        wild_cluster_bootstrap_t(
            _effects_dict(range(MIN_ESTIMABLE_BLOCKS)), n_bootstrap=0, seed=0)


def test_wct_zero_variance_data_is_defined_explicitly_never_nan():
    """Constant block effects make every replicate degenerate. On the null (the
    constant equals null_value) the data carry no evidence against it BY
    CONSTRUCTION, so p == 1.0; off the null the constant contradicts it in every
    block and the smallest reportable value is the resolution floor. Both return
    n_valid_replicates == 0 so a caller can floor the estimable-replicate count.
    """
    on_null = _effects_dict([0.7] * MIN_ESTIMABLE_BLOCKS)
    p, n_valid = wild_cluster_bootstrap_t(
        on_null, null_value=0.7, n_bootstrap=500, seed=0)
    assert p == 1.0
    assert n_valid == 0

    off_null = _effects_dict([0.7] * MIN_ESTIMABLE_BLOCKS)
    p, n_valid = wild_cluster_bootstrap_t(
        off_null, null_value=0.0, n_bootstrap=500, seed=0)
    assert p == 1.0 / 501
    assert n_valid == 0


# --------------------------------------------------------------------------- #
# Task 17: two-part usage summaries -- occupancy and intensity, never one number
# --------------------------------------------------------------------------- #
def _usage(n, *, hits=1, coefficient=1.0, searched=True, peak_mass=None):
    """`n` peaks that look alike, for one family."""
    mass = abs(coefficient) if peak_mass is None else peak_mass
    return [infer.PeakUsage(searched=searched, hit_count=hits,
                            coefficient_sum=coefficient,
                            abs_coefficient_sum=abs(coefficient),
                            peak_abs_coefficient_sum=mass)
            for _ in range(n)]


def _unused(n, *, searched=True, peak_mass=0.0):
    """`n` measured peaks in which the family was not used (NO_SEQUENCE_MATCH /
    HIT_BELOW_FLOOR), or -- with `searched=False` -- NOT_SEARCHED peaks."""
    return [infer.PeakUsage(searched=searched, hit_count=0, coefficient_sum=0.0,
                            abs_coefficient_sum=0.0,
                            peak_abs_coefficient_sum=peak_mass)
            for _ in range(n)]


def test_two_part_cancellation_is_visible_in_the_components_not_the_total():
    """The whole point of the split, as an executable claim.

    Query uses FAM_A in 30 of 40 peaks at intensity 1.0 (total 0.75); the
    comparator uses it in 15 of 40 at intensity 2.0 (total 0.75 as well). A
    one-part mean difference reports exactly 0.0 -- "no difference" -- for a
    family that is used twice as often on one side and twice as strongly on the
    other. Both components must be large and opposite.
    """
    query = _usage(30, coefficient=1.0) + _unused(10)
    comparator = _usage(15, coefficient=2.0) + _unused(25)
    result = infer.two_part_summary(
        query, comparator, family_id="FAM_A",
        usage_definition=infer.UsageDefinition.ANY_HIT)

    assert result.total_effect == pytest.approx(0.0)          # what one number would say
    assert result.probability_effect == pytest.approx(0.75 - 0.375)
    assert result.conditional_intensity_effect == pytest.approx(1.0 - 2.0)
    assert result.n_used_query == 30
    assert result.n_used_comparator == 15
    assert result.n_measured_query == result.n_measured_comparator == 40


def test_two_part_total_is_the_product_of_the_two_margins_on_each_side():
    """`E[mass] == P(used) * E[mass | used]`, so the total cannot drift from the
    components it is supposed to decompose."""
    query = _usage(12, coefficient=1.5) + _unused(8)
    comparator = _usage(5, coefficient=0.4) + _unused(15)
    r = infer.two_part_summary(query, comparator, family_id="FAM_A",
                               usage_definition=infer.UsageDefinition.ANY_HIT)
    p_q, p_c = 12 / 20, 5 / 20
    assert r.probability_effect == pytest.approx(p_q - p_c)
    assert r.conditional_intensity_effect == pytest.approx(1.5 - 0.4)
    assert r.total_effect == pytest.approx(p_q * 1.5 - p_c * 0.4)
    # and it equals the plain mean over measured peaks, which is what a one-part
    # summary computes -- the two-part result contains it, it does not replace it.
    assert r.total_effect == pytest.approx((12 * 1.5) / 20 - (5 * 0.4) / 20)


def test_two_part_requires_a_usage_definition_and_has_no_default():
    """No default may be selected silently -- checked as a signature property, so
    a later 'convenience' default is a test failure rather than a quiet change."""
    import inspect
    sig = inspect.signature(infer.two_part_summary)
    assert sig.parameters["usage_definition"].default is inspect.Parameter.empty
    assert sig.parameters["usage_definition"].kind is inspect.Parameter.KEYWORD_ONLY

    query, comparator = _usage(5), _usage(5)
    for bad in (None, "ANY_HIT", "any_hit"):
        with pytest.raises(InferError, match="no default"):
            infer.two_part_summary(query, comparator, family_id="FAM_A",
                                   usage_definition=bad)


def test_two_part_contribution_floor_refuses_without_a_frozen_null_derived_threshold():
    query, comparator = _usage(5, coefficient=1.0), _usage(5, coefficient=1.0)
    with pytest.raises(InferError, match="refusing to invent a cut-off"):
        infer.two_part_summary(query, comparator, family_id="FAM_A",
                               usage_definition=infer.UsageDefinition.CONTRIBUTION_FLOOR)


def test_two_part_budget_fraction_refuses_without_a_frozen_null_derived_threshold():
    query, comparator = _usage(5, coefficient=1.0), _usage(5, coefficient=1.0)
    with pytest.raises(InferError, match="refusing to invent a cut-off"):
        infer.two_part_summary(query, comparator, family_id="FAM_A",
                               usage_definition=infer.UsageDefinition.BUDGET_FRACTION)


def test_two_part_a_threshold_without_a_named_null_is_refused():
    """A bare number is an invented threshold, whatever it is called."""
    with pytest.raises(InferError, match="non-empty null_source"):
        infer.UsageThreshold(value=0.3, null_source="")
    with pytest.raises(InferError, match="non-empty null_source"):
        infer.UsageThreshold(value=0.3, null_source="   ")
    with pytest.raises(InferError, match="must be finite"):
        infer.UsageThreshold(value=float("nan"), null_source="dinucleotide_shuffle_v1")


def test_two_part_any_hit_refuses_a_threshold_it_would_not_use():
    """A recorded cut-off that changed nothing is a lie in the provenance."""
    with pytest.raises(InferError, match="ANY_HIT takes no usage threshold"):
        infer.two_part_summary(
            _usage(5), _usage(5), family_id="FAM_A",
            usage_definition=infer.UsageDefinition.ANY_HIT,
            usage_threshold=infer.UsageThreshold(value=0.1, null_source="null_v1"))


def test_two_part_contribution_floor_makes_a_weak_hit_a_measured_non_use():
    """A sub-threshold hit leaves the numerator and STAYS in the denominator.

    Dropping it instead would raise the reported probability of use by shrinking
    the denominator -- the peak was searched and the family was measured there.
    """
    floor = infer.UsageThreshold(value=0.5, null_source="dinucleotide_shuffle_v1")
    query = _usage(10, coefficient=1.0) + _usage(10, coefficient=0.1)   # 10 weak hits
    comparator = _usage(10, coefficient=1.0) + _unused(10)
    r = infer.two_part_summary(query, comparator, family_id="FAM_A",
                               usage_definition=infer.UsageDefinition.CONTRIBUTION_FLOOR,
                               usage_threshold=floor)
    assert r.n_used_query == 10
    assert r.n_measured_query == 20            # the weak hits are still measured
    assert r.probability_effect == pytest.approx(0.0)
    assert r.usage_threshold == 0.5
    assert r.usage_threshold_source == "dinucleotide_shuffle_v1"
    # Under ANY_HIT the same data give a different, equally valid answer -- which
    # is exactly why the definition may not be chosen silently.
    any_hit = infer.two_part_summary(query, comparator, family_id="FAM_A",
                                     usage_definition=infer.UsageDefinition.ANY_HIT)
    assert any_hit.n_used_query == 20
    assert any_hit.probability_effect == pytest.approx(0.5)


def test_two_part_budget_fraction_reads_the_share_not_the_magnitude():
    strong_but_minor = _usage(10, coefficient=2.0, peak_mass=100.0)   # 2% of the peak
    weak_but_dominant = _usage(10, coefficient=0.2, peak_mass=0.25)   # 80% of the peak
    share = infer.UsageThreshold(value=0.5, null_source="budget_null_v1")
    r = infer.two_part_summary(strong_but_minor + weak_but_dominant, _usage(4, peak_mass=1.0),
                               family_id="FAM_A",
                               usage_definition=infer.UsageDefinition.BUDGET_FRACTION,
                               usage_threshold=share)
    assert r.n_used_query == 10                 # the dominant ones, not the strong ones
    assert r.n_measured_query == 20


def test_two_part_budget_fraction_refuses_an_undefined_share():
    """Zero budget with a hit in it: the share has no denominator, and undefined
    is not `False` any more than it is `0.0`."""
    broken = [infer.PeakUsage(searched=True, hit_count=1, coefficient_sum=0.0,
                              abs_coefficient_sum=0.0, peak_abs_coefficient_sum=0.0)]
    with pytest.raises(InferError, match="has no denominator"):
        infer.two_part_summary(
            broken, _usage(4, peak_mass=1.0), family_id="FAM_A",
            usage_definition=infer.UsageDefinition.BUDGET_FRACTION,
            usage_threshold=infer.UsageThreshold(value=0.5, null_source="budget_null_v1"))


def test_two_part_not_searched_peaks_leave_every_denominator():
    """NOT_SEARCHED is not evidence of non-use, so it cannot dilute a probability."""
    query = _usage(10, coefficient=1.0) + _unused(10)
    comparator = _usage(10, coefficient=1.0) + _unused(10)
    baseline = infer.two_part_summary(query, comparator, family_id="FAM_A",
                                      usage_definition=infer.UsageDefinition.ANY_HIT)
    padded = infer.two_part_summary(query + _unused(50, searched=False), comparator,
                                    family_id="FAM_A",
                                    usage_definition=infer.UsageDefinition.ANY_HIT)
    assert padded.n_measured_query == baseline.n_measured_query == 20
    assert padded.probability_effect == baseline.probability_effect
    assert padded.total_effect == baseline.total_effect


def test_two_part_measured_non_use_states_stay_in_the_denominator():
    """The mirror image of the test above, so the two states are not confused.

    NO_SEQUENCE_MATCH and HIT_BELOW_FLOOR ARE measurements: adding them lowers
    the probability of use. Adding NOT_SEARCHED peaks does not.
    """
    query = _usage(10, coefficient=1.0)
    comparator = _usage(10, coefficient=1.0)
    with_measured_non_use = infer.two_part_summary(
        query + _unused(10), comparator, family_id="FAM_A",
        usage_definition=infer.UsageDefinition.ANY_HIT)
    with_not_searched = infer.two_part_summary(
        query + _unused(10, searched=False), comparator, family_id="FAM_A",
        usage_definition=infer.UsageDefinition.ANY_HIT)
    assert with_measured_non_use.probability_effect == pytest.approx(0.5 - 1.0)
    assert with_not_searched.probability_effect == pytest.approx(0.0)
    assert with_measured_non_use.probability_effect != with_not_searched.probability_effect


def test_two_part_conditional_intensity_is_undefined_not_zero_when_a_side_never_uses():
    """`None`, never `0.0`: a mean over an empty set has no value, and a zero
    there reads as 'used, at zero intensity'."""
    query = _usage(10, coefficient=1.2) + _unused(10)
    comparator = _unused(20)
    r = infer.two_part_summary(query, comparator, family_id="FAM_A",
                               usage_definition=infer.UsageDefinition.ANY_HIT)
    assert r.n_used_comparator == 0
    assert r.conditional_intensity_effect is None
    assert r.probability_effect == pytest.approx(0.5)
    assert r.total_effect == pytest.approx(0.5 * 1.2)     # still defined


def test_two_part_opposite_signed_hits_cancel_intensity_but_not_occupancy():
    """Task 1's separation, carried into the summary: a family whose signed mass
    cancels to zero is still *present* in every peak it occupies."""
    query = [infer.PeakUsage(searched=True, hit_count=2, coefficient_sum=0.0,
                             abs_coefficient_sum=2.0, peak_abs_coefficient_sum=2.0)
             for _ in range(10)] + _unused(10)
    comparator = _unused(20)
    r = infer.two_part_summary(query, comparator, family_id="FAM_A",
                               usage_definition=infer.UsageDefinition.ANY_HIT)
    assert r.probability_effect == pytest.approx(0.5)      # occupancy survives
    assert r.conditional_intensity_effect is None          # comparator never uses it
    assert r.total_effect == pytest.approx(0.0)            # signed mass cancelled


def test_two_part_refuses_when_a_side_has_no_measured_peak():
    with pytest.raises(InferError, match="measured peaks on both sides"):
        infer.two_part_summary(_usage(10), _unused(10, searched=False),
                               family_id="FAM_A",
                               usage_definition=infer.UsageDefinition.ANY_HIT)


def test_wct_chunk_size_cannot_change_the_result_at_a_misaligned_block_count(monkeypatch):
    """The twin above passes at G = 60 for the wrong reason.

    numpy's bounded int8 generator hands out four values per 32-bit word and
    throws the partial buffer away when the call returns, so splitting an
    (n_bootstrap, G) draw into per-chunk draws only reproduces the single-call
    bit stream when G is a multiple of four. G = 60 is; MIN_ESTIMABLE_BLOCKS = 30,
    the smallest block count this estimator will accept at all, is not -- and at
    G = 30 shrinking the chunk moved p from 0.7305389221556886 to
    0.7005988023952096 on identical data and an identical seed. A p-value that
    follows a memory constant is not a reproducible p-value, so the fixture here
    is deliberately misaligned.
    """
    for g in (MIN_ESTIMABLE_BLOCKS, MIN_ESTIMABLE_BLOCKS + 3, 61):
        assert g % 4 != 0, "the point of this fixture is a misaligned block count"
        effects = _effects_dict(np.random.default_rng(2).normal(0.0, 1.0, size=g))
        monkeypatch.setattr(infer, "_MAX_WEIGHTS_PER_CHUNK", 8_000_000)
        one_chunk = wild_cluster_bootstrap_t(effects, n_bootstrap=500, seed=41)
        monkeypatch.setattr(infer, "_MAX_WEIGHTS_PER_CHUNK", 7)  # forces chunk == 1
        per_replicate = wild_cluster_bootstrap_t(effects, n_bootstrap=500, seed=41)
        assert one_chunk == per_replicate, f"chunk size moved the p-value at G={g}"
