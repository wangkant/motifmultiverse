"""Tests for the BCa paired block bootstrap interval (`FP-15`, Task 15/Increment D).

Task 3 removed the percentile block bootstrap's p/q values because a percentile
bootstrap cannot license a hypothesis test. This module builds the interval that
legitimately can: the BCa (bias-corrected and accelerated) paired block bootstrap.
Task 16 later adds the wild cluster bootstrap-*t* p-value; only once both exist may
a result carry `InferenceCapability.INTERVAL_AND_TEST`.

The single most important correctness property under test: **the resampling AND
jackknife unit is the genomic block, never the peak.** Peaks within a block are
correlated (shared local coverage, shared regulatory context); resampling at the
peak level would silently understate the interval. `test_bca_block_level_resampling_is_much_wider_than_naive_peak_level`
makes this an executable, falsifiable claim rather than a comment.
"""
from __future__ import annotations

import math
import random

import pytest

from motifmultiverse.infer import MIN_ESTIMABLE_BLOCKS, InferError, bca_paired_block_interval


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
