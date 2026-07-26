"""infer stage -- see README.md in this directory for rule / failure / check.

`bca_paired_block_interval` is `FP-15`'s specified interval estimator
(`docs/ROADMAP.md` M4a, `docs/CONSTRAINTS.md` FP-15): a bias-corrected and
accelerated (BCa) bootstrap over whole genomic blocks, paired across a query and
a comparator peak set. It replaces the percentile block bootstrap that
`interpret` currently ships (`interpret.estimate_effects`) wherever a caller is
ready to license the stronger estimator; wiring it into that pipeline, and the
block-level wild cluster bootstrap-*t* p-value that would then let a result
carry `schema.InferenceCapability.INTERVAL_AND_TEST`, are later tasks (`FP-15`'s
"done when" in `docs/ROADMAP.md` needs both estimators before
`schema.IMPLEMENTED_ESTIMATORS` changes; see that file's M4a section).

The rest of this module -- `run` -- is still the pre-alpha skeleton described
below and in the README; it is unrelated to `bca_paired_block_interval` and is
untouched by this change.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from statistics import NormalDist

from motifmultiverse.schema import HealthFloors

__all__ = ["run", "InferError", "bca_paired_block_interval", "MIN_ESTIMABLE_BLOCKS"]

#: The block is the estimability unit everywhere else in this project
#: (`schema.HealthFloors.min_blocks`, `interpret.health_report`): for a clustered
#: peak set the effective sample size is the number of blocks, not the number of
#: peaks. Reused here rather than re-declared so the floor cannot drift between
#: the health-report path and the interval-estimator path.
MIN_ESTIMABLE_BLOCKS = HealthFloors().min_blocks

_NORMAL = NormalDist()

Block = tuple[str, int]


class InferError(ValueError):
    """A BCa interval was requested over too little genomic-block data to estimate."""


def _flatten(
    values: Mapping[Block, Sequence[float]],
    blocks: Sequence[Block],
    *,
    skip: Block | None = None,
) -> list[float]:
    """Concatenate the per-block sequences for `blocks`, in block-key order.

    Iterating `blocks` (already sorted by the caller) rather than `values` is
    what makes the result independent of the Mapping's insertion order.
    """
    out: list[float] = []
    for b in blocks:
        if b == skip:
            continue
        out.extend(values.get(b, ()))
    return out


def _quantile_linear(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolated quantile (`numpy.percentile` / `scipy.stats.quantile`
    default `method="linear"`), so the BCa percentile lookup uses the same
    interpolation convention as the reference implementation it is checked
    against.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = p * (n - 1)
    lo_i = math.floor(idx)
    hi_i = math.ceil(idx)
    if lo_i == hi_i:
        return sorted_values[int(idx)]
    frac = idx - lo_i
    return sorted_values[lo_i] * (1 - frac) + sorted_values[hi_i] * frac


def bca_paired_block_interval(
    query_values: Mapping[Block, Sequence[float]],
    comparator_values: Mapping[Block, Sequence[float]],
    *,
    statistic: Callable[[Sequence[float], Sequence[float]], float],
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """A BCa (bias-corrected and accelerated) paired block bootstrap interval.

    `query_values` / `comparator_values` map a genomic block (`chrom`,
    `start // block_size`, e.g. `schema.HitRecord.block`) to the per-peak values
    observed in that block for each side. A block need not appear in both
    mappings (mirrors `interpret.estimate_effects`, whose resampling frame is the
    union of blocks touched by either side).

    **The resampling AND jackknife unit is the genomic block, never the peak.**
    Peaks within a block are correlated -- shared local coverage, shared
    regulatory context -- so a peak-level bootstrap would silently understate the
    interval (`test_bca_block_level_resampling_is_much_wider_than_naive_peak_level`
    makes this an executable claim). Every draw below, and every jackknife
    leave-one-out replicate, removes or resamples a *whole* block: every peak
    that block carries on both sides moves together.

    Determinism: block keys are sorted before any resampling, so the result
    depends only on block membership and content, never on `Mapping` iteration
    order or on the internal order of a block's `Sequence[float]` (for an
    order-invariant `statistic`, e.g. a mean). The same `seed` reproduces
    byte-identical endpoints.

    Raises `InferError` if fewer than `MIN_ESTIMABLE_BLOCKS` blocks are available,
    if fewer than that many survive the leave-one-block-out jackknife (a block
    whose removal empties one whole side cannot be jackknifed and is excluded
    rather than propagating a NaN into the acceleration estimate), or if fewer
    than that many bootstrap replicates are estimable. Each of these is a refusal
    to report a number computed from too little data, not a best-effort fallback.
    """
    if n_bootstrap < 1:
        raise InferError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not 0.0 < alpha < 1.0:
        raise InferError(f"alpha must be in (0, 1), got {alpha}")

    blocks = sorted(set(query_values) | set(comparator_values))
    if len(blocks) < MIN_ESTIMABLE_BLOCKS:
        raise InferError(
            f"{len(blocks)} genomic block(s) is below the preregistered floor of "
            f"{MIN_ESTIMABLE_BLOCKS} (schema.HealthFloors.min_blocks); refusing to "
            "report an interval computed from too little data."
        )

    # Canonicalise each block's values to a fixed (sorted) order before anything
    # ever sums them. A block's `Sequence[float]` represents an UNORDERED set of
    # peak values; the caller's original list order is not semantic and must not
    # be observable in the output. Sorting once here -- rather than trusting
    # `statistic` to be order-invariant, or reaching for an order-independent
    # summation like `math.fsum` -- is what actually delivers that: `statistic`
    # is an arbitrary caller-supplied callable (e.g. a plain `sum()`-based mean
    # difference) that this module does not control and cannot make
    # order-invariant from the outside. `math.fsum` would only fix
    # order-sensitivity in a summation *this module* performs; it does nothing
    # for a caller's own reduction. Handing `statistic` a value sequence that is
    # always identical (same content -> same order) for a given block, no matter
    # what order the caller enumerated it in, is the only place in the pipeline
    # that can make the guarantee hold for an arbitrary `statistic`. (A prior
    # version skipped this: `_flatten` preserved caller order, so summing the
    # same multiset in a different order produced a different float via
    # non-associative addition -- usually by ~1 ULP, but that was enough to
    # occasionally reorder two close bootstrap replicates relative to each
    # other, and the BCa percentile lookup then interpolated between different
    # order statistics, shifting the reported endpoint by far more than 1 ULP.
    # 161/180 (seed, n_bootstrap) trials disagreed under pure within-block
    # reordering before this fix; see
    # test_bca_within_block_row_order_invariance_holds_across_many_seeds_and_bootstrap_sizes.)
    # `key=(isnan, x)` rather than a plain `sorted(...)`: NaN's non-transitive
    # comparisons (`nan < x` and `x < nan` are both False) break `sorted`'s usual
    # guarantee that equal inputs sort to the same output regardless of input
    # order -- a NaN anywhere in a block's raw values could otherwise let
    # within-block order leak back in for the *other*, perfectly-finite values
    # in that same block. Grouping on `isnan` first routes every NaN to one
    # consistent end and leaves the finite values to compare normally among
    # themselves, independent of where the caller put the NaN.
    query_values = {
        b: tuple(sorted(query_values.get(b, ()), key=lambda x: (math.isnan(x), x)))
        for b in blocks
    }
    comparator_values = {
        b: tuple(sorted(comparator_values.get(b, ()), key=lambda x: (math.isnan(x), x)))
        for b in blocks
    }

    theta_hat = statistic(_flatten(query_values, blocks), _flatten(comparator_values, blocks))

    # Jackknife acceleration, block-level: each block is dropped -- from BOTH
    # sides at once, since it is the whole block that is the resampling unit --
    # in turn. A block that is the sole holder of an entire side cannot be
    # dropped without collapsing that side to nothing, so it is excluded from
    # the jackknife rather than handed to `statistic` as an empty sequence.
    jackknife_values: list[float] = []
    for b in blocks:
        q = _flatten(query_values, blocks, skip=b)
        c = _flatten(comparator_values, blocks, skip=b)
        if not q or not c:
            continue
        val = statistic(q, c)
        if math.isfinite(val):
            jackknife_values.append(val)

    if len(jackknife_values) < MIN_ESTIMABLE_BLOCKS:
        raise InferError(
            f"only {len(jackknife_values)} block(s) support the leave-one-block-out "
            f"jackknife, below the preregistered floor of {MIN_ESTIMABLE_BLOCKS}; "
            "refusing to report an acceleration estimate computed from too little data."
        )

    theta_dot = sum(jackknife_values) / len(jackknife_values)
    deviations = [theta_dot - v for v in jackknife_values]
    numerator = sum(d ** 3 for d in deviations)
    denominator = sum(d ** 2 for d in deviations) ** 1.5
    a_hat = (numerator / (6.0 * denominator)) if denominator > 0 else 0.0

    # Bootstrap replicates: draw whole blocks with replacement, never peaks.
    rng = random.Random(seed)
    n_blocks = len(blocks)
    replicates: list[float] = []
    for _ in range(n_bootstrap):
        draw = [blocks[rng.randrange(n_blocks)] for _ in range(n_blocks)]
        q = _flatten(query_values, draw)
        c = _flatten(comparator_values, draw)
        if not q or not c:
            continue
        val = statistic(q, c)
        if math.isfinite(val):
            replicates.append(val)

    if len(replicates) < MIN_ESTIMABLE_BLOCKS:
        raise InferError(
            f"only {len(replicates)} of {n_bootstrap} bootstrap replicates were "
            f"estimable, below the preregistered floor of {MIN_ESTIMABLE_BLOCKS}"
        )
    replicates.sort()

    # Bias correction z0: inverse-normal-CDF of the tie-aware proportion of
    # replicates at or below theta_hat. Matches the "mean" convention scipy's BCa
    # implementation uses for percentile-of-score (ties count as one half), which
    # is what this function is cross-checked against
    # (test_bca_skewed_distribution_matches_checked_scipy_reference).
    n_rep = len(replicates)
    n_less = sum(1 for v in replicates if v < theta_hat)
    n_leq = sum(1 for v in replicates if v <= theta_hat)
    proportion = (n_less + n_leq) / (2.0 * n_rep)
    # Clamped away from the {0, 1} boundary so a unanimous bootstrap tail (every
    # replicate on one side of theta_hat) yields a large-but-finite z0 rather than
    # an infinite one propagating into a NaN interval.
    eps = 1.0 / (2 * n_rep)
    proportion = min(max(proportion, eps), 1.0 - eps)
    z0 = _NORMAL.inv_cdf(proportion)

    z_alpha = _NORMAL.inv_cdf(alpha / 2.0)
    z_1alpha = -z_alpha

    def _bca_percentile(z: float) -> float:
        num = z0 + z
        denom = 1.0 - a_hat * num
        if denom == 0:
            denom = 1e-12
        return _NORMAL.cdf(z0 + num / denom)

    p_lo = min(max(_bca_percentile(z_alpha), 0.0), 1.0)
    p_hi = min(max(_bca_percentile(z_1alpha), 0.0), 1.0)
    if p_lo > p_hi:
        p_lo, p_hi = p_hi, p_lo

    lo = _quantile_linear(replicates, p_lo)
    hi = _quantile_linear(replicates, p_hi)
    return (lo, hi)


def run(*args, **kwargs):
    """Not implemented in the pre-alpha skeleton."""
    raise NotImplementedError(
        "infer is a skeleton; see src/motifmultiverse/infer/README.md"
    )
