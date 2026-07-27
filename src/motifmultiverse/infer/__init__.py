"""infer stage -- see README.md in this directory for rule / failure / check.

`bca_paired_block_interval` is `FP-15`'s specified interval estimator
(`docs/ROADMAP.md` M4a, `docs/CONSTRAINTS.md` FP-15): a bias-corrected and
accelerated (BCa) bootstrap over whole genomic blocks, paired across a query and
a comparator peak set. `wild_cluster_bootstrap_t` is `FP-15`'s specified p value:
a block-level wild cluster bootstrap-t over per-block scalar effects, with the
null imposed by centering. Together they are the two halves a result needs to
carry `schema.InferenceCapability.INTERVAL_AND_TEST`; `interpret` wires them in
behind its `bca-wild-cluster` estimator selection.

`two_part_summary` answers a different question about the same peaks: not how
uncertain one number is, but whether ONE number should have been reported at
all. A family can be used in more query peaks and used less intensely where it
is used; multiplied together those cancel, and a single mean difference reports
"no effect" for two effects that are individually large and opposite. The
summary reports the two margins separately and never collapses them.

The rest of this module -- `run` -- is still the pre-alpha skeleton described
below and in the README; it is unrelated to the estimators above.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist

import numpy as np

from motifmultiverse.schema import HealthFloors

__all__ = [
    "run", "InferError", "bca_paired_block_interval", "wild_cluster_bootstrap_t",
    "MIN_ESTIMABLE_BLOCKS",
    "UsageDefinition", "UsageThreshold", "PeakUsage", "TwoPartEffect",
    "two_part_summary",
]

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


#: Upper bound on Rademacher weights drawn per chunk inside
#: `wild_cluster_bootstrap_t` (a chunk is a `(n, G)` int8 matrix; the bound keeps
#: the float64 working set beside it modest for genome-scale block counts).
#: Chunk boundaries are unobservable in the result: the bit stream is consumed
#: sequentially, so chunking changes nothing but peak memory
#: (`test_wct_chunk_size_cannot_change_the_result` makes that an executable
#: claim by monkeypatching this constant).
_MAX_WEIGHTS_PER_CHUNK = 8_000_000


def wild_cluster_bootstrap_t(
    block_effects: Mapping[Block, float],
    *,
    null_value: float = 0.0,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, int]:
    """A block-level wild cluster bootstrap-*t* p-value (`FP-15`'s specified test).

    `block_effects` maps a genomic block (`chrom`, `start // block_size`, e.g.
    `schema.HitRecord.block`) to ONE scalar effect observed in that block, and
    tests H0: the across-block mean effect equals `null_value` (two-sided).
    Because the input is already block-level, the block is the resampling unit
    *by construction*: each replicate draws one Rademacher `-1`/`+1` weight per
    BLOCK, never per observation -- there are no observations below block level
    for this function to mistakenly treat as independent.

    The estimator, pinned down so a reader does not have to reverse-engineer it:

    * **Observed statistic.** `t_obs = (mean(x) - null_value) / se(x)` with
      `se(x) = sqrt(s^2(x) / G)`, `s^2` the Bessel-corrected (`ddof=1`) sample
      variance across the `G` blocks. The associated degrees of freedom are
      `G - 1`; they enter ONLY through that `ddof=1` -- the Student-t CDF is
      never consulted, because the bootstrap replicates ARE the reference
      distribution (a Gaussian-data cross-check against `scipy.stats.t` lives in
      `test_wct_agrees_with_scipy_t_test_on_a_gaussian_null_within_a_wide_band`).
    * **Null-imposed.** The replicate world is centred under the null:
      `e_g = x_g - mean(x)`, so the bootstrap data-generating process has mean
      0 -- the null, in null-centred coordinates -- no matter how far the
      observed mean sits from `null_value`. Reflecting the data about
      `null_value` therefore negates every replicate statistic and leaves the
      two-sided p-value byte-identical
      (`test_wct_null_imposition_makes_p_reflection_invariant_about_the_null`).
    * **Studentized replicates.** Each replicate recomputes its own standard
      error: `t* = mean(w * e) / se(w * e)`, with `se` defined exactly as for
      the observed statistic. Dividing by a within-replicate SE (rather than
      comparing raw replicate means) is what makes this a bootstrap-*t*; on
      heterogeneous block effects the two procedures disagree materially
      (`test_wct_studentised_statistic_matches_independent_loop_reference`).
    * **Two-sided finite-sample p.** `p = (extreme + 1) / (B + 1)` with
      `B = n_bootstrap` and `extreme = #{|t*| >= |t_obs|}`; a tie counts as
      extreme, the conservative convention. A skipped (degenerate) replicate
      leaves the denominator at `B + 1` exactly as preregistered.
    * **Degenerate replicates.** A replicate whose weighted effects are all
      identical (zero variance -- tested structurally, since a computed variance
      of identical values is not guaranteed to be exactly 0.0 in floating point)
      has an undefined `t*`. Such replicates are SKIPPED: never counted as
      extreme, and excluded from the returned `n_valid_replicates`, so a caller
      can floor the estimable-replicate count (`interpret` refuses below
      `MIN_ESTIMABLE_BLOCKS`). Mid-stream degeneracy requires astronomically
      unlikely weight patterns at `G >= MIN_ESTIMABLE_BLOCKS`; the reachable
      case is constant INPUT, defined explicitly above: if the constant equals
      `null_value` the data sit exactly on the null and there is no evidence
      against it by construction -- return `(1.0, 0)`; otherwise the constant
      effect contradicts the null in every block and the smallest reportable
      value is returned, `(1 / (B + 1), 0)` (conservative relative to the
      sign-test-style exact value `2^(1-G)`).

    Determinism and row-order invariance: block keys are sorted before any
    summation and weights are drawn in that sorted order from
    `numpy.random.default_rng(seed)` (never global RNG), so the result depends
    only on the mapping's content -- IEEE-754 summation is non-associative and
    dict insertion order must not leak in (the Task 15 lesson; the permutation
    test is `test_wct_block_key_insertion_order_never_changes_the_result`).

    Returns `(p_value, n_valid_replicates)`. Raises `InferError` if fewer than
    `MIN_ESTIMABLE_BLOCKS` blocks are supplied, if any block effect or
    `null_value` is NaN/Inf (a non-finite input must become a refusal, never a
    silently contaminated p-value), or if `n_bootstrap < 1`.
    """
    if n_bootstrap < 1:
        raise InferError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not math.isfinite(null_value):
        raise InferError(f"null_value must be finite, got non-finite {null_value!r}")

    # Sort by block key BEFORE anything sums or assigns a weight: iteration
    # order of the Mapping must never be observable.
    items = sorted(block_effects.items())
    if len(items) < MIN_ESTIMABLE_BLOCKS:
        raise InferError(
            f"{len(items)} genomic block(s) is below the preregistered floor of "
            f"{MIN_ESTIMABLE_BLOCKS} (schema.HealthFloors.min_blocks); refusing to "
            "report a p value computed from too little data."
        )
    x = np.array([v for _, v in items], dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise InferError(
            "block_effects contains a non-finite value (NaN or Inf); refusing to "
            "report a p value computed from non-finite data."
        )

    g = len(items)

    if bool(np.all(x == x[0])):
        # Bitwise-constant input. This is checked STRUCTURALLY, before any SE is
        # computed: the variance of identical values is not guaranteed to be 0.0
        # in floating point (the computed mean can differ from every value by 1
        # ULP, leaving dust-level deviations), and comparing that computed mean
        # against `null_value` misfires the same way -- so the constant itself
        # is what is compared. Every centered effect is then 0 and every
        # replicate degenerate, so n_valid_replicates is honestly 0, and the
        # p-value is defined by which side of the null the constant sits on (see
        # the docstring).
        if float(x[0]) == null_value:
            return (1.0, 0)
        return (1.0 / (n_bootstrap + 1), 0)

    mean = float(x.mean())
    var = float(x.var(ddof=1))
    se_obs = math.sqrt(var / g)
    if se_obs == 0.0:
        # Variance underflowed to zero although the values are not bitwise
        # identical: at float64 resolution the data are constant. The replicate
        # world is degenerate at that resolution (n_valid honestly 0); a zero
        # numerator over a zero SE sits exactly on the null, a nonzero one is an
        # unboundedly large |t_obs| no replicate can match.
        if mean == null_value:
            return (1.0, 0)
        return (1.0 / (n_bootstrap + 1), 0)

    t_obs = (mean - null_value) / se_obs
    abs_t_obs = abs(t_obs)
    centered = x - mean  # the null-imposed replicate world

    rng = np.random.default_rng(seed)
    sqrt_g = math.sqrt(g)
    chunk = max(1, min(n_bootstrap, _MAX_WEIGHTS_PER_CHUNK // g))
    extreme = 0
    n_valid = 0
    remaining = n_bootstrap
    while remaining > 0:
        n = min(chunk, remaining)
        remaining -= n
        # int8 keeps the draw small; the float64 product below is the working set.
        weights = rng.integers(0, 2, size=(n, g), dtype=np.int8) * 2 - 1
        boot = weights.astype(np.float64) * centered
        # Degenerate == every weighted effect bitwise identical (the structural
        # zero-variance condition; see the docstring).
        valid = ~(boot == boot[:, :1]).all(axis=1)
        n_valid += int(valid.sum())
        if valid.any():
            bmean = boot[valid].mean(axis=1)
            bsd = boot[valid].std(axis=1, ddof=1)
            t_boot = bmean / (bsd / sqrt_g)
            extreme += int((np.abs(t_boot) >= abs_t_obs).sum())

    p_value = (extreme + 1) / (n_bootstrap + 1)
    return (p_value, n_valid)


# --------------------------------------------------------------------------- #
# Two-part usage summaries: occupancy and intensity, never one number
# --------------------------------------------------------------------------- #
class UsageDefinition(StrEnum):
    """What "this family is used in this peak" means. There is no default.

    The three are genuinely different questions, and a result computed under one
    does not answer either of the others:

    * ``ANY_HIT`` -- the family has at least one called hit in the peak. The
      caller's own floor is the only threshold; nothing further is imposed here.
    * ``CONTRIBUTION_FLOOR`` -- the family's absolute coefficient mass in the
      peak reaches a frozen, null-derived cut-off. A hit below it is a *measured
      non-use*, not a missing observation: it stays in the denominator.
    * ``BUDGET_FRACTION`` -- the family holds at least a given share of the
      peak's total absolute coefficient mass. Relative rather than absolute, so
      it does not track overall attribution magnitude.

    Not defaulted anywhere in this module, and not defaultable: "used" is a
    scientific definition, and silently choosing one for a caller who never
    chose it produces a number whose meaning only the code knows.
    """

    ANY_HIT = "ANY_HIT"
    CONTRIBUTION_FLOOR = "CONTRIBUTION_FLOOR"
    BUDGET_FRACTION = "BUDGET_FRACTION"


@dataclass(frozen=True)
class UsageThreshold:
    """A cut-off with the null that produced it attached.

    ``CONTRIBUTION_FLOOR`` and ``BUDGET_FRACTION`` both need a number that
    separates use from non-use, and this project's standing rule is that such a
    number is either derived from a stated null or is not supplied at all
    (``schema.criteria`` says the same thing as ``CRITERION_NOT_YET_DEFINED``:
    an undefined criterion defers, it does not guess). ``null_source`` is
    therefore required and must be non-empty -- it names the calibration the
    value came from, so a reader can check the threshold rather than trust it.
    """

    value: float
    null_source: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise InferError(f"usage threshold must be finite, got {self.value!r}")
        if not isinstance(self.null_source, str) or not self.null_source.strip():
            raise InferError(
                "a usage threshold needs a non-empty null_source naming the calibration "
                "it came from; a bare number is an invented threshold"
            )


@dataclass(frozen=True)
class PeakUsage:
    """One peak's contribution for ONE family, in the shape a usage rule needs.

    ``searched`` is the four-state missingness collapsed to the only distinction
    a denominator cares about: ``NOT_SEARCHED`` is ``False`` and leaves every
    denominator; ``NO_SEQUENCE_MATCH`` and ``HIT_BELOW_FLOOR`` are ``True`` with
    ``hit_count == 0`` -- they are measurements of non-use and stay in.

    ``coefficient_sum`` is signed and ``abs_coefficient_sum`` is not: opposite
    hits cancel the first and not the second, which is why both travel.
    ``peak_abs_coefficient_sum`` is the peak's total over *all* families, the
    denominator ``BUDGET_FRACTION`` needs.
    """

    searched: bool
    hit_count: int
    coefficient_sum: float
    abs_coefficient_sum: float
    peak_abs_coefficient_sum: float


@dataclass(frozen=True)
class TwoPartEffect:
    """Occupancy and intensity as two numbers, plus the total they multiply to.

    ``total_effect`` is reported so that the cancellation is *visible* -- it is
    exactly the single number a one-part summary would have produced, sitting
    beside the two components that explain it. It is never reported alone.

    ``conditional_intensity_effect`` is ``None``, never ``0.0``, when a side
    never uses the family: "the mean among peaks that use it" has no value when
    no peak uses it, and a zero there would be read as "used, at zero
    intensity".
    """

    family_id: str
    usage_definition: UsageDefinition
    probability_effect: float
    conditional_intensity_effect: float | None
    total_effect: float
    n_used_query: int
    n_used_comparator: int
    #: The denominators. A probability difference without them is uncheckable,
    #: and this project reports no ratio without the count under it.
    n_measured_query: int
    n_measured_comparator: int
    #: Provenance of the cut-off, when the definition took one. Carried on the
    #: result rather than left in the caller's config, so the record states what
    #: "used" meant when it was computed.
    usage_threshold: float | None
    usage_threshold_source: str | None


def _mean_of(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _usage_predicate(definition: UsageDefinition, threshold: UsageThreshold | None):
    """Resolve the definition to a per-peak predicate, refusing on a missing cut-off."""
    if definition is UsageDefinition.ANY_HIT:
        if threshold is not None:
            raise InferError(
                "ANY_HIT takes no usage threshold; supplying one would record a cut-off "
                "that did not affect the result"
            )
        return lambda u: u.hit_count > 0

    if threshold is None:
        raise InferError(
            f"{definition.value} needs a frozen null-derived usage threshold "
            "(infer.UsageThreshold); refusing to invent a cut-off. Use ANY_HIT if no "
            "calibrated threshold exists yet."
        )

    if definition is UsageDefinition.CONTRIBUTION_FLOOR:
        return lambda u: u.hit_count > 0 and u.abs_coefficient_sum >= threshold.value

    def _budget(u: PeakUsage) -> bool:
        if u.hit_count <= 0:
            return False
        if u.peak_abs_coefficient_sum <= 0.0:
            # A share of a zero budget is undefined, and undefined is not False
            # any more than it is 0.0 -- refusing is the only honest option.
            raise InferError(
                "BUDGET_FRACTION is undefined for a peak that has hits but zero total "
                "absolute coefficient mass; the share has no denominator"
            )
        return u.abs_coefficient_sum / u.peak_abs_coefficient_sum >= threshold.value

    return _budget


def two_part_summary(
    query: Sequence[PeakUsage],
    comparator: Sequence[PeakUsage],
    *,
    family_id: str,
    usage_definition: UsageDefinition,
    usage_threshold: UsageThreshold | None = None,
) -> TwoPartEffect:
    """Split one family's query-minus-comparator difference into its two margins.

    The decomposition, over *measured* peaks only:

    * ``probability_effect``  = P(used | query) - P(used | comparator)
    * ``conditional_intensity_effect`` = E[mass | used, query] - E[mass | used,
      comparator], or ``None`` if either side has no used peak
    * ``total_effect`` = E[mass | query] - E[mass | comparator], with a
      non-using peak contributing 0

    and on each side ``E[mass] == P(used) * E[mass | used]`` exactly, which is
    what makes the total the *product* of the two margins rather than a third
    independent quantity. A family used more often but less intensely therefore
    shows a positive probability effect, a negative intensity effect, and a
    total near zero -- the case a single mean difference reports as "nothing
    here".

    ``usage_definition`` is required and has no default (see
    :class:`UsageDefinition`). ``NOT_SEARCHED`` peaks (``searched=False``) are
    excluded from every denominator rather than counted as non-use.

    Raises `InferError` when a side has no measured peak, when a thresholded
    definition is given no frozen threshold, or when a budget share is
    undefined.
    """
    if not isinstance(usage_definition, UsageDefinition):
        raise InferError(
            f"usage_definition must be one of {[d.value for d in UsageDefinition]}, got "
            f"{usage_definition!r}. There is no default: 'used' is a scientific "
            "definition, not a formatting choice."
        )
    is_used = _usage_predicate(usage_definition, usage_threshold)

    q_measured = [u for u in query if u.searched]
    c_measured = [u for u in comparator if u.searched]
    if not q_measured or not c_measured:
        raise InferError(
            f"{family_id}: a two-part summary needs measured peaks on both sides, got "
            f"query={len(q_measured)}, comparator={len(c_measured)}. NOT_SEARCHED peaks "
            "are not measurements of non-use and do not count here."
        )

    q_used = [u for u in q_measured if is_used(u)]
    c_used = [u for u in c_measured if is_used(u)]

    p_query = len(q_used) / len(q_measured)
    p_comparator = len(c_used) / len(c_measured)

    intensity_query = _mean_of([u.coefficient_sum for u in q_used]) if q_used else None
    intensity_comparator = _mean_of([u.coefficient_sum for u in c_used]) if c_used else None
    conditional = (
        intensity_query - intensity_comparator
        if intensity_query is not None and intensity_comparator is not None
        else None
    )

    # The total is built from the same used/not-used split, so it is the product
    # of the two margins by construction rather than a separately-computed number
    # that merely ought to agree with them.
    total_query = p_query * intensity_query if intensity_query is not None else 0.0
    total_comparator = (
        p_comparator * intensity_comparator if intensity_comparator is not None else 0.0
    )

    return TwoPartEffect(
        family_id=family_id,
        usage_definition=usage_definition,
        probability_effect=p_query - p_comparator,
        conditional_intensity_effect=conditional,
        total_effect=total_query - total_comparator,
        n_used_query=len(q_used),
        n_used_comparator=len(c_used),
        n_measured_query=len(q_measured),
        n_measured_comparator=len(c_measured),
        usage_threshold=None if usage_threshold is None else usage_threshold.value,
        usage_threshold_source=None if usage_threshold is None else usage_threshold.null_source,
    )


def run(*args, **kwargs):
    """Not implemented in the pre-alpha skeleton."""
    raise NotImplementedError(
        "infer is a skeleton; see src/motifmultiverse/infer/README.md"
    )
