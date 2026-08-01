"""align stage -- see README.md in this directory for rule / failure / check.

Two constraints carry the science, and both are enforced structurally rather
than left to caller discipline:

**Registration is selected on UNSIGNED sequence content (PPM).** Every offset
and orientation candidate is scored by cosine similarity of the (non-negative)
PPM rows. Signed CWM similarity is computed only AFTER an offset has been
chosen this way, at that one registration -- it never enters the search, so it
can never move the search toward or away from any particular offset. This is
the fix for the failure `README.md` documents: an aligner that maximises
*signed* cosine is structurally blind to a sign-flipped motif (the signed
cosine at the true registration is near -1, so that offset can never win under
that objective), which manufactured a false "no sign flips found" verdict in
the reference implementation. `guards.sign_alignment` is the executable check
that a result was produced this way (`registered_on == "unsigned_ppm"`).

**The bilateral overlap requirement excludes short windows from candidacy
entirely**, not just from being reported. A 3-4bp window can reach a very high
(even perfect) local cosine purely by chance or by an unrelated shared k-mer;
without a floor on both `overlap_bp` and the overlap fraction of *each* motif's
own length, the search would happily return that window as "the" alignment.
The floor is applied before scoring is even compared across candidates: an
invalid window is not scored down, it is not a candidate at all.

**Registration happens on the trimmed core, never on the padded pattern
window.** TF-MoDISco emits every pattern at one fixed window width and pads the
flanks with near-uniform background, so two patterns of the same width share
most of their rows no matter what motif each one actually contains. Cosine over
the whole window measures that shared padding: measured on 29 real ChromBPNet
discovery patterns (50bp windows, cores 4-30bp), every one of the 406 pairs
scored at least 0.665 and the median was 0.821 -- a 4bp `CATC` core and an 11bp
`GCCCCGCCCCC` core registered at 0.945. Trimmed first, the same 406 pairs span
0.032 to 0.989 with a median of 0.611, and 212 of them stop clearing the
overlap floor at all. Worse, the bilateral overlap floor below is computed from
the window length, so on a uniform-width registry it is satisfied by every pair
and excludes nothing -- the same short-window failure it exists to prevent,
hidden one level down. `ingest` already records each node's contribution-bearing
span as `trimmed_core`; `align_registry` slices both the PPM and the CWM to it
before any candidate is scored, and refuses a node that declares none rather
than falling back to the padded window.

**The null re-runs the full pipeline per shuffle.** `calibrate_pair_null` calls
`register_pair` -- offset and orientation search included -- once per shuffle,
on freshly permuted data. A null that reused the observed offset and only
recomputed a score at it would be answering "how similar are these two
sequences at one fixed alignment", not "how surprising is it to find SOME
alignment this good" -- a different, easier question that inflates every
p-value computed against it.

**Parallelism buys wall-clock and nothing else.** `workers` splits the pair loop
across processes; it is the one speed lever that does not weaken the null, and
it is admissible only because it cannot reach the arithmetic. Each pair's null
generator is constructed inside `calibrate_pair_null` from the run seed alone,
per call, so a pair's null scores are a pure function of that seed and that
pair's own two matrices -- no generator, cache or accumulator is carried from
one pair to the next, and a worker therefore cannot see how many other workers
ran or in what order pairs were scheduled. Outcomes are reassembled by each
pair's position in `combinations(nodes, 2)` rather than by completion time, so
the row order of `alignment_edges.parquet` is a property of the registry and not
of the scheduling. The equality is the point, not the speed: `tests/test_align.py`
runs a whole registry at two worker counts and compares the written files byte
for byte, and pins the per-pair null against an independent recomputation from
the seed, so a future shared-state "optimisation" fails there rather than
shipping a table whose p-values depend on a `--workers` value nobody recorded.
That is also why `workers` is NOT carried on the edges the way `seed` and
`null_shuffles` are: those two change the measurement, and a reader must be able
to see them; the worker count cannot, and recording it on every row would imply
it could.

RECORDED, not fixed: the null generator is seeded from the run seed only, so two
pairs whose targets have the same trimmed-core length draw the *same* sequence
of row permutations. Their nulls are positively dependent, which matters to any
later procedure that treats these p-values as independent tests. Seeding per
pair instead would decorrelate them -- and would change every p-value this rule
version has ever produced, which is a decision about the null and not about
scheduling. Parallelism neither causes this nor is blocked by it (both worker
counts reproduce the same correlated draws exactly), so it is written down here
rather than quietly changed under cover of a performance patch.

**Every pair this stage considered is accounted for on disk.** The run
denominators used to be returned, printed, and written to no file, on the
argument that a count smuggled into a guard-outcome sentence would become the
on-disk denominator in the one artifact nobody would look for it in. That
argument is right about *where* a count must not go and wrong about whether one
belongs on disk at all. Run this stage on thirteen real TF-MoDISco outputs and
`evidence/` holds 5,171 edges with no recorded denominator: a reader cannot tell
5,171-of-9,591 from 5,171-of-5,171, and an excluded pair is indistinguishable
from a pair that was never considered -- the absence-versus-refusal distinction
this package enforces everywhere else, lost in the one stage whose output is a
*subset* of what it looked at. So the guard sentence still carries no count, and
the denominators go where a reader of `evidence/` will find them:
`alignment_run_summary.json` beside the edge table carries
`n_pairs_considered = n_edges + n_pairs_excluded` with the overlap floor that
decided the split, and `alignment_excluded_pairs.tsv` names every excluded pair
with the reason it was excluded, the registrability status of each endpoint, and
the two trimmed-core lengths that reason is a statement about -- so
`no_offset_meets_overlap_floor` can be re-derived from the row and the recorded
floor rather than taken on trust. On that thirteen-analysis run all 139 nodes
were registrable and all 4,420 exclusions were the overlap floor over cores of
4-49bp, which is a fact about core-width heterogeneity in the registry that no
artifact previously recorded.

BASE_ORDER documents a convention this module needs but nothing upstream
declares: PPM/CWM columns are `(A, C, G, T)`. This is a numeric axis
convention (needed only to build a reverse complement), not a parsed
identifier, so it is not a Rule 2 (`no_key_parsing`) violation.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from numbers import Integral, Real
from pathlib import Path
from typing import Any

__all__ = [
    "AlignmentError", "AlignmentEvidence", "AlignmentRunSummary",
    "register_pair", "calibrate_pair_null", "align_registry", "run",
    "BASE_ORDER", "REGISTRATION_RULE_VERSION",
    "DEFAULT_MIN_OVERLAP_BP", "DEFAULT_MIN_OVERLAP_FRAC", "DEFAULT_NULL_SHUFFLES",
    "DEFAULT_WORKERS",
    "NODE_REGISTRABLE", "NODE_NO_PPM_ARRAY", "NODE_NO_TRIMMED_CORE", "NODE_STATUSES",
    "EXCLUDED_NODE_NOT_REGISTRABLE", "EXCLUDED_NO_OFFSET_MEETS_FLOOR",
    "PAIR_EXCLUSION_REASONS", "RUN_SUMMARY_SCHEMA_VERSION",
]

#: PPM/CWM column convention. See module docstring.
BASE_ORDER = "ACGT"

#: Bumped whenever the registration rule itself changes (the overlap floor,
#: the scoring function, the orientation search, the window registered on).
#: Carried on every emitted edge so a downstream reader never has to guess
#: which rule produced a row. `unsigned_ppm_v1` registered on the untrimmed
#: pattern window, and its scores are not comparable to these: the flanking
#: background two same-width patterns share dominated its cosine.
REGISTRATION_RULE_VERSION = "unsigned_ppm_trimmed_core_v2"

#: The bilateral overlap floor. Both conditions must hold: raw overlap length
#: AND the fraction of each motif's own length that overlap represents -- a
#: floor on `overlap_bp` alone would still let a long motif's tail "cover" a
#: short one almost entirely while barely touching itself.
DEFAULT_MIN_OVERLAP_BP = 6
DEFAULT_MIN_OVERLAP_FRAC = 0.5

DEFAULT_NULL_SHUFFLES = 1000

#: Worker processes for the pair loop. ONE by default, deliberately: adding a
#: parameter must not change what any existing invocation does, and a stage that
#: silently started using every core on a shared machine would be doing exactly
#: that. Raising it is safe for the *result* -- see the module docstring -- but
#: it is the caller's decision, not this module's.
DEFAULT_WORKERS = 1

#: Whether a registry node contributed matrices to the pair loop, and if not,
#: why not. The two refusals are named separately rather than merged into one
#: "unusable" because they are different repairs: no PPM at all is a discovery
#: or ingest problem, an undeclared `trimmed_core` is a trim-threshold one. A
#: reader holding only "excluded" could not tell which.
NODE_REGISTRABLE = "registrable"
NODE_NO_PPM_ARRAY = "no_ppm_array"
NODE_NO_TRIMMED_CORE = "no_declared_trimmed_core"
NODE_STATUSES = (NODE_REGISTRABLE, NODE_NO_PPM_ARRAY, NODE_NO_TRIMMED_CORE)

#: Why a considered pair produced no edge. Exactly one applies to each excluded
#: pair, and with `n_edges` they partition `n_pairs_considered` -- which is the
#: property `alignment_run_summary.json` exists to let a reader check.
EXCLUDED_NODE_NOT_REGISTRABLE = "node_not_registrable"
EXCLUDED_NO_OFFSET_MEETS_FLOOR = "no_offset_meets_overlap_floor"
PAIR_EXCLUSION_REASONS = (
    EXCLUDED_NODE_NOT_REGISTRABLE, EXCLUDED_NO_OFFSET_MEETS_FLOOR,
)

#: Bumped if the run-summary payload's keys change. Carried in the file so a
#: reader parsing it never has to infer which shape it is looking at.
RUN_SUMMARY_SCHEMA_VERSION = "1"


class AlignmentError(ValueError):
    """A pair could not be registered under the bilateral overlap rule."""


@dataclass(frozen=True)
class AlignmentEvidence:
    """One pair's registration, plus the null that calibrates it.

    Field order matches the brief's interface exactly; `registration_rule_version`
    is appended rather than inserted, so positional construction of the
    original fields is unaffected.
    """

    source_node_id: str
    target_node_id: str
    orientation: str
    offset: int
    overlap_bp: int
    overlap_frac_source: float
    overlap_frac_target: float
    ppm_similarity: float
    signed_cwm_similarity: float | None
    empirical_p_value: float | None
    null_shuffles: int
    seed: int
    registered_on: str = "unsigned_ppm"
    registration_rule_version: str = REGISTRATION_RULE_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("source_node_id", self.source_node_id),
            ("target_node_id", self.target_node_id),
            ("orientation", self.orientation),
            ("registered_on", self.registered_on),
            ("registration_rule_version", self.registration_rule_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise AlignmentError(f"{name} must be a non-empty string")
        if self.source_node_id == self.target_node_id:
            raise AlignmentError("alignment evidence must connect two distinct nodes")
        if self.orientation not in {"+", "-"}:
            raise AlignmentError("orientation must be '+' or '-'")
        if self.registered_on != "unsigned_ppm":
            raise AlignmentError("registered_on must be 'unsigned_ppm'")
        if self.registration_rule_version != REGISTRATION_RULE_VERSION:
            raise AlignmentError(
                f"registration_rule_version must be {REGISTRATION_RULE_VERSION!r}"
            )
        for name, value in (
            ("offset", self.offset),
            ("overlap_bp", self.overlap_bp),
            ("null_shuffles", self.null_shuffles),
            ("seed", self.seed),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise AlignmentError(f"{name} must be an integer")
        if self.overlap_bp <= 0:
            raise AlignmentError("overlap_bp must be positive")
        if self.null_shuffles < 0:
            raise AlignmentError("null_shuffles must be non-negative")

        def finite_measure(name: str, value: Any, low: float, high: float) -> None:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not low <= float(value) <= high
            ):
                raise AlignmentError(
                    f"{name} must be a finite measure in [{low}, {high}]"
                )

        finite_measure("overlap_frac_source", self.overlap_frac_source, 0.0, 1.0)
        finite_measure("overlap_frac_target", self.overlap_frac_target, 0.0, 1.0)
        finite_measure("ppm_similarity", self.ppm_similarity, -1.0, 1.0)
        if self.signed_cwm_similarity is not None:
            finite_measure(
                "signed_cwm_similarity", self.signed_cwm_similarity, -1.0, 1.0
            )
        if self.null_shuffles == 0:
            if self.empirical_p_value is not None:
                raise AlignmentError(
                    "uncalibrated alignment evidence requires empirical_p_value=None"
                )
        else:
            if self.empirical_p_value is None:
                raise AlignmentError(
                    "calibrated alignment evidence requires an empirical_p_value"
                )
            finite_measure("empirical_p_value", self.empirical_p_value, 0.0, 1.0)

    @property
    def is_calibrated(self) -> bool:
        return self.null_shuffles > 0 and self.empirical_p_value is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlignmentRunSummary:
    """What one `align_registry` call did, independent of any one edge.

    `workers` records the scheduling this run used. It sits here and NOT on
    `AlignmentEvidence` because of what the two places mean: a field on an edge
    is part of the measurement that edge reports, and `seed` / `null_shuffles`
    are exactly that -- change either and the p-value changes. The worker count
    cannot change any number in the table, so putting it on every row would tell
    a reader to consider it when comparing rows. Here it answers "what did this
    run do", which is a fair question about a job that can take hours.

    The denominators are no longer only here: `align_registry` writes them to
    `alignment_run_summary.json` beside the edge table, and every excluded pair
    to `alignment_excluded_pairs.tsv`, so a reader who never saw this object can
    still recover `n_pairs_considered = n_edges + n_pairs_excluded` and say why
    each excluded pair was excluded. `workers` is the one field deliberately
    left out of that file, for the same reason it is not on an edge: it changes
    no number, and a run-summary artifact that differed between two runs of the
    same registry would invite a reader to look for a difference that is not
    there. The CLI prints it, where it belongs -- it explains the wall-clock.
    """

    n_nodes: int
    n_pairs_considered: int
    n_edges: int
    n_pairs_excluded: int
    null_shuffles: int
    seed: int
    registration_rule_version: str
    edges_path: str
    null_summary_path: str
    workers: int = DEFAULT_WORKERS
    # Appended, never inserted -- the same convention
    # `AlignmentEvidence.registration_rule_version` follows, so positional
    # construction of the original fields is unaffected. `align_registry` sets
    # all four on every run; the defaults exist for that compatibility and are
    # not a value any run of this stage can actually emit.
    min_overlap_bp: int = DEFAULT_MIN_OVERLAP_BP
    min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC
    excluded_pairs_path: str = ""
    run_summary_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Numeric core
# --------------------------------------------------------------------------- #
def _as_matrix(mat: Any, what: str):
    import numpy as np

    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise AlignmentError(
            f"expected a (length, 4) {what} array over BASE_ORDER={BASE_ORDER!r}, "
            f"got shape {arr.shape}"
        )
    return arr


def _reverse_complement(mat: Any):
    """Reverse the position order and complement each base column.

    `BASE_ORDER = "ACGT"`: complementing swaps column 0<->3 (A<->T) and
    column 1<->2 (C<->G), the standard DNA complement, applied to a PPM or a
    CWM alike -- both are `(length, 4)` matrices over the same column
    convention, so the same transform registers either.
    """
    return mat[::-1, [3, 2, 1, 0]]


def _cosine(a: Any, b: Any) -> float:
    """Cosine of two motif windows, on the unsigned PPM content.

    This was briefly replaced by ``sqrt(v @ v)`` to avoid ``np.linalg.norm``'s
    dispatch cost, on a measurement of 3.2s -> 1.83s for a 29-pattern registry.
    That measurement was wall-clock over the whole CLI and was dominated by
    interpreter start-up and I/O: re-measured on the inner loop alone, the
    replacement was worth about 10%, and the ``np.errstate`` needed to make it
    safe cost more than it saved -- the "optimised" version benchmarked *slower*
    than this one (5.78 vs 5.26 us/call over 120,000 calls, median of 5). It is
    reverted rather than tuned further, because the complexity it carried (an
    underflow fallback, a suppressed warning context, a subnormal-range caveat)
    bought nothing measurable.

    What the detour did surface is a real defect, and it is not about speed.
    ``np.linalg.norm`` rescales internally; ``np.dot`` does not. Above about
    1e153 the numerator overflows to +/-inf while the denominator overflows too,
    the quotient is NaN -- and ``max(-1.0, min(1.0, nan))`` silently yields
    **+1.0**, because a NaN comparison is False and the clamp keeps its first
    argument. Measured: two exactly anti-correlated windows at 1e155 reported
    ``+1``. In *this* module that is the failure the docstring above is written
    about: a sign-flipped pair reported as a perfect positive match. Below about
    1e-165 the numerator underflows instead and the cosine comes back 0.0.

    So a non-finite quotient now falls back to a scale-normalised computation,
    which is exact at any representable magnitude. Cosine is scale-invariant, so
    this changes no value it did not previously get wrong, and it costs one
    ``math.isfinite`` on the path everything real takes.
    """
    import numpy as np

    flat_a, flat_b = a.ravel(), b.ravel()
    norm_a, norm_b = float(np.linalg.norm(flat_a)), float(np.linalg.norm(flat_b))
    if norm_a and norm_b:
        value = float(np.dot(flat_a, flat_b) / (norm_a * norm_b))
    elif not (flat_a.any() and flat_b.any()):
        # An all-zero window has no direction. A window whose norm merely
        # underflowed still does, so the two cannot share this exit -- and telling
        # them apart costs a scan, which is why it happens here and not on the
        # path every real pair takes.
        return 0.0
    else:
        value = math.nan
    if not math.isfinite(value):
        scale_a, scale_b = float(np.max(np.abs(flat_a))), float(np.max(np.abs(flat_b)))
        unit_a, unit_b = flat_a / scale_a, flat_b / scale_b
        denominator = float(np.linalg.norm(unit_a)) * float(np.linalg.norm(unit_b))
        value = float(np.dot(unit_a, unit_b)) / denominator if denominator else 0.0
    # Roundoff can place a mathematically bounded cosine a few ulps outside
    # [-1, 1]; clamp the computed value before schema validation. NaN can no
    # longer reach here, which matters: the clamp would turn it into +1.0.
    return max(-1.0, min(1.0, value))


def _candidate_windows(source_len: int, target_len: int):
    """Every (offset, overlap) with overlap_bp>=1, in ascending offset order.

    `offset` is defined so that target row `t` sits under source row `t + offset`:
    the overlapping source range is `[max(0,offset), min(source_len,
    target_len+offset))`, and the matching target range is that same window
    shifted back by `offset`.
    """
    for offset in range(-(target_len - 1), source_len):
        s_start = max(0, offset)
        s_end = min(source_len, target_len + offset)
        overlap_bp = s_end - s_start
        if overlap_bp <= 0:
            continue
        t_start = s_start - offset
        t_end = s_end - offset
        yield offset, s_start, s_end, t_start, t_end, overlap_bp


def _declared_core(node: dict[str, Any], length: int) -> tuple[int, int] | None:
    """The half-open trimmed core this registry node declares, or None.

    `ingest` records `trimmed_core` for every node it writes: the span of the
    pattern whose contribution actually rises above the trim threshold. Anything
    outside it is the near-uniform background TF-MoDISco pads each fixed-width
    window with, and registering on that padding is the failure the module
    docstring describes.

    None means "this registry declares no core", and the caller drops the pair.
    Falling back to the full window would be worse than refusing: it would put
    the padded score back into the edge table under the same rule version as a
    trimmed one, where nothing downstream could tell the two apart.
    """
    core = node.get("trimmed_core")
    if not isinstance(core, (list, tuple)) or len(core) != 2:
        return None
    try:
        start, end = int(core[0]), int(core[1])
    except (TypeError, ValueError):
        return None
    if not 0 <= start < end <= length:
        return None
    return start, end


def register_pair(source_ppm: Any, target_ppm: Any,
                  source_cwm: Any | None = None, target_cwm: Any | None = None,
                  *, min_overlap_bp: int = DEFAULT_MIN_OVERLAP_BP,
                  min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
                  source_node_id: str = "source", target_node_id: str = "target",
                  ) -> AlignmentEvidence:
    """Register one pair: search offset x orientation on PPM, score signed CWM
    only at the winner.

    `source_node_id`/`target_node_id` are metadata only, not part of the search;
    `align_registry` fills in the real ids afterwards. This keeps the function
    usable directly on raw arrays, exactly as the brief's interface specifies.
    """
    src = _as_matrix(source_ppm, "PPM")
    tgt = _as_matrix(target_ppm, "PPM")
    source_len, target_len = src.shape[0], tgt.shape[0]

    best: tuple[float, str, int, int, int, int, int, int, float, float] | None = None
    for orientation, tgt_oriented in (("+", tgt), ("-", _reverse_complement(tgt))):
        for offset, s_start, s_end, t_start, t_end, overlap_bp in _candidate_windows(
            source_len, target_len,
        ):
            frac_source = overlap_bp / source_len
            frac_target = overlap_bp / target_len
            # The bilateral overlap requirement: a candidate that fails it is
            # not scored down, it is not a candidate. A 3-4bp window reaching a
            # perfect local cosine must never reach the comparison below.
            if (overlap_bp < min_overlap_bp
                    or frac_source < min_overlap_frac
                    or frac_target < min_overlap_frac):
                continue
            score = _cosine(src[s_start:s_end], tgt_oriented[t_start:t_end])
            if best is None or score > best[0]:
                best = (score, orientation, offset, s_start, s_end, t_start, t_end,
                        overlap_bp, frac_source, frac_target)

    if best is None:
        raise AlignmentError(
            f"no offset/orientation satisfies the bilateral overlap requirement "
            f"(min_overlap_bp={min_overlap_bp}, min_overlap_frac={min_overlap_frac}) for "
            f"source_len={source_len}, target_len={target_len}; refusing to register this pair "
            "rather than return an offset that does not meet it"
        )

    (score, orientation, offset, s_start, s_end, t_start, t_end,
     overlap_bp, frac_source, frac_target) = best

    signed_similarity: float | None = None
    if source_cwm is not None and target_cwm is not None:
        src_cwm = _as_matrix(source_cwm, "CWM")
        tgt_cwm = _as_matrix(target_cwm, "CWM")
        if src_cwm.shape[0] != source_len:
            raise AlignmentError(
                f"source_cwm length {src_cwm.shape[0]} does not match source_ppm length {source_len}"
            )
        if tgt_cwm.shape[0] != target_len:
            raise AlignmentError(
                f"target_cwm length {tgt_cwm.shape[0]} does not match target_ppm length {target_len}"
            )
        tgt_cwm_oriented = tgt_cwm if orientation == "+" else _reverse_complement(tgt_cwm)
        # Measured AT the chosen registration only -- never re-optimised. This
        # is the one line that keeps a sign-flipped pair from being invisible:
        # nothing above this point has ever looked at a CWM.
        signed_similarity = _cosine(src_cwm[s_start:s_end], tgt_cwm_oriented[t_start:t_end])

    return AlignmentEvidence(
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        orientation=orientation,
        offset=offset,
        overlap_bp=overlap_bp,
        overlap_frac_source=frac_source,
        overlap_frac_target=frac_target,
        ppm_similarity=score,
        signed_cwm_similarity=signed_similarity,
        empirical_p_value=None,
        null_shuffles=0,
        seed=0,
    )


def calibrate_pair_null(source_ppm: Any, target_ppm: Any, *, null_shuffles: int, seed: int,
                        min_overlap_bp: int = DEFAULT_MIN_OVERLAP_BP,
                        min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
                        ) -> tuple[float, list[float]]:
    """Empirical p-value for the observed PPM registration, against a null that
    re-runs the FULL registration pipeline on each shuffle.

    Each null draw permutes the target's row (position) order and calls
    `register_pair` again from scratch: a fresh offset x orientation search,
    not a rescore at the observed offset. Reusing the observed offset here
    would answer a different, easier question (see module docstring) and would
    make every alignment look more significant than it is.
    """
    import numpy as np

    if null_shuffles < 1:
        # The "no null shuffle can ever fail to register" argument in the loop
        # below only holds for null_shuffles >= 1; at 0 it degenerates to an
        # empty null_scores and would otherwise surface deep inside the loop
        # as an opaque "every null shuffle failed" refusal. Fail fast here
        # instead, with the actual cause.
        raise AlignmentError(f"null_shuffles must be >= 1 to calibrate a null; got {null_shuffles}")

    src = _as_matrix(source_ppm, "PPM")
    tgt = _as_matrix(target_ppm, "PPM")

    observed = register_pair(src, tgt, min_overlap_bp=min_overlap_bp,
                             min_overlap_frac=min_overlap_frac)
    observed_score = observed.ppm_similarity

    rng = np.random.default_rng(seed)
    null_scores: list[float] = []
    for _ in range(null_shuffles):
        shuffled_target = tgt[rng.permutation(tgt.shape[0])]
        try:
            shuffle_evidence = register_pair(
                src, shuffled_target, min_overlap_bp=min_overlap_bp,
                min_overlap_frac=min_overlap_frac,
            )
        except AlignmentError:
            # The overlap floor depends only on the two lengths, which a row
            # permutation never changes, so this is unreachable whenever the
            # observed call above succeeded. Kept defensive rather than
            # asserted away: a shuffle that cannot be registered is not
            # evidence for or against the observed alignment either way.
            continue
        null_scores.append(shuffle_evidence.ppm_similarity)

    if not null_scores:
        raise AlignmentError(
            "every null shuffle failed the bilateral overlap requirement; no null "
            "distribution could be calibrated for this pair"
        )

    n_at_least_as_extreme = sum(1 for s in null_scores if s >= observed_score)
    p_value = (n_at_least_as_extreme + 1) / (len(null_scores) + 1)
    return p_value, null_scores


# --------------------------------------------------------------------------- #
# The whole stage
# --------------------------------------------------------------------------- #
_EDGE_FIELDS = [f.name for f in AlignmentEvidence.__dataclass_fields__.values()]
#: Explicit dtypes for every edge column, applied whether or not there are any
#: rows. `pd.DataFrame([], columns=_EDGE_FIELDS)` alone infers `object` for
#: every column on an empty run (0 edges, or a registry of 0/1 nodes), which
#: silently loses the typed schema (int64/float64) a populated run produces --
#: a real difference for any downstream reader that assumes a fixed schema
#: across runs (e.g. concatenating edge tables, or numeric ops on `offset` /
#: `ppm_similarity`). `signed_cwm_similarity` is `float64` even though the
#: dataclass allows `None`: pandas already stores that column's `None` as NaN.
_EDGE_DTYPES: dict[str, str] = {
    "source_node_id": "string", "target_node_id": "string",
    "orientation": "string", "offset": "int64", "overlap_bp": "int64",
    "overlap_frac_source": "float64", "overlap_frac_target": "float64",
    "ppm_similarity": "float64", "signed_cwm_similarity": "float64",
    "empirical_p_value": "float64", "null_shuffles": "int64", "seed": "int64",
    "registered_on": "string", "registration_rule_version": "string",
}
_NULL_SUMMARY_FIELDS = [
    "source_node_id", "target_node_id", "null_shuffles", "seed",
    "empirical_p_value", "null_mean", "null_min", "null_max",
    "observed_ppm_similarity", "registration_rule_version",
]
#: One row per excluded pair -- the complement of `alignment_edges.parquet` over
#: the same enumeration, so the two together are every pair this run considered.
#: `*_node_status` says why an endpoint contributed no matrices, and is
#: `registrable` on both sides exactly when the pair itself is what failed.
#: `*_core_bp` is the trimmed-core length each side brought: the two numbers the
#: overlap floor recorded in `alignment_run_summary.json` is a statement about,
#: so `no_offset_meets_overlap_floor` is a claim a reader can re-derive from the
#: row rather than one that has to be believed. Empty when the node declared no
#: core to measure -- which is itself that node's reason, not a missing value.
_EXCLUDED_PAIR_FIELDS = [
    "source_node_id", "target_node_id", "exclusion_reason",
    "source_node_status", "target_node_status",
    "source_core_bp", "target_core_bp",
]


def _write_edges(out: Path, edges: list[AlignmentEvidence]) -> Path:
    import pandas as pd

    dest = out / "alignment_edges.parquet"
    df = pd.DataFrame([e.to_dict() for e in edges], columns=_EDGE_FIELDS)
    df = df.astype(_EDGE_DTYPES)
    df.to_parquet(dest, index=False)
    return dest


def _write_null_summary(out: Path, rows: list[dict[str, Any]]) -> Path:
    dest = out / "alignment_null_summary.tsv"
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_NULL_SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return dest


def _excluded_pair_row(source_node_id: str, target_node_id: str, reason: str,
                       source_status: str, target_status: str,
                       core_bp: dict[str, int]) -> dict[str, Any]:
    """One excluded pair, said in full: which pair, which reason, and the facts
    the reason is about.

    The core lengths are read off the trimmed matrices, not recomputed from the
    floor -- a second copy of the overlap rule here could disagree with the one
    that actually excluded the pair, and a row that explained an exclusion that
    did not happen that way would be worse than no row.
    """
    return {
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "exclusion_reason": reason,
        "source_node_status": source_status,
        "target_node_status": target_status,
        "source_core_bp": core_bp.get(source_node_id, ""),
        "target_core_bp": core_bp.get(target_node_id, ""),
    }


def _write_excluded_pairs(out: Path, rows: list[dict[str, Any]]) -> Path:
    """Write every excluded pair -- header first, and written even with no rows.

    A zero-row file with its header says "this run excluded nothing". A missing
    file says "this run did not record exclusions". Those are different claims,
    and the whole point of the artifact is that a reader never has to guess which
    one an empty directory entry meant.
    """
    dest = out / "alignment_excluded_pairs.tsv"
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_EXCLUDED_PAIR_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return dest


def _write_run_summary(out: Path, payload: dict[str, Any]) -> Path:
    """The denominator, beside the numerator.

    Sorted keys and a trailing newline so the file is stable text: two runs of
    the same registry at different worker counts must produce identical bytes
    here as well as in the two tables, which is why nothing about scheduling is
    in the payload.
    """
    dest = out / "alignment_run_summary.json"
    dest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return dest


#: One pair's outcome: its position in the pair order, the edge it produced (or
#: None if it was excluded), and the null-summary row that goes with the edge.
_PairOutcome = tuple[int, "AlignmentEvidence | None", "dict[str, Any] | None"]


def _register_and_calibrate(
    index: int, source_node_id: str, target_node_id: str,
    source_ppm: Any, source_cwm: Any | None, target_ppm: Any, target_cwm: Any | None,
    *, null_shuffles: int, seed: int, min_overlap_bp: int, min_overlap_frac: float,
) -> _PairOutcome:
    """Everything one pair needs, from registration to its two output rows.

    This is the unit `workers` distributes, and it is the only place the per-pair
    work is written down: the sequential path calls it directly and each worker
    process calls it through `_worker_pair`, so there is no second copy that an
    edit could leave behind. It takes its matrices as arguments, returns plain
    data, and keeps nothing between calls -- which is precisely what makes the
    result independent of how the calls were scheduled. A cache or a shared
    generator added here would be invisible in a serial run and would change the
    answer in a parallel one.

    `index` is carried through untouched so the caller can restore pair order
    from it; a worker never needs to know what it means.
    """
    try:
        evidence = register_pair(
            source_ppm, target_ppm, source_cwm=source_cwm, target_cwm=target_cwm,
            min_overlap_bp=min_overlap_bp, min_overlap_frac=min_overlap_frac,
        )
    except AlignmentError:
        return index, None, None
    p_value, null_scores = calibrate_pair_null(
        source_ppm, target_ppm, null_shuffles=null_shuffles, seed=seed,
        min_overlap_bp=min_overlap_bp, min_overlap_frac=min_overlap_frac,
    )
    evidence = replace(
        evidence, source_node_id=source_node_id, target_node_id=target_node_id,
        empirical_p_value=p_value, null_shuffles=null_shuffles, seed=seed,
    )
    null_row = {
        "source_node_id": source_node_id, "target_node_id": target_node_id,
        "null_shuffles": null_shuffles, "seed": seed,
        "empirical_p_value": p_value,
        "null_mean": sum(null_scores) / len(null_scores),
        "null_min": min(null_scores), "null_max": max(null_scores),
        "observed_ppm_similarity": evidence.ppm_similarity,
        "registration_rule_version": evidence.registration_rule_version,
    }
    return index, evidence, null_row


#: Filled once per worker process by `_worker_init`, and never written to again.
#: The trimmed matrices are read-only inputs that every pair a worker handles
#: draws from, so they are sent once per process rather than once per job: a
#: 240-node registry has ~28,000 pairs and would otherwise pickle the same
#: 30x4 arrays thousands of times over. It is not shared state in the sense that
#: matters here -- nothing accumulates in it, so two workers holding it compute
#: exactly what one worker holding it would.
_WORKER_MATRICES: dict[str, tuple[Any, Any | None]] = {}


def _worker_init(matrices: dict[str, tuple[Any, Any | None]]) -> None:
    global _WORKER_MATRICES
    _WORKER_MATRICES = matrices


def _worker_pair(job: tuple[int, str, str, dict[str, Any]]) -> _PairOutcome:
    """Look this pair's matrices up in the worker, then do the ordinary work.

    Deliberately holds no logic of its own: everything below the lookup is the
    same call the sequential path makes, so "what a worker computes" and "what a
    serial run computes" cannot drift apart.
    """
    index, source_node_id, target_node_id, params = job
    source_ppm, source_cwm = _WORKER_MATRICES[source_node_id]
    target_ppm, target_cwm = _WORKER_MATRICES[target_node_id]
    return _register_and_calibrate(
        index, source_node_id, target_node_id,
        source_ppm, source_cwm, target_ppm, target_cwm, **params,
    )


def _run_pairs(jobs: list[tuple[int, str, str, dict[str, Any]]],
               matrices: dict[str, tuple[Any, Any | None]],
               *, workers: int,
               progress: Callable[[int, int], None] | None) -> list[_PairOutcome]:
    """Run every pair job and return the outcomes in JOB order, not finish order.

    Row order is part of what "byte-identical at every worker count" means: a
    table ordered by whichever worker finished first would differ run to run on
    the same machine with every number in it unchanged, and nothing in the file
    would say why the bytes moved. Two things keep that from happening, and the
    redundancy is deliberate -- `Executor.map` yields in submission order, and
    the returned list is *also* indexed by the position each job carries. The
    second survives a switch to `as_completed`, which is a plausible future edit
    for tighter progress reporting.

    `progress` is called with (completed, total) and is the only reporting this
    module does -- it writes to no stream itself, so a caller parsing stdout
    cannot be polluted by a progress line it did not ask for. In the parallel
    path it counts outcomes as they are *collected*, which is a lower bound on
    the work actually finished: a chunk that is still running holds back the
    count of the chunks behind it. It is a progress report, not a scheduler
    trace, and understating progress is the safe direction for one.
    """
    total = len(jobs)
    outcomes: list[_PairOutcome | None] = [None] * total

    def note(done: int) -> None:
        if progress is not None:
            progress(done, total)

    if workers == 1 or total <= 1:
        for done, (index, source_node_id, target_node_id, params) in enumerate(jobs, start=1):
            source_ppm, source_cwm = matrices[source_node_id]
            target_ppm, target_cwm = matrices[target_node_id]
            outcomes[index] = _register_and_calibrate(
                index, source_node_id, target_node_id,
                source_ppm, source_cwm, target_ppm, target_cwm, **params,
            )
            note(done)
        return [outcome for outcome in outcomes if outcome is not None]

    # Small chunks on purpose. Pair cost varies several-fold with core length, so
    # a chunk per worker would leave most of them idle behind whichever one drew
    # the long motifs; the pickling this saves is per-job overhead measured in
    # microseconds against a job that runs for ~0.1s at the default shuffles.
    chunksize = max(1, total // (workers * 8))
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(matrices,)) as pool:
        for done, outcome in enumerate(
            pool.map(_worker_pair, jobs, chunksize=chunksize), start=1,
        ):
            outcomes[outcome[0]] = outcome
            note(done)
    return [outcome for outcome in outcomes if outcome is not None]


def align_registry(registry_dir: str | os.PathLike[str], out_dir: str | os.PathLike[str],
                   *, null_shuffles: int = DEFAULT_NULL_SHUFFLES, seed: int = 0,
                   min_overlap_bp: int = DEFAULT_MIN_OVERLAP_BP,
                   min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
                   workers: int = DEFAULT_WORKERS,
                   progress: Callable[[int, int], None] | None = None,
                   ) -> tuple[AlignmentRunSummary, list[AlignmentEvidence]]:
    """Register every pair of motifs in a registry, calibrate a null for each,
    and write both the edge table and the null summary.

    Every matrix is trimmed to the node's declared `trimmed_core` first, so an
    emitted `offset` and `overlap_bp` are in core coordinates, not window ones,
    and `overlap_frac_*` is a fraction of each core rather than of the padding
    around it. A node whose registry record declares no usable core is excluded
    with the pairs it would have joined; see `_declared_core`.

    Four files are written, not two. `alignment_edges.parquet` and
    `alignment_null_summary.tsv` are the registered pairs;
    `alignment_excluded_pairs.tsv` is their complement over the same
    enumeration, one row per excluded pair with the reason and the two core
    lengths behind it; and `alignment_run_summary.json` carries the denominators
    (`n_pairs_considered == n_edges + n_pairs_excluded`) together with the
    overlap floor that split them. So a reader of the output directory can
    recover what this run looked at, not only what it kept.

    `null_shuffles` and `seed` are threaded through as the provenance the
    non-negotiable constraints require: every emitted edge carries both
    (`AlignmentEvidence.null_shuffles` / `.seed`), not just the run as a whole.

    `workers` splits the pair loop across processes and changes nothing else; see
    the module docstring for why that is safe here and what is tested to keep it
    so. `progress` is called with (completed_pairs, total_pairs) after each pair
    finishes; this function writes to no stream of its own, so what a caller sees
    on stdout is exactly what it saw before.
    """
    from motifmultiverse import guards
    from motifmultiverse.guard_log import GuardLog
    from motifmultiverse.ingest import load_registry
    from motifmultiverse.provenance import record

    if null_shuffles < 1:
        # Fail fast on a run-level parameter, before opening the registry or
        # writing anything: every pair would otherwise hit this same refusal
        # one at a time inside the loop below (see calibrate_pair_null).
        raise AlignmentError(f"--null-shuffles must be >= 1; got {null_shuffles}")
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        # Same reason, one level up: a bad worker count must not be discovered
        # after the provenance record and the pair list already exist. `True`
        # is rejected explicitly because `bool` is an `Integral` and `workers=True`
        # would otherwise mean "one worker", reading as "yes, parallelise".
        raise AlignmentError(
            f"workers must be an integer >= 1; got {workers!r}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Provenance is written before the registry is even opened, same as
    # compile's pattern: a run that fails to load its registry still leaves a
    # record of what was attempted (T-09).
    prov = record("align", seed=seed)
    # Checksummed BEFORE the write, and before the registry is opened. `align` was
    # the one stage whose provenance carried `"inputs": {}`, so
    # `alignment_edges.parquet` could not be tied to the registry bytes behind it
    # -- the single thing this package's provenance discipline exists to make
    # possible. Hashing does not need `load_registry` to succeed, only the files
    # to be on disk, so this keeps T-09 intact: the record is still written before
    # the body runs, and a run that cannot load its registry still leaves one --
    # now naming what it tried to read. A second `prov.write` would have appended
    # a second record and moved the `provenance_records` count that guard outcomes
    # join on.
    for name in ("registry.json", "arrays.h5"):
        candidate = Path(registry_dir) / name
        if candidate.exists():
            prov.add_input(candidate, key=f"registry:{name}")
    prov.write(out)

    meta, nodes, arrays = load_registry(registry_dir)
    try:
        # Read and trim every node's matrices ONCE, then let go of the file. Two
        # reasons, and only one of them is that the old per-pair read re-read the
        # same node's arrays n-1 times. The other is that an open HDF5 handle
        # must not still be live when worker processes are created: a forked
        # child inherits the parent's file descriptor and HDF5's own cached
        # state, and concurrent reads through that shared handle are exactly the
        # use h5py documents as unsupported. Closing here means the workers never
        # touch the file at all -- they get plain arrays.
        matrices: dict[str, tuple[Any, Any | None]] = {}
        # Every node gets a status, including the ordinary one: a node absent
        # from this map would be a node the run never decided about, and the
        # pair loop below reads it rather than inferring registrability from
        # membership in `matrices` -- so "excluded" always comes with a reason.
        node_status: dict[str, str] = {}
        core_bp: dict[str, int] = {}
        for node in nodes:
            node_id = node["node_id"]
            node_arrays = arrays[node_id]
            if "ppm" not in node_arrays:
                # Never averaged, never guessed: a node with no PPM at all has
                # no unsigned content to register on, so its pairs are excluded
                # rather than silently registered on CWM alone.
                node_status[node_id] = NODE_NO_PPM_ARRAY
                continue
            ppm = node_arrays["ppm"][:]
            # Trim to the declared core before anything is scored. Both matrices
            # of a node take the same window, because `register_pair` measures
            # the signed CWM at the registration the PPM chose and the two would
            # otherwise no longer describe the same positions.
            core = _declared_core(node, ppm.shape[0])
            if core is None:
                node_status[node_id] = NODE_NO_TRIMMED_CORE
                continue
            cwm = node_arrays["cwm"][:] if "cwm" in node_arrays else None
            node_status[node_id] = NODE_REGISTRABLE
            core_bp[node_id] = core[1] - core[0]
            matrices[node_id] = (
                ppm[core[0]:core[1]],
                None if cwm is None else cwm[core[0]:core[1]],
            )
    finally:
        arrays.close()

    params = {"null_shuffles": null_shuffles, "seed": seed,
              "min_overlap_bp": min_overlap_bp, "min_overlap_frac": min_overlap_frac}
    jobs: list[tuple[int, str, str, dict[str, Any]]] = []
    # Excluded pairs are keyed by their position in `combinations(nodes, 2)`,
    # not appended in the order they are discovered: exclusions arise at two
    # different stages (node filter here, overlap floor after the pair loop) and
    # a table that ran the first stage's rows before the second's would not be
    # in the order of the enumeration it is the complement of.
    excluded_rows: dict[int, dict[str, Any]] = {}
    #: job index -> that pair's position in the considered enumeration.
    job_pair_index: list[int] = []
    n_considered = 0
    for a, b in combinations(nodes, 2):
        a_id, b_id = a["node_id"], b["node_id"]
        pair_index = n_considered
        n_considered += 1
        a_status, b_status = node_status[a_id], node_status[b_id]
        if a_status != NODE_REGISTRABLE or b_status != NODE_REGISTRABLE:
            # No PPM, or no declared trimmed core: excluded above, NAMED here,
            # never registered on padding or on CWM alone.
            excluded_rows[pair_index] = _excluded_pair_row(
                a_id, b_id, EXCLUDED_NODE_NOT_REGISTRABLE,
                a_status, b_status, core_bp,
            )
            continue
        # The index IS the pair's position among the jobs, which is
        # `combinations` order; `_run_pairs` restores that order from it.
        job_pair_index.append(pair_index)
        jobs.append((len(jobs), a_id, b_id, params))

    edges: list[AlignmentEvidence] = []
    null_rows: list[dict[str, Any]] = []
    for job_index, evidence, null_row in _run_pairs(
        jobs, matrices, workers=workers, progress=progress,
    ):
        if evidence is None or null_row is None:
            # No offset met the bilateral overlap requirement for this pair.
            # Both endpoints were registrable -- it is the two cores' widths that
            # admit no window, which is why the row carries them.
            _job, a_id, b_id, _params = jobs[job_index]
            excluded_rows[job_pair_index[job_index]] = _excluded_pair_row(
                a_id, b_id, EXCLUDED_NO_OFFSET_MEETS_FLOOR,
                NODE_REGISTRABLE, NODE_REGISTRABLE, core_bp,
            )
            continue
        edges.append(evidence)
        null_rows.append(null_row)

    exclusions = [excluded_rows[index] for index in sorted(excluded_rows)]
    if n_considered != len(edges) + len(exclusions):
        # Not arithmetic-by-construction: `_run_pairs` reassembles outcomes by
        # index and drops any slot it never filled, so a scheduling bug that lost
        # a job would silently shrink both numerators while `n_pairs_considered`
        # stayed put -- a denominator that no longer adds up, published as if it
        # did. Cheaper to refuse than to write it.
        raise AlignmentError(
            f"pair accounting does not close: {n_considered} considered but "
            f"{len(edges)} edges + {len(exclusions)} excluded = "
            f"{len(edges) + len(exclusions)}; refusing to write a run summary "
            "whose denominator cannot be reconciled with its rows"
        )

    GuardLog("align", out).record(
        guards.sign_alignment([e.to_dict() for e in edges]),
        # Still no edge count here, and for the reason this comment always gave:
        # a denominator smuggled into a guard-outcome sentence lands in the one
        # artifact nobody would look for it in, as prose rather than as a field.
        # What has changed is that it is no longer written nowhere -- it is in
        # `alignment_run_summary.json` beside the edges, with the excluded pairs
        # named one per row in `alignment_excluded_pairs.tsv`. A guard outcome
        # records whether the question was asked; it is not the denominator's
        # home.
        subject=(
            "the alignment edges about to be written to alignment_edges.parquet, "
            f"registered under rule version {REGISTRATION_RULE_VERSION} "
            f"(null_shuffles={null_shuffles}, seed={seed})"
        ),
    ).raise_if_failed()

    edges_path = _write_edges(out, edges)
    null_summary_path = _write_null_summary(out, null_rows)
    excluded_pairs_path = _write_excluded_pairs(out, exclusions)
    # Every member of both vocabularies is emitted, including the ones this run
    # never used. A reason absent from the payload would read as "not counted";
    # a reason present with 0 says the run looked and found none, and it also
    # tells a reader what the other possibilities were.
    run_summary_path = _write_run_summary(out, {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "n_nodes": len(nodes),
        "n_nodes_by_status": {
            status: sum(1 for value in node_status.values() if value == status)
            for status in NODE_STATUSES
        },
        "n_pairs_considered": n_considered,
        "n_edges": len(edges),
        "n_pairs_excluded": len(exclusions),
        "n_pairs_excluded_by_reason": {
            reason: sum(1 for row in exclusions if row["exclusion_reason"] == reason)
            for reason in PAIR_EXCLUSION_REASONS
        },
        "null_shuffles": null_shuffles,
        "seed": seed,
        # The floor is what decided the split, and until now it was recorded
        # nowhere -- so `no_offset_meets_overlap_floor` named a threshold a
        # reader had no way to know the value of.
        "min_overlap_bp": min_overlap_bp,
        "min_overlap_frac": min_overlap_frac,
        "registration_rule_version": REGISTRATION_RULE_VERSION,
        # File NAMES, not paths: these three sit in the directory this file sits
        # in, and an absolute path would make the artifact depend on where the
        # run happened to be written.
        "edges_file": edges_path.name,
        "null_summary_file": null_summary_path.name,
        "excluded_pairs_file": excluded_pairs_path.name,
    })

    summary = AlignmentRunSummary(
        n_nodes=len(nodes),
        n_pairs_considered=n_considered,
        n_edges=len(edges),
        n_pairs_excluded=len(exclusions),
        null_shuffles=null_shuffles,
        seed=seed,
        registration_rule_version=REGISTRATION_RULE_VERSION,
        edges_path=str(edges_path),
        null_summary_path=str(null_summary_path),
        workers=workers,
        min_overlap_bp=min_overlap_bp,
        min_overlap_frac=min_overlap_frac,
        excluded_pairs_path=str(excluded_pairs_path),
        run_summary_path=str(run_summary_path),
    )
    return summary, edges


#: The stage entry point named in `__all__`; kept as a plain alias so a caller
#: importing `align.run` gets exactly `align_registry`, not a wrapper that
#: could drift from it.
run = align_registry
