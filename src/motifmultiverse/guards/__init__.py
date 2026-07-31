"""Executable constraints.

Every guard here returns a :class:`GuardResult` and every guard has a
*falsification* test in ``tests/test_guards.py`` that feeds it a shifted,
permuted or reordered input and requires it to FAIL (T-15).

Why that requirement exists: in the reference implementation 5 framework guards
all passed, but a later falsification pass showed 2 of them still passed under a
row-shifted *and* a permuted lexicon index, and none of the 5 could detect a
reordered index. A guard that has never been shown to fail is not evidence, and
in a report a vacuous guard and a correct guard look identical. See
``docs/CONSTRAINTS.md``.

T-16: a guard never reads its own output, and never hard-codes an upstream row
count. Where an upstream count is needed it is read from the upstream artifact
and compared.
"""
from __future__ import annotations

import ast
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from motifmultiverse.schema import Missingness

__all__ = [
    "GuardResult", "GuardError", "PendingGuardInput", "ALL_GUARDS",
    "GUARDS_AWAITING_INPUT", "CROSS_MODEL_AXES", "run_all",
    "single_scale", "variant_id_unique", "no_key_parsing", "four_state_missingness",
    "no_cross_model_cwm_avg", "no_cross_estimand_pooling", "sign_alignment",
    "interaction_required",
    "estimability_floor", "stratum_parity", "short_motif_flag", "single_family_layer",
    "selection_provenance_declared", "health_before_effect", "comparator_declared",
    "index_order_matches_loader",
]


class GuardError(AssertionError):
    """A guard rejected its input."""


@dataclass(frozen=True)
class GuardResult:
    guard_id: str
    passed: bool
    detail: str

    def raise_if_failed(self) -> GuardResult:
        if not self.passed:
            raise GuardError(f"{self.guard_id}: {self.detail}")
        return self


def _ok(gid: str, detail: str = "ok") -> GuardResult:
    return GuardResult(gid, True, detail)


def _fail(gid: str, detail: str) -> GuardResult:
    return GuardResult(gid, False, detail)


# --------------------------------------------------------------------------- #
def single_scale(records: Iterable[Mapping[str, Any]]) -> GuardResult:
    """All results in one analysis must carry the same ``input_scale``.

    The hit caller is not input-scale invariant: in the reference implementation
    the same regions produced different discrete retention decisions depending on
    which *other* regions shared the input, with the onset measured between
    6,460 and 7,085 regions -- under 10% growth on the base set. Specifications
    must therefore be subsets of ONE frozen run, never re-called per
    specification.
    """
    gid = "single_scale"
    scales = {r.get("input_scale") for r in records}
    if not scales:
        return _fail(gid, "no records carried an input_scale provenance field")
    if None in scales:
        return _fail(gid, "a record is missing the input_scale provenance field")
    if len(scales) != 1:
        return _fail(gid, f"results span {len(scales)} input scales: {sorted(scales)}")
    return _ok(gid, f"single input_scale={scales.pop()}")


def variant_id_unique(nodes: Sequence[Any]) -> GuardResult:
    """``variant_id`` unique and 1:1 with ``denovo_pattern_id``."""
    gid = "variant_id_unique"
    vids = [getattr(n, "variant_id", None) or n["variant_id"] for n in nodes]
    keys = [getattr(n, "denovo_pattern_id", None) or n["denovo_pattern_id"] for n in nodes]
    if len(set(vids)) != len(vids):
        dupes = sorted({v for v in vids if vids.count(v) > 1})
        return _fail(gid, f"variant_id collisions: {dupes}")
    if len(set(keys)) != len(keys):
        return _fail(gid, "denovo_pattern_id collisions; the mapping cannot be 1:1")
    if len(set(zip(vids, keys, strict=True))) != len(set(vids)):
        return _fail(gid, "variant_id -> denovo_pattern_id is not 1:1")
    return _ok(gid, f"{len(vids)} variant_ids, 1:1, no collisions")


def no_key_parsing(source: str) -> GuardResult:
    """Heuristic static check: no semantics may be sliced out of an identifier string.

    This is the guard for the failure where a hit-caller row number was matched
    against a discovery manifest id, filing one factor's evidence under another's
    name.

    This is a syntactic AST scan, not a dataflow analysis: it only sees a slice
    or prefix test performed directly on one of a fixed set of watched names. An
    alias (``x = variant_id; x.split(...)``) or a value that reaches the same
    operation through a parameter, attribute, or return value is invisible to it.
    A pass here is evidence about the literal source text scanned, not a proof
    that no identifier's semantics are parsed anywhere in the call graph -- hence
    the success detail says "heuristic scan passed", and the frozen principle
    this guard partially covers is not classified ENFORCED on its strength alone.
    """
    gid = "no_key_parsing"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - defensive
        return _fail(gid, f"could not parse source: {exc}")
    ident_names = {"variant_id", "pattern_key", "motif_name", "node_id", "region_id", "peak_id"}
    offenders: list[str] = []
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.Subscript):
            target = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"split", "startswith", "endswith", "strip", "removeprefix"}:
                target = node.func.value
        if target is None:
            continue
        name = getattr(target, "id", None) or getattr(target, "attr", None)
        if name in ident_names:
            offenders.append(f"line {node.lineno}: {name}")
    if offenders:
        return _fail(gid, "semantics parsed from identifiers -> " + "; ".join(offenders))
    return _ok(gid, "heuristic scan passed")


@dataclass(frozen=True)
class PendingGuardInput:
    """Why one guard has no call site, in the three parts a reader can check.

    A prose note saying a guard is "waiting for the right input" is unfalsifiable:
    it never becomes wrong, so nobody ever revisits it. The entries below used to
    be exactly that, and two of them were vague enough to be re-derived from
    scratch by anyone who wanted to know whether the moment had arrived. Splitting
    the note into three fields fixes both halves of that:

    ``nearest_artifact``
        The dotted name of the thing this release DOES emit that comes closest to
        the guard's input. It is a real symbol: the test
        ``test_every_pending_entry_names_a_real_artifact`` resolves every one of
        them, so an entry cannot point at something that was renamed away or
        never existed.
    ``why_not_a_call_site``
        The failure that pointing the guard at that artifact would cause. There are
        only three kinds in this project and naming which one applies is the whole
        content of the entry: the guard would **corroborate itself** (the claim and
        the recomputation are the same code), it would be **vacuous** (it cannot
        fail on any input the artifact can produce), or it would **override a
        decision nobody has made** (the only passing state is one the record has no
        way to express).
    ``closes_when``
        The concrete thing whose existence closes the entry, stated so that a
        future reader recognises the moment instead of re-deriving this analysis.
        Where a scientific decision is still owed, it says so and names it, because
        an entry that lists only the missing field invites someone to add the field
        and wire a guard whose semantics are still undecided.

    ``__str__`` renders the three as one sentence, so a reader (or a report) that
    printed the old string value still gets a sentence rather than a repr.
    """

    nearest_artifact: str
    why_not_a_call_site: str
    closes_when: str

    def __str__(self) -> str:
        return (f"nearest artifact in this release: {self.nearest_artifact}. "
                f"{self.why_not_a_call_site} Closes when: {self.closes_when}")


#: Guards with no call site in this release, and the input each is waiting for.
#:
#: A guard that is defined, exported and never invoked reads as protection. Seven
#: of the fifteen were in that position, including `four_state_missingness` -- the
#: guard for this project's founding failure. Silence about that is the same shape
#: as the failure itself, so the gap is data here rather than something a reader
#: has to discover by grepping. `test_every_guard_is_called_or_declared_pending`
#: fails if a guard is neither invoked in `src/` nor listed below, so a new orphan
#: cannot appear quietly and a wired-up guard cannot stay on this list.
#:
#: One test in `tests/test_guards.py` accompanies each entry and asserts the
#: executable half of its claim -- that the interval nobody writes is still not
#: written, that the manifest still states no defined count, that a single-family
#: budget share still comes out 1.0. Those tests FAIL when the awaited input
#: arrives, which is the point: the registry announces the moment instead of
#: waiting to be re-read.
GUARDS_AWAITING_INPUT: dict[str, PendingGuardInput] = {
    "interaction_required": PendingGuardInput(
        nearest_artifact="interpret.FamilyEffect",
        why_not_a_call_site=(
            "This release estimates one family's effect against one comparator, never "
            "an interaction. `FamilyEffect` has no `is_specificity_claim` and no "
            "`interaction_ci`, and the guard skips every record that does not claim "
            "specificity -- so handed the effects table it would iterate over rows it "
            "never inspects and pass. What that leaves open is worth stating plainly, "
            "because it is the failure the guard was written for and this release does "
            "not prevent it: two per-family effects sitting side by side in one table "
            "are exactly the material a reader assembles a specificity claim from by "
            "differencing two significance statuses."
        ),
        closes_when=(
            "a stage emits a claim carrying an interaction estimate -- one family's "
            "effect in cell A minus its effect in cell B, with an interval on that "
            "difference rather than on either half -- and marks it "
            "`is_specificity_claim`. A record that merely gains the two fields while "
            "the interval is still computed on one cell at a time does not close this: "
            "the guard would then read an interval that answers a different question."
        ),
    ),
    "estimability_floor": PendingGuardInput(
        nearest_artifact="validate.StabilityResult",
        why_not_a_call_site=(
            "`StabilityResult` is the only record in the project with all three of the "
            "guard's parts in one place: an N (`n_affected_peaks`), an interval "
            "(`affected_interval`), and a quantity that N-limited interval would be "
            "read against (`paired_delta_reconstruction_all`, the all-peak median "
            "delta). Two things stop it being a call site. Nothing ever writes the "
            "interval -- `validate.evaluate_stability` is its only producer and leaves "
            "it None -- so the CI clause has no data. And the N clause is already an "
            "invariant of the record itself: `StabilityResult.__post_init__` refuses a "
            "result under `MIN_AFFECTED_PEAKS` that is not LOW_RISK_RARE_NOT_VALIDATED, "
            "so the guard could only fail on an object the constructor had already "
            "refused to build.\n"
            "`interpret.FamilyEffect` is the other candidate and fails differently. Its "
            "N is the block count, and `interpret.health_report` floors that against "
            "the run's pre-registered `HealthFloors.min_blocks` BEFORE any effect is "
            "computed, over a block set the effect frame can only add to -- so at the "
            "run's own floor the guard cannot fail, and at a hard-coded 30 it would "
            "override a floor the run declared, which is the opposite of "
            "pre-registration. `n_bootstrap_valid` is not an alternative N: it is a "
            "replicate count, and 2,000 replicates drawn from two blocks would clear "
            "any floor placed on it. There is also no reference -- the comparator is "
            "already differenced out, so the interval is read against zero alone and "
            "'contains zero AND the reference' has no second quantity to name."
        ),
        closes_when=(
            "either producer starts emitting the part it is missing: "
            "`evaluate_stability` computes `affected_interval` (at which point the "
            "reference is already there and the guard's CI clause becomes live), or "
            "effects are emitted per stratified cell, each with its own N and an "
            "interval read against a NAMED second estimate. One decision is owed "
            "before wiring either, and this entry is not closed without it: whether an "
            "under-powered cell is emitted carrying NOT_ESTIMABLE_UNDERPOWERED and no "
            "direction (`docs/CONSTRAINTS.md` FP-12's wording) or refused outright, the "
            "way `infer.bca_paired_block_interval` refuses below "
            "`infer.MIN_ESTIMABLE_BLOCKS`. The guard accepts either; the records above "
            "can express only the second.\n"
            "A SEPARATE question that used to be filed here has been decided and is "
            "not part of what this entry waits for. `_effects_percentile` floors "
            "replicates and never blocks, while the BCa and wild-cluster paths refuse "
            "below `infer.MIN_ESTIMABLE_BLOCKS` (30) -- so the same 6-block frame is an "
            "interval on one path and a refusal on the other. That asymmetry is kept, "
            "deliberately: 30 is derived from what a jackknife needs to estimate BCa\'s "
            "acceleration, a quantity the percentile interval does not compute, and "
            "importing it would be borrowing a constant derived for another estimator "
            "and presenting it as this one\'s requirement. What was missing was not a "
            "floor but disclosure, and every effect now records "
            "`estimator_min_blocks` -- None on the percentile path, saying in the "
            "artifact that no block floor was enforced there -- beside the "
            "`n_blocks` it was computed from and the run\'s own declared "
            "`HealthFloors.min_blocks`."
        ),
    ),
    "stratum_parity": PendingGuardInput(
        nearest_artifact="interpret.FamilyComposition",
        why_not_a_call_site=(
            "Nothing in this release stratifies. `interpret.compose` and "
            "`interpret.estimate_effects` partition by family, and a family is an "
            "identity read off the frozen lexicon, not a variable produced by a "
            "stratifying RULE -- which is what the guard compares. Passing family_id "
            "as a stratum variable would hand every cell the same single rule and make "
            "the guard incapable of failing, since two rules for one variable is the "
            "only thing it can detect."
        ),
        closes_when=(
            "cells are cut by a variable this pipeline COMPUTES -- a promoter/distal "
            "call, a GC bin, an accessibility tertile -- and each cell records the rule "
            "that produced each such variable, so two cells whose 'promoter' came from "
            "different definitions can be caught. `docs/CONSTRAINTS.md` FP-24 records "
            "that the ordering clause of the same principle (the cross-tabulation comes "
            "before any effect is visible) is a separate check this guard does not make."
        ),
    ),
    "single_family_layer": PendingGuardInput(
        nearest_artifact="infer.UsageDefinition.BUDGET_FRACTION",
        why_not_a_call_site=(
            "`interpret.FamilyComposition.peak_share` is not the share this guard is "
            "about, and the question of WHICH it is -- undefined, or defined but out "
            "of the guard's scope -- has an answer that can be measured rather than "
            "argued. It is DEFINED AND OUT OF SCOPE. Its denominator is searched "
            "PEAKS, so a single-family composition reports the fraction of searched "
            "peaks carrying the only family present, and single-familyness does not "
            "force that fraction to anything: on the real K562 substrate "
            "(576,589 rows, 33,917 peaks) a composition restricted to one family "
            "comes out 0.310670 for CTCF/CTCFL-like and 0.998143 for AP-1/bZIP. Both "
            "are measurements; neither is 1.0; a 1.0 there would be the measurement "
            "that every searched peak carries it. The share this guard rules on is "
            "the other kind -- a family's share of its peak's own family layer, whose "
            "denominator is the other families -- where one family forces 1.0 "
            "IDENTICALLY and the number carries no information. Giving "
            "FamilyComposition an estimability status would therefore not be wiring "
            "this guard; it would be inventing an estimability semantics for a "
            "quantity that is estimable, and marking a real 1.0 NOT_ESTIMABLE would "
            "suppress a finding.\n"
            "The share the guard IS about -- a family's share of its peak's own family "
            "layer -- does exist, in `infer._usage_predicate` under BUDGET_FRACTION: "
            "`abs_coefficient_sum / peak_abs_coefficient_sum`. On a peak whose only "
            "family with mass is that one, the ratio is exactly 1.0 and clears every "
            "threshold in (0, 1], so BUDGET_FRACTION degenerates to ANY_HIT there and "
            "the occupancy margin of `infer.two_part_summary` is inflated by peaks "
            "whose share was forced by there being nothing else to hold mass. That is "
            "this guard's failure, live, in an artifact this release emits. It still "
            "cannot be wired: `infer.PeakUsage` carries no count of families with mass, "
            "so 'one family' is not observable where the share is computed, and there "
            "is no per-peak estimability state to set -- the guard's only passing state "
            "for a single-family stratum is a NOT_ESTIMABLE the record cannot express, "
            "so wiring it today would refuse every BUDGET_FRACTION run outright."
        ),
        closes_when=(
            "`PeakUsage` carries the number of families with mass in its peak (or the "
            "share is computed in `interpret._peak_usage`, where that number is "
            "already known), AND a decision has been made about what a single-family "
            "peak does to a BUDGET_FRACTION denominator: dropped from it as "
            "NOT_ESTIMABLE, or kept with its share recorded as undefined rather than "
            "1.0. The field alone is not enough -- it would let the guard fire on every "
            "real substrate with no state in which the run could legitimately proceed. "
            "What does NOT close it, and has now been looked at twice: an estimability "
            "status on `FamilyComposition`. That entry point is settled above -- the "
            "share there is defined and the guard's rule does not reach it -- so a "
            "future round should spend its attention on PeakUsage and the decision, "
            "not on re-deriving the composition question."
        ),
    ),
}


def four_state_missingness(
    rows: Sequence[Mapping[str, Any]],
    claimed_coverage: float,
    claimed_defined: int,
    claimed_total: int,
    value_key: str,
    state_key: str = "missingness",
) -> GuardResult:
    """The claimed coverage/defined/total are checked, never trusted.

    This is the guard for the project's founding failure: a coverage figure
    computed AFTER an undefined value had already been filled reported
    1.000000, and because the number that shipped was never compared against
    anything, it supplied its own evidence of correctness. Checking rows for a
    literal 0 cannot catch that failure -- a fill can write any value, not just
    zero, and the arithmetic that produced 1.000000 ran downstream of the fill,
    on data that no longer showed which rows had been undefined. The only check
    that can catch it is an independent recomputation of ``defined``, ``total``
    and coverage from the raw missingness rows, compared against what the
    artifact claims. A mismatch fails even when no row contains a numeric zero.
    """
    gid = "four_state_missingness"
    missing_state = [i for i, r in enumerate(rows) if state_key not in r]
    if missing_state:
        return _fail(gid, f"{len(missing_state)} rows carry no missingness state")
    # The guard is named for four states, so it has to check that there are four.
    # It used to accept any string in the state column: a row reading
    # `missingness: "banana"` counted as not-used and passed, which makes the
    # recomputation below agree with a claim derived from the same nonsense.
    legal = {m.value for m in Missingness}
    illegal = sorted({str(r.get(state_key)) for r in rows} - legal)
    if illegal:
        return _fail(
            gid,
            f"missingness values outside the four states: {illegal[:5]} "
            f"(legal: {sorted(legal)})",
        )
    # The zero-collapse check is supplementary -- see the docstring -- but it must
    # actually run. `value_key` used to default to "statistic", which is the name
    # this module's own test fixtures use and not a column any artifact in this
    # project has: on real hit rows `.get` returned None, `None == 0` was False,
    # and the check silently passed on everything. The default agreed with the
    # tests instead of with the data, so the tests could not notice. It is now
    # required -- the guard cannot guess which column holds the value that a fill
    # could have written into -- and a column absent from every row fails rather
    # than passes, because a guard that no-ops certifies rather than checks.
    if value_key not in {k for r in rows for k in r}:
        return _fail(
            gid,
            f"no row carries the value column {value_key!r}; pass value_key= naming "
            "the column whose undefined entries could have been filled",
        )
    bad = [i for i, r in enumerate(rows)
           if r.get(state_key) != "used" and r.get(value_key) == 0]
    if bad:
        return _fail(gid, f"{len(bad)} undefined values collapsed to 0 (rows {bad[:5]})")
    total = len(rows)
    defined = sum(1 for r in rows if r.get(state_key) == "used")
    coverage = defined / total if total else float("nan")

    def _matches(claimed: float, recomputed: float) -> bool:
        if math.isnan(claimed) or math.isnan(recomputed):
            return math.isnan(claimed) and math.isnan(recomputed)
        # Coverage is conventionally reported rounded to 4 decimal places (e.g. a
        # report or plot label showing "0.6667" for 2/3). An exact-match tolerance
        # (the previous rel_tol=1e-9) makes this comparison effectively exact, so a
        # legitimately-rounded claim FALSE-FAILS against a full-precision
        # recomputation -- and a guard that cries wolf on correct input gets
        # disabled by the next maintainer, which is worse than a weaker guard. The
        # largest rounding error a 4-decimal display can introduce is 0.5e-4
        # (5e-5); abs_tol=1e-4 covers that with a 2x margin. It still catches the
        # project's founding failure (a claimed coverage of 1.000000 against a
        # true 0.5 -- four orders of magnitude past this tolerance) and a coarser
        # 2-decimal-rounded claim (error ~3.3e-3), neither of which is display
        # rounding.
        return math.isclose(claimed, recomputed, rel_tol=1e-4, abs_tol=1e-4)

    mismatches = []
    if claimed_total != total:
        mismatches.append(f"total: claimed {claimed_total}, recomputed {total}")
    if claimed_defined != defined:
        mismatches.append(f"defined: claimed {claimed_defined}, recomputed {defined}")
    if not _matches(claimed_coverage, coverage):
        mismatches.append(f"coverage: claimed {claimed_coverage!r}, recomputed {coverage:.6f}")
    if mismatches:
        return _fail(
            gid,
            "claimed coverage/defined/total do not match a recomputation from the raw "
            "rows -> " + "; ".join(mismatches),
        )
    return _ok(
        gid,
        f"claimed coverage={claimed_coverage:.6f}, defined={defined}, total={total} "
        "match a recomputation from the raw rows",
    )


#: The axes a CWM combination must hold fixed. Exported because the operations
#: log that feeds `no_cross_model_cwm_avg` has to report which of them each
#: operation held fixed: two hand-maintained lists of the same three strings
#: would be free to drift into disagreeing about what "cross-model" means, and
#: the disagreement would show up as a guard that quietly stops testing an axis.
CROSS_MODEL_AXES = ("model", "readout", "metacluster")


def no_cross_model_cwm_avg(operations: Iterable[Mapping[str, Any]]) -> GuardResult:
    """Averaging CWMs across model / readout / metacluster is a design prohibition.

    Each operation states what it did (``op``) and which axes it held fixed
    (``group_by``). The guard is only as good as where that pair comes from: an
    operation record written by the stage that performed the operation is that
    stage testifying about itself, and it passes for exactly as long as the
    testimony is kept up to date by hand. `compile.operations_log` therefore does
    not ask the writer -- it reads the emitted lexicon back and classifies each
    motif against the registry arrays it stands for, so ``op`` is a property of
    the bytes rather than of anyone's intent.

    What this guard does NOT cover, stated here because a reader would otherwise
    read a pass as broader than it is. Two things, and the second is the larger.

    1. A representative averaged WITHIN one model holds all three axes fixed and
       passes (`docs/CONSTRAINTS.md` FP-05). The prohibition on constructed
       representatives as such is a separate rule, enforced at compile by
       requiring a representative to be one of its own members.
    2. **The check reaches back exactly as far as the operations log does, and no
       further.** `compile.operations_log` classifies the emitted lexicon against
       the registry `ingest` wrote, so it sees combination performed between those
       two points -- and a cross-model mean performed *before* the registry, where
       a real meta-analysed-CWM stage would live, arrives as an ordinary registry
       motif and is classified `copy`. That is not a hypothetical: it is pinned by
       `test_ingest_compile`'s
       `test_a_cross_model_mean_made_upstream_of_the_registry_passes`, which builds
       one and asserts this guard passes on it. So a pass here means
       "nothing downstream of the registry averaged", never "this lexicon contains
       no cross-model average" -- and because the pass sentence is persisted
       verbatim by `guard_log` and printed verbatim by `report`, it says so
       itself rather than relying on a reader finding this docstring.
    """
    gid = "no_cross_model_cwm_avg"
    for op in operations:
        if op.get("op") not in {"mean", "average"}:
            continue
        grouped = set(op.get("group_by", ()))
        for axis in CROSS_MODEL_AXES:
            if axis not in grouped:
                return _fail(gid, f"CWM {op['op']} does not hold {axis} fixed (group_by={sorted(grouped)})")
    return _ok(
        gid,
        "no CWM combination recorded in this log averages across model, readout or "
        "metacluster; combination performed before the operations were recorded is "
        "outside what this checked",
    )


def sign_alignment(alignments: Iterable[Mapping[str, Any]]) -> GuardResult:
    """Registration must be established on unsigned PPM similarity.

    An aligner that maximises *signed* CWM cosine is structurally blind to a
    sign-flipped motif: at its true registration the signed cosine is near -1, so
    the correct offset can never win. Run that way, a sign survey returns "no
    flips" as an artefact of the instrument.
    """
    gid = "sign_alignment"
    for a in alignments:
        if a.get("registered_on") != "unsigned_ppm":
            return _fail(gid, f"alignment registered on {a.get('registered_on')!r}, not unsigned_ppm")
        if a.get("signed_similarity_used_for_registration"):
            return _fail(gid, "signed similarity was used to choose the offset")
    return _ok(gid, "registration is unsigned-PPM; signed similarity is a separate statistic")


def interaction_required(claims: Iterable[Mapping[str, Any]]) -> GuardResult:
    """A specificity claim needs an interaction CI excluding zero."""
    gid = "interaction_required"
    for c in claims:
        if not c.get("is_specificity_claim"):
            continue
        ci = c.get("interaction_ci")
        if ci is None:
            return _fail(gid, f"{c.get('id')}: specificity claimed with no interaction estimate")
        lo, hi = ci
        if lo <= 0.0 <= hi:
            return _fail(gid, f"{c.get('id')}: interaction CI [{lo}, {hi}] includes zero")
        if c.get("derived_from") == "difference_of_significance":
            return _fail(gid, f"{c.get('id')}: derived from two significance statuses")
    return _ok(gid, "every specificity claim rests on an interaction interval excluding zero")


def estimability_floor(cells: Iterable[Mapping[str, Any]], n_min: int = 30) -> GuardResult:
    """N >= 30, and a CI containing both zero and the reference is NOT_ESTIMABLE."""
    gid = "estimability_floor"
    for c in cells:
        n = c.get("n")
        if n is None:
            return _fail(gid, f"{c.get('id')}: no N recorded")
        ci, ref = c.get("ci"), c.get("reference")
        if n < n_min and c.get("status") != "NOT_ESTIMABLE_UNDERPOWERED":
            return _fail(gid, f"{c.get('id')}: N={n} < {n_min} but not marked NOT_ESTIMABLE_UNDERPOWERED")
        if ci is not None and ref is not None:
            lo, hi = ci
            if lo <= 0.0 <= hi and lo <= ref <= hi and c.get("status") != "NOT_ESTIMABLE_UNDERPOWERED":
                return _fail(gid, f"{c.get('id')}: CI contains both 0 and the reference {ref}")
    return _ok(gid, f"all cells meet N>={n_min} or are marked NOT_ESTIMABLE_UNDERPOWERED")


def stratum_parity(cells: Iterable[Mapping[str, Any]]) -> GuardResult:
    """Stratifying variables entering an interaction come from ONE rule."""
    gid = "stratum_parity"
    rules: dict[str, set[str]] = {}
    for c in cells:
        for var, rule in (c.get("stratum_rules") or {}).items():
            rules.setdefault(var, set()).add(rule)
    bad = {v: sorted(r) for v, r in rules.items() if len(r) > 1}
    if bad:
        return _fail(gid, f"stratum variables defined by more than one rule: {bad}")
    return _ok(gid, f"{len(rules)} stratum variables, each from a single rule")


def short_motif_flag(nodes: Sequence[Any]) -> GuardResult:
    """Short / weakly-supported motifs must carry ``low_confidence_annotation``.

    The thresholds are annotate/README.md's: PWM <= 6 bp, or TomTom q > 0.05, or
    seqlet count < 100.

    Two ways this used to pass what it exists to catch:

    * ``(motif_length or 99) <= 6`` reads a legitimate **zero** as absent, so the
      weakest possible motif -- length 0, zero seqlets -- was not short and not
      low-support, and passed unflagged. Missing and zero are different claims;
      ``or`` cannot tell them apart.
    * a non-numeric value, including this package's own ``MISSING_SENTINEL``,
      raised ``TypeError: '<=' not supported between 'str' and 'int'`` out of the
      guard. A guard that crashes on a value its own schema produces is not
      reporting on the data, it is reporting on itself.

    Absent (``None``) is still not an offence: no measurement is not evidence of a
    short motif. It is reported separately so it cannot be mistaken for a pass.
    """
    gid = "short_motif_flag"

    def g(n: Any, k: str) -> Any:
        return getattr(n, k) if hasattr(n, k) else n.get(k)

    def number(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return math.nan          # unmeasurable -> reported, never compared
        return float(value)

    offenders: list[Any] = []
    unusable: list[Any] = []
    for n in nodes:
        length, seqlets = number(g(n, "motif_length")), number(g(n, "seqlet_count"))
        q = number((g(n, "annotation_matches") or {}).get("tomtom_q"))
        if any(v is not None and math.isnan(v) for v in (length, seqlets, q)):
            unusable.append(g(n, "variant_id"))
            continue
        short = length is not None and length <= 6
        few = seqlets is not None and seqlets < 100
        weak_q = q is not None and q > 0.05
        if (short or weak_q or few) and not g(n, "low_confidence_annotation"):
            offenders.append(g(n, "variant_id"))
    if unusable:
        return _fail(
            gid,
            f"{len(unusable)} node(s) carry a non-numeric motif_length / seqlet_count / "
            f"tomtom_q, so the threshold cannot be applied: {unusable[:5]}",
        )
    if offenders:
        return _fail(gid, f"unflagged low-confidence annotations: {offenders[:5]}")
    return _ok(gid, "every short / weak / low-support motif is flagged")


def single_family_layer(strata: Iterable[Mapping[str, Any]]) -> GuardResult:
    """A stratum with one family has an undefined share, not a share of 1.0."""
    gid = "single_family_layer"
    for s in strata:
        if len(set(s.get("families", ()))) <= 1:
            if s.get("within_peak_share") == 1.0:
                return _fail(gid, f"{s.get('id')}: single-family stratum recorded share=1.0")
            if s.get("status") != "NOT_ESTIMABLE":
                return _fail(gid, f"{s.get('id')}: single-family stratum not marked NOT_ESTIMABLE")
    return _ok(gid, "single-family strata are NOT_ESTIMABLE, never 1.0")


# --------------------------------------------------------------------------- #
# The three guards below cover the peak-set query path (interpret). They are the
# executable half of FP-20 and BA-16 / BA-18.
# --------------------------------------------------------------------------- #
def selection_provenance_declared(queries: Iterable[Mapping[str, Any]]) -> GuardResult:
    """A peak-set query declares how it was selected, and silence costs it inference.

    Two failures, not one. The obvious one is an undeclared query. The one that
    actually happens is an undeclared query that was *treated as* externally
    defined, because the lookup that resolves the grade had a permissive default.
    A missing declaration must land in the most conservative mode.
    """
    gid = "selection_provenance_declared"
    from motifmultiverse.schema import (
        MOST_CONSERVATIVE_OUTPUT_MODE,
        SelectionProvenance,
        output_mode_for,
    )
    valid = {g.value for g in SelectionProvenance}
    for q in queries:
        qid = q.get("query_id", "<unnamed>")
        if "selection_provenance" not in q:
            return _fail(gid, f"{qid}: no selection_provenance field at all")
        grade = q.get("selection_provenance")
        mode = q.get("output_mode")
        if mode is None:
            return _fail(gid, f"{qid}: declared {grade!r} but emitted no output_mode")
        undeclared = (
            grade in (None, "", "NA")
            or grade == SelectionProvenance.DECLARATION_MISSING.value
            or grade not in valid
        )
        if undeclared and mode != MOST_CONSERVATIVE_OUTPUT_MODE.value:
            return _fail(
                gid,
                f"{qid}: selection_provenance {grade!r} is undeclared or unrecognised but the "
                f"query ran in {mode!r}; an undeclared query takes "
                f"{MOST_CONSERVATIVE_OUTPUT_MODE.value}, never a permissive default",
            )
        if not undeclared and mode != output_mode_for(grade).value:
            return _fail(
                gid,
                f"{qid}: grade {grade!r} dispatches to {output_mode_for(grade).value} "
                f"but the query ran in {mode!r}",
            )
    return _ok(gid, "every query declares its selection source and ran in the dispatched mode")


def health_before_effect(report: Mapping[str, Any]) -> GuardResult:
    """The three health numbers precede any effect, and a floor failure suppresses it.

    A disclaimer attached to an effect size is not a control, because the effect
    size is what gets quoted. If a submitted peak set barely intersects the
    universe, spans too few blocks to have an effective sample size, or is mostly
    unexplained by the frozen lexicon, the reading is withheld rather than
    annotated.
    """
    gid = "health_before_effect"
    required = ("intersection_coverage", "n_blocks", "explained_fraction")
    health = report.get("health") or {}
    failures = list(report.get("floor_failures") or ())
    absent = [k for k in required if k not in health]
    if absent:
        return _fail(gid, f"health numbers absent: {absent}")
    # A null number is admissible only as a recorded failure. An undefined ratio
    # that passes silently is the same defect as a coverage computed after filling.
    nulls = [k for k in required if health.get(k) is None]
    if nulls and not failures:
        return _fail(gid, f"health numbers are null but no floor failure was recorded: {nulls}")
    order = list(report.get("emitted_order") or ())
    if not order or order[0] != "health":
        return _fail(gid, f"health is not the first section emitted (order={order})")
    effects = report.get("effects") or []
    if failures and effects:
        return _fail(gid, f"{len(effects)} effects emitted while floors failed: {failures}")
    if failures and report.get("interpretation_emitted"):
        return _fail(gid, f"interpretation emitted while floors failed: {failures}")
    return _ok(gid, f"health first over {len(order)} sections; floor failures={failures or 'none'}")


def comparator_declared(claims: Iterable[Mapping[str, Any]]) -> GuardResult:
    """A cross-condition effect carries the identity of the peak set it is against.

    This is the most transferable rule the reference line produced. One set of
    measurements supported both "replicates exactly" and "four times stronger,
    prediction falsified"; the only difference was whether the baseline was the
    unselected universe or a residual subset from which the relevant peaks had
    already been removed. A frozen lexicon transfers between projects. A
    comparator does not.

    So: no baseline, no number. And one effect that carries two baselines is not
    one number with an ambiguity -- it is two results, and the guard refuses to
    let them be reported as one.
    """
    gid = "comparator_declared"
    seen: dict[str, set[str]] = {}
    for c in claims:
        if not c.get("is_cross_condition", True):
            continue
        cid = c.get("id", "<unnamed>")
        comparator = c.get("comparator_id")
        if comparator in (None, "", "NA"):
            return _fail(gid, f"{cid}: cross-condition effect with no comparator_id")
        if isinstance(comparator, (list, tuple, set)):
            names = sorted(str(x) for x in comparator)
            if len(set(names)) > 1:
                return _fail(gid, f"{cid}: one effect carrying {len(names)} baselines: {names}")
            comparator = names[0]
        seen.setdefault(str(cid), set()).add(str(comparator))
    ambiguous = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    if ambiguous:
        return _fail(
            gid,
            "the same effect is reported against more than one baseline, so no single number "
            f"is licensed: {ambiguous}",
        )
    return _ok(gid, f"{len(seen)} cross-condition effects, each against exactly one named baseline")


def index_order_matches_loader(index_names: Sequence[str],
                               loader_names: Sequence[str]) -> GuardResult:
    """A frozen index must be in the order the loader actually emits, compared by name.

    The loader walks its pattern groups in a fixed order -- positives before
    negatives -- and sorts within a group by the pattern's numeric suffix. A
    lexicon index sorted by metacluster ascending puts negatives first, so every
    positional read against it is off by the size of the negative set.

    In the reference implementation this was invisible: one model had no negative
    motifs at all, so the two orders coincided and the positional read looked
    correct. It would have silently mislabelled every hit for the other model.

    The comparison is by name and never by position, because comparing positions is
    the very assumption under test.
    """
    gid = "index_order_matches_loader"
    index_names, loader_names = list(index_names), list(loader_names)
    if not index_names or not loader_names:
        return _fail(gid, f"nothing to compare: {len(index_names)} index vs "
                          f"{len(loader_names)} loader names")
    for label, names in (("index", index_names), ("loader", loader_names)):
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            return _fail(gid, f"{label} order contains duplicate names: {dupes}")
    missing = [n for n in index_names if n not in set(loader_names)]
    extra = [n for n in loader_names if n not in set(index_names)]
    if missing or extra:
        return _fail(gid, f"index and loader disagree on membership: "
                          f"only-in-index={missing[:5]}, only-in-loader={extra[:5]}")
    if index_names != loader_names:
        first = next(i for i, (a, b) in enumerate(zip(index_names, loader_names, strict=True))
                     if a != b)
        return _fail(
            gid,
            f"same {len(index_names)} motifs, different order: position {first} is "
            f"{index_names[first]!r} in the index and {loader_names[first]!r} from the loader. "
            "Any positional read against this index is wrong."
        )
    return _ok(gid, f"{len(index_names)} motifs, index order matches the loader by name")


def no_cross_estimand_pooling(summaries: Iterable[Mapping[str, Any]],
                              specifications: Mapping[str, Mapping[str, Any]]) -> GuardResult:
    """A stability summary may not span two estimands.

    The failure this refuses is the one a specification multiverse is most likely
    to commit, and it does not look like an error while you are committing it: run
    the same contrast against two baseline populations, then report "the effect is
    stable across specifications". That number averages an answer to one question
    with an answer to another. Where the baseline is what moved, the finding is
    that the conclusion is *baseline-sensitive*, and a single score conceals
    exactly the thing worth reporting.

    Two arguments, because one would make this a guard over its own producer. The
    summary says which cells it covers; ``specifications`` is the manifest,
    written before the run by the code that enumerated the grid, and the estimand
    of each cell is read from **there**. So a summariser that groups wrongly
    cannot also tell the guard it grouped rightly -- and a summariser that lies
    about its members is caught by the second half, which refuses a cell the
    manifest does not contain rather than skipping it.
    """
    gid = "no_cross_estimand_pooling"
    n_groups = 0
    for summary in summaries:
        n_groups += 1
        key = summary.get("group_key", "<unnamed group>")
        members = list(summary.get("cell_ids", ()))
        if not members:
            return _fail(gid, f"{key}: a summary that names no cells cannot be checked")
        estimands = set()
        for cell_id in members:
            spec = specifications.get(cell_id)
            if spec is None:
                return _fail(
                    gid,
                    f"{key}: cell {cell_id} is not in the specification manifest, so the "
                    "estimand it belongs to cannot be established"
                )
            estimands.add(spec.get("estimand_id"))
        # A summary also states which estimand it is FOR, and that label is what a
        # reader joins on. One estimand among the members is not enough if the
        # label names a different one: the rows would be within-estimand and filed
        # under the wrong question, which reads exactly like the pooling this
        # refuses and is not caught by the count alone.
        claimed = summary.get("estimand_id")
        if claimed is not None and estimands and claimed not in estimands:
            return _fail(
                gid,
                f"{key}: filed under estimand {claimed}, but its cells belong to "
                f"{', '.join(sorted(map(str, estimands)))} according to the manifest"
            )
        if len(estimands) > 1:
            return _fail(
                gid,
                f"{key}: pools {len(estimands)} estimands ({', '.join(sorted(map(str, estimands)))}); "
                "an effect against one baseline population and an effect against another are "
                "answers to different questions and do not average"
            )
    return _ok(gid, f"each of {n_groups} summaries stays within one estimand")


ALL_GUARDS = {
    "single_scale": single_scale,
    "no_cross_estimand_pooling": no_cross_estimand_pooling,
    "variant_id_unique": variant_id_unique,
    "no_key_parsing": no_key_parsing,
    "four_state_missingness": four_state_missingness,
    "no_cross_model_cwm_avg": no_cross_model_cwm_avg,
    "sign_alignment": sign_alignment,
    "interaction_required": interaction_required,
    "estimability_floor": estimability_floor,
    "stratum_parity": stratum_parity,
    "short_motif_flag": short_motif_flag,
    "single_family_layer": single_family_layer,
    "selection_provenance_declared": selection_provenance_declared,
    "health_before_effect": health_before_effect,
    "comparator_declared": comparator_declared,
    "index_order_matches_loader": index_order_matches_loader,
}


def _required_parameters(fn: Any) -> list[str]:
    """The names of ``fn``'s parameters that have no default.

    Derived from the signature rather than listed here, so a guard that grows or
    loses an argument cannot leave a hand-maintained list behind (the same reason
    ``ALL_GUARDS`` is walked instead of enumerated in prose).
    """
    import inspect

    return [
        name for name, parameter in inspect.signature(fn).parameters.items()
        if parameter.default is inspect.Parameter.empty
    ]


def run_all(inputs: Mapping[str, Any]) -> list[GuardResult]:
    """Run each guard for which ``inputs`` supplies its argument(s).

    Most guards take one argument -- the rows, nodes or claims they check -- and
    ``inputs[guard_id]`` is that value. Two do not: ``four_state_missingness``
    needs the claimed coverage/defined/total it must recompute against, and
    ``index_order_matches_loader`` needs both orders it compares. Handing them a
    single positional value raised ``TypeError`` straight out of the "run every
    guard" entry point, so the one call that promises to run everything could not
    run two of the fifteen -- including the guard for this project's founding
    failure. For those guards ``inputs[guard_id]`` is a Mapping of keyword
    arguments instead.

    The shape is decided from the signature, never sniffed from the value: a
    Mapping is already the legitimate single argument of
    ``health_before_effect``, and guessing would silently splat a report into
    keyword arguments. A guard whose arguments arrive in the wrong shape raises
    ``TypeError`` naming what it wanted -- it does not return a failed
    ``GuardResult``, because "the caller passed the wrong thing" and "the data
    violated the rule" are different claims and a report cannot tell them apart
    once they look alike.
    """
    results = []
    for gid, fn in ALL_GUARDS.items():
        if gid not in inputs:
            continue
        value = inputs[gid]
        required = _required_parameters(fn)
        if len(required) == 1:
            results.append(fn(value))
        elif isinstance(value, Mapping):
            results.append(fn(**value))
        else:
            raise TypeError(
                f"{gid} takes {len(required)} arguments ({', '.join(required)}), so "
                f"inputs[{gid!r}] must be a mapping of keyword arguments, not "
                f"{type(value).__name__}"
            )
    return results
