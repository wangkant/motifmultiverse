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

__all__ = [
    "GuardResult", "GuardError", "ALL_GUARDS", "run_all",
    "single_scale", "variant_id_unique", "no_key_parsing", "four_state_missingness",
    "no_cross_model_cwm_avg", "sign_alignment", "interaction_required",
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


def four_state_missingness(
    rows: Sequence[Mapping[str, Any]],
    claimed_coverage: float,
    claimed_defined: int,
    claimed_total: int,
    value_key: str = "statistic",
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
        return math.isclose(claimed, recomputed, rel_tol=1e-9, abs_tol=1e-9)

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


def no_cross_model_cwm_avg(operations: Iterable[Mapping[str, Any]]) -> GuardResult:
    """Averaging CWMs across model / readout / metacluster is a design prohibition."""
    gid = "no_cross_model_cwm_avg"
    for op in operations:
        if op.get("op") not in {"mean", "average"}:
            continue
        grouped = set(op.get("group_by", ()))
        for axis in ("model", "readout", "metacluster"):
            if axis not in grouped:
                return _fail(gid, f"CWM {op['op']} does not hold {axis} fixed (group_by={sorted(grouped)})")
    return _ok(gid, "no CWM averaging crosses model, readout or metacluster")


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
    """Short / weakly-supported motifs must carry ``low_confidence_annotation``."""
    gid = "short_motif_flag"
    def g(n: Any, k: str) -> Any:
        return getattr(n, k) if hasattr(n, k) else n.get(k)
    offenders = []
    for n in nodes:
        short = (g(n, "motif_length") or 99) <= 6
        weak_q = (g(n, "annotation_matches") or {}).get("tomtom_q", 0.0) > 0.05
        few = (g(n, "seqlet_count") or 10**9) < 100
        if (short or weak_q or few) and not g(n, "low_confidence_annotation"):
            offenders.append(g(n, "variant_id"))
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


ALL_GUARDS = {
    "single_scale": single_scale,
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


def run_all(inputs: Mapping[str, Any]) -> list[GuardResult]:
    """Run each guard for which ``inputs`` supplies an argument."""
    results = []
    for gid, fn in ALL_GUARDS.items():
        if gid in inputs:
            results.append(fn(inputs[gid]))
    return results
