"""What a run records about the guards it ran, and what it must never record.

`guards` is the package's executable half; until this module existed, running one
left no trace. The property that matters most is the least obvious: a guard that
FAILS raises, so the run whose outcome a reader most needs is the run that
produces no result artifact at all. Every test here that is about ordering is
about that.

The second property is negative and is asserted as hard as the first. This log is
a record of calls, not a second implementation of any check: it must copy the
guard's own boolean and the guard's own sentence, and it must never turn a
failure into a log line -- a logger that swallows a refusal is the
disclaimer-instead-of-a-control failure wearing a new hat.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from motifmultiverse import guard_log, guards


def _outcomes(directory: Path) -> list[dict]:
    return json.loads((directory / guard_log.GUARD_OUTCOMES_FILENAME).read_text())


# --------------------------------------------------------------------------- #
# 1. The failing run is the one that must leave a record.
# --------------------------------------------------------------------------- #
def test_a_failing_guards_outcome_is_on_disk_before_its_refusal_escapes(tmp_path):
    """The whole reason this is a file and not a field of the result.

    `raise_if_failed` is how every call site in this package uses a guard, so a
    guard that says no ends the run and no result artifact is written. If the
    outcome travelled inside that result it would be missing from exactly the
    runs it was written for.
    """
    log = guard_log.GuardLog("interpret", tmp_path)
    failing = guards.single_scale([{"input_scale": 1}, {"input_scale": 2}])
    assert not failing.passed, "the fixture must actually fail or this proves nothing"

    with pytest.raises(guards.GuardError):
        log.record(failing, subject="two hit tables' worth of rows").raise_if_failed()

    recorded = _outcomes(tmp_path)
    assert [row["guard_id"] for row in recorded] == ["single_scale"]
    assert recorded[0]["passed"] is False
    assert recorded[0]["detail"] == failing.detail
    assert recorded[0]["subject"] == "two hit tables' worth of rows"


def test_record_hands_back_the_same_verdict_and_never_downgrades_it(tmp_path):
    """A log that softened a failure would be worse than no log."""
    log = guard_log.GuardLog("align", tmp_path)
    failing = guards.sign_alignment([{"registered_on": "signed_cwm"}])
    returned = log.record(failing, subject="one edge")

    assert returned is failing
    assert returned.passed is False
    with pytest.raises(guards.GuardError):
        returned.raise_if_failed()


def test_the_recorded_verdict_is_the_guards_own_and_is_not_recomputed(tmp_path):
    """Copied, not re-derived. A log that recomputes can disagree with the guard."""
    log = guard_log.GuardLog("annotate", tmp_path)
    for result in (
        guards.short_motif_flag([{"variant_id": "v1", "motif_length": 12,
                                  "seqlet_count": 400, "annotation_matches": {},
                                  "low_confidence_annotation": False}]),
        guards.short_motif_flag([{"variant_id": "v2", "motif_length": 4,
                                  "seqlet_count": 400, "annotation_matches": {},
                                  "low_confidence_annotation": False}]),
    ):
        log.record(result, subject="candidates")

    recorded = _outcomes(tmp_path)
    assert [row["passed"] for row in recorded] == [True, False]
    assert [row["detail"] for row in recorded] == [
        "every short / weak / low-support motif is flagged",
        "unflagged low-confidence annotations: ['v2']",
    ]


# --------------------------------------------------------------------------- #
# 2. The log is an append-log, like the provenance record beside it.
# --------------------------------------------------------------------------- #
def test_a_second_run_appends_and_never_destroys_the_first(tmp_path):
    guard_log.GuardLog("interpret", tmp_path).record(
        guards.single_scale([{"input_scale": 7}]), subject="first run")
    guard_log.GuardLog("infer", tmp_path).record(
        guards.single_scale([{"input_scale": 7}]), subject="second run")

    recorded = _outcomes(tmp_path)
    assert [row["stage"] for row in recorded] == ["interpret", "infer"]
    assert [row["subject"] for row in recorded] == ["first run", "second run"]


def test_an_unreadable_log_is_quarantined_rather_than_overwritten_or_raised(tmp_path):
    """Not overwritten, and not fatal either -- both halves matter.

    Overwriting would discard every earlier run's outcomes silently. But the first
    version of this module took the other branch and *raised*, and because
    `GuardLogError` is a `SchemaError` the refusal reached `cli.main` as exit 4:
    a corrupt bookkeeping file left in `--out` by something else discarded a
    completed interpretation, with every effect computed and every guard passed.
    A record of what happened does not get to decide whether it may happen.
    """
    dest = tmp_path / guard_log.GUARD_OUTCOMES_FILENAME
    dest.write_text('[{"guard_id": "single_scale"', encoding="utf-8")   # truncated

    log = guard_log.GuardLog("interpret", tmp_path)
    result = log.record(guards.single_scale([{"input_scale": 7}]), subject="rows")

    assert result.passed                                     # the guard still decides
    assert dest.read_text(encoding="utf-8") == '[{"guard_id": "single_scale"'
    quarantined = json.loads(
        (tmp_path / guard_log.QUARANTINE_FILENAME).read_text(encoding="utf-8"))
    assert [row["guard_id"] for row in quarantined] == ["single_scale"]
    assert log.degraded and guard_log.QUARANTINE_FILENAME in log.degraded


def test_a_quarantined_log_still_accumulates_every_outcome_of_the_run(tmp_path):
    """The outcomes are complete; only their place in the directory's log is lost."""
    (tmp_path / guard_log.GUARD_OUTCOMES_FILENAME).write_text("{", encoding="utf-8")

    log = guard_log.GuardLog("interpret", tmp_path)
    log.record(guards.single_scale([{"input_scale": 7}]), subject="first")
    log.record(guards.single_scale([{"input_scale": 9}]), subject="second")

    quarantined = json.loads(
        (tmp_path / guard_log.QUARANTINE_FILENAME).read_text(encoding="utf-8"))
    assert [row["subject"] for row in quarantined] == ["first", "second"]
    assert [outcome.subject for outcome in log.outcomes] == ["first", "second"]


def test_a_log_that_is_not_a_list_is_refused(tmp_path):
    (tmp_path / guard_log.GUARD_OUTCOMES_FILENAME).write_text('{"guard_id": "x"}')
    with pytest.raises(guard_log.GuardLogError):
        guard_log.read_guard_outcomes(tmp_path)


def test_absent_empty_and_unreadable_are_three_different_answers(tmp_path):
    """A reader that collapses them reports an absence as a result."""
    absent = tmp_path / "absent"
    absent.mkdir()
    assert guard_log.read_guard_outcomes(absent) is None

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / guard_log.GUARD_OUTCOMES_FILENAME).write_text("[]")
    assert guard_log.read_guard_outcomes(empty) == []

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / guard_log.GUARD_OUTCOMES_FILENAME).write_text("not json")
    with pytest.raises(guard_log.GuardLogError):
        guard_log.read_guard_outcomes(broken)


def test_an_unbound_log_writes_nothing_and_still_collects(tmp_path):
    """A library caller who named no directory is not refused for it."""
    log = guard_log.GuardLog("interpret")
    log.record(guards.single_scale([{"input_scale": 3}]), subject="rows")

    assert [o.guard_id for o in log.outcomes] == ["single_scale"]
    assert not list(tmp_path.iterdir())
    assert log.outcomes[0].provenance_records == guard_log.NOT_RECORDED


def test_an_entry_joins_to_the_run_whose_provenance_record_it_was_written_under(tmp_path):
    """The join `run_status.json` already uses: the length of the provenance log.

    Every subcommand appends its provenance record BEFORE it computes, so the
    number of records at the moment a guard returns identifies the run whose
    record is the last one.
    """
    (tmp_path / "provenance.json").write_text(json.dumps([{"subcommand": "interpret"}]))
    guard_log.GuardLog("interpret", tmp_path).record(
        guards.single_scale([{"input_scale": 7}]), subject="first run")

    (tmp_path / "provenance.json").write_text(
        json.dumps([{"subcommand": "interpret"}, {"subcommand": "interpret"}]))
    guard_log.GuardLog("interpret", tmp_path).record(
        guards.single_scale([{"input_scale": 7}]), subject="second run")

    assert [row["provenance_records"] for row in _outcomes(tmp_path)] == [1, 2]


def test_a_missing_provenance_log_costs_the_join_and_says_so(tmp_path):
    """`NOT_RECORDED`, never 0: 0 is a legitimate length and would join to nothing."""
    guard_log.GuardLog("interpret", tmp_path).record(
        guards.single_scale([{"input_scale": 7}]), subject="rows")
    assert _outcomes(tmp_path)[0]["provenance_records"] == guard_log.NOT_RECORDED


def test_the_log_refuses_something_that_is_not_a_guard_result(tmp_path):
    """It records what a guard returned; it cannot invent one."""
    with pytest.raises(guard_log.GuardLogError):
        guard_log.GuardLog("interpret", tmp_path).record(
            {"guard_id": "single_scale", "passed": True}, subject="rows")   # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 3. Every stage that calls a guard records what it returned.
# --------------------------------------------------------------------------- #
#: The one guard call in `src/` that deliberately records nothing, and why. It is
#: an allowlist rather than a blanket exemption so that a future unrecorded call
#: has to be argued for here instead of appearing quietly.
#: Keyed by (file, ENCLOSING FUNCTION, guard). The function is part of the key
#: because a file-level key exempts every call in the file, including ones written
#: later by someone who never read this reason -- `compile/__init__.py` holds two
#: calls of `index_order_matches_loader` and only one of them is the one argued
#: for here.
UNRECORDED_GUARD_CALLS = {
    ("compile/__init__.py", "probe_backend", "index_order_matches_loader"): (
        "compile.probe_backend checks the INSTALLED LOADER against a synthetic "
        "one-motif lexicon in a temporary directory, on behalf of `status`. It "
        "belongs to the environment and to no run's output directory, so there is "
        "no artifact for its outcome to sit beside."
    ),
}


def _enclosing_function(tree):
    """Every node mapped to the name of the innermost function containing it."""
    owner = {}

    def visit(node, fn):
        for child in ast.iter_child_nodes(node):
            owner[child] = fn
            visit(child, child.name
                  if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn)

    visit(tree, None)
    return owner


def _returns_a_guard_result(node):
    annotation = getattr(node, "returns", None)
    return (isinstance(annotation, ast.Name) and annotation.id == "GuardResult") or (
        isinstance(annotation, ast.Attribute) and annotation.attr == "GuardResult")


def test_every_guard_call_in_src_records_what_the_guard_returned():
    """A guard that ran and left no trace is the gap this module was written to close.

    Structural rather than textual: the AST is walked and each guard call's parent
    must be a `.record(...)` call. A test that grepped for the word `guard_log`
    near a call site would pass on a comment.

    "Guard call" is deliberately wider than `guards.<name>(...)`. A function
    annotated `-> guards.GuardResult` is one too -- `compile.verify_roundtrip` is
    the case that motivated this -- and it is the shape most likely to escape a
    narrower rule, because the guard invocation inside it looks recorded to nobody
    and its own call sites do not mention `guards` at all. Such a function is
    allowed to call a guard unrecorded *when it hands the result straight back*,
    since recording then belongs to whoever receives it; and every call to it must
    itself be recorded.
    """
    import motifmultiverse

    src = Path(motifmultiverse.__file__).resolve().parent
    files = [p for p in sorted(src.rglob("*.py"))
             if p.parts[-2:] != ("guards", "__init__.py") and p.name != "guard_log.py"]
    trees = {p: ast.parse(p.read_text(encoding="utf-8")) for p in files}
    guard_returning = {
        node.name
        for tree in trees.values() for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _returns_a_guard_result(node)
    }

    offenders = []
    for path, tree in trees.items():
        relative = str(path.relative_to(src))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        owner = _enclosing_function(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (isinstance(node.func, ast.Attribute)
                    and getattr(node.func.value, "id", None) == "guards"
                    and node.func.attr in guards.ALL_GUARDS):
                called = node.func.attr
            elif isinstance(node.func, ast.Name) and node.func.id in guard_returning:
                called = node.func.id
            else:
                continue
            enclosing = owner.get(node)
            if (relative, enclosing, called) in UNRECORDED_GUARD_CALLS:
                continue
            parent = parents.get(node)
            # Handed straight back out of a guard-returning function: the caller
            # records it, and the call site of THAT is checked in this same walk.
            if isinstance(parent, ast.Return) and enclosing in guard_returning:
                continue
            recorded = (isinstance(parent, ast.Call)
                        and isinstance(parent.func, ast.Attribute)
                        and parent.func.attr == "record")
            if not recorded:
                offenders.append(f"{relative}:{node.lineno}: {called} (in {enclosing})")
    assert not offenders, (
        "guard calls whose outcome is written down nowhere: " + "; ".join(offenders)
    )


def test_the_unrecorded_call_allowlist_is_not_a_dead_letter():
    """Each exemption must still name a real call, or it is protecting nothing."""
    import motifmultiverse

    src = Path(motifmultiverse.__file__).resolve().parent
    for (relative, function, guard_id), reason in UNRECORDED_GUARD_CALLS.items():
        tree = ast.parse((src / relative).read_text(encoding="utf-8"))
        owner = _enclosing_function(tree)
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and getattr(node.func.value, "id", None) == "guards"
                 and node.func.attr == guard_id and owner.get(node) == function]
        assert calls, (
            f"{relative}:{function} no longer calls guards.{guard_id}; the exemption "
            "is stale and now silently covers nothing"
        )
        assert len(reason) > 40, f"{relative}/{function}: an exemption needs a reason"
