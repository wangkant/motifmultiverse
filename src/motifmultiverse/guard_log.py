"""What the executable constraints returned, written where the results are written.

A tool whose thesis is that decisions carry their evidence did not record the
outcome of its own executable constraints. ``report`` discovered that and had to
print it in "what this report does not know": no artifact persisted a
:class:`~motifmultiverse.guards.GuardResult`, so the strongest thing a report
could say about a guard was which guards the *code path* calls -- a fact about
the source, not about the run in front of the reader. A guard that ran and passed
and a guard that was never reached produced identical directories.

**Why this is its own artifact and not a field of the provenance record.**
``provenance.json`` is an append-log written *before* the body runs (T-09), and
that ordering is the whole point of it: a record that arrives only on success is a
record the runs you most want to explain never get. A guard outcome does not exist
until after the body has run, so putting it in the provenance record would mean
either rewriting an already-appended record -- destroying the append-only property
that makes the log evidence -- or moving provenance to the end, which is the
failure T-09 exists to prevent. The decisive reason is the third one: **a failing
guard raises**, so the run that most needs its outcome recorded is exactly the run
that produces no result artifact at all. An outcome stored inside
``interpretation.json`` or ``stability_results.parquet`` would be missing from
precisely the refusals it was written for. So the shape is ``validate``'s
``backend_verification.tsv``: a separate, versioned file recording what a check
returned, sitting beside the results rather than inside them, and written *as each
outcome is produced* so that a refusal leaves it behind.

**What a record here is and is not.** Each entry is what the guard function
returned -- its id, its pass/fail, and its own ``detail`` string, copied
verbatim. Nothing here re-derives a verdict, recounts an input, or summarises a
run: this module runs no guard and second-guesses none, because a log that
recomputes is a second implementation of the check and free to disagree with it.
``subject`` is a caller-supplied sentence naming *what* was handed to the guard,
and it is prose on purpose -- a structured "n rows" field would be a count this
module computed beside a count the guard's own detail already states, and two
counts that can disagree are worse than one.

**What it cannot say.** A guard absent from this file is *not* recorded as
not-having-run. Guard calls made outside a stage that binds a log --
``compile.probe_backend``, which checks the installed loader against a synthetic
one-motif lexicon on behalf of ``status``, belongs to the environment and not to
any output directory -- leave no entry here. Readers (``report`` among them) must
treat the file as a positive record of the calls it lists and infer nothing from a
name's absence; that inversion is the failure ``report``'s module docstring is
built against.

**Joining an entry to an artifact.** Several runs legitimately write one ``--out``.
Each entry records ``provenance_records``: how many records ``provenance.json``
held when the outcome was written. Since every subcommand appends its provenance
record before it computes, that number identifies the run whose record is the last
one -- the same join ``run_status.json`` uses, and the reason
``run_status.artifacts_are_from.provenance_records`` can tell a reader which run
the artifacts lying in a directory belong to.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from motifmultiverse.guards import GuardResult
from motifmultiverse.schema import SchemaError

__all__ = [
    "GUARD_OUTCOMES_FILENAME", "GUARD_OUTCOMES_SCHEMA_VERSION", "NOT_RECORDED",
    "QUARANTINE_FILENAME", "GuardLogError", "GuardOutcome", "GuardLog",
    "read_guard_outcomes",
]

#: The artifact this module writes, beside ``provenance.json`` in the run's ``--out``.
GUARD_OUTCOMES_FILENAME = "guard_outcomes.json"

#: Artifacts here carry schema versions; so does this one.
GUARD_OUTCOMES_SCHEMA_VERSION = "1"

#: What ``provenance_records`` says when there is no readable provenance log to
#: count. A literal token rather than ``null`` or ``0``: ``0`` is a legitimate
#: length and would silently join to nothing, and ``null`` reads as "no run".
NOT_RECORDED = "NOT_RECORDED"

#: Where outcomes go when ``guard_outcomes.json`` is already there and unreadable.
#: The existing file is never overwritten -- it may hold every earlier run's
#: outcomes -- but the alternative to overwriting is not discarding the run: this
#: module is bookkeeping, and bookkeeping does not get to veto a computation that
#: has already happened and whose guards have already passed. So the outcomes go
#: to a sibling, and the run says where.
QUARANTINE_FILENAME = "guard_outcomes.unreadable-predecessor.json"


class GuardLogError(SchemaError):
    """A guard-outcome log cannot be written or read without losing what it holds.

    A subclass of ``SchemaError`` so ``cli.main`` already maps it to exit **4** --
    a refusal, named -- rather than letting a ``JSONDecodeError`` escape from
    inside an append as an undocumented exit 1.
    """


@dataclass(frozen=True)
class GuardOutcome:
    """One guard call: which guard, on what, and what it returned.

    ``detail`` is the guard's own sentence, copied and never rewritten. ``passed``
    is the guard's own boolean. ``subject`` is the caller's description of the
    input; it is the only field this module does not take from the
    :class:`~motifmultiverse.guards.GuardResult`, and it describes the *call*, not
    the data's correctness.
    """

    stage: str
    guard_id: str
    passed: bool
    detail: str
    subject: str
    recorded_utc: str
    provenance_records: int | str
    schema_version: str = GUARD_OUTCOMES_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _provenance_records(out_dir: Path) -> int | str:
    """How many records the provenance log in ``out_dir`` holds right now.

    Unreadable or absent is reported as :data:`NOT_RECORDED` and never raised: a
    guard outcome must not be lost because the log beside it could not be counted.
    What is lost in that case is the join, and the token says so.
    """
    try:
        records = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return NOT_RECORDED
    return len(records) if isinstance(records, list) else NOT_RECORDED


class GuardLog:
    """Collects guard outcomes for one stage, and appends each one as it happens.

    Bound to an output directory, every outcome is appended to
    ``guard_outcomes.json`` at the moment the guard returns -- *before* the caller
    gets the chance to raise on it. That ordering is not incidental: guards in this
    package are invoked as ``log.record(...).raise_if_failed()``, so a failing
    guard's outcome is already on disk when the refusal propagates and the run
    ends with no result artifact.

    Unbound (``out_dir=None``) it still collects into :attr:`outcomes`, so a
    library caller that never names a directory -- and every test in this suite --
    can read what ran without a filesystem.
    """

    def __init__(self, stage: str, out_dir: str | os.PathLike[str] | None = None) -> None:
        self.stage = stage
        self.out_dir = None if out_dir is None else Path(out_dir)
        self.outcomes: list[GuardOutcome] = []
        #: Set when the directory's log could not be appended to and this run's
        #: outcomes went to :data:`QUARANTINE_FILENAME` instead. A sentence for a
        #: human, or ``None``. Callers surface it; nothing in this package reads it
        #: to make a decision, because a bookkeeping failure is not a finding.
        self.degraded: str | None = None

    def record(self, result: GuardResult, *, subject: str) -> GuardResult:
        """Record ``result``, then hand it back unchanged.

        Returning the same object is what lets a call site read
        ``log.record(guards.x(rows), subject=...).raise_if_failed()``: the guard
        still decides, this module only writes down what it decided. It must not
        wrap, downgrade or swallow a failure -- a logger that turns a refusal into
        a log line is the disclaimer-instead-of-a-control failure in a new place.
        """
        if not isinstance(result, GuardResult):
            raise GuardLogError(
                f"{self.stage}: a guard outcome must be a GuardResult, not "
                f"{type(result).__name__}; this log records what a guard returned and "
                "cannot invent one"
            )
        outcome = GuardOutcome(
            stage=self.stage,
            guard_id=result.guard_id,
            passed=result.passed,
            detail=result.detail,
            subject=subject,
            recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            provenance_records=(NOT_RECORDED if self.out_dir is None
                                else _provenance_records(self.out_dir)),
        )
        self.outcomes.append(outcome)
        if self.out_dir is not None:
            self._append(outcome)
        return result

    def _append(self, outcome: GuardOutcome) -> None:
        """Append one outcome to the directory's log without ever destroying it.

        The same two properties ``provenance.ProvenanceRecord.write`` needs, for
        the same reasons. Atomic: the replacement is staged in a per-process
        sibling and moved with ``os.replace``, so a run killed mid-write leaves the
        old log rather than half a JSON document. And an unreadable log is not an
        empty one -- overwriting it would discard the recorded outcomes of every
        earlier run in the directory, so this never writes over it.

        What it does *instead* is the part worth stating, because the first version
        of this module got it wrong in a way that only showed up on real data: it
        raised. Since the raise reaches ``cli.main`` as a refusal, a corrupt
        bookkeeping file left in ``--out`` by something else discarded a completed
        interpretation -- every effect computed, every guard passed, and no
        ``interpretation.json`` written, because a JSON file this module owns could
        not be parsed. That inverts the module's own thesis: the record exists to
        say what happened, not to decide whether it may. So the outcomes go to
        :data:`QUARANTINE_FILENAME` beside it, :attr:`degraded` says so in a
        sentence, and the science survives its bookkeeping.
        """
        out = self.out_dir
        assert out is not None                       # only called when bound
        out.mkdir(parents=True, exist_ok=True)
        dest = out / GUARD_OUTCOMES_FILENAME
        try:
            existing = _read_log(dest, allow_absent=True)
        except GuardLogError as exc:
            dest = out / QUARANTINE_FILENAME
            if self.degraded is None:
                self.degraded = (
                    f"{GUARD_OUTCOMES_FILENAME} in this directory could not be read "
                    f"({exc.__cause__ or exc}); it has been left untouched and this "
                    f"run's guard outcomes were written to {QUARANTINE_FILENAME} "
                    "instead. The outcomes are complete; what is lost is their "
                    "place in the directory's single log."
                )
            # A quarantine file that is itself unreadable is not recoverable by
            # moving aside again, and silently starting a third file would be the
            # overwriting this method exists to prevent. That one still raises.
            existing = _read_log(dest, allow_absent=True)
        existing.append(outcome.to_dict())
        staged = dest.with_name(f".{dest.name}.{os.getpid()}.partial")
        try:
            staged.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(staged, dest)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise


def _read_log(dest: Path, *, allow_absent: bool) -> list[dict[str, Any]]:
    if not dest.exists():
        if allow_absent:
            return []
        raise FileNotFoundError(dest)
    try:
        payload = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise GuardLogError(
            f"{dest} is not a readable guard-outcome log ({exc}) and will not be "
            "overwritten; move it aside to keep the earlier outcomes, or delete it to "
            "start a new log."
        ) from exc
    if not isinstance(payload, list):
        raise GuardLogError(
            f"{dest} is not a guard-outcome log -- a log is a list of outcomes, this is "
            f"{type(payload).__name__} -- and will not be overwritten."
        )
    bad = [i for i, row in enumerate(payload) if not isinstance(row, dict)]
    if bad:
        raise GuardLogError(
            f"{dest} holds {len(bad)} entr{'y' if len(bad) == 1 else 'ies'} that are not "
            f"outcome records (first at index {bad[0]}); refusing to read a log whose "
            "entries cannot be told apart."
        )
    return payload


def read_guard_outcomes(out_dir: str | os.PathLike[str]) -> list[dict[str, Any]] | None:
    """The outcomes recorded in ``out_dir``, or ``None`` if there is no such file.

    Three states, kept apart on purpose, because a reader that collapses them
    reports the founding failure's shape:

    * **absent** -- ``None``. Every artifact produced before this module existed is
      in this state, and it means *nothing was recorded*, never *nothing ran*.
    * **present and readable** -- the list, verbatim.
    * **present and unreadable** -- :class:`GuardLogError`. Reporting a corrupt log
      as an absent one would let a consumer say "no outcomes were recorded here"
      about a directory that recorded some.
    """
    dest = Path(out_dir) / GUARD_OUTCOMES_FILENAME
    if not dest.exists():
        return None
    return _read_log(dest, allow_absent=False)
