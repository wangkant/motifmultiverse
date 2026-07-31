"""What happened to the run that last wrote this output directory.

The failure this closes was documented in ``cli``'s own docstring as a known
limitation, which is a good place to record a defect and a bad place to leave
one. Provenance is written *before* the body runs, deliberately: a record that
arrives only on success is a record the runs you most want to explain never get.
So a refused run appended its provenance record and produced nothing else -- and
if an earlier run had succeeded into the same ``--out``, the directory was left
holding that earlier result, a refusal record, and nothing relating the two. A
reader who looks at the directory reads the old result as this run's, and a
pipeline that globs for ``interpretation.json`` cannot tell the difference at
all. "Read the exit code, not the directory" is advice, and advice is not a
mechanism: the exit code is gone by the time anyone opens the folder.

``run_status.json`` is that mechanism, and it is a *statement about runs*, not a
staleness marker on the artifacts:

* it records the outcome of the run that most recently wrote here --
  ``SUCCESS`` / ``REFUSED`` / ``UNIMPLEMENTED`` / ``INPUT_MISSING`` / ``CRASHED``
  -- with the exit code the process returned and, for a refusal, the sentence
  that refused;
* ``artifacts_are_from`` carries the *last successful* run forward across
  subsequent failures, so a directory whose status is ``REFUSED`` still says
  which earlier run wrote what is lying in it, and says
  ``NO_SUCCESSFUL_RUN_RECORDED`` when no run this file has seen ever succeeded.

Nothing is deleted and nothing is renamed. Destroying an earlier real result to
prevent a misreading of it would trade a misreading for a data loss; what was
missing was never the removal, it was the sentence saying which run the files
belong to.

A downstream reader's contract is one line: **trust the artifacts only when
``status == "SUCCESS"``**, and otherwise read ``artifacts_are_from`` to find out
whose they are. ``provenance.json`` remains the append-log of every invocation;
this file is the outcome of the latest one, and the two join on
``provenance_records`` -- the number of records in the log when this run
finished, so this run's own record is the last one.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

__all__ = [
    "RUN_STATUS_FILENAME", "SCHEMA_VERSION", "STATUSES", "NO_SUCCESSFUL_RUN",
    "write_run_status", "read_run_status",
]

RUN_STATUS_FILENAME = "run_status.json"
SCHEMA_VERSION = "1"

#: Every outcome a run can end in, and the exit code that goes with it. Kept as
#: data so this file cannot claim an outcome the CLI never returns.
STATUSES = {
    "SUCCESS": 0,
    "INPUT_MISSING": 2,
    "UNIMPLEMENTED": 3,
    "REFUSED": 4,
    "CRASHED": 1,
}

#: What ``artifacts_are_from`` says when no successful run has ever been recorded
#: in this directory. A literal token, never ``null``: ``null`` beside a field
#: named "artifacts are from" reads as "from nowhere in particular", and this
#: says the stronger and true thing -- that nothing here was written by a run
#: this file has seen succeed.
NO_SUCCESSFUL_RUN = "NO_SUCCESSFUL_RUN_RECORDED"


def read_run_status(out_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The status document in ``out_dir``, or ``None`` if there is not a readable one.

    Unreadable is reported as absent *to this module only*, and never propagates:
    a corrupt status file must not stop the current run from recording its own
    outcome. What it does cost is the carry-forward -- see
    :func:`write_run_status`, which then says so rather than inventing a
    predecessor.
    """
    path = Path(out_dir) / RUN_STATUS_FILENAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def write_run_status(out_dir: str | os.PathLike[str], *, status: str, command: str,
                     subcommand: str, detail: str | None = None,
                     exit_code: int | None = None) -> Path:
    """Record how this run ended, and which run the artifacts here came from.

    Written last, when the outcome is known -- the opposite of the provenance
    record, and for the opposite reason. Provenance answers "what was attempted",
    so it must survive a run that dies; this answers "may the files here be
    read", which is only knowable at the end.

    Writing is atomic (staged sibling plus ``os.replace``) because a run
    interrupted mid-write would otherwise leave a status file that says nothing,
    in a directory whose whole problem was that nothing said anything.
    """
    if status not in STATUSES:
        raise ValueError(f"unknown run status {status!r}; expected one of {sorted(STATUSES)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    previous = read_run_status(out)
    n_records: int | str = "NOT_RECORDED"
    try:
        records = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        if isinstance(records, list):
            n_records = len(records)
    except (OSError, UnicodeDecodeError, ValueError):
        pass

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": STATUSES[status] if exit_code is None else exit_code,
        "subcommand": subcommand,
        "command": command,
        "finished_utc": finished,
        # The refusal sentence itself, so the directory carries the reason and not
        # only the fact. `null` for a run that did not refuse -- an empty string
        # would read as a refusal with nothing to say for itself.
        "detail": detail,
        "provenance_records": n_records,
    }
    if status == "SUCCESS":
        document["artifacts_are_from"] = {
            "status": "SUCCESS", "command": command, "subcommand": subcommand,
            "finished_utc": finished, "provenance_records": n_records,
        }
    elif previous is not None and "artifacts_are_from" in previous:
        # Carried forward verbatim: this run wrote no artifacts, so whatever is in
        # the directory still belongs to whoever the last file said it belonged to.
        document["artifacts_are_from"] = previous["artifacts_are_from"]
    elif previous is None and (out / RUN_STATUS_FILENAME).exists():
        # There was a status file and it could not be read. Saying
        # NO_SUCCESSFUL_RUN_RECORDED here would assert something this run does not
        # know; the honest value names the loss.
        document["artifacts_are_from"] = "PREVIOUS_RUN_STATUS_UNREADABLE"
    else:
        document["artifacts_are_from"] = NO_SUCCESSFUL_RUN

    dest = out / RUN_STATUS_FILENAME
    staged = dest.with_name(f".{RUN_STATUS_FILENAME}.{os.getpid()}.partial")
    try:
        staged.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        os.replace(staged, dest)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return dest
