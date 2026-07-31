"""Provenance recorder (T-09).

Every subcommand writes a provenance record, including the ones whose bodies are
still ``NotImplementedError``. Provenance is the most expensive thing to add
retroactively: once a result exists without its input checksums, command line,
software versions, seed and timestamp, those facts are usually unrecoverable.

The record is deliberately boring and dependency-free so it cannot fail for
environmental reasons.
"""
from __future__ import annotations

import getpass  # noqa: F401  (intentionally unused; see redaction note below)
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ProvenanceError", "ProvenanceRecord", "sha256_file", "record"]


class ProvenanceError(ValueError):
    """An input cannot be recorded without losing or confusing another.

    A subclass of ValueError so existing callers keep working, and typed so it
    reaches the CLI's refusal contract (exit 4, one sentence) instead of
    escaping as a traceback. A bare ValueError in that list would swallow real
    bugs; a refusal has to be recognisable as one.
    """

# NOTE ON REDACTION: this recorder deliberately does NOT capture the OS username,
# hostname, or absolute home paths. Provenance must be publishable; a record that
# cannot be released without a scrubbing pass is a record that will be released
# unscrubbed. Input paths are stored relative to the project root where possible.
#
# The one thing recorded verbatim is `command`, which echoes whatever paths the
# invoker typed. That is deliberate and it is the limit of the policy: a record
# that cannot say what was run describes nothing. If a project's paths are
# themselves sensitive, run from inside the project root so they are relative.


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ProvenanceRecord:
    """The five things every result must carry."""

    command: str
    subcommand: str
    inputs: dict[str, str] = field(default_factory=dict)      # relpath -> sha256
    software: dict[str, str] = field(default_factory=dict)
    random_seed: int | None = None
    input_scale: int | None = None
    substrate_id: str | None = None
    timestamp_utc: str = ""
    schema_version: str = "1"
    #: The redaction policy as DATA, not as a comment. A later step that bundles
    #: records for release has to know machine-readably which fields were left
    #: unredacted; "read the source comment" is not an export contract.
    redaction_policy: str = "basenames_only_except_command"

    def __post_init__(self) -> None:
        if not self.timestamp_utc:
            self.timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.software:
            from motifmultiverse import __version__

            self.software = {
                "motifmultiverse": __version__,
                "python": platform.python_version(),
            }

    def add_input(self, path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None,
                  key: str | None = None) -> None:
        """Record one input by checksum, under a key that *distinguishes* it.

        Keys stay relative, per the redaction note above: to ``root`` when given,
        else the basename. They are never absolute and never contain a home
        directory -- which is also why the basename is *not* silently qualified
        with its parent when it collides. ``/home/<user>/modisco.h5`` would then
        record the username, trading a lost input for a leaked one.

        A basename does not always identify an input. ``modisco.h5`` is the
        standard TF-MoDISco filename, so ``ingest`` over several discovery
        outputs used to record exactly one checksum and attach it to whichever
        file was read last: a record naming one input while describing several.
        The recorder now refuses that instead of overwriting. Callers that read
        several same-named files pass ``root`` (keys become project-relative) or
        an explicit ``key`` -- ``ingest`` uses the config's own ``analysis_id``,
        which identifies the discovery run better than any path does. A
        provenance record that loses an input describes nothing, and a recorder
        that guesses at the difference describes something else.
        """
        p = Path(path)
        digest = sha256_file(p)
        if key is None:
            key = str(p.relative_to(root)) if root else p.name
        previous = self.inputs.get(key)
        if previous is not None and previous != digest:
            raise ProvenanceError(
                f"provenance key {key!r} already names a different input "
                f"(sha256 {previous[:12]}... vs {digest[:12]}...). Pass root= or "
                "key= so the recorded keys distinguish these files."
            )
        self.inputs[key] = digest

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, out_dir: str | os.PathLike[str]) -> Path:
        """Append this record to the directory's log without ever destroying it.

        Appending is a read-modify-write, and two things about that sequence
        have to hold or the log stops describing anything.

        It has to be atomic. Writing in place truncates the file first, so a
        process killed mid-write leaves half a JSON document -- and because
        every subcommand writes its record *before* it computes, that half
        document is read by the next run, and the one after it. One
        interrupted run would take out every later run in the directory. The
        replacement is therefore staged in a sibling file and moved into place
        with ``os.replace``, which is atomic: a reader sees the old log or the
        new one, never a truncation.

        And an unreadable log is not an empty one. Overwriting it would
        silently discard the recorded history of every earlier run in the
        directory -- the one loss provenance exists to prevent -- so the
        recorder refuses instead. The refusal is a ``ProvenanceError`` so it
        reaches the CLI's exit-4 contract rather than escaping as a
        ``JSONDecodeError`` traceback from inside an append.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "provenance.json"
        existing: list[dict[str, Any]] = []
        if dest.exists():
            try:
                existing = json.loads(dest.read_text())
            except ValueError as exc:
                raise ProvenanceError(
                    f"{dest} is not readable provenance ({exc}) and will not be "
                    "overwritten; move it aside to keep the earlier records, or "
                    "delete it to start a new log."
                ) from exc
            if not isinstance(existing, list):
                raise ProvenanceError(
                    f"{dest} is not a provenance log -- a log is a list of records, "
                    f"this is {type(existing).__name__} -- and will not be overwritten."
                )
        existing.append(self.to_dict())
        # Named per-process so two runs writing the same directory cannot stage
        # over each other's partial file; the final os.replace still decides
        # which one the log ends up holding.
        staged = dest.with_name(f"{dest.name}.{os.getpid()}.partial")
        staged.write_text(json.dumps(existing, indent=2, sort_keys=True))
        os.replace(staged, dest)
        return dest


def record(subcommand: str, out_dir: str | os.PathLike[str] | None = None,
           seed: int | None = None, input_scale: int | None = None,
           substrate_id: str | None = None) -> ProvenanceRecord:
    """Build (and optionally write) the record for one invocation."""
    rec = ProvenanceRecord(
        command=" ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        subcommand=subcommand,
        random_seed=seed,
        input_scale=input_scale,
        substrate_id=substrate_id,
    )
    if out_dir is not None:
        rec.write(out_dir)
    return rec
