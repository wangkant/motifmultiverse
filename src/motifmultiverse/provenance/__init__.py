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

__all__ = ["ProvenanceRecord", "sha256_file", "record"]

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

    def add_input(self, path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> None:
        p = Path(path)
        key = str(p.relative_to(root)) if root else p.name
        self.inputs[key] = sha256_file(p)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, out_dir: str | os.PathLike[str]) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "provenance.json"
        existing: list[dict[str, Any]] = []
        if dest.exists():
            existing = json.loads(dest.read_text())
        existing.append(self.to_dict())
        dest.write_text(json.dumps(existing, indent=2, sort_keys=True))
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
