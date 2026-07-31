"""Canonical identity and manifest I/O for frozen hit substrates."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from motifmultiverse.schema import SchemaError
from motifmultiverse.schema.substrate import (
    SUBSTRATE_SCHEMA_VERSION,
    CallerSpecification,
    HitSubstrateManifest,
    _require_digest,
)

__all__ = [
    "SubstrateError", "canonical_json_bytes", "compute_substrate_id", "build_manifest",
    "read_manifest", "write_manifest", "OPPORTUNITY_LEDGER_SCHEMA_VERSION",
    "OpportunityLedger", "read_opportunity_ledger", "write_opportunity_ledger",
]


class SubstrateError(SchemaError):
    """An invalid semantic substrate identity or manifest."""


def _normalise_json(value: Any, *, key: str | None = None) -> Any:
    """Return strict JSON values, rejecting ambiguous identity inputs."""
    if isinstance(value, Path):
        if value.is_absolute():
            raise SubstrateError("absolute filenames cannot contribute to semantic identity")
        return str(value)
    if isinstance(value, str):
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise SubstrateError("absolute filenames cannot contribute to semantic identity")
        return value
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SubstrateError("NaN and Infinity cannot contribute to semantic identity")
        return 0 if value == 0 else value
    if isinstance(value, (set, frozenset)):
        raise SubstrateError("unordered sets cannot contribute to semantic identity")
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise SubstrateError("semantic identity mapping keys must be strings")
        return {k: _normalise_json(v, key=k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(v) for v in value]
    raise SubstrateError(f"{type(value).__name__} cannot contribute to semantic identity")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Canonical JSON for identity: sorted keys, finite scalars, no ambiguous values."""
    if not isinstance(payload, Mapping):
        raise SubstrateError("semantic identity must be a mapping")
    normalised = _normalise_json(payload)
    return json.dumps(
        normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def compute_substrate_id(manifest_without_id: Mapping[str, Any]) -> str:
    """Return the lowercase SHA-256 of strict canonical semantic JSON."""
    return hashlib.sha256(canonical_json_bytes(manifest_without_id)).hexdigest()


def _semantic_payload(
    *, peak_universe_hash: str, n_regions: int, caller_specification: CallerSpecification,
) -> dict[str, Any]:
    return {
        "schema_version": SUBSTRATE_SCHEMA_VERSION,
        "peak_universe_hash": peak_universe_hash,
        "n_regions": n_regions,
        "caller_specification": asdict(caller_specification),
    }


def build_manifest(
    *, peak_universe_hash: str, n_regions: int, caller_specification: CallerSpecification,
    input_files: Mapping[str, str], created_at: str,
) -> HitSubstrateManifest:
    """Build a manifest whose identity excludes non-semantic file provenance."""
    semantic = _semantic_payload(
        peak_universe_hash=peak_universe_hash, n_regions=n_regions,
        caller_specification=caller_specification,
    )
    return HitSubstrateManifest(
        substrate_id=compute_substrate_id(semantic), peak_universe_hash=peak_universe_hash,
        n_regions=n_regions, caller_specification=caller_specification,
        input_files=dict(input_files), created_at=created_at,
    )


def _manifest_payload(manifest: HitSubstrateManifest) -> dict[str, Any]:
    return asdict(manifest)


def write_manifest(manifest: HitSubstrateManifest, path: str | Path) -> Path:
    """Write a self-validating versioned manifest with checksum provenance."""
    _assert_identity(manifest)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(_manifest_payload(manifest), indent=2, sort_keys=True) + "\n")
    return dest


def _assert_identity(manifest: HitSubstrateManifest) -> None:
    expected = compute_substrate_id(_semantic_payload(
        peak_universe_hash=manifest.peak_universe_hash, n_regions=manifest.n_regions,
        caller_specification=manifest.caller_specification,
    ))
    if manifest.substrate_id != expected:
        raise SubstrateError("manifest substrate_id does not match its semantic caller specification")


def read_manifest(path: str | Path) -> HitSubstrateManifest:
    """Load and verify a manifest before a consumer trusts its substrate id."""
    p = Path(path)
    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise SubstrateError(f"{p}: invalid manifest JSON") from exc
    expected = {
        "schema_version", "substrate_id", "peak_universe_hash", "n_regions",
        "caller_specification", "input_files", "created_at",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise SubstrateError(f"{p}: manifest keys must be exactly {sorted(expected)}")
    try:
        specification = CallerSpecification(**payload["caller_specification"])
        manifest = HitSubstrateManifest(
            substrate_id=payload["substrate_id"], peak_universe_hash=payload["peak_universe_hash"],
            n_regions=payload["n_regions"], caller_specification=specification,
            input_files=payload["input_files"], created_at=payload["created_at"],
            schema_version=payload["schema_version"],
        )
    except (TypeError, KeyError, SchemaError) as exc:
        raise SubstrateError(f"{p}: invalid substrate manifest") from exc
    _assert_identity(manifest)
    return manifest


# --------------------------------------------------------------------------- #
# The opportunity ledger.
#
# `guards.four_state_missingness` -- the guard for this project's founding
# failure -- had no call site for one reason: nothing this package emits states a
# coverage independently of the rows the guard would recompute it from. Pointing
# it at `interpret.health_report` would have been a guard auditing its own
# producer, and putting the counts on `HitSubstrateManifest` would have re-identified
# every existing substrate, including the frozen K562 run whose id is embedded in
# published interpretations, test fixtures and a multiverse audit. Re-identifying a
# frozen artifact to carry two integers is the tail wagging the dog.
#
# So the counts live where the knowledge does: with the program that FROZE the run.
# That program already knows how many (region, variant) opportunities it
# materialised and how many it retained -- the real K562 substrate is built from an
# upstream table whose own vocabulary is USED / SEARCHED_NOT_RETAINED / NOT_SEARCHED
# -- and it simply did not carry them across. A ledger written there and read here
# is the `verify_against_manifest` shape: the claim comes from one producer and the
# recomputation from another's bytes, at zero cost to any existing identity.
#
# What binds it to the substrate is that it names the substrate it describes, and
# reading it against a different one is refused. A ledger that could be read beside
# any substrate would be a claim about nothing.
# --------------------------------------------------------------------------- #

#: Emitted artifacts here carry schema versions; so does this one.
OPPORTUNITY_LEDGER_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class OpportunityLedger:
    """How much the freezing program searched, and how much it retained.

    Two denominators rather than one, deliberately. ``n_retained /
    n_opportunities`` is what `guards.four_state_missingness` recomputes: how much
    of everything that could have been measured was kept. ``n_searched /
    n_opportunities`` is a different question -- how much was looked at -- and
    `interpret.peak_universe` answers a third, treating a searched-but-unretained
    opportunity as a measurement contributing nothing.

    Those are three legitimate quantities and the ambiguity between them is what
    kept this ledger unwritten. Recording two of them under distinct names, and
    letting the guard keep its own arithmetic, resolves it without redefining
    ``defined`` -- which would have been the tempting fix and would have destroyed
    the guard: a ``defined`` that counts NO_SEQUENCE_MATCH rows can no longer
    detect a fill that wrote a value into one, which is the failure the guard
    exists for.
    """

    substrate_id: str
    n_opportunities: int
    n_retained: int
    n_searched: int
    producer: str
    schema_version: str = OPPORTUNITY_LEDGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_digest("substrate_id", self.substrate_id)
        for name in ("n_opportunities", "n_retained", "n_searched"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SubstrateError(f"{name} must be a non-negative integer")
        if self.n_retained > self.n_searched:
            raise SubstrateError(
                f"n_retained ({self.n_retained}) exceeds n_searched ({self.n_searched}): "
                "an opportunity cannot be retained without having been searched"
            )
        if self.n_searched > self.n_opportunities:
            raise SubstrateError(
                f"n_searched ({self.n_searched}) exceeds n_opportunities "
                f"({self.n_opportunities})"
            )
        if not str(self.producer or "").strip():
            raise SubstrateError(
                "producer is required: a ledger is evidence because it comes from the "
                "program that froze the run, so it has to say which program that was"
            )
        if self.schema_version != OPPORTUNITY_LEDGER_SCHEMA_VERSION:
            raise SubstrateError(
                f"opportunity ledger schema_version {self.schema_version!r} is not "
                f"{OPPORTUNITY_LEDGER_SCHEMA_VERSION!r}"
            )

    @property
    def retained_coverage(self) -> float:
        """``n_retained / n_opportunities`` -- the fraction the guard recomputes."""
        return self.n_retained / self.n_opportunities if self.n_opportunities else float("nan")


def read_opportunity_ledger(path: str | Path, *, substrate_id: str) -> OpportunityLedger:
    """Read a ledger and refuse one that describes a different frozen run.

    ``substrate_id`` is required rather than optional. A ledger that can be read
    beside any substrate is a claim about nothing, and the failure it would hide is
    the ordinary one: a directory holding last week's ledger next to this week's
    hit table, agreeing with neither.
    """
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubstrateError(f"{p}: invalid opportunity ledger JSON") from exc
    expected = {"schema_version", "substrate_id", "n_opportunities", "n_retained",
                "n_searched", "producer"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise SubstrateError(f"{p}: ledger keys must be exactly {sorted(expected)}")
    try:
        ledger = OpportunityLedger(**payload)
    except (TypeError, SchemaError) as exc:
        raise SubstrateError(f"{p}: invalid opportunity ledger") from exc
    if ledger.substrate_id != substrate_id:
        raise SubstrateError(
            f"{p} describes substrate {ledger.substrate_id} but the hit table carries "
            f"{substrate_id}; refusing to check one frozen run's coverage against "
            "another's counts"
        )
    return ledger


def write_opportunity_ledger(ledger: OpportunityLedger, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(json.dumps(asdict(ledger), indent=2, sort_keys=True), encoding="utf-8")
    return p
