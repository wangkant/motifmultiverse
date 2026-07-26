"""Canonical identity and manifest I/O for frozen hit substrates."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path, PureWindowsPath
from typing import Any

from motifmultiverse.schema import SchemaError
from motifmultiverse.schema.substrate import (
    SUBSTRATE_SCHEMA_VERSION,
    CallerSpecification,
    HitSubstrateManifest,
)

__all__ = [
    "SubstrateError", "canonical_json_bytes", "compute_substrate_id", "build_manifest",
    "read_manifest", "write_manifest",
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
