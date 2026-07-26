"""Versioned schema for one frozen hit-caller substrate."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from . import SchemaError

__all__ = [
    "SUBSTRATE_SCHEMA_VERSION", "JsonValue", "CallerSpecification",
    "HitSubstrateManifest",
]

SUBSTRATE_SCHEMA_VERSION = "1"
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | Mapping[str, "JsonValue"]


def _require_digest(name: str, value: str) -> None:
    if (not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise SchemaError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CallerSpecification:
    """Every caller choice that makes one frozen run distinct."""

    caller_name: str
    caller_version: str
    lexicon_content_hash: str
    parameters: Mapping[str, JsonValue]
    preprocessing_contract_hash: str
    schema_version: str = SUBSTRATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (not isinstance(self.caller_name, str) or not isinstance(self.caller_version, str)
                or not self.caller_name.strip() or not self.caller_version.strip()):
            raise SchemaError("caller_name and caller_version are required")
        _require_digest("lexicon_content_hash", self.lexicon_content_hash)
        _require_digest("preprocessing_contract_hash", self.preprocessing_contract_hash)
        if not isinstance(self.parameters, Mapping):
            raise SchemaError("caller parameters must be a mapping")
        if not all(isinstance(key, str) for key in self.parameters):
            raise SchemaError("caller parameter keys must be strings")
        if not isinstance(self.schema_version, str):
            raise SchemaError("caller specification schema_version must be a string")
        if self.schema_version != SUBSTRATE_SCHEMA_VERSION:
            raise SchemaError(
                f"caller specification schema_version {self.schema_version!r} is unsupported"
            )


@dataclass(frozen=True)
class HitSubstrateManifest:
    """Content identity and non-semantic file-checksum provenance of a substrate."""

    substrate_id: str
    peak_universe_hash: str
    n_regions: int
    caller_specification: CallerSpecification
    input_files: Mapping[str, str]
    created_at: str
    schema_version: str = SUBSTRATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_digest("substrate_id", self.substrate_id)
        _require_digest("peak_universe_hash", self.peak_universe_hash)
        if type(self.n_regions) is not int or self.n_regions < 1:
            raise SchemaError("n_regions must be positive")
        if not isinstance(self.caller_specification, CallerSpecification):
            raise SchemaError("caller_specification must be a CallerSpecification")
        if not isinstance(self.input_files, Mapping):
            raise SchemaError("input_files provenance must be a mapping")
        if not isinstance(self.schema_version, str):
            raise SchemaError("substrate manifest schema_version must be a string")
        if self.schema_version != SUBSTRATE_SCHEMA_VERSION:
            raise SchemaError(f"substrate manifest schema_version {self.schema_version!r} is unsupported")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise SchemaError("created_at is required provenance")
        for name, checksum in self.input_files.items():
            if not name or not isinstance(name, str):
                raise SchemaError("input file provenance keys must be nonempty strings")
            _require_digest(f"input_files[{name!r}]", checksum)
