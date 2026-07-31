"""Affected-subset downstream stability validation.

The all-peak reconstruction summary is retained as a diagnostic only.  A merge
can change a small subset while the all-peak median is zero, so every decision
is evaluated and labelled on the subset whose hit identities or coefficients
actually changed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

from motifmultiverse.provenance import ProvenanceRecord, record, sha256_file
from motifmultiverse.schema import (
    PeakSplitManifest,
    SchemaError,
    SplitRole,
    build_peak_split_manifest,
    peak_split_manifest_checksum,
)

from .base import (
    SPLIT_ARTIFACT_SCHEMA_VERSION,
    AnalysisMode,
    CrossFitFold,
    DecisionSplitArtifact,
    ValidationSplitArtifact,
    assert_artifact_split_compatibility,
    assert_cross_fit_compatibility,
    assert_split_compatibility,
)

__all__ = [
    "AnalysisMode", "BackendUnavailable", "BackendVerification", "CrossFitFold",
    "DecisionSplitArtifact", "LexiconBinding", "PeakSplitManifest", "SPLIT_ARTIFACT_SCHEMA_VERSION",
    "STABILITY_SCHEMA_VERSION", "StabilityBackend", "StabilityProvenance", "StabilityResult", "SplitRole",
    "ValidationError", "ValidationSplitArtifact", "assert_artifact_split_compatibility",
    "assert_cross_fit_compatibility", "assert_split_compatibility",
    "build_peak_split_manifest", "evaluate_stability", "load_lexicon_binding",
    "normalize_backend_output", "peak_split_manifest_checksum", "run_backend_validation", "run",
    "stability_result_id",
    "write_stability_artifacts",
]

STABILITY_SCHEMA_VERSION = "1"
STABILITY_PROVENANCE_CONTRACT_VERSION = "stability-provenance-1"
MIN_AFFECTED_PEAKS = 30
_REQUIRED_COLUMNS = ("peak_id", "hit_id", "coefficient", "reconstruction")
_PERSISTED_RESULT_COLUMNS = (
    "decision_id", "n_affected_peaks", "n_affected_hits", "family_coefficient_share",
    "paired_delta_reconstruction_affected", "paired_delta_reconstruction_all", "hit_jaccard",
    "coefficient_conservation", "status", "power_statement", "affected_interval", "schema_version",
    "artifact_id", "split_manifest_checksum", "decision_artifact_id", "validation_artifact_id",
    "provenance",
)


class ValidationError(SchemaError):
    """Validation data, backend output, or persisted artifacts are malformed."""


class BackendUnavailable(ValidationError):
    """An optional backend cannot be used in this environment."""


@dataclass(frozen=True)
class LexiconBinding:
    """The exact compiled lexicon files and semantic hashes an adapter receives."""

    lexicon_identity: str
    entries: tuple[tuple[str, str, str, str], ...]
    schema_version: str = STABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.lexicon_identity, str) or not self.lexicon_identity.startswith("lexicons:"):
            raise ValidationError("lexicon_identity must be a content-addressed lexicons identity")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValidationError("lexicon binding requires at least one compiled manifest entry")
        if self.schema_version != STABILITY_SCHEMA_VERSION:
            raise ValidationError("lexicon binding has an unsupported schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {"lexicon_identity": self.lexicon_identity, "entries": self.entries,
                "schema_version": self.schema_version}


@dataclass(frozen=True)
class StabilityProvenance:
    """Full, immutable validation provenance plus non-circular artifact identities.

    ``validation_split_identity`` hashes the complete validation split semantics
    while deliberately excluding only ``ValidationSplitArtifact.result_id`` and
    its derived ``artifact_id``.  Those two values point back to the stability
    identity being computed; excluding anything broader would discard real
    provenance, while including either would make the identity impossible to
    construct.
    """

    command: str
    subcommand: str
    stage: str
    inputs: Mapping[str, str]
    software: Mapping[str, str]
    random_seed: int | None
    input_scale: int | None
    substrate_id: str
    timestamp_utc: str
    schema_version: str
    redaction_policy: str
    lexicon_identity: str
    split_manifest_checksum: str
    decision_artifact_id: str
    validation_split_identity: str
    contract_version: str = STABILITY_PROVENANCE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in (
            "command", "subcommand", "stage", "timestamp_utc", "schema_version",
            "redaction_policy", "lexicon_identity", "split_manifest_checksum",
            "decision_artifact_id", "validation_split_identity", "contract_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"stability provenance {name} must be a non-empty string")
        if self.subcommand != "validate" or self.stage != "validate":
            raise ValidationError("stability provenance subcommand and stage must both be 'validate'")
        if self.schema_version != "1":
            raise ValidationError("stability provenance recorder schema_version must be '1'")
        if self.contract_version != STABILITY_PROVENANCE_CONTRACT_VERSION:
            raise ValidationError("stability provenance contract_version is unsupported")
        if self.redaction_policy != "basenames_only_except_command":
            raise ValidationError("stability provenance redaction_policy is unsupported")
        if (
            not self.lexicon_identity.startswith("lexicons:")
            or not _is_sha256(self.lexicon_identity.removeprefix("lexicons:"))
        ):
            raise ValidationError("stability provenance lexicon_identity is malformed")
        for name in ("split_manifest_checksum", "validation_split_identity"):
            if not _is_sha256(getattr(self, name)):
                raise ValidationError(f"stability provenance {name} must be a SHA-256 digest")
        if (
            not self.decision_artifact_id.startswith("decision-split:")
            or not _is_sha256(self.decision_artifact_id.removeprefix("decision-split:"))
        ):
            raise ValidationError(
                "stability provenance decision_artifact_id must be a decision-split identity"
            )
        if isinstance(self.random_seed, bool) or (
            self.random_seed is not None and not isinstance(self.random_seed, int)
        ):
            raise ValidationError("stability provenance random_seed must be an integer or None")
        if isinstance(self.input_scale, bool) or (
            self.input_scale is not None
            and (not isinstance(self.input_scale, int) or self.input_scale < 0)
        ):
            raise ValidationError(
                "stability provenance input_scale must be a non-negative integer or None"
            )
        if not _is_sha256(self.substrate_id):
            raise ValidationError(
                "stability provenance substrate_id must be a lowercase SHA-256 digest"
            )
        frozen_inputs = _validated_string_mapping(self.inputs, "inputs", digests=True)
        frozen_software = _validated_string_mapping(self.software, "software", digests=False)
        if not frozen_inputs:
            raise ValidationError("stability provenance inputs cannot be empty")
        if not frozen_software:
            raise ValidationError("stability provenance software cannot be empty")
        object.__setattr__(self, "inputs", MappingProxyType(frozen_inputs))
        object.__setattr__(self, "software", MappingProxyType(frozen_software))

    @classmethod
    def from_record(
        cls,
        source: ProvenanceRecord,
        *,
        lexicon: LexiconBinding,
        manifest: PeakSplitManifest,
        decision: DecisionSplitArtifact,
        validation: ValidationSplitArtifact,
    ) -> StabilityProvenance:
        """Preserve every recorder field and add exact validation-stage identities."""
        if type(source) is not ProvenanceRecord:
            raise ValidationError("stability provenance source must be an exact ProvenanceRecord")
        _validate_identity_context(lexicon, manifest, decision, validation)
        return cls(
            command=source.command,
            subcommand=source.subcommand,
            stage="validate",
            inputs=dict(source.inputs),
            software=dict(source.software),
            random_seed=source.random_seed,
            input_scale=source.input_scale,
            substrate_id=source.substrate_id,
            timestamp_utc=source.timestamp_utc,
            schema_version=source.schema_version,
            redaction_policy=source.redaction_policy,
            lexicon_identity=lexicon.lexicon_identity,
            split_manifest_checksum=manifest.checksum,
            decision_artifact_id=decision.artifact_id,
            validation_split_identity=_validation_split_identity(
                manifest, decision, validation,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "subcommand": self.subcommand,
            "stage": self.stage,
            "inputs": dict(self.inputs),
            "software": dict(self.software),
            "random_seed": self.random_seed,
            "input_scale": self.input_scale,
            "substrate_id": self.substrate_id,
            "timestamp_utc": self.timestamp_utc,
            "schema_version": self.schema_version,
            "redaction_policy": self.redaction_policy,
            "lexicon_identity": self.lexicon_identity,
            "split_manifest_checksum": self.split_manifest_checksum,
            "decision_artifact_id": self.decision_artifact_id,
            "validation_split_identity": self.validation_split_identity,
            "contract_version": self.contract_version,
        }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_string_mapping(
    value: object,
    name: str,
    *,
    digests: bool,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"stability provenance {name} must be a mapping")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValidationError(f"stability provenance {name} keys must be non-empty strings")
        if not isinstance(item, str) or not item.strip():
            raise ValidationError(f"stability provenance {name} values must be non-empty strings")
        if digests and not _is_sha256(item):
            raise ValidationError("stability provenance inputs values must be SHA-256 digests")
        normalized[key] = item
    return dict(sorted(normalized.items()))


def load_lexicon_binding(lexicons: str | Path) -> LexiconBinding:
    """Validate the compiled lexicon manifests and bind every accompanying H5 byte stream."""
    root = Path(lexicons)
    if not root.exists():
        raise ValidationError(f"compiled lexicons path does not exist: {root}")
    if not root.is_dir():
        raise ValidationError(f"compiled lexicons path must be a directory: {root}")
    manifest_paths = sorted(root.glob("*.manifest.json"))
    if not manifest_paths:
        raise ValidationError(f"compiled lexicons directory has no *.manifest.json: {root}")
    from motifmultiverse.compile import CompileError, validate_compiled_lexicon

    entry_rows: list[tuple[str, str, str, str]] = []
    seen_tiers: set[str] = set()
    for manifest_path in manifest_paths:
        try:
            manifest = validate_compiled_lexicon(manifest_path)
        except (CompileError, OSError) as exc:
            raise ValidationError(f"{manifest_path} is not a valid compiled lexicon manifest: {exc}") from exc
        if manifest_path.name != f"{manifest.tier}.manifest.json":
            raise ValidationError(f"{manifest_path} filename must exactly match declared tier {manifest.tier!r}")
        if manifest.tier in seen_tiers:
            raise ValidationError(f"compiled lexicons contain duplicate tier manifest {manifest.tier!r}")
        seen_tiers.add(manifest.tier)
        h5_path = root / f"{manifest.tier}.h5"
        if not h5_path.is_file():
            raise ValidationError(f"compiled lexicon named by {manifest_path} is missing: {h5_path}")
        entry_rows.append((manifest.tier, manifest.lexicon_content_hash,
                           sha256_file(manifest_path), sha256_file(h5_path)))
    entries = tuple(sorted(entry_rows))
    h5_tiers = {path.stem for path in root.glob("*.h5")}
    if h5_tiers != seen_tiers:
        raise ValidationError(
            "compiled lexicon manifests and HDF5 companions are ambiguous: "
            f"manifest tiers={sorted(seen_tiers)} h5 tiers={sorted(h5_tiers)}"
        )
    digest = hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest()
    return LexiconBinding(lexicon_identity=f"lexicons:{digest}", entries=entries)


@dataclass(frozen=True)
class StabilityResult:
    """Versioned, affected-subset evidence for one downstream decision."""

    decision_id: str
    n_affected_peaks: int
    n_affected_hits: int
    family_coefficient_share: float
    paired_delta_reconstruction_affected: float | None
    paired_delta_reconstruction_all: float
    hit_jaccard: float | None
    coefficient_conservation: float | None
    status: str
    power_statement: str
    affected_interval: tuple[float, float] | None = None
    backend: str = ""
    backend_version: str = ""
    backend_result_id: str = ""
    lexicon_identity: str = ""
    schema_version: str = STABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.decision_id, str) or not self.decision_id.strip():
            raise ValidationError("stability decision_id must be a non-empty string")
        for name in ("n_affected_peaks", "n_affected_hits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"stability {name} must be a non-negative integer")
        if self.schema_version != STABILITY_SCHEMA_VERSION:
            raise ValidationError(
                f"stability schema_version must be {STABILITY_SCHEMA_VERSION!r}"
            )
        _finite("family_coefficient_share", self.family_coefficient_share)
        if not 0.0 <= self.family_coefficient_share <= 1.0:
            raise ValidationError("stability family_coefficient_share must be in [0, 1]")
        _finite("paired_delta_reconstruction_all", self.paired_delta_reconstruction_all)
        for name in (
            "paired_delta_reconstruction_affected", "hit_jaccard", "coefficient_conservation",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite(f"stability {name}", value)
        if self.hit_jaccard is not None and not 0.0 <= self.hit_jaccard <= 1.0:
            raise ValidationError("stability hit_jaccard must be in [0, 1]")
        if self.coefficient_conservation is not None and not -1.0 <= self.coefficient_conservation <= 1.0:
            raise ValidationError("stability coefficient_conservation must be in [-1, 1]")
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValidationError("stability status must be a non-empty string")
        if not isinstance(self.power_statement, str) or not self.power_statement.strip():
            raise ValidationError("stability power_statement must be a non-empty string")
        for name in ("backend", "backend_version", "backend_result_id", "lexicon_identity"):
            value = getattr(self, name)
            if value and (not isinstance(value, str) or not value.strip()):
                raise ValidationError(f"stability {name} must be a non-empty string when bound")
        if self.affected_interval is not None:
            if (
                not isinstance(self.affected_interval, tuple)
                or len(self.affected_interval) != 2
            ):
                raise ValidationError("stability affected_interval must be a two-value tuple or None")
            _finite("stability affected_interval[0]", self.affected_interval[0])
            _finite("stability affected_interval[1]", self.affected_interval[1])
        if self.n_affected_peaks < MIN_AFFECTED_PEAKS:
            if self.status != "LOW_RISK_RARE_NOT_VALIDATED":
                raise ValidationError(
                    "fewer than 30 affected peaks must be LOW_RISK_RARE_NOT_VALIDATED"
                )
            if self.affected_interval is not None:
                raise ValidationError("rare affected subsets must not report an interval")
            if "frequency-limited" not in self.power_statement.lower():
                raise ValidationError("rare affected subsets require a frequency-limited power statement")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendVerification:
    """Per-backend verification state; unverified is evidence, not absence."""

    backend: str
    backend_version: str
    status: str
    detail: str = ""
    backend_result_id: str | None = None
    lexicon_identity: str = ""
    schema_version: str = STABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("backend", "backend_version", "status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"backend verification {name} must be a non-empty string")
        if self.status not in {"VERIFIED", "UNVERIFIED"}:
            raise ValidationError("backend verification status must be VERIFIED or UNVERIFIED")
        if not isinstance(self.detail, str):
            raise ValidationError("backend verification detail must be a string")
        if self.backend_result_id is not None and (
            not isinstance(self.backend_result_id, str) or not self.backend_result_id.strip()
        ):
            raise ValidationError("backend verification backend_result_id must be non-empty or None")
        if self.lexicon_identity and (
            not isinstance(self.lexicon_identity, str) or not self.lexicon_identity.startswith("lexicons:")
        ):
            raise ValidationError("backend verification lexicon_identity must be bound when present")
        if self.schema_version != STABILITY_SCHEMA_VERSION:
            raise ValidationError("backend verification has an unsupported schema_version")


class StabilityBackend:
    """Backend adapter returning the pre-merge and post-merge standardized tables."""

    name = "backend"
    version = "unknown"
    optional = False

    def compare(self, lexicons: str | Path, decision_id: str) -> tuple[Any, Any]:
        raise NotImplementedError


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{name} must be a finite numeric value")
    return float(value)


def normalize_backend_output(output: Any, *, backend: str):
    """Require one exact semantic schema before any backend is compared.

    Backends may carry extra implementation columns, but their identity,
    coefficient, and reconstruction values cannot be inferred or coerced from
    backend-specific names.  Duplicate peak/hit rows are also refused: summing
    them would manufacture a coefficient change.
    """
    import pandas as pd

    if not isinstance(backend, str) or not backend.strip():
        raise ValidationError("backend name must be a non-empty string")
    if not isinstance(output, pd.DataFrame):
        raise ValidationError(f"{backend} output must be a pandas DataFrame")
    missing = [column for column in _REQUIRED_COLUMNS if column not in output.columns]
    if missing:
        raise ValidationError(f"{backend} output is missing standardized column(s): {missing}")
    frame = output.loc[:, _REQUIRED_COLUMNS].copy()
    if frame.empty:
        raise ValidationError(f"{backend} output cannot be empty")
    for column in ("peak_id", "hit_id"):
        if any(not isinstance(value, str) or not value.strip() for value in frame[column]):
            raise ValidationError(f"{backend} output {column} must contain non-empty string identities")
    for column in ("coefficient", "reconstruction"):
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in frame[column]):
            raise ValidationError(f"{backend} output {column} must have numeric values")
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{backend} output {column} must be numeric") from exc
        if not frame[column].map(lambda value: math.isfinite(float(value))).all():
            raise ValidationError(f"{backend} output {column} must be finite")
        frame[column] = frame[column].astype("float64")
    if frame.duplicated(["peak_id", "hit_id"]).any():
        raise ValidationError(f"{backend} output has duplicate standardized hit identities")
    return frame.sort_values(["peak_id", "hit_id"], kind="stable").reset_index(drop=True)


def _peak_reconstruction(frame, backend: str) -> dict[str, float]:
    grouped = frame.groupby("peak_id", sort=False)["reconstruction"].agg(["min", "max"])
    inconsistent = grouped.index[grouped["min"] != grouped["max"]].tolist()
    if inconsistent:
        raise ValidationError(
            f"{backend} output has multiple reconstruction values for peak(s): {inconsistent}"
        )
    return {str(peak): float(value) for peak, value in grouped["min"].items()}


def evaluate_stability(decision_id: str, before: Any, after: Any) -> StabilityResult:
    """Evaluate a merge on affected peaks and retain the all-peak dilution diagnostic."""
    import numpy as np

    before_frame = normalize_backend_output(before, backend="before")
    after_frame = normalize_backend_output(after, backend="after")
    before_reconstruction = _peak_reconstruction(before_frame, "before")
    after_reconstruction = _peak_reconstruction(after_frame, "after")
    if set(before_reconstruction) != set(after_reconstruction):
        raise ValidationError("before and after outputs must cover exactly the same peak identities")

    before_coefficients = {
        (str(row.peak_id), str(row.hit_id)): float(row.coefficient)
        for row in before_frame.itertuples(index=False)
    }
    after_coefficients = {
        (str(row.peak_id), str(row.hit_id)): float(row.coefficient)
        for row in after_frame.itertuples(index=False)
    }
    affected_peaks = {
        peak
        for peak, hit_id in set(before_coefficients) | set(after_coefficients)
        if before_coefficients.get((peak, hit_id)) != after_coefficients.get((peak, hit_id))
    }
    affected = sorted(affected_peaks)
    all_delta = [after_reconstruction[peak] - before_reconstruction[peak]
                 for peak in sorted(before_reconstruction)]
    affected_delta = [after_reconstruction[peak] - before_reconstruction[peak] for peak in affected]
    # The affected-peak set is built once and reused. Writing `key[0] in
    # set(affected)` inside these comprehensions looked equivalent but was not:
    # the condition is re-evaluated for every hit, so the set was rebuilt from
    # the sorted list once PER HIT ROW and the scan became quadratic in the hit
    # table. A real-sized validation (80,000 hits over 16,000 affected peaks)
    # took 99 seconds that way; the identical result now takes under a second.
    # Nothing about the values changes -- only how many times the same set is
    # constructed.
    affected_keys_before = {key for key in before_coefficients if key[0] in affected_peaks}
    affected_keys_after = {key for key in after_coefficients if key[0] in affected_peaks}
    affected_keys = affected_keys_before | affected_keys_after
    union = affected_keys_before | affected_keys_after
    hit_jaccard = None if not union else len(affected_keys_before & affected_keys_after) / len(union)
    shared = sorted(affected_keys_before & affected_keys_after)
    if len(shared) < 2:
        conservation = None
    else:
        left = np.asarray([before_coefficients[key] for key in shared], dtype=float)
        right = np.asarray([after_coefficients[key] for key in shared], dtype=float)
        conservation_value = float(np.corrcoef(left, right)[0, 1])
        conservation = conservation_value if math.isfinite(conservation_value) else None
    denominator = sum(abs(value) for value in after_coefficients.values())
    numerator = sum(abs(after_coefficients[key]) for key in affected_keys_after)
    share = 0.0 if denominator == 0 else numerator / denominator
    n_affected = len(affected)
    if n_affected < MIN_AFFECTED_PEAKS:
        status = "LOW_RISK_RARE_NOT_VALIDATED"
        power = (
            f"Evidence strength is frequency-limited: {n_affected} affected peaks is below "
            f"the preregistered floor of {MIN_AFFECTED_PEAKS}; no interval or inferential "
            "equivalence claim is reported."
        )
    elif affected_delta and math.isclose(float(np.median(affected_delta)), 0.0, abs_tol=1e-12):
        status = "STABLE_AFFECTED_SUBSET"
        power = (
            f"Affected-subset stability is descriptive for {n_affected} affected peaks; "
            "it is not an equivalence claim."
        )
    else:
        status = "CHANGED_AFFECTED_SUBSET"
        power = (
            f"Affected-subset change is descriptive for {n_affected} affected peaks; "
            "it is not an equivalence claim."
        )
    return StabilityResult(
        decision_id=decision_id,
        n_affected_peaks=n_affected,
        n_affected_hits=len(affected_keys),
        family_coefficient_share=float(share),
        paired_delta_reconstruction_affected=(
            None if not affected_delta else float(np.median(affected_delta))
        ),
        paired_delta_reconstruction_all=float(np.median(all_delta)),
        hit_jaccard=hit_jaccard,
        coefficient_conservation=conservation,
        status=status,
        power_statement=power,
    )


def run_backend_validation(
    lexicons: LexiconBinding,
    decision_id: str,
    backends: Sequence[StabilityBackend],
) -> tuple[tuple[StabilityResult, ...], tuple[BackendVerification, ...]]:
    """Run independent adapters without converting a missing optional one to success."""
    if not isinstance(lexicons, LexiconBinding):
        raise ValidationError("backend validation requires a validated LexiconBinding")
    results: list[StabilityResult] = []
    verification: list[BackendVerification] = []
    for backend in backends:
        name = getattr(backend, "name", type(backend).__name__)
        version = getattr(backend, "version", "unknown")
        optional = bool(getattr(backend, "optional", False))
        if not isinstance(name, str) or not name.strip() or not isinstance(version, str) or not version.strip():
            raise ValidationError("stability backend requires non-empty name and version")
        try:
            before, after = backend.compare(lexicons, decision_id)
            before = normalize_backend_output(before, backend=name)
            after = normalize_backend_output(after, backend=name)
            backend_result_id = "backend-result:" + hashlib.sha256(json.dumps({
                "after": after.to_dict("records"), "backend": name, "backend_version": version,
                "before": before.to_dict("records"), "decision_id": decision_id,
                "lexicon_identity": lexicons.lexicon_identity,
            }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            result = replace(
                evaluate_stability(decision_id, before, after),
                backend=name,
                backend_version=version,
                backend_result_id=backend_result_id,
                lexicon_identity=lexicons.lexicon_identity,
            )
            results.append(result)
            verification.append(BackendVerification(
                name, version, "VERIFIED", backend_result_id=backend_result_id,
                lexicon_identity=lexicons.lexicon_identity,
            ))
        except BackendUnavailable as exc:
            detail = f"{name} backend unavailable: {exc}"
            if not optional:
                raise BackendUnavailable(detail) from exc
            verification.append(BackendVerification(
                name, version, "UNVERIFIED", detail, lexicon_identity=lexicons.lexicon_identity,
            ))
    return tuple(results), tuple(verification)


def _split_binding(
    manifest: PeakSplitManifest,
    decision: DecisionSplitArtifact,
    validation: ValidationSplitArtifact,
) -> dict[str, Any]:
    """The result-ID binding intentionally excludes its own circular result_id/artifact_id."""
    return {
        "decision_artifact_id": decision.artifact_id,
        "manifest_checksum": manifest.checksum,
        "validation": {
            "cross_fit_folds": [
                {"fold_id": fold.fold_id, "decision_peak_ids": sorted(fold.decision_peak_ids),
                 "evaluation_peak_ids": sorted(fold.evaluation_peak_ids)}
                for fold in validation.cross_fit_folds
            ],
            "decision_id": validation.decision_id,
            "decision_peak_ids": sorted(validation.decision_peak_ids),
            "mode": validation.mode.value,
            "schema_version": validation.schema_version,
            "validation_peak_ids": sorted(validation.validation_peak_ids),
        },
    }


def _validation_split_identity(
    manifest: PeakSplitManifest,
    decision: DecisionSplitArtifact,
    validation: ValidationSplitArtifact,
) -> str:
    return hashlib.sha256(
        json.dumps(
            _split_binding(manifest, decision, validation),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_identity_context(
    lexicon: object,
    manifest: object,
    decision: object,
    validation: object,
) -> None:
    if not isinstance(lexicon, LexiconBinding):
        raise ValidationError("stability identity requires a validated LexiconBinding")
    if not isinstance(manifest, PeakSplitManifest):
        raise ValidationError("stability identity requires a PeakSplitManifest")
    if not isinstance(decision, DecisionSplitArtifact):
        raise ValidationError("stability identity requires a DecisionSplitArtifact")
    if not isinstance(validation, ValidationSplitArtifact):
        raise ValidationError("stability identity requires a ValidationSplitArtifact")
    assert_artifact_split_compatibility(manifest, decision, validation)


def _canonical_json(value: Any, *, what: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{what} is not canonical JSON: {exc}") from exc


def _canonical_records(
    values: object,
    record_type: type[StabilityResult] | type[BackendVerification],
    *,
    what: str,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ValidationError(f"{what} must be a sequence of {record_type.__name__} records")
    rows = tuple(values)
    for position, row in enumerate(rows):
        if type(row) is not record_type:
            raise ValidationError(
                f"{what}[{position}] must be an exact {record_type.__name__} record"
            )
        try:
            if record_type is StabilityResult:
                StabilityResult(**row.to_dict())
            else:
                BackendVerification(**asdict(row))
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{what}[{position}] is malformed: {exc}") from exc
    return tuple(sorted(
        rows,
        key=lambda row: _canonical_json(
            row.to_dict() if isinstance(row, StabilityResult) else asdict(row),
            what=what,
        ),
    ))


def _validated_stability_identity_inputs(
    results: object,
    verification: object,
    provenance: object,
    lexicon: object,
    manifest: object,
    decision: object,
    validation: object,
) -> tuple[tuple[StabilityResult, ...], tuple[BackendVerification, ...]]:
    _validate_identity_context(lexicon, manifest, decision, validation)
    if type(provenance) is not StabilityProvenance:
        raise ValidationError(
            "stability identity provenance must be an exact StabilityProvenance record"
        )
    # Reconstructing validates every field again before any bytes are hashed.
    StabilityProvenance(**provenance.to_dict())
    if provenance.lexicon_identity != lexicon.lexicon_identity:
        raise ValidationError("stability provenance lexicon identity does not match")
    if provenance.split_manifest_checksum != manifest.checksum:
        raise ValidationError("stability provenance split manifest identity does not match")
    if provenance.decision_artifact_id != decision.artifact_id:
        raise ValidationError("stability provenance decision split identity does not match")
    if provenance.validation_split_identity != _validation_split_identity(
        manifest, decision, validation,
    ):
        raise ValidationError("stability provenance validation split identity does not match")
    return (
        _canonical_records(results, StabilityResult, what="stability results"),
        _canonical_records(
            verification, BackendVerification, what="backend verification",
        ),
    )


def stability_result_id(
    results: Sequence[StabilityResult],
    verification: Sequence[BackendVerification],
    provenance: StabilityProvenance,
    lexicon: LexiconBinding,
    manifest: PeakSplitManifest,
    decision: DecisionSplitArtifact,
    validation: ValidationSplitArtifact,
) -> str:
    """Canonical emitted artifact identity, and the required ValidationSplitArtifact.result_id.

    The identity deliberately excludes only the output's self-reference.  It
    therefore commits to normalized backend results, verification rows, full
    provenance, every compiled lexicon byte/hash binding, and split semantics.
    """
    result_rows, verification_rows = _validated_stability_identity_inputs(
        results, verification, provenance, lexicon, manifest, decision, validation,
    )
    payload = {
        "lexicon": lexicon.to_dict(),
        "provenance": provenance.to_dict(),
        "results": [row.to_dict() for row in result_rows],
        "schema_version": STABILITY_SCHEMA_VERSION,
        "split_binding": _split_binding(manifest, decision, validation),
        "verification": [asdict(row) for row in verification_rows],
    }
    digest = hashlib.sha256(
        _canonical_json(payload, what="stability identity payload").encode("utf-8")
    ).hexdigest()
    return f"stability:{digest}"


def _publish_directory_noreplace(stage: Path, destination: Path) -> None:
    """Atomically publish ``stage`` only when ``destination`` is still absent.

    POSIX ``rename``/``replace`` may replace an empty destination directory, so
    a separate existence check cannot provide the no-clobber contract.  Linux
    ``renameat2(RENAME_NOREPLACE)`` makes the absence check and directory rename
    one filesystem operation.  Filesystems without that flag publish an atomic
    no-replace symlink to the already-complete private directory; readers still
    see the complete directory at ``destination`` in one namespace operation.
    """
    import ctypes
    import errno

    def publish_symlink() -> None:
        try:
            os.symlink(
                os.path.relpath(stage, destination.parent),
                destination,
                target_is_directory=True,
            )
        except FileExistsError as exc:
            raise ValidationError(
                f"validation output already exists and will not be overwritten: "
                f"{destination}"
            ) from exc

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        publish_symlink()
        return
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(stage),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ValidationError(
            f"validation output already exists and will not be overwritten: {destination}"
        )
    if error in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        publish_symlink()
        return
    raise OSError(error, os.strerror(error), os.fspath(destination))


def write_stability_artifacts(
    out_dir: str | Path,
    results: Sequence[StabilityResult],
    verification: Sequence[BackendVerification],
    *,
    manifest: PeakSplitManifest,
    decision: DecisionSplitArtifact,
    validation: ValidationSplitArtifact,
    provenance: StabilityProvenance,
    lexicon: LexiconBinding,
) -> tuple[Path, Path]:
    """Atomically publish a coherent, fully-bound stability artifact pair."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    result_list, verification_list = _validated_stability_identity_inputs(
        results, verification, provenance, lexicon, manifest, decision, validation,
    )
    if any(row.decision_id != decision.decision_id for row in result_list):
        raise ValidationError("stability result decision_id does not match the bound split artifact")
    backend_identities = [(row.backend, row.backend_version) for row in verification_list]
    if len(set(backend_identities)) != len(backend_identities):
        raise ValidationError("backend verification rows must have unique backend identities")
    result_ids = [row.backend_result_id for row in result_list]
    if len(set(result_ids)) != len(result_ids):
        raise ValidationError("stability results must have unique backend_result_id values")
    verified = [row for row in verification_list if row.status == "VERIFIED"]
    unverified = [row for row in verification_list if row.status == "UNVERIFIED"]
    if not result_list and not unverified:
        raise ValidationError("empty stability artifacts require at least one UNVERIFIED backend row")
    if len(verified) != len(result_list):
        raise ValidationError("each VERIFIED backend row requires exactly one stability result")
    for row in result_list:
        matches = [entry for entry in verification_list if (
            entry.status == "VERIFIED" and entry.backend == row.backend
            and entry.backend_version == row.backend_version
            and entry.backend_result_id == row.backend_result_id
            and entry.lexicon_identity == row.lexicon_identity == lexicon.lexicon_identity
        )]
        if len(matches) != 1:
            raise ValidationError("every stability result must link to exactly one VERIFIED backend row")
    verified_ids = [entry.backend_result_id for entry in verified]
    if set(verified_ids) != set(result_ids) or len(set(verified_ids)) != len(verified_ids):
        raise ValidationError("every VERIFIED backend row must link to exactly one stability result")
    if any(entry.backend_result_id is not None for entry in unverified):
        raise ValidationError("UNVERIFIED backend rows cannot claim a stability result")
    if any(entry.lexicon_identity != lexicon.lexicon_identity for entry in verification_list):
        raise ValidationError("backend verification lexicon identity does not match the bound lexicons")
    artifact_id = stability_result_id(
        result_list, verification_list, provenance, lexicon, manifest, decision, validation
    )
    if validation.result_id != artifact_id:
        raise ValidationError(
            "validation split artifact result_id must equal the canonical stability artifact identity"
        )
    provenance_json = _canonical_json(
        provenance.to_dict(), what="stability provenance",
    )
    # The analysis mode travels with every emitted row. It is already committed
    # to by the artifact identity, but only inside a SHA-256 -- so an EXPLORATORY
    # run, whose "validation" peaks are permitted to be the decision peaks
    # themselves, produced a stability_results.parquet that no reader could tell
    # apart from a clean held-out one. AnalysisMode.EXPLORATORY exists precisely
    # to record nonconfirmatory reuse (assert_split_compatibility tells a caller
    # to declare it); a record that never reaches the artifact is a waiver, not a
    # record, so the declared mode is written in plain text beside the numbers it
    # qualifies.
    analysis_mode = validation.mode.value
    metadata = {
        b"motifmultiverse.artifact_id": artifact_id.encode(),
        b"motifmultiverse.schema_version": STABILITY_SCHEMA_VERSION.encode(),
        b"motifmultiverse.provenance": provenance_json.encode(),
        b"motifmultiverse.split_manifest_checksum": manifest.checksum.encode(),
        b"motifmultiverse.decision_artifact_id": decision.artifact_id.encode(),
        b"motifmultiverse.validation_artifact_id": validation.artifact_id.encode(),
        b"motifmultiverse.lexicon_identity": lexicon.lexicon_identity.encode(),
        b"motifmultiverse.analysis_mode": analysis_mode.encode(),
    }
    rows: list[dict[str, Any]] = []
    for result in result_list:
        row = result.to_dict()
        row["affected_interval"] = (
            None if result.affected_interval is None else json.dumps(result.affected_interval)
        )
        row.update({
            "artifact_id": artifact_id,
            "split_manifest_checksum": manifest.checksum,
            "decision_artifact_id": decision.artifact_id,
            "validation_artifact_id": validation.artifact_id,
            "lexicon_identity": lexicon.lexicon_identity,
            "analysis_mode": analysis_mode,
            "provenance": provenance_json,
        })
        rows.append(row)
    out = Path(out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(out):
        raise ValidationError(
            f"validation output already exists and will not be overwritten: {out}"
        )
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.stage-", dir=out.parent))
    result_path = stage / "stability_results.parquet"
    arrow_schema = pa.schema([
        pa.field("decision_id", pa.string()), pa.field("n_affected_peaks", pa.int64()),
        pa.field("n_affected_hits", pa.int64()), pa.field("family_coefficient_share", pa.float64()),
        pa.field("paired_delta_reconstruction_affected", pa.float64()),
        pa.field("paired_delta_reconstruction_all", pa.float64()), pa.field("hit_jaccard", pa.float64()),
        pa.field("coefficient_conservation", pa.float64()), pa.field("status", pa.string()),
        pa.field("power_statement", pa.string()), pa.field("affected_interval", pa.string()),
        pa.field("backend", pa.string()), pa.field("backend_version", pa.string()),
        pa.field("backend_result_id", pa.string()), pa.field("lexicon_identity", pa.string()),
        pa.field("schema_version", pa.string()), pa.field("artifact_id", pa.string()),
        pa.field("split_manifest_checksum", pa.string()), pa.field("decision_artifact_id", pa.string()),
        pa.field("validation_artifact_id", pa.string()), pa.field("analysis_mode", pa.string()),
        pa.field("provenance", pa.string()),
    ], metadata=metadata)
    try:
        table = pa.Table.from_pylist(rows, schema=arrow_schema)
        pq.write_table(table, result_path)
        verification_path = stage / "backend_verification.tsv"
        fields = ["backend", "backend_version", "status", "detail", "backend_result_id",
                  "lexicon_identity", "schema_version", "artifact_id", "split_manifest_checksum",
                  "decision_artifact_id", "validation_artifact_id", "analysis_mode", "provenance"]
        with verification_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for item in verification_list:
                writer.writerow({
                    **asdict(item), "artifact_id": artifact_id,
                    "split_manifest_checksum": manifest.checksum,
                    "decision_artifact_id": decision.artifact_id,
                    "validation_artifact_id": validation.artifact_id,
                    "analysis_mode": analysis_mode,
                    "provenance": provenance_json,
                })
        (stage / "provenance.json").write_text(
            json.dumps([provenance.to_dict()], indent=2, sort_keys=True), encoding="utf-8"
        )
        _publish_directory_noreplace(stage, out)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return out / "stability_results.parquet", out / "backend_verification.tsv"


class _FrozenHitTableBackend(StabilityBackend):
    name = "frozen-hit-table"
    version = STABILITY_SCHEMA_VERSION

    def __init__(
        self,
        before_path: str | Path,
        after_path: str | Path,
        validation_peak_ids: frozenset[str],
        expected_substrate_id: str | None = None,
    ) -> None:
        self.before_path = Path(before_path)
        self.after_path = Path(after_path)
        self.validation_peak_ids = validation_peak_ids
        self.expected_substrate_id = expected_substrate_id
        self.substrate_id: str | None = None

    def compare(self, lexicons: str | Path, decision_id: str) -> tuple[Any, Any]:
        import pandas as pd

        def read(path: Path):
            if not path.exists():
                raise ValidationError(f"frozen-hit-table input does not exist: {path}")
            try:
                return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, sep="\t")
            except (OSError, ValueError) as exc:
                raise ValidationError(f"frozen-hit-table cannot read {path}: {exc}") from exc

        before_raw = read(self.before_path)
        after_raw = read(self.after_path)

        def substrate_identity(frame: Any, label: str) -> str:
            if "substrate_id" not in frame.columns:
                raise ValidationError(
                    f"frozen-hit-table {label} rows must carry substrate_id"
                )
            values = list(frame["substrate_id"])
            if not values or any(not _is_sha256(value) for value in values):
                raise ValidationError(
                    f"frozen-hit-table {label} substrate_id values must be "
                    "lowercase SHA-256 digests"
                )
            identities = set(values)
            if len(identities) != 1:
                raise ValidationError(
                    f"frozen-hit-table {label} rows must belong to one substrate"
                )
            return next(iter(identities))

        before_substrate = substrate_identity(before_raw, "before")
        after_substrate = substrate_identity(after_raw, "after")
        if before_substrate != after_substrate:
            raise ValidationError(
                "frozen-hit-table before and after rows must use the same substrate"
            )
        if (
            self.expected_substrate_id is not None
            and before_substrate != self.expected_substrate_id
        ):
            raise ValidationError(
                "frozen-hit-table substrate_id does not match the validated "
                "substrate manifest"
            )
        before = normalize_backend_output(before_raw, backend=self.name)
        after = normalize_backend_output(after_raw, backend=self.name)
        for label, frame in (("before", before), ("after", after)):
            peak_ids = frozenset(str(value) for value in frame["peak_id"])
            if peak_ids != self.validation_peak_ids:
                raise ValidationError(
                    f"frozen-hit-table {label} rows must contain exactly the split-bound "
                    f"validation peaks; got {sorted(peak_ids)} expected "
                    f"{sorted(self.validation_peak_ids)}"
                )
        self.substrate_id = before_substrate
        return before, after


def _read_manifest(path: str | Path) -> PeakSplitManifest:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        assignments = {peak_id: SplitRole(role) for peak_id, role in payload["assignments"].items()}
        return PeakSplitManifest(
            schema_version=payload["schema_version"], assignments=assignments, checksum=payload["checksum"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{source} is not a valid peak split manifest: {exc}") from exc


def _read_artifact(path: str | Path, kind: str, manifest: PeakSplitManifest):
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        cls = DecisionSplitArtifact if kind == "decision" else ValidationSplitArtifact
        return cls.from_dict(payload, manifest=manifest)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{source} is not a valid {kind} split artifact: {exc}") from exc


def _bind_provenance_input(
    provenance: ProvenanceRecord,
    role: str,
    path: str | Path,
) -> None:
    if not isinstance(role, str) or not role.strip() or role in provenance.inputs:
        raise ValidationError(f"validation provenance input role is ambiguous: {role!r}")
    provenance.inputs[role] = sha256_file(path)


def run(
    lexicons: str | Path,
    out_dir: str | Path,
    *,
    before_hits: str | Path,
    after_hits: str | Path,
    substrate_manifest: str | Path,
    split_manifest: str | Path,
    decision_artifact: str | Path,
    validation_artifact: str | Path,
) -> tuple[tuple[StabilityResult, ...], tuple[BackendVerification, ...]]:
    """Validate frozen before/after hit tables under an exact split binding."""
    lexicon = load_lexicon_binding(lexicons)
    from motifmultiverse.substrate import SubstrateError
    from motifmultiverse.substrate import read_manifest as read_substrate_manifest

    try:
        substrate = read_substrate_manifest(substrate_manifest)
    except (OSError, SubstrateError) as exc:
        raise ValidationError(
            f"{substrate_manifest} is not a valid substrate manifest: {exc}"
        ) from exc
    lexicon_content_hashes = {entry[1] for entry in lexicon.entries}
    if substrate.caller_specification.lexicon_content_hash not in lexicon_content_hashes:
        raise ValidationError(
            "substrate manifest caller lexicon identity does not match any bound "
            "compiled lexicon"
        )
    manifest = _read_manifest(split_manifest)
    decision = _read_artifact(decision_artifact, "decision", manifest)
    validation = _read_artifact(validation_artifact, "validation", manifest)
    assert_artifact_split_compatibility(manifest, decision, validation)
    provenance_record = record("validate")
    try:
        for role, source in (
            ("before_hits", before_hits),
            ("after_hits", after_hits),
            ("substrate_manifest", substrate_manifest),
            ("split_manifest", split_manifest),
            ("decision_artifact", decision_artifact),
        ):
            _bind_provenance_input(provenance_record, role, source)
        for manifest_path in sorted(Path(lexicons).glob("*.manifest.json")):
            tier = manifest_path.name.removesuffix(".manifest.json")
            _bind_provenance_input(
                provenance_record, f"lexicon_manifest:{tier}", manifest_path,
            )
            _bind_provenance_input(
                provenance_record,
                f"lexicon_h5:{tier}",
                Path(lexicons) / f"{tier}.h5",
            )
    except OSError:
        raise
    backend = _FrozenHitTableBackend(
        before_hits,
        after_hits,
        validation.validation_peak_ids,
        expected_substrate_id=substrate.substrate_id,
    )
    results, verification = run_backend_validation(
        lexicon,
        decision.decision_id,
        [backend],
    )
    if backend.substrate_id != substrate.substrate_id:
        raise ValidationError(
            "validated frozen-hit-table substrate identity was not established"
        )
    # The raw validation artifact contains result_id, which points back to the
    # stability identity being computed.  Exclude exactly those self-referential
    # bytes; bind its complete non-circular split semantics instead.
    provenance_record.inputs["validation_split_binding"] = _validation_split_identity(
        manifest, decision, validation,
    )
    provenance_record.input_scale = len(validation.validation_peak_ids)
    provenance_record.substrate_id = substrate.substrate_id
    provenance = StabilityProvenance.from_record(
        provenance_record,
        lexicon=lexicon,
        manifest=manifest,
        decision=decision,
        validation=validation,
    )
    expected_result_id = stability_result_id(
        results, verification, provenance, lexicon, manifest, decision, validation,
    )
    if validation.result_id == "pending":
        validation = replace(validation, result_id=expected_result_id, artifact_id="")
    write_stability_artifacts(
        out_dir,
        results,
        verification,
        manifest=manifest,
        decision=decision,
        validation=validation,
        provenance=provenance,
        lexicon=lexicon,
    )
    return results, verification
