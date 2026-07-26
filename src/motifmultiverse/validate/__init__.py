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
from dataclasses import asdict, dataclass, fields, replace
from numbers import Real
from pathlib import Path
from typing import Any

from motifmultiverse.provenance import record, sha256_file
from motifmultiverse.schema import (
    LexiconManifest,
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
    "STABILITY_SCHEMA_VERSION", "StabilityBackend", "StabilityResult", "SplitRole",
    "ValidationError", "ValidationSplitArtifact", "assert_artifact_split_compatibility",
    "assert_cross_fit_compatibility", "assert_split_compatibility",
    "build_peak_split_manifest", "evaluate_stability", "load_lexicon_binding",
    "normalize_backend_output", "peak_split_manifest_checksum", "run_backend_validation", "run",
    "stability_result_id",
    "write_stability_artifacts",
]

STABILITY_SCHEMA_VERSION = "1"
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
    import h5py
    import numpy as np

    from motifmultiverse.compile import _content_hash

    entry_rows: list[tuple[str, str, str, str]] = []
    manifest_fields = {item.name for item in fields(LexiconManifest)}
    seen_tiers: set[str] = set()
    for manifest_path in manifest_paths:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError("manifest must be an object")
            manifest = LexiconManifest(**{key: value for key, value in payload.items() if key in manifest_fields})
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValidationError(f"{manifest_path} is not a valid compiled lexicon manifest: {exc}") from exc
        if (
            not isinstance(manifest.lexicon_content_hash, str)
            or len(manifest.lexicon_content_hash) != 64
            or any(char not in "0123456789abcdef" for char in manifest.lexicon_content_hash)
        ):
            raise ValidationError(f"{manifest_path} has no valid lexicon_content_hash")
        if manifest_path.name != f"{manifest.tier}.manifest.json":
            raise ValidationError(f"{manifest_path} filename must exactly match declared tier {manifest.tier!r}")
        if manifest.tier in seen_tiers:
            raise ValidationError(f"compiled lexicons contain duplicate tier manifest {manifest.tier!r}")
        seen_tiers.add(manifest.tier)
        h5_path = root / f"{manifest.tier}.h5"
        if not h5_path.is_file():
            raise ValidationError(f"compiled lexicon named by {manifest_path} is missing: {h5_path}")
        try:
            index = payload["index"]
            if not isinstance(index, list) or len(index) != manifest.n_motifs:
                raise TypeError("manifest index must describe every motif")
            if [row.get("pattern_tag") for row in index] != manifest.pattern_order:
                raise TypeError("manifest index pattern order is not authoritative")
            if [row.get("node_id") for row in index] != manifest.node_ids:
                raise TypeError("manifest index node order is not authoritative")
            ordered = []
            arrays: dict[str, dict[str, Any]] = {}
            with h5py.File(h5_path, "r") as h5:
                for row in index:
                    group, pattern = str(row["pattern_tag"]).split(".", 1)
                    if group not in h5 or pattern not in h5[group]:
                        raise TypeError(f"missing compiled motif {group}.{pattern}")
                    motif = h5[group][pattern]
                    if "contrib_scores" not in motif:
                        raise TypeError(f"compiled motif {group}.{pattern} lacks contrib_scores")
                    arrays[str(row["node_id"])] = {
                        "cwm": np.asarray(motif["contrib_scores"]),
                        **({"hypothetical_cwm": np.asarray(motif["hypothetical_contribs"])}
                           if "hypothetical_contribs" in motif else {}),
                        **({"ppm": np.asarray(motif["sequence"])} if "sequence" in motif else {}),
                    }
                    ordered.append((group, pattern, {"node_id": str(row["node_id"])}))
            recomputed = _content_hash(
                ordered, arrays, schema_version=manifest.schema_version,
                trim_threshold=manifest.trim_threshold, motif_type=manifest.motif_type,
                include_rc=manifest.include_rc, loader_backend=manifest.loader_backend,
                loader_parameters=manifest.loader_parameters,
            )
        except (OSError, TypeError, ValueError, KeyError, IndexError) as exc:
            raise ValidationError(f"{h5_path} is not the verified compiled lexicon named by {manifest_path}: {exc}") from exc
        if recomputed != manifest.lexicon_content_hash:
            raise ValidationError(f"{manifest_path} lexicon_content_hash does not match HDF5 content and loader semantics")
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
    affected = sorted({
        peak
        for peak, hit_id in set(before_coefficients) | set(after_coefficients)
        if before_coefficients.get((peak, hit_id)) != after_coefficients.get((peak, hit_id))
    })
    all_delta = [after_reconstruction[peak] - before_reconstruction[peak]
                 for peak in sorted(before_reconstruction)]
    affected_delta = [after_reconstruction[peak] - before_reconstruction[peak] for peak in affected]
    affected_keys_before = {key for key in before_coefficients if key[0] in set(affected)}
    affected_keys_after = {key for key in after_coefficients if key[0] in set(affected)}
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


def stability_result_id(
    results: Sequence[StabilityResult],
    verification: Sequence[BackendVerification],
    provenance: Mapping[str, Any],
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
    payload = {
        "lexicon": lexicon.to_dict(),
        "provenance": dict(provenance),
        "results": [row.to_dict() for row in sorted(results, key=lambda row: row.backend_result_id)],
        "schema_version": STABILITY_SCHEMA_VERSION,
        "split_binding": _split_binding(manifest, decision, validation),
        "verification": [asdict(row) for row in sorted(
            verification, key=lambda row: (row.backend, row.backend_version, row.status, row.detail)
        )],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"stability:{digest}"


def write_stability_artifacts(
    out_dir: str | Path,
    results: Sequence[StabilityResult],
    verification: Sequence[BackendVerification],
    *,
    manifest: PeakSplitManifest,
    decision: DecisionSplitArtifact,
    validation: ValidationSplitArtifact,
    provenance: Mapping[str, Any],
    lexicon: LexiconBinding,
) -> tuple[Path, Path]:
    """Atomically publish a coherent, fully-bound stability artifact pair."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    assert_artifact_split_compatibility(manifest, decision, validation)
    if not isinstance(lexicon, LexiconBinding):
        raise ValidationError("stability artifacts require a validated LexiconBinding")
    result_list = tuple(sorted(results, key=lambda row: json.dumps(
        row.to_dict(), sort_keys=True, separators=(",", ":")
    )))
    verification_list = tuple(sorted(verification, key=lambda row: (
        row.backend, row.backend_version, row.status, row.detail,
    )))
    if any(row.decision_id != decision.decision_id for row in result_list):
        raise ValidationError("stability result decision_id does not match the bound split artifact")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValidationError("stability artifacts require provenance")
    for row in result_list:
        matches = [entry for entry in verification_list if (
            entry.status == "VERIFIED" and entry.backend == row.backend
            and entry.backend_version == row.backend_version
            and entry.backend_result_id == row.backend_result_id
            and entry.lexicon_identity == row.lexicon_identity == lexicon.lexicon_identity
        )]
        if len(matches) != 1:
            raise ValidationError("every stability result must link to exactly one VERIFIED backend row")
    verified_ids = {entry.backend_result_id for entry in verification_list if entry.status == "VERIFIED"}
    if verified_ids != {row.backend_result_id for row in result_list}:
        raise ValidationError("every VERIFIED backend row must link to exactly one stability result")
    if any(entry.lexicon_identity != lexicon.lexicon_identity for entry in verification_list):
        raise ValidationError("backend verification lexicon identity does not match the bound lexicons")
    artifact_id = stability_result_id(
        result_list, verification_list, provenance, lexicon, manifest, decision, validation
    )
    if validation.result_id != artifact_id:
        raise ValidationError(
            "validation split artifact result_id must equal the canonical stability artifact identity"
        )
    metadata = {
        b"motifmultiverse.artifact_id": artifact_id.encode(),
        b"motifmultiverse.schema_version": STABILITY_SCHEMA_VERSION.encode(),
        b"motifmultiverse.provenance": json.dumps(
            dict(provenance), sort_keys=True, separators=(",", ":")
        ).encode(),
        b"motifmultiverse.split_manifest_checksum": manifest.checksum.encode(),
        b"motifmultiverse.decision_artifact_id": decision.artifact_id.encode(),
        b"motifmultiverse.validation_artifact_id": validation.artifact_id.encode(),
        b"motifmultiverse.lexicon_identity": lexicon.lexicon_identity.encode(),
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
            "provenance": json.dumps(dict(provenance), sort_keys=True, separators=(",", ":")),
        })
        rows.append(row)
    out = Path(out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
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
        pa.field("validation_artifact_id", pa.string()), pa.field("provenance", pa.string()),
    ], metadata=metadata)
    try:
        table = pa.Table.from_pylist(rows, schema=arrow_schema)
        pq.write_table(table, result_path)
        verification_path = stage / "backend_verification.tsv"
        fields = ["backend", "backend_version", "status", "detail", "backend_result_id",
                  "lexicon_identity", "schema_version", "artifact_id", "split_manifest_checksum",
                  "decision_artifact_id", "validation_artifact_id", "provenance"]
        with verification_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for item in verification_list:
                writer.writerow({
                    **asdict(item), "artifact_id": artifact_id,
                    "split_manifest_checksum": manifest.checksum,
                    "decision_artifact_id": decision.artifact_id,
                    "validation_artifact_id": validation.artifact_id,
                    "provenance": json.dumps(dict(provenance), sort_keys=True, separators=(",", ":")),
                })
        (stage / "provenance.json").write_text(
            json.dumps([dict(provenance)], indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(stage, out)
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
    ) -> None:
        self.before_path = Path(before_path)
        self.after_path = Path(after_path)
        self.validation_peak_ids = validation_peak_ids

    def compare(self, lexicons: str | Path, decision_id: str) -> tuple[Any, Any]:
        import pandas as pd

        def read(path: Path):
            if not path.exists():
                raise ValidationError(f"frozen-hit-table input does not exist: {path}")
            try:
                return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, sep="\t")
            except (OSError, ValueError) as exc:
                raise ValidationError(f"frozen-hit-table cannot read {path}: {exc}") from exc

        before = normalize_backend_output(read(self.before_path), backend=self.name)
        after = normalize_backend_output(read(self.after_path), backend=self.name)
        for label, frame in (("before", before), ("after", after)):
            peak_ids = frozenset(str(value) for value in frame["peak_id"])
            if peak_ids != self.validation_peak_ids:
                raise ValidationError(
                    f"frozen-hit-table {label} rows must contain exactly the split-bound "
                    f"validation peaks; got {sorted(peak_ids)} expected "
                    f"{sorted(self.validation_peak_ids)}"
                )
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


def run(
    lexicons: str | Path,
    out_dir: str | Path,
    *,
    before_hits: str | Path,
    after_hits: str | Path,
    split_manifest: str | Path,
    decision_artifact: str | Path,
    validation_artifact: str | Path,
) -> tuple[tuple[StabilityResult, ...], tuple[BackendVerification, ...]]:
    """Validate frozen before/after hit tables under an exact split binding."""
    lexicon = load_lexicon_binding(lexicons)
    manifest = _read_manifest(split_manifest)
    decision = _read_artifact(decision_artifact, "decision", manifest)
    validation = _read_artifact(validation_artifact, "validation", manifest)
    assert_artifact_split_compatibility(manifest, decision, validation)
    provenance_record = record("validate")
    try:
        for source in (before_hits, after_hits, split_manifest, decision_artifact):
            provenance_record.add_input(source)
        for manifest_path in sorted(Path(lexicons).glob("*.manifest.json")):
            provenance_record.add_input(manifest_path)
            provenance_record.add_input(Path(lexicons) / f"{manifest_path.name.removesuffix('.manifest.json')}.h5")
    except OSError:
        raise
    results, verification = run_backend_validation(
        lexicon,
        decision.decision_id,
        [_FrozenHitTableBackend(before_hits, after_hits, validation.validation_peak_ids)],
    )
    # The validation artifact's raw bytes include its required output identity;
    # hashing those bytes here would create a circular identity.  Its complete
    # split semantics are instead bound below and fingerprinted as provenance.
    validation_binding_digest = hashlib.sha256(json.dumps(
        _split_binding(manifest, decision, validation), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    provenance = {
        "stage": "validate",
        "inputs": {**dict(provenance_record.inputs), "validation_split_binding": validation_binding_digest},
        "software": dict(provenance_record.software),
        "lexicon": lexicon.to_dict(),
    }
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
