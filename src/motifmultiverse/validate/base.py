"""Split-bound contracts shared by downstream decision and validation stages."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from motifmultiverse.schema import PeakSplitManifest, SchemaError, SplitRole

__all__ = [
    "AnalysisMode",
    "CrossFitFold",
    "DecisionSplitArtifact",
    "SPLIT_ARTIFACT_SCHEMA_VERSION",
    "ValidationSplitArtifact",
    "assert_split_compatibility",
    "assert_cross_fit_compatibility",
    "assert_artifact_split_compatibility",
]


class AnalysisMode(StrEnum):
    """Primary analysis, explicitly nonconfirmatory exploration, or cross-fitting."""

    PRIMARY = "PRIMARY"
    EXPLORATORY = "EXPLORATORY"
    CROSS_FIT = "CROSS_FIT"


SPLIT_ARTIFACT_SCHEMA_VERSION = "1"


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_peak_ids(name: str, peak_ids: set[str] | frozenset[str]) -> None:
    if not isinstance(peak_ids, (set, frozenset)) or not peak_ids:
        raise SchemaError(f"{name} must be a non-empty set of peak IDs")
    if any(not _is_nonempty_string(peak_id) for peak_id in peak_ids):
        raise SchemaError(f"{name} must contain only non-empty string peak IDs")


def _require_manifest_members(
    manifest: PeakSplitManifest, peak_ids: Iterable[str], name: str
) -> None:
    unknown = sorted(set(peak_ids) - set(manifest.assignments))
    if unknown:
        raise SchemaError(f"{name} contains peak IDs absent from the split manifest: {unknown}")


def _require_checksum(name: str, checksum: str) -> None:
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise SchemaError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CrossFitFold:
    """One held-out fold with its own decision/training and evaluation peak sets."""

    fold_id: str
    decision_peak_ids: frozenset[str]
    evaluation_peak_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.fold_id, str) or not self.fold_id.strip():
            raise SchemaError("cross-fit fold_id must be a non-empty string")
        _require_peak_ids("cross-fit decision_peak_ids", self.decision_peak_ids)
        _require_peak_ids("cross-fit evaluation_peak_ids", self.evaluation_peak_ids)
        if self.decision_peak_ids & self.evaluation_peak_ids:
            raise SchemaError("a cross-fit fold cannot evaluate peaks used for its decision")


def _canonical_artifact_id(
    kind: str,
    *,
    decision_id: str,
    result_id: str | None,
    manifest_checksum: str,
    mode: AnalysisMode,
    folds: tuple[CrossFitFold, ...],
    decision_peak_ids: frozenset[str],
    validation_peak_ids: frozenset[str],
) -> str:
    """Content identity for a complete split-bound decision or validation artifact."""
    payload = {
        "cross_fit_folds": [
            {
                "decision_peak_ids": sorted(fold.decision_peak_ids),
                "evaluation_peak_ids": sorted(fold.evaluation_peak_ids),
                "fold_id": fold.fold_id,
            }
            for fold in sorted(folds, key=lambda item: item.fold_id)
        ],
        "decision_id": decision_id,
        "decision_peak_ids": sorted(decision_peak_ids),
        "kind": kind,
        "mode": mode.value,
        "result_id": result_id,
        "schema_version": SPLIT_ARTIFACT_SCHEMA_VERSION,
        "split_manifest_checksum": manifest_checksum,
        "validation_peak_ids": sorted(validation_peak_ids),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{kind}-split:{digest}"


def _require_artifact_id(kind: str, artifact_id: object, expected: str) -> None:
    if not isinstance(artifact_id, str) or artifact_id != expected:
        raise SchemaError(f"{kind} split artifact artifact_id does not match its content")


def _canonical_folds(kind: str, folds: tuple[CrossFitFold, ...]) -> tuple[CrossFitFold, ...]:
    if not isinstance(folds, tuple) or any(not isinstance(fold, CrossFitFold) for fold in folds):
        raise SchemaError(f"{kind} split artifact cross_fit_folds must be CrossFitFold values")
    return tuple(sorted(folds, key=lambda fold: fold.fold_id))


def _validate_artifact_fields(
    kind: str,
    *,
    manifest: PeakSplitManifest,
    decision_id: str,
    result_id: str | None,
    decision_peak_ids: frozenset[str],
    validation_peak_ids: frozenset[str],
    checksum: str,
    mode: AnalysisMode,
    folds: tuple[CrossFitFold, ...],
    artifact_id: str,
) -> str:
    if not isinstance(manifest, PeakSplitManifest):
        raise SchemaError(f"{kind} split artifact requires a PeakSplitManifest")
    if not _is_nonempty_string(decision_id):
        raise SchemaError(f"{kind} split artifact requires a non-empty decision_id")
    if result_id is not None and not _is_nonempty_string(result_id):
        raise SchemaError(f"{kind} split artifact result_id must be a non-empty string")
    _require_peak_ids(f"{kind} split artifact decision_peak_ids", decision_peak_ids)
    _require_peak_ids(f"{kind} split artifact validation_peak_ids", validation_peak_ids)
    _require_checksum(f"{kind} split artifact split_manifest_checksum", checksum)
    if checksum != manifest.checksum:
        raise SchemaError(f"{kind} split artifact checksum does not match the split manifest")
    if not isinstance(mode, AnalysisMode):
        raise SchemaError(f"{kind} split artifact mode must be an AnalysisMode")
    _canonical_folds(kind, folds)
    if mode is AnalysisMode.CROSS_FIT:
        if not folds:
            raise SchemaError(f"{kind} cross-fit artifact requires fold-specific memberships")
        if decision_peak_ids != validation_peak_ids:
            raise SchemaError(
                f"{kind} cross-fit artifact must carry the same complete decision and validation peak set"
            )
        assert_cross_fit_compatibility(manifest, decision_peak_ids, folds)
    else:
        if folds:
            raise SchemaError(f"{kind} artifact cannot carry cross-fit folds outside CROSS_FIT mode")
        assert_split_compatibility(
            manifest, decision_peak_ids, validation_peak_ids, mode=mode
        )
    expected = _canonical_artifact_id(
        kind,
        decision_id=decision_id,
        result_id=result_id,
        manifest_checksum=checksum,
        mode=mode,
        folds=folds,
        decision_peak_ids=decision_peak_ids,
        validation_peak_ids=validation_peak_ids,
    )
    if artifact_id:
        _require_artifact_id(kind, artifact_id, expected)
    return expected


def _fold_payload(folds: tuple[CrossFitFold, ...]) -> list[dict[str, object]]:
    return [
        {
            "fold_id": fold.fold_id,
            "decision_peak_ids": sorted(fold.decision_peak_ids),
            "evaluation_peak_ids": sorted(fold.evaluation_peak_ids),
        }
        for fold in folds
    ]


def _folds_from_payload(value: object) -> tuple[CrossFitFold, ...]:
    if not isinstance(value, list):
        raise SchemaError("split artifact cross_fit_folds must be a list")
    folds: list[CrossFitFold] = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
            "fold_id", "decision_peak_ids", "evaluation_peak_ids"
        }:
            raise SchemaError("split artifact fold metadata is malformed")
        decision_ids = row["decision_peak_ids"]
        evaluation_ids = row["evaluation_peak_ids"]
        if not isinstance(decision_ids, list) or not isinstance(evaluation_ids, list):
            raise SchemaError("split artifact fold peak IDs must be lists")
        folds.append(CrossFitFold(
            row["fold_id"], frozenset(decision_ids), frozenset(evaluation_ids)
        ))
    return tuple(folds)


def _artifact_payload(
    kind: str,
    artifact_id: str,
    manifest_checksum: str,
    decision_id: str,
    result_id: str | None,
    mode: AnalysisMode,
    decision_peak_ids: frozenset[str],
    validation_peak_ids: frozenset[str],
    folds: tuple[CrossFitFold, ...],
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "cross_fit_folds": _fold_payload(folds),
        "decision_id": decision_id,
        "decision_peak_ids": sorted(decision_peak_ids),
        "kind": kind,
        "mode": mode.value,
        "result_id": result_id,
        "schema_version": SPLIT_ARTIFACT_SCHEMA_VERSION,
        "split_manifest_checksum": manifest_checksum,
        "validation_peak_ids": sorted(validation_peak_ids),
    }


@dataclass(frozen=True)
class DecisionSplitArtifact:
    """A self-validating, manifest-bound downstream decision split artifact."""

    manifest: PeakSplitManifest
    decision_id: str
    decision_peak_ids: frozenset[str]
    validation_peak_ids: frozenset[str]
    split_manifest_checksum: str = ""
    mode: AnalysisMode = AnalysisMode.PRIMARY
    cross_fit_folds: tuple[CrossFitFold, ...] = ()
    artifact_id: str = ""
    schema_version: str = SPLIT_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPLIT_ARTIFACT_SCHEMA_VERSION:
            raise SchemaError(
                f"decision split artifact schema_version must be {SPLIT_ARTIFACT_SCHEMA_VERSION!r}"
            )
        checksum = (
            getattr(self.manifest, "checksum", None)
            if self.split_manifest_checksum == ""
            else self.split_manifest_checksum
        )
        folds = _canonical_folds("decision", self.cross_fit_folds)
        expected = _validate_artifact_fields(
            "decision",
            manifest=self.manifest,
            decision_id=self.decision_id,
            result_id=None,
            decision_peak_ids=self.decision_peak_ids,
            validation_peak_ids=self.validation_peak_ids,
            checksum=checksum,
            mode=self.mode,
            folds=folds,
            artifact_id=self.artifact_id,
        )
        object.__setattr__(self, "split_manifest_checksum", checksum)
        object.__setattr__(self, "cross_fit_folds", folds)
        object.__setattr__(self, "artifact_id", expected)

    @classmethod
    def create(cls, **kwargs: Any) -> DecisionSplitArtifact:
        return cls(**kwargs)

    def validate(self) -> None:
        _validate_artifact_fields(
            "decision", manifest=self.manifest, decision_id=self.decision_id,
            result_id=None, decision_peak_ids=self.decision_peak_ids,
            validation_peak_ids=self.validation_peak_ids,
            checksum=self.split_manifest_checksum, mode=self.mode,
            folds=self.cross_fit_folds, artifact_id=self.artifact_id,
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return _artifact_payload(
            "decision", self.artifact_id, self.split_manifest_checksum, self.decision_id,
            None, self.mode, self.decision_peak_ids, self.validation_peak_ids,
            self.cross_fit_folds,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], *, manifest: PeakSplitManifest) -> DecisionSplitArtifact:
        expected_keys = {
            "artifact_id", "cross_fit_folds", "decision_id", "decision_peak_ids", "kind",
            "mode", "result_id", "schema_version", "split_manifest_checksum", "validation_peak_ids",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected_keys or payload.get("kind") != "decision":
            raise SchemaError("decision split artifact payload is malformed")
        if payload["result_id"] is not None or not isinstance(payload["decision_peak_ids"], list) or not isinstance(payload["validation_peak_ids"], list):
            raise SchemaError("decision split artifact payload has invalid identity fields")
        _require_checksum("decision split artifact payload split_manifest_checksum", payload["split_manifest_checksum"])
        try:
            mode = AnalysisMode(payload["mode"])
        except (TypeError, ValueError) as exc:
            raise SchemaError("decision split artifact payload has an invalid mode") from exc
        return cls(
            manifest=manifest, decision_id=payload["decision_id"],
            decision_peak_ids=frozenset(payload["decision_peak_ids"]),
            validation_peak_ids=frozenset(payload["validation_peak_ids"]),
            split_manifest_checksum=payload["split_manifest_checksum"], mode=mode,
            cross_fit_folds=_folds_from_payload(payload["cross_fit_folds"]),
            artifact_id=payload["artifact_id"], schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class ValidationSplitArtifact:
    """A self-validating, manifest-bound downstream validation split artifact."""

    manifest: PeakSplitManifest
    decision_id: str
    result_id: str
    decision_peak_ids: frozenset[str]
    validation_peak_ids: frozenset[str]
    split_manifest_checksum: str = ""
    mode: AnalysisMode = AnalysisMode.PRIMARY
    cross_fit_folds: tuple[CrossFitFold, ...] = ()
    artifact_id: str = ""
    schema_version: str = SPLIT_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPLIT_ARTIFACT_SCHEMA_VERSION:
            raise SchemaError(
                f"validation split artifact schema_version must be {SPLIT_ARTIFACT_SCHEMA_VERSION!r}"
            )
        checksum = (
            getattr(self.manifest, "checksum", None)
            if self.split_manifest_checksum == ""
            else self.split_manifest_checksum
        )
        folds = _canonical_folds("validation", self.cross_fit_folds)
        expected = _validate_artifact_fields(
            "validation",
            manifest=self.manifest,
            decision_id=self.decision_id,
            result_id=self.result_id,
            decision_peak_ids=self.decision_peak_ids,
            validation_peak_ids=self.validation_peak_ids,
            checksum=checksum,
            mode=self.mode,
            folds=folds,
            artifact_id=self.artifact_id,
        )
        object.__setattr__(self, "split_manifest_checksum", checksum)
        object.__setattr__(self, "cross_fit_folds", folds)
        object.__setattr__(self, "artifact_id", expected)

    @classmethod
    def create(cls, **kwargs: Any) -> ValidationSplitArtifact:
        return cls(**kwargs)

    def validate(self) -> None:
        _validate_artifact_fields(
            "validation", manifest=self.manifest, decision_id=self.decision_id,
            result_id=self.result_id, decision_peak_ids=self.decision_peak_ids,
            validation_peak_ids=self.validation_peak_ids,
            checksum=self.split_manifest_checksum, mode=self.mode,
            folds=self.cross_fit_folds, artifact_id=self.artifact_id,
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return _artifact_payload(
            "validation", self.artifact_id, self.split_manifest_checksum, self.decision_id,
            self.result_id, self.mode, self.decision_peak_ids, self.validation_peak_ids,
            self.cross_fit_folds,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], *, manifest: PeakSplitManifest) -> ValidationSplitArtifact:
        expected_keys = {
            "artifact_id", "cross_fit_folds", "decision_id", "decision_peak_ids", "kind",
            "mode", "result_id", "schema_version", "split_manifest_checksum", "validation_peak_ids",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected_keys or payload.get("kind") != "validation":
            raise SchemaError("validation split artifact payload is malformed")
        if not _is_nonempty_string(payload["result_id"]) or not isinstance(payload["decision_peak_ids"], list) or not isinstance(payload["validation_peak_ids"], list):
            raise SchemaError("validation split artifact payload has invalid identity fields")
        _require_checksum("validation split artifact payload split_manifest_checksum", payload["split_manifest_checksum"])
        try:
            mode = AnalysisMode(payload["mode"])
        except (TypeError, ValueError) as exc:
            raise SchemaError("validation split artifact payload has an invalid mode") from exc
        return cls(
            manifest=manifest, decision_id=payload["decision_id"], result_id=payload["result_id"],
            decision_peak_ids=frozenset(payload["decision_peak_ids"]),
            validation_peak_ids=frozenset(payload["validation_peak_ids"]),
            split_manifest_checksum=payload["split_manifest_checksum"], mode=mode,
            cross_fit_folds=_folds_from_payload(payload["cross_fit_folds"]),
            artifact_id=payload["artifact_id"], schema_version=payload["schema_version"],
        )


def assert_split_compatibility(
    manifest: PeakSplitManifest,
    decision_peak_ids: set[str] | frozenset[str],
    validation_peak_ids: set[str] | frozenset[str],
    *,
    mode: AnalysisMode = AnalysisMode.PRIMARY,
) -> None:
    """Refuse primary decision/validation reuse and role-contaminated inputs."""
    if not isinstance(manifest, PeakSplitManifest):
        raise SchemaError("split compatibility requires a PeakSplitManifest")
    _require_peak_ids("decision_peak_ids", decision_peak_ids)
    _require_peak_ids("validation_peak_ids", validation_peak_ids)
    if not isinstance(mode, AnalysisMode):
        raise SchemaError("split compatibility mode must be an AnalysisMode")
    if mode is AnalysisMode.CROSS_FIT:
        raise SchemaError("CROSS_FIT requires fold-specific memberships; use assert_cross_fit_compatibility")
    _require_manifest_members(manifest, decision_peak_ids, "decision_peak_ids")
    _require_manifest_members(manifest, validation_peak_ids, "validation_peak_ids")
    if mode is AnalysisMode.EXPLORATORY:
        return

    overlap = sorted(decision_peak_ids & validation_peak_ids)
    if overlap:
        raise SchemaError(
            "primary decision and validation peak sets overlap; declare EXPLORATORY "
            f"mode to record nonconfirmatory reuse: {overlap}"
        )
    decision_roles = {manifest.assignments[peak_id] for peak_id in decision_peak_ids}
    validation_roles = {manifest.assignments[peak_id] for peak_id in validation_peak_ids}
    if disallowed := decision_roles - {SplitRole.DISCOVERY, SplitRole.ADJUDICATION}:
        raise SchemaError(f"primary decision uses validation/inference role(s): {sorted(disallowed)}")
    if disallowed := validation_roles - {SplitRole.VALIDATION}:
        raise SchemaError(f"primary validation uses non-validation role(s): {sorted(disallowed)}")


def assert_cross_fit_compatibility(
    manifest: PeakSplitManifest,
    analysis_peak_ids: frozenset[str],
    folds: tuple[CrossFitFold, ...],
) -> None:
    """Validate a genuine cross-fit partition, not a relabelled reused split."""
    if not isinstance(manifest, PeakSplitManifest):
        raise SchemaError("cross-fit compatibility requires a PeakSplitManifest")
    _require_peak_ids("cross-fit analysis_peak_ids", analysis_peak_ids)
    _require_manifest_members(manifest, analysis_peak_ids, "cross-fit analysis_peak_ids")
    if not isinstance(folds, tuple) or len(folds) < 2:
        raise SchemaError("cross-fit requires at least two fold-specific memberships")
    if any(not isinstance(fold, CrossFitFold) for fold in folds):
        raise SchemaError("cross-fit folds must be CrossFitFold values")
    fold_ids = [fold.fold_id for fold in folds]
    if len(set(fold_ids)) != len(fold_ids):
        raise SchemaError("cross-fit fold IDs must be unique")

    evaluation_counts = {peak_id: 0 for peak_id in analysis_peak_ids}
    for fold in folds:
        _require_manifest_members(manifest, fold.decision_peak_ids, "cross-fit decision_peak_ids")
        _require_manifest_members(manifest, fold.evaluation_peak_ids, "cross-fit evaluation_peak_ids")
        extra = (fold.decision_peak_ids | fold.evaluation_peak_ids) - analysis_peak_ids
        if extra:
            raise SchemaError(f"cross-fit fold {fold.fold_id!r} includes peaks outside analysis: {sorted(extra)}")
        expected_decision = analysis_peak_ids - fold.evaluation_peak_ids
        if fold.decision_peak_ids != expected_decision:
            raise SchemaError(
                f"cross-fit fold {fold.fold_id!r} must record all and only the "
                "complementary decision peaks"
            )
        for peak_id in fold.evaluation_peak_ids:
            evaluation_counts[peak_id] += 1
    duplicate = sorted(peak_id for peak_id, count in evaluation_counts.items() if count > 1)
    if duplicate:
        raise SchemaError(f"cross-fit evaluation membership is duplicate for peak IDs: {duplicate}")
    missing = sorted(peak_id for peak_id, count in evaluation_counts.items() if count == 0)
    if missing:
        raise SchemaError(f"cross-fit evaluation membership is missing for peak IDs: {missing}")


def assert_artifact_split_compatibility(
    manifest: PeakSplitManifest,
    decision: DecisionSplitArtifact,
    validation: ValidationSplitArtifact,
) -> None:
    """Bind paired output contracts to the exact frozen split manifest."""
    if not isinstance(manifest, PeakSplitManifest):
        raise SchemaError("artifact split compatibility requires a PeakSplitManifest")
    if not isinstance(decision, DecisionSplitArtifact) or not isinstance(validation, ValidationSplitArtifact):
        raise SchemaError("artifact split compatibility requires decision and validation split artifacts")
    decision.validate()
    validation.validate()
    if decision.decision_id != validation.decision_id:
        raise SchemaError("decision and validation split artifacts must name the same decision_id")
    if decision.split_manifest_checksum != manifest.checksum:
        raise SchemaError("decision split artifact checksum does not match the split manifest")
    if validation.split_manifest_checksum != manifest.checksum:
        raise SchemaError("validation split artifact checksum does not match the split manifest")
    if decision.mode is not validation.mode:
        raise SchemaError("decision and validation split artifacts must declare the same analysis mode")
    if (
        decision.decision_peak_ids != validation.decision_peak_ids
        or decision.validation_peak_ids != validation.validation_peak_ids
        or decision.cross_fit_folds != validation.cross_fit_folds
    ):
        raise SchemaError("decision and validation artifacts must carry the same split semantics")
