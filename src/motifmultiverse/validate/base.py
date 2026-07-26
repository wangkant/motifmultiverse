"""Split-bound contracts shared by downstream decision and validation stages."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from motifmultiverse.schema import PeakSplitManifest, SchemaError, SplitRole

__all__ = [
    "AnalysisMode",
    "CrossFitFold",
    "DecisionSplitArtifact",
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


@dataclass(frozen=True)
class DecisionSplitArtifact:
    """The split provenance carried by one downstream-aware decision artifact."""

    decision_id: str
    peak_ids: frozenset[str]
    split_manifest_checksum: str
    mode: AnalysisMode = AnalysisMode.PRIMARY
    cross_fit_folds: tuple[CrossFitFold, ...] = ()

    def __post_init__(self) -> None:
        _validate_artifact_fields(
            "decision", self.decision_id, self.peak_ids, self.split_manifest_checksum,
            self.mode, self.cross_fit_folds,
        )


@dataclass(frozen=True)
class ValidationSplitArtifact:
    """The split provenance carried by one validation artifact for a decision."""

    decision_id: str
    peak_ids: frozenset[str]
    split_manifest_checksum: str
    mode: AnalysisMode = AnalysisMode.PRIMARY
    cross_fit_folds: tuple[CrossFitFold, ...] = ()

    def __post_init__(self) -> None:
        _validate_artifact_fields(
            "validation", self.decision_id, self.peak_ids, self.split_manifest_checksum,
            self.mode, self.cross_fit_folds,
        )


def _validate_artifact_fields(
    kind: str,
    decision_id: str,
    peak_ids: frozenset[str],
    checksum: str,
    mode: AnalysisMode,
    folds: tuple[CrossFitFold, ...],
) -> None:
    if not isinstance(decision_id, str) or not decision_id.strip():
        raise SchemaError(f"{kind} split artifact requires a non-empty decision_id")
    _require_peak_ids(f"{kind} split artifact peak_ids", peak_ids)
    _require_checksum(f"{kind} split artifact split_manifest_checksum", checksum)
    if not isinstance(mode, AnalysisMode):
        raise SchemaError(f"{kind} split artifact mode must be an AnalysisMode")
    if not isinstance(folds, tuple) or any(not isinstance(fold, CrossFitFold) for fold in folds):
        raise SchemaError(f"{kind} split artifact cross_fit_folds must be CrossFitFold values")
    if mode is AnalysisMode.CROSS_FIT and not folds:
        raise SchemaError(f"{kind} cross-fit artifact requires fold-specific memberships")
    if mode is not AnalysisMode.CROSS_FIT and folds:
        raise SchemaError(f"{kind} artifact cannot carry cross-fit folds outside CROSS_FIT mode")


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
    if decision.decision_id != validation.decision_id:
        raise SchemaError("decision and validation split artifacts must name the same decision_id")
    if decision.split_manifest_checksum != manifest.checksum:
        raise SchemaError("decision split artifact checksum does not match the split manifest")
    if validation.split_manifest_checksum != manifest.checksum:
        raise SchemaError("validation split artifact checksum does not match the split manifest")
    if decision.mode is not validation.mode:
        raise SchemaError("decision and validation split artifacts must declare the same analysis mode")
    if decision.mode is AnalysisMode.CROSS_FIT:
        if decision.peak_ids != validation.peak_ids or decision.cross_fit_folds != validation.cross_fit_folds:
            raise SchemaError("cross-fit artifacts must carry the same complete fold-specific memberships")
        assert_cross_fit_compatibility(manifest, decision.peak_ids, decision.cross_fit_folds)
        return
    assert_split_compatibility(
        manifest, decision.peak_ids, validation.peak_ids, mode=decision.mode
    )
