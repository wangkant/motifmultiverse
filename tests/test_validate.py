"""Validation split provenance tests (Task 13)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from motifmultiverse.schema import SchemaError
from motifmultiverse.validate import (
    AnalysisMode,
    CrossFitFold,
    DecisionSplitArtifact,
    PeakSplitManifest,
    ValidationSplitArtifact,
    assert_artifact_split_compatibility,
    assert_cross_fit_compatibility,
    assert_split_compatibility,
    build_peak_split_manifest,
)


def _manifest():
    return build_peak_split_manifest({
        "p-discovery": "DISCOVERY",
        "p-adjudication": "ADJUDICATION",
        "p-validation-a": "VALIDATION",
        "p-validation-b": "VALIDATION",
        "p-inference": "INFERENCE",
    })


def test_split_manifest_checksum_is_canonical_and_tampering_is_refused():
    first = _manifest()
    second = build_peak_split_manifest(dict(reversed(tuple(_manifest().assignments.items()))))
    assert first.checksum == second.checksum

    with pytest.raises(SchemaError, match="checksum"):
        replace(first, checksum="0" * 64)


def test_split_manifest_rejects_role_corruption_and_is_frozen():
    with pytest.raises(SchemaError, match="SplitRole"):
        build_peak_split_manifest({"p": "not-a-role"})

    manifest = _manifest()
    with pytest.raises(TypeError):
        manifest.assignments["other"] = manifest.assignments["p-discovery"]
    with pytest.raises(SchemaError, match="SplitRole"):
        PeakSplitManifest(
            schema_version=manifest.schema_version,
            assignments={"p-discovery": "DISCOVERY"},
            checksum=manifest.checksum,
        )


def test_split_primary_roles_are_mutually_exclusive_and_overlap_is_refused():
    manifest = _manifest()
    with pytest.raises(SchemaError, match="validation"):
        assert_split_compatibility(manifest, {"p-validation-a"}, {"p-validation-b"})
    with pytest.raises(SchemaError, match="overlap"):
        assert_split_compatibility(manifest, {"p-discovery"}, {"p-discovery"})


def test_split_overlap_requires_explicit_exploratory_mode():
    manifest = _manifest()
    assert_split_compatibility(
        manifest,
        {"p-discovery"},
        {"p-discovery"},
        mode=AnalysisMode.EXPLORATORY,
    )


def test_split_artifacts_bind_and_validate_the_manifest_checksum():
    manifest = _manifest()
    decision = DecisionSplitArtifact(
        decision_id="decision-1",
        peak_ids=frozenset({"p-discovery", "p-adjudication"}),
        split_manifest_checksum=manifest.checksum,
    )
    validation = ValidationSplitArtifact(
        decision_id="decision-1",
        peak_ids=frozenset({"p-validation-a", "p-validation-b"}),
        split_manifest_checksum=manifest.checksum,
    )
    assert_artifact_split_compatibility(manifest, decision, validation)

    with pytest.raises(SchemaError, match="checksum"):
        assert_artifact_split_compatibility(
            manifest,
            decision,
            replace(validation, split_manifest_checksum="f" * 64),
        )


def test_split_artifacts_require_an_actual_manifest_for_checksum_validation():
    manifest = _manifest()
    decision = DecisionSplitArtifact(
        decision_id="decision-1",
        peak_ids=frozenset({"p-discovery"}),
        split_manifest_checksum=manifest.checksum,
    )
    validation = ValidationSplitArtifact(
        decision_id="decision-1",
        peak_ids=frozenset({"p-validation-a"}),
        split_manifest_checksum=manifest.checksum,
    )
    with pytest.raises(SchemaError, match="PeakSplitManifest"):
        assert_artifact_split_compatibility(object(), decision, validation)


def test_split_cross_fit_cannot_be_claimed_by_an_ordinary_reused_split():
    manifest = _manifest()
    with pytest.raises(SchemaError, match="fold"):
        assert_split_compatibility(
            manifest,
            {"p-discovery"},
            {"p-discovery"},
            mode=AnalysisMode.CROSS_FIT,
        )


def test_split_cross_fit_requires_a_complete_nonduplicated_evaluation_partition():
    manifest = _manifest()
    all_peaks = frozenset({"p-discovery", "p-adjudication", "p-validation-a", "p-validation-b"})
    missing = (
        CrossFitFold("one", all_peaks - {"p-discovery"}, frozenset({"p-discovery"})),
        CrossFitFold("two", all_peaks - {"p-adjudication"}, frozenset({"p-adjudication"})),
    )
    with pytest.raises(SchemaError, match="missing"):
        assert_cross_fit_compatibility(manifest, all_peaks, missing)

    duplicate = (
        CrossFitFold("one", all_peaks - {"p-discovery"}, frozenset({"p-discovery"})),
        CrossFitFold("two", all_peaks - {"p-discovery"}, frozenset({"p-discovery"})),
    )
    with pytest.raises(SchemaError, match="duplicate"):
        assert_cross_fit_compatibility(manifest, all_peaks, duplicate)


def test_split_cross_fit_records_each_fold_specific_decision_and_evaluation_sets():
    manifest = _manifest()
    all_peaks = frozenset({"p-discovery", "p-adjudication", "p-validation-a", "p-validation-b"})
    folds = tuple(
        CrossFitFold(
            fold_id=peak_id,
            decision_peak_ids=all_peaks - {peak_id},
            evaluation_peak_ids=frozenset({peak_id}),
        )
        for peak_id in sorted(all_peaks)
    )
    assert_cross_fit_compatibility(manifest, all_peaks, folds)
