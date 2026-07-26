"""Validation split provenance tests (Task 13)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from motifmultiverse.schema import SchemaError, SplitRole, peak_split_manifest_checksum
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
    decision = DecisionSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        decision_peak_ids=frozenset({"p-discovery", "p-adjudication"}),
        validation_peak_ids=frozenset({"p-validation-a", "p-validation-b"}),
    )
    validation = ValidationSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        result_id="stability-1",
        decision_peak_ids=frozenset({"p-discovery", "p-adjudication"}),
        validation_peak_ids=frozenset({"p-validation-a", "p-validation-b"}),
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
    decision = DecisionSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation-a"}),
    )
    validation = ValidationSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        result_id="stability-1",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation-a"}),
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


@pytest.mark.parametrize("assignments", [
    {1: "DISCOVERY"},
    {True: "DISCOVERY"},
    {"p": "DISCOVERY", 1: "VALIDATION"},
])
def test_split_manifest_rejects_nonstring_and_mixed_peak_ids_as_schema_errors(assignments):
    with pytest.raises(SchemaError, match="peak IDs"):
        build_peak_split_manifest(assignments)


def test_split_manifest_public_checksum_helper_rejects_malformed_direct_inputs():
    with pytest.raises(SchemaError, match="peak IDs"):
        peak_split_manifest_checksum("1", {1: SplitRole.DISCOVERY})
    with pytest.raises(SchemaError, match="SplitRole"):
        peak_split_manifest_checksum("1", {"p": "DISCOVERY"})


def test_split_artifacts_are_created_against_the_exact_manifest_and_canonical_identity():
    manifest = _manifest()
    decision = DecisionSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        decision_peak_ids=frozenset({"p-discovery", "p-adjudication"}),
        validation_peak_ids=frozenset({"p-validation-a", "p-validation-b"}),
    )
    validation = ValidationSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        result_id="stability-1",
        decision_peak_ids=frozenset({"p-discovery", "p-adjudication"}),
        validation_peak_ids=frozenset({"p-validation-a", "p-validation-b"}),
    )
    assert decision.split_manifest_checksum == manifest.checksum
    assert decision.artifact_id != validation.artifact_id
    assert DecisionSplitArtifact.from_dict(decision.to_dict(), manifest=manifest) == decision
    assert ValidationSplitArtifact.from_dict(validation.to_dict(), manifest=manifest) == validation


def test_split_artifact_identity_rejects_checksum_and_mode_tampering():
    manifest = _manifest()
    decision = DecisionSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation-a"}),
    )
    with pytest.raises(SchemaError, match="checksum"):
        replace(decision, split_manifest_checksum="f" * 64)
    with pytest.raises(SchemaError, match="artifact_id"):
        replace(decision, mode=AnalysisMode.EXPLORATORY)
    payload = decision.to_dict()
    payload["mode"] = AnalysisMode.EXPLORATORY.value
    with pytest.raises(SchemaError, match="artifact_id"):
        DecisionSplitArtifact.from_dict(payload, manifest=manifest)


@pytest.mark.parametrize("checksum", [None, 1, "F" * 64, "not-a-digest"])
def test_split_artifact_rejects_malformed_checksum_controls(checksum):
    manifest = _manifest()
    with pytest.raises(SchemaError, match="checksum"):
        DecisionSplitArtifact(
            manifest=manifest,
            decision_id="decision-1",
            decision_peak_ids=frozenset({"p-discovery"}),
            validation_peak_ids=frozenset({"p-validation-a"}),
            split_manifest_checksum=checksum,
        )


def test_split_artifact_rejects_unknown_members_and_non_cross_fit_folds():
    manifest = _manifest()
    with pytest.raises(SchemaError, match="absent"):
        DecisionSplitArtifact.create(
            manifest=manifest,
            decision_id="decision-1",
            decision_peak_ids=frozenset({"unknown"}),
            validation_peak_ids=frozenset({"p-validation-a"}),
        )
    fold = CrossFitFold(
        "one", frozenset({"p-discovery"}), frozenset({"p-validation-a"})
    )
    with pytest.raises(SchemaError, match="outside CROSS_FIT"):
        DecisionSplitArtifact.create(
            manifest=manifest,
            decision_id="decision-1",
            decision_peak_ids=frozenset({"p-discovery"}),
            validation_peak_ids=frozenset({"p-validation-a"}),
            cross_fit_folds=(fold,),
        )


def test_split_artifact_rejects_cross_fit_without_matching_full_analysis_membership():
    manifest = _manifest()
    peaks = frozenset({"p-discovery", "p-validation-a"})
    folds = (
        CrossFitFold("one", frozenset({"p-validation-a"}), frozenset({"p-discovery"})),
        CrossFitFold("two", frozenset({"p-discovery"}), frozenset({"p-validation-a"})),
    )
    with pytest.raises(SchemaError, match="same complete"):
        DecisionSplitArtifact.create(
            manifest=manifest,
            decision_id="decision-1",
            decision_peak_ids=peaks,
            validation_peak_ids=frozenset({"p-validation-a"}),
            mode=AnalysisMode.CROSS_FIT,
            cross_fit_folds=folds,
        )
    artifact = DecisionSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        decision_peak_ids=peaks,
        validation_peak_ids=peaks,
        mode=AnalysisMode.CROSS_FIT,
        cross_fit_folds=folds,
    )
    assert artifact.to_dict()["cross_fit_folds"] == [
        {
            "fold_id": "one",
            "decision_peak_ids": ["p-validation-a"],
            "evaluation_peak_ids": ["p-discovery"],
        },
        {
            "fold_id": "two",
            "decision_peak_ids": ["p-discovery"],
            "evaluation_peak_ids": ["p-validation-a"],
        },
    ]


def test_split_cross_fit_artifact_normalizes_fold_order_for_canonical_serialization():
    manifest = _manifest()
    peaks = frozenset({"p-discovery", "p-validation-a"})
    folds = (
        CrossFitFold("two", frozenset({"p-discovery"}), frozenset({"p-validation-a"})),
        CrossFitFold("one", frozenset({"p-validation-a"}), frozenset({"p-discovery"})),
    )
    artifact = DecisionSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        decision_peak_ids=peaks,
        validation_peak_ids=peaks,
        mode=AnalysisMode.CROSS_FIT,
        cross_fit_folds=folds,
    )
    assert [fold.fold_id for fold in artifact.cross_fit_folds] == ["one", "two"]
    assert [row["fold_id"] for row in artifact.to_dict()["cross_fit_folds"]] == ["one", "two"]


def test_split_cross_fit_rejects_duplicate_fold_ids_and_noncomplement_decisions():
    manifest = _manifest()
    peaks = frozenset({"p-discovery", "p-adjudication", "p-validation-a"})
    duplicate_ids = (
        CrossFitFold("one", frozenset({"p-validation-a", "p-adjudication"}), frozenset({"p-discovery"})),
        CrossFitFold("one", frozenset({"p-discovery", "p-adjudication"}), frozenset({"p-validation-a"})),
    )
    with pytest.raises(SchemaError, match="fold IDs"):
        assert_cross_fit_compatibility(manifest, peaks, duplicate_ids)
    noncomplement = (
        CrossFitFold("one", frozenset({"p-validation-a"}), frozenset({"p-discovery"})),
        CrossFitFold("two", frozenset({"p-discovery", "p-adjudication"}), frozenset({"p-validation-a"})),
        CrossFitFold("three", frozenset({"p-discovery", "p-validation-a"}), frozenset({"p-adjudication"})),
    )
    with pytest.raises(SchemaError, match="complementary"):
        assert_cross_fit_compatibility(manifest, peaks, noncomplement)


def test_split_artifact_pair_rejects_decision_id_and_mode_mismatches():
    manifest = _manifest()
    decision = DecisionSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation-a"}),
    )
    wrong_decision = ValidationSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-2",
        result_id="stability-1",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation-a"}),
    )
    with pytest.raises(SchemaError, match="decision_id"):
        assert_artifact_split_compatibility(manifest, decision, wrong_decision)
    exploratory = ValidationSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        result_id="stability-1",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-discovery"}),
        mode=AnalysisMode.EXPLORATORY,
    )
    with pytest.raises(SchemaError, match="same analysis mode"):
        assert_artifact_split_compatibility(manifest, decision, exploratory)
    altered_peaks = ValidationSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        result_id="stability-1",
        decision_peak_ids=frozenset({"p-adjudication"}),
        validation_peak_ids=frozenset({"p-validation-a"}),
    )
    with pytest.raises(SchemaError, match="same split semantics"):
        assert_artifact_split_compatibility(manifest, decision, altered_peaks)


@pytest.mark.parametrize("fold", [
    CrossFitFold("one", frozenset({"p-discovery"}), frozenset({"p-validation-a"})),
])
def test_split_cross_fit_fold_controls_cover_unknown_members_and_empty_fold_ids(fold):
    manifest = _manifest()
    with pytest.raises(SchemaError, match="absent"):
        assert_cross_fit_compatibility(
            manifest,
            frozenset({"p-discovery", "unknown"}),
            (fold, fold),
        )
    with pytest.raises(SchemaError, match="fold_id"):
        CrossFitFold(" ", frozenset({"p-discovery"}), frozenset({"p-validation-a"}))


def test_split_manifest_controls_cover_unsupported_version_and_empty_assignments():
    with pytest.raises(SchemaError, match="schema_version"):
        peak_split_manifest_checksum("2", {"p": SplitRole.DISCOVERY})
    with pytest.raises(SchemaError, match="non-empty mapping"):
        build_peak_split_manifest({})


def test_split_cross_fit_fold_rejects_overlapping_or_nonset_memberships():
    with pytest.raises(SchemaError, match="cannot evaluate"):
        CrossFitFold("one", frozenset({"p"}), frozenset({"p"}))
    with pytest.raises(SchemaError, match="non-empty set"):
        CrossFitFold("one", frozenset(), frozenset({"p"}))


def test_split_artifact_deserialization_refuses_missing_or_empty_manifest_checksum():
    manifest = _manifest()
    decision = DecisionSplitArtifact.create(
        manifest=manifest,
        decision_id="decision-1",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation-a"}),
    )
    payload = decision.to_dict()
    payload["split_manifest_checksum"] = ""
    with pytest.raises(SchemaError, match="checksum"):
        DecisionSplitArtifact.from_dict(payload, manifest=manifest)
    payload = decision.to_dict()
    del payload["split_manifest_checksum"]
    with pytest.raises(SchemaError, match="malformed"):
        DecisionSplitArtifact.from_dict(payload, manifest=manifest)
