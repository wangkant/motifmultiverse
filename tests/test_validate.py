"""Validation split provenance tests (Task 13)."""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

import motifmultiverse.adjudicate as adjudicate
from motifmultiverse.schema import SchemaError, SplitRole, peak_split_manifest_checksum
from motifmultiverse.validate import (
    AnalysisMode,
    BackendUnavailable,
    BackendVerification,
    CrossFitFold,
    DecisionSplitArtifact,
    LexiconBinding,
    PeakSplitManifest,
    StabilityBackend,
    StabilityResult,
    ValidationSplitArtifact,
    assert_artifact_split_compatibility,
    assert_cross_fit_compatibility,
    assert_split_compatibility,
    build_peak_split_manifest,
    evaluate_stability,
    load_lexicon_binding,
    normalize_backend_output,
    run_backend_validation,
    stability_result_id,
    write_stability_artifacts,
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


def _persisted_artifact(kind: str):
    manifest = _manifest()
    common = {
        "manifest": manifest,
        "decision_id": "decision-1",
        "decision_peak_ids": frozenset({"p-discovery"}),
        "validation_peak_ids": frozenset({"p-validation-a"}),
    }
    if kind == "decision":
        artifact = DecisionSplitArtifact.create(**common)
        return artifact, DecisionSplitArtifact, manifest
    artifact = ValidationSplitArtifact.create(result_id="stability-1", **common)
    return artifact, ValidationSplitArtifact, manifest


@pytest.mark.parametrize("kind", ["decision", "validation"])
@pytest.mark.parametrize("artifact_id", ["", None, "not-an-artifact-id"])
def test_split_persisted_artifact_requires_a_nonempty_exact_identity(kind, artifact_id):
    artifact, artifact_type, manifest = _persisted_artifact(kind)
    payload = artifact.to_dict()
    payload["artifact_id"] = artifact_id
    with pytest.raises(SchemaError, match="artifact_id"):
        artifact_type.from_dict(payload, manifest=manifest)


@pytest.mark.parametrize("kind", ["decision", "validation"])
def test_split_persisted_artifact_requires_an_identity_field(kind):
    artifact, artifact_type, manifest = _persisted_artifact(kind)
    payload = artifact.to_dict()
    del payload["artifact_id"]
    with pytest.raises(SchemaError, match="malformed"):
        artifact_type.from_dict(payload, manifest=manifest)


@pytest.mark.parametrize("kind", ["decision", "validation"])
def test_split_persisted_artifact_rejects_relabelled_mode_when_identity_is_blanked(kind):
    artifact, artifact_type, manifest = _persisted_artifact(kind)
    payload = artifact.to_dict()
    payload["mode"] = AnalysisMode.EXPLORATORY.value
    payload["artifact_id"] = ""
    with pytest.raises(SchemaError, match="artifact_id"):
        artifact_type.from_dict(payload, manifest=manifest)


@pytest.mark.parametrize("kind", ["decision", "validation"])
@pytest.mark.parametrize("field, value", [
    ("decision_peak_ids", [["nested"]]),
    ("decision_peak_ids", [True]),
    ("validation_peak_ids", [1]),
    ("validation_peak_ids", "not-a-list"),
])
def test_split_persisted_artifact_rejects_malformed_peak_collections_with_field_paths(
    kind, field, value,
):
    artifact, artifact_type, manifest = _persisted_artifact(kind)
    payload = artifact.to_dict()
    payload[field] = value
    with pytest.raises(SchemaError, match=field):
        artifact_type.from_dict(payload, manifest=manifest)


@pytest.mark.parametrize("kind", ["decision", "validation"])
def test_split_persisted_artifact_rejects_nested_fold_collections_with_field_paths(kind):
    manifest = _manifest()
    peaks = frozenset({"p-discovery", "p-validation-a"})
    folds = (
        CrossFitFold("one", frozenset({"p-validation-a"}), frozenset({"p-discovery"})),
        CrossFitFold("two", frozenset({"p-discovery"}), frozenset({"p-validation-a"})),
    )
    common = {
        "manifest": manifest,
        "decision_id": "decision-1",
        "decision_peak_ids": peaks,
        "validation_peak_ids": peaks,
        "mode": AnalysisMode.CROSS_FIT,
        "cross_fit_folds": folds,
    }
    if kind == "decision":
        artifact = DecisionSplitArtifact.create(**common)
        artifact_type = DecisionSplitArtifact
    else:
        artifact = ValidationSplitArtifact.create(result_id="stability-1", **common)
        artifact_type = ValidationSplitArtifact
    payload = artifact.to_dict()
    payload["cross_fit_folds"][0]["decision_peak_ids"] = [["nested"]]
    with pytest.raises(SchemaError, match=r"cross_fit_folds\[0\].decision_peak_ids\[0\]"):
        artifact_type.from_dict(payload, manifest=manifest)


@pytest.mark.parametrize("kind", ["decision", "validation"])
@pytest.mark.parametrize("replacement, match", [
    ("not-a-list", "cross_fit_folds must be a list"),
    ([[]], r"cross_fit_folds\[0\] metadata"),
    ([
        {
            "fold_id": True,
            "decision_peak_ids": ["p-validation-a"],
            "evaluation_peak_ids": ["p-discovery"],
        },
    ], r"cross_fit_folds\[0\].fold_id"),
    ([
        {
            "fold_id": "one",
            "decision_peak_ids": [1],
            "evaluation_peak_ids": ["p-discovery"],
        },
    ], r"cross_fit_folds\[0\].decision_peak_ids\[0\]"),
])
def test_split_persisted_artifact_rejects_malformed_fold_parser_values(
    kind, replacement, match,
):
    manifest = _manifest()
    peaks = frozenset({"p-discovery", "p-validation-a"})
    folds = (
        CrossFitFold("one", frozenset({"p-validation-a"}), frozenset({"p-discovery"})),
        CrossFitFold("two", frozenset({"p-discovery"}), frozenset({"p-validation-a"})),
    )
    common = {
        "manifest": manifest,
        "decision_id": "decision-1",
        "decision_peak_ids": peaks,
        "validation_peak_ids": peaks,
        "mode": AnalysisMode.CROSS_FIT,
        "cross_fit_folds": folds,
    }
    if kind == "decision":
        artifact = DecisionSplitArtifact.create(**common)
        artifact_type = DecisionSplitArtifact
    else:
        artifact = ValidationSplitArtifact.create(result_id="stability-1", **common)
        artifact_type = ValidationSplitArtifact
    payload = artifact.to_dict()
    payload["cross_fit_folds"] = replacement
    with pytest.raises(SchemaError, match=match):
        artifact_type.from_dict(payload, manifest=manifest)


def _hit_table(*, affected: int = 20, total: int = 200, merged: bool = False):
    rows = []
    for number in range(total):
        changed = merged and number < affected
        rows.append({
            "peak_id": f"peak-{number:03d}",
            "hit_id": "family-merged" if changed else "family-original",
            "coefficient": 2.0 if changed else 1.0,
            "reconstruction": 1.0 if changed else 0.0,
        })
    return pd.DataFrame(rows)


def _stability_result() -> StabilityResult:
    return evaluate_stability(
        "decision:affected-subset",
        _hit_table(),
        _hit_table(merged=True),
    )


def test_stability_uses_affected_subset_not_all_peak_median_dilution():
    """A 20/200 changed subset cannot be hidden by an all-peak median of zero."""
    result = _stability_result()

    assert result.n_affected_peaks == 20
    assert result.paired_delta_reconstruction_all == 0.0
    assert result.paired_delta_reconstruction_affected == 1.0
    assert result.status == "LOW_RISK_RARE_NOT_VALIDATED"
    assert "no interval or inferential equivalence claim" in result.power_statement.lower()


def test_stability_rare_events_omit_interval_and_state_frequency_limited_power():
    result = evaluate_stability("decision:rare", _hit_table(total=40), _hit_table(
        affected=29, total=40, merged=True,
    ))

    assert result.n_affected_peaks == 29
    assert result.status == "LOW_RISK_RARE_NOT_VALIDATED"
    assert result.affected_interval is None
    assert "frequency-limited" in result.power_statement


@pytest.mark.parametrize("column, value", [
    ("peak_id", ""),
    ("hit_id", None),
    ("coefficient", "1.0"),
    ("coefficient", float("inf")),
    ("reconstruction", float("nan")),
])
def test_stability_normalization_rejects_corrupt_backend_output(column, value):
    rows = _hit_table(total=1).copy()
    if isinstance(value, str) and column in {"coefficient", "reconstruction"}:
        rows[column] = rows[column].astype(object)
    rows.loc[0, column] = value
    with pytest.raises(SchemaError, match=column):
        normalize_backend_output(rows, backend="test-backend")


@pytest.mark.parametrize("output, match", [
    ({"peak_id": ["p"]}, "DataFrame"),
    (pd.DataFrame(), "missing standardized"),
    (pd.DataFrame(columns=["peak_id", "hit_id", "coefficient", "reconstruction"]), "cannot be empty"),
    (pd.DataFrame([{"peak_id": "p", "hit_id": "h", "coefficient": True, "reconstruction": 0.0}]), "numeric"),
    (pd.DataFrame([
        {"peak_id": "p", "hit_id": "h", "coefficient": 1.0, "reconstruction": 0.0},
        {"peak_id": "p", "hit_id": "h", "coefficient": 2.0, "reconstruction": 0.0},
    ]), "duplicate"),
])
def test_stability_normalization_negative_controls_cover_type_empty_and_identity(output, match):
    with pytest.raises(SchemaError, match=match):
        normalize_backend_output(output, backend="control")


def test_stability_refuses_inconsistent_peak_reconstruction_and_unpaired_peak_sets():
    inconsistent = pd.DataFrame([
        {"peak_id": "p", "hit_id": "a", "coefficient": 1.0, "reconstruction": 0.0},
        {"peak_id": "p", "hit_id": "b", "coefficient": 1.0, "reconstruction": 1.0},
    ])
    with pytest.raises(SchemaError, match="multiple reconstruction"):
        evaluate_stability("decision:control", inconsistent, inconsistent)
    with pytest.raises(SchemaError, match="same peak identities"):
        evaluate_stability("decision:control", _hit_table(total=1), _hit_table(total=2))


def test_stability_coefficient_cancellation_and_none_similarity_semantics():
    before = pd.DataFrame([
        {"peak_id": "p", "hit_id": "old", "coefficient": 1.0, "reconstruction": 0.0},
    ])
    after = pd.DataFrame([
        {"peak_id": "p", "hit_id": "new", "coefficient": 0.0, "reconstruction": 1.0},
    ])
    changed = evaluate_stability("decision:control", before, after)
    assert changed.family_coefficient_share == 0.0
    assert changed.hit_jaccard == 0.0
    assert changed.coefficient_conservation is None
    unchanged = evaluate_stability("decision:control", before, before)
    assert unchanged.hit_jaccard is None
    assert unchanged.coefficient_conservation is None


class _UnavailableBackend(StabilityBackend):
    name = "optional-missing"
    version = "1"
    optional = True

    def compare(self, lexicons, decision_id):
        raise BackendUnavailable("optional-missing backend is not installed")


class _AvailableBackend(StabilityBackend):
    name = "available"
    version = "1"

    def compare(self, lexicons, decision_id):
        return _hit_table(), _hit_table(merged=True)


def test_missing_optional_backend_is_persisted_unverified_without_erasing_verified_results(tmp_path):
    results, verification = run_backend_validation(
        load_lexicon_binding(_lexicons(tmp_path)), "decision:backend",
        [_UnavailableBackend(), _AvailableBackend()],
    )

    assert [result.decision_id for result in results] == ["decision:backend"]
    assert {(row.backend, row.status) for row in verification} == {
        ("optional-missing", "UNVERIFIED"), ("available", "VERIFIED"),
    }
    assert "optional-missing backend is not installed" in verification[0].detail


def test_stability_result_satisfies_the_adjudication_evidence_protocol():
    assert isinstance(_stability_result(), adjudicate.StabilityEvidence)


def test_stability_artifacts_bind_split_identity_and_provenance(tmp_path):
    binding = load_lexicon_binding(_lexicons(tmp_path))
    manifest = _manifest()
    decision = DecisionSplitArtifact.create(
        manifest=manifest,
        decision_id="decision:artifact",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation-a"}),
    )
    provisional = ValidationSplitArtifact.create(
        manifest=manifest,
        decision_id="decision:artifact",
        result_id="pending",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation-a"}),
    )
    results, verification = run_backend_validation(binding, decision.decision_id, [_AvailableBackend()])
    provenance = {"stage": "test", "input": "fixed", "lexicon": binding.to_dict()}
    validation = ValidationSplitArtifact.create(
        manifest=manifest, decision_id=decision.decision_id,
        result_id=stability_result_id(results, verification, provenance, binding, manifest, decision, provisional),
        decision_peak_ids=provisional.decision_peak_ids,
        validation_peak_ids=provisional.validation_peak_ids,
    )
    results_path, verification_path = write_stability_artifacts(
        tmp_path / "out",
        results,
        verification,
        manifest=manifest,
        decision=decision,
        validation=validation,
        provenance=provenance,
        lexicon=binding,
    )

    import pyarrow.parquet as pq

    metadata = pq.read_schema(results_path).metadata
    assert metadata[b"motifmultiverse.split_manifest_checksum"] == manifest.checksum.encode()
    assert metadata[b"motifmultiverse.decision_artifact_id"] == decision.artifact_id.encode()
    assert metadata[b"motifmultiverse.validation_artifact_id"] == validation.artifact_id.encode()
    assert "UNVERIFIED" not in verification_path.read_text(encoding="utf-8")


def test_stability_artifact_writer_refuses_a_manifest_mismatch(tmp_path):
    binding = load_lexicon_binding(_lexicons(tmp_path))
    manifest = _manifest()
    other_manifest = build_peak_split_manifest({
        **manifest.assignments,
        "p-validation-extra": "VALIDATION",
    })
    decision = DecisionSplitArtifact.create(
        manifest=manifest,
        decision_id="decision:mismatch",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation-a"}),
    )
    validation = ValidationSplitArtifact.create(
        manifest=other_manifest,
        decision_id="decision:mismatch",
        result_id="stability:mismatch",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation-a"}),
    )
    result = evaluate_stability("decision:mismatch", _hit_table(), _hit_table(merged=True))

    with pytest.raises(SchemaError, match="checksum"):
        write_stability_artifacts(
            tmp_path, [result], [], manifest=manifest, decision=decision,
            validation=validation, provenance={"stage": "test"}, lexicon=binding,
        )


def _lexicons(tmp_path, *, content_hash="a" * 64):
    import h5py
    import numpy as np

    from motifmultiverse.compile import _content_hash

    lexicons = tmp_path / "lexicons"
    lexicons.mkdir()
    with h5py.File(lexicons / "core.h5", "w") as h5:
        motif = h5.create_group("pos").create_group("pattern_0")
        motif.create_dataset("contrib_scores", data=np.asarray([[1.0, 0.0, 0.0, 0.0]]))
    content_hash = _content_hash(
        [("pos", "pattern_0", {"node_id": "node-0"})],
        {"node-0": {"cwm": np.asarray([[1.0, 0.0, 0.0, 0.0]])}},
        schema_version="1.0", trim_threshold=0.3, motif_type="cwm", include_rc=False,
        loader_backend="finemo", loader_parameters={},
    )
    (lexicons / "core.manifest.json").write_text(json.dumps({
        "tier": "core", "lexicon_content_hash": content_hash, "n_motifs": 1,
        "pattern_order": ["pos.pattern_0"], "node_ids": ["node-0"],
        "index": [{"pattern_tag": "pos.pattern_0", "node_id": "node-0"}],
        "schema_version": "1.0", "trim_threshold": 0.3, "motif_type": "cwm",
        "include_rc": False, "loader_backend": "finemo", "loader_parameters": {},
    }), encoding="utf-8")
    return lexicons


def test_lexicon_binding_requires_a_real_content_addressed_compiled_lexicon(tmp_path):
    with pytest.raises(SchemaError, match="does not exist"):
        load_lexicon_binding(tmp_path / "missing")
    lexicons = _lexicons(tmp_path)
    binding = load_lexicon_binding(lexicons)
    assert isinstance(binding, LexiconBinding)
    assert binding.lexicon_identity.startswith("lexicons:")
    (lexicons / "core.h5").write_bytes(b"tampered")
    with pytest.raises(SchemaError, match="verified compiled lexicon"):
        load_lexicon_binding(lexicons)


def test_backend_results_are_bound_to_the_backend_and_lexicon_identity(tmp_path):
    binding = load_lexicon_binding(_lexicons(tmp_path))
    results, verification = run_backend_validation(
        binding, "decision:backend", [_AvailableBackend()],
    )

    assert results[0].backend == verification[0].backend == "available"
    assert results[0].backend_result_id == verification[0].backend_result_id
    assert results[0].lexicon_identity == verification[0].lexicon_identity == binding.lexicon_identity


def test_stability_result_id_changes_with_lexicon_identity_and_cannot_match_an_arbitrary_result_id(tmp_path):
    binding = load_lexicon_binding(_lexicons(tmp_path))
    manifest = _manifest()
    decision = DecisionSplitArtifact.create(
        manifest=manifest, decision_id="decision:identity",
        decision_peak_ids=frozenset({"p-discovery"}), validation_peak_ids=frozenset({"p-validation-a"}),
    )
    validation = ValidationSplitArtifact.create(
        manifest=manifest, decision_id="decision:identity", result_id="arbitrary",
        decision_peak_ids=frozenset({"p-discovery"}), validation_peak_ids=frozenset({"p-validation-a"}),
    )
    results, verification = run_backend_validation(binding, decision.decision_id, [_AvailableBackend()])
    provenance = {"stage": "test", "inputs": {}, "software": {}, "lexicon": binding.to_dict()}
    expected = stability_result_id(results, verification, provenance, binding, manifest, decision, validation)

    assert expected.startswith("stability:")
    assert expected != validation.result_id
    with pytest.raises(SchemaError, match="result_id"):
        write_stability_artifacts(
            tmp_path / "out", results, verification, manifest=manifest, decision=decision,
            validation=validation, provenance=provenance, lexicon=binding,
        )


def test_empty_stability_artifact_has_explicit_schema_and_unverified_backend_row(tmp_path):
    binding = load_lexicon_binding(_lexicons(tmp_path))
    manifest = _manifest()
    decision = DecisionSplitArtifact.create(
        manifest=manifest, decision_id="decision:empty",
        decision_peak_ids=frozenset({"p-discovery"}), validation_peak_ids=frozenset({"p-validation-a"}),
    )
    provisional = ValidationSplitArtifact.create(
        manifest=manifest, decision_id=decision.decision_id, result_id="pending",
        decision_peak_ids=decision.decision_peak_ids, validation_peak_ids=decision.validation_peak_ids,
    )
    verification = [BackendVerification(
        "missing", "1", "UNVERIFIED", "missing backend", lexicon_identity=binding.lexicon_identity,
    )]
    provenance = {"stage": "test", "inputs": {}, "software": {}, "lexicon": binding.to_dict()}
    validation = ValidationSplitArtifact.create(
        manifest=manifest, decision_id=decision.decision_id,
        result_id=stability_result_id([], verification, provenance, binding, manifest, decision, provisional),
        decision_peak_ids=decision.decision_peak_ids, validation_peak_ids=decision.validation_peak_ids,
    )
    result_path, verification_path = write_stability_artifacts(
        tmp_path / "out", [], verification, manifest=manifest, decision=decision,
        validation=validation, provenance=provenance, lexicon=binding,
    )
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(result_path)
    assert schema.field("n_affected_peaks").type == pa.int64()
    assert schema.metadata[b"motifmultiverse.artifact_id"] == validation.result_id.encode()
    assert "UNVERIFIED" in verification_path.read_text(encoding="utf-8")


def test_unlinked_backend_result_refuses_before_any_artifact_is_published(tmp_path):
    binding = load_lexicon_binding(_lexicons(tmp_path))
    manifest = _manifest()
    decision = DecisionSplitArtifact.create(
        manifest=manifest, decision_id="decision:linked",
        decision_peak_ids=frozenset({"p-discovery"}), validation_peak_ids=frozenset({"p-validation-a"}),
    )
    provisional = ValidationSplitArtifact.create(
        manifest=manifest, decision_id=decision.decision_id, result_id="pending",
        decision_peak_ids=decision.decision_peak_ids, validation_peak_ids=decision.validation_peak_ids,
    )
    results, verification = run_backend_validation(binding, decision.decision_id, [_AvailableBackend()])
    provenance = {"stage": "test", "inputs": {}, "software": {}, "lexicon": binding.to_dict()}
    validation = ValidationSplitArtifact.create(
        manifest=manifest, decision_id=decision.decision_id,
        result_id=stability_result_id(results, verification, provenance, binding, manifest, decision, provisional),
        decision_peak_ids=decision.decision_peak_ids, validation_peak_ids=decision.validation_peak_ids,
    )
    with pytest.raises(SchemaError, match="exactly one VERIFIED"):
        write_stability_artifacts(
            tmp_path / "refused", [replace(results[0], backend_result_id="forged")], verification,
            manifest=manifest, decision=decision, validation=validation,
            provenance=provenance, lexicon=binding,
        )
    assert not (tmp_path / "refused").exists()
