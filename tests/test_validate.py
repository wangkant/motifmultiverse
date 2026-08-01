"""Validation split provenance tests (Task 13)."""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

import motifmultiverse.adjudicate as adjudicate
from motifmultiverse.provenance import ProvenanceRecord
from motifmultiverse.schema import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SchemaError,
    SplitRole,
    peak_split_manifest_checksum,
)
from motifmultiverse.validate import (
    DEVICE_NULL_ABS_COEFFICIENT_DELTA,
    AnalysisMode,
    BackendUnavailable,
    BackendVerification,
    CrossFitFold,
    DecisionSplitArtifact,
    LexiconBinding,
    PeakSplitManifest,
    StabilityBackend,
    StabilityProvenance,
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


def test_the_split_checksum_seals_the_ROLE_and_not_only_the_set_of_peak_ids():
    """A seal that covers the peak IDs but not their roles seals nothing that matters.

    Every existing exercise of this checksum varies the peak-ID set, so the digest
    would keep passing them all while covering only half of what it claims. Held
    out the other way: dropping `role` from the hashed payload leaves the entire
    suite green and ruff clean.

    What that would ship is the whole point of the mechanism. Move one peak from
    VALIDATION to DISCOVERY in a committed `--split-manifest` and leave the
    recorded checksum alone: `PeakSplitManifest.__post_init__` recomputes the
    digest, gets the recorded value back and accepts the manifest as sealed. The
    split artifacts then bind to it, `_canonical_artifact_id` folds in the
    unchanged checksum, and `write_stability_artifacts` stamps it into the parquet
    footer. The published result claims held-out validation on a peak that was
    used to make the decision -- the exact axis `HELD_OUT_INFERENCE` is granted on
    -- and every checksum on the artifact agrees the split was never touched.

    So the assertion holds the peak-ID SET fixed and varies only the roles.
    """
    honest = {"p-1": SplitRole.DISCOVERY, "p-42": SplitRole.VALIDATION}
    moved = {"p-1": SplitRole.DISCOVERY, "p-42": SplitRole.DISCOVERY}
    swapped = {"p-1": SplitRole.VALIDATION, "p-42": SplitRole.DISCOVERY}

    digest = peak_split_manifest_checksum(SPLIT_MANIFEST_SCHEMA_VERSION, honest)
    assert peak_split_manifest_checksum(SPLIT_MANIFEST_SCHEMA_VERSION, moved) != digest
    assert peak_split_manifest_checksum(SPLIT_MANIFEST_SCHEMA_VERSION, swapped) != digest

    # And the seal must refuse the tampered assignments under the honest digest,
    # which is the shape the defect actually takes on disk.
    with pytest.raises(SchemaError, match="checksum"):
        PeakSplitManifest(
            schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
            assignments=moved,
            checksum=digest,
        )


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
    assert changed.affected_coefficient_share == 0.0
    assert changed.hit_jaccard == 0.0
    assert changed.coefficient_conservation is None
    unchanged = evaluate_stability("decision:control", before, before)
    assert unchanged.hit_jaccard is None
    assert unchanged.coefficient_conservation is None


def test_stability_coefficient_share_uses_absolute_mass_under_real_signed_cancellation():
    before = pd.DataFrame([
        {"peak_id": "affected", "hit_id": "plus", "coefficient": 0.5, "reconstruction": 0.0},
        {"peak_id": "affected", "hit_id": "minus", "coefficient": -0.5, "reconstruction": 0.0},
        {"peak_id": "unchanged", "hit_id": "other", "coefficient": 2.0, "reconstruction": 0.0},
    ])
    after = pd.DataFrame([
        {"peak_id": "affected", "hit_id": "plus", "coefficient": 1.0, "reconstruction": 1.0},
        {"peak_id": "affected", "hit_id": "minus", "coefficient": -1.0, "reconstruction": 1.0},
        {"peak_id": "unchanged", "hit_id": "other", "coefficient": 2.0, "reconstruction": 0.0},
    ])

    result = evaluate_stability("decision:signed-cancellation", before, after)

    assert after.loc[after["peak_id"] == "affected", "coefficient"].sum() == 0.0
    assert result.affected_coefficient_share == 0.5


def _refit_hit_table(*, changed: int, total: int, after: bool, jitter: float = 1e-09):
    """A hit table shaped like a real iterative caller's re-fit.

    `changed` peaks genuinely change: one hit is replaced and the surviving
    coefficients move by 0.5. EVERY other peak keeps its exact hit set and moves
    only by `jitter` -- which is what FiNeMo does to the ~93,000 hits a single
    dropped motif never touched, and which is three orders of magnitude below the
    measured device-null coefficient floor (`DEVICE_NULL_ABS_COEFFICIENT_DELTA`).
    """
    rows = []
    for number in range(total):
        genuinely_changed = after and number < changed
        moved = after and number >= changed
        for index, name in enumerate(("a", "b", "c")):
            hit_id = "c-merged" if (genuinely_changed and name == "c") else name
            coefficient = 1.0 + index
            if genuinely_changed:
                coefficient += 0.5
            elif moved:
                coefficient += jitter
            rows.append({
                "peak_id": f"peak-{number:04d}",
                "hit_id": hit_id,
                "coefficient": coefficient,
                "reconstruction": 1.0 if genuinely_changed else 0.0,
            })
    return pd.DataFrame(rows)


def test_affected_set_is_not_every_coefficient_that_moved_in_the_last_bits():
    """Exact float inequality made a re-fit look like a merge effect.

    `before != after` counted a peak as affected when any coefficient differed at
    all. A hit caller re-fits competitively when the lexicon changes, so on the
    thirteen-analysis case study 20% of the hits a merge never touched still
    differed -- by a median of 0.0 and a 99th percentile of 4.1e-06 -- and the
    affected set inflated about fivefold. Dropping `neg_patterns.pattern_18` (327
    hits in 276 of 2,639 peaks) reported 2,121 affected peaks; the hit set changed
    in 360 and 414 peaks cleared the measured device-null floor.

    Both definitions stay reachable and both are named on the result. Exact
    inequality is `coefficient_tolerance=0.0` -- still available, no longer the
    silent default.
    """
    before = _refit_hit_table(changed=40, total=400, after=False)
    after = _refit_hit_table(changed=40, total=400, after=True)

    default = evaluate_stability("decision:refit", before, after)
    hit_set_only = evaluate_stability(
        "decision:refit", before, after, coefficient_tolerance=None,
    )
    exact = evaluate_stability("decision:refit", before, after, coefficient_tolerance=0.0)

    assert default.n_affected_peaks == 40
    assert hit_set_only.n_affected_peaks == 40
    assert exact.n_affected_peaks == 400          # the pristine behaviour, on demand
    assert default.n_affected_hits == 40 * 4      # a, b, c and c-merged
    assert exact.n_affected_hits == 40 * 4 + 360 * 3
    assert default.affected_definition == (
        f"hit_set_change|abs_coefficient_delta>{DEVICE_NULL_ABS_COEFFICIENT_DELTA:g}")
    assert hit_set_only.affected_definition == "hit_set_change"
    assert exact.affected_definition == "hit_set_change|abs_coefficient_delta>0.0"


def test_a_coefficient_that_moved_past_the_tolerance_is_still_affected():
    """The narrower definition is not "hit identity only" by the back door.

    A peak whose hit set is untouched but whose coefficient genuinely moved is
    affected under the default, and is excluded only when the caller explicitly
    asks for `coefficient_tolerance=None`. Without this the fix would trade a
    fivefold over-count for an under-count that hides real redistribution.
    """
    before = pd.DataFrame([
        {"peak_id": "redistributed", "hit_id": "kept", "coefficient": 1.0, "reconstruction": 0.0},
        {"peak_id": "untouched", "hit_id": "kept", "coefficient": 2.0, "reconstruction": 0.0},
    ])
    after = pd.DataFrame([
        {"peak_id": "redistributed", "hit_id": "kept", "coefficient": 1.4, "reconstruction": 1.0},
        {"peak_id": "untouched", "hit_id": "kept", "coefficient": 2.0, "reconstruction": 0.0},
    ])

    assert evaluate_stability("decision:redistribution", before, after).n_affected_peaks == 1


@pytest.mark.parametrize("direction, moved_to", [("up", 1.4), ("down", 0.6)])
def test_a_coefficient_is_affected_whichever_WAY_it_moved(direction, moved_to):
    """`abs` is in the rule's own name, and only one side of it was ever exercised.

    The persisted rule reads `hit_set_change|abs_coefficient_delta>3.63e-07`. Every
    other test here moves a shared coefficient UP, so dropping the `abs` --
    comparing a signed delta instead -- left the whole suite green and ruff clean.

    What that ships is an under-count in the ordinary direction. A merge moves mass
    ONTO the survivor and OFF the collapsed motif's shared hits, so the peaks that
    lost mass are exactly the ones a signed comparison stops counting.
    `n_affected_peaks` collapses; below MIN_AFFECTED_PEAKS the record flips to
    LOW_RISK_RARE_NOT_VALIDATED with a "frequency-limited" power statement -- the
    artifact reporting that the merge touched too little to validate, while every
    affected peak lost mass. `affected_coefficient_share`, which `criteria.v1`
    names as TRUE_DUPLICATE required evidence, is understated with it, so
    adjudication reads the merge as safer than the data say. Nothing raises: every
    field stays inside its validated range.

    A signed comparison is not even a coherent alternative rule. `left` and `right`
    are chosen by `len()`, so which direction it tests depends on which table has
    more rows -- a peak's classification would depend on data outside that peak.
    """
    before = pd.DataFrame([
        {"peak_id": "moved", "hit_id": "kept", "coefficient": 1.0, "reconstruction": 0.0},
        {"peak_id": "untouched", "hit_id": "kept", "coefficient": 2.0, "reconstruction": 0.0},
    ])
    after = pd.DataFrame([
        {"peak_id": "moved", "hit_id": "kept", "coefficient": moved_to, "reconstruction": 1.0},
        {"peak_id": "untouched", "hit_id": "kept", "coefficient": 2.0, "reconstruction": 0.0},
    ])

    result = evaluate_stability("decision:direction", before, after)

    assert result.n_affected_peaks == 1, (
        f"a shared coefficient that moved {direction} was not counted as affected"
    )
    assert result.affected_coefficient_share > 0.0
    assert evaluate_stability(
        "decision:redistribution", before, after, coefficient_tolerance=None,
    ).n_affected_peaks == 0


def test_the_status_label_is_no_longer_diluted_by_peaks_that_only_wobbled():
    """The module's own dilution failure, one level down.

    `paired_delta_reconstruction_affected` is a median over the affected set, so
    padding that set with peaks whose delta is exactly zero drives the median to
    zero and prints STABLE_AFFECTED_SUBSET. On the case study this was not
    hypothetical: pairs 2, 5 and 6 were labelled STABLE over 753, 772 and 921
    "affected" peaks of which 172, 144 and 173 had actually changed, and all
    three become CHANGED once the set is the peaks that changed.
    """
    before = _refit_hit_table(changed=40, total=400, after=False)
    after = _refit_hit_table(changed=40, total=400, after=True)

    default = evaluate_stability("decision:dilution", before, after)
    exact = evaluate_stability("decision:dilution", before, after, coefficient_tolerance=0.0)

    assert exact.paired_delta_reconstruction_affected == 0.0
    assert exact.status == "STABLE_AFFECTED_SUBSET"
    assert default.paired_delta_reconstruction_affected == 1.0
    assert default.status == "CHANGED_AFFECTED_SUBSET"


def test_the_recorded_definition_is_the_one_that_produced_the_numbers():
    """The string is a re-derivation instruction, not a label.

    Parsing `affected_definition` back out and re-running under exactly that
    tolerance has to reproduce the record, or the field documents a rule other
    than the one applied.
    """
    from motifmultiverse.validate import _parse_affected_definition

    before = _refit_hit_table(changed=40, total=400, after=False)
    after = _refit_hit_table(changed=40, total=400, after=True)

    for tolerance in (None, 0.0, DEVICE_NULL_ABS_COEFFICIENT_DELTA, 1e-03):
        result = evaluate_stability(
            "decision:roundtrip", before, after, coefficient_tolerance=tolerance,
        )
        recovered = _parse_affected_definition(result.affected_definition)
        assert recovered == tolerance
        replayed = evaluate_stability(
            "decision:roundtrip", before, after, coefficient_tolerance=recovered,
        )
        assert replayed == result


def test_the_share_of_an_all_affected_run_is_one_on_both_supported_interpreters():
    """Naive float summation let a subset exceed the whole it is a subset of.

    `affected_coefficient_share` divided a sum taken over `affected_keys_after`
    -- a set, iterated in hash order -- by a sum over `after_coefficients`, a
    dict iterated in insertion order. When every peak is affected the two add
    exactly the same numbers in different orders, and float addition is not
    associative: the numerator landed one ulp above the denominator, the share
    came out 1.0000000000000004 and `StabilityResult` refused its own output.

    Python 3.12 concealed it, because builtin `sum` uses Neumaier compensated
    summation there and 3.11 does not -- so the share this package REPORTED
    differed between the two interpreters its classifiers claim. `math.fsum` is
    correctly rounded and order-independent, which is the property the ratio
    needed all along and the reason the number is now reproducible.
    """
    before = _refit_hit_table(changed=40, total=400, after=False)
    after = _refit_hit_table(changed=40, total=400, after=True)

    exact = evaluate_stability("decision:share", before, after, coefficient_tolerance=0.0)
    default = evaluate_stability("decision:share", before, after)

    assert exact.n_affected_peaks == 400              # the affected subset IS the whole
    assert exact.affected_coefficient_share == 1.0    # so the share is one, not one-plus-an-ulp
    assert 0.0 <= default.affected_coefficient_share < 1.0


@pytest.mark.parametrize("definition", [
    "",
    "   ",
    "everything_that_moved",
    "hit_set_change|abs_coefficient_delta>",
    "hit_set_change|abs_coefficient_delta>loose",
    "hit_set_change|abs_coefficient_delta>nan",
    "hit_set_change|abs_coefficient_delta>inf",
    "hit_set_change|abs_coefficient_delta>-1e-06",
])
def test_a_stability_result_refuses_an_affected_definition_no_reader_can_decode(definition):
    with pytest.raises(SchemaError, match="affected_definition"):
        StabilityResult(
            decision_id="d", n_affected_peaks=40, n_affected_hits=40,
            affected_coefficient_share=0.5, paired_delta_reconstruction_affected=1.0,
            paired_delta_reconstruction_all=0.0, hit_jaccard=None,
            coefficient_conservation=None, status="CHANGED_AFFECTED_SUBSET",
            power_statement="descriptive", affected_definition=definition,
        )


@pytest.mark.parametrize("tolerance", [-1e-09, float("nan"), float("inf"), True, "0.1"])
def test_evaluate_stability_refuses_a_tolerance_that_is_not_a_tolerance(tolerance):
    with pytest.raises(SchemaError, match="coefficient_tolerance"):
        evaluate_stability(
            "decision:bad-tolerance", _hit_table(), _hit_table(merged=True),
            coefficient_tolerance=tolerance,
        )


def test_run_backend_validation_records_the_definition_it_compared_backends_under():
    """Two backends are only comparable under one stated affected-set rule."""
    binding = LexiconBinding(
        lexicon_identity="lexicons:" + "0" * 64,
        entries=(("core", "a" * 64, "b" * 64, "c" * 64),),
    )
    results, _ = run_backend_validation(
        binding, "decision:definition", [_AvailableBackend()], coefficient_tolerance=None,
    )
    assert [row.affected_definition for row in results] == ["hit_set_change"]

    with pytest.raises(SchemaError, match="coefficient_tolerance"):
        run_backend_validation(
            binding, "decision:definition", [_AvailableBackend()], coefficient_tolerance=-1.0,
        )


def test_the_affected_definition_reaches_the_persisted_artifact(tmp_path):
    """A number whose definition lives only in a function signature is unreadable.

    `n_affected_peaks` moves by up to 5x with the definition, so the parquet row
    that carries it carries the rule beside it.
    """
    (
        binding, manifest, decision, validation, provenance, results, verification,
    ) = _valid_artifact_bundle(tmp_path)
    result_path, _ = write_stability_artifacts(
        tmp_path / "definition", results, verification, manifest=manifest,
        decision=decision, validation=validation, provenance=provenance, lexicon=binding,
    )

    frame = pd.read_parquet(result_path)
    assert list(frame["affected_definition"]) == [
        f"hit_set_change|abs_coefficient_delta>{DEVICE_NULL_ABS_COEFFICIENT_DELTA:g}"
    ]
    assert "affected_coefficient_share" in frame.columns
    assert "family_coefficient_share" not in frame.columns


def test_the_coefficient_share_field_is_named_for_the_subset_it_actually_measures():
    """It is affected-peak mass over total mass, and never was a family share.

    The standardized backend table is (peak_id, hit_id, coefficient,
    reconstruction): no family_id, no variant_id, no marker for the collapsed
    node. The per-family quantity the old name claimed is not computable from
    these inputs, so the name was corrected to the arithmetic rather than the
    arithmetic left disagreeing with the name. `criteria.v1.yaml` names this
    field as TRUE_DUPLICATE evidence, so the two must agree.

    Pinned against v1 BY NAME, not against "the default". v1 happens to be the
    default again, and that is beside the point: what this test owns is the field
    NAME, and v1 is the registry that speaks it. v2's TRUE_DUPLICATE requires the
    pair geometry and not this field at all -- a separate, deliberate regression,
    recorded as a known cost in the v2 criterion's own `declared_rationale` and
    pinned by tests/test_true_duplicate_criterion.py. If the default moves again,
    this assertion must not move with it.
    """
    from motifmultiverse.schema.criteria import load_criteria
    from motifmultiverse.validate import _REQUIRED_COLUMNS

    before = pd.DataFrame([
        {"peak_id": "affected", "hit_id": "GATA::v1", "coefficient": 1.0, "reconstruction": 0.0},
        {"peak_id": "quiet", "hit_id": "GATA::v2", "coefficient": 3.0, "reconstruction": 0.0},
    ])
    after = pd.DataFrame([
        {"peak_id": "affected", "hit_id": "GATA::v1", "coefficient": 1.0, "reconstruction": 1.0},
        {"peak_id": "affected", "hit_id": "GATA::v3", "coefficient": 1.0, "reconstruction": 1.0},
        {"peak_id": "quiet", "hit_id": "GATA::v2", "coefficient": 3.0, "reconstruction": 0.0},
    ])

    result = evaluate_stability("decision:share", before, after)

    # Affected mass / total mass -- 2 of 5 -- and NOT any grouping of the GATA
    # family the hit_id strings above happen to spell out.
    assert result.affected_coefficient_share == 0.4
    assert not hasattr(result, "family_coefficient_share")
    assert "family_id" not in _REQUIRED_COLUMNS and "variant_id" not in _REQUIRED_COLUMNS

    criteria = load_criteria(adjudicate.packaged_v1_criteria_path())
    assert "affected_coefficient_share" in criteria["TRUE_DUPLICATE"].required_evidence
    assert "family_coefficient_share" not in criteria["TRUE_DUPLICATE"].required_evidence


@pytest.mark.parametrize(
    "observed",
    [
        frozenset({"p-validation-a"}),
        frozenset({"p-validation-a", "p-validation-b", "p-discovery"}),
    ],
    ids=["missing-validation-peak", "extra-nonvalidation-peak"],
)
def test_frozen_backend_refuses_both_missing_and_extra_validation_peaks(
    tmp_path, observed,
):
    from motifmultiverse.validate import _FrozenHitTableBackend

    rows = [
        {
            "peak_id": peak_id,
            "hit_id": "family",
            "coefficient": 1.0,
            "reconstruction": 0.0,
            "substrate_id": "a" * 64,
        }
        for peak_id in sorted(observed)
    ]
    before = tmp_path / "before.parquet"
    after = tmp_path / "after.parquet"
    pd.DataFrame(rows).to_parquet(before, index=False)
    pd.DataFrame(rows).to_parquet(after, index=False)
    backend = _FrozenHitTableBackend(
        before, after, frozenset({"p-validation-a", "p-validation-b"}),
    )

    with pytest.raises(SchemaError, match="exactly the split-bound validation peaks"):
        backend.compare(object(), "decision:peaks")


@pytest.mark.parametrize(
    ("before_substrates", "after_substrates", "match"),
    [
        (None, ["a" * 64, "a" * 64], "substrate_id"),
        (["not-a-digest", "not-a-digest"], ["a" * 64, "a" * 64], "substrate_id"),
        (["a" * 64, "b" * 64], ["a" * 64, "a" * 64], "one substrate"),
        (["a" * 64, "a" * 64], ["b" * 64, "b" * 64], "same substrate"),
    ],
    ids=["missing", "malformed", "mixed", "before-after-mismatch"],
)
def test_frozen_backend_requires_one_shared_canonical_substrate_identity(
    tmp_path, before_substrates, after_substrates, match,
):
    from motifmultiverse.validate import _FrozenHitTableBackend

    def rows(substrates):
        values = [
            {
                "peak_id": peak_id,
                "hit_id": "family",
                "coefficient": 1.0,
                "reconstruction": 0.0,
            }
            for peak_id in ("p-validation-a", "p-validation-b")
        ]
        if substrates is not None:
            for row, substrate_id in zip(values, substrates, strict=True):
                row["substrate_id"] = substrate_id
        return values

    before = tmp_path / "before.parquet"
    after = tmp_path / "after.parquet"
    pd.DataFrame(rows(before_substrates)).to_parquet(before, index=False)
    pd.DataFrame(rows(after_substrates)).to_parquet(after, index=False)
    backend = _FrozenHitTableBackend(
        before, after, frozenset({"p-validation-a", "p-validation-b"}),
    )

    with pytest.raises(SchemaError, match=match):
        backend.compare(object(), "decision:substrate")


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
    provenance = _stability_provenance(
        binding, manifest, decision, provisional,
    )
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
    assert json.loads(metadata[b"motifmultiverse.provenance"]) == provenance.to_dict()
    result_rows = pq.read_table(results_path).to_pandas()
    assert json.loads(result_rows.loc[0, "provenance"]) == provenance.to_dict()
    assert json.loads((results_path.parent / "provenance.json").read_text()) == [
        provenance.to_dict(),
    ]
    verification_rows = pd.read_csv(verification_path, sep="\t")
    assert json.loads(verification_rows.loc[0, "provenance"]) == provenance.to_dict()
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


def _lexicons(tmp_path):
    import h5py
    import numpy as np

    from motifmultiverse.compile import lexicon_semantic_hash

    lexicons = tmp_path / "lexicons"
    lexicons.mkdir()
    motif_arrays = {
        "node-0": {"cwm": np.asarray([[1.0, 0.0, 0.0, 0.0]])},
        "node-1": {"cwm": np.asarray([[0.0, 1.0, 0.0, 0.0]])},
    }
    with h5py.File(lexicons / "core.h5", "w") as h5:
        patterns = h5.create_group("pos_patterns")
        for number, node_id in enumerate(motif_arrays):
            patterns.create_group(f"pattern_{number}").create_dataset(
                "contrib_scores", data=motif_arrays[node_id]["cwm"],
            )
    ordered = [
        ("pos_patterns", f"pattern_{number}",
         {"node_id": node_id, "variant_id": f"MA_FAM_{number + 1:02d}"})
        for number, node_id in enumerate(motif_arrays)
    ]
    content_hash = lexicon_semantic_hash(
        ordered,
        motif_arrays,
        schema_version="1.0", trim_threshold=0.3, motif_type="cwm", include_rc=False,
        loader_backend="finemo", loader_parameters={"motif_lambda_default": 0.7},
    )
    (lexicons / "core.manifest.json").write_text(json.dumps({
        "tier": "core", "lexicon_content_hash": content_hash, "n_motifs": 2,
        "pattern_order": ["pos_patterns.pattern_0", "pos_patterns.pattern_1"],
        "node_ids": ["node-0", "node-1"],
        "index": [
            {
                "index": number,
                "pattern_tag": f"pos_patterns.pattern_{number}",
                "node_id": node_id,
                "variant_id": f"MA_FAM_{number + 1:02d}",
                "metacluster": "pos",
            }
            for number, node_id in enumerate(motif_arrays)
        ],
        "schema_version": "1.0", "trim_threshold": 0.3, "motif_type": "cwm",
        "include_rc": False, "loader_backend": "finemo",
        "loader_parameters": {"motif_lambda_default": 0.7},
        "comparisons": {}, "source_registry": "registry", "sensitivity_triggers": {},
        "project": "test-project", "cross_model_claims_restricted": True,
    }), encoding="utf-8")
    return lexicons


@pytest.mark.parametrize("extra_kind", ["root_group", "motif", "dataset"])
def test_lexicon_binding_rejects_every_extra_hdf5_object(tmp_path, extra_kind):
    """Deleting exact HDF5-universe comparison makes each mutation pass."""
    import h5py
    import numpy as np

    lexicons = _lexicons(tmp_path)
    with h5py.File(lexicons / "core.h5", "a") as h5:
        if extra_kind == "root_group":
            h5.create_group("unindexed_root")
        elif extra_kind == "motif":
            h5["pos_patterns"].create_group("pattern_2").create_dataset(
                "contrib_scores", data=np.asarray([[0.0, 0.0, 1.0, 0.0]]),
            )
        else:
            h5["pos_patterns"]["pattern_0"].create_dataset(
                "unindexed_scores", data=np.asarray([[9.0, 9.0, 9.0, 9.0]]),
            )

    with pytest.raises(SchemaError, match="compiled lexicon|HDF5|universe"):
        load_lexicon_binding(lexicons)


def test_lexicon_binding_rejects_a_manifest_that_omits_a_real_hdf5_motif(tmp_path):
    """An internally consistent one-row manifest cannot hide the second loader motif."""
    import numpy as np

    from motifmultiverse.compile import lexicon_semantic_hash

    lexicons = _lexicons(tmp_path)
    manifest_path = lexicons / "core.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["n_motifs"] = 1
    payload["pattern_order"] = payload["pattern_order"][:1]
    payload["node_ids"] = payload["node_ids"][:1]
    payload["index"] = payload["index"][:1]
    payload["lexicon_content_hash"] = lexicon_semantic_hash(
        [("pos_patterns", "pattern_0",
          {"node_id": "node-0", "variant_id": "MA_FAM_01"})],
        {"node-0": {"cwm": np.asarray([[1.0, 0.0, 0.0, 0.0]])}},
        schema_version="1.0", trim_threshold=0.3, motif_type="cwm", include_rc=False,
        loader_backend="finemo", loader_parameters={"motif_lambda_default": 0.7},
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SchemaError, match="index|universe|motif"):
        load_lexicon_binding(lexicons)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(unknown_top_level="silently ignored"),
        lambda payload: payload.pop("project"),
        lambda payload: payload.update(cross_model_claims_restricted=1),
        lambda payload: payload["index"][0].update(unknown_index_field="silently ignored"),
        lambda payload: payload["index"][0].pop("variant_id"),
        lambda payload: payload["index"][0].update(index=True),
    ],
    ids=[
        "unknown-top-level", "missing-top-level", "wrong-top-level-type",
        "unknown-index-field", "missing-index-field", "wrong-index-type",
    ],
)
def test_lexicon_binding_requires_the_exact_compiler_manifest_schema(tmp_path, mutation):
    lexicons = _lexicons(tmp_path)
    manifest_path = lexicons / "core.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(payload)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SchemaError, match="manifest|index|schema"):
        load_lexicon_binding(lexicons)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("comparisons", {"expanded": []}),
        ("comparisons", {"core": {"positive_sets_identical": True}}),
        ("sensitivity_triggers", {"cluster": "threshold_sensitive"}),
        ("sensitivity_triggers", {"cluster": ["threshold_sensitive", 7]}),
    ],
    ids=[
        "comparison-not-object",
        "comparison-wrong-shape",
        "triggers-not-array",
        "trigger-not-string",
    ],
)
def test_lexicon_binding_requires_exact_nested_manifest_schemas(tmp_path, field, value):
    lexicons = _lexicons(tmp_path)
    manifest_path = lexicons / "core.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload[field] = value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SchemaError, match="comparisons|sensitivity_triggers|schema"):
        load_lexicon_binding(lexicons)


@pytest.mark.parametrize("identity", ["pattern", "node", "variant"])
def test_lexicon_binding_rejects_duplicate_manifest_identities(tmp_path, identity):
    lexicons = _lexicons(tmp_path)
    manifest_path = lexicons / "core.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if identity == "pattern":
        payload["pattern_order"][1] = payload["pattern_order"][0]
        payload["index"][1]["pattern_tag"] = payload["index"][0]["pattern_tag"]
    elif identity == "node":
        payload["node_ids"][1] = payload["node_ids"][0]
        payload["index"][1]["node_id"] = payload["index"][0]["node_id"]
    else:
        payload["index"][1]["variant_id"] = payload["index"][0]["variant_id"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SchemaError, match=f"duplicate.*{identity}"):
        load_lexicon_binding(lexicons)


def test_lexicon_binding_rejects_a_malformed_variant_identity(tmp_path):
    lexicons = _lexicons(tmp_path)
    manifest_path = lexicons / "core.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["index"][0]["variant_id"] = "semantic identity with spaces"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SchemaError, match="variant"):
        load_lexicon_binding(lexicons)


@pytest.mark.parametrize(
    "mutation",
    ["motif-is-dataset", "contrib-is-group", "nonnumeric-contrib", "wrong-float-dtype"],
)
def test_lexicon_binding_rejects_malformed_hdf5_structure(tmp_path, mutation):
    import h5py
    import numpy as np

    lexicons = _lexicons(tmp_path)
    with h5py.File(lexicons / "core.h5", "a") as h5:
        patterns = h5["pos_patterns"]
        if mutation == "motif-is-dataset":
            del patterns["pattern_1"]
            patterns.create_dataset("pattern_1", data=np.asarray([1.0]))
        elif mutation == "contrib-is-group":
            motif = patterns["pattern_1"]
            del motif["contrib_scores"]
            motif.create_group("contrib_scores")
        elif mutation == "nonnumeric-contrib":
            motif = patterns["pattern_1"]
            del motif["contrib_scores"]
            motif.create_dataset(
                "contrib_scores", data=np.asarray([["a", "b", "c", "d"]], dtype="S1"),
            )
        else:
            motif = patterns["pattern_1"]
            values = motif["contrib_scores"][:]
            del motif["contrib_scores"]
            motif.create_dataset("contrib_scores", data=values.astype("float32"))

    with pytest.raises(SchemaError, match="HDF5|dataset|numeric|float64|compiled lexicon"):
        load_lexicon_binding(lexicons)


@pytest.mark.parametrize("link_kind", ["external", "soft", "hard-alias"])
def test_lexicon_binding_rejects_nonlocal_or_aliased_hdf5_links(tmp_path, link_kind):
    import h5py
    import numpy as np

    from motifmultiverse.compile import lexicon_semantic_hash

    lexicons = _lexicons(tmp_path)
    h5_path = lexicons / "core.h5"
    manifest_path = lexicons / "core.manifest.json"
    if link_kind == "external":
        external = tmp_path / "external.h5"
        with h5py.File(h5_path, "r") as source, h5py.File(external, "w") as target:
            source.copy("pos_patterns", target)
        with h5py.File(h5_path, "a") as h5:
            del h5["pos_patterns"]
            h5["pos_patterns"] = h5py.ExternalLink(str(external), "/pos_patterns")
    else:
        with h5py.File(h5_path, "a") as h5:
            del h5["pos_patterns"]["pattern_1"]
            if link_kind == "soft":
                h5["pos_patterns"]["pattern_1"] = h5py.SoftLink(
                    "/pos_patterns/pattern_0",
                )
            else:
                h5["pos_patterns"]["pattern_1"] = h5["pos_patterns"]["pattern_0"]
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        arrays = {
            "node-0": {"cwm": np.asarray([[1.0, 0.0, 0.0, 0.0]])},
            "node-1": {"cwm": np.asarray([[1.0, 0.0, 0.0, 0.0]])},
        }
        payload["lexicon_content_hash"] = lexicon_semantic_hash(
            [
                ("pos_patterns", "pattern_0",
                 {"node_id": "node-0", "variant_id": "MA_FAM_01"}),
                ("pos_patterns", "pattern_1",
                 {"node_id": "node-1", "variant_id": "MA_FAM_02"}),
            ],
            arrays,
            schema_version="1.0",
            trim_threshold=0.3,
            motif_type="cwm",
            include_rc=False,
            loader_backend="finemo",
            loader_parameters={"motif_lambda_default": 0.7},
        )
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SchemaError, match="link|alias|compiler-emitted"):
        load_lexicon_binding(lexicons)


def test_lexicon_binding_requires_the_dataset_selected_by_motif_type(tmp_path):
    import numpy as np

    from motifmultiverse.compile import lexicon_semantic_hash

    lexicons = _lexicons(tmp_path)
    manifest_path = lexicons / "core.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["motif_type"] = "hcwm"
    payload["lexicon_content_hash"] = lexicon_semantic_hash(
        [
            ("pos_patterns", "pattern_0",
             {"node_id": "node-0", "variant_id": "MA_FAM_01"}),
            ("pos_patterns", "pattern_1",
             {"node_id": "node-1", "variant_id": "MA_FAM_02"}),
        ],
        {
            "node-0": {"cwm": np.asarray([[1.0, 0.0, 0.0, 0.0]])},
            "node-1": {"cwm": np.asarray([[0.0, 1.0, 0.0, 0.0]])},
        },
        schema_version="1.0",
        trim_threshold=0.3,
        motif_type="hcwm",
        include_rc=False,
        loader_backend="finemo",
        loader_parameters={"motif_lambda_default": 0.7},
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SchemaError, match="motif_type|hypothetical_contribs"):
        load_lexicon_binding(lexicons)


def test_lexicon_binding_requires_a_real_content_addressed_compiled_lexicon(tmp_path):
    with pytest.raises(SchemaError, match="does not exist"):
        load_lexicon_binding(tmp_path / "missing")
    lexicons = _lexicons(tmp_path)
    binding = load_lexicon_binding(lexicons)
    assert isinstance(binding, LexiconBinding)
    assert binding.lexicon_identity.startswith("lexicons:")
    (lexicons / "core.h5").write_bytes(b"tampered")
    with pytest.raises(SchemaError, match="valid compiled lexicon|readable HDF5"):
        load_lexicon_binding(lexicons)


def test_backend_results_are_bound_to_the_backend_and_lexicon_identity(tmp_path):
    binding = load_lexicon_binding(_lexicons(tmp_path))
    results, verification = run_backend_validation(
        binding, "decision:backend", [_AvailableBackend()],
    )

    assert results[0].backend == verification[0].backend == "available"
    assert results[0].backend_result_id == verification[0].backend_result_id
    assert results[0].lexicon_identity == verification[0].lexicon_identity == binding.lexicon_identity


def _stability_provenance(binding, manifest, decision, validation, **changes):
    base = ProvenanceRecord(
        command="motifmultiverse validate fixture",
        subcommand="validate",
        inputs={
            "before.parquet": "1" * 64,
            "after.parquet": "2" * 64,
            "split-manifest.json": "3" * 64,
            "decision-split.json": "4" * 64,
        },
        software={"motifmultiverse": "0.test", "python": "3.test"},
        random_seed=17,
        input_scale=2,
        substrate_id="d" * 64,
        timestamp_utc="2026-07-26T12:00:00Z",
        schema_version="1",
        redaction_policy="basenames_only_except_command",
    )
    provenance = StabilityProvenance.from_record(
        base,
        lexicon=binding,
        manifest=manifest,
        decision=decision,
        validation=validation,
    )
    return replace(provenance, **changes) if changes else provenance


def _identity_context(tmp_path, *, decision_id="decision:canonical"):
    binding = load_lexicon_binding(_lexicons(tmp_path))
    manifest = _manifest()
    decision = DecisionSplitArtifact.create(
        manifest=manifest, decision_id=decision_id,
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation-a"}),
    )
    provisional = ValidationSplitArtifact.create(
        manifest=manifest, decision_id=decision_id, result_id="pending",
        decision_peak_ids=decision.decision_peak_ids,
        validation_peak_ids=decision.validation_peak_ids,
    )
    provenance = _stability_provenance(
        binding, manifest, decision, provisional,
    )
    return binding, manifest, decision, provisional, provenance


def _valid_artifact_bundle(tmp_path, *, decision_id="decision:publication"):
    binding, manifest, decision, provisional, provenance = _identity_context(
        tmp_path, decision_id=decision_id,
    )
    results, verification = run_backend_validation(
        binding, decision.decision_id, [_AvailableBackend()],
    )
    validation = replace(
        provisional,
        result_id=stability_result_id(
            results, verification, provenance,
            binding, manifest, decision, provisional,
        ),
        artifact_id="",
    )
    return (
        binding, manifest, decision, validation, provenance,
        list(results), list(verification),
    )


def test_stability_result_id_canonicalizes_complete_records_when_partial_keys_tie(tmp_path):
    binding, manifest, decision, provisional, provenance = _identity_context(tmp_path)
    first = replace(
        _stability_result(),
        decision_id=decision.decision_id,
        backend_result_id="",
        paired_delta_reconstruction_all=0.0,
    )
    second = replace(
        first,
        paired_delta_reconstruction_all=2.0,
        power_statement=first.power_statement + " Distinct complete record.",
    )
    verification = [
        BackendVerification(
            "tied", "1", "VERIFIED", "same partial key",
            backend_result_id="result:a", lexicon_identity=binding.lexicon_identity,
        ),
        BackendVerification(
            "tied", "1", "VERIFIED", "same partial key",
            backend_result_id="result:b", lexicon_identity=binding.lexicon_identity,
        ),
    ]

    forward = stability_result_id(
        [first, second], verification, provenance,
        binding, manifest, decision, provisional,
    )
    reversed_inputs = stability_result_id(
        [second, first], list(reversed(verification)), provenance,
        binding, manifest, decision, provisional,
    )

    assert reversed_inputs == forward


@pytest.mark.parametrize(
    "field,value",
    [
        ("command", "motifmultiverse validate changed"),
        ("timestamp_utc", "2026-07-26T12:00:01Z"),
        ("random_seed", 18),
        ("input_scale", 3),
        ("substrate_id", "e" * 64),
        ("software", {"motifmultiverse": "0.other", "python": "3.test"}),
        ("inputs", {"before.parquet": "f" * 64}),
    ],
)
def test_stability_result_id_binds_every_full_provenance_field(tmp_path, field, value):
    binding, manifest, decision, provisional, provenance = _identity_context(tmp_path)

    original = stability_result_id(
        [], [], provenance, binding, manifest, decision, provisional,
    )
    changed = stability_result_id(
        [], [], replace(provenance, **{field: value}),
        binding, manifest, decision, provisional,
    )

    assert changed != original


def test_stability_result_id_refuses_unvalidated_public_inputs_before_hashing(tmp_path):
    binding, manifest, decision, provisional, provenance = _identity_context(tmp_path)

    with pytest.raises(SchemaError, match="StabilityResult"):
        stability_result_id(
            [object()], [], provenance, binding, manifest, decision, provisional,
        )
    with pytest.raises(SchemaError, match="provenance"):
        stability_result_id(
            [], [], {"stage": "validate"}, binding, manifest, decision, provisional,
        )
    with pytest.raises(SchemaError, match="LexiconBinding"):
        stability_result_id(
            [], [], provenance, "not-a-binding", manifest, decision, provisional,
        )


def test_stability_provenance_rejects_tamper_missing_and_wrong_types(tmp_path):
    binding, manifest, decision, provisional, provenance = _identity_context(tmp_path)

    with pytest.raises(SchemaError, match="lexicon"):
        stability_result_id(
            [], [], replace(provenance, lexicon_identity="lexicons:" + "f" * 64),
            binding, manifest, decision, provisional,
        )
    for field, value in (
        ("split_manifest_checksum", "f" * 64),
        ("decision_artifact_id", "decision-split:" + "f" * 64),
        ("validation_split_identity", "f" * 64),
    ):
        with pytest.raises(SchemaError, match="split identity|manifest identity"):
            stability_result_id(
                [], [], replace(provenance, **{field: value}),
                binding, manifest, decision, provisional,
            )
    with pytest.raises(SchemaError, match="command"):
        replace(provenance, command="")
    with pytest.raises(SchemaError, match="inputs"):
        replace(provenance, inputs={"before.parquet": 7})
    for substrate_id in (None, "not-a-substrate-digest"):
        with pytest.raises(SchemaError, match="substrate_id"):
            replace(provenance, substrate_id=substrate_id)


def test_stability_writer_rejects_a_reduced_mapping_even_when_nonempty(tmp_path):
    (
        binding, manifest, decision, validation, _provenance,
        results, verification,
    ) = _valid_artifact_bundle(tmp_path)

    with pytest.raises(SchemaError, match="exact StabilityProvenance"):
        write_stability_artifacts(
            tmp_path / "out",
            results,
            verification,
            manifest=manifest,
            decision=decision,
            validation=validation,
            provenance={"stage": "validate"},
            lexicon=binding,
        )

    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda results, verification, lexicon: (
                results,
                [*verification, verification[0]],
            ),
            "unique backend identities",
        ),
        (
            lambda results, verification, lexicon: (
                [*results, results[0]],
                verification,
            ),
            "unique backend_result_id",
        ),
        (
            lambda results, verification, lexicon: (
                results,
                [
                    *verification,
                    BackendVerification(
                        "orphan", "1", "VERIFIED",
                        backend_result_id="backend-result:orphan",
                        lexicon_identity=lexicon.lexicon_identity,
                    ),
                ],
            ),
            "each VERIFIED backend row",
        ),
        (
            lambda results, verification, lexicon: (results, []),
            "each VERIFIED backend row",
        ),
        (
            lambda results, verification, lexicon: (
                results,
                [
                    *verification,
                    BackendVerification(
                        "ambiguous", "1", "UNVERIFIED", "not run",
                        backend_result_id="backend-result:claimed",
                        lexicon_identity=lexicon.lexicon_identity,
                    ),
                ],
            ),
            "UNVERIFIED backend rows cannot claim",
        ),
    ],
    ids=[
        "duplicate-backend-identity",
        "duplicate-result-identity",
        "orphan-verified-association",
        "missing-verified-association",
        "unverified-claims-result",
    ],
)
def test_stability_writer_refuses_duplicate_or_ambiguous_backend_associations(
    tmp_path, mutation, match,
):
    (
        binding, manifest, decision, validation, provenance,
        results, verification,
    ) = _valid_artifact_bundle(tmp_path)
    mutated_results, mutated_verification = mutation(
        results, verification, binding,
    )

    with pytest.raises(SchemaError, match=match):
        write_stability_artifacts(
            tmp_path / "out",
            mutated_results,
            mutated_verification,
            manifest=manifest,
            decision=decision,
            validation=validation,
            provenance=provenance,
            lexicon=binding,
        )

    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "failure_point",
    ["parquet", "tsv", "provenance", "rename"],
)
def test_stability_publication_failure_cleans_only_its_unique_owned_stage(
    tmp_path, monkeypatch, failure_point,
):
    import csv
    from pathlib import Path

    import pyarrow.parquet as pq

    import motifmultiverse.validate as validate_mod

    (
        binding, manifest, decision, validation, provenance,
        results, verification,
    ) = _valid_artifact_bundle(tmp_path)
    out = tmp_path / "publication"
    unrelated_stage = tmp_path / ".publication.stage-unrelated"
    unrelated_stage.mkdir()
    (unrelated_stage / "owner.txt").write_text("someone else", encoding="utf-8")
    unrelated_sibling = tmp_path / "unrelated-sibling"
    unrelated_sibling.write_text("preserve", encoding="utf-8")

    def injected_failure(*_args, **_kwargs):
        raise OSError(f"injected {failure_point} failure")

    if failure_point == "parquet":
        monkeypatch.setattr(pq, "write_table", injected_failure)
    elif failure_point == "tsv":
        monkeypatch.setattr(csv.DictWriter, "writeheader", injected_failure)
    elif failure_point == "provenance":
        original_write_text = Path.write_text

        def fail_provenance(path, *args, **kwargs):
            if path.name == "provenance.json":
                raise OSError("injected provenance failure")
            return original_write_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_provenance)
    else:
        monkeypatch.setattr(
            validate_mod, "_publish_directory_noreplace", injected_failure,
        )

    with pytest.raises(OSError, match=f"injected {failure_point} failure"):
        write_stability_artifacts(
            out,
            results,
            verification,
            manifest=manifest,
            decision=decision,
            validation=validation,
            provenance=provenance,
            lexicon=binding,
        )

    assert not out.exists()
    assert (unrelated_stage / "owner.txt").read_text(encoding="utf-8") == "someone else"
    assert unrelated_sibling.read_text(encoding="utf-8") == "preserve"
    assert list(tmp_path.glob(".publication.stage-*")) == [unrelated_stage]


def test_stability_writer_never_touches_an_existing_output(tmp_path):
    (
        binding, manifest, decision, validation, provenance,
        results, verification,
    ) = _valid_artifact_bundle(tmp_path)
    out = tmp_path / "publication"
    out.mkdir()
    sentinel = out / "existing.txt"
    sentinel.write_text("original", encoding="utf-8")
    unrelated_stage = tmp_path / ".publication.stage-unrelated"
    unrelated_stage.mkdir()

    with pytest.raises(SchemaError, match="will not be overwritten"):
        write_stability_artifacts(
            out,
            results,
            verification,
            manifest=manifest,
            decision=decision,
            validation=validation,
            provenance=provenance,
            lexicon=binding,
        )

    assert sentinel.read_text(encoding="utf-8") == "original"
    assert list(out.iterdir()) == [sentinel]
    assert unrelated_stage.is_dir()


def test_stability_writer_does_not_clobber_an_output_created_at_publish_time(
    tmp_path, monkeypatch,
):
    import motifmultiverse.validate as validate_mod

    (
        binding, manifest, decision, validation, provenance,
        results, verification,
    ) = _valid_artifact_bundle(tmp_path)
    out = tmp_path / "publication"
    original_publish = validate_mod._publish_directory_noreplace

    def racing_publish(stage, target):
        out.mkdir()
        return original_publish(stage, target)

    monkeypatch.setattr(
        validate_mod, "_publish_directory_noreplace", racing_publish,
    )

    with pytest.raises(SchemaError, match="will not be overwritten|already exists"):
        write_stability_artifacts(
            out,
            results,
            verification,
            manifest=manifest,
            decision=decision,
            validation=validation,
            provenance=provenance,
            lexicon=binding,
        )

    assert out.is_dir()
    assert list(out.iterdir()) == []
    assert not list(tmp_path.glob(".publication.stage-*"))


def test_stability_writer_treats_a_dangling_output_symlink_as_existing(tmp_path):
    (
        binding, manifest, decision, validation, provenance,
        results, verification,
    ) = _valid_artifact_bundle(tmp_path)
    out = tmp_path / "publication"
    out.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(SchemaError, match="will not be overwritten|already exists"):
        write_stability_artifacts(
            out,
            results,
            verification,
            manifest=manifest,
            decision=decision,
            validation=validation,
            provenance=provenance,
            lexicon=binding,
        )

    assert out.is_symlink()
    assert out.readlink() == tmp_path / "missing-target"


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
    provenance = _stability_provenance(
        binding, manifest, decision, validation,
    )
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
    provenance = _stability_provenance(
        binding, manifest, decision, provisional,
    )
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
    provenance = _stability_provenance(
        binding, manifest, decision, provisional,
    )
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


def test_affected_subset_scan_stays_linear_in_the_hit_table():
    """`evaluate_stability` rebuilt the affected-peak set once per hit row.

    The membership tests were written `key[0] in set(affected)` inside a
    comprehension, so the set was reconstructed from the sorted list for every
    hit in the table and the scan became quadratic. A validation of 80,000 hits
    over 16,000 affected peaks took 99 seconds; the same call now takes about
    1.7. This is a wall-clock test because the defect has no other observable --
    the numbers were always right, only the number of times the same set was
    built was wrong -- so the ceiling is set an order of magnitude above the
    linear time and well below the quadratic one, and the result is asserted
    alongside it so a "fix" that changed an answer would fail here too.
    """
    import time

    n_peaks, hits_per_peak = 16_000, 5
    before_rows, after_rows = [], []
    for peak in range(n_peaks):
        for hit in range(hits_per_peak):
            common = {"peak_id": f"peak-{peak:06d}", "hit_id": f"hit-{hit}",
                      "reconstruction": 0.25}
            # Varying, not constant: a constant coefficient column makes the
            # conservation correlation undefined and buries this test in numpy
            # divide-by-zero warnings that have nothing to do with what it checks.
            before_rows.append({**common, "coefficient": 1.0 + hit})
            after_rows.append({**common, "coefficient": 2.0 + hit})
    before, after = pd.DataFrame(before_rows), pd.DataFrame(after_rows)

    started = time.perf_counter()
    result = evaluate_stability("decision:linear", before, after)
    elapsed = time.perf_counter() - started

    assert result.n_affected_peaks == n_peaks
    assert result.n_affected_hits == n_peaks * hits_per_peak
    assert result.hit_jaccard == 1.0
    assert elapsed < 20.0, (
        f"{len(before_rows)} hits took {elapsed:.1f}s; the quadratic rebuild took 99s "
        "and the linear scan takes under 2s"
    )


def test_an_exploratory_run_is_marked_exploratory_in_the_artifacts_it_writes(tmp_path):
    """A run whose validation peaks ARE its decision peaks must say so on paper.

    `AnalysisMode.EXPLORATORY` is the waiver `assert_split_compatibility` tells a
    caller to declare when the two peak sets overlap, so it is the record that the
    result is nonconfirmatory. That record reached the artifact only inside the
    SHA-256 of the validation split identity, which means a reader comparing a
    reused-peak run against a genuinely held-out one saw two files with identical
    columns, identical statuses and identical power statements. A waiver nobody
    can read is not a waiver.
    """
    import pyarrow.parquet as pq

    binding = load_lexicon_binding(_lexicons(tmp_path))
    manifest = _manifest()
    reused = frozenset({"p-discovery"})
    decision = DecisionSplitArtifact.create(
        manifest=manifest, decision_id="decision:reuse",
        decision_peak_ids=reused, validation_peak_ids=reused,
        mode=AnalysisMode.EXPLORATORY,
    )
    provisional = ValidationSplitArtifact.create(
        manifest=manifest, decision_id=decision.decision_id, result_id="pending",
        decision_peak_ids=reused, validation_peak_ids=reused,
        mode=AnalysisMode.EXPLORATORY,
    )
    provenance = _stability_provenance(binding, manifest, decision, provisional)
    results, verification = run_backend_validation(
        binding, decision.decision_id, [_AvailableBackend()],
    )
    validation = replace(
        provisional,
        result_id=stability_result_id(
            results, verification, provenance, binding, manifest, decision, provisional,
        ),
        artifact_id="",
    )
    out = tmp_path / "exploratory"
    result_path, verification_path = write_stability_artifacts(
        out, results, verification, manifest=manifest, decision=decision,
        validation=validation, provenance=provenance, lexicon=binding,
    )

    frame = pd.read_parquet(result_path)
    assert list(frame["analysis_mode"]) == ["EXPLORATORY"] * len(frame)
    metadata = pq.read_schema(result_path).metadata
    assert metadata[b"motifmultiverse.analysis_mode"] == b"EXPLORATORY"
    header, *rows = verification_path.read_text(encoding="utf-8").strip().split("\n")
    column = header.split("\t").index("analysis_mode")
    assert {row.split("\t")[column] for row in rows} == {"EXPLORATORY"}


def test_a_primary_run_is_marked_primary_rather_than_left_blank(tmp_path):
    """The mode column is not an EXPLORATORY-only annotation.

    A field that appears only when something is wrong is read as noise the first
    time it is empty; the twin of the exploratory test pins that a clean
    held-out run states its mode too, so the column always carries a claim.
    """
    (
        binding, manifest, decision, validation, provenance, results, verification,
    ) = _valid_artifact_bundle(tmp_path)
    result_path, verification_path = write_stability_artifacts(
        tmp_path / "primary", results, verification, manifest=manifest,
        decision=decision, validation=validation, provenance=provenance, lexicon=binding,
    )

    assert list(pd.read_parquet(result_path)["analysis_mode"]) == ["PRIMARY"]
    assert "PRIMARY" in verification_path.read_text(encoding="utf-8")
