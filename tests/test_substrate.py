"""Frozen hit-substrate identity tests."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from motifmultiverse import interpret
from motifmultiverse.schema import SchemaError
from motifmultiverse.schema.substrate import CallerSpecification, HitSubstrateManifest
from motifmultiverse.substrate import (
    SubstrateError,
    build_manifest,
    compute_substrate_id,
    read_manifest,
    write_manifest,
)


def _manifest(*, lambda_value: float):
    specification = CallerSpecification(
        caller_name="finemo",
        caller_version="0.3.1",
        lexicon_content_hash="a" * 64,
        parameters={"lambda": lambda_value, "min_score": 0.25},
        preprocessing_contract_hash="b" * 64,
    )
    return build_manifest(
        peak_universe_hash="c" * 64,
        n_regions=42,
        caller_specification=specification,
        input_files={"peaks.tsv": "d" * 64},
        created_at="2026-07-25T00:00:00Z",
    )


def test_substrate_id_is_deterministic_and_parameter_sensitive():
    """Changing caller lambda must make a different frozen substrate."""
    a = _manifest(lambda_value=0.7)
    b = _manifest(lambda_value=0.7)
    c = _manifest(lambda_value=0.8)

    assert a.substrate_id == b.substrate_id
    assert a.substrate_id != c.substrate_id


def test_hit_table_cannot_mix_substrate_ids(tmp_path):
    """Rows from distinct caller specifications cannot masquerade as one run."""
    table = tmp_path / "mixed.tsv"
    table.write_text(
        "\t".join([
            "region_id", "chrom", "start", "end", "variant_id", "family_id",
            "hit_coefficient", "missingness", "input_scale", "lexicon_id", "substrate_id",
        ])
        + "\n"
        + "r1\tchr1\t0\t10\tUA_FAMA_01\tFAM_A\t1.0\tused\t9999\tlex_v1\t"
        + "a" * 64
        + "\n"
        + "r2\tchr1\t10\t20\tUA_FAMA_02\tFAM_A\t1.0\tused\t9999\tlex_v1\t"
        + "b" * 64
        + "\n"
    )

    with pytest.raises(interpret.InterpretError, match="mixes substrates"):
        interpret.read_hit_table(table)


def test_identity_changes_for_every_semantic_caller_input():
    """Caller, version, lexicon, parameters, preprocessing, and universe all bind identity."""
    baseline = _manifest(lambda_value=0.7)
    specification = baseline.caller_specification
    variants = [
        replace(specification, caller_name="fimo"),
        replace(specification, caller_version="0.3.2"),
        replace(specification, lexicon_content_hash="e" * 64),
        replace(specification, parameters={"lambda": 0.8, "min_score": 0.25}),
        replace(specification, preprocessing_contract_hash="f" * 64),
    ]
    manifests = [
        build_manifest(
            peak_universe_hash="c" * 64, n_regions=42, caller_specification=changed,
            input_files={"peaks.tsv": "d" * 64}, created_at="2026-07-25T00:00:00Z",
        )
        for changed in variants
    ]
    manifests.extend([
        build_manifest(
            peak_universe_hash="f" * 64, n_regions=42, caller_specification=specification,
            input_files={"peaks.tsv": "d" * 64}, created_at="2026-07-25T00:00:00Z",
        ),
        build_manifest(
            peak_universe_hash="c" * 64, n_regions=43, caller_specification=specification,
            input_files={"peaks.tsv": "d" * 64}, created_at="2026-07-25T00:00:00Z",
        ),
    ])

    assert all(candidate.substrate_id != baseline.substrate_id for candidate in manifests)


@pytest.mark.parametrize("bad", [
    {"parameter": float("nan")},
    {"parameter": float("inf")},
    {"parameter": {"unordered"}},
    {"input_file": "/private/project/peaks.tsv"},
    {"parameters": {"working_dir": "/private/project"}},
])
def test_canonical_identity_refuses_ambiguous_or_path_dependent_values(bad):
    """Non-JSON and machine-dependent values must not silently become an identity."""
    with pytest.raises(SubstrateError):
        compute_substrate_id(bad)


def test_manifest_keeps_file_checksums_as_provenance_not_semantic_identity(tmp_path):
    """Changing checksum provenance alone cannot create a second caller substrate."""
    first = _manifest(lambda_value=0.7)
    second = build_manifest(
        peak_universe_hash=first.peak_universe_hash, n_regions=first.n_regions,
        caller_specification=first.caller_specification,
        input_files={"renamed-input.tsv": "e" * 64}, created_at="2026-07-26T00:00:00Z",
    )
    assert second.substrate_id == first.substrate_id

    path = write_manifest(first, tmp_path / "substrate.manifest.json")
    payload = json.loads(path.read_text())
    payload["substrate_id"] = "f" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(SubstrateError):
        read_manifest(path)


def test_manifest_schema_refuses_wrong_runtime_types():
    """Invalid manifest shapes must become structured refusals, not later crashes."""
    with pytest.raises(SchemaError):
        CallerSpecification(
            caller_name="finemo", caller_version="0.3.1", lexicon_content_hash="a" * 64,
            parameters=[], preprocessing_contract_hash="b" * 64,
        )
    with pytest.raises(SchemaError):
        HitSubstrateManifest(
            substrate_id="a" * 64, peak_universe_hash="b" * 64, n_regions=True,
            caller_specification=_manifest(lambda_value=0.7).caller_specification,
            input_files=[], created_at="2026-07-25T00:00:00Z",
        )
