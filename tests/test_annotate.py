"""Annotation retains competing database evidence without adjudicating it."""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from motifmultiverse.schema import MISSING_SENTINEL, MotifNode


def _node(*, node_id: str = "node-a", motif_length: int = 10,
          trimmed_core=(0, 10), seqlet_count: int | None = 150):
    return MotifNode(
        node_id=node_id,
        model="model", readout="readout", context="context", metacluster="pos",
        denovo_pattern_id="pattern", variant_id="UA_UNASSIGNED_01",
        family_id=MISSING_SENTINEL, motif_length=motif_length,
        trimmed_core=None if trimmed_core is None else list(trimmed_core),
        seqlet_count=seqlet_count,
    )


class _StaticBackend:
    optional = True

    def __init__(self, name: str, version: str, candidates):
        self.name = name
        self.version = version
        self._candidates = candidates

    def annotate(self, nodes):
        return self._candidates


class _UnavailableBackend:
    name = "homer"
    version = "5.0"
    optional = True

    def annotate(self, nodes):
        raise RuntimeError("HOMER database is unavailable")


def _candidate(**changes):
    """A valid direct schema row that one test can corrupt at a time."""
    from motifmultiverse.schema.annotation import AnnotationCandidate

    candidate = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )
    return replace(candidate, **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"node_id": ""}, "identity fields"),
        ({"candidate_id": "annotation:corrupted"}, "stable annotation match identity"),
        ({"schema_version": "999"}, "candidate schema_version"),
        ({"motif_length": 0}, "motif_length"),
        ({"trimmed_core_length": -1}, "trimmed_core_length cannot be negative"),
        ({"trimmed_core_length": 11}, "trimmed_core_length exceeds its motif_length"),
        ({"seqlet_count": -1}, "seqlet_count"),
        ({"q_value": 1.01}, "q_value"),
        ({"chance_occurrence_probability": 1.01}, "chance_occurrence_probability"),
        ({"observed_to_null_ratio": -0.01}, "observed_to_null_ratio"),
    ],
)
def test_annotation_candidate_schema_refuses_each_corrupted_guarded_value(changes, message):
    """Removing the named AnnotationCandidate validation branch fails its row."""
    from motifmultiverse.schema import SchemaError

    with pytest.raises(SchemaError, match=message):
        _candidate(**changes)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"trimmed_core_length": 6}, "short motif"),
        ({"source": "tomtom", "q_value": 0.0501}, "weak TomTom match"),
        ({"seqlet_count": 99}, "low seqlet support"),
    ],
)
def test_annotation_candidate_refuses_false_low_confidence_for_each_trigger(changes, reason):
    """Each documented confidence trigger must reject a falsely-clear row."""
    from motifmultiverse.schema import SchemaError
    from motifmultiverse.schema.annotation import AnnotationCandidate

    payload = {
        "node_id": "node-a", "proposed_family_id": "FAM_ALPHA", "source": "tomtom",
        "source_version": "5.5", "matched_motif_id": "JASPAR:MA0001", "motif_length": 10,
        "trimmed_core_length": 10, "seqlet_count": 150,
    }
    payload.update(changes)
    candidate = AnnotationCandidate.create(**payload)
    with pytest.raises(SchemaError, match="must be flagged"):
        replace(candidate, low_confidence_annotation=False)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"backend": ""}, "backend name and version"),
        ({"backend": 42}, "backend name and version"),
        ({"backend_version": ""}, "backend name and version"),
        ({"backend_version": 42}, "backend name and version"),
        ({"status": "NOT_A_STATUS"}, "backend log status"),
        ({"candidate_count": -1}, "candidate_count"),
        ({"schema_version": "999"}, "backend log schema_version"),
    ],
)
def test_annotation_backend_log_refuses_each_corrupted_guarded_value(changes, message):
    """Removing the named AnnotationBackendLog validation branch fails its row."""
    from motifmultiverse.schema import SchemaError
    from motifmultiverse.schema.annotation import AnnotationBackendLog, BackendStatus

    payload = {
        "backend": "tomtom", "backend_version": "5.5", "status": BackendStatus.VERIFIED,
        "candidate_count": 1,
    }
    payload.update(changes)
    with pytest.raises(SchemaError, match=message):
        AnnotationBackendLog(**payload)


def test_annotation_keeps_conflicting_candidates_without_mutating_node_assignment():
    """Deleting candidate aggregation or assigning a family to the node fails this."""
    from motifmultiverse.annotate import annotate_nodes
    from motifmultiverse.schema.annotation import AnnotationCandidate

    node = _node()
    candidates = [
        AnnotationCandidate.create(
            node_id=node.node_id, proposed_family_id="FAM_ALPHA", source="tomtom",
            source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10, trimmed_core_length=10,
            seqlet_count=150,
        ),
        AnnotationCandidate.create(
            node_id=node.node_id, proposed_family_id="FAM_BETA", source="homer",
            source_version="4.11", matched_motif_id="HOMER:TF_BETA", motif_length=10, trimmed_core_length=10,
            seqlet_count=150,
        ),
    ]

    result = annotate_nodes(
        [node], [_StaticBackend("tomtom", "5.5", [candidates[0]]),
                 _StaticBackend("homer", "4.11", [candidates[1]])],
    )

    assert {row.proposed_family_id for row in result.candidates} == {"FAM_ALPHA", "FAM_BETA"}
    assert len(result.candidates) == 2
    assert node.family_id == MISSING_SENTINEL
    assert node.putative_tf_label == MISSING_SENTINEL
    assert node.annotation_matches == {}
    assert node.family_assignment_source == MISSING_SENTINEL
    assert node.family_assignment_confidence is None


def test_candidate_identity_is_stable_for_same_match_independent_of_backend_row_order():
    """Replacing the content identity with a row counter must fail this."""
    from motifmultiverse.annotate import annotate_nodes
    from motifmultiverse.schema.annotation import AnnotationCandidate

    left, right = _node(node_id="node-left"), _node(node_id="node-right")
    left_match = AnnotationCandidate.create(
        node_id="node-left", proposed_family_id="FAM_ALPHA", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )
    right_match = AnnotationCandidate.create(
        node_id="node-right", proposed_family_id="FAM_BETA", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0002", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )

    forward = annotate_nodes([left, right], [_StaticBackend("tomtom", "5.5", [left_match, right_match])])
    reverse = annotate_nodes([right, left], [_StaticBackend("tomtom", "5.5", [right_match, left_match])])

    by_match_forward = {row.matched_motif_id: row.candidate_id for row in forward.candidates}
    by_match_reverse = {row.matched_motif_id: row.candidate_id for row in reverse.candidates}
    assert by_match_forward == by_match_reverse
    assert by_match_forward["JASPAR:MA0001"] != by_match_forward["JASPAR:MA0002"]


def test_candidate_identity_uses_the_documented_match_tuple_not_a_proposed_label():
    """Adding a non-match label field to the stable-ID payload must fail this."""
    from motifmultiverse.schema.annotation import AnnotationCandidate

    first = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )
    relabelled = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA_REVISED", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )

    assert first.candidate_id == relabelled.candidate_id


@pytest.mark.parametrize(
    ("trimmed_core_length", "source", "q_value", "seqlet_count", "expected"),
    [
        (6, "homer", None, 150, True),
        (7, "tomtom", 0.0501, 150, True),
        (7, "tomtom", 0.05, 150, False),
        (7, "homer", None, 99, True),
        (0, "homer", None, 150, True),      # a core that trimmed to nothing is short
        (None, "homer", None, 150, False),  # no core declared is "not measured"
    ],
)
def test_low_confidence_annotation_uses_the_documented_boundaries(
    trimmed_core_length, source, q_value, seqlet_count, expected,
):
    """Changing <=6, TomTom q>0.05, or <100 seqlets must fail its row.

    The width under test is the trimmed core, and `motif_length` is held at the
    padded window width real tfmodisco-lite output carries (50 on all 139 nodes
    of the case-study registry) so that a rule reading the window instead of the
    core cannot satisfy any of these rows.
    """
    from motifmultiverse.schema.annotation import AnnotationCandidate

    candidate = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source=source,
        source_version="v1", matched_motif_id="database:match", motif_length=50,
        trimmed_core_length=trimmed_core_length,
        seqlet_count=seqlet_count, q_value=q_value,
    )

    assert candidate.low_confidence_annotation is expected
    assert candidate.proposed_family_id == "FAM_ALPHA"
    assert candidate.matched_motif_id == "database:match"


def test_a_short_core_in_a_padded_window_is_low_confidence():
    """The defect, at the schema: `motif_length` is the padded discovery window.

    It was 50 for every one of the 139 nodes of the thirteen-analysis case study,
    so `motif_length <= 6` could not fire on any real row -- while 40 of those
    nodes declare a contribution-bearing core of 6bp or less.
    """
    from motifmultiverse.schema.annotation import AnnotationCandidate

    candidate = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source="homer",
        source_version="4.11", matched_motif_id="HOMER:TF_ALPHA", motif_length=50,
        trimmed_core_length=4, seqlet_count=400,
    )

    assert candidate.low_confidence_annotation is True
    assert candidate.motif_length == 50
    assert candidate.trimmed_core_length == 4


def test_the_core_length_cannot_be_omitted_when_building_a_candidate():
    """A defaulted width would silently restore the reading that could never fire."""
    from motifmultiverse.schema.annotation import AnnotationCandidate

    with pytest.raises(TypeError, match="trimmed_core_length"):
        AnnotationCandidate.create(
            node_id="node-a", proposed_family_id="FAM_ALPHA", source="homer",
            source_version="4.11", matched_motif_id="HOMER:TF_ALPHA", motif_length=50,
            seqlet_count=400,
        )


def test_the_backend_takes_the_core_length_from_the_node_it_annotates(tmp_path):
    """End to end in the shape the case study runs: window 50, core 4.

    `ConfiguredAnnotationBackend` has the node in hand, so it reads the declared
    `trimmed_core` rather than handing the padded window to the confidence rule.
    """
    from motifmultiverse.annotate.base import ConfiguredAnnotationBackend

    node = _node(motif_length=50, trimmed_core=(25, 29), seqlet_count=400)
    config = tmp_path / "db.json"
    config.write_text(json.dumps({"tomtom": {
        "version": "5.5",
        "matches": [{"node_id": "node-a", "proposed_family_id": "FAM_ALPHA",
                     "matched_motif_id": "JASPAR:MA0001", "q_value": 0.001}],
    }}))

    row = ConfiguredAnnotationBackend("tomtom", config).annotate([node])[0]

    assert row.motif_length == 50
    assert row.trimmed_core_length == 4
    assert row.low_confidence_annotation is True


def test_a_node_that_declares_no_core_carries_none_rather_than_its_window(tmp_path):
    """None is "not measured". Substituting `motif_length` is what made this vacuous."""
    from motifmultiverse.annotate.base import ConfiguredAnnotationBackend

    node = _node(motif_length=50, trimmed_core=None, seqlet_count=400)
    config = tmp_path / "db.json"
    config.write_text(json.dumps({"tomtom": {
        "version": "5.5",
        "matches": [{"node_id": "node-a", "proposed_family_id": "FAM_ALPHA",
                     "matched_motif_id": "JASPAR:MA0001", "q_value": 0.001}],
    }}))

    row = ConfiguredAnnotationBackend("tomtom", config).annotate([node])[0]

    assert row.trimmed_core_length is None
    assert row.low_confidence_annotation is False


def test_the_core_length_survives_the_candidate_table_round_trip(tmp_path):
    """`adjudicate` re-reads this table and re-validates the flag against the rule.

    If the width did not travel in the artifact, the re-read would evaluate the
    short clause against nothing -- a second place the clause could not fire.
    """
    from motifmultiverse.adjudicate import _read_annotation_candidates
    from motifmultiverse.annotate import annotate_nodes, write_annotation_artifacts
    from motifmultiverse.schema.annotation import AnnotationCandidate

    node = _node(motif_length=50, trimmed_core=(25, 29), seqlet_count=400)
    candidate = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source="homer",
        source_version="4.11", matched_motif_id="HOMER:TF_ALPHA", motif_length=50,
        trimmed_core_length=4, seqlet_count=400,
    )
    result = annotate_nodes([node], [_StaticBackend("homer", "4.11", [candidate])])
    candidates_path, _ = write_annotation_artifacts(tmp_path, result)

    table = pd.read_parquet(candidates_path)
    reread = _read_annotation_candidates(candidates_path)

    assert table["trimmed_core_length"].tolist() == [4]
    assert [row.trimmed_core_length for row in reread] == [4]
    assert [row.low_confidence_annotation for row in reread] == [True]


def test_occurrence_null_fields_remain_none_without_input_and_preserve_supplied_values():
    """Inventing a null probability or ratio when no table was supplied fails this."""
    from motifmultiverse.annotate import annotate_nodes
    from motifmultiverse.schema.annotation import AnnotationCandidate

    candidate = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )
    backend = _StaticBackend("tomtom", "5.5", [candidate])
    missing = annotate_nodes([_node()], [backend]).candidates[0]
    supplied = annotate_nodes(
        [_node()], [backend],
        occurrence_nulls={candidate.candidate_id: {
            "chance_occurrence_probability": 0.125, "observed_to_null_ratio": 2.5,
        }},
    ).candidates[0]

    assert missing.chance_occurrence_probability is None
    assert missing.observed_to_null_ratio is None
    assert supplied.chance_occurrence_probability == pytest.approx(0.125)
    assert supplied.observed_to_null_ratio == pytest.approx(2.5)


def test_optional_backend_failure_is_unverified_and_does_not_remove_successful_candidate(tmp_path):
    """Letting an optional failure abort or clear prior rows fails this."""
    from motifmultiverse.annotate import annotate_nodes, write_annotation_artifacts
    from motifmultiverse.schema.annotation import AnnotationCandidate

    candidate = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )
    result = annotate_nodes(
        [_node()], [_StaticBackend("tomtom", "5.5", [candidate]), _UnavailableBackend()],
    )
    candidates_path, logs_path = write_annotation_artifacts(tmp_path, result)

    table = pd.read_parquet(candidates_path)
    logs = json.loads(logs_path.read_text())
    assert table["proposed_family_id"].tolist() == ["FAM_ALPHA"]
    assert {row["backend"]: row["status"] for row in logs["backends"]} == {
        "tomtom": "VERIFIED", "homer": "UNVERIFIED",
    }
    assert "HOMER database is unavailable" in logs["backends"][1]["detail"]
    assert (tmp_path / "provenance.json").exists() is False


def test_failed_backend_does_not_retain_a_partial_prefix_of_its_candidates():
    """Merging a backend row before its later row is rejected must fail this."""
    from motifmultiverse.annotate import annotate_nodes
    from motifmultiverse.schema.annotation import AnnotationCandidate

    valid = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )
    foreign = AnnotationCandidate.create(
        node_id="node-not-in-run", proposed_family_id="FAM_BETA", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0002", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )

    result = annotate_nodes([_node()], [_StaticBackend("tomtom", "5.5", [valid, foreign])])

    assert result.candidates == ()
    assert result.backend_logs[0].status.value == "UNVERIFIED"
    assert result.backend_logs[0].candidate_count == 0


@pytest.mark.parametrize(
    ("candidate_source", "candidate_version"),
    [("tomtom", "5.5"), ("homer", "4.12")],
)
def test_backend_source_or_version_mismatch_is_unverified_without_erasing_success(
    candidate_source, candidate_version,
):
    """Trusting a candidate's provenance over the reporting backend must fail this."""
    from motifmultiverse.annotate import annotate_nodes
    from motifmultiverse.schema.annotation import AnnotationCandidate

    successful = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )
    mismatched = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_BETA", source=candidate_source,
        source_version=candidate_version, matched_motif_id="DATABASE:MISMATCH", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )

    result = annotate_nodes(
        [_node()], [
            _StaticBackend("tomtom", "5.5", [successful]),
            _StaticBackend("homer", "4.11", [mismatched]),
        ],
    )

    assert [row.proposed_family_id for row in result.candidates] == ["FAM_ALPHA"]
    assert [(log.backend, log.backend_version, log.status.value, log.candidate_count)
            for log in result.backend_logs] == [
        ("tomtom", "5.5", "VERIFIED", 1),
        ("homer", "4.11", "UNVERIFIED", 0),
    ]
    assert (f"{candidate_source}/{candidate_version}" in result.backend_logs[1].detail
            and "homer/4.11" in result.backend_logs[1].detail)


def test_unreadable_optional_database_is_logged_without_erasing_another_backend(tmp_path):
    """Checksumming an absent optional config before adapters run must fail this."""
    from motifmultiverse.annotate import annotate_registry
    from motifmultiverse.annotate.tomtom import TomTomBackend
    from motifmultiverse.schema.annotation import AnnotationCandidate

    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "registry.json").write_text(json.dumps({
        "registry_metadata": {}, "nodes": [_node().to_dict()],
    }))
    candidate = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source="homer",
        source_version="4.11", matched_motif_id="HOMER:TF_ALPHA", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )
    absent_config = tmp_path / "not-installed.yaml"
    out = tmp_path / "annotation"

    result = annotate_registry(
        registry, out,
        backends=[_StaticBackend("homer", "4.11", [candidate]), TomTomBackend(absent_config)],
        provenance_inputs=[absent_config],
    )

    assert [row.proposed_family_id for row in result.candidates] == ["FAM_ALPHA"]
    assert [entry.status.value for entry in result.backend_logs] == ["VERIFIED", "UNVERIFIED"]
    assert (out / "annotation_candidates.parquet").exists()
    assert (out / "annotation_backend_logs.json").exists()
    assert (out / "provenance.json").exists()


def test_cli_annotate_writes_candidate_table_logs_and_provenance(tmp_path, capsys):
    """Routing annotate back through `_not_implemented` fails this end-to-end case."""
    from motifmultiverse.cli import main

    registry = tmp_path / "registry"
    registry.mkdir()
    node = _node().to_dict()
    (registry / "registry.json").write_text(json.dumps({
        "registry_metadata": {
            "project": "p", "peak_universe_id": "u", "analyses": [], "n_models": 1,
            "cross_model_claims_restricted": True, "metacluster_states": {}, "trim_threshold": 0.3,
        },
        "nodes": [node],
    }))
    databases = tmp_path / "databases.json"
    databases.write_text(json.dumps({
        "tomtom": {
            "version": "5.5", "matches": [{
                "node_id": "node-a", "proposed_family_id": "FAM_ALPHA",
                "matched_motif_id": "JASPAR:MA0001", "q_value": 0.01,
            }],
        },
        "homer": {"version": "4.11", "error": "HOMER data not installed"},
    }))
    out = tmp_path / "annotation"

    assert main([
        "annotate", str(tmp_path / "evidence"), "--registry", str(registry),
        "--tomtom", "--homer", "--databases", str(databases), "--out", str(out),
    ]) == 0

    table = pd.read_parquet(out / "annotation_candidates.parquet")
    logs = json.loads((out / "annotation_backend_logs.json").read_text())
    provenance = json.loads((out / "provenance.json").read_text())
    assert table[["node_id", "proposed_family_id"]].to_dict("records") == [
        {"node_id": "node-a", "proposed_family_id": "FAM_ALPHA"},
    ]
    assert {entry["backend"]: entry["status"] for entry in logs["backends"]} == {
        "tomtom": "VERIFIED", "homer": "UNVERIFIED",
    }
    assert provenance[0]["subcommand"] == "annotate"
    assert "annotation_candidates.parquet" in capsys.readouterr().out


def test_tomtom_adapter_reads_the_yaml_database_path_advertised_by_the_cli(tmp_path):
    """Treating the default ``config/db.yaml`` as JSON must fail this adapter test."""
    from motifmultiverse.annotate.tomtom import TomTomBackend

    config = tmp_path / "db.yaml"
    config.write_text(
        "tomtom:\n"
        "  version: '5.5'\n"
        "  matches:\n"
        "    - node_id: node-a\n"
        "      proposed_family_id: FAM_ALPHA\n"
        "      matched_motif_id: JASPAR:MA0001\n"
    )

    rows = TomTomBackend(config).annotate([_node()])

    assert [(row.source, row.source_version, row.proposed_family_id) for row in rows] == [
        ("tomtom", "5.5", "FAM_ALPHA"),
    ]


# --- regression: the shipped example must be readable by the adapter it feeds --
def test_shipped_db_example_is_a_shape_the_adapter_can_read(tmp_path):
    """`config/db.example.yaml` described binaries and thresholds -- a shape no
    adapter here reads. Feeding it to --tomtom raised "tomtom database
    configuration has no version": an example that could not work.
    """
    from pathlib import Path

    import motifmultiverse
    from motifmultiverse.annotate.tomtom import TomTomBackend

    root = Path(motifmultiverse.__file__).resolve().parents[2]
    example = root / "config" / "db.example.yaml"
    if not example.exists():
        pytest.skip("config/ not present in this installation")

    backend = TomTomBackend(str(example))
    assert backend.version, "the example must carry a database version"
    # No registry nodes -> the adapter must refuse the example's own matches by
    # name rather than silently returning nothing, which proves it parsed them.
    from motifmultiverse.annotate.base import AnnotationBackendError
    with pytest.raises(AnnotationBackendError, match="absent from the registry"):
        backend.annotate([])


def test_shipped_db_example_declares_a_version_for_every_backend_section():
    from pathlib import Path

    import yaml

    import motifmultiverse

    root = Path(motifmultiverse.__file__).resolve().parents[2]
    example = root / "config" / "db.example.yaml"
    if not example.exists():
        pytest.skip("config/ not present in this installation")
    payload = yaml.safe_load(example.read_text())
    for name in ("tomtom", "homer"):
        section = payload.get(name)
        assert isinstance(section, dict), f"{name} section missing"
        assert section.get("version"), f"{name} section has no version; the adapter refuses it"
        assert isinstance(section.get("matches"), list), f"{name} matches must be a list"


def test_a_dropped_backend_names_its_reason_but_the_adjudicator_cannot_see_it():
    """KNOWN LIMITATION, pinned: annotation completeness does not cross the stage seam.

    The test above states the retention rule: a backend that returns one row for
    a node outside the run is not trusted for any of its rows. That is
    deliberate -- output that disagrees with the registry cannot be verified row
    by row -- and the drop is not silent, because the backend log carries
    UNVERIFIED and names the node that caused it.

    What is nowhere recorded is the consequence, which this test pins. The
    dropped rows were the family evidence, and ``adjudicate`` reads
    ``annotation_candidates.parquet`` and nothing else. So the component below
    loses the family conflict that refused its merge, and adjudicates as a
    duplicate instead, on annotation evidence it has no way to know is partial.
    This USED to be a DEFERRED rather than a collapse, only because TRUE_DUPLICATE
    was CRITERION_NOT_YET_DEFINED. TRUE_DUPLICATE v2 is a FROZEN_DECLARED_HEURISTIC
    that decides, so the limitation is no longer hypothetical, and the assertions
    below say exactly how far it gets. On this fixture the shipped criterion
    REFUSES -- but only by an accident of the fixture's numbers: its
    ``empirical_p_value`` is 0.001, a hair above its own null floor of
    1/(1000+1) = 0.000999. Move it to the floor, which real duplicate pairs sit
    at, and the shipped criterion collapses a pair whose own annotation evidence
    said FAM_ALPHA vs FAM_BETA. That is pinned below too, because that is the
    live severity, and nothing in v2 pays it down.

    It is pinned rather than patched because both obvious repairs invent a rule
    the design has not decided. Keeping the good rows contradicts the retention
    rule above. Deferring every component whenever a backend is UNVERIFIED would
    defer whole runs at any site without HOMER, since a backend that failed
    cannot say which nodes it would have covered -- which is exactly why
    "annotation evidence is incomplete" has to become a declared adjudication
    input by design before it can become one in code. See annotate/README.md.
    """
    from motifmultiverse.adjudicate import adjudicate_component, packaged_criteria_path
    from motifmultiverse.align import AlignmentEvidence
    from motifmultiverse.annotate import annotate_nodes
    from motifmultiverse.schema import Decision
    from motifmultiverse.schema.annotation import AnnotationCandidate
    from motifmultiverse.schema.criteria import (
        Criterion,
        CriterionStatus,
        Predicate,
        load_criteria,
    )

    def _match(node_id, family_id, matched):
        return AnnotationCandidate.create(
            node_id=node_id, proposed_family_id=family_id, source="tomtom",
            source_version="5.5", matched_motif_id=matched, motif_length=10, trimmed_core_length=10,
            seqlet_count=150,
        )

    nodes = [_node(node_id="node-a"), _node(node_id="node-b")]
    nodes = [replace(nodes[0], variant_id="UA_ALPHA_01"),
             replace(nodes[1], variant_id="UA_BETA_02")]
    conflicting = [_match("node-a", "FAM_ALPHA", "JASPAR:MA0001"),
                   _match("node-b", "FAM_BETA", "JASPAR:MA0002")]
    stray = _match("node-ghost", "FAM_GAMMA", "JASPAR:MA0003")

    complete = annotate_nodes(nodes, [_StaticBackend("tomtom", "5.5", conflicting)])
    dropped = annotate_nodes(nodes, [_StaticBackend("tomtom", "5.5", [*conflicting, stray])])

    assert len(complete.candidates) == 2
    assert dropped.candidates == ()
    # Not silent: the reason survives, and it names the row that caused it.
    log = dropped.backend_logs[0]
    assert log.status.value == "UNVERIFIED"
    assert "node-ghost" in log.detail

    edge = AlignmentEvidence(
        source_node_id="node-a", target_node_id="node-b", orientation="+", offset=0,
        overlap_bp=10, overlap_frac_source=1.0, overlap_frac_target=1.0,
        ppm_similarity=0.99, signed_cwm_similarity=0.99, empirical_p_value=0.001,
        null_shuffles=1000, seed=7,
    )
    metadata = {
        node_id: {"variant_id": variant_id, "motif_completeness": 1.0,
                  "seqlet_count": 150, "core_ic": 10.0, "cross_context_recurrence": 1}
        for node_id, variant_id in (("node-a", "UA_ALPHA_01"), ("node-b", "UA_BETA_02"),
                                    ("node-ghost", "UA_GAMMA_03"))
    }

    def _adjudicate(candidates, criteria):
        return adjudicate_component(["node-a", "node-b"], [edge], candidates, [],
                                    criteria, "test", node_metadata=metadata)

    shipped = load_criteria(packaged_criteria_path())
    assert _adjudicate(complete.candidates, shipped).relationship == "AMBIGUOUS_CROSS_FAMILY"
    assert _adjudicate(complete.candidates, shipped).decision == Decision.REFUSE_MERGE
    assert _adjudicate(dropped.candidates, shipped).relationship == "TRUE_DUPLICATE"
    # Complete evidence, one unmet predicate (p = 0.001 > the 1/1001 null floor)
    # -> the fail-safe REFUSE_MERGE, which v1 could not express and v2 can.
    assert _adjudicate(dropped.candidates, shipped).decision == Decision.REFUSE_MERGE

    # ...and the live severity, one number away. At the null floor the shipped
    # heuristic COLLAPSES the pair, on annotation evidence it cannot know is
    # partial. This is the cost of freezing the criterion, stated as a passing
    # assertion rather than as prose.
    at_floor = replace(edge, empirical_p_value=1.0 / (edge.null_shuffles + 1))
    assert adjudicate_component(
        ["node-a", "node-b"], [at_floor], dropped.candidates, [], shipped, "test",
        node_metadata=metadata,
    ).decision == Decision.COLLAPSE

    frozen_duplicate = dict(shipped)
    frozen_duplicate["TRUE_DUPLICATE"] = Criterion(
        criterion_id="TRUE_DUPLICATE", version="project-1",
        status=CriterionStatus.FROZEN_DECLARED_HEURISTIC,
        relationship="TRUE_DUPLICATE", required_evidence=("ppm_similarity",),
        predicates=(Predicate(field="ppm_similarity", operator="ge", value=0.9,
                              provenance="declared", basis="test fixture"),),
        insufficient_evidence_action=Decision.DEFERRED,
        decision_if_matched=Decision.COLLAPSE,
        declared_rationale="a site-local threshold, invented for this test",
        replacement_evidence=("nothing; this criterion exists only in this test",),
    )
    assert _adjudicate(complete.candidates, frozen_duplicate).decision == Decision.REFUSE_MERGE
    assert _adjudicate(dropped.candidates, frozen_duplicate).decision == Decision.COLLAPSE


def test_annotate_records_what_its_guard_returned_beside_the_candidates(tmp_path):
    """The stage's own threshold check leaves a record, not only an exception path."""
    import json

    from motifmultiverse import guard_log
    from motifmultiverse.annotate import annotate_registry
    from motifmultiverse.schema.annotation import AnnotationCandidate

    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "registry.json").write_text(json.dumps({
        "registry_metadata": {}, "nodes": [_node().to_dict()],
    }))
    candidate = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source="homer",
        source_version="4.11", matched_motif_id="HOMER:TF_ALPHA", motif_length=10, trimmed_core_length=10,
        seqlet_count=150,
    )
    out = tmp_path / "annotation"
    annotate_registry(registry, out,
                      backends=[_StaticBackend("homer", "4.11", [candidate])])

    recorded = json.loads((out / guard_log.GUARD_OUTCOMES_FILENAME).read_text())
    assert [row["guard_id"] for row in recorded] == ["short_motif_flag"]
    assert recorded[0]["stage"] == "annotate" and recorded[0]["passed"] is True
    assert "homer/4.11" in recorded[0]["subject"]
