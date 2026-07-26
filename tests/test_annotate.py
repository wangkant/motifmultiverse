"""Annotation retains competing database evidence without adjudicating it."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from motifmultiverse.schema import MISSING_SENTINEL, MotifNode


def _node(*, node_id: str = "node-a", motif_length: int = 10, seqlet_count: int | None = 150):
    return MotifNode(
        node_id=node_id,
        model="model", readout="readout", context="context", metacluster="pos",
        denovo_pattern_id="pattern", variant_id="UA_UNASSIGNED_01",
        family_id=MISSING_SENTINEL, motif_length=motif_length, seqlet_count=seqlet_count,
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


def test_annotation_keeps_conflicting_candidates_without_mutating_node_assignment():
    """Deleting candidate aggregation or assigning a family to the node fails this."""
    from motifmultiverse.annotate import annotate_nodes
    from motifmultiverse.schema.annotation import AnnotationCandidate

    node = _node()
    candidates = [
        AnnotationCandidate.create(
            node_id=node.node_id, proposed_family_id="FAM_ALPHA", source="tomtom",
            source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10,
            seqlet_count=150,
        ),
        AnnotationCandidate.create(
            node_id=node.node_id, proposed_family_id="FAM_BETA", source="homer",
            source_version="4.11", matched_motif_id="HOMER:TF_BETA", motif_length=10,
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
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10,
        seqlet_count=150,
    )
    right_match = AnnotationCandidate.create(
        node_id="node-right", proposed_family_id="FAM_BETA", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0002", motif_length=10,
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
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10,
        seqlet_count=150,
    )
    relabelled = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA_REVISED", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10,
        seqlet_count=150,
    )

    assert first.candidate_id == relabelled.candidate_id


@pytest.mark.parametrize(
    ("motif_length", "source", "q_value", "seqlet_count", "expected"),
    [
        (6, "homer", None, 150, True),
        (7, "tomtom", 0.0501, 150, True),
        (7, "tomtom", 0.05, 150, False),
        (7, "homer", None, 99, True),
    ],
)
def test_low_confidence_annotation_uses_the_documented_boundaries(
    motif_length, source, q_value, seqlet_count, expected,
):
    """Changing <=6, TomTom q>0.05, or <100 seqlets must fail its row."""
    from motifmultiverse.schema.annotation import AnnotationCandidate

    candidate = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source=source,
        source_version="v1", matched_motif_id="database:match", motif_length=motif_length,
        seqlet_count=seqlet_count, q_value=q_value,
    )

    assert candidate.low_confidence_annotation is expected
    assert candidate.proposed_family_id == "FAM_ALPHA"
    assert candidate.matched_motif_id == "database:match"


def test_occurrence_null_fields_remain_none_without_input_and_preserve_supplied_values():
    """Inventing a null probability or ratio when no table was supplied fails this."""
    from motifmultiverse.annotate import annotate_nodes
    from motifmultiverse.schema.annotation import AnnotationCandidate

    candidate = AnnotationCandidate.create(
        node_id="node-a", proposed_family_id="FAM_ALPHA", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10,
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
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10,
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
        source_version="5.5", matched_motif_id="JASPAR:MA0001", motif_length=10,
        seqlet_count=150,
    )
    foreign = AnnotationCandidate.create(
        node_id="node-not-in-run", proposed_family_id="FAM_BETA", source="tomtom",
        source_version="5.5", matched_motif_id="JASPAR:MA0002", motif_length=10,
        seqlet_count=150,
    )

    result = annotate_nodes([_node()], [_StaticBackend("tomtom", "5.5", [valid, foreign])])

    assert result.candidates == ()
    assert result.backend_logs[0].status.value == "UNVERIFIED"
    assert result.backend_logs[0].candidate_count == 0


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
        source_version="4.11", matched_motif_id="HOMER:TF_ALPHA", motif_length=10,
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
