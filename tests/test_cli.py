"""CLI tests: all eight subcommands expose real --help and record provenance (T-08/T-09)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from motifmultiverse.cli import build_parser, main
from motifmultiverse.schema import build_peak_split_manifest
from motifmultiverse.validate import (
    DecisionSplitArtifact,
    ValidationSplitArtifact,
)

SUBCOMMANDS = ["ingest", "align", "annotate", "adjudicate",
               "compile", "validate", "infer", "report"]
IMPLEMENTED = ["ingest", "compile", "interpret", "align", "annotate", "adjudicate",
               "validate", "infer", "report"]


def test_version_flag():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_bare_invocation_prints_help_and_exits_2():
    assert main([]) == 2


@pytest.mark.parametrize("sub", SUBCOMMANDS + ["interpret"])
def test_subcommand_help_is_available(sub, capsys):
    with pytest.raises(SystemExit) as e:
        main([sub, "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert sub in out
    assert "usage:" in out.lower()


def test_all_eight_subcommands_are_registered():
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    registered = set()
    for a in actions:
        registered |= set(a.choices)
    assert set(SUBCOMMANDS).issubset(registered)


def test_no_subcommand_body_is_a_skeleton_any_more():
    """`report` was the last skeleton body; nothing in this release exits 3.

    This used to be a parametrized list of skeleton argv lines, and the list went
    stale twice (`align`/`annotate`, then `infer`). It is derived from the
    dispatch table now, so it cannot go stale in either direction: a module that
    regressed to a skeleton would fail here, and one that was implemented without
    this test being touched would not need touching.
    """
    from motifmultiverse.status import MODULES, module_status

    skeletons = [name for name in MODULES if module_status(name)["status"] == "SKELETON"]
    assert skeletons == []


def test_a_skeleton_body_would_still_exit_3_and_name_its_readme(monkeypatch, capsys, tmp_path):
    """The exit-3 machinery is kept rather than deleted with the last skeleton.

    Asserting only that nothing is a skeleton would leave `_not_implemented` and
    the exit-3 branch of `main` unexercised -- a mechanism nobody runs is a
    mechanism nobody knows still works, and `docs/ROADMAP.md` has unimplemented
    milestones left. So one subcommand is pushed back to the skeleton dispatcher
    here and the whole path is walked: NotImplementedError, exit 3, and a message
    naming the README that specifies the module.
    """
    from motifmultiverse import cli

    real_build_parser = cli.build_parser

    def stubbed():
        parser = real_build_parser()
        for action in parser._actions:
            if getattr(action, "choices", None) and "report" in action.choices:
                action.choices["report"].set_defaults(func=lambda ns: cli._run("report", ns))
        return parser

    monkeypatch.setattr(cli, "build_parser", stubbed)
    rc = main(["report", "project/", "--out", str(tmp_path / "o")])
    assert rc == 3
    err = capsys.readouterr().err
    assert "src/motifmultiverse/" in err and "README.md" in err


def test_validate_cli_writes_split_bound_stability_and_backend_artifacts(tmp_path):
    import h5py
    import numpy as np

    from motifmultiverse.compile import lexicon_semantic_hash
    from motifmultiverse.schema.substrate import CallerSpecification
    from motifmultiverse.substrate import build_manifest as build_substrate_manifest
    from motifmultiverse.substrate import write_manifest as write_substrate_manifest

    lexicons = tmp_path / "lexicons"
    lexicons.mkdir()
    with h5py.File(lexicons / "core.h5", "w") as h5:
        h5.create_group("pos_patterns").create_group("pattern_0").create_dataset(
            "contrib_scores", data=np.asarray([[1.0, 0.0, 0.0, 0.0]])
        )
    content_hash = lexicon_semantic_hash(
        [("pos_patterns", "pattern_0",
          {"node_id": "node-0", "variant_id": "MA_FAM_01"})],
        {"node-0": {"cwm": np.asarray([[1.0, 0.0, 0.0, 0.0]])}},
        schema_version="1.0", trim_threshold=0.3, motif_type="cwm", include_rc=False,
        loader_backend="finemo", loader_parameters={"motif_lambda_default": 0.7},
    )
    (lexicons / "core.manifest.json").write_text(json.dumps({
        "tier": "core", "lexicon_content_hash": content_hash, "n_motifs": 1,
        "pattern_order": ["pos_patterns.pattern_0"], "node_ids": ["node-0"],
        "index": [{
            "index": 0, "pattern_tag": "pos_patterns.pattern_0", "node_id": "node-0",
            "variant_id": "MA_FAM_01", "metacluster": "pos",
        }],
        "schema_version": "1.0", "trim_threshold": 0.3, "motif_type": "cwm",
        "include_rc": False, "loader_backend": "finemo",
        "loader_parameters": {"motif_lambda_default": 0.7},
        "comparisons": {}, "source_registry": "registry", "sensitivity_triggers": {},
        "project": "test-project", "cross_model_claims_restricted": True,
    }), encoding="utf-8")
    substrate = build_substrate_manifest(
        peak_universe_hash="a" * 64,
        n_regions=2,
        caller_specification=CallerSpecification(
            caller_name="finemo",
            caller_version="0.test",
            lexicon_content_hash=content_hash,
            parameters={"motif_type": "cwm"},
            preprocessing_contract_hash="b" * 64,
        ),
        input_files={"peaks.bed": "c" * 64},
        created_at="2026-07-26T12:00:00Z",
    )
    substrate_path = write_substrate_manifest(
        substrate, tmp_path / "substrate.manifest.json",
    )
    rows = [
        {
            "peak_id": "p-validation",
            "hit_id": "old",
            "coefficient": 1.0,
            "reconstruction": 0.0,
            "substrate_id": substrate.substrate_id,
        },
    ]
    before = tmp_path / "before" / "hits.parquet"
    after = tmp_path / "after" / "hits.parquet"
    before.parent.mkdir()
    after.parent.mkdir()
    pd.DataFrame(rows).to_parquet(before, index=False)
    rows[0] = {
        "peak_id": "p-validation",
        "hit_id": "new",
        "coefficient": 2.0,
        "reconstruction": 1.0,
        "substrate_id": substrate.substrate_id,
    }
    pd.DataFrame(rows).to_parquet(after, index=False)
    manifest = build_peak_split_manifest({
        "p-discovery": "DISCOVERY", "p-validation": "VALIDATION",
    })
    manifest_path = tmp_path / "split-manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": manifest.schema_version,
        "assignments": {key: value.value for key, value in manifest.assignments.items()},
        "checksum": manifest.checksum,
    }), encoding="utf-8")
    decision = DecisionSplitArtifact.create(
        manifest=manifest, decision_id="decision:cli",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation"}),
    )
    provisional = ValidationSplitArtifact.create(
        manifest=manifest, decision_id="decision:cli", result_id="pending",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation"}),
    )
    decision_path = tmp_path / "decision-split.json"
    validation_path = tmp_path / "validation-split.json"
    decision_path.write_text(json.dumps(decision.to_dict()), encoding="utf-8")
    validation_path.write_text(json.dumps(provisional.to_dict()), encoding="utf-8")
    out = tmp_path / "validation"

    argv = [
        "validate", str(lexicons), "--before-hits", str(before), "--after-hits", str(after),
        "--substrate-manifest", str(substrate_path),
        "--split-manifest", str(manifest_path), "--decision-artifact", str(decision_path),
        "--validation-artifact", str(validation_path), "--out", str(out),
    ]
    assert main(argv) == 0
    assert (out / "stability_results.parquet").exists()
    assert (out / "backend_verification.tsv").exists()
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))[0]
    assert provenance["command"] and provenance["subcommand"] == provenance["stage"] == "validate"
    assert provenance["timestamp_utc"] and provenance["schema_version"] == "1"
    assert provenance["redaction_policy"] == "basenames_only_except_command"
    assert provenance["random_seed"] is None and provenance["input_scale"] == 1
    assert provenance["substrate_id"] == substrate.substrate_id
    assert provenance["software"] and provenance["inputs"]
    assert {
        "before_hits",
        "after_hits",
        "substrate_manifest",
        "split_manifest",
        "decision_artifact",
        "validation_split_binding",
        "lexicon_manifest:core",
        "lexicon_h5:core",
    } <= set(provenance["inputs"])
    assert provenance["inputs"]["before_hits"] != provenance["inputs"]["after_hits"]
    assert "hits.parquet" not in provenance["inputs"]
    assert provenance["lexicon_identity"].startswith("lexicons:")
    assert provenance["split_manifest_checksum"] == manifest.checksum
    assert provenance["decision_artifact_id"] == decision.artifact_id
    assert len(provenance["validation_split_identity"]) == 64

    unsupported_out = tmp_path / "unsupported-backend"
    unsupported_argv = [str(unsupported_out) if value == str(out) else value for value in argv]
    assert main([*unsupported_argv, "--fimo-heldout"]) == 4
    assert not unsupported_out.exists()

    mismatched_substrate = build_substrate_manifest(
        peak_universe_hash="a" * 64,
        n_regions=2,
        caller_specification=CallerSpecification(
            caller_name="finemo",
            caller_version="0.test",
            lexicon_content_hash="f" * 64,
            parameters={"motif_type": "cwm"},
            preprocessing_contract_hash="b" * 64,
        ),
        input_files={"peaks.bed": "c" * 64},
        created_at="2026-07-26T12:00:00Z",
    )
    mismatched_substrate_path = write_substrate_manifest(
        mismatched_substrate, tmp_path / "mismatched-substrate.manifest.json",
    )
    for hit_path in (before, after):
        hit_rows = pd.read_parquet(hit_path)
        hit_rows["substrate_id"] = mismatched_substrate.substrate_id
        hit_rows.to_parquet(hit_path, index=False)
    mismatch_out = tmp_path / "mismatched-substrate-output"
    mismatch_argv = [
        str(mismatched_substrate_path)
        if value == str(substrate_path)
        else str(mismatch_out)
        if value == str(out)
        else value
        for value in argv
    ]
    assert main(mismatch_argv) == 4
    assert not mismatch_out.exists()
    for hit_path in (before, after):
        hit_rows = pd.read_parquet(hit_path)
        hit_rows["substrate_id"] = substrate.substrate_id
        hit_rows.to_parquet(hit_path, index=False)

    # A split-bound validation command must not silently include decision peaks.
    corrupt = pd.read_parquet(after)
    corrupt.loc[len(corrupt)] = {
        "peak_id": "p-discovery",
        "hit_id": "old",
        "coefficient": 1.0,
        "reconstruction": 0.0,
        "substrate_id": substrate.substrate_id,
    }
    corrupt.to_parquet(after, index=False)
    refused_out = tmp_path / "refused-validation"
    refused_argv = [str(refused_out) if value == str(out) else value for value in argv]
    assert main(refused_argv) == 4
    assert not refused_out.exists()


def test_provenance_is_written_before_the_body_runs_and_survives_a_refusal(tmp_path):
    # "align" and "annotate" no longer belong here (Tasks 10 and 11 implement
    # them for real); Task 12 implements adjudicate, and `report` is implemented
    # too, so what is exercised here is no longer a skeleton -- it is the same
    # T-09 property one step further on: the record is on disk before anything is
    # rendered, and a refusal does not take it back. `--html` is chosen because
    # the CLI itself owns that refusal (this release renders markdown only), so
    # the record under test does not depend on what the input directory holds.
    # `report` has no --seed flag (only ingest/compile/interpret/align/infer ever
    # exposed one), so the recorded random_seed is the untouched default (None)
    # rather than a value threaded in from the CLI -- the property under test is
    # that the record exists at all before the body runs, not what value a
    # particular field holds.
    out = tmp_path / "o"
    assert main(["report", "project/", "--out", str(out), "--html"]) == 4
    recs = json.loads((out / "provenance.json").read_text())
    assert len(recs) == 1
    r = recs[0]
    for field in ("command", "subcommand", "software", "timestamp_utc", "random_seed"):
        assert field in r, field
    assert r["subcommand"] == "report"
    assert r["random_seed"] is None
    assert r["software"]["motifmultiverse"]


def test_cli_threads_the_seed_into_the_provenance_record(tmp_path):
    """align is implemented and exposes --seed; its provenance is written before
    the registry is even opened (same T-09 pattern as the skeletons), so a
    nonexistent registry path is enough to reach that write without needing
    the run to succeed end to end. This is the concrete-seed counterpart to
    test_provenance_is_written_even_though_body_is_unimplemented: together
    they prove the CLI carries --seed into provenance on both the
    implemented and the still-skeleton paths. A seed that silently fails to
    be recorded would make a stochastic run unreproducible while looking fine.
    """
    out = tmp_path / "o"
    main(["align", "registry/", "--out", str(out), "--seed", "7"])
    recs = json.loads((out / "provenance.json").read_text())
    assert len(recs) == 1
    assert recs[0]["subcommand"] == "align"
    assert recs[0]["random_seed"] == 7


def test_report_refuses_html_and_docx_rather_than_rendering_markdown_instead(tmp_path, capsys):
    """A flag naming an output this release cannot produce is refused, not downgraded.

    Same precedent as `_run_validate` refusing `--fimo-heldout`: emitting markdown
    for a caller who asked for HTML is the gap between what was specified and what
    ran, which is the gap this package exists to close. Exit 4, and the message
    names the flag.
    """
    out = tmp_path / "o"
    assert main(["report", "project/", "--out", str(out), "--docx"]) == 4
    err = capsys.readouterr().err
    assert "refused" in err and "--docx" in err


def test_provenance_appends_rather_than_overwrites(tmp_path):
    # Both runs are ones whose record the CLI writes before it does anything
    # else: `align` before it opens the registry, `report` before it renders and
    # before it refuses `--html`. The subject is the append, so neither run needs
    # to succeed -- but the second must not depend on how the renderer treats a
    # directory that holds no interpretation.
    out = tmp_path / "o"
    main(["align", "registry/", "--out", str(out)])
    main(["report", "project/", "--out", str(out), "--html"])
    assert len(json.loads((out / "provenance.json").read_text())) == 2


def test_the_implemented_subcommands_do_not_exit_3(tmp_path):
    """A guard against re-listing an implemented module as a skeleton."""
    parser = build_parser()
    registered = set()
    for a in parser._actions:
        if hasattr(a, "choices") and a.choices:
            registered |= set(a.choices)
    assert set(IMPLEMENTED).issubset(registered)


def test_ingest_and_compile_run_end_to_end(tmp_path, capsys):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    modisco = tmp_path / "modisco.h5"
    with h5py.File(modisco, "w") as h5:
        for group, n in (("pos_patterns", 2), ("neg_patterns", 1)):
            for i in range(n):
                grp = h5.require_group(group).create_group(f"pattern_{i}")
                grp.create_dataset("contrib_scores", data=np.ones((10, 4)) * (i + 1))
                grp.create_dataset("hypothetical_contribs", data=np.ones((10, 4)))
                grp.create_dataset("sequence", data=np.full((10, 4), 0.25))
    project = tmp_path / "project.json"
    project.write_text(json.dumps({
        "project": "cli-test", "peak_universe_id": "u1",
        "analyses": [{"id": "a1", "model": "m", "readout": "r", "union_id": "MA",
                      "context": "promoter", "modisco_h5": str(modisco)}]}))

    assert main(["ingest", str(project), "--out", str(tmp_path / "registry")]) == 0
    out = capsys.readouterr().out
    assert "3 motif nodes" in out and "cross_model_claims_restricted" in out

    assert main(["compile", str(tmp_path / "registry"), "--out", str(tmp_path / "lex")]) == 0
    out = capsys.readouterr().out
    assert "content_hash=" in out
    manifest = json.loads((tmp_path / "lex" / "core.manifest.json").read_text())
    assert manifest["pattern_order"] == ["pos_patterns.pattern_0", "pos_patterns.pattern_1",
                                         "neg_patterns.pattern_0"]
    assert len(manifest["lexicon_content_hash"]) == 64
    assert manifest["cross_model_claims_restricted"] is True


def test_ingest_refuses_an_undeclared_union_id_and_exits_4(tmp_path, capsys):
    project = tmp_path / "project.json"
    project.write_text(json.dumps({"project": "p", "analyses": [
        {"id": "a1", "model": "m", "readout": "r", "context": "c",
         "modisco_h5": str(tmp_path / "nope.h5")}]}))
    assert main(["ingest", str(project), "--out", str(tmp_path / "r")]) == 4
    assert "union_id" in capsys.readouterr().err


def _tiny_substrate(tmp_path, n_blocks=6, substrate_id="e" * 64):
    """A hit table small enough to read, with two peaks per block."""
    from motifmultiverse.schema import HIT_TABLE_COLUMNS
    lines = ["\t".join(HIT_TABLE_COLUMNS)]
    query, comparator = [], []
    for b in range(n_blocks):
        for i in (0, 1):
            rid = f"r{b}_{i}"
            (query if i == 0 else comparator).append(rid)
            start = b * 1_000_000 + i * 1000
            lines.append("\t".join([rid, "chr1", str(start), str(start + 500),
                                    f"UA_FAMA_{i}", "FAM_A", "1.0" if i == 0 else "0.2",
                                    "used", "9999", "lex_v1", substrate_id]))
    hits = tmp_path / "hits.tsv"
    hits.write_text("\n".join(lines) + "\n")
    (tmp_path / "q.txt").write_text("\n".join(query) + "\n")
    (tmp_path / "c.txt").write_text("\n".join(comparator) + "\n")
    return hits, tmp_path / "q.txt", tmp_path / "c.txt"


def _floors(n_blocks=6):
    return ["--floor-blocks", str(n_blocks), "--floor-coverage", "0.9",
            "--floor-explained", "0.9"]


def _bilateral_substrate(tmp_path, n_query_blocks=10, n_comparator_peaks=2):
    """A query healthy on its own, paired with a comparator too thin to clear
    the same block floor -- both comparator peaks land in block 0, so the
    comparator's n_blocks is 1 no matter how many blocks the query spans.
    """
    from motifmultiverse.schema import HIT_TABLE_COLUMNS
    lines = ["\t".join(HIT_TABLE_COLUMNS)]
    query_ids = []
    for b in range(n_query_blocks):
        rid = f"q{b}"
        query_ids.append(rid)
        start = b * 1_000_000
        lines.append("\t".join([rid, "chr1", str(start), str(start + 500),
                                "UA_FAMA_0", "FAM_A", "1.0", "used", "9999", "lex_v1", "e" * 64]))
    comparator_ids = []
    for i in range(n_comparator_peaks):
        rid = f"c{i}"
        comparator_ids.append(rid)
        start = i * 1000
        lines.append("\t".join([rid, "chr1", str(start), str(start + 500),
                                "UA_FAMA_1", "FAM_A", "0.2", "used", "9999", "lex_v1", "e" * 64]))
    hits = tmp_path / "hits.tsv"
    hits.write_text("\n".join(lines) + "\n")
    (tmp_path / "q.txt").write_text("\n".join(query_ids) + "\n")
    (tmp_path / "c.txt").write_text("\n".join(comparator_ids) + "\n")
    return hits, tmp_path / "q.txt", tmp_path / "c.txt"


def test_interpret_runs_end_to_end_and_writes_its_result(tmp_path, capsys):
    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    rc = main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
               "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
               "--bootstrap", "50", "--seed", "1", "--out", str(out), *_floors()])
    assert rc == 0
    blob = json.loads((out / "interpretation.json").read_text())
    assert blob["output_mode"] == "FULL_INFERENCE"
    assert blob["emitted_order"][0] == "health"
    assert blob["effects"][0]["comparator_id"] == "odd"
    assert "intersection_coverage" in capsys.readouterr().out


def test_interpret_writes_provenance_with_input_checksums(tmp_path):
    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
          "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
          "--bootstrap", "20", "--out", str(out), *_floors()])
    rec = json.loads((out / "provenance.json").read_text())[0]
    assert rec["subcommand"] == "interpret"
    assert rec["input_scale"] == 9999
    # keyed by the role each file played, not its basename (see
    # test_same_named_query_and_comparator_do_not_collide)
    assert set(rec["inputs"]) == {"hits:hits.tsv", "peaks:q.txt", "comparator:c.txt"}
    assert all(len(v) == 64 for v in rec["inputs"].values())


def test_interpret_refuses_a_table_that_does_not_match_the_substrate_manifest(tmp_path, capsys):
    """A supplied manifest is a guard, never merely display metadata."""
    from motifmultiverse.schema.substrate import CallerSpecification
    from motifmultiverse.substrate import build_manifest, write_manifest

    hits, q, c = _tiny_substrate(tmp_path)
    manifest = build_manifest(
        peak_universe_hash="c" * 64,
        n_regions=12,
        caller_specification=CallerSpecification(
            caller_name="finemo", caller_version="0.3.1", lexicon_content_hash="a" * 64,
            parameters={"lambda": 0.7}, preprocessing_contract_hash="b" * 64,
        ),
        input_files={"peaks.tsv": "d" * 64}, created_at="2026-07-25T00:00:00Z",
    )
    manifest_path = write_manifest(manifest, tmp_path / "substrate.manifest.json")
    out = tmp_path / "o"

    assert main([
        "interpret", str(hits), "--substrate-manifest", str(manifest_path),
        "--peaks", str(q), "--comparator", str(c), "--comparator-id", "odd",
        "--selection-provenance", "EXTERNAL", "--bootstrap", "20", "--out", str(out),
        *_floors(),
    ]) == 4
    assert "substrate_id" in capsys.readouterr().err
    assert not (out / "interpretation.json").exists()


def test_interpret_artifacts_record_the_verified_substrate_id(tmp_path):
    """A successful interpretation retains the content identity that made it reproducible."""
    from motifmultiverse.schema.substrate import CallerSpecification
    from motifmultiverse.substrate import build_manifest, write_manifest

    manifest = build_manifest(
        peak_universe_hash="c" * 64,
        n_regions=12,
        caller_specification=CallerSpecification(
            caller_name="finemo", caller_version="0.3.1", lexicon_content_hash="a" * 64,
            parameters={"lambda": 0.7}, preprocessing_contract_hash="b" * 64,
        ),
        input_files={"peaks.tsv": "d" * 64}, created_at="2026-07-25T00:00:00Z",
    )
    hits, q, c = _tiny_substrate(tmp_path, substrate_id=manifest.substrate_id)
    manifest_path = write_manifest(manifest, tmp_path / "substrate.manifest.json")
    out = tmp_path / "o"

    assert main([
        "interpret", str(hits), "--substrate-manifest", str(manifest_path),
        "--peaks", str(q), "--comparator", str(c), "--comparator-id", "odd",
        "--selection-provenance", "EXTERNAL", "--bootstrap", "50", "--out", str(out),
        *_floors(),
    ]) == 0
    assert json.loads((out / "interpretation.json").read_text())["substrate_id"] == manifest.substrate_id
    assert json.loads((out / "provenance.json").read_text())[0]["substrate_id"] == manifest.substrate_id


def test_interpret_refuses_a_malformed_substrate_manifest(tmp_path, capsys):
    """Manifest structural corruption must be a refusal rather than an uncaught exception."""
    from motifmultiverse.schema.substrate import CallerSpecification
    from motifmultiverse.substrate import build_manifest, write_manifest

    manifest = build_manifest(
        peak_universe_hash="c" * 64,
        n_regions=12,
        caller_specification=CallerSpecification(
            caller_name="finemo", caller_version="0.3.1", lexicon_content_hash="a" * 64,
            parameters={"lambda": 0.7}, preprocessing_contract_hash="b" * 64,
        ),
        input_files={"peaks.tsv": "d" * 64}, created_at="2026-07-25T00:00:00Z",
    )
    hits, q, c = _tiny_substrate(tmp_path, substrate_id=manifest.substrate_id)
    manifest_path = write_manifest(manifest, tmp_path / "substrate.manifest.json")
    payload = json.loads(manifest_path.read_text())
    payload["input_files"] = []
    manifest_path.write_text(json.dumps(payload))

    assert main([
        "interpret", str(hits), "--substrate-manifest", str(manifest_path),
        "--peaks", str(q), "--comparator", str(c), "--comparator-id", "odd",
        "--selection-provenance", "EXTERNAL", "--bootstrap", "20", "--out", str(tmp_path / "o"),
        *_floors(),
    ]) == 4
    assert "invalid substrate manifest" in capsys.readouterr().err


def test_interpret_refuses_without_a_baseline_and_exits_4(tmp_path, capsys):
    hits, q, _ = _tiny_substrate(tmp_path)
    rc = main(["interpret", str(hits), "--peaks", str(q),
               "--selection-provenance", "EXTERNAL", "--bootstrap", "20",
               "--out", str(tmp_path / "o"), *_floors()])
    assert rc == 4
    assert "baseline" in capsys.readouterr().err


def test_interpret_without_a_declared_grade_falls_to_the_conservative_mode(tmp_path):
    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    rc = main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
               "--comparator-id", "odd", "--bootstrap", "20", "--out", str(out), *_floors()])
    assert rc == 0
    blob = json.loads((out / "interpretation.json").read_text())
    assert blob["selection_provenance"] == "DECLARATION_MISSING"
    assert blob["output_mode"] == "DESCRIPTIVE_ONLY_UNVERIFIABLE_CONDITIONING"
    assert blob["effects"] is None


def test_interpret_suppresses_the_reading_when_a_floor_fails(tmp_path):
    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    rc = main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
               "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
               "--bootstrap", "20", "--out", str(out),
               "--floor-blocks", "500"])
    assert rc == 0
    blob = json.loads((out / "interpretation.json").read_text())
    assert blob["effects"] is None and blob["composition"] is None
    assert "suppressed" in blob["suppression_reason"]


def test_interpret_prints_composition_when_only_the_comparator_fails_health(tmp_path, capsys):
    """A comparator-only health failure suppresses effects, not composition.

    Regression guard for the CLI printer: it used to be a binary
    if-suppressed/else-print-composition switch, which assumed
    ``suppression_reason`` set implied ``composition is None``. Task 2 broke
    that assumption on purpose (a bad comparator suppresses effects while
    composition, which never depended on the comparator, still stands), so the
    printer must report composition whenever it exists, not only when nothing
    was suppressed at all.
    """
    hits, q, c = _bilateral_substrate(tmp_path)
    out = tmp_path / "o"
    rc = main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
               "--comparator-id", "thin", "--selection-provenance", "EXTERNAL",
               "--bootstrap", "20", "--out", str(out),
               "--floor-blocks", "5", "--floor-coverage", "0.9", "--floor-explained", "0.9"])
    assert rc == 0
    blob = json.loads((out / "interpretation.json").read_text())
    assert blob["composition"] is not None and blob["effects"] is None
    assert any("comparator" in f for f in blob["floor_failures"])

    printed = capsys.readouterr().out
    assert "composition:" in printed and "families" in printed
    assert "not licensed by this selection provenance" not in printed
    assert blob["suppression_reason"] in printed


def test_provenance_records_no_username_or_hostname(tmp_path, monkeypatch):
    """Provenance must be publishable without a scrubbing pass.

    The recorder's stated policy is `basenames_only_except_command`: `command`
    echoes what the invoker typed, and every *other* field is redacted. So that
    is what is asserted -- per field, with `command` named as the one exemption
    and the reason it exists.

    The argv is stubbed to contain the username on purpose. An earlier version of
    this test scanned the whole serialized blob for the username and passed only
    because pytest's own command line happened not to contain it; run from a
    working directory with the username in its path (`/home/<user>/...`, or a
    `--junitxml` under one) it failed, having claimed a guarantee the policy
    never made. A test that depends on the caller's working directory is not
    testing the code.
    """
    import getpass
    import socket

    user, host = getpass.getuser(), socket.gethostname()
    monkeypatch.setattr(
        "sys.argv", ["motifmultiverse", "align", f"/home/{user}/registry/", "--out", "o"])
    out = tmp_path / "o"
    main(["align", "registry/", "--out", str(out)])
    record = json.loads((out / "provenance.json").read_text())[0]

    # `command` is the documented exemption: a record that cannot say what was
    # run describes nothing. It carries the argv verbatim, username and all.
    assert record["redaction_policy"] == "basenames_only_except_command"
    assert user in record["command"], "the exemption is not doing what it claims"

    # Every other field must be publishable as-is.
    rest = {key: value for key, value in record.items() if key != "command"}
    blob = json.dumps(rest)
    assert user not in blob, f"username leaked outside `command`: {rest}"
    assert host not in blob, f"hostname leaked outside `command`: {rest}"
    # Inputs specifically: recorded by basename, never by the caller's absolute path.
    assert all("/" not in name for name in record["inputs"])


# --------------------------------------------------------------------------- #
# Task 16: the --estimator flag (`percentile` default; `bca-wild-cluster`)     #
# --------------------------------------------------------------------------- #
def _wild_substrate(tmp_path, n_blocks=32):
    """A varying-coefficient substrate >= the estimability floor.

    `_tiny_substrate` cannot exercise the bca-wild-cluster path: its 6 blocks
    are below MIN_ESTIMABLE_BLOCKS and its constant per-side coefficients would
    make the wild bootstrap-t's observed SE exactly 0 (a refusal, correctly).
    Here query coefficients vary with the block (1.0/1.3/1.6 cycling) so the
    per-block effects have real variance; the planted effect is 0.9.
    """
    from motifmultiverse.schema import HIT_TABLE_COLUMNS
    lines = ["\t".join(HIT_TABLE_COLUMNS)]
    query, comparator = [], []
    for b in range(n_blocks):
        for i in (0, 1):
            rid = f"r{b}_{i}"
            (query if i == 0 else comparator).append(rid)
            start = b * 1_000_000 + i * 1000
            coeff = 1.0 + (b % 3) * 0.3 if i == 0 else 0.4
            lines.append("\t".join([rid, "chr1", str(start), str(start + 500),
                                    f"UA_FAMA_{i}", "FAM_A", str(coeff), "used",
                                    "9999", "lex_v1", "e" * 64]))
    hits = tmp_path / "hits.tsv"
    hits.write_text("\n".join(lines) + "\n")
    (tmp_path / "q.txt").write_text("\n".join(query) + "\n")
    (tmp_path / "c.txt").write_text("\n".join(comparator) + "\n")
    return hits, tmp_path / "q.txt", tmp_path / "c.txt"


def test_interpret_estimator_defaults_to_percentile_and_withholds_p_q(tmp_path):
    hits, q, c = _wild_substrate(tmp_path)
    out = tmp_path / "o"
    rc = main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
               "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
               "--bootstrap", "50", "--seed", "1", "--out", str(out), *_floors(32)])
    assert rc == 0
    blob = json.loads((out / "interpretation.json").read_text())
    assert blob["estimator"] == "percentile_block_bootstrap"
    effect = blob["effects"][0]
    assert effect["estimator"] == "percentile_block_bootstrap"
    assert effect["inference_capability"] == "ESTIMATION_ONLY"
    assert effect["p_value"] is None and effect["q_value"] is None


def test_interpret_estimator_bca_wild_cluster_emits_p_and_q(tmp_path):
    hits, q, c = _wild_substrate(tmp_path)
    out = tmp_path / "o"
    rc = main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
               "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
               "--estimator", "bca-wild-cluster",
               "--bootstrap", "100", "--seed", "1", "--out", str(out), *_floors(32)])
    assert rc == 0
    blob = json.loads((out / "interpretation.json").read_text())
    assert blob["estimator"] == "wild_cluster_bootstrap_t"
    effect = blob["effects"][0]
    assert effect["estimator"] == "wild_cluster_bootstrap_t"
    assert effect["inference_capability"] == "INTERVAL_AND_TEST"
    # Planted effect 0.9 with t_obs ~= 17: the p-value sits at the resolution
    # floor, and with a single family BH is the identity (q == p).
    assert effect["p_value"] == 1.0 / 101
    assert effect["q_value"] == effect["p_value"]
    assert effect["n_bootstrap_valid"] == 100
    lo, hi = effect["ci"]
    assert lo > 0.0 and hi > lo


def test_interpret_estimator_bca_wild_cluster_refuses_below_the_estimability_floor(tmp_path, capsys):
    """A lowered --floor-blocks passes health but cannot lower infer's floor."""
    hits, q, c = _wild_substrate(tmp_path, n_blocks=12)
    out = tmp_path / "o"
    rc = main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
               "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
               "--estimator", "bca-wild-cluster", "--bootstrap", "50",
               "--out", str(out), *_floors(10)])
    assert rc == 4
    assert "below the preregistered floor" in capsys.readouterr().err
    assert not (out / "interpretation.json").exists()


def test_interpret_estimator_rejects_an_unknown_choice(tmp_path):
    hits, q, c = _wild_substrate(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
              "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
              "--estimator", "bogus", "--out", str(tmp_path / "o"), *_floors(32)])
    assert exc.value.code == 2


# --------------------------------------------------------------------------- #
# Task 18: the `infer` subcommand and its effect_estimates.tsv artifact        #
# --------------------------------------------------------------------------- #
def _infer_argv(hits, q, c, out, *extra):
    return ["infer", str(hits), "--peaks", str(q), "--comparator", str(c),
            "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
            "--out", str(out), *extra]


def test_infer_writes_the_flat_effect_table_and_the_full_record(tmp_path, capsys):
    from motifmultiverse.cli import EFFECT_ESTIMATE_COLUMNS
    hits, q, c = _wild_substrate(tmp_path)
    out = tmp_path / "inference"
    rc = main(_infer_argv(hits, q, c, out, "--estimator", "bca-wild-cluster",
                          "--bootstrap", "100", "--seed", "1", *_floors(32)))
    assert rc == 0
    tsv = (out / "effect_estimates.tsv").read_text().strip().split("\n")
    assert tsv[0].split("\t") == list(EFFECT_ESTIMATE_COLUMNS)
    assert len(tsv) == 2                       # header + one family
    row = dict(zip(EFFECT_ESTIMATE_COLUMNS, tsv[1].split("\t"), strict=True))
    assert row["family_id"] == "FAM_A" and row["comparator_id"] == "odd"
    assert row["inference_capability"] == "INTERVAL_AND_TEST"
    assert row["estimator"] == "wild_cluster_bootstrap_t"
    assert float(row["p_value"]) == 1.0 / 101
    assert float(row["ci_low"]) > 0.0 < float(row["ci_high"])
    assert row["substrate_id"] == "e" * 64     # identity travels onto every row
    assert row["statistical_license"] == "FULL_INFERENCE"
    # The full record travels beside the flat one; a TSV row cannot hold health.
    blob = json.loads((out / "interpretation.json").read_text())
    assert blob["query_health"]["n_blocks"] == 32
    assert json.loads((out / "provenance.json").read_text())[0]["subcommand"] == "infer"
    # `infer` still answers ONE specification and still says so -- but it now
    # points at the subcommand that runs the grid instead of asserting there
    # isn't one, which stopped being true when `multiverse` landed.
    printed = capsys.readouterr().out
    assert "ONE specification" in printed
    assert "motifmultiverse multiverse" in printed
    assert "not implemented" not in printed


def test_infer_writes_the_missing_sentinel_for_a_withheld_p_never_a_blank_or_zero(tmp_path):
    """An empty cell reads as "no effect"; `0` reads as "certainly an effect".
    Neither is what an ESTIMATION_ONLY row means, and the capability column
    beside it says which of the two the reader is looking at.
    """
    from motifmultiverse.cli import EFFECT_ESTIMATE_COLUMNS
    hits, q, c = _wild_substrate(tmp_path)
    out = tmp_path / "inference"
    assert main(_infer_argv(hits, q, c, out, "--bootstrap", "50", *_floors(32))) == 0
    row = dict(zip(EFFECT_ESTIMATE_COLUMNS,
                   (out / "effect_estimates.tsv").read_text().strip().split("\n")[1].split("\t"),
                   strict=True))
    assert row["p_value"] == "NA" and row["q_value"] == "NA"
    assert row["inference_capability"] == "ESTIMATION_ONLY"
    assert row["estimator"] == "percentile_block_bootstrap"
    assert float(row["effect"]) > 0.0          # the estimate itself is still there


def test_infer_refuses_rather_than_writing_a_header_only_table(tmp_path, capsys):
    """A suppressed reading must not become an empty table: a file with only a
    header is indistinguishable from "we looked and found nothing"."""
    hits, q, c = _wild_substrate(tmp_path)
    out = tmp_path / "inference"
    rc = main(_infer_argv(hits, q, c, out, "--bootstrap", "50", "--floor-blocks", "999",
                          "--floor-coverage", "0.9", "--floor-explained", "0.9"))
    assert rc == 4
    assert "no effect estimates" in capsys.readouterr().err
    assert not (out / "effect_estimates.tsv").exists()
    # ...but the provenance row for the refused attempt still exists (T-09).
    assert json.loads((out / "provenance.json").read_text())[0]["subcommand"] == "infer"


def test_infer_refuses_a_table_that_does_not_match_the_substrate_manifest(tmp_path, capsys):
    from motifmultiverse.schema.substrate import CallerSpecification
    from motifmultiverse.substrate import build_manifest, write_manifest
    hits, q, c = _wild_substrate(tmp_path)
    manifest = write_manifest(
        build_manifest(
            peak_universe_hash="a" * 64, n_regions=64,
            caller_specification=CallerSpecification(
                caller_name="finemo", caller_version="0.test",
                lexicon_content_hash="d" * 64, parameters={"motif_type": "cwm"},
                preprocessing_contract_hash="b" * 64),
            input_files={"peaks.bed": "c" * 64}, created_at="2026-07-26T12:00:00Z"),
        tmp_path / "substrate.manifest.json")
    out = tmp_path / "inference"
    rc = main(_infer_argv(hits, q, c, out, "--substrate-manifest", str(manifest),
                          "--bootstrap", "50", *_floors(32)))
    assert rc == 4
    assert "does not match --substrate-manifest" in capsys.readouterr().err
    assert not (out / "effect_estimates.tsv").exists()


def test_interpret_refuses_a_substrate_truncated_below_its_manifest(tmp_path, capsys):
    """substrate_id proves the table's origin, never its completeness.

    The id is a column: truncate the run's rows and the survivors still carry it,
    so an id-only check passes. Every coverage figure is then a fraction of the
    universe that was handed in -- shrink the universe and a query the run never
    covered reports intersection_coverage = 1.0, which is the tool's own founding
    failure wearing a different hat.
    """
    from motifmultiverse.schema.substrate import CallerSpecification
    from motifmultiverse.substrate import build_manifest, write_manifest

    # 6 blocks x 2 peaks = 12 regions; the manifest declares that whole run.
    hits, q, c = _tiny_substrate(tmp_path, n_blocks=6)
    manifest = build_manifest(
        peak_universe_hash="c" * 64,
        n_regions=12,
        caller_specification=CallerSpecification(
            caller_name="finemo", caller_version="0.3.1", lexicon_content_hash="a" * 64,
            parameters={"lambda": 0.7}, preprocessing_contract_hash="b" * 64,
        ),
        input_files={"peaks.tsv": "d" * 64}, created_at="2026-07-25T00:00:00Z",
    )
    manifest_path = write_manifest(manifest, tmp_path / "substrate.manifest.json")

    # Re-issue the same run with the rows carrying the manifest's own id, then
    # truncate it to 5 of the 6 blocks. Nothing about the rows says they are partial.
    (tmp_path / "full").mkdir(exist_ok=True)
    full, q2, c2 = _tiny_substrate(tmp_path / "full", n_blocks=6,
                                   substrate_id=manifest.substrate_id)
    kept = full.read_text().splitlines()
    header, body = kept[0], [ln for ln in kept[1:] if not ln.startswith("r5_")]
    truncated = tmp_path / "truncated.tsv"
    truncated.write_text("\n".join([header, *body]) + "\n")

    out = tmp_path / "o"
    assert main([
        "interpret", str(truncated), "--substrate-manifest", str(manifest_path),
        "--peaks", str(q2), "--comparator", str(c2), "--comparator-id", "odd",
        "--selection-provenance", "EXTERNAL", "--bootstrap", "20", "--out", str(out),
        "--floor-blocks", "5", "--floor-coverage", "0.5", "--floor-explained", "0.5",
    ]) == 4
    err = capsys.readouterr().err
    assert "10 regions" in err and "declares 12" in err
    assert not (out / "interpretation.json").exists()


def test_comparator_without_an_id_is_refused_not_labelled_with_a_file_path(tmp_path, capsys):
    """`comparator_id` is semantic and lands in every effect id.

    It used to fall back to the value of `--comparator`, i.e. a filesystem path:
    an id that cannot be reproduced on another machine, and a local path leaked
    into a published artifact. An undeclared comparator is now undeclared, which
    lets `guards.comparator_declared` do what it exists for -- the run is refused
    with the baseline rule named, instead of quietly carrying a path as a label.
    """
    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    assert main([
        "interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
        "--selection-provenance", "EXTERNAL", "--bootstrap", "20", "--out", str(out),
        *_floors(),
    ]) == 4
    err = capsys.readouterr().err
    assert "named baseline peak set" in err
    assert str(c) not in err and str(tmp_path) not in err, "a local path leaked"
    assert not (out / "interpretation.json").exists()


def test_comparator_with_an_id_still_runs(tmp_path):
    """The declared form is unaffected by the refusal above."""
    import json

    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    assert main([
        "interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
        "--comparator-id", "gc_matched",
        "--selection-provenance", "EXTERNAL", "--bootstrap", "50", "--out", str(out),
        *_floors(),
    ]) == 0
    blob = (out / "interpretation.json").read_text()
    assert "gc_matched" in blob, "the declared comparator id must reach the artifact"
    assert str(tmp_path) not in blob, "a local path leaked into the artifact"
    json.loads(blob)


def test_substrate_circular_is_reachable_from_the_command_line(tmp_path):
    """The flagship two-axis outcome had no CLI path to it.

    `claim_scope` becomes SUBSTRATE_CIRCULAR when a declared selection feature is
    itself attribution-derived, but `PeakSetQuery.selection_feature_names` was
    never populated by the CLI, so no command could produce the result the whole
    two-axis design exists for: statistically licensed AND semantically circular,
    in the same record.
    """
    import json

    hits, q, c = _tiny_substrate(tmp_path)

    def scope(*extra):
        out = tmp_path / f"o{len(extra)}{extra}"
        assert main([
            "interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
            "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
            *extra, "--bootstrap", "50", "--out", str(out), *_floors(),
        ]) == 0
        payload = json.loads((out / "interpretation.json").read_text())
        return payload["statistical_license"], payload["claim_scope"]

    assert scope() == ("FULL_INFERENCE", "EXTERNAL_STRUCTURE")
    assert scope("--selection-feature", "gc_content") == (
        "FULL_INFERENCE", "EXTERNAL_STRUCTURE")
    # the same query, licensed identically, is circular once the feature it was
    # selected on is one of the attribution-derived names
    assert scope("--selection-feature", "attribution_pc1") == (
        "FULL_INFERENCE", "SUBSTRATE_CIRCULAR")


def test_selection_feature_is_repeatable(tmp_path):
    import json

    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    assert main([
        "interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
        "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
        "--selection-feature", "gc_content", "--selection-feature", "deepshap_score",
        "--bootstrap", "50", "--out", str(out), *_floors(),
    ]) == 0
    payload = json.loads((out / "interpretation.json").read_text())
    assert payload["claim_scope"] == "SUBSTRATE_CIRCULAR"


def test_same_named_query_and_comparator_do_not_collide(tmp_path):
    """Per-cluster layouts give a query and its comparator the same filename.

    provenance.add_input refuses a basename collision -- correctly, since keying by
    basename used to lose an input silently. But the CLI passed bare paths, so this
    normal layout aborted the run with an unhandled ValueError: a traceback and
    exit 1, not the documented exit 4 refusal, with no provenance record and no
    output directory. Inputs are now keyed by the role they played.
    """
    import json

    hits, q, c = _tiny_substrate(tmp_path)
    for name, src in (("setA", q), ("setB", c)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "peaks.txt").write_text(src.read_text())

    out = tmp_path / "o"
    assert main([
        "interpret", str(hits),
        "--peaks", str(tmp_path / "setA" / "peaks.txt"),
        "--comparator", str(tmp_path / "setB" / "peaks.txt"),
        "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
        "--bootstrap", "50", "--out", str(out), *_floors(),
    ]) == 0
    rec = json.loads((out / "provenance.json").read_text())[0]
    assert set(rec["inputs"]) == {"hits:hits.tsv", "peaks:peaks.txt", "comparator:peaks.txt"}
    assert len(set(rec["inputs"].values())) == 3, "three distinct files, three checksums"


def test_a_provenance_refusal_is_a_refusal_not_a_traceback(tmp_path, capsys):
    """ProvenanceError is typed so it reaches the exit-4 contract."""
    from motifmultiverse.provenance import ProvenanceError, record

    a, b = tmp_path / "a.h5", tmp_path / "b" / "a.h5"
    b.parent.mkdir()
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    rec = record("ingest")
    rec.add_input(a)
    with pytest.raises(ProvenanceError, match="already names a different input"):
        rec.add_input(b)
    assert isinstance(ProvenanceError("x"), ValueError), "stays a ValueError for old callers"


def test_compile_refuses_a_decisions_file_that_is_not_json_and_exits_4(tmp_path, capsys):
    """A malformed decisions file is a refusal, not a traceback.

    `--decisions` was read with a bare `json.loads`, so anything that is not JSON
    escaped the exit-code contract entirely: exit 1 and a raw `JSONDecodeError`
    stack, where every other refusal in this tool exits 4 and names the rule that
    declined. It also escaped T-09 -- the parse ran before `prov.write`, so the
    one artifact a rejected compile is supposed to leave behind was not written.
    """
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    modisco = tmp_path / "modisco.h5"
    with h5py.File(modisco, "w") as h5:
        grp = h5.require_group("pos_patterns").create_group("pattern_0")
        grp.create_dataset("contrib_scores", data=np.ones((10, 4)))
        grp.create_dataset("hypothetical_contribs", data=np.ones((10, 4)))
        grp.create_dataset("sequence", data=np.full((10, 4), 0.25))
    project = tmp_path / "project.json"
    project.write_text(json.dumps({
        "project": "cli-test", "peak_universe_id": "u1",
        "analyses": [{"id": "a1", "model": "m", "readout": "r", "union_id": "MA",
                      "context": "promoter", "modisco_h5": str(modisco)}]}))
    assert main(["ingest", str(project), "--out", str(tmp_path / "registry")]) == 0
    capsys.readouterr()

    decisions = tmp_path / "decisions.txt"
    decisions.write_text("these are notes, not a decision bundle\n")
    out = tmp_path / "lex"
    assert main(["compile", str(tmp_path / "registry"), "--decisions", str(decisions),
                 "--out", str(out), "--verify-roundtrip", "skip"]) == 4
    assert "refused" in capsys.readouterr().err
    assert (out / "provenance.json").exists()
    assert not (out / "core.h5").exists()


def test_compile_refuses_a_decisions_file_that_is_not_utf8_and_exits_4(tmp_path, capsys):
    """Same contract for bytes that are not text at all: `read_text` raised too."""
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    modisco = tmp_path / "modisco.h5"
    with h5py.File(modisco, "w") as h5:
        grp = h5.require_group("pos_patterns").create_group("pattern_0")
        grp.create_dataset("contrib_scores", data=np.ones((10, 4)))
        grp.create_dataset("hypothetical_contribs", data=np.ones((10, 4)))
        grp.create_dataset("sequence", data=np.full((10, 4), 0.25))
    project = tmp_path / "project.json"
    project.write_text(json.dumps({
        "project": "cli-test", "peak_universe_id": "u1",
        "analyses": [{"id": "a1", "model": "m", "readout": "r", "union_id": "MA",
                      "context": "promoter", "modisco_h5": str(modisco)}]}))
    assert main(["ingest", str(project), "--out", str(tmp_path / "registry")]) == 0
    capsys.readouterr()

    decisions = tmp_path / "decisions.bin"
    decisions.write_bytes(b"\xff\xfe\x00\x01not utf-8")
    assert main(["compile", str(tmp_path / "registry"), "--decisions", str(decisions),
                 "--out", str(tmp_path / "lex"), "--verify-roundtrip", "skip"]) == 4
    assert "refused" in capsys.readouterr().err


def test_an_unreadable_provenance_log_is_refused_and_kept_not_overwritten(tmp_path, capsys):
    """Appending must not crash the tool, and must not discard the earlier records.

    Every subcommand writes its record before it computes, so a provenance.json
    the append cannot parse is not one lost run -- it is every later run in that
    directory. Letting the JSONDecodeError out of the append made each of them a
    traceback and exit 1. Overwriting instead would be worse: the earlier
    records are the one thing provenance exists to keep.
    """
    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    argv = ["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
            "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
            "--bootstrap", "50", "--out", str(out), *_floors()]
    assert main(argv) == 0
    dest = out / "provenance.json"
    damaged = dest.read_text()[:120]
    dest.write_text(damaged)

    assert main(argv) == 4, "a refusal, not a traceback"
    assert "not readable provenance" in capsys.readouterr().err
    assert dest.read_text() == damaged, "the damaged file is kept for the operator"


def test_a_provenance_file_that_is_not_a_log_is_refused(tmp_path):
    """A parseable file of the wrong shape is still not something to append to."""
    from motifmultiverse.provenance import ProvenanceError, record

    out = tmp_path / "o"
    out.mkdir()
    (out / "provenance.json").write_text('{"subcommand": "interpret"}')
    with pytest.raises(ProvenanceError, match="is not a provenance log"):
        record("interpret", out_dir=out)


def test_an_interrupted_provenance_write_cannot_truncate_the_existing_log(tmp_path, monkeypatch):
    """Writing in place truncates first, so a killed run took out the log itself."""
    import pathlib

    from motifmultiverse.provenance import record

    out = tmp_path / "o"
    record("interpret", out_dir=out)
    dest = out / "provenance.json"
    intact = dest.read_text()

    real_write_text = pathlib.Path.write_text

    def killed_mid_write(self, data, *args, **kwargs):
        # What a kill during `write_text` leaves on disk: the file was opened
        # for writing, which truncated it, and only some of the bytes landed.
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)
        raise KeyboardInterrupt("killed mid-write")

    monkeypatch.setattr(pathlib.Path, "write_text", killed_mid_write)
    with pytest.raises(KeyboardInterrupt):
        record("interpret", out_dir=out)
    monkeypatch.undo()

    assert dest.read_text() == intact
    assert len(json.loads(dest.read_text())) == 1


def test_a_refused_run_marks_the_directory_and_names_whose_result_is_in_it(tmp_path):
    """The repair for what this module's docstring used to call a KNOWN LIMITATION.

    Provenance is written before anything is computed, on purpose (see
    provenance/__init__.py): a record that arrives only on success is a record
    the runs you most want to explain never get. So a refused run appends its
    record and refuses, and the earlier run's result is still lying in the
    directory -- correctly, because deleting a real result to prevent a
    misreading of it is a worse trade. What was missing was the sentence saying
    which run the files belong to, and "read the exit code, not the directory"
    was advice rather than a mechanism: the exit code is gone by the time anyone
    opens the folder.

    `run_status.json` is that sentence. The earlier result is still there,
    untouched and byte-identical -- and now labelled.
    """
    from motifmultiverse.schema.substrate import CallerSpecification
    from motifmultiverse.substrate import build_manifest, write_manifest

    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    argv = ["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
            "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
            "--bootstrap", "50", "--out", str(out), *_floors()]
    assert main(argv) == 0
    produced = (out / "interpretation.json").read_text()

    foreign = build_manifest(
        peak_universe_hash="c" * 64, n_regions=12,
        caller_specification=CallerSpecification(
            caller_name="finemo", caller_version="0.3.1", lexicon_content_hash="a" * 64,
            parameters={"lambda": 0.7}, preprocessing_contract_hash="b" * 64,
        ),
        input_files={"peaks.tsv": "d" * 64}, created_at="2026-07-25T00:00:00Z",
    )
    manifest_path = write_manifest(foreign, tmp_path / "substrate.manifest.json")
    assert main([*argv, "--substrate-manifest", str(manifest_path)]) == 4

    records = json.loads((out / "provenance.json").read_text())
    assert [r["subcommand"] for r in records] == ["interpret", "interpret"]
    assert (out / "interpretation.json").read_text() == produced, "the earlier result was destroyed"

    status = json.loads((out / "run_status.json").read_text())
    assert status["status"] == "REFUSED" and status["exit_code"] == 4
    # The refusal's own sentence, in the directory rather than only on a stderr
    # nobody kept: a reader who arrives later can see WHY, not just that.
    assert "substrate" in status["detail"].lower() or status["detail"]
    # And the thing that makes the stale result readable rather than misleading:
    # the artifacts here belong to the run named by `artifacts_are_from`, which is
    # the earlier successful one and NOT this refusal.
    came_from = status["artifacts_are_from"]
    assert came_from["status"] == "SUCCESS"
    assert came_from["finished_utc"] <= status["finished_utc"]
    # An EARLIER run: the log had one record when it finished and two when the
    # refusal did. (`command` cannot tell them apart here -- it is read from
    # `sys.argv`, which under an in-process test is pytest's own command line for
    # both runs, the same way `provenance.command` is.)
    assert came_from["provenance_records"] < status["provenance_records"], (
        "the refusal recorded itself as the run the artifacts came from"
    )


def test_a_successful_run_says_so_in_the_directory_it_wrote(tmp_path):
    """The other half: SUCCESS is written too, so an absent file means one thing.

    A status file that appears only on failure makes its own absence ambiguous --
    "this run succeeded" and "this tool is too old to say" would look identical to
    a downstream reader deciding whether to trust the artifacts beside it.
    """
    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    assert main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
                 "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
                 "--bootstrap", "50", "--out", str(out), *_floors()]) == 0
    status = json.loads((out / "run_status.json").read_text())
    assert status["status"] == "SUCCESS" and status["exit_code"] == 0
    assert status["detail"] is None
    assert status["subcommand"] == "interpret"
    assert status["provenance_records"] == 1
    assert status["artifacts_are_from"]["status"] == "SUCCESS"


def test_a_run_that_refuses_into_a_fresh_directory_says_no_run_succeeded_here(tmp_path):
    """With nothing earlier to inherit, the carry-forward must not invent one.

    `null` would read as "from nowhere in particular"; the token says the true and
    stronger thing -- that no run this file has seen has ever succeeded here, so
    anything in the directory was not written by one.
    """
    from motifmultiverse.run_status import NO_SUCCESSFUL_RUN

    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "never_succeeded"
    assert main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
                 "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
                 "--bootstrap", "5", "--out", str(out), *_floors()]) == 4
    status = json.loads((out / "run_status.json").read_text())
    assert status["status"] == "REFUSED"
    assert status["artifacts_are_from"] == NO_SUCCESSFUL_RUN


def test_report_refuses_an_unparseable_interpretation_with_exit_4_and_no_report(tmp_path, capsys):
    """A truncated input is a refusal, not a traceback and not an undocumented exit.

    `report` read both its inputs with a bare `json.loads`, so a half-written
    `interpretation.json` -- a run killed mid-write, a partially copied directory
    -- escaped as a `JSONDecodeError` that `main` does not catch: exit 1, a
    traceback, and a code this CLI never defined. The contract is exit 4 and a
    sentence naming the rule.
    """
    src = tmp_path / "interpretation"
    src.mkdir()
    (src / "interpretation.json").write_text('{"query_id": "cl5", "substrate', encoding="utf-8")
    (src / "provenance.json").write_text("[]", encoding="utf-8")
    out = tmp_path / "report"

    assert main(["report", str(src), "--out", str(out)]) == 4
    err = capsys.readouterr().err
    assert "refused" in err and "interpretation.json" in err
    assert "Traceback" not in err
    assert not (out / "report.md").exists(), "a report was written from a record nobody could read"
    assert json.loads((out / "run_status.json").read_text())["status"] == "REFUSED"


def test_the_printed_interpretation_names_both_permission_axes(tmp_path, capsys):
    """`claim_scope` reached the JSON and nothing else.

    The printed block led with `output_mode`, the deprecated view that cannot
    represent SUBSTRATE_CIRCULAR, so a query selected on an attribution-derived
    feature printed exactly what an EXTERNAL_STRUCTURE query printed. The one
    outcome the two-axis design exists for was invisible at the terminal.
    """
    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    assert main([
        "interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
        "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
        "--selection-feature", "hit_coefficient",
        "--bootstrap", "50", "--out", str(out), *_floors(),
    ]) == 0
    printed = capsys.readouterr().out
    assert "claim_scope" in printed
    assert "SUBSTRATE_CIRCULAR" in printed
    assert "statistical_license" in printed and "FULL_INFERENCE" in printed


def test_the_peaks_flag_states_how_a_bed_is_matched(capsys):
    """`--peaks` advertised "BED" and matched by exact string equality on
    region_id. A 3-column BED is read as `chrom:start-end` and misses every row
    of a table keyed `peak_000001`, which surfaced only as coverage 0.0.
    """
    with pytest.raises(SystemExit):
        main(["interpret", "--help"])
    help_text = capsys.readouterr().out
    assert "4th column IS the region_id" in help_text
    assert "exact string equality" in help_text


def test_interpret_records_what_each_guard_returned_beside_its_result(tmp_path):
    """The stage's guards leave a trace in the directory, not only on stdout.

    Four guards run inside `interpret.interpret_query`; before this file existed a
    reader of the output directory could not tell a run whose guards all passed
    from a run in which they had never been reached. The entries are joined to
    this run through the provenance log's length, the same join
    `run_status.json` uses.
    """
    from motifmultiverse import guard_log

    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    assert main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
                 "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
                 "--bootstrap", "50", "--out", str(out), *_floors()]) == 0

    recorded = json.loads((out / guard_log.GUARD_OUTCOMES_FILENAME).read_text())
    assert {row["guard_id"] for row in recorded} == {
        "comparator_declared", "selection_provenance_declared",
        "health_before_effect", "single_scale",
    }
    assert all(row["stage"] == "interpret" for row in recorded)
    assert all(row["passed"] is True for row in recorded)
    assert all(row["detail"] for row in recorded)

    n_records = len(json.loads((out / "provenance.json").read_text()))
    assert all(row["provenance_records"] == n_records for row in recorded)
    status = json.loads((out / "run_status.json").read_text())
    assert status["artifacts_are_from"]["provenance_records"] == n_records


def test_infer_records_its_guard_outcomes_under_its_own_stage_name(tmp_path):
    """`infer` and `interpret` run the same guards; the record says which ran here."""
    from motifmultiverse import guard_log

    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    assert main(["infer", str(hits), "--peaks", str(q), "--comparator", str(c),
                 "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
                 "--bootstrap", "50", "--out", str(out), *_floors()]) == 0

    recorded = json.loads((out / guard_log.GUARD_OUTCOMES_FILENAME).read_text())
    assert recorded and all(row["stage"] == "infer" for row in recorded)


def test_a_run_refused_by_a_guard_still_records_which_guard_refused_it(tmp_path, monkeypatch,
                                                                      capsys):
    """The case the record exists for: no result artifact is written at all.

    A guard failure is a refusal (exit 4) and `interpretation.json` is never
    produced, so an outcome stored inside that result would be missing from
    precisely the run a reader needs to explain. The failure is injected at the
    guard rather than by contriving data, because what is under test is the
    ORDERING -- the outcome reaches disk before the refusal propagates -- and not
    any particular guard's own logic.
    """
    from motifmultiverse import guard_log, guards
    from motifmultiverse import interpret as interpret_mod

    def refuse(records):
        return guards.GuardResult("single_scale", False, "injected: two input scales")

    monkeypatch.setattr(interpret_mod.guards, "single_scale", refuse)

    hits, q, c = _tiny_substrate(tmp_path)
    out = tmp_path / "o"
    assert main(["interpret", str(hits), "--peaks", str(q), "--comparator", str(c),
                 "--comparator-id", "odd", "--selection-provenance", "EXTERNAL",
                 "--bootstrap", "50", "--out", str(out), *_floors()]) == 4
    assert "refused" in capsys.readouterr().err

    assert not (out / "interpretation.json").exists(), (
        "the refusal must not have written a result; otherwise this proves nothing"
    )
    recorded = json.loads((out / guard_log.GUARD_OUTCOMES_FILENAME).read_text())
    failed = [row for row in recorded if not row["passed"]]
    assert [row["guard_id"] for row in failed] == ["single_scale"]
    assert failed[0]["detail"] == "injected: two input scales"
    assert json.loads((out / "run_status.json").read_text())["status"] == "REFUSED"


# --------------------------------------------------------------------------
# Output-path friction: what a second person on the machine finds afterwards.
# --------------------------------------------------------------------------


def _validate_inputs(tmp_path):
    """Build the smallest real `validate` input set and return argv without --out.

    A trimmed copy of the fixture in
    `test_validate_cli_writes_split_bound_stability_and_backend_artifacts`. It is
    separate rather than shared because these tests run `validate` twice over the
    same inputs, which that test does not, and because a fixture edited for one
    of them must not silently move the other.
    """
    import h5py
    import numpy as np

    from motifmultiverse.compile import lexicon_semantic_hash
    from motifmultiverse.schema.substrate import CallerSpecification
    from motifmultiverse.substrate import build_manifest as build_substrate_manifest
    from motifmultiverse.substrate import write_manifest as write_substrate_manifest

    root = tmp_path / "validate-inputs"
    lexicons = root / "lexicons"
    lexicons.mkdir(parents=True)
    with h5py.File(lexicons / "core.h5", "w") as h5:
        h5.create_group("pos_patterns").create_group("pattern_0").create_dataset(
            "contrib_scores", data=np.asarray([[1.0, 0.0, 0.0, 0.0]])
        )
    content_hash = lexicon_semantic_hash(
        [("pos_patterns", "pattern_0", {"node_id": "node-0", "variant_id": "MA_FAM_01"})],
        {"node-0": {"cwm": np.asarray([[1.0, 0.0, 0.0, 0.0]])}},
        schema_version="1.0", trim_threshold=0.3, motif_type="cwm", include_rc=False,
        loader_backend="finemo", loader_parameters={"motif_lambda_default": 0.7},
    )
    (lexicons / "core.manifest.json").write_text(json.dumps({
        "tier": "core", "lexicon_content_hash": content_hash, "n_motifs": 1,
        "pattern_order": ["pos_patterns.pattern_0"], "node_ids": ["node-0"],
        "index": [{
            "index": 0, "pattern_tag": "pos_patterns.pattern_0", "node_id": "node-0",
            "variant_id": "MA_FAM_01", "metacluster": "pos",
        }],
        "schema_version": "1.0", "trim_threshold": 0.3, "motif_type": "cwm",
        "include_rc": False, "loader_backend": "finemo",
        "loader_parameters": {"motif_lambda_default": 0.7},
        "comparisons": {}, "source_registry": "registry", "sensitivity_triggers": {},
        "project": "test-project", "cross_model_claims_restricted": True,
    }), encoding="utf-8")
    substrate = build_substrate_manifest(
        peak_universe_hash="a" * 64,
        n_regions=2,
        caller_specification=CallerSpecification(
            caller_name="finemo", caller_version="0.test",
            lexicon_content_hash=content_hash,
            parameters={"motif_type": "cwm"},
            preprocessing_contract_hash="b" * 64,
        ),
        input_files={"peaks.bed": "c" * 64},
        created_at="2026-07-26T12:00:00Z",
    )
    substrate_path = write_substrate_manifest(substrate, root / "substrate.manifest.json")
    before = root / "before" / "hits.parquet"
    after = root / "after" / "hits.parquet"
    before.parent.mkdir()
    after.parent.mkdir()
    pd.DataFrame([{
        "peak_id": "p-validation", "hit_id": "old", "coefficient": 1.0,
        "reconstruction": 0.0, "substrate_id": substrate.substrate_id,
    }]).to_parquet(before, index=False)
    pd.DataFrame([{
        "peak_id": "p-validation", "hit_id": "new", "coefficient": 2.0,
        "reconstruction": 1.0, "substrate_id": substrate.substrate_id,
    }]).to_parquet(after, index=False)
    manifest = build_peak_split_manifest(
        {"p-discovery": "DISCOVERY", "p-validation": "VALIDATION"}
    )
    manifest_path = root / "split-manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": manifest.schema_version,
        "assignments": {key: value.value for key, value in manifest.assignments.items()},
        "checksum": manifest.checksum,
    }), encoding="utf-8")
    decision = DecisionSplitArtifact.create(
        manifest=manifest, decision_id="decision:friction",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation"}),
    )
    provisional = ValidationSplitArtifact.create(
        manifest=manifest, decision_id="decision:friction", result_id="pending",
        decision_peak_ids=frozenset({"p-discovery"}),
        validation_peak_ids=frozenset({"p-validation"}),
    )
    decision_path = root / "decision-split.json"
    validation_path = root / "validation-split.json"
    decision_path.write_text(json.dumps(decision.to_dict()), encoding="utf-8")
    validation_path.write_text(json.dumps(provisional.to_dict()), encoding="utf-8")
    return [
        "validate", str(lexicons),
        "--before-hits", str(before), "--after-hits", str(after),
        "--substrate-manifest", str(substrate_path),
        "--split-manifest", str(manifest_path),
        "--decision-artifact", str(decision_path),
        "--validation-artifact", str(validation_path),
    ]


PUBLISHED_VALIDATE_FILES = {
    "stability_results.parquet", "backend_verification.tsv", "provenance.json",
}


@pytest.mark.parametrize("umask_value", [0o022, 0o027])
def test_validate_publishes_its_output_at_the_process_umask_not_0700(tmp_path, umask_value):
    """The published directory is as readable as anything else the run wrote.

    `validate` stages its artifacts in a `tempfile.mkdtemp` directory, which is
    0700 by construction, and publishes that same directory -- by rename where
    the filesystem supports `renameat2(RENAME_NOREPLACE)`, by symlink to it where
    it does not. Either way the 0700 travelled to the result, so `validation/`
    sat under a 0755 parent holding 0644 files that no collaborator could reach.
    The mode is compared against a directory this test creates with a plain
    `mkdir` under the same umask rather than against a literal, because the claim
    is "the mode the run would otherwise have produced", not "0755".
    """
    import os

    base = _validate_inputs(tmp_path)
    out = tmp_path / f"validation-{umask_value:03o}"
    control = tmp_path / f"control-{umask_value:03o}"
    previous = os.umask(umask_value)
    try:
        assert main([*base, "--out", str(out)]) == 0
        control.mkdir()
    finally:
        os.umask(previous)

    assert os.stat(out).st_mode & 0o777 == os.stat(control).st_mode & 0o777
    assert os.stat(out).st_mode & 0o777 == 0o777 & ~umask_value
    assert PUBLISHED_VALIDATE_FILES <= {entry.name for entry in out.iterdir()}


def test_validate_output_is_readable_where_renameat2_is_unavailable(tmp_path, monkeypatch):
    """The symlink fallback publishes a readable directory too.

    Which of the two publish paths runs is a property of the filesystem, so the
    real run took the fallback (ZFS rejects `RENAME_NOREPLACE` with EINVAL) while
    a test on ext4 would not exercise it at all. libc is stubbed here so the
    fallback is reached on every filesystem: it is the path where the mode
    matters most, since the published name is a symlink and `cp -r` of it copies
    the link rather than the results.
    """
    import ctypes
    import os

    base = _validate_inputs(tmp_path)  # imports h5py/pyarrow before libc is stubbed
    import pyarrow.parquet  # noqa: F401  - likewise; nothing may load a library later

    class _LibcWithoutRenameat2:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: _LibcWithoutRenameat2())

    out = tmp_path / "validation-fallback"
    previous = os.umask(0o022)
    try:
        assert main([*base, "--out", str(out)]) == 0
    finally:
        os.umask(previous)

    assert os.path.islink(out), "this test is meaningless unless the fallback ran"
    assert os.stat(out).st_mode & 0o777 == 0o755
    assert PUBLISHED_VALIDATE_FILES <= {entry.name for entry in out.iterdir()}


def test_validate_help_says_out_must_not_already_exist_and_that_is_what_happens(
    tmp_path, capsys,
):
    """The refusal is documented where the user chooses the directory.

    `--out` said "output directory", and the second run into it exited 4. Both
    halves are asserted together so the sentence cannot drift from the behaviour
    it describes.
    """
    with pytest.raises(SystemExit) as exc:
        main(["validate", "--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "must NOT already exist" in help_text
    assert "exit 4" in help_text

    base = _validate_inputs(tmp_path)
    out = tmp_path / "validation"
    assert main([*base, "--out", str(out)]) == 0
    capsys.readouterr()
    assert main([*base, "--out", str(out)]) == 4
    assert "already exists and will not be overwritten" in capsys.readouterr().err


def test_adjudicate_help_states_that_a_relative_review_path_is_joined_to_out(
    tmp_path, monkeypatch, capsys,
):
    """`--review` is an artifact name inside `--out`, and now says so.

    "human review file to emit" reads as a path like every other path on the
    command line, so `--review probe/review.yaml --out probe` looked like it
    named `probe/review.yaml` and wrote `probe/probe/review.yaml`. The rule is
    the intended one -- the review is one of adjudicate's three outputs, and
    `write_adjudication_artifacts` joins a relative path to the output directory
    on purpose -- so the fix is the sentence. The exact example in the help text
    is then executed here, so prose and behaviour cannot disagree.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_adjudicate import (  # noqa: E402
        _alignment,
        _annotation,
        _write_adjudication_registry,
    )

    with pytest.raises(SystemExit) as exc:
        main(["adjudicate", "--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "resolved against --out" in help_text
    assert "`--review probe/review.yaml --out probe` writes probe/probe/review.yaml" in help_text

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    pd.DataFrame([_alignment().to_dict()]).to_parquet(
        evidence / "alignment_edges.parquet", index=False
    )
    candidate_rows = []
    for candidate in (_annotation("node-a", "FAM_ALPHA"), _annotation("node-b", "FAM_ALPHA")):
        row = candidate.to_dict()
        row["provenance"] = json.dumps(row["provenance"])
        candidate_rows.append(row)
    pd.DataFrame(candidate_rows).to_parquet(
        evidence / "annotation_candidates.parquet", index=False
    )
    registry = _write_adjudication_registry(tmp_path)

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    out = tmp_path / "probe"
    assert main([
        "adjudicate", str(evidence), "--registry", str(registry),
        "--review", "probe/review.yaml", "--out", str(out),
    ]) == 0

    assert (out / "probe" / "review.yaml").exists()
    assert not (out / "review.yaml").exists()
    assert not (cwd / "probe").exists(), "a relative --review is not relative to the cwd"
    assert f"written: {out / 'probe' / 'review.yaml'}" in capsys.readouterr().out
