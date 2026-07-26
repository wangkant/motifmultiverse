"""CLI tests: all eight subcommands expose real --help and record provenance (T-08/T-09)."""
from __future__ import annotations

import json

import pytest

from motifmultiverse.cli import build_parser, main

SUBCOMMANDS = ["ingest", "align", "annotate", "adjudicate",
               "compile", "validate", "infer", "report"]
IMPLEMENTED = ["ingest", "compile", "interpret", "align"]


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


@pytest.mark.parametrize("argv", [
    ["adjudicate", "evidence/"],
    ["validate", "lexicons/"],
    ["infer", "instances/"],
    ["report", "project/"],
])
def test_unimplemented_bodies_exit_3_and_name_their_readme(argv, capsys, tmp_path):
    rc = main(argv + ["--out", str(tmp_path / "o")])
    assert rc == 3
    err = capsys.readouterr().err
    assert "src/motifmultiverse/" in err and "README.md" in err


def test_provenance_is_written_even_though_body_is_unimplemented(tmp_path):
    # "align" and "annotate" no longer belong here (Tasks 10 and 11 implement
    # them for real); "adjudicate" remains a genuine skeleton. It has no --seed flag
    # (only ingest/compile/interpret/align ever exposed one), so the recorded
    # random_seed is the untouched default (None) rather than a value threaded
    # in from the CLI -- the property under test is that the record exists
    # at all before the body raises, not what value a particular field holds.
    out = tmp_path / "o"
    assert main(["adjudicate", "evidence/", "--out", str(out)]) == 3
    recs = json.loads((out / "provenance.json").read_text())
    assert len(recs) == 1
    r = recs[0]
    for field in ("command", "subcommand", "software", "timestamp_utc", "random_seed"):
        assert field in r, field
    assert r["subcommand"] == "adjudicate"
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


def test_provenance_appends_rather_than_overwrites(tmp_path):
    out = tmp_path / "o"
    main(["align", "registry/", "--out", str(out)])
    main(["adjudicate", "evidence/", "--out", str(out)])
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
    assert set(rec["inputs"]) == {"hits.tsv", "q.txt", "c.txt"}
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
        "--selection-provenance", "EXTERNAL", "--bootstrap", "20", "--out", str(out),
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


def test_provenance_records_no_username_or_hostname(tmp_path):
    """Provenance must be publishable without a scrubbing pass."""
    import getpass
    import socket
    out = tmp_path / "o"
    main(["align", "registry/", "--out", str(out)])
    blob = (out / "provenance.json").read_text()
    assert getpass.getuser() not in blob
    assert socket.gethostname() not in blob
