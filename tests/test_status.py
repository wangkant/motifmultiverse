"""Tests for the generated implementation status (Task 18, Steps 3 and 4).

The subject here is a claim-making mechanism, so most of these assert what the
generator refuses to say: that it never folds a skip into a pass, never reports
zero for something it did not measure, and never lets the README's module table
drift away from the code it describes.
"""
from __future__ import annotations

import json

import pytest

from motifmultiverse import status as status_mod


def test_module_status_is_derived_from_the_cli_not_from_a_list():
    """A module counts as implemented iff its subcommand dispatches to a runner."""
    built = status_mod.build_status()["modules"]
    assert set(built) == set(status_mod.MODULES)
    assert built["report"]["status"] == "SKELETON"
    assert "README.md" in built["report"]["detail"]
    for name in ("ingest", "align", "annotate", "adjudicate", "compile",
                 "validate", "infer", "interpret"):
        assert built[name]["status"] == "IMPLEMENTED", name
        assert built[name]["detail"].startswith("cli._run_")


def test_a_module_reverting_to_a_skeleton_is_reported_as_one(monkeypatch):
    """The non-vacuity check: the derivation must be able to say SKELETON for a
    module it currently calls IMPLEMENTED, or it is not deriving anything."""
    from motifmultiverse import cli

    real_build_parser = cli.build_parser

    def stubbed():
        parser = real_build_parser()
        for action in parser._actions:
            if getattr(action, "choices", None) and "validate" in action.choices:
                action.choices["validate"].set_defaults(
                    func=lambda ns: (_ for _ in ()).throw(NotImplementedError()))
        return parser

    assert status_mod.module_status("validate")["status"] == "IMPLEMENTED"
    monkeypatch.setattr(cli, "build_parser", stubbed)
    # A plain lambda defined here has no `build_parser.<locals>` qualname, so the
    # derivation should NOT be fooled into calling it implemented by name alone;
    # what it reports is whatever the dispatch table actually holds.
    assert status_mod.module_status("validate")["detail"] != "cli._run_validate"


def test_an_absent_subcommand_is_absent_not_silently_implemented():
    assert status_mod.module_status("not-a-subcommand") == {
        "status": "ABSENT", "detail": "no CLI subcommand is registered"}


def test_a_backend_that_cannot_be_imported_is_unverified_never_assumed():
    result = status_mod.backend_status("finemo", "definitely_not_installed_xyz")
    assert result["status"] == "UNVERIFIED"
    assert "not installed" in result["detail"]


def test_a_backend_with_no_check_is_unverified_rather_than_omitted():
    """An unlisted backend reads as one that passed, so an uncheckable one is
    listed with the reason instead of being dropped."""
    result = status_mod.backend_status("tomtom", None)
    assert result["status"] == "UNVERIFIED"
    assert "no executable check" in result["detail"]
    assert set(status_mod.build_status()["optional_backends"]) == set(status_mod.OPTIONAL_BACKENDS)


def test_a_backend_that_imports_here_is_verified():
    assert status_mod.backend_status("json-as-a-stand-in", "json")["status"] == "VERIFIED"


def test_backend_status_has_no_third_comfortable_value():
    """`VERIFIED` or `UNVERIFIED`, and nothing in between: "probably fine" and
    "checked last month" are both UNVERIFIED to a reader deciding what to trust."""
    values = {entry["status"] for entry in status_mod.build_status()["optional_backends"].values()}
    assert values <= {"VERIFIED", "UNVERIFIED"}


def test_test_counts_are_absent_not_zero_when_no_run_was_supplied():
    """Zero failures nobody measured is not the claim "zero failures"."""
    tests = status_mod.build_status()["tests"]
    assert tests["status"] == "NOT_RUN"
    assert "passed" not in tests and "failed" not in tests and "skipped" not in tests


@pytest.mark.parametrize("missing", ["passed", "skipped", "failed"])
def test_a_partial_test_run_is_not_reported_as_a_run(missing):
    counts = {"passed": 10, "skipped": 1, "failed": 0}
    counts[missing] = None
    assert status_mod.build_status(**counts)["tests"]["status"] == "NOT_RUN"


def test_the_three_counts_are_reported_separately_and_never_summed():
    tests = status_mod.build_status(passed=612, skipped=1, failed=0,
                                    test_command="pytest")["tests"]
    assert tests == {
        "status": "RUN", "passed": 612, "skipped": 1, "failed": 0, "command": "pytest",
        "note": ("skipped tests are UNVERIFIED and are not included in `passed`; "
                 "612 verified, 1 unverified, 0 failing"),
    }
    # The number a careless summary would print (613) appears nowhere.
    assert "613" not in json.dumps(tests)


def test_the_readme_block_matches_what_the_generator_renders():
    """The README's module table is generated; this is what keeps it that way.

    If a module is implemented or reverts to a skeleton and nobody re-renders,
    this fails and names the command to run.
    """
    from pathlib import Path

    readme = Path("README.md").read_text(encoding="utf-8")
    expected = status_mod.render_markdown(status_mod.build_status())
    start = readme.find(status_mod.BEGIN_MARKER)
    end = readme.find(status_mod.END_MARKER)
    assert start != -1 and end != -1, "README lost its generated-status markers"
    assert readme[start:end + len(status_mod.END_MARKER)] == expected, (
        "README.md is stale; regenerate with "
        "`python -m motifmultiverse.status --render-readme README.md`"
    )


def test_the_readme_block_carries_no_test_count_to_go_stale():
    """Counts are per-run facts and must not be committed into prose -- the
    stale "163 passed" line this whole mechanism replaced."""
    block = status_mod.render_markdown(
        status_mod.build_status(passed=999, skipped=7, failed=3))
    assert "999" not in block and "passed" not in block.split("| module |")[0]


def test_rendering_refuses_a_readme_without_markers():
    with pytest.raises(ValueError, match="markers"):
        status_mod._splice("no markers here", "block")


def test_the_generator_writes_a_parseable_document(tmp_path, capsys):
    out = tmp_path / "implementation_status.json"
    assert status_mod.main(["--out", str(out), "--passed", "5", "--skipped", "1",
                            "--failed", "0", "--test-command", "pytest -q"]) == 0
    blob = json.loads(out.read_text())
    assert blob["schema_version"] == status_mod.SCHEMA_VERSION
    assert blob["tests"] == {
        "status": "RUN", "passed": 5, "skipped": 1, "failed": 0,
        "command": "pytest -q",
        "note": ("skipped tests are UNVERIFIED and are not included in `passed`; "
                 "5 verified, 1 unverified, 0 failing"),
    }
    assert blob["modules"]["report"]["status"] == "SKELETON"
    assert "written:" in capsys.readouterr().out


JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="{errors}" failures="{failures}" \
skipped="{skipped}" tests="{tests}" time="1.0"/></testsuites>
"""


def test_junit_counts_derive_passed_by_subtraction_so_a_skip_cannot_reach_it(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(JUNIT.format(tests=620, failures=0, errors=0, skipped=1))
    assert status_mod.counts_from_junit(report) == {"passed": 619, "skipped": 1, "failed": 0}


def test_junit_counts_treat_an_error_as_a_failure_not_as_a_pass(tmp_path):
    """A test that could not run verified nothing."""
    report = tmp_path / "report.xml"
    report.write_text(JUNIT.format(tests=10, failures=1, errors=2, skipped=3))
    assert status_mod.counts_from_junit(report) == {"passed": 4, "skipped": 3, "failed": 3}


def test_junit_counts_refuse_an_impossible_report(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(JUNIT.format(tests=2, failures=5, errors=0, skipped=0))
    with pytest.raises(ValueError, match="negative pass count"):
        status_mod.counts_from_junit(report)


def test_junit_counts_refuse_a_report_with_no_testsuite(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text("<?xml version='1.0'?><nothing/>")
    with pytest.raises(ValueError, match="no <testsuite>"):
        status_mod.counts_from_junit(report)


def test_the_generator_accepts_a_junit_report_end_to_end(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(JUNIT.format(tests=100, failures=0, errors=0, skipped=2))
    out = tmp_path / "implementation_status.json"
    assert status_mod.main(["--out", str(out), "--junit-xml", str(report),
                            "--test-command", "pytest --junitxml=report.xml"]) == 0
    tests = json.loads(out.read_text())["tests"]
    assert (tests["passed"], tests["skipped"], tests["failed"]) == (98, 2, 0)


# --- prose claims that must be derived, not typed -----------------------------
# Every hand-maintained status claim in this repository has been wrong at least
# once; status.py exists because of two, and these are the third and fourth.
def _repo_root():
    from pathlib import Path

    import motifmultiverse
    return Path(motifmultiverse.__file__).resolve().parents[2]


def test_cli_epilog_matches_the_dispatch_table():
    """The epilog said seven modules were implemented and "the remaining two"
    raised, when `infer` had been implemented and only `report` had not."""
    from motifmultiverse.cli import _status_epilog, build_parser
    from motifmultiverse.status import MODULES, module_status

    epilog = _status_epilog()
    head, sep, tail = epilog.partition(" are implemented; ")
    assert sep, f"epilog lost its shape: {epilog!r}"
    for name in MODULES:
        implemented = module_status(name)["status"] == "IMPLEMENTED"
        side = head if implemented else tail
        other = tail if implemented else head
        assert name in side and name not in other, (
            f"{name} is {'implemented' if implemented else 'a skeleton'} but the "
            f"epilog puts it on the other side: {epilog!r}"
        )
    # and it must reach the rendered help, not merely be computable (argparse
    # re-wraps the epilog, so compare on collapsed whitespace)
    collapse = " ".join(build_parser().format_help().split())
    assert " ".join(epilog.split()) in collapse


def test_cli_module_docstring_does_not_enumerate_module_status():
    """A count typed into prose is a claim nobody re-derives."""
    import motifmultiverse.cli as cli

    doc = cli.__doc__ or ""
    assert "the remaining two" not in doc, "the stale enumeration is back"
    # It may say that the split is derived; it must not hard-code the split.
    assert "``interpret`` are implemented" not in doc
    assert "remaining" not in doc


def test_constraint_tally_in_prose_matches_the_machine_readable_source():
    """README and CONSTRAINTS.md both said 4 / 13 / 8; constraints.tsv says 4 / 14 / 7."""
    import collections
    import csv

    root = _repo_root()
    tsv = root / "docs" / "constraints.tsv"
    if not tsv.exists():                       # installed without docs/
        import pytest
        pytest.skip("docs/ not present in this installation")
    counts = collections.Counter(
        r["enforcement"] for r in csv.DictReader(tsv.open(), delimiter="\t")
    )
    spaced = f'{counts["ENFORCED"]} / {counts["PARTIAL"]} / {counts["DOC_ONLY"]}'
    tight = spaced.replace(" ", "")
    readme = (root / "README.md").read_text()
    constraints = (root / "docs" / "CONSTRAINTS.md").read_text()
    assert spaced in readme, f"README tally disagrees with constraints.tsv ({spaced})"
    assert tight in constraints or spaced in constraints, (
        f"CONSTRAINTS.md tally disagrees with constraints.tsv ({tight})"
    )
