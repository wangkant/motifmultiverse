"""Tests for the generated implementation status (Task 18, Steps 3 and 4).

The subject here is a claim-making mechanism, so most of these assert what the
generator refuses to say: that it never folds a skip into a pass, never reports
zero for something it did not measure, and never lets the README's module table
drift away from the code it describes.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from motifmultiverse import status as status_mod


def test_module_status_is_derived_from_the_cli_not_from_a_list():
    """A module counts as implemented iff its subcommand dispatches to a runner."""
    built = status_mod.build_status()["modules"]
    assert set(built) == set(status_mod.MODULES)
    # `report` was the last SKELETON here; it is asserted by name rather than
    # only inside the loop below because the loop would keep passing if the
    # module vanished from MODULES entirely.
    assert built["report"] == {"status": "IMPLEMENTED", "detail": "cli._run_report"}
    for name in ("ingest", "align", "annotate", "adjudicate", "compile",
                 "validate", "infer", "interpret", "report"):
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


def _probe(module: str, run) -> status_mod.BackendProbe:
    return status_mod.BackendProbe(
        module=module, capability="the stand-in capability under test", run=run)


def test_a_backend_that_cannot_be_imported_is_unverified_never_assumed():
    result = status_mod.backend_status(
        "finemo", _probe("definitely_not_installed_xyz", lambda: "unreachable"))
    assert result["status"] == "UNVERIFIED"
    assert "not installed" in result["detail"]


def test_a_backend_with_no_check_is_unverified_rather_than_omitted():
    """An unlisted backend reads as one that passed, so an uncheckable one is
    listed with the reason instead of being dropped."""
    result = status_mod.backend_status("tomtom", None)
    assert result["status"] == "UNVERIFIED"
    assert "no executable check" in result["detail"]
    assert set(status_mod.build_status()["optional_backends"]) == set(status_mod.OPTIONAL_BACKENDS)


def test_verified_requires_a_capability_that_ran_here_not_merely_an_import():
    """The claim behind VERIFIED, and the fact that an import cannot make it.

    `backend_status` used to return VERIFIED the moment `import_module` returned,
    which this package's own code contradicts: `compile` separates
    `BackendMissing` from `BackendIncompatible` precisely because an importable
    backend may be unable to read a lexicon back, and finemo 0.40 renamed an
    argument so that every importable installation of it was in exactly that
    state. A status document that called those machines VERIFIED was describing
    the import system, not the backend.
    """
    ran: list[str] = []

    def capable() -> str:
        ran.append("probe")
        return "the stand-in round trip returned what it was given"

    result = status_mod.backend_status("json-as-a-stand-in", _probe("json", capable))
    assert result["status"] == "VERIFIED"
    assert ran == ["probe"], "VERIFIED was reported without running the probe"
    assert "the stand-in round trip returned what it was given" in result["detail"]
    assert "the stand-in capability under test" in result["detail"], (
        "a VERIFIED must say what was verified; a reader cannot go and look the probe up"
    )


def test_an_importable_backend_whose_capability_check_fails_is_unverified():
    """The falsification half: the probe must be able to turn a green into a red.

    This is the state an incompatible backend release puts a machine in --
    installed, importable, and unable to perform the one operation `compile`
    exists to guarantee. Without this test the capability check could silently
    stop running and every backend would go on reporting VERIFIED, which is the
    guard-that-has-never-failed shape the README's finding 4 is about.
    """
    def incapable() -> str:
        raise RuntimeError("the loader read back ['pattern_1'], not ['pattern_0']")

    result = status_mod.backend_status("json-as-a-stand-in", _probe("json", incapable))
    assert result["status"] == "UNVERIFIED"
    assert "imports but the capability check failed" in result["detail"]
    assert "pattern_1" in result["detail"], "the reason a reader would act on is dropped"


def test_a_probe_that_fails_unexpectedly_still_only_costs_a_verification():
    """A broken probe must not take the status document down with it.

    The document exists to be generated on machines whose backends are in unknown
    states; a generator that raises on a surprising failure produces no status at
    all, which is strictly less informative than UNVERIFIED.
    """
    def explodes() -> str:
        raise KeyboardInterrupt("something no caller predicted")

    result = status_mod.backend_status("json-as-a-stand-in", _probe("json", explodes))
    assert result["status"] == "UNVERIFIED"
    assert "KeyboardInterrupt" in result["detail"]


def test_the_shipped_finemo_probe_is_a_round_trip_and_not_an_import(monkeypatch):
    """The probe this package actually ships must exercise the real loader.

    Asserted against the shipped `OPTIONAL_BACKENDS` entry rather than a stand-in,
    because the stand-ins above prove the mechanism and this proves the wiring:
    the entry could be a lambda returning "ok" and every test above would still
    pass.
    """
    entry = status_mod.OPTIONAL_BACKENDS["finemo"]
    assert entry is not None and entry.module == "finemo"
    assert "read back" in entry.capability

    called: dict[str, object] = {}

    def fake_probe_backend(**kwargs):
        called["called"] = True
        called.update(kwargs)
        return "stand-in evidence"

    from motifmultiverse import compile as compile_mod

    monkeypatch.setattr(compile_mod, "probe_backend", fake_probe_backend)
    monkeypatch.setitem(sys.modules, "finemo", types.ModuleType("finemo"))
    result = status_mod.backend_status("finemo", entry)
    assert result["status"] == "VERIFIED"
    assert called.get("called"), "the shipped finemo probe does not call compile.probe_backend"


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

    # Resolved against this file, not the cwd. With a bare relative path the whole
    # suite went red from any directory but the repository root -- a clean checkout
    # run as `pytest path/to/tests` failed on FileNotFoundError: README.md.
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
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


def test_the_table_states_what_IMPLEMENTED_covers_and_shows_the_evidence_per_row():
    """A status word that can be separated from its scope is the `VERIFIED` defect.

    `BackendProbe.capability` is rendered into the status document for a reason
    this file states out loud: so that "nobody has to open the source to learn
    what green covers". The module table made the same kind of claim without the
    same discipline. It printed IMPLEMENTED and stopped -- no statement of what
    the word establishes, and no sign of `module_status`'s `detail`, which names
    the runner the subcommand dispatches to and IS the evidence the derivation
    ran on. The scope sat in hand-written prose outside the markers, where it can
    be edited, moved, or quoted away from the table it qualifies, while the block
    itself travels on its own into `PKG-INFO` and onto the project page.

    So the block now carries both. `IMPLEMENTED` is a claim about dispatch and
    says so, next to the dispatch it is claiming.
    """
    block = status_mod.render_markdown(status_mod.build_status())
    scope = block.split("| module |")[0]

    assert "dispatches" in scope, "the block does not say what IMPLEMENTED covers"
    assert "not a claim" in scope, "the block does not say what IMPLEMENTED does not cover"
    for name in status_mod.MODULES:
        detail = status_mod.module_status(name)["detail"]
        assert detail in block, f"{name}: the table drops the evidence it derived from"


def test_the_scope_sentence_is_generated_rather_than_typed_beside_the_table():
    """Non-vacuity: the scope must come from the same render as the rows.

    A caveat that lives in the hand-written prose passes any test that greps the
    README, and drifts the first time somebody rewrites the surrounding section.
    This asserts it is inside the markers -- the part `--render-readme` rewrites
    wholesale and `test_the_readme_block_matches_what_the_generator_renders`
    holds to the code.
    """
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    start = readme.find(status_mod.BEGIN_MARKER)
    end = readme.find(status_mod.END_MARKER) + len(status_mod.END_MARKER)
    committed_block = readme[start:end]

    assert "dispatches" in committed_block.split("| module |")[0]


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
    assert blob["modules"]["report"]["status"] == "IMPLEMENTED"
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


def test_no_module_readme_claims_to_be_unimplemented_while_it_is_implemented():
    """The module READMEs ship inside the wheel, so a stale one reaches users.

    `annotate/README.md` opened with "Every other module in this package is
    *unimplemented*" and tabulated six implemented modules as unimplemented,
    while its own body was described as raising NotImplementedError. A reader who
    pip-installed the package was told the tool does nothing.
    """

    from motifmultiverse.status import MODULES, module_status

    root = _repo_root() / "src" / "motifmultiverse"
    stale: list[str] = []
    for name in MODULES:
        readme = root / name / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text()
        implemented = module_status(name)["status"] == "IMPLEMENTED"
        # A module that IS implemented must not say its own body raises.
        if implemented and "The body raises `NotImplementedError`" in text:
            stale.append(f"{name}: says its body raises but it is implemented")
        # ... and must not tabulate other implemented modules as unimplemented.
        for other in MODULES:
            if module_status(other)["status"] != "IMPLEMENTED":
                continue
            if f"unimplemented (`{other}`" in text or f"`{other}`, " in text.split("unimplemented (")[-1][:200]:
                if "unimplemented (" in text and f"`{other}`" in text.split("unimplemented (")[1][:200]:
                    stale.append(f"{name}: tabulates implemented module {other} as unimplemented")
    assert not stale, "stale module READMEs:\n  " + "\n  ".join(sorted(set(stale)))


def test_every_readme_quickstart_command_parses():
    """The README's own `validate` line omitted a required argument and exited 2.

    A documented command that the parser rejects is a claim the tool contradicts
    the moment anyone copies it.

    This checks parsing across EVERY fenced block, including the ones whose inputs
    a reader has to supply. The quickstart's own sequence generates its inputs and
    is therefore executed rather than parsed, in
    `test_the_readme_quickstart_actually_runs`.
    """
    import shlex

    from motifmultiverse.cli import build_parser

    readme = (_repo_root() / "README.md").read_text()
    commands: list[str] = []
    for block in readme.split("```")[1::2]:
        buffer = ""
        for raw in block.splitlines():
            line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
            if not line.strip():
                continue
            buffer += line[:-1] + " " if line.endswith("\\") else line
            if not line.endswith("\\"):
                if buffer.strip().startswith("motifmultiverse "):
                    commands.append(buffer.strip())
                buffer = ""
    assert commands, "no motifmultiverse commands found in README code blocks"

    parser = build_parser()
    failures = []
    for command in commands:
        argv = shlex.split(command)[1:]
        if argv and argv[0].startswith("-"):        # --help / --version
            continue
        try:
            parser.parse_args(argv)
        except SystemExit:
            failures.append(command)
    assert not failures, "README commands the parser rejects:\n  " + "\n  ".join(failures)


def test_no_enforced_row_rests_only_on_a_guard_with_no_call_site():
    """`FP-12` and `FP-21` were `ENFORCED` on guards nothing ever calls.

    Both rows named exactly one mechanism -- `guards.estimability_floor` and
    `guards.interaction_required` -- and `guards.GUARDS_AWAITING_INPUT` records
    both as having no call site in this release. So an artifact violating either
    principle came out of this tool unchallenged while the table said the rule was
    enforced, which is the failure `guards/__init__.py` names outright: a guard
    that is defined, exported and never invoked reads as protection.

    The check is derived from the machine-readable table and the live registry
    rather than typed, so relabelling a row `ENFORCED` without wiring the guard up
    fails here instead of shipping.
    """
    import csv
    import re

    from motifmultiverse.guards import GUARDS_AWAITING_INPUT

    root = _repo_root()
    tsv = root / "docs" / "constraints.tsv"
    if not tsv.exists():                       # installed without docs/
        import pytest
        pytest.skip("docs/ not present in this installation")

    # A cited mechanism is a dotted reference: `guards.x`, `schema.Y`,
    # `validate.z`. A row is vacuous when every mechanism it cites is a guard the
    # registry says is waiting for an input it never receives.
    reference = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*")
    vacuous = []
    for row in csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"):
        if row["enforcement"] != "ENFORCED":
            continue
        cited = reference.findall(row["enforced_by"])
        assert cited, f"{row['principle_id']}: ENFORCED but names no mechanism at all"
        awaiting = [
            name for name in cited
            if name.startswith("guards.")
            and name.removeprefix("guards.") in GUARDS_AWAITING_INPUT
        ]
        if len(awaiting) == len(cited):
            vacuous.append(f"{row['principle_id']} rests only on {', '.join(awaiting)}")
    assert not vacuous, (
        "ENFORCED rows whose only named enforcement has no call site:\n  "
        + "\n  ".join(vacuous)
    )


def test_no_row_claims_enforcement_from_a_guard_with_no_call_site_without_saying_so():
    """The correction applied to two `ENFORCED` rows had left four others alone.

    `test_no_enforced_row_rests_only_on_a_guard_with_no_call_site` looks at
    `ENFORCED` rows only, and at whether the row rests *solely* on a pending guard.
    Underneath it, `FP-02` said `guards.no_cross_model_cwm_avg` "enforces the
    cross-model prohibition", `FP-19` said `guards.single_family_layer` "enforces
    the NOT_ESTIMABLE share", `FP-22` listed `guards.four_state_missingness` among
    the mechanisms of an `ENFORCED` row, and `FP-24` credited
    `guards.stratum_parity` with the one-rule clause -- all four of guards the
    registry itself records as never having met an artifact. A reader deciding how
    much of this table to trust cannot tell those citations from the wired ones.

    So the rule is not about the label: ANY row that cites a pending guard has to
    say, in the same cell, that it has no call site. Prose is graded by one literal
    phrase on purpose -- a checkable claim beats a well-phrased one.
    """
    import csv
    import re

    from motifmultiverse.guards import GUARDS_AWAITING_INPUT

    root = _repo_root()
    tsv = root / "docs" / "constraints.tsv"
    if not tsv.exists():                       # installed without docs/
        import pytest
        pytest.skip("docs/ not present in this installation")

    cited = re.compile(r"\bguards\.([A-Za-z_][A-Za-z0-9_]*)")
    undisclosed = []
    for row in csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"):
        pending = sorted({name for name in cited.findall(row["enforced_by"])
                          if name in GUARDS_AWAITING_INPUT})
        if pending and "no call site" not in row["enforced_by"].lower():
            undisclosed.append(f"{row['principle_id']} cites {', '.join(pending)}")
    assert not undisclosed, (
        "rows citing a guard that never meets an artifact, without saying so:\n  "
        + "\n  ".join(undisclosed)
    )


def _prose_units(text: str):
    """The unit a claim is made in: one table row, or one paragraph.

    A markdown table is a single blank-line-delimited block, so checking blocks
    would let one honest row excuse every other row in the same table. Rows are
    therefore split out and the rest is read a paragraph at a time, which is the
    span a qualifying clause can plausibly reach across.
    """
    for block in text.split("\n\n"):
        rows = [line for line in block.splitlines() if line.lstrip().startswith("|")]
        if rows:
            yield from rows
            yield "\n".join(line for line in block.splitlines() if line not in rows)
        else:
            yield block


def test_no_shipped_document_cites_a_guard_with_no_call_site_as_a_check():
    """Six guards, cited as mechanisms in nine places, none of which ran.

    `docs/BIAS_LEDGER.md` listed `guards.estimability_floor` as what handles the
    estimability-floor bias; `infer/README.md` answered "how to check it" with three
    guards that check nothing here; `interpret/README.md` said 30 was "the floor
    `guards.estimability_floor` applies to N", which it applies to nothing. Each
    sentence is true about the guard and false about this release, and the reader it
    misleads is the one budgeting how much to trust the tool.

    A citation counts as disclosed if its unit says "no call site" or marks the name
    with the README's dagger. Both are literal strings on purpose: a claim a test can
    check is worth more than a claim that reads well. `CHANGELOG.md` is exempt --
    it records what was true when it was written, and editing it would be a different
    kind of dishonesty.
    """
    import re

    from motifmultiverse.guards import GUARDS_AWAITING_INPUT

    root = _repo_root()
    if not (root / "docs").exists():           # installed without docs/
        import pytest
        pytest.skip("docs/ not present in this installation")

    documents = [root / "README.md", *sorted((root / "docs").glob("*.md")),
                 *sorted((root / "src" / "motifmultiverse").rglob("README.md"))]
    assert len(documents) > 5, "the document set collapsed; this test would pass vacuously"

    offenders = []
    for path in documents:
        text = path.read_text()
        for unit in _prose_units(text):
            named = [name for name in GUARDS_AWAITING_INPUT
                     if re.search(rf"(?<![\w.]){re.escape(name)}\b", unit)]
            if not named or "no call site" in unit.lower():
                continue
            undaggered = [n for n in named if f"`{n}`†" not in unit]
            if undaggered:
                offenders.append(
                    f"{path.relative_to(root)}: {', '.join(undaggered)} cited as a check "
                    f"in -- {unit.strip()[:90]}..."
                )
    assert not offenders, (
        "documents citing a guard that has never met an artifact, without saying so:\n  "
        + "\n  ".join(offenders)
    )


def test_the_readme_marks_exactly_the_guards_that_have_no_call_site():
    """The guard list is the most-read place where an unwired guard reads as protection.

    Fifteen names in a row, nothing distinguishing the six that have never been
    handed an artifact. The dagger set is derived from `GUARDS_AWAITING_INPUT` here
    rather than typed, so wiring a guard without unmarking it -- or marking one that
    is wired -- fails instead of shipping, and the README cannot drift into
    flattering the code the way the constraint tally once did.
    """
    import re

    from motifmultiverse.guards import ALL_GUARDS, GUARDS_AWAITING_INPUT

    readme = (_repo_root() / "README.md").read_text()
    marked = set(re.findall(r"`([a-z_]+)`†", readme))
    assert marked == set(GUARDS_AWAITING_INPUT), (
        "README daggers disagree with guards.GUARDS_AWAITING_INPUT: "
        f"only in README={sorted(marked - set(GUARDS_AWAITING_INPUT))}, "
        f"only in registry={sorted(set(GUARDS_AWAITING_INPUT) - marked)}"
    )
    listed = set(re.findall(r"`([a-z_]+)`", readme))
    assert set(ALL_GUARDS) <= listed, (
        "README no longer lists every guard: " + ", ".join(sorted(set(ALL_GUARDS) - listed))
    )


def test_citation_file_stays_in_sync_with_the_package():
    """A citation record that names the wrong version cites something else.

    Every other hand-maintained claim in this repository has drifted at least
    once, which is why status.py derives them. CITATION.cff cannot be derived --
    it is the authoritative statement of authorship -- so it is checked instead.
    """
    import yaml

    import motifmultiverse

    root = _repo_root()
    path = root / "CITATION.cff"
    if not path.exists():
        pytest.skip("CITATION.cff is not present in this installation")
    cff = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert cff["cff-version"] == "1.2.0"
    assert cff["version"] == motifmultiverse.__version__, (
        f"CITATION.cff cites {cff['version']}, package is {motifmultiverse.__version__}"
    )
    assert cff["type"] == "software"
    assert cff["authors"] and all(a.get("orcid") for a in cff["authors"]), (
        "every author needs an ORCID; a name alone does not disambiguate a person"
    )
    # No invented identifier. A placeholder DOI resolves to nothing while looking
    # like a citable record, which is worse than having none.
    assert "doi" not in cff, "a DOI appeared; check it resolves before pinning it here"


def test_every_file_that_states_a_version_states_the_same_one():
    """Four files name a version and only one of them was ever checked.

    `CITATION.cff` is held to `__version__` by the test above. `pyproject.toml`,
    the README badge and the CHANGELOG heading were not held to anything, and
    that was measured rather than assumed: bumping `pyproject.toml` to 0.2.0 and
    leaving `_version.py` alone left the whole suite green, and so did moving the
    badge and the CHANGELOG heading to a version that exists nowhere else.

    The pyproject drift is the one that matters. `provenance.__init__` writes
    `software.motifmultiverse` from `__version__`, so a release built from a
    bumped `pyproject.toml` installs as one version and stamps every artifact it
    produces with another. A provenance record that names software that was never
    released is not a lesser bug than a wrong number -- it is the record this
    package exists to make trustworthy, saying something untrue about itself, and
    nobody downstream can tell.

    `_version.py` is the single source. Everything else restates it and is
    therefore checked against it.
    """
    import re
    import tomllib

    import motifmultiverse

    root = _repo_root()
    version = motifmultiverse.__version__

    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == version, (
        f"pyproject.toml declares {pyproject['project']['version']}, package is {version}; "
        "the wheel would install under one version and stamp provenance with the other"
    )

    # shields.io escapes a literal dash as `--`, so undo that before comparing
    # rather than forbidding a version that contains one.
    badge = re.search(r"version-([^)\s]+?)-blue", (root / "README.md").read_text(encoding="utf-8"))
    assert badge, "the README lost its version badge"
    assert badge.group(1).replace("--", "-") == version, (
        f"README badge shows {badge.group(1)}, package is {version}"
    )

    changelog = re.search(r"^## \[([^\]]+)\]", (root / "CHANGELOG.md").read_text(encoding="utf-8"),
                          flags=re.MULTILINE)
    assert changelog, "the CHANGELOG lost its version heading"
    assert changelog.group(1) == version, (
        f"CHANGELOG's newest section is {changelog.group(1)}, package is {version}"
    )


def test_citation_license_matches_the_license_file():
    import yaml

    root = _repo_root()
    path = root / "CITATION.cff"
    if not path.exists():
        pytest.skip("CITATION.cff is not present in this installation")
    declared = yaml.safe_load(path.read_text(encoding="utf-8"))["license"]
    first_line = (root / "LICENSE").read_text(encoding="utf-8").splitlines()[0].strip()
    assert declared.split("-")[0].lower() in first_line.lower(), (
        f"CITATION.cff says {declared}, LICENSE begins {first_line!r}"
    )


#: Shapes that must never reach a published file. Kept as data, and as a test,
#: because a scan typed fresh at each release is a scan that misses a category:
#: the pattern used before this test existed looked for `kant/envs` and sailed
#: past `envs/chrombpnet_local` in a shipped README, and past a real absolute
#: path with a username in it that a fixture had baked in verbatim.
_LEAK_PATTERNS = (
    (r"/data1/|/data/scratch/", "an absolute path from the authoring machine"),
    (r"/afs/", "an AFS path"),
    (r"\bcsail\b|\.mit\.edu\b", "an institutional hostname"),
    (r"miniforge3|/envs/[a-z_]+|conda/envs", "a local environment name"),
    (r"claude-\d{4,}|scratchpad/|pytest-of-", "an agent or test scratch path"),
    (r"GPU-[0-9a-f]{8}-", "a GPU UUID"),
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", "an e-mail address"),
)


def test_no_published_file_carries_a_local_path_or_identity():
    """The repository is published; anything tracked is published with it.

    Both leaks this test was written after were introduced by commits whose own
    subject lines were about correctness, not about paths -- a fixture pasted a
    real provenance record verbatim (`command` is unredacted by policy, so a real
    record carries real paths), and a README named the environment a comparison
    had been run in. Neither author was careless; both were writing about
    something else. A scan nobody has to remember to run is the only kind that
    holds.
    """
    import re
    import subprocess

    root = _repo_root()
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=False,
    )
    if tracked.returncode != 0:
        pytest.skip("not a git checkout; nothing is tracked here to scan")
    names = [n for n in tracked.stdout.split("\0") if n]
    assert names, "git ls-files returned nothing"

    offences: list[str] = []
    for name in names:
        if name == "LICENSE":              # the licence carries the author's name by design
            continue
        if name == "tests/test_status.py":  # this file defines the patterns it scans for
            continue
        path = root / name
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue                        # binary or unreadable: nothing to leak in text
        for pattern, what in _LEAK_PATTERNS:
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                offences.append(f"{name}:{line}: {what} -- {match.group(0)!r}")
    assert not offences, "published files carry local identity:\n  " + "\n  ".join(offences[:20])


def test_the_readme_quickstart_actually_runs(tmp_path):
    """Not "parses" -- runs, from a checkout, with nothing downloaded.

    `test_every_readme_quickstart_command_parses` checks the parser because the
    older examples named files a reader had to supply. The quickstart generates
    its own inputs, so the stronger claim is available and is the one worth
    making: a documented sequence that has never been executed is a claim about
    software nobody ran.

    The matrices are seeded noise. This asserts the pipeline completes and writes
    the artifacts the README says it writes -- not that any number is meaningful.
    """
    import subprocess
    import sys

    root = _repo_root()
    inputs = tmp_path / "quickstart_inputs"
    subprocess.run([sys.executable, str(root / "examples/quickstart/make_inputs.py"),
                    str(inputs)], check=True, capture_output=True)
    assert (inputs / "project.json").exists()

    def cli(*args):
        result = subprocess.run(
            [sys.executable, "-m", "motifmultiverse.cli", *args],
            cwd=inputs, capture_output=True, text=True)
        assert result.returncode == 0, f"{args[0]} exited {result.returncode}: {result.stderr}"
        return result

    cli("ingest", "project.json", "--out", "registry")
    cli("align", "registry", "--out", "evidence")
    cli("annotate", "evidence", "--registry", "registry", "--out", "evidence")
    cli("adjudicate", "evidence", "--registry", "registry", "--out", "decisions")
    cli("compile", "registry", "--decisions", "decisions/merge_decisions.json",
        "--out", "lexicons")

    # The four files the README tells a reader to open.
    for relative in ("decisions/review.yaml", "evidence/alignment_null_summary.tsv",
                     "lexicons/core.manifest.json", "lexicons/provenance.json"):
        assert (inputs / relative).exists(), relative
    for stage in ("registry", "evidence", "decisions", "lexicons"):
        status = json.loads((inputs / stage / "run_status.json").read_text())
        assert status["status"] == "SUCCESS", (stage, status)
