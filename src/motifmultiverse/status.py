"""Machine-readable implementation status (T-18).

Every hand-maintained status claim in this repository has been wrong at least
once: the README said "163 passed, 1 skipped" while the suite was at 169, and
called `validate` a skeleton after it was implemented. A count typed into prose
is a claim nobody re-derives, and the failure mode is silent -- the number stays
plausible while it drifts.

So the claims are derived here instead, from the code and from an actual test
run, and the prose renders from the result:

* **module status** is read from the CLI dispatch table -- a module is a skeleton
  when its subcommand raises `NotImplementedError`, which is a fact about the
  code rather than an opinion about it;
* **optional backend status** is `VERIFIED` only when a capability probe *ran
  here and did the thing the backend is for*, and `UNVERIFIED` otherwise. There
  is no third, comfortable value. It used to be `VERIFIED` on a bare
  `import_module`, which this package's own code contradicts: `compile`
  separates `BackendMissing` from `BackendIncompatible` precisely because a
  backend that imports may still be unable to read a lexicon back -- finemo 0.40
  renamed an argument and every importable installation of it raised `TypeError`
  from inside `compile_lexicons`. An import is not a capability, and a status
  document whose green means "imports" is the guard-that-cannot-fail shape this
  repository is organised against;
* **test counts** are three separate numbers -- passed, skipped, failed -- and
  are absent (`NOT_RUN`) rather than zero when no run was supplied. A skipped
  test is never added to the passed count anywhere in this file, because that
  addition is exactly how a suite comes to claim more than it verified.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from motifmultiverse import __version__

__all__ = [
    "SCHEMA_VERSION", "MODULES", "OPTIONAL_BACKENDS", "BackendProbe",
    "module_status", "backend_status", "build_status", "counts_from_junit",
    "render_markdown", "main",
]

SCHEMA_VERSION = "1"

#: Every analysis module, in pipeline order, with the CLI subcommand that drives
#: it. Order is the declared reference path (docs/ARCHITECTURE.md), not
#: alphabetical, so the rendered table reads as the path a user would run.
MODULES = (
    "ingest", "align", "annotate", "adjudicate", "compile",
    "validate", "infer", "report", "interpret",
)

@dataclass(frozen=True)
class BackendProbe:
    """An executable check that a backend can do what this package needs of it.

    `capability` is the claim a `VERIFIED` makes, in a reader's terms -- it is
    rendered into the status document, so nobody has to open the source to learn
    what green covers. `run` performs it and returns the evidence; raising is how
    it fails, and every exception type is a failure, including the ones nobody
    predicted. A probe that swallowed an unexpected exception would be reporting
    a backend as working on the strength of not having crashed loudly enough.
    """

    module: str
    capability: str
    run: Callable[[], str]


def _finemo_roundtrip_probe() -> str:
    from motifmultiverse.compile import probe_backend

    return probe_backend()


#: Optional backends and what would prove each one works. `None` marks a backend
#: whose check is not implemented -- reported as `UNVERIFIED` with that reason,
#: never quietly omitted, since an unlisted backend reads as one that passed.
#:
#: `tomtom` and `homer` stay `None` deliberately. Both adapters read *precomputed*
#: output rather than invoking the tool (see `annotate/README.md`), so there is no
#: installed binary here whose capability could be probed, and inventing a check
#: that passes by reading a fixture would be a `VERIFIED` that verifies nothing --
#: the same defect as the import check this file has just stopped making.
OPTIONAL_BACKENDS: dict[str, BackendProbe | None] = {
    "finemo": BackendProbe(
        module="finemo",
        capability=("a lexicon compiled by this package is read back by the real "
                    "hit-caller loader, in the order its manifest records"),
        run=_finemo_roundtrip_probe,
    ),
    "tomtom": None,
    "homer": None,
}


def module_status(name: str) -> dict[str, str]:
    """Derive one module's status from what its CLI subcommand actually does.

    Read off the parser rather than a hand-kept list: a module counts as
    implemented when its subcommand dispatches to a real runner, and as a
    skeleton when the dispatcher is the `NotImplementedError` stub. Adding a
    module to a list is easy to do wrongly and impossible to notice; deriving it
    means the status changes exactly when the code does.
    """
    from motifmultiverse.cli import build_parser

    parser = build_parser()
    subparsers = {
        choice: sub
        for action in parser._actions if getattr(action, "choices", None)
        for choice, sub in action.choices.items()
    }
    if name not in subparsers:
        return {"status": "ABSENT", "detail": "no CLI subcommand is registered"}
    func = subparsers[name].get_default("func")
    qualname = getattr(func, "__qualname__", "")
    if qualname.startswith("build_parser.<locals>"):
        # The lambda that calls `_run(module, ns)`, i.e. the skeleton dispatcher.
        return {"status": "SKELETON",
                "detail": f"the subcommand raises NotImplementedError; "
                          f"see src/motifmultiverse/{name}/README.md"}
    return {"status": "IMPLEMENTED", "detail": f"cli.{qualname}"}


def backend_status(name: str, probe: BackendProbe | None) -> dict[str, str]:
    """`VERIFIED` only if a capability probe ran here and passed. Everything else is UNVERIFIED.

    Deliberately two-valued. "Probably fine", "verified previously" and "not
    checked" are all the same thing to a reader deciding whether to trust an
    artifact, and collapsing them into one honest value is the point.

    Three distinct situations now land on `UNVERIFIED`, and each says which it
    is: no probe exists for this backend; the backend is not installed; the
    backend is installed and the probe failed. The third is the one that used to
    be reported as `VERIFIED`, and it is the state an incompatible backend
    release puts a machine in -- installed, importable, and unable to read a
    single lexicon back.
    """
    if probe is None:
        return {"status": "UNVERIFIED",
                "detail": "no executable check exists for this backend in this release"}
    try:
        import_module(probe.module)
    except ImportError as exc:
        return {"status": "UNVERIFIED",
                "detail": f"{probe.module} is not installed ({exc}); "
                          f"UNVERIFIED: {probe.capability}"}
    try:
        evidence = probe.run()
    except BaseException as exc:  # noqa: BLE001 - see BackendProbe: any failure is a failure
        return {"status": "UNVERIFIED",
                "detail": (f"{probe.module} imports but the capability check failed "
                           f"({type(exc).__name__}: {exc}). Importable is not usable; "
                           f"UNVERIFIED: {probe.capability}")}
    return {"status": "VERIFIED", "detail": f"{probe.capability} -- {evidence}"}


def build_status(*, passed: int | None = None, skipped: int | None = None,
                 failed: int | None = None, test_command: str | None = None) -> dict[str, Any]:
    """Assemble the status document.

    Test counts stay three separate fields. If no run was supplied they are
    absent under `"status": "NOT_RUN"` rather than defaulted to zero -- zero
    failures that nobody measured is not the same claim as zero failures that
    somebody did.
    """
    if passed is None or skipped is None or failed is None:
        tests: dict[str, Any] = {
            "status": "NOT_RUN",
            "detail": "no test run was supplied to the generator; counts are absent, not zero",
        }
    else:
        tests = {
            "status": "RUN",
            "passed": passed,
            "skipped": skipped,
            "failed": failed,
            "command": test_command or "pytest",
            # Spelled out because the whole point is that it is not the sum.
            "note": ("skipped tests are UNVERIFIED and are not included in `passed`; "
                     f"{passed} verified, {skipped} unverified, {failed} failing"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "motifmultiverse.status",
        "package_version": __version__,
        "modules": {name: module_status(name) for name in MODULES},
        "optional_backends": {
            name: backend_status(name, probe)
            for name, probe in sorted(OPTIONAL_BACKENDS.items())
        },
        "tests": tests,
    }


#: Markers delimiting the rendered block in README.md. The prose around them is
#: written by hand; everything between them is generated and must not be edited.
BEGIN_MARKER = "<!-- BEGIN GENERATED STATUS -->"
END_MARKER = "<!-- END GENERATED STATUS -->"


def render_markdown(status: dict[str, Any]) -> str:
    """Render the README block: the module table, and nothing environment-dependent.

    Only module status goes into committed prose. Backend verification and test
    counts are properties of *a particular run on a particular machine* -- a
    README asserting `finemo: UNVERIFIED` would be false on a machine that has
    it, and a README asserting a pass count is the stale-number problem this
    module exists to end. Those live in `implementation_status.json`, which is
    generated per run, and the README points at it rather than restating it.
    """
    lines = [
        BEGIN_MARKER,
        "<!-- generated by `python -m motifmultiverse.status --render-readme README.md`; "
        f"schema {status['schema_version']}. Do not edit by hand. -->",
        "",
        "| module | status |",
        "|---|---|",
    ]
    for name, entry in status["modules"].items():
        lines.append(f"| `{name}` | {entry['status']} |")
    lines += [
        "",
        "Optional-backend verification and the three test counts (passed / skipped /",
        "failed, never summed) are per-run facts, not repository facts: see",
        "`implementation_status.json`, which CI regenerates and uploads on every run.",
        END_MARKER,
    ]
    return "\n".join(lines)


def _splice(text: str, block: str) -> str:
    """Replace the generated block in `text`, refusing if the markers are absent."""
    start = text.find(BEGIN_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"the generated-status markers {BEGIN_MARKER} / {END_MARKER} are missing or "
            "out of order; refusing to guess where the block belongs"
        )
    return text[:start] + block + text[end + len(END_MARKER):]


def counts_from_junit(path: str | Path) -> dict[str, int]:
    """Read the three counts out of a pytest JUnit XML report.

    `passed` is derived by subtraction -- `tests - failures - errors - skipped` --
    so a skip can never land in it: the skip count is removed before the pass
    count exists, rather than being trusted not to have been added. Errors count
    as failures, because a test that could not run did not verify anything.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(Path(path)).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        raise ValueError(f"{path} contains no <testsuite> element")
    total = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0)) for s in suites)
    errors = sum(int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    passed = total - failures - errors - skipped
    if passed < 0:
        raise ValueError(
            f"{path} reports {total} tests but {failures + errors + skipped} "
            "failed/errored/skipped; refusing to report a negative pass count"
        )
    return {"passed": passed, "skipped": skipped, "failed": failures + errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m motifmultiverse.status",
        description=(
            "Generate implementation_status.json from the code and, optionally, from "
            "a test run. Pass --passed/--skipped/--failed from an actual pytest run; "
            "omitting them records NOT_RUN rather than zeros."
        ),
    )
    parser.add_argument("--out", default="implementation_status.json")
    parser.add_argument("--passed", type=int, default=None)
    parser.add_argument("--skipped", type=int, default=None)
    parser.add_argument("--failed", type=int, default=None)
    parser.add_argument("--test-command", default=None)
    parser.add_argument("--junit-xml", default=None,
                        help="a pytest JUnit XML report; the three counts are read from it "
                             "instead of being passed in by hand")
    parser.add_argument("--render-readme", default=None,
                        help="also splice the rendered block into this README")
    ns = parser.parse_args(argv)

    counts = {"passed": ns.passed, "skipped": ns.skipped, "failed": ns.failed}
    if ns.junit_xml:
        counts = counts_from_junit(ns.junit_xml)
    status = build_status(**counts, test_command=ns.test_command)
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {out}")
    if ns.render_readme:
        readme = Path(ns.render_readme)
        readme.write_text(_splice(readme.read_text(encoding="utf-8"),
                                  render_markdown(status)), encoding="utf-8")
        print(f"rendered: {readme}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
