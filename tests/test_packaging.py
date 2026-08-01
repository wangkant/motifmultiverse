"""What the distribution must contain, checked the way the code reaches it.

The failure these are written after: `report`'s default bias ledger was
`docs/bias_ledger.tsv`, resolved by walking up from `report/__init__.py`. Under
`pip install -e .` that path exists, so every test and every CI run found it.
Under `pip install motifmultiverse` it does not -- `docs/` was never package data
-- so the *default* invocation of a subcommand refused, naming a file the wheel
had never been asked to carry. A refusal that reads as the tool being careful and
is in fact a packaging defect is worse than a crash, because nobody investigates
it.

Two rules follow, and both are executable here rather than remembered:

1. **A resource the code reads is package data, and is checked through
   `importlib.resources`** -- the mechanism the code uses -- never by looking for
   a file in the source tree, where it is present whether or not it ships.
2. **`pyproject.toml`'s `package-data` list is checked against the resources that
   exist**, in both directions: an entry naming nothing is a stale declaration, a
   resource named by nothing is one wheel away from being missing.

`.github/workflows/ci.yml`'s `wheel` job is the other half of this and cannot be
replaced by these tests: it builds the wheel, installs it into a clean
environment, and runs the CLI from a directory with no repository in it. These
tests catch the declaration going wrong; that job catches the build going wrong.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import motifmultiverse

#: Every packaged resource, with the accessor the code actually calls. A resource
#: added to `pyproject.toml` and reached by a hand-built path is not covered by
#: this list, which is what `test_every_declared_package_data_entry_matches_a_file`
#: exists to notice.
PACKAGED_RESOURCES = {
    # v1 is no longer the default, but it must still SHIP: `--criteria` pointing at
    # it is the documented way to reproduce a pre-v2 run, and a wheel that dropped
    # it would make that impossible from an installed package.
    "adjudicate/criteria.v1.yaml": "motifmultiverse.adjudicate:packaged_legacy_criteria_path",
    "adjudicate/criteria.v2.yaml": "motifmultiverse.adjudicate:packaged_criteria_path",
    "report/bias_ledger.tsv": "motifmultiverse.report:packaged_bias_ledger_path",
}


def _package_root() -> Path:
    return Path(motifmultiverse.__file__).resolve().parent


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    pytest.skip("no pyproject.toml above tests/; the declaration cannot be read here")


def _declared_package_data() -> list[str]:
    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["tool"]["setuptools"]["package-data"]["motifmultiverse"]


@pytest.mark.parametrize("resource,accessor", sorted(PACKAGED_RESOURCES.items()))
def test_each_packaged_resource_resolves_through_its_own_accessor(resource, accessor):
    """The accessor, not a path spelled out here: that is the code path that broke."""
    from importlib import import_module

    module_name, _, function_name = accessor.partition(":")
    path = getattr(import_module(module_name), function_name)()
    assert path.is_file(), f"{accessor}() -> {path}, which does not exist"
    assert path.read_text(encoding="utf-8").strip(), f"{path} is empty"
    assert path.resolve().is_relative_to(_package_root()), (
        f"{accessor}() resolved to {path}, outside the package at {_package_root()}. A wheel "
        "carries the package and not the repository around it, so a resource reached from "
        "outside it is a resource an installed user does not have."
    )
    assert path.resolve() == (_package_root() / resource).resolve()


def test_every_declared_package_data_entry_matches_a_file():
    """A declaration naming nothing ships nothing, and says so to no one."""
    root = _package_root()
    for pattern in _declared_package_data():
        assert list(root.glob(pattern)), (
            f"package-data declares {pattern!r} and it matches no file under {root}"
        )


def test_every_packaged_resource_is_declared_in_package_data():
    """The direction that actually fails: a file the code reads and the wheel omits.

    Checked against the declaration rather than against the built wheel because
    the declaration is what a source checkout has. The wheel itself is exercised
    by the `wheel` job in CI, which installs it and runs `report` with its default
    ledger from a directory containing no `docs/`.
    """
    declared = _declared_package_data()
    root = _package_root()
    for resource in PACKAGED_RESOURCES:
        matched = [p for p in declared if (root / resource) in set(root.glob(p))]
        assert matched, (
            f"{resource} is read by this package and matches no package-data pattern in "
            f"pyproject.toml ({declared}); it would be absent from a wheel"
        )


def test_the_module_readmes_ship_because_the_error_messages_name_them():
    """`cli._not_implemented` tells the reader to open `<module>/README.md`.

    An instruction to read a file the distribution does not contain is worse than
    no instruction: the reader goes looking for a defect in their installation.
    """
    root = _package_root()
    for module in ("ingest", "align", "annotate", "adjudicate", "compile",
                   "validate", "infer", "interpret", "report"):
        assert (root / module / "README.md").is_file(), f"{module}/README.md is not packaged"


def test_the_bias_ledger_exists_exactly_once_in_the_repository():
    """One ledger, so a copy cannot drift from the copy the renderer reads.

    It moved out of `docs/` into the package; leaving a copy behind for readers
    would recreate the two-sources problem that `docs/BIAS_LEDGER.md` already
    resolves by declaring the TSV authoritative.
    """
    root = _repo_root()
    found = sorted(p.relative_to(root) for p in root.rglob("bias_ledger.tsv")
                   if ".git" not in p.parts and "build" not in p.parts)
    assert [str(p) for p in found] == ["src/motifmultiverse/report/bias_ledger.tsv"], found
