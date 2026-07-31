"""Shared gates for the optional `finemo` backend.

The round trip -- compile a lexicon, hand the file to the *real* hit-caller
loader, compare the order it returns against the manifest -- is the one check
that proves what `compile` exists to guarantee. Everything else in the suite
proves only that this package can read its own output.

Two failures came out of gating it on `pytest.importorskip`:

**A skip is unverified, and unverified read as green.** With the backend absent
the round trip skipped, and the suite reported a pass count that did not include
it. That is the guard-that-cannot-fail shape the repository is organised against,
so the skip here is *opt-out*: set `MOTIFMULTIVERSE_REQUIRE_FINEMO=1` (as CI
should, alongside `pip install -e \".[dev,finemo]\"`) and a missing backend fails
the run instead of quietly shrinking it.

**Its mirror image skipped too, and nobody noticed.** The companion tests assert
the *no-backend* path -- that `--verify-roundtrip require` refuses rather than
passing when nothing can read the lexicon back -- and they skipped whenever the
backend WAS installed. So no single environment ran both halves: with finemo the
refusal path was unverified, without it the round trip was. The `no_finemo_backend`
fixture removes the environment from that question by making the import fail on
demand, so both halves run in every environment.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

#: Set to "1" where a missing backend must fail the run rather than skip it.
#: Named rather than inferred from CI-detection heuristics: a machine that
#: cannot install the backend and a machine that forgot to are different
#: situations, and only the operator knows which one this is.
REQUIRE_FINEMO_ENV = "MOTIFMULTIVERSE_REQUIRE_FINEMO"

#: The import name the backend provides. The *distribution* is also called
#: `finemo` (PyPI 0.40, 0.41; the pre-PyPI 0.30 installs from
#: github.com/kundajelab/Fi-NeMo, formerly austintwang/finemo_gpu). There is no
#: `finemo-gpu` distribution on PyPI -- `pip install finemo-gpu` fails with "no
#: matching distribution" -- which is what kept the extra uninstallable and the
#: round trip skipped.
FINEMO_IMPORT_NAME = "finemo"


def finemo_installed() -> bool:
    """Whether the backend can be imported, without importing it.

    `find_spec` rather than a `try: import`, because importing finemo pulls in
    torch: several seconds and a large allocation for a question that is answered
    by the module's presence on the path.
    """
    return importlib.util.find_spec(FINEMO_IMPORT_NAME) is not None


def require_finemo_backend() -> None:
    """Skip when the backend is absent -- or fail, if this run says it must be there.

    The skip message says what was *not* verified rather than what was skipped,
    because "skipped: needs finemo" is read as "fine" and "the round trip did not
    run; the lexicon was never read back by the tool that consumes it" is not.
    """
    if finemo_installed():
        return
    unverified = (
        f"the {FINEMO_IMPORT_NAME} backend is not installed, so the loader round trip "
        "DID NOT RUN: nothing here has read a compiled lexicon back with the tool that "
        'will consume it, and this run is not evidence that one loads. Install it with '
        'pip install -e ".[finemo]".'
    )
    if os.environ.get(REQUIRE_FINEMO_ENV) == "1":
        pytest.fail(
            f"{REQUIRE_FINEMO_ENV}=1 declares this run must verify the round trip, but "
            f"{unverified}"
        )
    pytest.skip(f"UNVERIFIED, not passed: {unverified}")


@pytest.fixture
def no_finemo_backend(monkeypatch):
    """Make the backend unimportable for one test, whether or not it is installed.

    Both the parent package and the submodule are blanked: `from finemo.data_io
    import ...` consults `sys.modules["finemo.data_io"]` first, so blanking only
    the parent leaves an already-imported submodule reachable and the test
    silently exercises the installed backend instead of the absent-backend path.
    """
    for name in (FINEMO_IMPORT_NAME, f"{FINEMO_IMPORT_NAME}.data_io"):
        monkeypatch.setitem(sys.modules, name, None)
    yield
    # monkeypatch restores sys.modules; nothing to undo by hand.
