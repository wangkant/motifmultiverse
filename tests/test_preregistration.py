"""The preregistration document and the criterion it freezes may not drift apart.

`FP-13` requires a merge/split rule's parameters to be written down and
checksummed before the result is seen, and says plainly that the package has
"no preregistration store". `docs/MERGE_CRITERION_PREREGISTRATION.md` is that
store, and this module is what makes it a store rather than a note: the checksum
in the document is compared against the file on every run, so editing the
criterion without re-registering it fails the suite instead of passing quietly.

That is the whole mechanism. It cannot tell whether the recorded checksum
predates the evidence -- nothing in a repository can, since a timestamp is as
editable as the file -- so it enforces the part that is enforceable and the
document states the part that is not.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from motifmultiverse.adjudicate import packaged_criteria_path
from motifmultiverse.schema.criteria import load_criteria

PREREGISTRATION = Path(__file__).resolve().parents[1] / "docs" / "MERGE_CRITERION_PREREGISTRATION.md"


def _document() -> str:
    if not PREREGISTRATION.exists():  # pragma: no cover - guarded by its own test
        pytest.fail(f"the preregistration document is missing: {PREREGISTRATION}")
    return PREREGISTRATION.read_text()


def test_the_registered_checksum_is_the_shipped_criteria_file():
    """Change the criterion and this fails until the registration is redone.

    A preregistration nobody re-checks is a comment. This is the check.
    """
    document = _document()
    registered = re.search(r"\*\*sha256\*\*\s*\|\s*`([0-9a-f]{64})`", document)
    assert registered, "the preregistration document states no sha256"

    actual = hashlib.sha256(packaged_criteria_path().read_bytes()).hexdigest()
    assert registered.group(1) == actual, (
        "criteria.v2.yaml no longer matches the checksum recorded in "
        f"{PREREGISTRATION.name}. Either the edit was unintended, or the "
        "criterion has genuinely changed and must be RE-registered -- new "
        "checksum, and predictions restated before the next validation run. "
        "FP-13 does not permit quietly re-pointing an existing registration at a "
        "changed rule."
    )


def test_the_document_registers_the_two_declared_magnitudes_that_are_in_force():
    """The prose must name the values the loader actually applies."""
    document = _document()
    criterion = load_criteria(packaged_criteria_path())["TRUE_DUPLICATE"]

    declared = {p.field: p.value for p in criterion.predicates if p.provenance == "declared"}
    assert declared == {"ppm_similarity": 0.90, "overlap_bp": 8}, (
        "the declared magnitudes changed; re-register before running anything"
    )
    for field, value in declared.items():
        assert f"`{field} ge {value}`" in document or f"{field} ge {value}" in document, (
            f"{field} ge {value} is in force but is not named in the preregistration"
        )


def test_the_document_states_falsifiers_and_the_limit_on_what_they_test():
    """A preregistration with no falsifier is a press release.

    And one that lets a transfer result be read as a correctness result is worse
    than none, because Section 7 shows the only downstream test that could carry
    a correctness claim has no power. Both must stay written down.
    """
    document = _document()

    assert re.search(r"^\|\s*\*\*F1\*\*", document, re.M), "no falsifier table"
    falsifiers = re.findall(r"\*\*F(\d)\*\*", document)
    assert len(set(falsifiers)) >= 5, (
        f"only {len(set(falsifiers))} distinct falsifiers are named; the "
        "criterion is being presented as harder to refute than it is"
    )
    assert "cannot establish" in document and "correctness" in document.lower(), (
        "the document must keep saying that GM12878 tests transfer and not "
        "correctness"
    )


def test_the_document_keeps_the_naive_baseline_finding():
    """The criterion's contribution on this registry is refusal, and a reader
    deciding whether to trust it has to be told so.

    Zero pairs are merged by the criterion and not by plain TomTom at q<0.05;
    674 go the other way. Deleting five random motifs of comparable mass is
    statistically indistinguishable from its own merge on reconstruction. If
    either finding is cut from the document, this fails.
    """
    document = _document()

    assert "674" in document, "the refusal count is the criterion's contribution"
    assert re.search(r"strict subset|subset of the naive", document), (
        "the document must say the merges are a subset of the naive baseline's"
    )
    assert "2.41e-4" in document or "2.41e-04" in document, (
        "the random-deletion control must stay in the document"
    )
    assert "necessary condition" in document and "not a sufficient one" in document, (
        "'no detectable reconstruction loss' must not be allowed to read as "
        "evidence that a merge was correct"
    )
