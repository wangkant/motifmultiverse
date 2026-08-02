"""`provenance: derived` must be resolvable, not self-reported.

The v2 criteria format made a threshold say whether it was `declared` (chosen by a
maintainer) or `derived` (follows from something). An adversarial review then wrote

    - field: ppm_similarity
      operator: ge
      value: 0.7314159
      provenance: derived
      basis: "It follows from the structure of the problem."

and the loader accepted it, inside a criterion it then reported as FROZEN. Every
guarantee the label carried was the author's own say-so. `declared` is honest by
construction -- it claims nothing a reader could check -- but `derived` is a claim
about where a number came from, and an unchecked claim about provenance is worth
less than no claim, because it *reads* as evidence.

So a derived threshold must now name its source in a closed, machine-resolvable
vocabulary, and the loader RECOMPUTES the value from that source and refuses the
file when they differ. The vocabulary has one member, `evidence_domain`, because
one member is what the shipped criteria actually need and a speculative second
would be another unexercised claim.

WHAT THIS ESTABLISHES, EXACTLY: that the value is a structural landmark of the
interval the field's own data validator holds it to -- an endpoint, or the sign
boundary of a signed measure -- rather than a number someone picked. It does NOT
establish that gating there is the right rule. `overlap_frac_source ge 1.0` is
checkably the top of [0.0, 1.0]; whether full bilateral containment means two
motifs are the same motif is an argument, and the argument is in `basis`. A value
that cannot be written as such a landmark is not derived and must be `declared`.
"""
from __future__ import annotations

import pytest
import yaml

from motifmultiverse.schema.criteria import (
    CRITERIA_SCHEMA_VERSION,
    CriteriaError,
    Predicate,
    load_criteria,
)


def _registry(tmp_path, predicates, *, schema_version=CRITERIA_SCHEMA_VERSION, status="FROZEN"):
    """A one-criterion registry file wrapping the predicates under test."""
    entry = {
        "criterion_id": "X",
        "version": "1",
        "status": status,
        "relationship": "X",
        "required_evidence": sorted({p["field"] for p in predicates}),
        "predicates": predicates,
        "insufficient_evidence_action": "deferred",
        "decision_if_matched": "refuse_merge",
    }
    if status == "FROZEN_DECLARED_HEURISTIC":
        entry["declared_rationale"] = "test fixture"
        entry["replacement_evidence"] = ["test fixture"]
    payload = {"criteria": [entry]}
    if schema_version is not False:
        payload["schema_version"] = schema_version
    path = tmp_path / "criteria.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _derived(field, value, **derived_from):
    return {
        "field": field,
        "operator": "ge",
        "value": value,
        "provenance": "derived",
        "basis": "test fixture",
        "derived_from": dict(derived_from) or None,
    }


# --------------------------------------------------------------------------- #
# The forgery the review actually wrote.
# --------------------------------------------------------------------------- #

def test_the_reviewers_forgery_no_longer_loads(tmp_path):
    """Verbatim: an invented magnitude, labelled derived, with a prose basis.

    This loaded and produced a criterion the package reported as FROZEN. It is the
    single test this whole module exists for.
    """
    path = _registry(tmp_path, [{
        "field": "ppm_similarity",
        "operator": "ge",
        "value": 0.7314159,
        "provenance": "derived",
        "basis": "It follows from the structure of the problem.",
    }])

    with pytest.raises(CriteriaError, match="derived_from"):
        load_criteria(path)


def test_naming_a_landmark_the_value_is_not_is_refused_with_both_numbers(tmp_path):
    """Naming a source is not enough; the value has to actually be that source.

    Otherwise `derived_from` degrades into a second self-report -- a longer way of
    writing the same unchecked claim.
    """
    path = _registry(tmp_path, [
        _derived("ppm_similarity", 0.7314159, evidence_domain="ppm_similarity", endpoint="max"),
    ])

    with pytest.raises(CriteriaError) as excinfo:
        load_criteria(path)
    message = str(excinfo.value)
    assert "0.7314159" in message, "the reader must be shown the value that was written"
    assert "1.0" in message, "and the value the named source actually resolves to"


def test_a_derived_threshold_may_not_borrow_another_fields_domain(tmp_path):
    """`ppm_similarity ge 1.0` justified by *overlap_frac_source*'s range is refused.

    Without this, every field with a [0, 1] domain lends its endpoints to every
    other field, and the check resolves to "is this number 0.0 or 1.0 anywhere in
    the package", which is not a provenance claim.
    """
    path = _registry(tmp_path, [
        _derived("ppm_similarity", 1.0, evidence_domain="overlap_frac_source", endpoint="max"),
    ])

    with pytest.raises(CriteriaError, match="own domain|its own"):
        load_criteria(path)


def test_the_sign_boundary_is_refused_on_a_field_that_has_no_sign(tmp_path):
    """`overlap_frac_source ge 0.0` is the bottom of its range, not a sign boundary.

    They are the same number and a different claim. `overlap_frac_source` is
    validated into [0.0, 1.0], so it never takes a negative value and "gate on the
    sign" is not a thing that can be said about it -- writing it that way would
    make a floor look like a driver/repressor distinction.
    """
    path = _registry(tmp_path, [
        _derived("overlap_frac_source", 0.0,
                 evidence_domain="overlap_frac_source", endpoint="sign_boundary"),
    ])

    with pytest.raises(CriteriaError, match="sign_boundary"):
        load_criteria(path)


def test_the_sign_boundary_is_accepted_on_a_signed_field(tmp_path):
    """The positive control: on [-1.0, 1.0] the same endpoint resolves to 0.0.

    Without this the four refusals above are also satisfied by a check that refuses
    everything, which would be a different bug with identical test output.
    """
    path = _registry(tmp_path, [
        _derived("signed_cwm_similarity", 0.0,
                 evidence_domain="signed_cwm_similarity", endpoint="sign_boundary"),
    ])

    criterion = load_criteria(path)["X"]
    assert criterion.predicates[0].value == 0.0
    assert criterion.predicates[0].provenance == "derived"


def test_an_unknown_evidence_field_is_refused_rather_than_defaulted(tmp_path):
    path = _registry(tmp_path, [
        _derived("invented_score", 1.0, evidence_domain="invented_score", endpoint="max"),
    ])

    with pytest.raises(CriteriaError, match="invented_score"):
        load_criteria(path)


def test_an_unknown_endpoint_is_refused(tmp_path):
    path = _registry(tmp_path, [
        _derived("ppm_similarity", 1.0, evidence_domain="ppm_similarity", endpoint="q90"),
    ])

    with pytest.raises(CriteriaError, match="endpoint"):
        load_criteria(path)


def test_derived_from_may_not_be_attached_to_a_declared_threshold(tmp_path):
    """A chosen number does not become derived by naming a landmark next to it."""
    path = _registry(
        tmp_path,
        [{
            "field": "ppm_similarity", "operator": "ge", "value": 0.9,
            "provenance": "declared", "basis": "test fixture",
            "derived_from": {"evidence_domain": "ppm_similarity", "endpoint": "max"},
        }],
        status="FROZEN_DECLARED_HEURISTIC",
    )

    with pytest.raises(CriteriaError, match="declared"):
        load_criteria(path)


def test_the_check_binds_to_the_predicate_not_only_to_the_file(tmp_path):
    """A Predicate built in-process is resolved too.

    `adjudicate` and the test suite construct Predicates directly, so a check that
    lived only in `load_criteria` would leave the in-process path exactly as
    unchecked as the YAML path was.
    """
    with pytest.raises(CriteriaError):
        Predicate(
            field="ppm_similarity", operator="ge", value=0.7314159,
            provenance="derived", basis="prose",
            derived_from={"evidence_domain": "ppm_similarity", "endpoint": "max"},
        )


# --------------------------------------------------------------------------- #
# The table is the one the data is actually held to.
# --------------------------------------------------------------------------- #

def test_the_domain_table_is_the_one_the_alignment_validator_enforces():
    """One table, two consumers -- a second copy would drift silently.

    If `EVIDENCE_FIELD_DOMAINS` were documentation maintained beside the real
    bounds, `derived` would again mean "matches a number in a file nobody checks".
    So `AlignmentEvidence` must validate against exactly these intervals: each
    field accepts both endpoints and refuses immediately outside them.
    """
    from motifmultiverse.align import AlignmentError, AlignmentEvidence
    from motifmultiverse.schema import EVIDENCE_FIELD_DOMAINS

    def edge(**overrides):
        fields = dict(
            source_node_id="a", target_node_id="b", orientation="+", offset=0,
            overlap_bp=10, overlap_frac_source=1.0, overlap_frac_target=1.0,
            ppm_similarity=0.9, signed_cwm_similarity=0.9,
            empirical_p_value=0.001, null_shuffles=1000, seed=7,
        )
        fields.update(overrides)
        return AlignmentEvidence(**fields)

    assert set(EVIDENCE_FIELD_DOMAINS) == {
        "overlap_frac_source", "overlap_frac_target",
        "ppm_similarity", "signed_cwm_similarity", "empirical_p_value",
    }
    for name, (low, high) in EVIDENCE_FIELD_DOMAINS.items():
        edge(**{name: low})
        edge(**{name: high})
        for outside in (low - 1e-9, high + 1e-9):
            with pytest.raises(AlignmentError, match=name):
                edge(**{name: outside})


def test_every_derived_threshold_in_the_v2_registry_resolves():
    """Whatever else changes, v2 may not reacquire a bare `derived` label.

    ABOUT v2 by name, not about the default. `provenance`/`derived_from` are a v2
    format feature -- the default registry (v1) has no predicates carrying them at
    all -- so asking this of "whatever is default" would pass vacuously the moment
    the default is a v1 file, which is exactly what it is.
    """
    from motifmultiverse.adjudicate import packaged_v2_criteria_path

    resolved = 0
    for criterion in load_criteria(packaged_v2_criteria_path()).values():
        for predicate in criterion.predicates:
            if predicate.provenance == "derived":
                assert predicate.derived_from, (
                    f"{criterion.criterion_id}.{predicate.field} claims 'derived' with "
                    "nothing a loader can resolve"
                )
                resolved += 1
    assert resolved, "no derived predicate was examined; this test proved nothing"


def test_a_v1_registry_is_not_held_to_the_new_requirement():
    """The rule the package applies to itself: a checksummed run keeps its meaning.

    v1 files carry no `provenance` at all, so there is nothing for `derived_from`
    to attach to, and a v1 run must not start failing because a later release grew
    a field. v1 is also the default, which makes this the ordinary case rather
    than the compatibility case -- but the claim is about the v1 FORMAT, so it
    names v1.
    """
    from motifmultiverse.adjudicate import packaged_v1_criteria_path

    v1 = load_criteria(packaged_v1_criteria_path())
    assert v1["TRUE_DUPLICATE"].predicates == ()
    assert all(
        p.provenance is None
        for criterion in v1.values()
        for p in criterion.predicates
    )


# --------------------------------------------------------------------------- #
# The other way a provenance claim goes wrong: not unresolvable, but resolvable
# to a different string depending on which machine resolved it.
#
# Above, `derived` was a claim the reader could not check. Below, the recorded
# input keys are checkable -- and were checkably different on two hosts that ran
# the same command over the same bytes. Same failure shape: something that reads
# as evidence about the run is actually evidence about the environment.
# --------------------------------------------------------------------------- #

def test_a_root_relative_input_key_is_recorded_in_posix_spelling_on_every_host(tmp_path):
    """`add_input(..., root=)` keys must not carry the host's path separator.

    The key was built with ``str(p.relative_to(root))``, and ``str()`` of a
    relative Path is host-flavoured, so the identical pipeline over the identical
    inputs recorded ``sub/dir/hits.tsv`` on Linux and ``sub\\dir\\hits.tsv`` on
    Windows. Measured here before the fix:
    ``tests/test_schema.py::test_add_input_root_distinguishes_same_basename``
    failed on this box for exactly that reason and passes on CI.

    That is an identity bug, not a cosmetic one. ``adjudicate_evidence`` copies
    ``provenance_record.inputs`` verbatim into the mapping that
    ``_scientific_artifact_id`` hashes into the ``ontology-decisions:`` and
    ``review:`` ids -- which are then written into every row of
    ``ontology_decisions.parquet`` and into its file metadata -- and into
    ``schema.decision_bundle_artifact_id``, whose ``merge-decisions:`` id
    ``schema`` re-derives on read and refuses on mismatch. Measured on the same
    inputs, the two spellings gave ``merge-decisions:a372f62a...`` and
    ``merge-decisions:fccd0571...``. Two collaborators would conclude their runs
    differed when the only thing that differed was the operating system.

    Asserting the POSIX spelling rather than ``os.sep`` is the point: a test that
    asked for the host's separator would pass on both machines and re-describe
    the bug as the contract.
    """
    from motifmultiverse.provenance import ProvenanceRecord, sha256_file
    from motifmultiverse.schema import decision_bundle_artifact_id

    nested = tmp_path / "sub" / "dir"
    nested.mkdir(parents=True)
    hits = nested / "hits.tsv"
    hits.write_bytes(b"region_id\thit_coefficient\nr1\t1.0\n")

    rec = ProvenanceRecord(command="mm adjudicate", subcommand="adjudicate")
    rec.add_input(hits, root=tmp_path)

    assert rec.inputs == {"sub/dir/hits.tsv": sha256_file(hits)}

    # And the keys are load-bearing, so pinning the spelling is worth doing: the
    # same digests under the two spellings are two different artifact ids.
    digest = sha256_file(hits)
    assert (
        decision_bundle_artifact_id((), {}, {"inputs": dict(rec.inputs)})
        != decision_bundle_artifact_id((), {}, {"inputs": {"sub\\dir\\hits.tsv": digest}})
    )
