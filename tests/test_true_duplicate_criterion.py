"""The shipped TRUE_DUPLICATE v2 criterion, exercised on real registered pairs.

Every number in this file is a value that ``align`` actually produced on the K562
13-cluster run (thirteen TF-MoDISco outputs, one per Leiden cluster of the ALLPEAKS
universe, same ChromBPNet model and counts-head readout), read out of
``casestudy/evidence/alignment_edges.parquet``. Nothing here is a hand-tuned
fixture designed to land on the right side of a threshold: the pairs were selected
by asking the real table which pairs the criterion collapses, which it refuses, and
which it cannot decide, and then transcribing those rows.

That matters because the criterion is a ``FROZEN_DECLARED_HEURISTIC``. Two of its
magnitudes were chosen by a maintainer, and a test suite that only ever showed them
synthetic inputs would be checking arithmetic rather than checking that the rule
does the job it was frozen to do.

The three behaviours pinned here are the three a criterion can have:

* COLLAPSE a real duplicate -- the five-way CTCF component.
* REFUSE a real non-duplicate -- a pos/neg pair that is 0.977 similar on UNSIGNED
  ppm and sign-flipped on the CWM, i.e. a driver against a repressor.
* DEFER when the evidence is not there -- an uncalibrated edge, where "no null was
  computed" must not be read as "this pair failed its null".
"""
from __future__ import annotations

from motifmultiverse.adjudicate import adjudicate_component, packaged_criteria_path
from motifmultiverse.align import AlignmentEvidence
from motifmultiverse.schema import Decision
from motifmultiverse.schema.criteria import CriterionStatus, load_criteria

# `at_alignment_null_floor` is imported inside the two tests that need it rather
# than at module scope, deliberately: a module-level import of a symbol this change
# introduces turns every test in the file into one collection error, and a
# collection error is a much weaker red than each test failing for its own reason.

#: How the case study aligned: `motifmultiverse align registry --null-shuffles 1000
#: --seed 20260731`. The floor of that null is 1/(1000+1) = 0.000999..., and every
#: `empirical_p_value` below is the literal value the run recorded.
NULL_SHUFFLES = 1000
SEED = 20260731
NULL_FLOOR_P = 0.000999


def test_the_shipped_duplicate_criterion_is_an_evaluable_declared_heuristic():
    """The precondition for everything else in this file, asserted once and named.

    v1 shipped this criterion as CRITERION_NOT_YET_DEFINED, which always DEFERS, so
    none of the behaviours below were reachable at all.
    """
    criterion = load_criteria(packaged_criteria_path())["TRUE_DUPLICATE"]
    assert criterion.status is CriterionStatus.FROZEN_DECLARED_HEURISTIC
    assert criterion.version == "2"
    assert criterion.decision_if_matched is Decision.COLLAPSE


def _shipped():
    return load_criteria(packaged_criteria_path())


def _metadata(**nodes):
    """Registry metadata exactly as ``adjudicate._read_node_metadata`` yields it.

    `variant_assignment_source` is "NA" because `ingest` assigned no variant
    identity to any node on this run -- the `variant_id` values are its own
    per-node counter, not a claim (`schema.variant_claim_is_assigned`).
    """
    return {
        node_id: {
            "family_id": "NA",
            "variant_id": variant_id,
            "variant_assignment_source": "NA",
            "motif_completeness": completeness,
            "seqlet_count": seqlets,
            "core_ic": core_ic,
        }
        for node_id, (variant_id, completeness, seqlets, core_ic) in nodes.items()
    }


# --------------------------------------------------------------------------- #
# 1. It collapses a real duplicate.
# --------------------------------------------------------------------------- #

#: One arm of the five-way CTCF component the shipped criterion collapses on the
#: real run: cl0/pattern_3 against the elected medoid cl6/pattern_2. Both nodes'
#: best TomTom match is CTCF_MA0139.1 (q = 2.02e-10 and 4.13e-09 respectively,
#: casestudy/evidence/annotation_candidates.parquet). The geometry is transcribed
#: from casestudy/evidence/alignment_edges.parquet.
CTCF_SOURCE = "cbp_counts_cl0::pos_patterns.pattern_3"
CTCF_TARGET = "cbp_counts_cl6::pos_patterns.pattern_2"


def _ctcf_edge(**overrides) -> AlignmentEvidence:
    fields = dict(
        source_node_id=CTCF_SOURCE,
        target_node_id=CTCF_TARGET,
        orientation="+",
        offset=0,
        overlap_bp=13,
        overlap_frac_source=1.0,
        overlap_frac_target=1.0,
        ppm_similarity=0.99541,
        signed_cwm_similarity=0.99402,
        empirical_p_value=NULL_FLOOR_P,
        null_shuffles=NULL_SHUFFLES,
        seed=SEED,
    )
    fields.update(overrides)
    return AlignmentEvidence(**fields)


def _ctcf_metadata():
    return _metadata(**{
        CTCF_SOURCE: ("CBPK562_UNASSIGNED_05", 0.26, 145, 10.44875497165125),
        CTCF_TARGET: ("CBPK562_UNASSIGNED_99", 0.26, 169, 9.636727679090463),
    })


def test_it_collapses_a_real_duplicate():
    """Two CTCF detectors found independently in two Leiden clusters collapse.

    This is the package's core promise, and until TRUE_DUPLICATE was frozen it was
    undelivered: v1 was CRITERION_NOT_YET_DEFINED, so this pair -- the same motif,
    same model, same readout, discovered twice -- could only ever be DEFERRED, and
    the compiled lexicon carried both copies.
    """
    decision = adjudicate_component(
        [CTCF_SOURCE, CTCF_TARGET], [_ctcf_edge()], [], [],
        _shipped(), "test", node_metadata=_ctcf_metadata(),
    )

    assert decision.relationship == "TRUE_DUPLICATE"
    assert decision.decision is Decision.COLLAPSE
    # The survivor is an OBSERVED member, never a synthesised average.
    assert decision.representative_node_id in {CTCF_SOURCE, CTCF_TARGET}
    assert decision.criterion_id == "TRUE_DUPLICATE"
    assert decision.criterion_version == "2"


def test_the_collapse_carries_the_declared_magnitudes_into_the_decision_record():
    """A reader of ontology_decisions.parquet must be told, without opening the
    registry YAML, that two of the numbers behind this collapse were chosen.
    """
    decision = adjudicate_component(
        [CTCF_SOURCE, CTCF_TARGET], [_ctcf_edge()], [], [],
        _shipped(), "test", node_metadata=_ctcf_metadata(),
    )

    assert "DECLARED-NOT-DERIVED" in decision.rationale
    assert "ppm_similarity ge 0.9" in decision.rationale
    assert "overlap_bp ge 8" in decision.rationale
    # And it names its own way out, so the caveat is actionable rather than decorative.
    assert "Replacement evidence required" in decision.rationale


def test_every_predicate_is_load_bearing_on_the_real_duplicate():
    """Break each gate in turn, using the same real pair, and the collapse stops.

    A criterion whose predicates are individually inert is a criterion that is not
    really deciding anything. Each perturbation below is a value the real registry
    contains somewhere: 7 bp cores, sign-flipped CWMs, and unilateral overlap are
    all common on this run.
    """
    # Positive control first: without it, a criterion that collapses NOTHING would
    # satisfy every `is not COLLAPSE` assertion below and this test would pass
    # vacuously -- which is exactly what it looks like on the pre-v2 package.
    assert adjudicate_component(
        [CTCF_SOURCE, CTCF_TARGET], [_ctcf_edge()], [], [],
        _shipped(), "test", node_metadata=_ctcf_metadata(),
    ).decision is Decision.COLLAPSE

    breaks = {
        # DECLARED. 7 bp is the modal registered overlap; here it drops the pair.
        "overlap_bp": _ctcf_edge(overlap_bp=7),
        # DERIVED, sign boundary: a repressor is not the same motif as a driver.
        "signed_cwm_similarity": _ctcf_edge(signed_cwm_similarity=-0.99402),
        # DERIVED, instrument resolution: one shuffle matched this alignment.
        "at_alignment_null_floor": _ctcf_edge(empirical_p_value=0.001998),
    }
    for field, edge in breaks.items():
        decision = adjudicate_component(
            [CTCF_SOURCE, CTCF_TARGET], [edge], [], [],
            _shipped(), "test", node_metadata=_ctcf_metadata(),
        )
        assert decision.decision is not Decision.COLLAPSE, (
            f"breaking {field} left the collapse standing, so that predicate does "
            "no work and should not be in a criterion that claims it does"
        )

    # Unilateral overlap is not a near-miss duplicate, it is a different
    # relationship, and it must be routed to the criterion that owns it.
    fragment = adjudicate_component(
        [CTCF_SOURCE, CTCF_TARGET], [_ctcf_edge(overlap_frac_source=0.777778)], [], [],
        _shipped(), "test", node_metadata=_ctcf_metadata(),
    )
    assert fragment.relationship == "FRAGMENT_MATCH"
    assert fragment.decision is Decision.DEFERRED


# --------------------------------------------------------------------------- #
# 2. It refuses a real non-duplicate.
# --------------------------------------------------------------------------- #

#: The hardest real refusal on the run, and the reason the sign gate exists. These
#: two are 0.977486 similar on the UNSIGNED ppm that `align` registers on, sit at
#: the null floor, and cover each other completely (both overlap fracs 1.0) -- a
#: similarity-only rule collapses them. Their signed CWM similarity is -0.976132:
#: one is a POSITIVE-metacluster pattern (a driver) and the other a NEGATIVE one
#: (a repressor). Row 222 of casestudy/evidence/alignment_edges.parquet; it is the
#: only sign-flipped pair among the 107 bilateral, at-floor edges.
FLIPPED_POS = "cbp_counts_cl0::pos_patterns.pattern_11"
FLIPPED_NEG = "cbp_counts_cl7::neg_patterns.pattern_1"

FLIPPED_EDGE = AlignmentEvidence(
    source_node_id=FLIPPED_POS,
    target_node_id=FLIPPED_NEG,
    orientation="-",
    offset=0,
    overlap_bp=6,
    overlap_frac_source=1.0,
    overlap_frac_target=1.0,
    ppm_similarity=0.977486,
    signed_cwm_similarity=-0.976132,
    empirical_p_value=NULL_FLOOR_P,
    null_shuffles=NULL_SHUFFLES,
    seed=SEED,
)


def _flipped_metadata():
    return _metadata(**{
        FLIPPED_POS: ("CBPK562_UNASSIGNED_03", 0.12, 21, 11.447609144704122),
        FLIPPED_NEG: ("CBPK562_UNASSIGNED_121", 0.12, 25, 9.692155962636622),
    })


def test_it_refuses_a_real_non_duplicate():
    """A driver and a repressor, 0.977 similar on unsigned ppm, are not collapsed.

    The refusal is a REFUSAL, not a deferral: the evidence is complete and a
    predicate did not hold. v1 could not express that distinction -- it deferred
    everything -- so "we looked and said no" and "we could not look" were the same
    record.
    """
    decision = adjudicate_component(
        [FLIPPED_POS, FLIPPED_NEG], [FLIPPED_EDGE], [], [],
        _shipped(), "test", node_metadata=_flipped_metadata(),
    )

    assert decision.relationship == "TRUE_DUPLICATE"
    assert decision.decision is Decision.REFUSE_MERGE
    # A refused cluster has no surviving member by definition.
    assert decision.representative_node_id is None
    # The caveat rides on the refusal too: "it missed a threshold" means something
    # different when the threshold was chosen rather than derived.
    assert "DECLARED-NOT-DERIVED" in decision.rationale


def test_the_refusal_survives_raising_similarity_to_the_maximum():
    """Even at ppm_similarity 1.0 the sign-flipped pair is still refused.

    Which is the point of registering on unsigned ppm and then gating on the sign
    separately: an unsigned similarity threshold, however high, cannot tell these
    two apart, and the criterion does not rely on one to.
    """
    from dataclasses import replace

    decision = adjudicate_component(
        [FLIPPED_POS, FLIPPED_NEG],
        [replace(FLIPPED_EDGE, ppm_similarity=1.0, overlap_bp=13)],
        [], [], _shipped(), "test", node_metadata=_flipped_metadata(),
    )
    assert decision.decision is Decision.REFUSE_MERGE


# --------------------------------------------------------------------------- #
# 3. It defers when the evidence is missing.
# --------------------------------------------------------------------------- #

def test_it_defers_when_the_null_was_never_computed():
    """`align --null-shuffles 0` leaves no null, and that is not a failed null.

    The same CTCF pair that collapses above, registered without calibration, must
    DEFER. "This run computed no null" and "this pair did not beat its null" are
    different facts; conflating them would let a cheap run collapse pairs the
    expensive one refuses. `at_alignment_null_floor` returns None for exactly this
    reason, and None is missing evidence, which can never license a collapse.
    """
    from motifmultiverse.adjudicate import at_alignment_null_floor

    uncalibrated = _ctcf_edge(empirical_p_value=None, null_shuffles=0)
    assert at_alignment_null_floor(uncalibrated) is None

    decision = adjudicate_component(
        [CTCF_SOURCE, CTCF_TARGET], [uncalibrated], [], [],
        _shipped(), "test", node_metadata=_ctcf_metadata(),
    )

    assert decision.relationship == "TRUE_DUPLICATE"
    assert decision.decision is Decision.DEFERRED
    assert decision.representative_node_id is None
    assert "missing required evidence" in decision.rationale
    assert "at_alignment_null_floor" in decision.rationale


def test_the_null_floor_is_the_instruments_resolution_not_a_fixed_alpha():
    """The gate must move with `--null-shuffles`, or it is a chosen alpha in disguise.

    At 1000 shuffles the case study's p = 0.000999 is AT the floor. Re-express the
    identical pair at 100 shuffles and the same p-value is comfortably below a floor
    of 1/101 = 0.00990, still at the floor; but a pair that merely reached 1/101 is
    NOT at the floor when 1000 shuffles were run. A literal `empirical_p_value le
    0.001` predicate could not say that, which is why the criterion does not use one.
    """
    from motifmultiverse.adjudicate import at_alignment_null_floor

    assert at_alignment_null_floor(_ctcf_edge()) is True
    assert at_alignment_null_floor(
        _ctcf_edge(empirical_p_value=1 / 101, null_shuffles=100)
    ) is True
    assert at_alignment_null_floor(
        _ctcf_edge(empirical_p_value=1 / 101, null_shuffles=NULL_SHUFFLES)
    ) is False


# --------------------------------------------------------------------------- #
# 4. The known cost of freezing it, pinned rather than described.
# --------------------------------------------------------------------------- #

#: The paralog merge the criterion makes on the real run, and does not avoid.
#: cl4/pattern_2 best-matches CTCF_MOUSE.H11MO.0.A at q = 1.61e-10; cl5/pattern_4
#: best-matches CTCFL_MOUSE.H11MO.0.A at q = 3.06e-11
#: (casestudy/evidence/annotation_candidates.parquet). In K562, CTCFL/BORIS is a
#: real, separately regulated factor. Both resolve to C2H2_ZINC_FINGER_FACTORS, so
#: `family_conflict` never fires, and the pair clears every predicate comfortably.
PARALOG_CTCF = "cbp_counts_cl4::pos_patterns.pattern_2"
PARALOG_CTCFL = "cbp_counts_cl5::pos_patterns.pattern_4"

PARALOG_EDGE = AlignmentEvidence(
    source_node_id=PARALOG_CTCF,
    target_node_id=PARALOG_CTCFL,
    orientation="+",
    offset=0,
    overlap_bp=14,
    overlap_frac_source=1.0,
    overlap_frac_target=1.0,
    ppm_similarity=0.97413,
    signed_cwm_similarity=0.98134,
    empirical_p_value=NULL_FLOOR_P,
    null_shuffles=NULL_SHUFFLES,
    seed=SEED,
)


def test_the_criterion_merges_ctcf_with_its_paralog_and_this_is_not_fixed():
    """KNOWN COST, pinned as a passing assertion so it cannot be forgotten.

    This is one of the four collapses the shipped criterion licenses on the real
    run -- 25% of its output -- and the error direction is DELETION, which is
    irreversible for the reader of the lexicon, rather than the status quo's
    inflation. It is not a near-miss: every predicate is cleared with room to
    spare. No predicate in v2 can distinguish these two, because the evidence that
    does (tomtom's best match, five orders of magnitude apart) is not carried on a
    field any criterion can read.

    If someone adds the `best_matched_motif_id` discriminator named first in the
    criterion's `replacement_evidence`, this test SHOULD start failing. That is the
    signal to delete it, not to weaken it.
    """
    metadata = _metadata(**{
        PARALOG_CTCF: ("CBPK562_UNASSIGNED_84", 0.26, 210, 12.0),
        PARALOG_CTCFL: ("CBPK562_UNASSIGNED_92", 0.28, 260, 12.5),
    })
    decision = adjudicate_component(
        [PARALOG_CTCF, PARALOG_CTCFL], [PARALOG_EDGE], [], [],
        _shipped(), "test", node_metadata=metadata,
    )
    assert decision.decision is Decision.COLLAPSE, (
        "the paralog merge stopped happening; if that is because a sub-family "
        "discriminator now exists, delete this test -- do not relax it"
    )

    # The criterion's own exit route names this as the first thing to fix.
    criterion = load_criteria(packaged_criteria_path())["TRUE_DUPLICATE"]
    assert any(
        "best_matched_motif_id" in entry for entry in criterion.replacement_evidence
    ), "the fix for the paralog merge must stay written down in replacement_evidence"


# --------------------------------------------------------------------------- #
# 5. The two disclosures a reader needs, pinned so they cannot be quietly cut.
# --------------------------------------------------------------------------- #

def test_the_rationale_reports_that_a_stricter_threshold_removes_MORE_motifs():
    """The sweep table shipped in an earlier draft reported only the reassuring
    direction: ppm_similarity 0.99 removes 2, down from the plateau's 9. It omitted
    0.96 and 0.97, which remove 10 -- MORE than the plateau -- because the
    criterion's own predicates double as the component-proposal edge filter
    (`adjudicate.edge_admits_duplicate_candidate`), so raising a threshold
    re-partitions the graph instead of merely dropping candidates. A pair buried in
    a component refused for family_conflict can be freed into a two-node component
    that then passes.

    A reader deciding whether 0.90 is conservative has to be told that "higher" is
    not "safer" here. This test fails if that disclosure is deleted.
    """
    criterion = load_criteria(packaged_criteria_path())["TRUE_DUPLICATE"]
    prose = (criterion.declared_rationale or "") + " ".join(
        p.basis or "" for p in criterion.predicates
    )

    assert "0.96" in prose and "0.97" in prose, (
        "the two thresholds that remove MORE motifs than the plateau are not named"
    )
    assert "MORE" in prose or "not monotone" in prose.lower(), (
        "the direction is not stated; a table of numbers a reader must diff for "
        "themselves is how this was concealed the first time"
    )


def test_the_rationale_names_the_fp08_evidence_v2_dropped():
    """v1 required both fields FP-08 names for a redundancy claim; v2 requires
    neither. Dropping reconstruction is argued (it lacks power). Dropping
    `affected_coefficient_share` was never mentioned at all.

    Both absences are pinned here against the legacy registry, so this test states
    the regression rather than describing it.
    """
    from motifmultiverse.adjudicate import packaged_legacy_criteria_path

    legacy = load_criteria(packaged_legacy_criteria_path())["TRUE_DUPLICATE"]
    shipped = load_criteria(packaged_criteria_path())["TRUE_DUPLICATE"]

    fp08_fields = {"paired_delta_reconstruction_affected", "affected_coefficient_share"}
    assert fp08_fields <= set(legacy.required_evidence), (
        "v1 is the baseline this regression is measured against"
    )
    assert not (fp08_fields & set(shipped.required_evidence)), (
        "if v2 has reacquired either field, this test has become a lie -- delete "
        "it and delete the KNOWN COST 2 paragraph with it"
    )
    assert "FP-08" in (shipped.declared_rationale or ""), (
        "a dropped constraint that is not named is a dropped constraint nobody "
        "will find"
    )
    assert "affected_coefficient_share" in (shipped.declared_rationale or ""), (
        "reconstruction's removal is argued; the coefficient share's removal must "
        "at least be admitted"
    )
