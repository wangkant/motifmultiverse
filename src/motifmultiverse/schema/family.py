"""Family assignment as an adjudicated consensus over annotation candidates.

``annotate`` retains database-label candidates; nothing decided a family from
them, and ``annotate/README.md`` records that as a *design* gap rather than an
implementation one. What was missing was not code but an answer to "what must a
family assignment satisfy".

This module supplies one answer and refuses to hide its shape. A family here is
**not** a protein taxonomy class. It is whatever the site's declared, versioned
mapping says, and the only thing this module contributes is the decision rule:
a family is assigned when independent candidates for the same node AGREE, and
is a typed hole when they do not.

The rule exists because of a specific failure. A de-novo contribution-weight
matrix is matched to a database by a similarity search that returns a ranked
list, and the top hit alone is not evidence: on real data a pattern matched
``ATF4_HUMAN.H11MO.0.A`` at ``q = 1.000`` -- chance -- and inherited that
matrix's family, which then dominated its layer. Taking the top hit is what
made that possible. Requiring two of the ranked hits to name the same family
makes a chance hit unable to name anything by itself, without imposing a
q-value ceiling -- which ``annotate/README.md`` pre-refutes, because q is a
function of motif length and a ceiling would silently select on length.

A family that could not be agreed is ``NOT_ASSIGNED_*``, never a sentinel that
downstream arithmetic can mistake for a family. Which refusal fired is part of
the record: "the candidates disagreed" and "there were not enough candidates to
disagree" are different facts about the evidence, and a reader who cannot tell
them apart cannot tell a contested motif from an unsearched one.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from . import SchemaError

__all__ = [
    "FAMILY_ASSIGNMENT_SCHEMA_VERSION", "FamilyAssignment", "FamilyAssignmentState",
    "FamilyVocabulary", "assign_family_by_consensus", "stable_family_assignment_id",
]

#: Version "1": first release of the consensus rule. Bump when the RULE changes,
#: not when a vocabulary is re-versioned -- the vocabulary carries its own
#: ``version`` and ``content_sha256`` on every assignment it produces.
FAMILY_ASSIGNMENT_SCHEMA_VERSION = "1"

#: The consensus rule's identity, recorded on every row so a later rule cannot
#: be mistaken for this one after the fact.
CONSENSUS_RULE_ID = "top_k_candidate_agreement"
CONSENSUS_RULE_VERSION = "1"


class FamilyAssignmentState(StrEnum):
    """Assigned with the strength of its agreement, or a named refusal.

    There is no state meaning "assigned, trust unknown". A caller that wants to
    weight an assignment reads ``n_agreeing`` / ``n_candidates``; a caller that
    wants to exclude the weak ones filters on ``ASSIGNED_UNANIMOUS``.
    """

    #: Every candidate that resolved to a family named the same one.
    ASSIGNED_UNANIMOUS = "ASSIGNED_UNANIMOUS"
    #: A strict majority agreed; at least one candidate named something else.
    ASSIGNED_MAJORITY = "ASSIGNED_MAJORITY"
    #: Candidates resolved, and no family was named more than once.
    NOT_ASSIGNED_SPLIT = "NOT_ASSIGNED_SPLIT"
    #: Fewer resolved candidates than the rule requires to agree at all.
    NOT_ASSIGNED_TOO_FEW = "NOT_ASSIGNED_TOO_FEW"
    #: Candidates exist but none of their matched motifs is in the vocabulary.
    NOT_ASSIGNED_UNDECLARED = "NOT_ASSIGNED_UNDECLARED"
    #: The node carries no annotation candidate at all.
    NOT_ASSIGNED_NO_CANDIDATE = "NOT_ASSIGNED_NO_CANDIDATE"


_ASSIGNED = frozenset(
    {FamilyAssignmentState.ASSIGNED_UNANIMOUS, FamilyAssignmentState.ASSIGNED_MAJORITY})


def stable_family_assignment_id(*, node_id: str, rule_id: str, rule_version: str,
                                vocabulary_id: str, vocabulary_version: str) -> str:
    """Row identity that admits a SECOND proposer instead of overwriting the first.

    Keyed on the rule and the vocabulary as well as the node, so a later
    assignment made by a different rule is an additional row and can never
    silently replace this one. ``docs/DATA_MODEL.md`` asks for exactly that
    shape; a table keyed on ``node_id`` alone cannot represent two proposers
    disagreeing, which is the state most worth being able to see.
    """
    identity = {
        "node_id": node_id, "rule_id": rule_id, "rule_version": rule_version,
        "vocabulary_id": vocabulary_id, "vocabulary_version": vocabulary_version,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"family:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class FamilyVocabulary:
    """A versioned, checksummed ``matched_motif_id -> family_id`` mapping.

    Declared by the site, never shipped: which grouping of motifs counts as one
    family is a scientific choice this package must not make on a caller's
    behalf. What it does insist on is that the choice be *citable* -- an
    unversioned mapping cannot be, so one is refused rather than defaulted, the
    same rule ``annotate`` already applies to a database section's ``version``.

    ``family_id`` must be an opaque join token. It is validated against the
    label map rather than derived from it, because a published taxonomy may
    reuse a display name across distinct groups: in the Vierstra v1.0 archetype
    release, 286 clusters carry 282 distinct names, and the cluster *named*
    ``ZNF143`` is not the cluster whose seed motif is ZNF143. Keying a family on
    its display name merges those two, which is a collapse no downstream number
    can recover from.
    """

    vocabulary_id: str
    version: str
    content_sha256: str
    #: matched_motif_id -> opaque family_id
    mapping: Mapping[str, str]
    #: family_id -> human-readable label. Display only; never a join key.
    labels: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for name in ("vocabulary_id", "version", "content_sha256"):
            if not str(getattr(self, name) or "").strip():
                raise SchemaError(
                    f"FamilyVocabulary.{name} is required: an unversioned, unchecksummed "
                    "vocabulary cannot be cited by a family-level number, and defaulting "
                    "it would make every such number untraceable.")
        if not self.mapping:
            raise SchemaError(
                "FamilyVocabulary.mapping is empty. An empty vocabulary assigns nothing, "
                "which is indistinguishable from the stage not having run; declare the "
                "mapping or do not declare the vocabulary.")
        object.__setattr__(self, "labels", dict(self.labels or {}))

    def label(self, family_id: str) -> str:
        """Display label, falling back to the token itself -- never to a guess."""
        return dict(self.labels).get(family_id, family_id)


@dataclass(frozen=True)
class FamilyAssignment:
    """One node's family, or one node's named refusal, with its whole basis."""

    assignment_id: str
    node_id: str
    state: FamilyAssignmentState
    family_id: str | None
    family_label: str | None
    n_candidates: int
    n_resolved: int
    n_agreeing: int
    rule_id: str
    rule_version: str
    vocabulary_id: str
    vocabulary_version: str
    vocabulary_sha256: str
    #: Every (matched_motif_id, family_id-or-None) the rule saw, in rank order.
    evidence: tuple[tuple[str, str | None], ...] = ()

    def __post_init__(self) -> None:
        assigned = self.state in _ASSIGNED
        if assigned and not self.family_id:
            raise SchemaError(
                f"{self.node_id}: state {self.state} claims a family but names none.")
        if not assigned and self.family_id is not None:
            raise SchemaError(
                f"{self.node_id}: state {self.state} is a refusal, so it must not carry "
                f"a family_id (got {self.family_id!r}). A refusal that names a family is "
                "the collapse this type exists to prevent.")
        if self.state is FamilyAssignmentState.ASSIGNED_UNANIMOUS and \
                self.n_agreeing != self.n_resolved:
            raise SchemaError(
                f"{self.node_id}: UNANIMOUS requires every resolved candidate to agree "
                f"({self.n_agreeing} of {self.n_resolved} did).")
        if self.n_resolved > self.n_candidates:
            raise SchemaError(
                f"{self.node_id}: {self.n_resolved} resolved candidates out of "
                f"{self.n_candidates} -- resolution cannot create candidates.")

    @property
    def is_assigned(self) -> bool:
        return self.state in _ASSIGNED


def assign_family_by_consensus(
    *,
    node_id: str,
    matched_motif_ids: Sequence[str],
    vocabulary: FamilyVocabulary,
    min_agreeing: int = 2,
) -> FamilyAssignment:
    """Assign a family when ``min_agreeing`` candidates name the same one.

    ``matched_motif_ids`` is the ranked candidate list for one node, best first.
    Rank is deliberately NOT used to break a tie: a rule that falls back to the
    top hit whenever the vote is inconclusive is the top-hit rule wearing a
    disguise, and the top hit is the thing this rule exists not to trust. An
    inconclusive vote is ``NOT_ASSIGNED_SPLIT``.

    Unresolvable candidates -- a matched motif the vocabulary does not declare --
    are counted and reported but never voted with, and they do not lower the
    bar for the ones that did resolve.
    """
    if min_agreeing < 2:
        raise SchemaError(
            f"min_agreeing={min_agreeing} would let a single candidate decide, which is "
            "the top-hit rule this function replaces. Use 2 or more.")

    seen = [(m, vocabulary.mapping.get(m)) for m in matched_motif_ids]
    resolved = [f for _, f in seen if f is not None]
    counts = Counter(resolved)

    if not seen:
        state, fam, agree = FamilyAssignmentState.NOT_ASSIGNED_NO_CANDIDATE, None, 0
    elif not resolved:
        state, fam, agree = FamilyAssignmentState.NOT_ASSIGNED_UNDECLARED, None, 0
    else:
        top, agree = counts.most_common(1)[0]
        tied = [f for f, n in counts.items() if n == agree]
        if agree < min_agreeing:
            state, fam = ((FamilyAssignmentState.NOT_ASSIGNED_TOO_FEW
                           if len(resolved) < min_agreeing
                           else FamilyAssignmentState.NOT_ASSIGNED_SPLIT), None)
            agree = 0
        elif len(tied) > 1:
            # Two families equally supported. Rank would break it; rank is not evidence.
            state, fam, agree = FamilyAssignmentState.NOT_ASSIGNED_SPLIT, None, 0
        elif agree == len(resolved):
            state, fam = FamilyAssignmentState.ASSIGNED_UNANIMOUS, top
        else:
            state, fam = FamilyAssignmentState.ASSIGNED_MAJORITY, top

    return FamilyAssignment(
        assignment_id=stable_family_assignment_id(
            node_id=node_id, rule_id=CONSENSUS_RULE_ID, rule_version=CONSENSUS_RULE_VERSION,
            vocabulary_id=vocabulary.vocabulary_id, vocabulary_version=vocabulary.version),
        node_id=node_id, state=state, family_id=fam,
        family_label=vocabulary.label(fam) if fam else None,
        n_candidates=len(seen), n_resolved=len(resolved), n_agreeing=agree,
        rule_id=CONSENSUS_RULE_ID, rule_version=CONSENSUS_RULE_VERSION,
        vocabulary_id=vocabulary.vocabulary_id, vocabulary_version=vocabulary.version,
        vocabulary_sha256=vocabulary.content_sha256, evidence=tuple(seen))
