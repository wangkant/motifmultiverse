"""Two independent axes, where the reference implementation had one conflated grade.

``SelectionProvenance`` used to answer a single question -- "what may this query
emit" -- by dispatching straight to an ``OutputMode``. That conflates two
questions that do not covary:

1. **``StatisticalLicense``** -- may this query support inference at all? A
   held-out split earns this even when the selection itself was clustered.
2. **``ClaimScope``** -- what can the resulting number be a claim *about*? A
   result can be statistically licensed and still be circular with respect to
   the substrate it was measured against, if the feature that chose the peak
   set is the same signal now being measured.

The canonical example is a held-out attribution cluster: split correctly, its
number is ``HELD_OUT_INFERENCE`` (fully licensed) *and*, because the selection
feature is attribution-derived, ``SUBSTRATE_CIRCULAR`` (a claim about the
model's own attribution surface, not about anything external to it). Collapsing
these into one grade would have to pick a side and either lose the license or
lose the circularity warning; keeping them apart lets a caller state both.

``resolve_query_permissions`` computes the two independently from
(``provenance``, ``selection_feature_names``): neither return value is derived
from the other. Unknown or missing provenance is refused to the floor of
*both* axes -- ``DESCRIPTIVE_ONLY`` and ``CONDITIONING_UNVERIFIABLE`` -- because
an undeclared selection is not a safe selection on either question.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "StatisticalLicense", "ClaimScope", "RepresentationId", "VariantId",
    "resolve_query_permissions", "DEFAULT_ATTRIBUTION_DERIVED_FEATURE_NAMES",
    "IDENTITY_SCHEMA_VERSION",
]

#: This module's schema revision. It defaults onto each identity value below so
#: every new schema object carries its version while callers retain the compact
#: five- and three-field construction forms specified for Task 7.
IDENTITY_SCHEMA_VERSION = "1"


class StatisticalLicense(StrEnum):
    """May this query's numbers support inference at all.

    This is read off ``selection_provenance`` alone: a held-out split earns
    ``HELD_OUT_INFERENCE`` regardless of what feature chose the cluster, because
    the license is about whether the *estimator* is valid, not about what the
    estimate is evidence of.
    """

    FULL_INFERENCE = "FULL_INFERENCE"
    HELD_OUT_INFERENCE = "HELD_OUT_INFERENCE"
    DESCRIPTIVE_ONLY = "DESCRIPTIVE_ONLY"


class ClaimScope(StrEnum):
    """What the resulting number can be a claim about.

    Independent of :class:`StatisticalLicense`: a fully licensed number can
    still be ``SUBSTRATE_CIRCULAR`` if the selection feature came from the same
    attribution surface the number now describes.
    """

    EXTERNAL_STRUCTURE = "EXTERNAL_STRUCTURE"
    INTERNAL_DECOMPOSITION = "INTERNAL_DECOMPOSITION"
    SUBSTRATE_CIRCULAR = "SUBSTRATE_CIRCULAR"
    CONDITIONING_UNVERIFIABLE = "CONDITIONING_UNVERIFIABLE"


@dataclass(frozen=True)
class RepresentationId:
    """The same five fields ``MotifNode`` already carries, as one hashable key.

    ``model``/``readout``/``context``/``metacluster``/``local_pattern_id`` name
    one discovered pattern before it is assigned a ``variant_id``. Bundling them
    lets a later step key a table on "this representation" without parsing one
    out of ``node_id`` (Rule 2 of ``schema/__init__.py``). ``schema_version``
    travels with the value, per the global schema contract.
    """

    model: str
    readout: str
    context: str
    metacluster: str
    local_pattern_id: str
    schema_version: str = IDENTITY_SCHEMA_VERSION


@dataclass(frozen=True)
class VariantId:
    """A structured counterpart to the ``<UNION>_<FAMILY>_<NN>`` variant_id string.

    ``namespace`` is the union id's namespace (as in ``NamespacedId``);
    ``family_id`` and ``value`` are kept apart so a caller does not have to
    re-parse the family back out of the string form.
    """

    family_id: str
    namespace: str
    value: str
    schema_version: str = IDENTITY_SCHEMA_VERSION


#: Feature names known to be derived FROM model attribution: DeepSHAP/DeepLIFT
#: projections, MoDISco seqlet/contribution scores, hit-caller coefficients. A
#: peak set selected on one of these and then measured for an attribution-based
#: effect over that same set is circular with respect to the substrate, whether
#: or not the selection also held out a verification half -- which is exactly
#: why this feeds ``ClaimScope`` and not ``StatisticalLicense``. An explicit
#: function parameter in :func:`resolve_query_permissions` besides, so a caller
#: with a different attribution surface can supply its own registry instead.
DEFAULT_ATTRIBUTION_DERIVED_FEATURE_NAMES: frozenset[str] = frozenset({
    "attribution_pc1", "attribution_pc2", "attribution_score", "attribution_magnitude",
    "deepshap_score", "deeplift_score", "hypothetical_contrib_sum", "contrib_score_sum",
    "modisco_seqlet_score", "hit_coefficient", "hit_coefficient_sum",
})


def resolve_query_permissions(
    provenance: Any,
    selection_feature_names: Sequence[str],
    attribution_derived_registry: set[str],
) -> tuple[StatisticalLicense, ClaimScope]:
    """Resolve both axes from one declared provenance and the features it used.

    ``statistical_license`` is read off ``provenance`` alone. ``claim_scope`` is
    read off ``provenance``'s conditioning-verifiability and, separately,
    whether any declared selection feature is in ``attribution_derived_registry``
    -- never off ``statistical_license``. Each axis is a read of the inputs, not
    of the other axis's result, which is what keeps them free to disagree (a
    held-out cluster selected on an attribution feature is licensed on the first
    axis and circular on the second, in the same call).

    Unknown, unrecognised, or missing ``provenance`` (``None``, an empty string,
    a value from some future ledger revision) is refused to the floor of both
    axes: ``DESCRIPTIVE_ONLY`` and ``CONDITIONING_UNVERIFIABLE``. It must never
    resolve to the permissive value of either -- an undeclared selection is not
    a safe selection.
    """
    # Local import: `identity` is a submodule of `schema`, and `schema/__init__`
    # imports names from here at module scope, so importing `schema` back at
    # module scope would be circular. By the time any query is actually
    # resolved, `schema/__init__` has finished executing, so this is safe.
    from motifmultiverse.schema import SelectionProvenance

    try:
        resolved_provenance = SelectionProvenance(provenance)
    except (ValueError, TypeError):
        return StatisticalLicense.DESCRIPTIVE_ONLY, ClaimScope.CONDITIONING_UNVERIFIABLE

    license_by_provenance: dict[SelectionProvenance, StatisticalLicense] = {
        SelectionProvenance.EXTERNAL: StatisticalLicense.FULL_INFERENCE,
        SelectionProvenance.PROGRAMMATIC_RULE: StatisticalLicense.FULL_INFERENCE,
        SelectionProvenance.CLUSTERED_WITH_SPLIT: StatisticalLicense.HELD_OUT_INFERENCE,
        SelectionProvenance.CLUSTERED_NO_SPLIT: StatisticalLicense.DESCRIPTIVE_ONLY,
        SelectionProvenance.EYEBALLED: StatisticalLicense.DESCRIPTIVE_ONLY,
        SelectionProvenance.MODEL_SELECTED_NO_TRANSCRIPT: StatisticalLicense.DESCRIPTIVE_ONLY,
        SelectionProvenance.DECLARATION_MISSING: StatisticalLicense.DESCRIPTIVE_ONLY,
    }
    statistical_license = license_by_provenance.get(
        resolved_provenance, StatisticalLicense.DESCRIPTIVE_ONLY)

    conditioning_unverifiable_provenances = {
        SelectionProvenance.MODEL_SELECTED_NO_TRANSCRIPT,
        SelectionProvenance.DECLARATION_MISSING,
    }
    if resolved_provenance in conditioning_unverifiable_provenances:
        claim_scope = ClaimScope.CONDITIONING_UNVERIFIABLE
    elif any(name in attribution_derived_registry for name in selection_feature_names):
        claim_scope = ClaimScope.SUBSTRATE_CIRCULAR
    elif resolved_provenance is SelectionProvenance.EXTERNAL:
        claim_scope = ClaimScope.EXTERNAL_STRUCTURE
    else:
        claim_scope = ClaimScope.INTERNAL_DECOMPOSITION

    return statistical_license, claim_scope
