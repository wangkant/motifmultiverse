"""Compile tiered lexicons from a registry, in the order the loader will read them.

Output per tier: one hit-caller-compatible HDF5 plus a manifest carrying the index,
the content hash and an explicit statement of how this tier differs from the others.

Three constraints here come from reading the actual loader rather than from a
specification:

**Order is the loader's, not the writer's.** The loader walks
``['pos_patterns', 'neg_patterns']`` in that fixed order and sorts within a group
by the integer suffix of ``pattern_N``. A frozen index sorted by metacluster
ascending (``neg`` < ``pos``) therefore does **not** match what comes back, and any
positional read against it is wrong. In the reference implementation that mistake
was invisible because one model had no negative motifs at all, so the two orders
happened to coincide. ``guards.index_order_matches_loader`` compares by name.

**A tier contrast that changes nothing must say so.** If ``core`` and ``expanded``
hold the same positive motifs, the manifest records ``positive_sets_identical``.
The reference implementation's did not, and a sensitivity analysis that looked as
though it varied lexicon width had in fact varied only the negative half.

**A lexicon is content-addressed.** ``FP-11`` requires every family-level number to
state the lexicon it was computed under; without a hash there is nothing to state.

And one thing this module deliberately does **not** have: a numeric threshold on
merge confidence. Sensitivity-lexicon membership is decided by three named
triggers, never by a cut-off -- see :func:`triggers_by_cluster`.

Out of scope: seqlet-consuming loader entry points. What is written here targets
the motif loader (CWM / hypothetical CWM / PPM).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from motifmultiverse import guards
from motifmultiverse.guard_log import GuardLog
from motifmultiverse.ingest import GROUP_METACLUSTER, MODISCO_GROUPS, load_registry
from motifmultiverse.provenance import record
from motifmultiverse.schema import (
    LEXICON_MANIFEST_SCHEMA_VERSION,
    MISSING_SENTINEL,
    Decision,
    DecisionBundle,
    DecisionRecord,
    LexiconManifest,
    SchemaError,
    Tier,
    sensitivity_triggers,
)

__all__ = [
    "CompileError", "BackendMissing", "BackendIncompatible", "TIERS",
    "compile_lexicons", "lexicon_semantic_hash", "load_back", "probe_backend",
    "operations_log", "validate_compiled_lexicon", "verify_roundtrip",
]

TIERS = ("core", "expanded", "sensitivity")


class CompileError(ValueError):
    """A lexicon cannot be compiled as declared."""


class BackendMissing(RuntimeError):
    """No usable backend is available to perform verification.

    Raised when the backend cannot be imported at all, and -- via
    :class:`BackendIncompatible` -- when it imports but cannot be called. Both
    mean the same thing to a reader of the artifact: *no round trip happened*.
    Keeping them one catchable type is what lets ``--verify-roundtrip auto``
    treat them alike (write the lexicon, claim nothing) while ``require`` refuses
    for either reason.
    """


class BackendIncompatible(BackendMissing):
    """The backend is installed but does not accept the call this package makes.

    Distinct from a missing backend because the diagnosis is different -- the fix
    is a version, not an install -- but a subclass because the *consequence* is
    identical: nothing read the lexicon back.

    The failure this exists to prevent: ``load_back`` passed the loader's
    arguments by keyword inside a ``try`` that caught only ``ImportError``, so
    when the backend renamed one of them the call raised a bare ``TypeError``
    from the middle of ``compile_lexicons``. Under ``--verify-roundtrip auto``,
    which is the default and is documented as "verify if a backend is there,
    otherwise carry on", that aborted the whole compile: the lexicon was not
    written at all, for a reason no message connected to the backend. Naming the
    condition keeps the diagnosis in the message and keeps ``auto`` doing what it
    says.
    """


# --------------------------------------------------------------------------- #
# Tier membership
# --------------------------------------------------------------------------- #
def _apply_tiers(nodes: list[dict[str, Any]], overrides: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply per-node ``tiers`` overrides, then re-check the invariant they can break.

    ``MotifNode.__post_init__`` refuses ``discovery_tier != analysis_tier`` unless
    ``tier_reason`` is given explicitly. Ingest always constructs nodes with
    ``discovery_tier == analysis_tier == CORE``, so that check is trivially
    satisfied there -- the only place a divergence can be *introduced* is here,
    by an override that changes one of the pair but not the other. Because this
    function works over plain dicts (not ``MotifNode`` instances), applying an
    override never re-runs the dataclass validation that would have rejected the
    resulting state had it been constructed from scratch. Re-run it explicitly:
    an override is licensed to diverge the two tiers, but only alongside the same
    explicit reason ``MotifNode`` would have demanded.
    """
    out = []
    for node in nodes:
        node = dict(node)
        override = overrides.get(node["node_id"]) or {}
        node.update({k: v for k, v in override.items()
                     if k in {"discovery_tier", "analysis_tier", "tier_reason"}})
        discovery_tier, analysis_tier = node.get("discovery_tier"), node.get("analysis_tier")
        if (discovery_tier is not None and analysis_tier is not None
                and discovery_tier != analysis_tier
                and node.get("tier_reason", MISSING_SENTINEL) == MISSING_SENTINEL):
            raise CompileError(
                f"{node['node_id']}: tier override diverges discovery_tier "
                f"({discovery_tier!r}) from analysis_tier ({analysis_tier!r}) with no "
                "tier_reason. This is the same invariant MotifNode.__post_init__ "
                "enforces at ingest time; an override that introduces the divergence "
                "must supply the reason too, not silently apply."
            )
        if node.get("analysis_tier") == Tier.EXCLUDED.value:
            continue
        out.append(node)
    return out


def triggers_by_cluster(decisions: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Which named triggers fire for each collapse decision.

    Membership of the sensitivity lexicon is decided by three named conditions --
    ``merge_confidence != HIGH``, ``family_ambiguity``, ``threshold_sensitive`` --
    and by no number. There is no scalar merge confidence to threshold: the design
    lists "moderate-confidence merge" as a trigger without defining it, and the
    reference implementation produced the value by looking up a family name. Any
    cut-off would invent a continuous quantity that does not exist.
    """
    return {str(d.get("cluster_id")): sensitivity_triggers(d) for d in decisions
            if d.get("decision") == Decision.COLLAPSE.value}


def _assert_members_known(decisions: list[DecisionRecord], known_ids: set[str]) -> None:
    """Every node a decision names must exist somewhere in this registry.

    ``DecisionBundle.from_dict`` validates the payload's internal consistency but
    has no registry to check against; this is the one place that does. A stale
    decision -- one left over from a registry that has since been re-ingested --
    is refused rather than silently dropped, because a dropped member changes the
    lexicon's content without saying so.
    """
    for d in decisions:
        for member in d.members:
            if member not in known_ids:
                raise CompileError(
                    f"decision {d.cluster_id!r} names unknown decision member "
                    f"{member!r}; it is not a node_id anywhere in this registry. A "
                    "stale decision naming a node that no longer exists is refused, "
                    "not silently ignored."
                )


def _assert_tier_overrides_known(tiers: dict[str, dict[str, str]], known_ids: set[str]) -> None:
    """Every node a ``tiers`` override names must exist somewhere in this registry.

    Mirrors ``_assert_members_known``: ``DecisionBundle.from_dict`` validates that
    an override's *value* names a real Tier, but has no registry to check its
    *key* -- the node id -- against. Before this check, an override naming a node
    that has since been dropped from the registry (a stale re-ingest) was simply
    never looked up by ``_apply_tiers`` and silently did nothing (round-1 review
    finding 1a).
    """
    for node_id in tiers:
        if node_id not in known_ids:
            raise CompileError(
                f"tier override names unknown node {node_id!r}; it is not a "
                "node_id anywhere in this registry. A stale override naming a "
                "node that no longer exists is refused, not silently ignored."
            )


def _members_for_tier(nodes: list[dict[str, Any]], decisions: list[dict[str, Any]],
                      tier: str) -> list[dict[str, Any]]:
    """Which nodes make up one tier's lexicon, after applicable collapses."""
    if tier == "core":
        kept = [n for n in nodes if n.get("analysis_tier") == Tier.CORE.value]
    else:
        kept = [n for n in nodes if n.get("analysis_tier") in
                (Tier.CORE.value, Tier.EXPANDED.value)]
    by_id = {n["node_id"]: n for n in kept}

    collapsed_away: set[str] = set()
    for d in decisions:
        if d.get("decision") != Decision.COLLAPSE.value:
            continue
        members = [m for m in (d.get("members") or []) if m in by_id]
        if not members:
            continue
        if tier == "sensitivity" and sensitivity_triggers(d):
            continue          # left split in the sensitivity lexicon, by named trigger
        # The representative's membership was already validated against the
        # decision's own member list (DecisionRecord.__post_init__); what is
        # checked here is narrower and tier-specific: this collapse is about to
        # be applied in `tier`, and its surviving members must not be silently
        # dropped just because the representative itself did not survive into
        # this particular tier.
        representative = d.get("representative")
        if representative not in by_id:
            raise CompileError(
                f"collapse {d.get('cluster_id')!r}: representative is absent from "
                f"tier {tier} (representative={representative!r}), even though "
                f"member(s) {sorted(members)} survive into it. The surviving "
                "members are never silently dropped to accommodate a "
                "representative missing from this tier."
            )
        collapsed_away.update(m for m in members if m != representative)
    return [n for n in kept if n["node_id"] not in collapsed_away]


# --------------------------------------------------------------------------- #
# Writing, in loader order
# --------------------------------------------------------------------------- #
def loader_order(members: list[dict[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
    """Return ``(group, pattern_name, node)`` in the order the loader emits.

    Positive group first, then negative -- because that is the order the loader
    iterates, not because positives matter more. Within a group, ``pattern_N`` is
    assigned in the members' existing order, so the loader's numeric sort
    reproduces exactly this sequence.
    """
    by_group = {g: [] for g in MODISCO_GROUPS}
    for node in members:
        group = next((g for g, mc in GROUP_METACLUSTER.items()
                      if mc == node.get("metacluster")), None)
        if group is None:
            raise CompileError(
                f"{node['node_id']}: metacluster {node.get('metacluster')!r} maps to no "
                f"loader group; expected one of {sorted(GROUP_METACLUSTER.values())}"
            )
        by_group[group].append(node)
    ordered = []
    for group in MODISCO_GROUPS:
        for i, node in enumerate(by_group[group]):
            ordered.append((group, f"pattern_{i}", node))
    return ordered


def lexicon_semantic_hash(ordered: list[tuple[str, str, dict[str, Any]]], arrays: Any, *,
                  schema_version: str, trim_threshold: float, motif_type: str,
                  include_rc: bool, loader_backend: str,
                  loader_parameters: dict[str, Any]) -> str:
    """A lexicon's identity: canonical loader configuration, then ordered array bytes.

    Two lexicons built from byte-identical motif arrays but compiled to be read
    back under different loader settings (a different ``trim_threshold``,
    ``motif_type``, ``include_rc``, or ``loader_parameters``) load differently and
    must not collide on identity. The metadata blob is hashed first, as canonical
    JSON (sorted keys, tight separators, no whitespace or key-order dependence),
    so the hash is deterministic across runs and machines rather than resting on a
    ``dict`` iteration order or a ``repr``. Array identity follows: loader order,
    node id, **variant id**, and each array's dtype and shape ahead of its bytes,
    so two arrays that happen to serialize to the same byte length at different
    shapes cannot collide either.

    ``variant_id`` is in there because it is what downstream binds to. The hit
    caller returns ``pattern_tag``; the manifest index is the only table that
    resolves a tag to the semantic identity a family-level number is grouped by
    (``interpret/README.md`` step 3), and ``schema`` calls it the *only* stable
    semantic identity. Hashing tag, node id and arrays alone let two lexicons that
    resolve the same tags to entirely different variant identities share one
    ``lexicon_content_hash`` -- so a number citing that hash under ``FP-11``
    named a lexicon that could not be told apart from the other one, which is
    exactly the citation ``FP-11`` exists to make meaningful.
    """
    h = hashlib.sha256()
    metadata = {
        "schema_version": schema_version,
        "trim_threshold": trim_threshold,
        "motif_type": motif_type,
        "include_rc": include_rc,
        "loader_backend": loader_backend,
        "loader_parameters": loader_parameters,
    }
    h.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for group, pattern_name, node in ordered:
        try:
            identity = f"{group}.{pattern_name}\t{node['node_id']}\t{node['variant_id']}\n"
        except KeyError as exc:
            raise CompileError(
                f"{group}.{pattern_name} names no {exc.args[0]}. A lexicon's identity "
                "covers the variant_id each pattern tag resolves to, so it cannot be "
                "computed from a node that does not carry one; supplying the tag alone "
                "would content-address two different variant assignments identically."
            ) from exc
        h.update(identity.encode())
        grp = arrays[node["node_id"]]
        for key in ("cwm", "hypothetical_cwm", "ppm"):
            h.update(key.encode())
            if key in grp:
                arr = grp[key][:].astype("float64")
                h.update(f"{arr.dtype}\t{arr.shape}\n".encode())
                h.update(arr.tobytes())
    return h.hexdigest()


# Compatibility for callers from the initial implementation; validation uses the
# public function above so compiler and verifier share one authoritative algorithm.
_content_hash = lexicon_semantic_hash


_COMPILED_MANIFEST_FIELDS = {
    *(item.name for item in fields(LexiconManifest)),
    "index",
    "project",
    "cross_model_claims_restricted",
}
_COMPILED_INDEX_FIELDS = {
    "index", "pattern_tag", "node_id", "variant_id", "metacluster",
}
_COMPILED_MOTIF_DATASETS = {"contrib_scores", "hypothetical_contribs", "sequence"}
#: ``motif_type`` -> (the HDF5 dataset the loader reads, the registry array key).
#: The **keys are the loader's vocabulary**, not ours: ``motif_type`` is handed
#: verbatim to ``load_modisco_motifs``, which dispatches on ``cwm`` / ``hcwm`` /
#: ``pfm`` / ``pfm_softmax`` and has no branch for anything else -- an unknown
#: value leaves its local motif variables unbound and raises ``UnboundLocalError``
#: deep inside the backend rather than being rejected. This table once said
#: ``ppm``, which is *our* name for the registry array (the values' second
#: element), so ``compile(motif_type="ppm")`` wrote a lexicon that the very loader
#: its manifest declared could not read. The two namespaces meet here, in a table,
#: and nowhere else.
_MOTIF_TYPE_DATASET = {
    "cwm": ("contrib_scores", "cwm"),
    "hcwm": ("hypothetical_contribs", "hypothetical_cwm"),
    "pfm": ("sequence", "ppm"),
}
_VARIANT_ID_RE = re.compile(r"^[A-Za-z0-9]+_[A-Za-z0-9]+_\d{2,}$")
_COMPARISON_FIELDS = {
    "positive_sets_identical",
    "negative_sets_identical",
    "n_positive_here",
    "n_positive_there",
    "n_negative_here",
    "n_negative_there",
    "only_here",
    "only_there",
}
_SENSITIVITY_TRIGGER_VALUES = {
    "merge_confidence_not_high",
    "family_ambiguity",
    "threshold_sensitive",
}


def _require_exact_json_fields(
    payload: dict[str, Any],
    expected: set[str],
    *,
    what: str,
) -> None:
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {sorted(missing)}")
        if unknown:
            detail.append(f"unknown {sorted(unknown)}")
        raise CompileError(f"{what} schema is not exact: {', '.join(detail)}")


def _require_json_type(
    payload: dict[str, Any],
    key: str,
    expected: type | tuple[type, ...],
    *,
    what: str,
) -> Any:
    value = payload[key]
    if isinstance(value, bool) and expected is not bool:
        raise CompileError(f"{what} {key} has type bool, not {expected}")
    if not isinstance(value, expected):
        raise CompileError(f"{what} {key} has type {type(value).__name__}, not {expected}")
    return value


def _motif_type_dataset(motif_type: object, *, what: str) -> tuple[str, str]:
    if not isinstance(motif_type, str) or motif_type not in _MOTIF_TYPE_DATASET:
        raise CompileError(
            f"{what} motif_type must be one of {sorted(_MOTIF_TYPE_DATASET)}"
        )
    return _MOTIF_TYPE_DATASET[motif_type]


def _validate_nested_manifest_schema(payload: dict[str, Any], *, what: str) -> None:
    for other_tier, comparison in payload["comparisons"].items():
        if (
            not isinstance(other_tier, str)
            or other_tier not in TIERS
            or other_tier == payload["tier"]
        ):
            raise CompileError(f"{what} comparisons has an invalid tier key")
        if type(comparison) is not dict:
            raise CompileError(f"{what} comparisons[{other_tier!r}] must be an object")
        comparison_fields = set(comparison)
        if comparison_fields not in (
            _COMPARISON_FIELDS,
            _COMPARISON_FIELDS | {"warning"},
        ):
            raise CompileError(
                f"{what} comparisons[{other_tier!r}] schema is not exact"
            )
        for name in ("positive_sets_identical", "negative_sets_identical"):
            if type(comparison[name]) is not bool:
                raise CompileError(
                    f"{what} comparisons[{other_tier!r}].{name} must be a boolean"
                )
        for name in (
            "n_positive_here",
            "n_positive_there",
            "n_negative_here",
            "n_negative_there",
        ):
            if type(comparison[name]) is not int or comparison[name] < 0:
                raise CompileError(
                    f"{what} comparisons[{other_tier!r}].{name} "
                    "must be a non-negative integer"
                )
        for name in ("only_here", "only_there"):
            values = comparison[name]
            if (
                type(values) is not list
                or any(type(value) is not str or not value.strip() for value in values)
                or values != sorted(set(values))
            ):
                raise CompileError(
                    f"{what} comparisons[{other_tier!r}].{name} "
                    "must be a sorted unique string array"
                )
        if "warning" in comparison and (
            type(comparison["warning"]) is not str
            or not comparison["warning"].strip()
        ):
            raise CompileError(
                f"{what} comparisons[{other_tier!r}].warning "
                "must be a non-empty string"
            )

    for cluster_id, values in payload["sensitivity_triggers"].items():
        if type(cluster_id) is not str or not cluster_id.strip():
            raise CompileError(f"{what} sensitivity_triggers keys must be non-empty strings")
        if (
            type(values) is not list
            or not values
            or any(type(value) is not str for value in values)
            or values != list(dict.fromkeys(values))
            or any(value not in _SENSITIVITY_TRIGGER_VALUES for value in values)
        ):
            raise CompileError(
                f"{what} sensitivity_triggers[{cluster_id!r}] must be a unique "
                "non-empty array of named compiler triggers"
            )


def validate_compiled_lexicon(
    manifest_path: str | os.PathLike[str],
    h5_path: str | os.PathLike[str] | None = None,
) -> LexiconManifest:
    """Load and prove one exact compiler-emitted manifest/HDF5 pair.

    This is the public structural loader shared by compile consumers.  It
    enumerates the complete HDF5 tree emitted by :func:`_write_h5`, rather than
    trusting a manifest-controlled subset: every root group, motif group, and
    motif dataset must be accounted for.  The semantic hash is then recomputed
    with :func:`lexicon_semantic_hash`, the same public algorithm used to emit
    the manifest.
    """
    import h5py
    import numpy as np

    manifest_source = Path(manifest_path)
    h5_source = (
        Path(h5_path)
        if h5_path is not None
        else manifest_source.with_name(manifest_source.name.removesuffix(".manifest.json") + ".h5")
    )
    try:
        payload = json.loads(manifest_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompileError(f"{manifest_source} is not valid JSON: {exc}") from exc
    if type(payload) is not dict:
        raise CompileError(f"{manifest_source} manifest must be a JSON object")
    _require_exact_json_fields(
        payload, _COMPILED_MANIFEST_FIELDS, what=f"{manifest_source} manifest",
    )
    string_fields = {
        "tier", "lexicon_content_hash", "schema_version", "motif_type",
        "loader_backend", "source_registry", "project",
    }
    for name in string_fields:
        value = _require_json_type(payload, name, str, what=f"{manifest_source} manifest")
        if not value.strip():
            raise CompileError(f"{manifest_source} manifest {name} must be non-empty")
    _require_json_type(payload, "n_motifs", int, what=f"{manifest_source} manifest")
    trim_threshold = _require_json_type(
        payload, "trim_threshold", (int, float), what=f"{manifest_source} manifest",
    )
    if not math.isfinite(trim_threshold):
        raise CompileError(f"{manifest_source} manifest trim_threshold must be finite")
    for name in ("include_rc", "cross_model_claims_restricted"):
        if type(payload[name]) is not bool:
            raise CompileError(f"{manifest_source} manifest {name} must be a boolean")
    for name in ("loader_parameters", "comparisons", "sensitivity_triggers"):
        if type(payload[name]) is not dict:
            raise CompileError(f"{manifest_source} manifest {name} must be an object")
    for name in ("pattern_order", "node_ids", "index"):
        if type(payload[name]) is not list:
            raise CompileError(f"{manifest_source} manifest {name} must be an array")
    required_dataset, _required_semantic_name = _motif_type_dataset(
        payload["motif_type"], what=f"{manifest_source} manifest",
    )
    motif_lambda = payload["loader_parameters"].get("motif_lambda_default")
    if (
        isinstance(motif_lambda, bool)
        or not isinstance(motif_lambda, (int, float))
        or not math.isfinite(motif_lambda)
    ):
        raise CompileError(
            f"{manifest_source} manifest loader_parameters must contain the finite "
            "resolved motif_lambda_default"
        )
    _validate_nested_manifest_schema(
        payload, what=f"{manifest_source} manifest",
    )

    manifest_fields = {item.name for item in fields(LexiconManifest)}
    try:
        manifest = LexiconManifest(**{key: payload[key] for key in manifest_fields})
    except (TypeError, ValueError) as exc:
        raise CompileError(f"{manifest_source} manifest is malformed: {exc}") from exc
    if manifest.schema_version != LEXICON_MANIFEST_SCHEMA_VERSION:
        raise CompileError(
            f"{manifest_source} manifest schema_version must be "
            f"{LEXICON_MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest.tier not in TIERS:
        raise CompileError(f"{manifest_source} manifest has unknown tier {manifest.tier!r}")
    if manifest.n_motifs < 1:
        raise CompileError(f"{manifest_source} manifest must describe at least one motif")
    if (
        len(manifest.lexicon_content_hash) != 64
        or any(character not in "0123456789abcdef"
               for character in manifest.lexicon_content_hash)
    ):
        raise CompileError(f"{manifest_source} manifest has no lowercase SHA-256 content hash")
    for name, values in (
        ("pattern_order", manifest.pattern_order),
        ("node_ids", manifest.node_ids),
    ):
        if any(type(value) is not str or not value.strip() for value in values):
            raise CompileError(f"{manifest_source} manifest {name} must contain strings")
        if len(set(values)) != len(values):
            raise CompileError(f"{manifest_source} manifest has duplicate {name}")

    index = payload["index"]
    if len(index) != manifest.n_motifs:
        raise CompileError(f"{manifest_source} manifest index must describe every motif")
    for position, row in enumerate(index):
        if type(row) is not dict:
            raise CompileError(f"{manifest_source} manifest index row {position} must be an object")
        _require_exact_json_fields(
            row, _COMPILED_INDEX_FIELDS,
            what=f"{manifest_source} manifest index row {position}",
        )
        if type(row["index"]) is not int:
            raise CompileError(
                f"{manifest_source} manifest index row {position} index must be an integer"
            )
        if row["index"] != position:
            raise CompileError(
                f"{manifest_source} manifest index row {position} has noncanonical index "
                f"{row['index']!r}"
            )
        for name in ("pattern_tag", "node_id", "variant_id", "metacluster"):
            if type(row[name]) is not str or not row[name].strip():
                raise CompileError(
                    f"{manifest_source} manifest index row {position} {name} "
                    "must be a non-empty string"
                )
    pattern_tags = [row["pattern_tag"] for row in index]
    node_ids = [row["node_id"] for row in index]
    variant_ids = [row["variant_id"] for row in index]
    if len(set(pattern_tags)) != len(pattern_tags):
        raise CompileError(f"{manifest_source} manifest index has duplicate pattern identity")
    if len(set(node_ids)) != len(node_ids):
        raise CompileError(f"{manifest_source} manifest index has duplicate node identity")
    if any(not _VARIANT_ID_RE.fullmatch(variant_id) for variant_id in variant_ids):
        raise CompileError(
            f"{manifest_source} manifest index has a malformed variant identity"
        )
    if len(set(variant_ids)) != len(variant_ids):
        raise CompileError(f"{manifest_source} manifest index has duplicate variant identity")
    if pattern_tags != manifest.pattern_order or node_ids != manifest.node_ids:
        raise CompileError(
            f"{manifest_source} manifest index order does not match pattern_order/node_ids"
        )

    expected_by_group: dict[str, list[str]] = {group: [] for group in MODISCO_GROUPS}
    ordered: list[tuple[str, str, dict[str, Any]]] = []
    for row in index:
        pieces = row["pattern_tag"].split(".")
        if len(pieces) != 2 or pieces[0] not in expected_by_group:
            raise CompileError(
                f"{manifest_source} manifest index pattern_tag "
                f"{row['pattern_tag']!r} is not compiler-emitted"
            )
        group, pattern = pieces
        expected_pattern = f"pattern_{len(expected_by_group[group])}"
        if pattern != expected_pattern:
            raise CompileError(
                f"{manifest_source} manifest index pattern {pattern!r} is not the next "
                f"compiler-emitted name {expected_pattern!r}"
            )
        if row["metacluster"] != GROUP_METACLUSTER[group]:
            raise CompileError(
                f"{manifest_source} manifest index metacluster does not match {group}"
            )
        expected_by_group[group].append(pattern)
        # variant_id travels with node_id because the semantic hash covers both;
        # by here the index row's variant_id has already been checked for shape
        # and uniqueness, so recomputing the hash from it is a like-for-like
        # replay of what the compiler hashed.
        ordered.append((group, pattern,
                        {"node_id": row["node_id"], "variant_id": row["variant_id"]}))
    expected_order = [
        f"{group}.{pattern}"
        for group in MODISCO_GROUPS
        for pattern in expected_by_group[group]
    ]
    if pattern_tags != expected_order:
        raise CompileError(f"{manifest_source} manifest index is not in compiler loader order")

    arrays: dict[str, dict[str, Any]] = {}
    try:
        with h5py.File(h5_source, "r") as h5:
            seen_objects: dict[int, str] = {}

            def require_local_object(parent: Any, name: str, path: str) -> Any:
                link = parent.get(name, getlink=True)
                if type(link) is not h5py.HardLink:
                    raise CompileError(
                        f"{h5_source} HDF5 {path} uses a nonlocal or soft link; "
                        "compiler-emitted trees contain local hard links only"
                    )
                obj = parent[name]
                address = int(h5py.h5o.get_info(obj.id).addr)
                if address in seen_objects:
                    raise CompileError(
                        f"{h5_source} HDF5 {path} aliases compiler object "
                        f"{seen_objects[address]!r}"
                    )
                seen_objects[address] = path
                return obj

            expected_groups = {
                group for group, patterns in expected_by_group.items() if patterns
            }
            actual_groups = set(h5.keys())
            if actual_groups != expected_groups:
                raise CompileError(
                    f"{h5_source} HDF5 root groups do not exactly match the manifest "
                    f"universe: actual={sorted(actual_groups)} expected={sorted(expected_groups)}"
                )
            metaclusters = {
                group: require_local_object(h5, group, group)
                for group in expected_groups
            }
            motif_lengths: set[int] = set()
            for group in MODISCO_GROUPS:
                patterns = expected_by_group[group]
                if not patterns:
                    continue
                metacluster = metaclusters[group]
                if not isinstance(metacluster, h5py.Group):
                    raise CompileError(f"{h5_source} HDF5 {group} must be a group")
                if set(metacluster.keys()) != set(patterns):
                    raise CompileError(
                        f"{h5_source} HDF5 motifs in {group} do not exactly match the "
                        f"manifest universe: actual={sorted(metacluster.keys())} "
                        f"expected={sorted(patterns)}"
                    )
            for group, pattern, node in ordered:
                motif = require_local_object(
                    metaclusters[group], pattern, f"{group}.{pattern}",
                )
                if not isinstance(motif, h5py.Group):
                    raise CompileError(
                        f"{h5_source} HDF5 motif {group}.{pattern} must be a group"
                    )
                dataset_names = set(motif.keys())
                if "contrib_scores" not in dataset_names:
                    raise CompileError(
                        f"{h5_source} HDF5 motif {group}.{pattern} lacks contrib_scores"
                    )
                if required_dataset not in dataset_names:
                    raise CompileError(
                        f"{h5_source} HDF5 motif {group}.{pattern} lacks "
                        f"{required_dataset} required by motif_type={manifest.motif_type!r}"
                    )
                if not dataset_names <= _COMPILED_MOTIF_DATASETS:
                    raise CompileError(
                        f"{h5_source} HDF5 motif {group}.{pattern} has unindexed dataset/group "
                        f"{sorted(dataset_names - _COMPILED_MOTIF_DATASETS)}"
                    )
                loaded: dict[str, Any] = {}
                for source_name, semantic_name in (
                    ("contrib_scores", "cwm"),
                    ("hypothetical_contribs", "hypothetical_cwm"),
                    ("sequence", "ppm"),
                ):
                    if source_name not in motif:
                        continue
                    dataset = require_local_object(
                        motif,
                        source_name,
                        f"{group}.{pattern}/{source_name}",
                    )
                    if not isinstance(dataset, h5py.Dataset):
                        raise CompileError(
                            f"{h5_source} HDF5 motif {group}.{pattern}/{source_name} "
                            "must be a dataset"
                        )
                    values = np.asarray(dataset)
                    if values.ndim != 2 or values.shape[1] != 4:
                        raise CompileError(
                            f"{h5_source} HDF5 motif {group}.{pattern}/{source_name} "
                            "must have shape (width, 4)"
                        )
                    if values.dtype != np.dtype(float):
                        raise CompileError(
                            f"{h5_source} HDF5 motif {group}.{pattern}/{source_name} "
                            "must contain compiler-emitted float64 values"
                        )
                    loaded[semantic_name] = values
                motif_lengths.add(int(loaded["cwm"].shape[0]))
                arrays[node["node_id"]] = loaded
            if len(motif_lengths) != 1:
                raise CompileError(
                    f"{h5_source} HDF5 motifs have mixed widths {sorted(motif_lengths)}"
                )
    except OSError as exc:
        raise CompileError(f"{h5_source} is not readable HDF5: {exc}") from exc

    recomputed = lexicon_semantic_hash(
        ordered,
        arrays,
        schema_version=manifest.schema_version,
        trim_threshold=manifest.trim_threshold,
        motif_type=manifest.motif_type,
        include_rc=manifest.include_rc,
        loader_backend=manifest.loader_backend,
        loader_parameters=manifest.loader_parameters,
    )
    if recomputed != manifest.lexicon_content_hash:
        raise CompileError(
            f"{manifest_source} lexicon_content_hash does not match the complete HDF5 "
            "motif universe and loader semantics"
        )
    return manifest


def _write_h5(path: Path, ordered: list[tuple[str, str, dict[str, Any]]], arrays: Any) -> None:
    import h5py
    import numpy as np

    lengths = {int(arrays[node["node_id"]]["cwm"].shape[0]) for _, _, node in ordered}
    if len(lengths) > 1:
        raise CompileError(
            f"this lexicon mixes motif lengths {sorted(lengths)}. The loader stacks every "
            "motif into one array, so a lexicon whose patterns differ in length cannot be "
            "read back at all."
        )
    with h5py.File(path, "w") as h5:
        for group, pattern_name, node in ordered:
            src = arrays[node["node_id"]]
            dest = h5.require_group(group).create_group(pattern_name)
            dest.create_dataset("contrib_scores", data=np.asarray(src["cwm"][:], dtype=float))
            if "hypothetical_cwm" in src:
                dest.create_dataset("hypothetical_contribs",
                                    data=np.asarray(src["hypothetical_cwm"][:], dtype=float))
            if "ppm" in src:
                dest.create_dataset("sequence", data=np.asarray(src["ppm"][:], dtype=float))


#: The emitted HDF5 dataset -> the registry array it must have been copied from.
#: The reverse of what :func:`_write_h5` writes. ``_MOTIF_TYPE_DATASET`` maps the
#: *loader's* vocabulary onto ours for the motif the loader will read; this maps
#: the *file's*, because :func:`operations_log` has to open the written file and
#: account for every dataset in it, not only the one the loader wants.
_EMITTED_DATASET_SOURCE = {
    "contrib_scores": "cwm",
    "hypothetical_contribs": "hypothetical_cwm",
    "sequence": "ppm",
}

#: Relative tolerance for recognising an emitted matrix AS the arithmetic mean of
#: the matrices it stands for. "Unchanged" is tested exactly -- a copy is a copy,
#: and any inexactness there is already a change -- but a mean is *computed*, and
#: the same mean computed in a different summation order differs in the last bits.
#: An operation must not escape classification (and therefore the guard) because
#: whoever wrote it summed in a different order than this classifier does.
_MEAN_MATCH_RTOL = 1e-12


def _registry_matrix(arrays: Any, registry_node: str, array_key: str) -> Any:
    """One node's registry matrix as float64, or None if the registry has none."""
    import numpy as np

    group = arrays[registry_node]
    if array_key not in group:
        return None
    return np.asarray(group[array_key][:], dtype=float)


def operations_log(tier: str,
                   ordered: list[tuple[str, str, dict[str, Any]]],
                   decisions: list[dict[str, Any]],
                   nodes_by_id: dict[str, dict[str, Any]],
                   registry_arrays: Any,
                   emitted_h5_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """What each emitted motif was built from, classified from the bytes.

    This is the input :func:`guards.no_cross_model_cwm_avg` reads, and where it
    comes from is the whole point of it. The obvious operations log is one the
    combining stage writes about itself -- ``{"op": "medoid", "group_by": [...]}``
    appended next to the code that picked the medoid. A guard fed that log audits
    a claim produced by the code it is auditing: it passes for exactly as long as
    somebody keeps the claim honest by hand, which is the same unchecked
    invariant the guard was written to replace, one level further in. It is the
    same reason the four-state missingness guard is deliberately not wired into
    ``interpret.health_report``, where the claim and the recomputation would be
    the same expression.

    So nothing here asks the writer what it did. The lexicon is opened and read
    back, and each emitted motif is classified against the arrays ``ingest``
    recorded for the nodes it stands for:

    ``copy``
        the emitted matrices are bit-identical to the registry's for this node,
        and no collapse absorbed anything into it.
    ``select_representative``
        the same, for a motif that IS a collapse's representative: the collapse
        was applied by choosing an observed member, not by combining members.
    ``mean``
        the emitted matrices are the elementwise arithmetic mean of the matrices
        of the nodes this motif stands for. This is the entry the guard rules on,
        and no stage in this release produces it -- but a stage that started to
        would produce it *here*, without editing this function, because the
        classification is of the file.

    Anything else is refused rather than filed under a label that does not fit:
    an emitted matrix that is neither one of its inputs nor their mean was
    produced by an operation this package cannot name, and an operation nobody
    can name cannot be audited for which axes it held fixed.

    ``group_by`` is the subset of ``guards.CROSS_MODEL_AXES`` on which the inputs
    all agree -- what the operation held fixed, read from the registry records
    rather than declared. It is recorded for every entry, including the ones the
    guard skips, because "this collapse crossed two models but averaged nothing"
    is worth being able to see.

    Three limits, stated because a pass here should not be read as more than it
    is:

    * If every input carries identical matrices, their mean IS one of them and
      this classifier calls it ``copy``. An averaging step over indistinguishable
      inputs is invisible in the artifact -- to this reader and to any other.
    * ``inputs`` is the representative plus the decision's other members that
      this tier did not emit. A member that this tier dropped for an unrelated
      reason (an ``EXCLUDED`` analysis tier) is therefore in the input set as
      well; that can only widen the set a mean is checked against, so it can
      cause a refusal to classify, never a silent pass.
    * The guard this feeds says nothing about a representative averaged *within*
      one model (`docs/CONSTRAINTS.md` FP-05). That clause is carried by
      ``_members_for_tier``, which refuses a representative that is not one of
      its own members.
    """
    import h5py
    import numpy as np

    emitted_ids = {node["node_id"] for _, _, node in ordered}
    absorbed_by_representative: dict[str, tuple[str, list[str]]] = {}
    for decision in decisions:
        if decision.get("decision") != Decision.COLLAPSE.value:
            continue
        representative = decision.get("representative")
        if representative not in emitted_ids:
            continue
        absorbed = [member for member in (decision.get("members") or ())
                    if member != representative
                    and member not in emitted_ids
                    and member in nodes_by_id]
        if absorbed:
            absorbed_by_representative[representative] = (
                str(decision.get("cluster_id")), absorbed,
            )

    log: list[dict[str, Any]] = []
    with h5py.File(emitted_h5_path, "r") as h5:
        for group, pattern_name, node in ordered:
            emitted_node = node["node_id"]
            tag = f"{group}.{pattern_name}"
            motif = h5[group][pattern_name]
            unknown = sorted(set(motif) - set(_EMITTED_DATASET_SOURCE))
            if unknown:
                raise CompileError(
                    f"{tier}: emitted motif {tag} carries dataset(s) {unknown} that no "
                    "registry array corresponds to, so what produced them cannot be "
                    "classified and no operations log can account for this lexicon."
                )
            emitted = {name: np.asarray(motif[name][:], dtype=float) for name in motif}
            cluster_id, absorbed = absorbed_by_representative.get(emitted_node, (None, []))
            inputs = [emitted_node, *absorbed]

            unchanged = True
            for name, values in emitted.items():
                own = _registry_matrix(registry_arrays, emitted_node,
                                       _EMITTED_DATASET_SOURCE[name])
                if own is None or not np.array_equal(values, own):
                    unchanged = False
                    break
            operation: str | None = None
            if unchanged:
                operation = "select_representative" if absorbed else "copy"
            else:
                parts_per_dataset = {
                    name: [_registry_matrix(registry_arrays, source,
                                            _EMITTED_DATASET_SOURCE[name])
                           for source in inputs]
                    for name in emitted
                }
                if all(
                    all(part is not None for part in parts)
                    and len({part.shape for part in parts}) == 1
                    and np.allclose(emitted[name], sum(parts[1:], parts[0]) / len(parts),
                                    rtol=_MEAN_MATCH_RTOL, atol=0.0)
                    for name, parts in parts_per_dataset.items()
                ):
                    operation = "mean"
            if operation is None:
                raise CompileError(
                    f"{tier}: emitted motif {tag} ({emitted_node}) is neither the "
                    f"registry's own matrices for that node nor the elementwise mean of "
                    f"{sorted(inputs)}. This package cannot name the operation that "
                    "produced it, and an operation it cannot name cannot be audited "
                    "for the axes it held fixed (guards.no_cross_model_cwm_avg). "
                    "Refusing to publish a lexicon whose contents are unaccounted for."
                )

            log.append({
                "tier": tier,
                "pattern_tag": tag,
                "op": operation,
                "emitted_node": emitted_node,
                "cluster_id": cluster_id,
                "inputs": inputs,
                "group_by": [axis for axis in guards.CROSS_MODEL_AXES
                             if len({nodes_by_id[source].get(axis)
                                     for source in inputs}) == 1],
            })
    return log


def _compare(tier: str, index: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """State, for every other tier, what this contrast does and does not vary."""
    def split(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
        pos = {r["node_id"] for r in rows if r["metacluster"] == GROUP_METACLUSTER["pos_patterns"]}
        neg = {r["node_id"] for r in rows if r["metacluster"] == GROUP_METACLUSTER["neg_patterns"]}
        return pos, neg

    mine_pos, mine_neg = split(index[tier])
    out: dict[str, dict[str, Any]] = {}
    for other, rows in index.items():
        if other == tier:
            continue
        other_pos, other_neg = split(rows)
        out[other] = {
            "positive_sets_identical": mine_pos == other_pos,
            "negative_sets_identical": mine_neg == other_neg,
            "n_positive_here": len(mine_pos), "n_positive_there": len(other_pos),
            "n_negative_here": len(mine_neg), "n_negative_there": len(other_neg),
            "only_here": sorted((mine_pos | mine_neg) - (other_pos | other_neg)),
            "only_there": sorted((other_pos | other_neg) - (mine_pos | mine_neg)),
        }
        if out[other]["positive_sets_identical"]:
            out[other]["warning"] = (
                f"the {tier} vs {other} contrast does not vary the positive lexicon; "
                "any sensitivity read from it concerns the negative half only"
            )
    return out


#: The loader-side defaults ``load_back`` applies for any key ``loader_parameters``
#: leaves unset. Kept as data, in one place, so ``_resolve_loader_parameters`` is
#: the *only* place a fallback is filled in -- not duplicated between the hasher
#: and the reader, which is what let ``None`` and ``{}`` (behaviourally identical:
#: both fall back to this same default) content-address differently (round-1
#: review finding 1).
_LOADER_PARAMETER_DEFAULTS: dict[str, Any] = {"motif_lambda_default": 0.7}


def _resolve_loader_parameters(loader_parameters: dict[str, Any] | None) -> dict[str, Any]:
    """The fully-resolved, effective loader parameters -- what ``load_back`` will use.

    Called once, before the result is hashed, stored on the manifest, or handed to
    ``load_back``, so that every spelling which resolves to the same effective
    configuration (``None``, ``{}``, an explicit
    ``{"motif_lambda_default": 0.7}``) always produces the *same* dict -- and
    therefore the same ``lexicon_content_hash`` -- while a spelling that actually
    changes a value (``{"motif_lambda_default": 0.5}``) still changes it.
    """
    return {**_LOADER_PARAMETER_DEFAULTS, **(loader_parameters or {})}


# --------------------------------------------------------------------------- #
# Reading back, with the real loader
# --------------------------------------------------------------------------- #
#: What the backend has called "the trim threshold applied to every motif that
#: names no override of its own", newest spelling first. finemo 0.30 called it
#: ``trim_threshold``; 0.40 renamed it to ``trim_threshold_default`` and added the
#: per-motif ``trim_coords`` / ``trim_thresholds`` this package never sets.
#:
#: Resolved from the installed signature rather than assumed, because assuming it
#: is how the rename reached users as a bare ``TypeError``: the two spellings are
#: one letter apart, both plausible, and nothing at the call site could tell them
#: apart until the call itself failed.
_LOADER_TRIM_THRESHOLD_ALIASES = ("trim_threshold_default", "trim_threshold")

#: Loader parameters this package deliberately leaves unset, because a compiled
#: lexicon declares no *per-motif* overrides: every motif in it is read under the
#: single manifest-recorded configuration. They carry no default in the backend's
#: signature, so "leave unset" has to be spelled as an explicit ``None`` -- and
#: only for the ones the installed backend actually has, since the set has grown
#: across releases.
_LOADER_UNSET_PARAMETERS = (
    "trim_coords", "trim_thresholds", "motifs_include", "motif_name_map",
    "motif_lambdas",
)


def _loader_call_kwargs(loader: Callable[..., Any], *, trim_threshold: float,
                        motif_type: str, include_rc: bool,
                        extra: dict[str, Any]) -> dict[str, Any]:
    """Bind this package's loader settings to the *installed* backend's signature.

    Two things must never happen quietly here, and each gets its own refusal.

    A setting we mean to pass may not exist under that name any more (the
    ``trim_threshold`` -> ``trim_threshold_default`` rename). Passing it anyway
    raises ``TypeError`` from inside the backend, which reads as a bug in this
    package rather than as a version mismatch.

    A parameter the backend *requires* may be one this package has never heard
    of. That one is worse: the tempting repair is to let the backend default it,
    but these parameters have no defaults precisely because they change what the
    loader returns, and a round trip that verified an order produced under
    settings the manifest does not record has verified the wrong lexicon. So an
    unrecognised required parameter is a refusal, naming it, not a guess.
    """
    try:
        params = inspect.signature(loader).parameters
    except (TypeError, ValueError) as exc:                      # pragma: no cover
        raise BackendIncompatible(
            f"the installed loader {loader!r} has no inspectable signature ({exc}), so the "
            "settings this lexicon's manifest records cannot be bound to it"
        ) from exc
    accepts_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    threshold_param = next(
        (name for name in _LOADER_TRIM_THRESHOLD_ALIASES if name in params), None)
    if threshold_param is None:
        raise BackendIncompatible(
            "the installed loader takes no trim-threshold argument under any name this "
            f"package knows ({', '.join(_LOADER_TRIM_THRESHOLD_ALIASES)}); its parameters "
            f"are {list(params)}. The manifest records a trim_threshold, so reading back "
            "without passing it would verify a lexicon nobody compiled."
        )
    kwargs: dict[str, Any] = {
        threshold_param: trim_threshold,
        "motif_type": motif_type,
        "include_rc": include_rc,
        **extra,
    }
    kwargs.update({name: None for name in _LOADER_UNSET_PARAMETERS if name in params})
    if not accepts_var_keyword:
        unknown = sorted(name for name in kwargs if name not in params)
        if unknown:
            raise BackendIncompatible(
                f"the installed loader does not accept {unknown}, which this lexicon's "
                f"manifest records as loader settings; its parameters are {list(params)}"
            )
    # The first parameter is the path, which is passed positionally; everything
    # else without a default has to come from `kwargs` or we do not know what the
    # backend would do with it.
    positional = list(params)[:1]
    unsatisfied = sorted(
        name for name, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                       inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs and name not in positional
    )
    if unsatisfied:
        raise BackendIncompatible(
            f"the installed loader requires {unsatisfied}, which this package does not "
            "set and whose effect on the returned motif order is unknown to it. Letting "
            "the backend choose would make the round trip verify an order the manifest "
            "does not describe."
        )
    return kwargs


def load_back(h5_path: str | os.PathLike[str], trim_threshold: float = 0.3,
             motif_type: str = "cwm", include_rc: bool = False,
             loader_parameters: dict[str, Any] | None = None) -> list[str]:
    """Read a compiled lexicon with the **real** hit-caller loader; return its names.

    Verification is behavioural, not structural: asserting that the file has the
    groups we just wrote proves only that we can read our own output. The three
    named arguments plus ``loader_parameters`` (forwarded as extra keyword
    arguments to the real loader, e.g. ``motif_lambda_default``) are exactly the
    settings a lexicon's manifest records; a caller that wants to verify *this*
    lexicon must pass *its* settings, not whatever this function defaults to.
    """
    try:
        from finemo.data_io import load_modisco_motifs
    except ImportError as exc:
        raise BackendMissing(
            "round-trip verification needs the finemo backend (pip install finemo). "
            "Without it the H5 is written but never read back by anything but this package."
        ) from exc
    extra = _resolve_loader_parameters(loader_parameters)
    kwargs = _loader_call_kwargs(
        load_modisco_motifs, trim_threshold=trim_threshold, motif_type=motif_type,
        include_rc=include_rc, extra=extra,
    )
    _motifs_df, _cwms, _trim_masks, names = load_modisco_motifs(str(h5_path), **kwargs)
    return [str(n) for n in names]


#: The one motif the capability probe compiles. Deterministic, tiny, and shaped
#: like a real motif rather than noise: a flat or random array can trim to nothing
#: under the loader's own trimming, and a probe that failed because its input was
#: degenerate would report a working backend as broken.
_PROBE_MOTIF_LENGTH = 20
_PROBE_CORE = slice(8, 12)


def probe_backend(trim_threshold: float = 0.3, motif_type: str = "cwm",
                  include_rc: bool = False,
                  loader_parameters: dict[str, Any] | None = None) -> str:
    """Compile one lexicon and read it back. Returns what that proved, in a sentence.

    This is the capability behind a ``VERIFIED`` backend in
    :mod:`motifmultiverse.status`, and it exists because *importable* and *usable*
    are different claims about this backend specifically. :func:`load_back`
    already separates them at call time -- :class:`BackendMissing` when the import
    fails, :class:`BackendIncompatible` when it imports but will not accept the
    call this package makes, which is not hypothetical: finemo 0.40 renamed the
    trim-threshold argument, and every installed-and-imported backend of that
    release raised ``TypeError`` from inside :func:`compile_lexicons`. A status
    line that said ``VERIFIED`` because ``import finemo`` returned would have been
    reporting green on exactly those machines.

    So the probe performs the smallest complete instance of the thing being
    claimed: write an H5 with this module's own writer, hand it to the real
    loader, and compare the names that come back with
    ``guards.index_order_matches_loader`` -- the same comparison
    :func:`verify_roundtrip` makes on a real lexicon. It raises
    :class:`BackendMissing` / :class:`BackendIncompatible` when the backend cannot
    do it, and :class:`CompileError` when it can be called but returns something
    other than the lexicon it was given.
    """
    import tempfile

    import numpy as np

    cwm = np.zeros((_PROBE_MOTIF_LENGTH, 4), dtype=float)
    cwm[_PROBE_CORE, 0] = 1.0
    ppm = np.full((_PROBE_MOTIF_LENGTH, 4), 0.25, dtype=float)
    ppm[_PROBE_CORE] = [0.85, 0.05, 0.05, 0.05]
    arrays = {"probe_node": {"cwm": cwm, "hypothetical_cwm": cwm, "ppm": ppm}}
    ordered = [("pos_patterns", "pattern_0", {"node_id": "probe_node"})]
    expected = ["pos_patterns.pattern_0"]

    with tempfile.TemporaryDirectory(prefix="motifmultiverse-probe-") as tmp:
        h5_path = Path(tmp) / "probe.h5"
        # This module's real writer, not an H5 hand-rolled for the probe: a probe
        # that writes its own file format verifies the backend against something
        # this package does not actually produce.
        _write_h5(h5_path, ordered, arrays)
        names = load_back(h5_path, trim_threshold=trim_threshold, motif_type=motif_type,
                          include_rc=include_rc, loader_parameters=loader_parameters)
    result = guards.index_order_matches_loader(expected, names)
    if not result.passed:
        raise CompileError(
            f"the installed loader read a one-motif lexicon back as {names}, not {expected} "
            f"({result.detail}). It is callable, so this is not a version-binding failure: "
            "what it returns does not correspond to what this package writes, and no lexicon "
            "compiled here could be verified against it."
        )
    return (f"compiled a one-motif lexicon and read it back with the real loader; "
            f"the loader returned {names}, matching the order the manifest records")


def verify_roundtrip(h5_path: str | os.PathLike[str],
                     manifest: LexiconManifest) -> guards.GuardResult:
    """Write-then-read: the loader's order must be the manifest's order, by name.

    Reads back under the manifest's *own* loader configuration -- not this
    module's defaults -- because that configuration is exactly what
    ``lexicon_content_hash`` asserts the lexicon's identity depends on.
    """
    names = load_back(
        h5_path, trim_threshold=manifest.trim_threshold, motif_type=manifest.motif_type,
        include_rc=manifest.include_rc, loader_parameters=manifest.loader_parameters,
    )
    return guards.index_order_matches_loader(manifest.pattern_order, names)


# --------------------------------------------------------------------------- #
# The whole step
# --------------------------------------------------------------------------- #
def compile_lexicons(registry_dir: str | os.PathLike[str], out_dir: str | os.PathLike[str],
                     decisions_path: str | os.PathLike[str] | None = None,
                     tiers: tuple[str, ...] = TIERS,
                     verify: str = "auto",
                     seed: int | None = None,
                     trim_threshold: float = 0.3,
                     motif_type: str = "cwm",
                     include_rc: bool = False,
                     loader_backend: str = "finemo",
                     loader_parameters: dict[str, Any] | None = None,
                     ) -> dict[str, LexiconManifest]:
    """Compile one lexicon per tier, each with its manifest, in loader order.

    ``trim_threshold``, ``motif_type``, ``include_rc``, ``loader_backend`` and
    ``loader_parameters`` are the loader-affecting settings that used to be
    hard-coded inside :func:`load_back` (``trim_threshold=0.3, motif_type="cwm",
    motif_lambda_default=0.7, include_rc=False``). Two lexicons compiled with
    different values here load differently under the real loader and must not
    share a ``lexicon_content_hash`` -- so these are threaded into the manifest
    and the hash, and into round-trip verification, instead of staying implicit.
    Defaults reproduce exactly what was previously hard-coded, so existing callers
    are unaffected.
    """
    unknown = [t for t in tiers if t not in TIERS]
    if unknown:
        raise CompileError(f"unknown tiers {unknown}; expected a subset of {list(TIERS)}")
    _required_dataset, required_semantic_name = _motif_type_dataset(
        motif_type, what="compile",
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Lexicon artifacts are built here and moved into `out` only once every tier
    # has been written and verified. A tier is refused after earlier tiers are
    # already on disk -- mixed motif lengths in `expanded`, a round-trip the
    # backend cannot perform under `--verify-roundtrip require` -- and the
    # directory that was left behind is not merely untidy: `validate` binds a
    # lexicon set by globbing `*.manifest.json`, so a compile that refused two of
    # three tiers read downstream as a perfectly valid one-tier lexicon, with a
    # `lexicon_identity` computed over the surviving fragment. Staging inside
    # `out` keeps the move on one filesystem; the leading dot keeps a crashed run's
    # leftovers out of that glob, and the next run clears them.
    staging = out / ".compile-staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    # Resolved once, to its effective form, before it is hashed or stored: any
    # spelling that reads back identically (``None``, ``{}``, an explicit
    # ``{"motif_lambda_default": 0.7}``) must produce the same manifest and the
    # same lexicon_content_hash. `load_back` resolves the same way, from the same
    # function, so the two can never drift apart again.
    loader_parameters = _resolve_loader_parameters(loader_parameters)

    prov = record("compile", seed=seed)
    # Bound to `out`, not to the staging directory: the outcome of a round trip is
    # a fact about the run whether or not the tier it verified is ever published,
    # and staging is deleted on refusal. Same reason the provenance record above
    # goes to `out` before anything is compiled.
    guard_log = GuardLog("compile", out)
    meta, nodes, arrays = load_registry(registry_dir)
    try:
        payload: dict[str, Any] = {}
        # Read the bytes, but do not decode or parse them yet. A decisions file
        # that is not UTF-8 JSON is a refusal like any other, and it has to reach
        # the same two rules the refusals below already obey: provenance is
        # written first (T-09), and the caller sees `CompileError` -- exit 4, the
        # rule that declined -- rather than a `JSONDecodeError` traceback and an
        # exit code the CLI never documented.
        decisions_bytes: bytes | None = None
        if decisions_path is not None:
            prov.add_input(decisions_path)
            decisions_bytes = Path(decisions_path).read_bytes()
        # Provenance is written before the decisions payload is validated, same as
        # every other refusal below (a tier with no motifs, mixed motif lengths):
        # a rejected compile still leaves a record of what was attempted (T-09).
        prov.write(out)
        if decisions_bytes is not None:
            try:
                payload = json.loads(decisions_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CompileError(
                    f"{decisions_path} is not valid JSON: {exc}"
                ) from exc
        try:
            bundle = (
                DecisionBundle.from_adjudication_artifact(payload)
                if decisions_path is not None
                else DecisionBundle.from_dict(payload)
            )
        except SchemaError as exc:
            raise CompileError(str(exc)) from exc
        known_ids = {n["node_id"] for n in nodes}
        _assert_members_known(bundle.decisions, known_ids)
        _assert_tier_overrides_known(bundle.tiers, known_ids)
        # `_members_for_tier` / `triggers_by_cluster` predate `DecisionBundle` and
        # still work over plain dicts; converting once here keeps them unchanged.
        decisions = [asdict(d) for d in bundle.decisions]
        overrides = bundle.tiers

        tiered = _apply_tiers(nodes, overrides)
        index: dict[str, list[dict[str, Any]]] = {}
        ordered_by_tier: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        for tier in tiers:
            members = _members_for_tier(tiered, decisions, tier)
            if not members:
                raise CompileError(f"tier {tier!r} would contain no motifs")
            ordered = loader_order(members)
            ordered_by_tier[tier] = ordered
            index[tier] = [
                {"index": i, "pattern_tag": f"{g}.{p}", "node_id": n["node_id"],
                 "variant_id": n["variant_id"], "metacluster": n["metacluster"]}
                for i, (g, p, n) in enumerate(ordered)
            ]

        missing_loader_arrays = sorted({
            node["node_id"]
            for ordered in ordered_by_tier.values()
            for _, _, node in ordered
            if required_semantic_name not in arrays[node["node_id"]]
        })
        if missing_loader_arrays:
            raise CompileError(
                f"motif_type={motif_type!r} requires {required_semantic_name!r} "
                "for every emitted motif; missing for "
                f"{missing_loader_arrays}"
            )

        manifests: dict[str, LexiconManifest] = {}
        nodes_by_id = {node["node_id"]: node for node in nodes}
        operations: list[dict[str, Any]] = []
        for tier in tiers:
            ordered = ordered_by_tier[tier]
            h5_path = staging / f"{tier}.h5"
            _write_h5(h5_path, ordered, arrays)
            # Read back what was just written and classify it against the
            # registry. Deliberately not a note this loop makes about its own
            # intentions: see `operations_log`.
            operations += operations_log(
                tier, ordered, decisions, nodes_by_id, arrays, h5_path)
            content_hash = lexicon_semantic_hash(
                ordered, arrays,
                schema_version=LEXICON_MANIFEST_SCHEMA_VERSION,
                trim_threshold=trim_threshold, motif_type=motif_type,
                include_rc=include_rc, loader_backend=loader_backend,
                loader_parameters=loader_parameters,
            )
            manifest = LexiconManifest(
                tier=tier,
                lexicon_content_hash=content_hash,
                n_motifs=len(ordered),
                pattern_order=[f"{g}.{p}" for g, p, _ in ordered],
                node_ids=[n["node_id"] for _, _, n in ordered],
                schema_version=LEXICON_MANIFEST_SCHEMA_VERSION,
                trim_threshold=trim_threshold,
                motif_type=motif_type,
                include_rc=include_rc,
                loader_backend=loader_backend,
                # A fresh copy per tier: the three manifests must not share one
                # dict object. Nothing in this package mutates it in place today,
                # but a future in-place mutation on one tier's manifest must not
                # silently corrupt the other two.
                loader_parameters=dict(loader_parameters),
                comparisons=_compare(tier, index),
                source_registry=str(Path(registry_dir).name),
                sensitivity_triggers={k: v for k, v in triggers_by_cluster(decisions).items() if v},
            )
            (staging / f"{tier}.manifest.json").write_text(
                json.dumps({**asdict(manifest), "index": index[tier],
                            "project": meta.project,
                            "cross_model_claims_restricted": meta.cross_model_claims_restricted},
                           indent=2, sort_keys=True))
            manifests[tier] = manifest

            if verify == "skip":
                continue
            try:
                guard_log.record(
                    verify_roundtrip(h5_path, manifest),
                    subject=(
                        f"tier {tier!r}: the pattern order recorded in its manifest "
                        f"(lexicon_content_hash={manifest.lexicon_content_hash}) against the "
                        f"order the real {loader_backend} loader returns for {tier}.h5"
                    ),
                ).raise_if_failed()
            except BackendMissing:
                # Nothing is recorded for this tier: the loader could not be
                # called, so no guard ran and no outcome exists. Writing a
                # "not verified" entry here would put a guard id in the record
                # of a run where that guard never saw the lexicon, which is the
                # absence-as-a-result failure the log exists to avoid. The
                # backend's absence is already `status`'s UNVERIFIED and the
                # CLI's "round-trip: NOT verified" line.
                if verify == "require":
                    raise

        # Every emitted motif in every tier, and what it was built from. The
        # guard runs before publication, so a lexicon holding a CWM averaged
        # across models is refused rather than written and annotated -- and the
        # log itself is shipped, so a reader can re-run the guard against the
        # lexicon without this package's cooperation.
        guard_log.record(
            guards.no_cross_model_cwm_avg(operations),
            subject=(
                f"every motif emitted into tiers {list(tiers)}, each classified "
                "against the registry arrays ingest recorded for the nodes it "
                "stands for -- combination performed at or before ingest is not "
                "visible to this classification"
            ),
        ).raise_if_failed()
        (staging / "combination_operations.json").write_text(
            json.dumps(operations, indent=2, sort_keys=True))

        rows = ["\t".join(["tier", "index", "pattern_tag", "node_id", "variant_id",
                           "metacluster", "lexicon_content_hash"])]
        for tier in tiers:
            for row in index[tier]:
                rows.append("\t".join([tier, str(row["index"]), row["pattern_tag"],
                                       row["node_id"], row["variant_id"], row["metacluster"],
                                       manifests[tier].lexicon_content_hash]))
        (staging / "manifest.tsv").write_text("\n".join(rows) + "\n")

        # Publication. Every refusal this function can raise has already had its
        # chance, so nothing above this line was ever visible under `out`; from
        # here the set appears tier by tier, each file by an atomic rename.
        for name in sorted(path.name for path in staging.iterdir()):
            os.replace(staging / name, out / name)
    finally:
        arrays.close()
        shutil.rmtree(staging, ignore_errors=True)
    return manifests
