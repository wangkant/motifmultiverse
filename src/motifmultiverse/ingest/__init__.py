"""Normalise TF-MoDISco outputs into a registry, with provenance attached first.

Scope, deliberately narrow: read the discovery HDF5s named by a project config,
extract each pattern's CWM, hypothetical CWM, PPM and seqlet count, and write a
registry in which every node carries the six field groups of ``docs/DATA_MODEL.md``
and every input carries its checksum.

Two things here are not conveniences:

**Three ways for a metacluster to be absent.** ``group_absent`` (no group in the
file), ``group_empty`` (a group with no patterns) and ``not_searched`` (this run
never looked) are recorded separately and never collapsed. In the reference
implementation four discovery leaves had no negative group *at all*, and reading
that as "no repressive motifs" is the discovery-stage form of ``BA-01``.

**Identifiers are opaque.** ``denovo_pattern_id`` and friends are join tokens.
Nothing here parses a number out of one -- a reference-implementation key read
``CBP_2048_...`` while the real input width was 2114, and that was harmless only
because no code ever read the digits. ``union_id`` is declared in the config for
the same reason: deriving it from a filename would be the same mistake.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from motifmultiverse.provenance import ProvenanceRecord, record, sha256_file
from motifmultiverse.schema import (
    MISSING_SENTINEL,
    REGISTRY_SCHEMA_VERSION,
    UNION_ID_RE,
    AnalysisConfig,
    MetaclusterState,
    MotifNode,
    RegistryMetadata,
    SchemaError,
    Tier,
)

__all__ = [
    "IngestError", "MODISCO_GROUPS", "GROUP_METACLUSTER", "DEFAULT_TRIM_THRESHOLD",
    "read_project", "ingest_project", "load_registry",
]


class IngestError(ValueError):
    """A project config or a discovery file cannot be ingested as declared."""


#: The loader contract: these names, in this order. See ``compile``.
MODISCO_GROUPS = ("pos_patterns", "neg_patterns")
GROUP_METACLUSTER = {"pos_patterns": "pos", "neg_patterns": "neg"}
DEFAULT_TRIM_THRESHOLD = 0.3

_REQUIRED_ANALYSIS_FIELDS = ("id", "model", "readout", "union_id", "context", "modisco_h5")


def read_project(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a project config. YAML if PyYAML is present, otherwise JSON."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise IngestError(
                "reading a YAML project config needs PyYAML; install it or supply JSON"
            ) from exc
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise IngestError(f"{p}: a project config must be a mapping")
    return cfg


def validate_project(cfg: dict[str, Any]) -> AnalysisConfig:
    """Every semantic attribute of an analysis is an explicit field, or this fails."""
    analyses = cfg.get("analyses") or []
    if not analyses:
        raise IngestError("the project declares no analyses")
    for a in analyses:
        missing = [f for f in _REQUIRED_ANALYSIS_FIELDS if not a.get(f)]
        if missing:
            raise IngestError(f"analysis {a.get('id', '<unnamed>')!r} is missing {missing}")
        if not UNION_ID_RE.match(str(a["union_id"])):
            raise IngestError(
                f"analysis {a['id']!r}: union_id {a['union_id']!r} must be alphanumeric. "
                "It is declared, never derived from a filename or an analysis id -- "
                "deriving it would be parsing semantics out of an identifier (BA-11)."
            )
    return AnalysisConfig(
        project=str(cfg.get("project") or "unnamed-project"),
        analyses=list(analyses),
        peak_universe_id=str(cfg.get("peak_universe_id") or MISSING_SENTINEL),
        # Read, not ignored. `or None` is wrong here: an explicitly empty list is
        # the claim "checked, nothing shared", which is not the same as the key
        # being absent, so the key's presence decides and its value is passed
        # through untouched. AnalysisConfig refuses a group naming an unknown id,
        # naming one analysis twice, or naming fewer than two analyses.
        shared_attribution_groups=cfg.get("shared_attribution_groups"),
    )


#: The root group of an *original* (pre-``tfmodisco-lite``) TF-MoDISco HDF5. Its
#: patterns live at ``metacluster_idx_to_submetacluster_results/metacluster_N/
#: seqlets_to_patterns_result/patterns/pattern_N`` -- a layout this reader does
#: not read. Recognised by name only, to refuse the file; nothing is parsed out of
#: it and no pattern is read through it.
PRE_LITE_ROOT_GROUP = "metacluster_idx_to_submetacluster_results"


def assert_readable_layout(h5: Any, analysis_id: str, h5_path: Any) -> None:
    """Refuse a discovery file whose patterns this reader cannot see at all.

    ``group_absent`` is a claim, not a shrug: ``docs/DATA_MODEL.md`` defines it as
    *discovery ran and the group never formed*, which is evidence about the
    admission gate. An original-TF-MoDISco file has neither ``pos_patterns`` nor
    ``neg_patterns`` at its root, so reading it produced exactly that claim twice
    over -- and exit 0, and an empty registry -- for a file that may hold dozens of
    patterns under the older layout. That is a measurement invented from the
    reader's own blindness, the discovery-stage form of ``BA-01`` the three
    absences exist to prevent. The three absences are unchanged; what is refused
    is answering with any of them about a file that was never read.
    """
    if PRE_LITE_ROOT_GROUP in h5:
        raise IngestError(
            f"{analysis_id}: {h5_path} is an original TF-MoDISco output (it carries "
            f"{PRE_LITE_ROOT_GROUP!r}), whose patterns this reader does not read. "
            "It cannot be recorded as two group_absent metaclusters: group_absent "
            "claims discovery ran and the group never formed, and nothing here "
            "observed that. Convert it to tfmodisco-lite layout (pos_patterns / "
            "neg_patterns) and re-declare the analysis."
        )


def group_state(h5: Any, group: str, searched: bool) -> MetaclusterState:
    """The three absences, kept apart (V-08)."""
    if not searched:
        return MetaclusterState.NOT_SEARCHED
    if group not in h5:
        return MetaclusterState.GROUP_ABSENT
    if len(h5[group].keys()) == 0:
        return MetaclusterState.GROUP_EMPTY
    return MetaclusterState.PRESENT


def _trim(cwm: Any, threshold: float) -> tuple[int, int]:
    """Trimmed core as a half-open window, by the rule the hit caller uses."""
    per_pos = [max(abs(float(v)) for v in row) for row in cwm]
    if not per_pos:
        return (0, 0)
    cutoff = max(per_pos) * threshold
    keep = [i for i, v in enumerate(per_pos) if v >= cutoff]
    return (keep[0], keep[-1] + 1) if keep else (0, len(per_pos))


def _core_ic(ppm: Any, start: int, end: int) -> float | None:
    """Information content summed over the trimmed core, or None if there is no PPM."""
    if ppm is None:
        return None
    total = 0.0
    for row in list(ppm)[start:end]:
        probs = [float(v) for v in row]
        s = math.fsum(probs)
        if s <= 0:
            continue
        probs = [v / s for v in probs]
        entropy = -math.fsum(p * math.log2(p) for p in probs if p > 0)
        total += 2.0 - entropy
    return total


def _seqlet_count(pattern: Any) -> int | None:
    seqlets = pattern.get("seqlets")
    if seqlets is None:
        return None
    if "n_seqlets" in seqlets:
        value = seqlets["n_seqlets"][()]
        return int(value.item() if hasattr(value, "item") else value)
    for key in ("start", "example_idx"):
        if key in seqlets:
            return int(len(seqlets[key]))
    return None


def ingest_project(project_path: str | os.PathLike[str], out_dir: str | os.PathLike[str],
                   trim_threshold: float = DEFAULT_TRIM_THRESHOLD,
                   seed: int | None = None) -> tuple[RegistryMetadata, list[MotifNode]]:
    """Read every declared discovery output into one registry."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise IngestError("ingest needs h5py and numpy") from exc

    cfg = validate_project(read_project(project_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prov: ProvenanceRecord = record("ingest", seed=seed)
    prov.add_input(project_path)

    nodes: list[MotifNode] = []
    states: dict[str, dict[str, str]] = {}
    arrays: dict[str, dict[str, Any]] = {}
    counters_by_union: dict[str, int] = {}

    for analysis in cfg.analyses:
        analysis_id = str(analysis["id"])
        h5_path = Path(str(analysis["modisco_h5"]))
        if not h5_path.exists():
            raise IngestError(f"{analysis_id}: {h5_path} does not exist")
        # Keyed by analysis_id, not by basename: `modisco.h5` is the standard
        # TF-MoDISco filename, so a project with several discovery runs recorded
        # one checksum under one key and attributed it to whichever file was read
        # last. The config's own id is unique by construction and leaks no path.
        prov.add_input(h5_path, key=f"{analysis_id}:{h5_path.name}")
        # Hashed once per file, not once per pattern. The digest goes on to every
        # node this file produces, and re-reading a 17 MB discovery output for each
        # of its patterns re-derived a value that cannot have changed mid-loop.
        h5_digest = sha256_file(h5_path)
        # An analysis may declare which metaclusters it looked for at all.
        declared = analysis.get("search_metaclusters") or {}
        searched = {g: bool(declared.get(g, True)) for g in MODISCO_GROUPS}

        with h5py.File(h5_path, "r") as h5:
            assert_readable_layout(h5, analysis_id, h5_path)
            states[analysis_id] = {g: group_state(h5, g, searched[g]).value
                                   for g in MODISCO_GROUPS}
            for group in MODISCO_GROUPS:
                if states[analysis_id][group] != MetaclusterState.PRESENT.value:
                    continue
                for pattern_name in sorted(h5[group].keys()):
                    pattern = h5[group][pattern_name]
                    cwm = np.asarray(pattern["contrib_scores"][:], dtype=float)
                    hcwm = (np.asarray(pattern["hypothetical_contribs"][:], dtype=float)
                            if "hypothetical_contribs" in pattern else None)
                    ppm = (np.asarray(pattern["sequence"][:], dtype=float)
                           if "sequence" in pattern else None)
                    start, end = _trim(cwm, trim_threshold)
                    node_id = f"{analysis_id}::{group}.{pattern_name}"
                    union_id = str(analysis["union_id"])
                    counter = counters_by_union.get(union_id, 0)
                    nodes.append(MotifNode(
                        node_id=node_id,
                        model=str(analysis["model"]),
                        readout=str(analysis["readout"]),
                        context=str(analysis["context"]),
                        metacluster=GROUP_METACLUSTER[group],
                        # An opaque join token. Nothing parses digits out of it (V-09).
                        denovo_pattern_id=node_id,
                        # The middle segment is a placeholder, not a claim: family_id
                        # is the authoritative field, and annotate is unspecified.
                        variant_id=f"{union_id}_UNASSIGNED_{counter:02d}",
                        # ...and this is where that placeholder-ness is RECORDED,
                        # rather than left legible only in the value's spelling
                        # (which V-09 forbids a consumer from reading).
                        variant_assignment_source=MISSING_SENTINEL,
                        family_id=MISSING_SENTINEL,
                        motif_length=int(cwm.shape[0]),
                        trimmed_core=[start, end],
                        seqlet_count=_seqlet_count(pattern),
                        core_ic=_core_ic(ppm, start, end),
                        # Completeness is the observed contribution-bearing core
                        # span as a fraction of the observed motif span. It is
                        # explicit registry data, not motif length under a new
                        # name and not a value reconstructed from an identifier.
                        motif_completeness=(
                            (end - start) / int(cwm.shape[0])
                            if int(cwm.shape[0]) > 0
                            else None
                        ),
                        discovery_tier=Tier.CORE,
                        analysis_tier=Tier.CORE,
                        provenance={"analysis_id": analysis_id,
                                    "modisco_h5_sha256": h5_digest,
                                    "trim_threshold": trim_threshold},
                    ))
                    arrays[node_id] = {"cwm": cwm, "hypothetical_cwm": hcwm, "ppm": ppm}
                    counters_by_union[union_id] = counter + 1

    from motifmultiverse.guards import variant_id_unique

    identity_guard = variant_id_unique(nodes)
    if not identity_guard.passed:
        raise IngestError(identity_guard.detail)

    meta = RegistryMetadata(
        project=cfg.project,
        peak_universe_id=cfg.peak_universe_id,
        analyses=[dict(a) for a in cfg.analyses],
        n_models=cfg.n_models,
        cross_model_claims_restricted=cfg.cross_model_claims_restricted,
        metacluster_states=states,
        trim_threshold=trim_threshold,
        schema_version=REGISTRY_SCHEMA_VERSION,
        shared_attribution_groups=cfg.shared_attribution_groups,
        n_attribution_sources=cfg.n_attribution_sources,
    )
    _write_registry(out, meta, nodes, arrays)
    prov.write(out)
    return meta, nodes


def _write_registry(out: Path, meta: RegistryMetadata, nodes: list[MotifNode],
                    arrays: dict[str, dict[str, Any]]) -> None:
    import h5py

    payload = {
        "registry_metadata": asdict(meta),
        "nodes": [{k: v for k, v in n.to_dict().items()
                   if k not in {"cwm", "hypothetical_cwm", "ppm"}} for n in nodes],
    }
    (out / "registry.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    with h5py.File(out / "arrays.h5", "w") as h5:
        for node_id, mats in arrays.items():
            grp = h5.create_group(node_id)
            for name, mat in mats.items():
                if mat is not None:
                    grp.create_dataset(name, data=mat)


def load_registry(registry_dir: str | os.PathLike[str]) -> tuple[RegistryMetadata, list[dict[str, Any]], Any]:
    """Read a registry back: metadata, node records, and an open arrays handle."""
    import h5py

    d = Path(registry_dir)
    blob = json.loads((d / "registry.json").read_text())
    try:
        if "schema_version" not in blob["registry_metadata"]:
            raise SchemaError(
                f"{d}/registry.json registry_metadata is missing required schema_version"
            )
        meta = RegistryMetadata(**blob["registry_metadata"])
        nodes = [MotifNode(**node) for node in blob["nodes"]]
        from motifmultiverse.guards import variant_id_unique

        identity_guard = variant_id_unique(nodes)
        if not identity_guard.passed:
            raise SchemaError(identity_guard.detail)
    except (TypeError, KeyError, SchemaError) as exc:
        raise SchemaError(f"{d}/registry.json is not a registry: {exc}") from exc
    return meta, blob["nodes"], h5py.File(d / "arrays.h5", "r")
