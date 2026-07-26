"""Command-line interface (T-08).

Nine subcommands are wired with their real arguments and real ``--help``.
``ingest``, ``compile`` and ``interpret`` are implemented; the other six raise
:class:`NotImplementedError` naming the module README that specifies them.

Every subcommand writes a provenance record before it raises (T-09), because a
skeleton that defers provenance is a tool that never has it.

Exit codes: ``0`` success, ``2`` no subcommand, ``3`` unimplemented body,
``4`` refusal -- the tool declined to produce a number and says which rule
declined it.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from motifmultiverse import __version__
from motifmultiverse.compile import TIERS as COMPILE_TIERS
from motifmultiverse.compile import BackendMissing, CompileError
from motifmultiverse.guards import GuardError
from motifmultiverse.ingest import DEFAULT_TRIM_THRESHOLD, IngestError
from motifmultiverse.interpret import DEFAULT_BLOCK_SIZE, DEFAULT_BOOTSTRAP, InterpretError
from motifmultiverse.provenance import record
from motifmultiverse.schema import (
    MISSING_SENTINEL,
    HealthFloors,
    PeakSetQuery,
    SchemaError,
    SelectionProvenance,
)

_README = "src/motifmultiverse/{module}/README.md"


def _not_implemented(module: str) -> NotImplementedError:
    return NotImplementedError(
        f"'{module}' is a skeleton in this pre-alpha release. "
        f"Its rule, the failure that produced it, and how to check it are specified in "
        f"{_README.format(module=module)}"
    )


def _run(module: str, args: argparse.Namespace) -> int:
    record(module, out_dir=getattr(args, "out", None) or None,
           seed=getattr(args, "seed", None))
    raise _not_implemented(module)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motifmultiverse",
        description=(
            "Bias-aware harmonization and robust inference of attribution-derived "
            "regulatory motifs across models and methods."
        ),
        epilog=("pre-alpha: ingest, compile and interpret are implemented; the other six "
                "subcommands raise NotImplementedError and exit 3. Exit 4 means the tool "
                "refused to produce a number. See docs/ROADMAP.md"),
    )
    p.add_argument("--version", action="version", version=f"motifmultiverse {__version__}")
    sub = p.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")

    a = sub.add_parser(
        "ingest", help="normalise TF-MoDISco outputs into a motif registry",
        description=(
            "Read every discovery HDF5 named by the project config into one registry. "
            "A metacluster group that contributes no patterns is recorded as one of "
            "group_absent / group_empty / not_searched -- three different claims, "
            "never collapsed into 'no motifs'."
        ),
    )
    a.add_argument("project", help="project.yaml (or .json)")
    a.add_argument("--trim-threshold", type=float, default=DEFAULT_TRIM_THRESHOLD,
                   help=f"fraction of peak contribution defining the trimmed core "
                        f"(default: {DEFAULT_TRIM_THRESHOLD})")
    a.add_argument("--seed", type=int, default=None, help="random seed, recorded in provenance")
    a.add_argument("--out", default="registry/", help="output registry directory")
    a.set_defaults(func=_run_ingest)

    a = sub.add_parser("align", help="build the alignment evidence graph")
    a.add_argument("registry", help="registry/ from ingest")
    a.add_argument("--null-shuffles", type=int, default=1000,
                   help="per-pair null shuffles (default: 1000)")
    a.add_argument("--out", default="evidence/", help="output evidence directory")
    a.add_argument("--seed", type=int, default=None, help="random seed, recorded in provenance")
    a.set_defaults(func=lambda ns: _run("align", ns))

    a = sub.add_parser("annotate", help="attach database matches and family labels")
    a.add_argument("evidence", help="evidence/ from align")
    a.add_argument("--tomtom", action="store_true", help="use TomTom matches")
    a.add_argument("--homer", action="store_true", help="use HOMER matches")
    a.add_argument("--databases", default="config/db.yaml", help="database path config")
    a.add_argument("--out", default="evidence/", help="output directory")
    a.set_defaults(func=lambda ns: _run("annotate", ns))

    a = sub.add_parser("adjudicate", help="decide merges, emit a human review file")
    a.add_argument("evidence", help="evidence/ from annotate")
    a.add_argument("--policy", choices=["conservative", "permissive"], default="conservative")
    a.add_argument("--review", default="review.yaml", help="human review file to emit")
    a.add_argument("--out", default="evidence/", help="output directory")
    a.set_defaults(func=lambda ns: _run("adjudicate", ns))

    a = sub.add_parser(
        "compile", help="compile tiered lexicons from a registry and its decisions",
        description=(
            "Write one hit-caller-compatible HDF5 per tier, in the order the loader "
            "emits, each with a manifest carrying its index, content hash and an "
            "explicit statement of what the tier contrast does and does not vary. "
            "Round-trip verification calls the real loader and therefore needs the "
            "finemo backend; without it the file is written but read back by nothing "
            "outside this package (see --verify-roundtrip)."
        ),
    )
    a.add_argument("registry", help="registry/ from ingest")
    a.add_argument("--decisions", default=None,
                   help="decisions JSON: {'tiers': {...}, 'decisions': [...]}")
    a.add_argument("--tiers", default=",".join(COMPILE_TIERS),
                   help=f"comma-separated tiers (default: {','.join(COMPILE_TIERS)})")
    a.add_argument("--verify-roundtrip", choices=["auto", "require", "skip"], default="auto",
                   help="read each lexicon back with the real loader: 'auto' verifies when "
                        "the finemo backend is installed and says so when it is not, "
                        "'require' fails without it, 'skip' never verifies")
    a.add_argument("--seed", type=int, default=None, help="random seed, recorded in provenance")
    a.add_argument("--out", default="lexicons/", help="output lexicon directory")
    a.set_defaults(func=_run_compile)

    a = sub.add_parser("validate", help="downstream stability of the compiled lexicons")
    a.add_argument("lexicons", help="lexicons/ from compile")
    a.add_argument("--fimo-heldout", action="store_true", help="held-out FIMO coverage")
    a.add_argument("--finemo-pilot", action="store_true", help="FiNeMo pilot instance calling")
    a.add_argument("--matched-peaks", action="store_true", help="matched-peak controls")
    a.add_argument("--out", default="validation/", help="output directory")
    a.set_defaults(func=lambda ns: _run("validate", ns))

    a = sub.add_parser("infer", help="robust inference across a specification multiverse")
    a.add_argument("instances", help="instances/ from validate")
    a.add_argument("--unit", choices=["peak", "instance", "region"], default="peak")
    a.add_argument("--multiverse", default="specifications.yaml", help="specification axes")
    a.add_argument("--out", default="inference/", help="output directory")
    a.set_defaults(func=lambda ns: _run("infer", ns))

    a = sub.add_parser("report", help="render the audit report")
    a.add_argument("project", help="project directory")
    a.add_argument("--html", action="store_true", help="render HTML")
    a.add_argument("--docx", action="store_true", help="render DOCX")
    a.add_argument("--out", default="report/", help="output directory")
    a.set_defaults(func=lambda ns: _run("report", ns))

    a = sub.add_parser(
        "interpret",
        help="describe a peak set at the strength its selection provenance licenses",
        description=(
            "Answer a subset query over ONE frozen hit table. Three health numbers are "
            "computed before any effect; if one falls below its pre-registered floor the "
            "reading is suppressed rather than annotated."
        ),
    )
    a.add_argument("hits", help="frozen hit table (.tsv or .parquet)")
    a.add_argument("--peaks", required=True, help="queried peak set (BED, or one region_id per line)")
    a.add_argument("--selection-provenance", default=None,
                   choices=[g.value for g in SelectionProvenance],
                   help="how the peak set was chosen; omitting it is recorded as "
                        "DECLARATION_MISSING and costs the query its inference")
    a.add_argument("--selection-rule", default=None,
                   help="the executable rule, required by PROGRAMMATIC_RULE")
    a.add_argument("--comparator", default=None, help="baseline peak set; required for inference")
    a.add_argument("--comparator-id", default=None, help="name of the baseline, carried on every effect")
    a.add_argument("--held-out", default=None, help="held-out peak set, required by CLUSTERED_WITH_SPLIT")
    a.add_argument("--query-id", default="query", help="name of this query")
    a.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                   help=f"genomic block size for the bootstrap (default: {DEFAULT_BLOCK_SIZE})")
    a.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP,
                   help=f"block bootstrap replicates (default: {DEFAULT_BOOTSTRAP})")
    a.add_argument("--floor-coverage", type=float, default=HealthFloors().min_intersection_coverage,
                   help="pre-registered floor on intersection coverage")
    a.add_argument("--floor-blocks", type=int, default=HealthFloors().min_blocks,
                   help="pre-registered floor on blocks spanned")
    a.add_argument("--floor-explained", type=float, default=HealthFloors().min_explained_fraction,
                   help="pre-registered floor on the fraction the frozen lexicon explains")
    a.add_argument("--seed", type=int, default=0, help="random seed, recorded with every interval")
    a.add_argument("--out", default="interpretation/", help="output directory")
    a.set_defaults(func=_run_interpret)

    return p


def _run_ingest(ns: argparse.Namespace) -> int:
    from motifmultiverse import ingest as ingest_mod

    meta, nodes = ingest_mod.ingest_project(
        ns.project, ns.out, trim_threshold=ns.trim_threshold, seed=ns.seed)
    print(f"registry: {len(nodes)} motif nodes from {len(meta.analyses)} analyses "
          f"({meta.n_models} models)")
    for analysis_id, states in sorted(meta.metacluster_states.items()):
        print(f"  {analysis_id}: " + ", ".join(f"{g}={s}" for g, s in sorted(states.items())))
    if meta.cross_model_claims_restricted:
        print(f"  cross_model_claims_restricted: n_models={meta.n_models} < 3, so "
              "between-model heterogeneity is not estimable; sign consistency and "
              "leave-one-model-out only")
    print(f"written: {ns.out}")
    return 0


def _run_compile(ns: argparse.Namespace) -> int:
    from motifmultiverse import compile as compile_mod

    tiers = tuple(t.strip() for t in ns.tiers.split(",") if t.strip())
    manifests = compile_mod.compile_lexicons(
        ns.registry, ns.out, decisions_path=ns.decisions, tiers=tiers,
        verify=ns.verify_roundtrip, seed=ns.seed)
    for tier, manifest in manifests.items():
        print(f"{tier}: {manifest.n_motifs} motifs  "
              f"content_hash={manifest.lexicon_content_hash[:16]}...")
        for other, cmp in sorted(manifest.comparisons.items()):
            if cmp.get("warning"):
                print(f"  vs {other}: {cmp['warning']}")
    if ns.verify_roundtrip != "skip":
        try:
            compile_mod.load_back(Path(ns.out) / f"{tiers[0]}.h5")
            print("round-trip: verified against the real loader")
        except compile_mod.BackendMissing as exc:
            print(f"round-trip: NOT verified -- {exc}")
    print(f"written: {ns.out}")
    return 0


def _run_interpret(ns: argparse.Namespace) -> int:
    from motifmultiverse import interpret as interpret_mod

    # Provenance is written after the inputs are read and checksummed, but before
    # anything is computed: a record that cannot name its inputs describes nothing.
    rec = record("interpret", seed=ns.seed)
    hits = interpret_mod.read_hit_table(ns.hits)
    rec.input_scale = hits[0].input_scale
    for path in (ns.hits, ns.peaks, ns.comparator, ns.held_out):
        if path:
            rec.add_input(path)
    rec.write(ns.out)

    query = PeakSetQuery(
        query_id=ns.query_id,
        region_ids=interpret_mod.read_peak_set(ns.peaks),
        selection_provenance=(ns.selection_provenance
                              or SelectionProvenance.DECLARATION_MISSING),
        selection_rule=ns.selection_rule or MISSING_SENTINEL,
        comparator_id=ns.comparator_id or (MISSING_SENTINEL if not ns.comparator else ns.comparator),
        comparator_region_ids=interpret_mod.read_peak_set(ns.comparator) if ns.comparator else [],
        held_out_region_ids=interpret_mod.read_peak_set(ns.held_out) if ns.held_out else [],
    )
    result = interpret_mod.interpret_query(
        hits, query,
        floors=HealthFloors(min_intersection_coverage=ns.floor_coverage,
                            min_blocks=ns.floor_blocks,
                            min_explained_fraction=ns.floor_explained),
        block_size=ns.block_size, n_bootstrap=ns.bootstrap, seed=ns.seed,
    )
    dest = result.write(ns.out)

    h = result.health
    print(f"query {result.query_id}: {result.selection_provenance} -> {result.output_mode}")
    print(f"  intersection_coverage : {h['intersection_coverage']} "
          f"({h['n_in_universe']}/{h['n_submitted']} submitted peaks in the universe)")
    print(f"  n_blocks              : {h['n_blocks']} (block size {h['block_size']})")
    print(f"  explained_fraction    : {h['explained_fraction']} "
          f"({h['n_with_used_hit']}/{h['n_searched']} searched peaks)")
    # Composition and effects are suppressed independently (Task 2: a bad
    # comparator withholds effects but not composition, which never depended
    # on it), so they are reported independently too -- printing one line for
    # "there is no composition" while the composition line above it lists the
    # actual families would be exactly the self-corroborating wrongness this
    # project exists to prevent.
    if result.composition is not None:
        if result.effects is not None:
            effects = str(len(result.effects))
        elif result.suppression_reason:
            effects = "suppressed (see below)"
        else:
            effects = "not licensed by this selection provenance"
        print(f"  composition: {len(result.composition)} families; effects: {effects}")
    if result.suppression_reason:
        print(f"  {result.suppression_reason}")
    for note in result.notes:
        print(f"  note: {note}")
    print(f"written: {dest}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if not getattr(ns, "subcommand", None):
        parser.print_help()
        return 2
    try:
        return ns.func(ns)
    except NotImplementedError as exc:
        print(f"motifmultiverse: {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"motifmultiverse: {exc.filename}: no such file", file=sys.stderr)
        return 2
    except (GuardError, InterpretError, IngestError, CompileError,
            BackendMissing, SchemaError) as exc:
        # A refusal is not a crash. Exit 4 means the tool declined to produce a
        # number, and the message says which rule declined it.
        print(f"motifmultiverse: refused: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
