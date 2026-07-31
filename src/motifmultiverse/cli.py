"""Command-line interface (T-08).

Nine subcommands are wired with their real arguments and real ``--help``. Which
of them are implemented is deliberately *not* written here: this sentence used to
enumerate them and had gone stale for the third time in this repository. The
split is derived from the dispatch table by :func:`_status_epilog` and shown in
``--help``; a subcommand that is still a skeleton raises
:class:`NotImplementedError` naming the module README that specifies it.

Every subcommand writes a provenance record before it raises (T-09), because a
skeleton that defers provenance is a tool that never has it.

Exit codes: ``0`` success, ``2`` no subcommand, ``3`` unimplemented body,
``4`` refusal -- the tool declined to produce a number and says which rule
declined it.

KNOWN LIMITATION: a refusal appends its provenance record and produces nothing
else, so it neither writes nor removes result artifacts. Re-running into a
``--out`` that already holds a result therefore leaves that earlier result in
place beside the refusal's record, with nothing in the directory relating the
two. Read the exit code, not the directory. Deleting the earlier result to
prevent the misreading would destroy a real result, and marking it stale needs
a decision about what an output directory promises that this design has not
made; ``tests/test_cli.py::test_a_refused_run_leaves_the_earlier_result_beside_its_own_refusal``
pins the behaviour so it stays a known state rather than a surprise.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from motifmultiverse import __version__
from motifmultiverse.align import DEFAULT_NULL_SHUFFLES, DEFAULT_WORKERS, AlignmentError
from motifmultiverse.annotate import AnnotationError
from motifmultiverse.compile import TIERS as COMPILE_TIERS
from motifmultiverse.compile import BackendMissing, CompileError
from motifmultiverse.guards import GuardError
from motifmultiverse.infer import InferError
from motifmultiverse.ingest import DEFAULT_TRIM_THRESHOLD, IngestError
from motifmultiverse.interpret import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_BOOTSTRAP,
    ESTIMATOR_CHOICES,
    MIN_PERCENTILE_REPLICATES,
    InterpretError,
)
from motifmultiverse.provenance import ProvenanceError, record
from motifmultiverse.report import ReportError
from motifmultiverse.schema import (
    MISSING_SENTINEL,
    Decision,
    HealthFloors,
    PeakSetQuery,
    SchemaError,
    SelectionProvenance,
)

_README = "src/motifmultiverse/{module}/README.md"


def _status_epilog() -> str:
    """Render the implemented/skeleton split from the dispatch table, not by hand.

    This sentence was hand-maintained and had gone stale for the third time: it
    named seven implemented modules and "the remaining two" when `infer` had been
    implemented, leaving only `report`. The README's module table was moved to
    generation for exactly this reason (`status.py`'s docstring lists the earlier
    two); the CLI's own copy of the same claim was left behind. Deriving it here
    means the text cannot disagree with the code it describes.

    Called at help-format time rather than at parser-construction time:
    `status.module_status` reads the dispatch table by building a parser, so
    computing this while `build_parser` is still running recurses forever.
    """
    from motifmultiverse.status import MODULES, module_status

    done, skeleton = [], []
    for name in MODULES:
        (done if module_status(name)["status"] == "IMPLEMENTED" else skeleton).append(name)
    if skeleton:
        tail = (f"{'; '.join(skeleton)} raise{'s' if len(skeleton) == 1 else ''} "
                "NotImplementedError and exit 3")
    else:
        # Reachable since `report` landed. "every subcommand is implemented"
        # restated the head of the sentence; what a reader needs from the empty
        # case is the consequence -- that exit 3 is not a thing that happens now.
        tail = "nothing is a skeleton and no subcommand exits 3"
    return (f"pre-alpha: {', '.join(done)} are implemented; {tail}. "
            "Exit 4 means the tool refused to produce a number. See docs/ROADMAP.md")


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that fills its epilog from the dispatch table when asked.

    The epilog states which subcommands are implemented, and that claim is
    derived rather than typed (see :func:`_status_epilog`). It has to be produced
    after construction finishes, so the derivation is deferred to the moment help
    is actually formatted.
    """

    def format_help(self) -> str:
        self.epilog = _status_epilog()
        return super().format_help()


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
    p = _Parser(
        prog="motifmultiverse",
        description=(
            "Bias-aware harmonization and robust inference of attribution-derived "
            "regulatory motifs across models and methods."
        ),
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

    a = sub.add_parser(
        "align", help="build the alignment evidence graph",
        description=(
            "Register every pair of motifs on UNSIGNED PPM content (offset x "
            "orientation search under a bilateral overlap floor); signed CWM "
            "similarity is measured only at the winning registration, never "
            "re-optimized. Each pair's null re-runs the full registration on "
            "freshly shuffled data, not a rescore at the observed offset."
        ),
    )
    a.add_argument("registry", help="registry/ from ingest")
    a.add_argument("--null-shuffles", type=int, default=DEFAULT_NULL_SHUFFLES,
                   help=f"per-pair null shuffles, re-registered from scratch each time "
                        f"(default: {DEFAULT_NULL_SHUFFLES})")
    a.add_argument("--out", default="evidence/", help="output evidence directory")
    a.add_argument("--seed", type=int, default=0,
                   help="random seed for the null shuffles; required provenance on "
                        "every emitted edge (default: 0)")
    a.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"worker processes for the pair loop (default: {DEFAULT_WORKERS}). "
                        "Each pair's null is drawn from the run seed alone, so the written "
                        "tables are byte-identical at every worker count; only wall-clock "
                        "changes. Progress is reported on stderr, never on stdout.")
    a.set_defaults(func=_run_align)

    a = sub.add_parser("annotate", help="retain database-label candidates for later adjudication")
    a.add_argument("evidence", help="evidence/ from align")
    a.add_argument("--registry", default=None,
                   help="motif registry; defaults to <evidence>/registry when embedded")
    a.add_argument("--tomtom", action="store_true",
                   help="read precomputed TomTom results from --databases "
                        "(does not invoke TomTom)")
    a.add_argument("--homer", action="store_true",
                   help="read precomputed HOMER results from --databases "
                        "(does not invoke HOMER)")
    a.add_argument("--databases", default="config/db.yaml",
                   help="precomputed database results; site-specific and not shipped. "
                        "See config/db.example.yaml for the required shape "
                        "(default: config/db.yaml)")
    a.add_argument("--occurrence-null", default=None,
                   help="precomputed JSON map of candidate IDs to optional occurrence-null values")
    a.add_argument("--out", default="evidence/", help="output directory")
    a.set_defaults(func=_run_annotate)

    a = sub.add_parser("adjudicate", help="decide merges, emit a human review file")
    a.add_argument("evidence", help="evidence/ from annotate")
    a.add_argument("--policy", choices=["conservative"], default="conservative",
                   help="only the frozen conservative policy is currently defined")
    a.add_argument(
        "--registry",
        required=True,
        help="authoritative versioned registry/ supplying variant and medoid metadata",
    )
    a.add_argument("--criteria", default=None,
                   help="versioned executable criterion registry "
                        "(default: the criteria.v1.yaml packaged with adjudicate)")
    a.add_argument("--review", default="review.yaml", help="human review file to emit")
    a.add_argument("--out", default="evidence/", help="output directory")
    a.set_defaults(func=_run_adjudicate)

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
                   help="identity-bearing merge_decisions.json emitted by adjudicate")
    a.add_argument("--tiers", default=",".join(COMPILE_TIERS),
                   help=f"comma-separated tiers (default: {','.join(COMPILE_TIERS)})")
    a.add_argument("--verify-roundtrip", choices=["auto", "require", "skip"], default="auto",
                   help="read each lexicon back with the real loader: 'auto' verifies when "
                        "the finemo backend is installed and callable and says so when it "
                        "is not, 'require' fails without it, 'skip' never verifies")
    a.add_argument("--seed", type=int, default=None, help="random seed, recorded in provenance")
    a.add_argument("--out", default="lexicons/", help="output lexicon directory")
    a.set_defaults(func=_run_compile)

    a = sub.add_parser("validate", help="downstream stability of the compiled lexicons")
    a.add_argument("lexicons", help="lexicons/ from compile")
    a.add_argument("--before-hits", required=True,
                   help="frozen pre-merge standardized hit table (.parquet or .tsv)")
    a.add_argument("--after-hits", required=True,
                   help="frozen post-merge standardized hit table (.parquet or .tsv)")
    a.add_argument("--substrate-manifest", required=True,
                   help="validated manifest for the one frozen caller substrate")
    a.add_argument("--split-manifest", required=True,
                   help="exact frozen PeakSplitManifest JSON")
    a.add_argument("--decision-artifact", required=True,
                   help="manifest-bound decision split artifact JSON")
    a.add_argument("--validation-artifact", required=True,
                   help="manifest-bound validation split artifact JSON")
    a.add_argument("--fimo-heldout", action="store_true", help="held-out FIMO coverage")
    a.add_argument("--finemo-pilot", action="store_true", help="FiNeMo pilot instance calling")
    a.add_argument("--matched-peaks", action="store_true", help="matched-peak controls")
    a.add_argument("--out", default="validation/", help="output directory")
    a.set_defaults(func=_run_validate)

    a = sub.add_parser(
        "infer", help="emit effect estimates for one specification over a frozen substrate",
        description=(
            "The pipeline's inference stage: read ONE frozen hit table, estimate the "
            "per-family query-minus-comparator effect with the selected estimator, and "
            "write a flat effect_estimates.tsv for the report to consume. It runs the "
            "same code interpret runs -- there is one implementation of these estimators, "
            "not two -- and differs only in emitting a tabular artifact rather than "
            "answering an ad-hoc query. "
            "NOT IMPLEMENTED: the specification MULTIVERSE. This estimates ONE "
            "specification. Sweeping specification axes and reporting the dropped cells "
            "with reasons is FP-15's remaining half (docs/ROADMAP.md M4); running one "
            "specification and calling the result a multiverse would be the exact "
            "overstatement this tool exists to prevent."
        ),
    )
    a.add_argument("hits", help="frozen hit table (.tsv or .parquet), as interpret reads")
    a.add_argument("--substrate-manifest", default=None,
                   help="verified manifest for the one frozen caller specification")
    a.add_argument("--peaks", required=True,
                   help="queried peak set: one region_id per line, or a BED whose 4th "
                        "column IS the region_id. Matched to the hit table by exact "
                        "string equality on region_id, never by interval overlap")
    a.add_argument("--comparator", required=True,
                   help="baseline peak set; an effect without a baseline is not an effect")
    a.add_argument("--comparator-id", required=True, help="name of the baseline, on every row")
    a.add_argument("--selection-provenance", default=None,
                   choices=[g.value for g in SelectionProvenance],
                   help="how the peak set was chosen; omitting it is recorded as "
                        "DECLARATION_MISSING and costs the query its inference")
    a.add_argument("--selection-rule", default=None,
                   help="the executable rule, required by PROGRAMMATIC_RULE")
    a.add_argument("--selection-feature", action="append", default=None,
                   metavar="NAME", dest="selection_features",
                   help="a feature the peak set was selected ON; repeatable. This is "
                        "what makes SUBSTRATE_CIRCULAR reachable: a selection feature "
                        "that is itself attribution-derived (attribution_pc1, "
                        "deepshap_score, hit_coefficient, ...) means the number would "
                        "describe the surface it was chosen from, however well "
                        "licensed the query is on the statistical axis")
    a.add_argument("--held-out", default=None, help="held-out peak set, required by CLUSTERED_WITH_SPLIT")
    a.add_argument("--query-id", default="query", help="name of this specification")
    a.add_argument("--estimator", default="percentile", choices=sorted(ESTIMATOR_CHOICES),
                   help="see `interpret --estimator`; only bca-wild-cluster emits p and q")
    a.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                   help=f"genomic block size for the bootstrap (default: {DEFAULT_BLOCK_SIZE})")
    a.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP,
                   # `%%` because argparse %-formats help text; a bare `%` here
                   # raised ValueError out of --help, which is the one place a
                   # tool must never fail.
                   help=f"block bootstrap replicates (default: {DEFAULT_BOOTSTRAP}). The "
                        f"percentile path refuses below {MIN_PERCENTILE_REPLICATES}: fewer "
                        "replicates cannot resolve a 2.5%% tail, so both endpoints would be "
                        "extreme replicates rather than the 95%% interval they are labelled")
    a.add_argument("--floor-coverage", type=float, default=HealthFloors().min_intersection_coverage,
                   help="pre-registered floor on intersection coverage")
    a.add_argument("--floor-blocks", type=int, default=HealthFloors().min_blocks,
                   help="pre-registered floor on blocks spanned")
    a.add_argument("--floor-explained", type=float, default=HealthFloors().min_explained_fraction,
                   help="pre-registered floor on the fraction the frozen lexicon explains")
    a.add_argument("--seed", type=int, default=0, help="random seed, recorded with every interval")
    a.add_argument("--out", default="inference/", help="output directory")
    a.set_defaults(func=_run_infer)

    a = sub.add_parser(
        "report", help="render the audit report",
        description=(
            "Render one markdown audit report from the artifacts a stage actually "
            "wrote: the interpretation.json in the given directory and the "
            "provenance.json log beside it, plus the bias ledger. Every number is a "
            "recorded field printed beside the denominator the stage recorded and "
            "named; nothing here recomputes a number another stage already recorded, "
            "and no absent field gets a default. Markdown is the only rendering form "
            "in this release: --html and --docx refuse rather than emitting markdown "
            "under another name."
        ),
    )
    a.add_argument("interpretation",
                   help="directory holding interpretation.json and the provenance.json "
                        "log beside it, as written by `interpret --out` / `infer --out`")
    a.add_argument("--bias-ledger", default="docs/bias_ledger.tsv",
                   help="the bias ledger to render (default: docs/bias_ledger.tsv). The "
                        "TSV is authoritative where docs/BIAS_LEDGER.md's English gloss "
                        "differs from it")
    a.add_argument("--html", action="store_true",
                   help="NOT IMPLEMENTED: refuses (exit 4). Rendering markdown while the "
                        "caller asked for HTML is the specified-versus-ran gap this "
                        "package exists to close")
    a.add_argument("--docx", action="store_true",
                   help="NOT IMPLEMENTED: refuses (exit 4), for the same reason as --html")
    a.add_argument("--out", default="report/", help="output directory")
    a.set_defaults(func=_run_report)

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
    a.add_argument("--substrate-manifest", default=None,
                   help="verified manifest for the one frozen caller specification")
    a.add_argument("--peaks", required=True,
                   help="queried peak set: one region_id per line, or a BED whose 4th "
                        "column IS the region_id. Peaks are matched to the hit table by "
                        "exact string equality on region_id, never by interval overlap, "
                        "so a 3-column BED is read as 'chrom:start-end' strings and "
                        "matches only a table whose region_id is spelled that way")
    a.add_argument("--selection-provenance", default=None,
                   choices=[g.value for g in SelectionProvenance],
                   help="how the peak set was chosen; omitting it is recorded as "
                        "DECLARATION_MISSING and costs the query its inference")
    a.add_argument("--selection-rule", default=None,
                   help="the executable rule, required by PROGRAMMATIC_RULE")
    a.add_argument("--selection-feature", action="append", default=None,
                   metavar="NAME", dest="selection_features",
                   help="a feature the peak set was selected ON; repeatable. This is "
                        "what makes SUBSTRATE_CIRCULAR reachable: a selection feature "
                        "that is itself attribution-derived (attribution_pc1, "
                        "deepshap_score, hit_coefficient, ...) means the number would "
                        "describe the surface it was chosen from, however well "
                        "licensed the query is on the statistical axis")
    a.add_argument("--comparator", default=None, help="baseline peak set; required for inference")
    a.add_argument("--comparator-id", default=None, help="name of the baseline, carried on every effect")
    a.add_argument("--held-out", default=None, help="held-out peak set, required by CLUSTERED_WITH_SPLIT")
    a.add_argument("--query-id", default="query", help="name of this query")
    a.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                   help=f"genomic block size for the bootstrap (default: {DEFAULT_BLOCK_SIZE})")
    a.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP,
                   # `%%` because argparse %-formats help text; a bare `%` here
                   # raised ValueError out of --help, which is the one place a
                   # tool must never fail.
                   help=f"block bootstrap replicates (default: {DEFAULT_BOOTSTRAP}). The "
                        f"percentile path refuses below {MIN_PERCENTILE_REPLICATES}: fewer "
                        "replicates cannot resolve a 2.5%% tail, so both endpoints would be "
                        "extreme replicates rather than the 95%% interval they are labelled")
    a.add_argument("--estimator", default="percentile", choices=sorted(ESTIMATOR_CHOICES),
                   help="percentile: block bootstrap interval, no p or q value "
                        "(ESTIMATION_ONLY, the conservative default). bca-wild-cluster: "
                        "FP-15's specified pair -- BCa paired genomic-block interval plus "
                        "block-level wild cluster bootstrap-t p value (INTERVAL_AND_TEST), "
                        "with BH q values over the families in the run")
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


def _align_progress(min_interval: float = 2.0):
    """A progress callback for `align`, writing to STDERR and nowhere else.

    `align` prints its counts and the paths it wrote to stdout and callers parse
    that; a progress line interleaved into it would corrupt what they read. The
    alternative is not silence -- the stage is quadratic in the registry with a
    thousand re-registrations per pair, and a job that says nothing for an hour
    is indistinguishable from one that has hung. So the lines go to the other
    stream, where a pipeline reading stdout never sees them and a human watching
    the terminal does.

    Throttled to one line per `min_interval` seconds, because a 28,000-pair run
    would otherwise put 28,000 lines in a log; the first and last pair always
    print, so even a run that finishes instantly reports what it did, and the
    first line carries the pair total so a reader knows what they are waiting
    for. Plain lines rather than a carriage-returned bar: this stream is as often
    a log file as a terminal.
    """
    import time

    last_emitted = 0.0

    def report(completed: int, total: int) -> None:
        nonlocal last_emitted
        now = time.monotonic()
        if completed in (1, total) or now - last_emitted >= min_interval:
            last_emitted = now
            percent = 100.0 * completed / total if total else 100.0
            # "candidate pairs processed", not "edges": a pair whose best offset
            # misses the bilateral overlap floor is finished work but not an
            # edge, and a progress line that counted edges would appear to stall
            # on a registry full of them.
            print(f"align: {completed}/{total} candidate pairs processed ({percent:.0f}%)",
                  file=sys.stderr, flush=True)

    return report


def _run_align(ns: argparse.Namespace) -> int:
    from motifmultiverse import align as align_mod

    summary, edges = align_mod.run(
        ns.registry, ns.out, null_shuffles=ns.null_shuffles, seed=ns.seed,
        workers=ns.workers, progress=_align_progress())
    print(f"align: {summary.n_nodes} motif nodes, {summary.n_pairs_considered} candidate pairs, "
          f"{summary.n_edges} alignment edges ({summary.n_pairs_excluded} excluded: no PPM or "
          "no offset met the bilateral overlap floor)")
    print(f"  null_shuffles={summary.null_shuffles} seed={summary.seed} "
          f"registered_on=unsigned_ppm registration_rule_version={summary.registration_rule_version}")
    # Reported, but as scheduling rather than provenance: the tables above are
    # byte-identical at every worker count, so this line explains the wall-clock
    # and nothing about the numbers.
    print(f"  workers={summary.workers} (scheduling only; the written tables do not depend on it)")
    for e in edges[:5]:
        print(f"  {e.source_node_id} <-> {e.target_node_id}: orientation={e.orientation} "
              f"offset={e.offset} overlap_bp={e.overlap_bp} ppm_similarity={e.ppm_similarity:.3f} "
              f"p={e.empirical_p_value:.4f}")
    print(f"written: {summary.edges_path}")
    print(f"written: {summary.null_summary_path}")
    return 0


def _run_annotate(ns: argparse.Namespace) -> int:
    from motifmultiverse import annotate as annotate_mod
    from motifmultiverse.annotate.homer import HomerBackend
    from motifmultiverse.annotate.tomtom import TomTomBackend

    backends = []
    if ns.tomtom:
        backends.append(TomTomBackend(ns.databases))
    if ns.homer:
        backends.append(HomerBackend(ns.databases))
    occurrence_nulls = None
    if ns.occurrence_null:
        try:
            import json

            occurrence_nulls = json.loads(Path(ns.occurrence_null).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AnnotationError(f"--occurrence-null is not valid JSON: {exc}") from exc
        if not isinstance(occurrence_nulls, dict):
            raise AnnotationError("--occurrence-null must be a JSON object keyed by candidate ID")
    registry = ns.registry or str(Path(ns.evidence) / "registry")
    provenance_inputs = [ns.databases] if backends else []
    if ns.occurrence_null:
        provenance_inputs.append(ns.occurrence_null)
    result = annotate_mod.run(
        registry, ns.out, backends=backends, occurrence_nulls=occurrence_nulls,
        provenance_inputs=provenance_inputs,
    )
    print(f"annotate: {len(result.candidates)} candidates from {len(result.backend_logs)} backends")
    for backend in result.backend_logs:
        detail = f" ({backend.detail})" if backend.detail else ""
        print(f"  {backend.backend} {backend.backend_version}: {backend.status.value} "
              f"({backend.candidate_count} candidates){detail}")
    print(f"written: {Path(ns.out) / 'annotation_candidates.parquet'}")
    print(f"written: {Path(ns.out) / 'annotation_backend_logs.json'}")
    return 0


def _run_adjudicate(ns: argparse.Namespace) -> int:
    from motifmultiverse import adjudicate as adjudicate_mod

    decisions = adjudicate_mod.run(
        ns.evidence,
        ns.out,
        criteria_path=ns.criteria,
        review_path=ns.review,
        policy=ns.policy,
        registry_dir=ns.registry,
    )
    counts = {decision.value: 0 for decision in Decision}
    for row in decisions:
        counts[row.decision.value] += 1
    print(
        f"adjudicate: {len(decisions)} considered clusters "
        f"({', '.join(f'{name}={count}' for name, count in counts.items() if count)})"
    )
    print(f"written: {Path(ns.out) / 'ontology_decisions.parquet'}")
    print(f"written: {Path(ns.out) / 'merge_decisions.json'}")
    review = Path(ns.review)
    print(f"written: {review if review.is_absolute() else Path(ns.out) / review}")
    return 0


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


def _run_validate(ns: argparse.Namespace) -> int:
    from motifmultiverse import validate as validate_mod

    requested = [
        flag for flag, enabled in (
            ("--fimo-heldout", ns.fimo_heldout),
            ("--finemo-pilot", ns.finemo_pilot),
            ("--matched-peaks", ns.matched_peaks),
        ) if enabled
    ]
    if requested:
        raise SchemaError(
            f"{', '.join(requested)} is not yet an adapter input; refusing a semantic no-op. "
            "Supply normalized frozen hit tables only."
        )
    results, verification = validate_mod.run(
        ns.lexicons,
        ns.out,
        before_hits=ns.before_hits,
        after_hits=ns.after_hits,
        substrate_manifest=ns.substrate_manifest,
        split_manifest=ns.split_manifest,
        decision_artifact=ns.decision_artifact,
        validation_artifact=ns.validation_artifact,
    )
    for result in results:
        print(
            f"validate: {result.decision_id} affected_peaks={result.n_affected_peaks} "
            f"affected_delta={result.paired_delta_reconstruction_affected} "
            f"all_peak_delta={result.paired_delta_reconstruction_all} status={result.status}"
        )
        print(f"  {result.power_statement}")
    for backend in verification:
        detail = f" ({backend.detail})" if backend.detail else ""
        print(f"  {backend.backend} {backend.backend_version}: {backend.status}{detail}")
    print(f"written: {Path(ns.out) / 'stability_results.parquet'}")
    print(f"written: {Path(ns.out) / 'backend_verification.tsv'}")
    return 0


#: Columns of `inference/effect_estimates.tsv`, in order. `ci` is split into two
#: columns because a TSV cell holds one value; every other field keeps the name
#: it has on `interpret.FamilyEffect`, so the flat artifact and the JSON record
#: are the same record in two shapes rather than two vocabularies.
EFFECT_ESTIMATE_COLUMNS = (
    "id", "family_id", "comparator_id", "is_cross_condition",
    "effect", "ci_low", "ci_high", "p_value", "q_value",
    "inference_capability", "estimator",
    "n_query_peaks", "n_comparator_peaks", "n_blocks",
    "n_bootstrap", "n_bootstrap_valid", "block_size", "random_seed",
    "substrate_id", "lexicon_id", "input_scale",
    "statistical_license", "claim_scope",
)


def _effect_estimate_rows(result) -> list[list[str]]:
    """Flatten an `Interpretation`'s effects to TSV cells.

    An undefined value is written as the `MISSING_SENTINEL` (`NA`), never as 0 or
    an empty cell: `p_value` is undefined for every ESTIMATION_ONLY row, and a
    blank there reads as "no evidence of an effect" rather than "this estimator
    is not licensed to test". The `inference_capability` column beside it says
    which of the two the reader is looking at.
    """
    rows = []
    for effect in result.effects:
        low, high = effect["ci"] if effect["ci"] is not None else (None, None)
        flat = {
            **effect,
            "ci_low": low,
            "ci_high": high,
            "substrate_id": result.substrate_id,
            "lexicon_id": result.lexicon_id,
            "input_scale": result.input_scale,
            "statistical_license": result.statistical_license,
            "claim_scope": result.claim_scope,
        }
        rows.append([
            MISSING_SENTINEL if flat.get(col) is None else str(flat[col])
            for col in EFFECT_ESTIMATE_COLUMNS
        ])
    return rows


def _run_infer(ns: argparse.Namespace) -> int:
    from motifmultiverse import interpret as interpret_mod
    from motifmultiverse.substrate import read_manifest

    rec = record("infer", seed=ns.seed)
    hits = interpret_mod.read_hit_table(ns.hits)
    rec.input_scale = hits[0].input_scale
    rec.substrate_id = hits[0].substrate_id
    # Keyed by the ROLE each file played, not by its basename. Per-cluster layouts
    # legitimately give a query and its comparator the same filename in different
    # directories, which used to collide; the role is also what a reader of the
    # record actually wants to know.
    for role, path in (("hits", ns.hits), ("substrate_manifest", ns.substrate_manifest),
                       ("peaks", ns.peaks), ("comparator", ns.comparator),
                       ("held_out", ns.held_out)):
        if path:
            rec.add_input(path, key=f"{role}:{Path(path).name}")
    rec.write(ns.out)

    if ns.substrate_manifest:
        manifest = read_manifest(ns.substrate_manifest)
        interpret_mod.verify_against_manifest(hits, manifest, "infer")
    query = PeakSetQuery(
        query_id=ns.query_id,
        region_ids=interpret_mod.read_peak_set(ns.peaks),
        selection_provenance=(ns.selection_provenance
                              or SelectionProvenance.DECLARATION_MISSING),
        selection_rule=ns.selection_rule or MISSING_SENTINEL,
        selection_feature_names=list(ns.selection_features or []),
        comparator_id=ns.comparator_id,
        comparator_region_ids=interpret_mod.read_peak_set(ns.comparator),
        held_out_region_ids=interpret_mod.read_peak_set(ns.held_out) if ns.held_out else [],
    )
    result = interpret_mod.interpret_query(
        hits, query,
        floors=HealthFloors(min_intersection_coverage=ns.floor_coverage,
                            min_blocks=ns.floor_blocks,
                            min_explained_fraction=ns.floor_explained),
        block_size=ns.block_size, n_bootstrap=ns.bootstrap, seed=ns.seed,
        estimator=ns.estimator,
    )
    if result.effects is None:
        # No table at all, rather than an empty one. An `effect_estimates.tsv`
        # holding only a header is indistinguishable from "we looked and found
        # nothing", which is not what happened: the reading was suppressed, and
        # the reason belongs in the refusal rather than in a file nobody re-reads.
        raise InterpretError(
            f"no effect estimates: {result.suppression_reason or 'this selection provenance ' + f'({result.output_mode}) licenses no effect'}"
        )

    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "effect_estimates.tsv"
    lines = ["\t".join(EFFECT_ESTIMATE_COLUMNS)]
    lines += ["\t".join(row) for row in _effect_estimate_rows(result)]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # The full record travels beside the flat one: the TSV is for reading, the
    # JSON carries the health views and notes a TSV row cannot hold.
    result.write(out)

    print(f"infer: {len(result.effects)} family effects, estimator={result.estimator} "
          f"({result.effects[0]['inference_capability']})")
    print("  ONE specification. The specification multiverse is not implemented; "
          "see docs/ROADMAP.md M4.")
    for effect in result.effects[:5]:
        ci = effect["ci"]
        interval = "NA" if ci is None else f"[{ci[0]:.4g}, {ci[1]:.4g}]"
        p = "withheld" if effect["p_value"] is None else f"{effect['p_value']:.4g}"
        print(f"  {effect['family_id']}: effect={effect['effect']:.4g} ci={interval} p={p}")
    for note in result.notes:
        print(f"  note: {note}")
    print(f"written: {dest}")
    print(f"written: {out / 'interpretation.json'}")
    return 0


def _run_interpret(ns: argparse.Namespace) -> int:
    from motifmultiverse import interpret as interpret_mod
    from motifmultiverse.substrate import read_manifest

    # Provenance is written after the inputs are read and checksummed, but before
    # anything is computed: a record that cannot name its inputs describes nothing.
    rec = record("interpret", seed=ns.seed)
    hits = interpret_mod.read_hit_table(ns.hits)
    substrate_id = hits[0].substrate_id
    rec.input_scale = hits[0].input_scale
    rec.substrate_id = substrate_id
    # Keyed by the ROLE each file played, not by its basename. Per-cluster layouts
    # legitimately give a query and its comparator the same filename in different
    # directories, which used to collide; the role is also what a reader of the
    # record actually wants to know.
    for role, path in (("hits", ns.hits), ("substrate_manifest", ns.substrate_manifest),
                       ("peaks", ns.peaks), ("comparator", ns.comparator),
                       ("held_out", ns.held_out)):
        if path:
            rec.add_input(path, key=f"{role}:{Path(path).name}")
    rec.write(ns.out)

    if ns.substrate_manifest:
        manifest = read_manifest(ns.substrate_manifest)
        interpret_mod.verify_against_manifest(hits, manifest, "interpret")
    query = PeakSetQuery(
        query_id=ns.query_id,
        region_ids=interpret_mod.read_peak_set(ns.peaks),
        selection_provenance=(ns.selection_provenance
                              or SelectionProvenance.DECLARATION_MISSING),
        selection_rule=ns.selection_rule or MISSING_SENTINEL,
        selection_feature_names=list(ns.selection_features or []),
        # An undeclared comparator is undeclared. Falling back to `ns.comparator`
        # put a filesystem path into a semantic field that lands in every effect
        # id: not reproducible on another machine, and a local path leaked into a
        # published artifact. The sentinel is what `guards.comparator_declared`
        # exists to act on.
        comparator_id=ns.comparator_id or MISSING_SENTINEL,
        comparator_region_ids=interpret_mod.read_peak_set(ns.comparator) if ns.comparator else [],
        held_out_region_ids=interpret_mod.read_peak_set(ns.held_out) if ns.held_out else [],
    )
    result = interpret_mod.interpret_query(
        hits, query,
        floors=HealthFloors(min_intersection_coverage=ns.floor_coverage,
                            min_blocks=ns.floor_blocks,
                            min_explained_fraction=ns.floor_explained),
        block_size=ns.block_size, n_bootstrap=ns.bootstrap, seed=ns.seed,
        estimator=ns.estimator,
    )
    dest = result.write(ns.out)

    h = result.health
    # Both permission axes, because `output_mode` is the deprecated view that
    # cannot represent SUBSTRATE_CIRCULAR: a query selected on `hit_coefficient`
    # and a query selected on genomic position printed the same line, and the
    # difference between them existed only in the JSON nobody re-opens.
    print(f"query {result.query_id}: {result.selection_provenance} -> {result.output_mode}")
    print(f"  statistical_license   : {result.statistical_license}")
    print(f"  claim_scope           : {result.claim_scope}")
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


def _run_report(ns: argparse.Namespace) -> int:
    """Render the audit report from what a stage recorded, and nothing else.

    Same shape as `_run_interpret`: the provenance record is written before
    anything is rendered, so a refused run still says what was asked for. It is
    written before the `--html` / `--docx` refusal too -- T-09 is "every
    subcommand records", not "every subcommand that gets as far as producing
    something".

    Only the inputs that exist are checksummed. A record naming a checksum for an
    absent file would be a fabricated fact about the run, and the renderer's own
    refusal names the missing file better than a synthesised digest could.
    """
    from motifmultiverse import report as report_mod

    src = Path(ns.interpretation)
    rec = record("report")
    # Keyed by the ROLE each file played, as `interpret` and `infer` key theirs:
    # `interpretation.json` and `provenance.json` are fixed basenames that several
    # directories legitimately share.
    for role, path in (("interpretation", src / "interpretation.json"),
                       ("provenance", src / "provenance.json"),
                       ("bias_ledger", Path(ns.bias_ledger))):
        if path.is_file():
            rec.add_input(path, key=f"{role}:{path.name}")
    rec.write(ns.out)

    requested = [flag for flag, enabled in (("--html", ns.html), ("--docx", ns.docx)) if enabled]
    if requested:
        # The precedent is `_run_validate` refusing `--fimo-heldout`: a flag that
        # names an output this release cannot produce is refused, not silently
        # downgraded to the one it can. Emitting markdown for a caller who asked
        # for HTML is exactly the gap between what was specified and what ran.
        raise ReportError(
            f"{', '.join(requested)} is not a rendering backend in this release; refusing "
            "a semantic no-op. This renderer emits markdown only."
        )

    dest = report_mod.run(src, ns.out, bias_ledger=ns.bias_ledger)
    print(f"report: rendered from {src / 'interpretation.json'} "
          f"and the provenance log beside it")
    print(f"  bias ledger: {ns.bias_ledger}")
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
    except (GuardError, InterpretError, InferError, IngestError, CompileError,
            BackendMissing, SchemaError, AlignmentError, AnnotationError,
            ProvenanceError, ReportError) as exc:
        # A refusal is not a crash. Exit 4 means the tool declined to produce a
        # number, and the message says which rule declined it.
        print(f"motifmultiverse: refused: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
