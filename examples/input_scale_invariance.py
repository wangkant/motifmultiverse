#!/usr/bin/env python
"""Measure whether the hit caller is input-scale invariant.

README finding 1 -- "the hit caller is not input-scale invariant; the onset was
bracketed to (6,460, 7,085] regions on a 6,460-region base" -- is marked
*externally sourced, not reproducible from this repository*.  It is the finding
the architecture rests on: it is why ``guards.single_scale`` exists, why the hit
substrate is frozen, and why a specification may never re-call the caller.

This script turns that citation into a measurement.

THE EXPERIMENT
--------------
One fixed lexicon.  One frozen region universe.  Call hits over a base set of
``N`` regions, then over supersets of ``N + k`` regions that *contain* the base
set, and compare the discrete decisions on the regions present in both.  Input
scale invariance predicts the shared decisions are identical at every ``k``.

Two arms, because "which other regions share the input" has two readings and
they are not the same experiment:

``append``
    The superset is the base rows, in the base order, followed by the extra
    rows.  Every base region keeps its row index.  This isolates *scale*: the
    only thing that changed is that more regions exist.
``sorted``
    The superset is base plus extras, all re-sorted into universe order, so the
    extras interleave and every base region's row index moves.  This is what an
    enlarged peak universe actually looks like, and it varies scale and
    position together.

Running ``sorted`` at the base scale against ``append`` at the base scale gives
a third, free comparison: same regions, same count, different row order.  That
one separates position from scale.

WHAT COUNTS AS A DECISION CHANGE
--------------------------------
A hit key present in one run and absent in the other.  Nothing else.  A key is
``(peak_name, motif_name, start_untrimmed, strand)`` -- all four are properties
of the genome and the lexicon, none of them is a row number, so keys are
comparable across runs of different composition (the caller's own ``peak_id``
is a row index and is not).

Coefficient movement on a *shared* key is reported but is deliberately NOT a
decision change.  This package measured a device null -- identical lexicon,
identical regions, GPU against CPU -- at max ``|coefficient delta|`` 3.63e-07
over 93,661 shared keys, recorded as
``motifmultiverse.validate.DEVICE_NULL_ABS_COEFFICIENT_DELTA``.  That figure is
used here as the floor below which a coefficient difference is instrument
noise; it is not re-derived and no other tolerance is invented.

NEGATIVE CONTROL
----------------
The base set is called twice, at the same scale, on the same device, from the
same input file.  If those two runs disagree the instrument cannot tell a scale
effect from run-to-run noise and every other number here is void.  It is run
first, before any superset.

USAGE
-----
    python examples/input_scale_invariance.py build   --out RUNDIR ...
    python examples/input_scale_invariance.py call    --out RUNDIR --arm append
    python examples/input_scale_invariance.py compare --out RUNDIR

``call`` skips any step whose hit table already exists, so it is resumable and
the two arms can run concurrently on two pinned GPUs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Defaults describing the substrate this was run against.  Every one of them is
# a path on disk or a number taken from a file, never a guess.

# None of these ship with the repository, so none of them is hard-coded to a path
# that only existed on one machine on one day. Each reads an environment variable
# and otherwise must be passed explicitly. The values the reported run used are
# recorded in `examples/results/run_manifest.json` and in
# `docs/INPUT_SCALE_INVARIANCE.md`, which is where provenance belongs -- a default
# pointing into a scratch directory is not a default, it is a dangling reference.
UNIVERSE_NPZ = os.environ.get("MMV_SCALE_UNIVERSE_NPZ")
LEXICON_H5 = os.environ.get("MMV_SCALE_LEXICON_H5")
FINEMO = os.environ.get("MMV_FINEMO", "finemo")

#: Frozen caller settings, copied verbatim from the case study's own
#: hits/before_core/parameters.json so the instrument is the same instrument.
CALL_MODE = "pp"
CWM_TRIM_THRESHOLD = 0.3
GLOBAL_LAMBDA = 0.7
BATCH_SIZE = 2000

#: Seed for the permutation that defines the nested ladder.  Same seed the case
#: study's RUN.sh passes to every stage.
SEED = 20260731

BASE_N = 6460
#: (label, N).  Ordered by how much each one is worth: the negative control
#: first, then the reference's own upper bracket, then the rest.
LADDER: tuple[tuple[str, int], ...] = (
    ("base", 6460),
    ("ctrl", 6460),
    ("sup_7085", 7085),
    ("sup_6525", 6525),
    ("sup_6783", 6783),
    ("sup_12920", 12920),
    ("sup_9690", 9690),
)

ARMS = ("append", "sorted")

#: motifmultiverse.validate.DEVICE_NULL_ABS_COEFFICIENT_DELTA.  Imported when
#: the package is importable so the two can never drift; the literal is the
#: fallback for running this file outside the installed package.
try:  # pragma: no cover - import shape, not logic
    from motifmultiverse.validate import DEVICE_NULL_ABS_COEFFICIENT_DELTA
except Exception:  # pragma: no cover
    DEVICE_NULL_ABS_COEFFICIENT_DELTA = 3.63e-07


# --------------------------------------------------------------------------- #
def sha256(path: str | Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        read = 0
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            read += len(chunk)
            if limit is not None and read >= limit:
                break
    return h.hexdigest()


def ladder_indices(n_universe: int) -> dict[str, np.ndarray]:
    """Nested row-index sets: every step is a superset of the one before it.

    A single seeded permutation defines the whole ladder -- step ``N`` is its
    first ``N`` entries -- so nesting is structural rather than checked.
    """
    perm = np.random.default_rng(SEED).permutation(n_universe)
    out: dict[str, np.ndarray] = {}
    for label, n in LADDER:
        if n > n_universe:
            raise SystemExit(f"ladder step {label}={n} exceeds universe {n_universe}")
        out[label] = perm[:n].copy()
    return out


def arm_order(rows: np.ndarray, arm: str, base_rows: np.ndarray) -> np.ndarray:
    """Row order actually written into the .npz for one arm."""
    if arm == "sorted":
        return np.sort(rows)
    if arm != "append":
        raise SystemExit(f"unknown arm {arm}")
    # base rows first, in base order, then the extras in permutation order.
    row_list = rows.tolist()
    row_set = set(row_list)
    base_set = set(base_rows.tolist())
    if not base_set <= row_set:
        raise SystemExit("ladder step is not a superset of the base set")
    head = [r for r in base_rows.tolist() if r in row_set]
    tail = [r for r in row_list if r not in base_set]
    return np.asarray(head + tail, dtype=rows.dtype)


# --------------------------------------------------------------------------- #
def cmd_build(args: argparse.Namespace) -> int:
    out = Path(args.out)
    (out / "regions").mkdir(parents=True, exist_ok=True)

    src = np.load(args.universe)
    seqs = src["sequences"]
    contribs = src["contributions"]
    chrom = src["chr"]
    chrom_id = src["chr_id"]
    start = src["start"]
    names = src["peak_name"]
    n_universe = seqs.shape[0]
    if len(set(names.tolist())) != n_universe:
        raise SystemExit("universe peak_name is not unique; it cannot key hits")

    ladder = ladder_indices(n_universe)
    base_rows = ladder["base"]

    manifest = {
        "universe_npz": str(args.universe),
        "n_universe": int(n_universe),
        "region_width": int(seqs.shape[2]),
        "lexicon_h5": str(args.lexicon),
        "lexicon_sha256": sha256(args.lexicon),
        "seed": SEED,
        "caller": {
            "mode": CALL_MODE,
            "cwm_trim_threshold": CWM_TRIM_THRESHOLD,
            "global_lambda": GLOBAL_LAMBDA,
            "batch_size": BATCH_SIZE,
        },
        "device_null_abs_coefficient_delta": DEVICE_NULL_ABS_COEFFICIENT_DELTA,
        "steps": [],
    }

    for arm in ARMS:
        for label, n in LADDER:
            rows = ladder[label]
            if label == "ctrl":
                # The negative control is the SAME input file as `base`, called
                # a second time.  Writing a second copy would test the writer.
                manifest["steps"].append(
                    {"arm": arm, "label": label, "n": n,
                     "regions_npz": f"regions/{arm}_base.npz", "repeat_of": "base"}
                )
                continue
            order = arm_order(rows, arm, base_rows)
            path = out / "regions" / f"{arm}_{label}.npz"
            if not path.exists():
                np.savez(
                    path,
                    sequences=seqs[order],
                    contributions=contribs[order],
                    chr=chrom[order],
                    chr_id=chrom_id[order],
                    start=start[order],
                    peak_id=np.arange(len(order), dtype=np.uint32),
                    peak_name=names[order],
                )
            np.save(out / "regions" / f"{arm}_{label}.rows.npy", order)
            manifest["steps"].append(
                {"arm": arm, "label": label, "n": int(len(order)),
                 "regions_npz": f"regions/{arm}_{label}.npz"}
            )
            print(f"[build] {arm}/{label}: n={len(order)} -> {path.name}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build] manifest -> {out / 'manifest.json'}")
    return 0


# --------------------------------------------------------------------------- #
def cmd_call(args: argparse.Namespace) -> int:
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    env = dict(os.environ)
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    timings = []
    for step in manifest["steps"]:
        if step["arm"] != args.arm:
            continue
        label = step["label"]
        hits_dir = out / "hits" / f"{args.arm}_{label}"
        if (hits_dir / "hits.tsv").exists():
            print(f"[call] skip {args.arm}/{label} (exists)", flush=True)
            continue
        regions = out / step["regions_npz"]
        cmd = [
            args.finemo, "call-hits", "-M", CALL_MODE, "-r", str(regions),
            "-m", str(manifest["lexicon_h5"]), "-o", str(hits_dir),
            "-t", str(CWM_TRIM_THRESHOLD), "-l", str(GLOBAL_LAMBDA),
            "-b", str(BATCH_SIZE),
        ]
        t0 = time.time()
        with open(out / "hits" / f"{args.arm}_{label}.log", "w") as log:
            rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        print(f"[call] {args.arm}/{label} n={step['n']} rc={rc} {dt:.0f}s", flush=True)
        if rc != 0:
            return rc
        timings.append({"arm": args.arm, "label": label, "n": step["n"], "seconds": round(dt, 1)})
    if timings:
        p = out / f"timings_{args.arm}.json"
        prev = json.loads(p.read_text()) if p.exists() else []
        p.write_text(json.dumps(prev + timings, indent=2))
    return 0


def cmd_call_one(args: argparse.Namespace) -> int:
    """One extra call outside the ladder, with an explicit output label.

    Used for the device control: the two arms run on two different pinned GPUs,
    so the cross-arm (order-only) comparison would otherwise confound row order
    with GPU instance.  Re-calling the `sorted` base on the `append` arm's GPU
    separates them, and doubles as a device null measured on this substrate
    rather than borrowed from the case study's.
    """
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    hits_dir = out / "hits" / args.label
    if (hits_dir / "hits.tsv").exists():
        print(f"[call-one] skip {args.label} (exists)")
        return 0
    env = dict(os.environ)
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    cmd = [
        args.finemo, "call-hits", "-M", CALL_MODE, "-r", str(out / args.regions),
        "-m", str(manifest["lexicon_h5"]), "-o", str(hits_dir),
        "-t", str(CWM_TRIM_THRESHOLD), "-l", str(GLOBAL_LAMBDA),
        "-b", str(args.batch_size),
    ]
    t0 = time.time()
    with open(out / "hits" / f"{args.label}.log", "w") as log:
        rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"[call-one] {args.label} rc={rc} {time.time() - t0:.0f}s", flush=True)
    return rc


# --------------------------------------------------------------------------- #
def load_hits(path: Path) -> dict[tuple[str, str, str, str], float]:
    """key -> hit_coefficient.  The key carries no row number."""
    hits: dict[tuple[str, str, str, str], float] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            key = (row["peak_name"], row["motif_name"], row["start_untrimmed"], row["strand"])
            if key in hits:
                raise SystemExit(f"duplicate hit key {key} in {path}")
            hits[key] = float(row["hit_coefficient"])
    return hits


def compare(a: dict, b: dict, shared_peaks: set[str]) -> dict:
    """Compare two hit tables over an explicitly supplied region set.

    ``shared_peaks`` comes from the *inputs* -- the row-index files that built
    the two .npz -- and never from the hit tables.  Deriving it from the hit
    tables would silently drop exactly the regions the experiment is looking
    for: a region whose every hit disappeared in one run has no rows there, so
    an intersection of hit-table peak names would remove it and score the
    largest possible decision change as no change at all.
    """
    keys_a = {k for k in a if k[0] in shared_peaks}
    keys_b = {k for k in b if k[0] in shared_peaks}
    shared = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    deltas = [abs(a[k] - b[k]) for k in shared]
    max_delta = max(deltas) if deltas else 0.0
    over = sum(1 for d in deltas if d > DEVICE_NULL_ABS_COEFFICIENT_DELTA)
    changed_peaks = {k[0] for k in (only_a | only_b)}
    # How big were the hits whose presence changed?  A decision change on a hit
    # of coefficient 1e-05 is the caller's sparsity boundary flickering; one on
    # a coefficient of 1 is a motif call appearing or disappearing.  The base
    # run's median hit coefficient is ~1.19, so 1.0 is used as the "real hit"
    # mark and stated rather than tuned.
    changed_coefs = sorted(
        [a[k] for k in only_a] + [b[k] for k in only_b]
    )
    n_mid = len(changed_coefs) // 2
    return {
        "n_shared_regions": len(shared_peaks),
        "n_regions_with_zero_hits_a": len(shared_peaks - {k[0] for k in keys_a}),
        "n_regions_with_zero_hits_b": len(shared_peaks - {k[0] for k in keys_b}),
        "n_hits_base_on_shared": len(shared) + len(only_a),
        "n_hits_other_on_shared": len(shared) + len(only_b),
        "n_shared_keys": len(shared),
        "n_dropped": len(only_a),
        "n_gained": len(only_b),
        "n_decision_changes": len(only_a) + len(only_b),
        "hit_jaccard": (len(shared) / len(shared | only_a | only_b)) if (shared or only_a or only_b) else 1.0,
        "n_regions_with_decision_change": len(changed_peaks),
        "frac_regions_with_decision_change": (
            len(changed_peaks) / len(shared_peaks) if shared_peaks else 0.0
        ),
        "n_decision_changes_coefficient_ge_1": sum(1 for c in changed_coefs if c >= 1.0),
        "changed_hit_coefficient_median": changed_coefs[n_mid] if changed_coefs else 0.0,
        "changed_hit_coefficient_max": changed_coefs[-1] if changed_coefs else 0.0,
        "max_abs_coefficient_delta": max_delta,
        "n_shared_keys_over_device_null": over,
        # Private: consumed by cmd_compare to key the solver-step diagnostic, and
        # popped before the row is written. Not a result column.
        "_changed_peaks": changed_peaks,
    }


def solver_steps(qc_path: Path, keep: set[str]) -> dict[str, int]:
    """peak_name -> ``num_steps``, the solver iterations that region received.

    This is the mechanism channel, and it is read from the caller's own
    ``peaks_qc.tsv`` rather than inferred. FiNeMo holds a rolling buffer of
    ``--batch-size`` regions and refills each converged slot from the input in
    order, so a region's iteration count is a property of WHERE IT LANDED IN THE
    SCHEDULE, not of the region. If a manipulation moves decisions it should move
    this too; if it moves neither, it did not perturb the solver at all.
    """
    steps: dict[str, int] = {}
    with open(qc_path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["peak_name"] in keep:
                steps[row["peak_name"]] = int(row["num_steps"])
    return steps


def compare_steps(qc_a: Path, qc_b: Path, shared_peaks: set[str],
                  changed_peaks: set[str]) -> dict:
    a = solver_steps(qc_a, shared_peaks)
    b = solver_steps(qc_b, shared_peaks)
    common = set(a) & set(b)
    moved = {p for p in common if a[p] != b[p]}
    return {
        "n_regions_num_steps_differs": len(moved),
        "max_abs_num_steps_delta": max((abs(a[p] - b[p]) for p in common), default=0),
        "n_changed_regions_also_resolved_differently": len(changed_peaks & moved),
    }


def cmd_compare(args: argparse.Namespace) -> int:
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    hits_root = out / "hits"

    # The comparison set, taken from the inputs: the peak names of the base
    # rows.  Every ladder step contains all of them by construction.
    universe_names = np.load(manifest["universe_npz"])["peak_name"]
    base_rows = np.load(out / "regions" / "append_base.rows.npy")
    shared_peaks = {str(x) for x in universe_names[base_rows]}
    if len(shared_peaks) != BASE_N:
        raise SystemExit(f"base region set is {len(shared_peaks)} names, expected {BASE_N}")

    def have(arm: str, label: str) -> Path | None:
        p = hits_root / f"{arm}_{label}" / "hits.tsv"
        return p if p.exists() else None

    cache: dict[str, dict] = {}

    def get(arm: str, label: str) -> dict | None:
        p = have(arm, label)
        if p is None:
            return None
        k = f"{arm}_{label}"
        if k not in cache:
            cache[k] = load_hits(p)
        return cache[k]

    def qc(arm: str, label: str) -> Path:
        # The negative control re-runs `base`'s input under its own output dir,
        # so its peaks_qc is that dir's, not base's.
        return hits_root / f"{arm}_{label}" / "peaks_qc.tsv"

    rows = []
    for arm in ARMS:
        base = get(arm, "base")
        if base is None:
            continue
        for label, n in LADDER:
            if label == "base":
                continue
            other = get(arm, label)
            if other is None:
                continue
            res = compare(base, other, shared_peaks)
            kind = "negative_control" if label == "ctrl" else "scale"
            rows.append({
                "comparison": f"{arm}:base_{BASE_N}_vs_{label}",
                "arm": arm, "kind": kind,
                "n_base": BASE_N, "n_other": n,
                "pct_scale_increase": round(100.0 * (n - BASE_N) / BASE_N, 3),
                **res,
                "_qc": (qc(arm, "base"), qc(arm, label)),
            })
    # Order-only control: same regions, same count, different row order.  The
    # two arms ran on two different pinned GPUs, so this row confounds order
    # with device; the two rows after it separate them.
    a, b = get("append", "base"), get("sorted", "base")
    if a is not None and b is not None:
        rows.append({
            "comparison": "append_base_vs_sorted_base",
            "arm": "cross", "kind": "order_only_cross_device",
            "n_base": BASE_N, "n_other": BASE_N, "pct_scale_increase": 0.0,
            **compare(a, b, shared_peaks),
            "_qc": (qc("append", "base"), qc("sorted", "base")),
        })
    extra = hits_root / "sorted_base_on_append_gpu" / "hits.tsv"
    if extra.exists():
        c = load_hits(extra)
        extra_qc = extra.with_name("peaks_qc.tsv")
        if b is not None:
            rows.append({
                "comparison": "sorted_base_vs_sorted_base_other_gpu",
                "arm": "cross", "kind": "device_control",
                "n_base": BASE_N, "n_other": BASE_N, "pct_scale_increase": 0.0,
                **compare(b, c, shared_peaks),
                "_qc": (qc("sorted", "base"), extra_qc),
            })
        if a is not None:
            rows.append({
                "comparison": "append_base_vs_sorted_base_same_gpu",
                "arm": "cross", "kind": "order_only_same_device",
                "n_base": BASE_N, "n_other": BASE_N, "pct_scale_increase": 0.0,
                **compare(a, c, shared_peaks),
                "_qc": (qc("append", "base"), extra_qc),
            })

    # Batch-size control: the base input file, the base row order, one GPU, and
    # only `--batch-size` changed.  Scale and order are both held fixed, so a
    # decision change here cannot be attributed to either -- which is the whole
    # question this experiment turned out to be about.  Produced by
    # `call-one --regions regions/append_base.npz --batch-size N --label batch_N`.
    for batch_dir in sorted(hits_root.glob("batch_*")):
        if not (batch_dir / "hits.tsv").exists() or a is None:
            continue
        rows.append({
            "comparison": f"append_base_vs_{batch_dir.name}",
            "arm": "cross", "kind": "batch_size_only",
            "n_base": BASE_N, "n_other": BASE_N, "pct_scale_increase": 0.0,
            **compare(a, load_hits(batch_dir / "hits.tsv"), shared_peaks),
            "_qc": (qc("append", "base"), batch_dir / "peaks_qc.tsv"),
        })

    if not rows:
        print("no comparable hit tables yet", file=sys.stderr)
        return 1

    # Mechanism channel. Resolved after the fact so a missing peaks_qc.tsv costs
    # the diagnostic and not the result: the decision columns above stand on
    # their own, and this only explains them.
    for row in rows:
        qc_a, qc_b = row.pop("_qc")
        changed = row.pop("_changed_peaks")
        if qc_a.exists() and qc_b.exists():
            row.update(compare_steps(qc_a, qc_b, shared_peaks, changed))
        else:
            row.update({
                "n_regions_num_steps_differs": "",
                "max_abs_num_steps_delta": "",
                "n_changed_regions_also_resolved_differently": "",
            })

    cols = list(rows[0])
    tsv = out / "scale_invariance_results.tsv"
    with open(tsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {tsv}")
    widths = {c: max(len(c), *(len(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(
            (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])).ljust(widths[c]) for c in cols
        ))
    (out / "compare_provenance.json").write_text(json.dumps({
        "lexicon_sha256": manifest["lexicon_sha256"],
        "universe_npz": manifest["universe_npz"],
        "device_null_abs_coefficient_delta": DEVICE_NULL_ABS_COEFFICIENT_DELTA,
        "decision_change": "hit key present on one side and absent on the other; "
                           "key = (peak_name, motif_name, start_untrimmed, strand)",
    }, indent=2))
    return 0


# --------------------------------------------------------------------------- #
def _stability_frame(hits_path: Path, qc_path: Path, keep: set[str]):
    """The package's four standardized columns, built from one FiNeMo run.

    ``reconstruction`` must be one value per peak; the caller's own per-peak
    negative log-likelihood (``peaks_qc.tsv``) is that value.  Nothing here is
    inferred from a column name -- the mapping is written out.
    """
    import pandas as pd

    qc = pd.read_csv(qc_path, sep="\t")
    nll = dict(zip(qc["peak_name"].astype(str), qc["nll"].astype(float), strict=True))
    hits = pd.read_csv(hits_path, sep="\t")
    hits = hits[hits["peak_name"].astype(str).isin(keep)]
    return pd.DataFrame({
        "peak_id": hits["peak_name"].astype(str),
        "hit_id": (
            hits["motif_name"].astype(str) + "@"
            + hits["start_untrimmed"].astype(str) + ":" + hits["strand"].astype(str)
        ),
        "coefficient": hits["hit_coefficient"].astype(float),
        "reconstruction": [nll[p] for p in hits["peak_name"].astype(str)],
    })


def cmd_stability(args: argparse.Namespace) -> int:
    """Re-express the same comparisons through ``validate.evaluate_stability``."""
    import pandas as pd

    from motifmultiverse.validate import evaluate_stability

    out = Path(args.out)
    rows = []
    for arm in ARMS:
        base_dir = out / "hits" / f"{arm}_base"
        if not (base_dir / "hits.tsv").exists():
            continue
        for label, n in LADDER:
            other_dir = out / "hits" / f"{arm}_{label}"
            if label == "base" or not (other_dir / "hits.tsv").exists():
                continue
            b_names = set(pd.read_csv(base_dir / "hits.tsv", sep="\t")["peak_name"].astype(str))
            o_names = set(pd.read_csv(other_dir / "hits.tsv", sep="\t")["peak_name"].astype(str))
            keep = b_names & o_names
            before = _stability_frame(base_dir / "hits.tsv", base_dir / "peaks_qc.tsv", keep)
            after = _stability_frame(other_dir / "hits.tsv", other_dir / "peaks_qc.tsv", keep)
            res = evaluate_stability(f"{arm}:base_vs_{label}", before, after)
            rows.append({
                "comparison": f"{arm}:base_{BASE_N}_vs_{label}", "n_other": n,
                "n_peaks_compared": len(keep),
                "n_peaks_excluded_no_hits_one_side": len(b_names ^ o_names),
                "n_affected_peaks": res.n_affected_peaks,
                "n_affected_hits": res.n_affected_hits,
                "hit_jaccard": res.hit_jaccard,
                "coefficient_conservation": res.coefficient_conservation,
                "status": res.status,
                "affected_definition": res.affected_definition,
            })
    if not rows:
        print("no hit tables to evaluate", file=sys.stderr)
        return 1
    path = out / "scale_invariance_stability.tsv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")
    for r in rows:
        print(r)
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--universe", default=UNIVERSE_NPZ, required=UNIVERSE_NPZ is None,
                   help="FiNeMo regions .npz for the FULL universe "
                        "(env: MMV_SCALE_UNIVERSE_NPZ)")
    b.add_argument("--lexicon", default=LEXICON_H5, required=LEXICON_H5 is None,
                   help="compiled lexicon .h5, frozen for every step "
                        "(env: MMV_SCALE_LEXICON_H5)")
    b.set_defaults(fn=cmd_build)

    c = sub.add_parser("call")
    c.add_argument("--out", required=True)
    c.add_argument("--arm", required=True, choices=ARMS)
    c.add_argument("--gpu", default="", help="CUDA_VISIBLE_DEVICES value; pin by UUID")
    c.add_argument("--finemo", default=FINEMO)
    c.set_defaults(fn=cmd_call)

    c1 = sub.add_parser("call-one")
    c1.add_argument("--out", required=True)
    c1.add_argument("--regions", required=True, help="path relative to --out")
    c1.add_argument("--label", required=True)
    c1.add_argument("--gpu", default="")
    c1.add_argument("--finemo", default=FINEMO)
    c1.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="caller --batch-size. Exposed because it is a THIRD way to move the "
             "decisions at fixed region set and fixed order, and the one that "
             "shows the effect is not about scale: it changes which solver slot "
             "and step a region occupies, and nothing about the region.",
    )
    c1.set_defaults(fn=cmd_call_one)

    k = sub.add_parser("compare")
    k.add_argument("--out", required=True)
    k.set_defaults(fn=cmd_compare)

    s = sub.add_parser("stability")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_stability)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
