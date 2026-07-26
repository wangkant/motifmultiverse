"""align stage -- see README.md in this directory for rule / failure / check.

Two constraints carry the science, and both are enforced structurally rather
than left to caller discipline:

**Registration is selected on UNSIGNED sequence content (PPM).** Every offset
and orientation candidate is scored by cosine similarity of the (non-negative)
PPM rows. Signed CWM similarity is computed only AFTER an offset has been
chosen this way, at that one registration -- it never enters the search, so it
can never move the search toward or away from any particular offset. This is
the fix for the failure `README.md` documents: an aligner that maximises
*signed* cosine is structurally blind to a sign-flipped motif (the signed
cosine at the true registration is near -1, so that offset can never win under
that objective), which manufactured a false "no sign flips found" verdict in
the reference implementation. `guards.sign_alignment` is the executable check
that a result was produced this way (`registered_on == "unsigned_ppm"`).

**The bilateral overlap requirement excludes short windows from candidacy
entirely**, not just from being reported. A 3-4bp window can reach a very high
(even perfect) local cosine purely by chance or by an unrelated shared k-mer;
without a floor on both `overlap_bp` and the overlap fraction of *each* motif's
own length, the search would happily return that window as "the" alignment.
The floor is applied before scoring is even compared across candidates: an
invalid window is not scored down, it is not a candidate at all.

**The null re-runs the full pipeline per shuffle.** `calibrate_pair_null` calls
`register_pair` -- offset and orientation search included -- once per shuffle,
on freshly permuted data. A null that reused the observed offset and only
recomputed a score at it would be answering "how similar are these two
sequences at one fixed alignment", not "how surprising is it to find SOME
alignment this good" -- a different, easier question that inflates every
p-value computed against it.

BASE_ORDER documents a convention this module needs but nothing upstream
declares: PPM/CWM columns are `(A, C, G, T)`. This is a numeric axis
convention (needed only to build a reverse complement), not a parsed
identifier, so it is not a Rule 2 (`no_key_parsing`) violation.
"""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any

__all__ = [
    "AlignmentError", "AlignmentEvidence", "AlignmentRunSummary",
    "register_pair", "calibrate_pair_null", "align_registry", "run",
    "BASE_ORDER", "REGISTRATION_RULE_VERSION",
    "DEFAULT_MIN_OVERLAP_BP", "DEFAULT_MIN_OVERLAP_FRAC", "DEFAULT_NULL_SHUFFLES",
]

#: PPM/CWM column convention. See module docstring.
BASE_ORDER = "ACGT"

#: Bumped whenever the registration rule itself changes (the overlap floor,
#: the scoring function, the orientation search). Carried on every emitted
#: edge so a downstream reader never has to guess which rule produced a row.
REGISTRATION_RULE_VERSION = "unsigned_ppm_v1"

#: The bilateral overlap floor. Both conditions must hold: raw overlap length
#: AND the fraction of each motif's own length that overlap represents -- a
#: floor on `overlap_bp` alone would still let a long motif's tail "cover" a
#: short one almost entirely while barely touching itself.
DEFAULT_MIN_OVERLAP_BP = 6
DEFAULT_MIN_OVERLAP_FRAC = 0.5

DEFAULT_NULL_SHUFFLES = 1000


class AlignmentError(ValueError):
    """A pair could not be registered under the bilateral overlap rule."""


@dataclass(frozen=True)
class AlignmentEvidence:
    """One pair's registration, plus the null that calibrates it.

    Field order matches the brief's interface exactly; `registration_rule_version`
    is appended rather than inserted, so positional construction of the
    original fields is unaffected.
    """

    source_node_id: str
    target_node_id: str
    orientation: str
    offset: int
    overlap_bp: int
    overlap_frac_source: float
    overlap_frac_target: float
    ppm_similarity: float
    signed_cwm_similarity: float | None
    empirical_p_value: float
    null_shuffles: int
    seed: int
    registered_on: str = "unsigned_ppm"
    registration_rule_version: str = REGISTRATION_RULE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlignmentRunSummary:
    """What one `align_registry` call did, independent of any one edge."""

    n_nodes: int
    n_pairs_considered: int
    n_edges: int
    n_pairs_excluded: int
    null_shuffles: int
    seed: int
    registration_rule_version: str
    edges_path: str
    null_summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Numeric core
# --------------------------------------------------------------------------- #
def _as_matrix(mat: Any, what: str):
    import numpy as np

    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise AlignmentError(
            f"expected a (length, 4) {what} array over BASE_ORDER={BASE_ORDER!r}, "
            f"got shape {arr.shape}"
        )
    return arr


def _reverse_complement(mat: Any):
    """Reverse the position order and complement each base column.

    `BASE_ORDER = "ACGT"`: complementing swaps column 0<->3 (A<->T) and
    column 1<->2 (C<->G), the standard DNA complement, applied to a PPM or a
    CWM alike -- both are `(length, 4)` matrices over the same column
    convention, so the same transform registers either.
    """
    return mat[::-1, [3, 2, 1, 0]]


def _cosine(a: Any, b: Any) -> float:
    import numpy as np

    flat_a, flat_b = a.ravel(), b.ravel()
    norm_a, norm_b = float(np.linalg.norm(flat_a)), float(np.linalg.norm(flat_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(flat_a, flat_b) / (norm_a * norm_b))


def _candidate_windows(source_len: int, target_len: int):
    """Every (offset, overlap) with overlap_bp>=1, in ascending offset order.

    `offset` is defined so that target row `t` sits under source row `t + offset`:
    the overlapping source range is `[max(0,offset), min(source_len,
    target_len+offset))`, and the matching target range is that same window
    shifted back by `offset`.
    """
    for offset in range(-(target_len - 1), source_len):
        s_start = max(0, offset)
        s_end = min(source_len, target_len + offset)
        overlap_bp = s_end - s_start
        if overlap_bp <= 0:
            continue
        t_start = s_start - offset
        t_end = s_end - offset
        yield offset, s_start, s_end, t_start, t_end, overlap_bp


def register_pair(source_ppm: Any, target_ppm: Any,
                  source_cwm: Any | None = None, target_cwm: Any | None = None,
                  *, min_overlap_bp: int = DEFAULT_MIN_OVERLAP_BP,
                  min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
                  source_node_id: str = "source", target_node_id: str = "target",
                  ) -> AlignmentEvidence:
    """Register one pair: search offset x orientation on PPM, score signed CWM
    only at the winner.

    `source_node_id`/`target_node_id` are metadata only, not part of the search;
    `align_registry` fills in the real ids afterwards. This keeps the function
    usable directly on raw arrays, exactly as the brief's interface specifies.
    """
    src = _as_matrix(source_ppm, "PPM")
    tgt = _as_matrix(target_ppm, "PPM")
    source_len, target_len = src.shape[0], tgt.shape[0]

    best: tuple[float, str, int, int, int, int, int, int, float, float] | None = None
    for orientation, tgt_oriented in (("+", tgt), ("-", _reverse_complement(tgt))):
        for offset, s_start, s_end, t_start, t_end, overlap_bp in _candidate_windows(
            source_len, target_len,
        ):
            frac_source = overlap_bp / source_len
            frac_target = overlap_bp / target_len
            # The bilateral overlap requirement: a candidate that fails it is
            # not scored down, it is not a candidate. A 3-4bp window reaching a
            # perfect local cosine must never reach the comparison below.
            if (overlap_bp < min_overlap_bp
                    or frac_source < min_overlap_frac
                    or frac_target < min_overlap_frac):
                continue
            score = _cosine(src[s_start:s_end], tgt_oriented[t_start:t_end])
            if best is None or score > best[0]:
                best = (score, orientation, offset, s_start, s_end, t_start, t_end,
                        overlap_bp, frac_source, frac_target)

    if best is None:
        raise AlignmentError(
            f"no offset/orientation satisfies the bilateral overlap requirement "
            f"(min_overlap_bp={min_overlap_bp}, min_overlap_frac={min_overlap_frac}) for "
            f"source_len={source_len}, target_len={target_len}; refusing to register this pair "
            "rather than return an offset that does not meet it"
        )

    (score, orientation, offset, s_start, s_end, t_start, t_end,
     overlap_bp, frac_source, frac_target) = best

    signed_similarity: float | None = None
    if source_cwm is not None and target_cwm is not None:
        src_cwm = _as_matrix(source_cwm, "CWM")
        tgt_cwm = _as_matrix(target_cwm, "CWM")
        if src_cwm.shape[0] != source_len:
            raise AlignmentError(
                f"source_cwm length {src_cwm.shape[0]} does not match source_ppm length {source_len}"
            )
        if tgt_cwm.shape[0] != target_len:
            raise AlignmentError(
                f"target_cwm length {tgt_cwm.shape[0]} does not match target_ppm length {target_len}"
            )
        tgt_cwm_oriented = tgt_cwm if orientation == "+" else _reverse_complement(tgt_cwm)
        # Measured AT the chosen registration only -- never re-optimised. This
        # is the one line that keeps a sign-flipped pair from being invisible:
        # nothing above this point has ever looked at a CWM.
        signed_similarity = _cosine(src_cwm[s_start:s_end], tgt_cwm_oriented[t_start:t_end])

    return AlignmentEvidence(
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        orientation=orientation,
        offset=offset,
        overlap_bp=overlap_bp,
        overlap_frac_source=frac_source,
        overlap_frac_target=frac_target,
        ppm_similarity=score,
        signed_cwm_similarity=signed_similarity,
        empirical_p_value=float("nan"),
        null_shuffles=0,
        seed=0,
    )


def calibrate_pair_null(source_ppm: Any, target_ppm: Any, *, null_shuffles: int, seed: int,
                        min_overlap_bp: int = DEFAULT_MIN_OVERLAP_BP,
                        min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
                        ) -> tuple[float, list[float]]:
    """Empirical p-value for the observed PPM registration, against a null that
    re-runs the FULL registration pipeline on each shuffle.

    Each null draw permutes the target's row (position) order and calls
    `register_pair` again from scratch: a fresh offset x orientation search,
    not a rescore at the observed offset. Reusing the observed offset here
    would answer a different, easier question (see module docstring) and would
    make every alignment look more significant than it is.
    """
    import numpy as np

    if null_shuffles < 1:
        # The "no null shuffle can ever fail to register" argument in the loop
        # below only holds for null_shuffles >= 1; at 0 it degenerates to an
        # empty null_scores and would otherwise surface deep inside the loop
        # as an opaque "every null shuffle failed" refusal. Fail fast here
        # instead, with the actual cause.
        raise AlignmentError(f"null_shuffles must be >= 1 to calibrate a null; got {null_shuffles}")

    src = _as_matrix(source_ppm, "PPM")
    tgt = _as_matrix(target_ppm, "PPM")

    observed = register_pair(src, tgt, min_overlap_bp=min_overlap_bp,
                             min_overlap_frac=min_overlap_frac)
    observed_score = observed.ppm_similarity

    rng = np.random.default_rng(seed)
    null_scores: list[float] = []
    for _ in range(null_shuffles):
        shuffled_target = tgt[rng.permutation(tgt.shape[0])]
        try:
            shuffle_evidence = register_pair(
                src, shuffled_target, min_overlap_bp=min_overlap_bp,
                min_overlap_frac=min_overlap_frac,
            )
        except AlignmentError:
            # The overlap floor depends only on the two lengths, which a row
            # permutation never changes, so this is unreachable whenever the
            # observed call above succeeded. Kept defensive rather than
            # asserted away: a shuffle that cannot be registered is not
            # evidence for or against the observed alignment either way.
            continue
        null_scores.append(shuffle_evidence.ppm_similarity)

    if not null_scores:
        raise AlignmentError(
            "every null shuffle failed the bilateral overlap requirement; no null "
            "distribution could be calibrated for this pair"
        )

    n_at_least_as_extreme = sum(1 for s in null_scores if s >= observed_score)
    p_value = (n_at_least_as_extreme + 1) / (len(null_scores) + 1)
    return p_value, null_scores


# --------------------------------------------------------------------------- #
# The whole stage
# --------------------------------------------------------------------------- #
_EDGE_FIELDS = [f.name for f in AlignmentEvidence.__dataclass_fields__.values()]
#: Explicit dtypes for every edge column, applied whether or not there are any
#: rows. `pd.DataFrame([], columns=_EDGE_FIELDS)` alone infers `object` for
#: every column on an empty run (0 edges, or a registry of 0/1 nodes), which
#: silently loses the typed schema (int64/float64) a populated run produces --
#: a real difference for any downstream reader that assumes a fixed schema
#: across runs (e.g. concatenating edge tables, or numeric ops on `offset` /
#: `ppm_similarity`). `signed_cwm_similarity` is `float64` even though the
#: dataclass allows `None`: pandas already stores that column's `None` as NaN.
_EDGE_DTYPES: dict[str, str] = {
    "source_node_id": "string", "target_node_id": "string",
    "orientation": "string", "offset": "int64", "overlap_bp": "int64",
    "overlap_frac_source": "float64", "overlap_frac_target": "float64",
    "ppm_similarity": "float64", "signed_cwm_similarity": "float64",
    "empirical_p_value": "float64", "null_shuffles": "int64", "seed": "int64",
    "registered_on": "string", "registration_rule_version": "string",
}
_NULL_SUMMARY_FIELDS = [
    "source_node_id", "target_node_id", "null_shuffles", "seed",
    "empirical_p_value", "null_mean", "null_min", "null_max",
    "observed_ppm_similarity", "registration_rule_version",
]


def _write_edges(out: Path, edges: list[AlignmentEvidence]) -> Path:
    import pandas as pd

    dest = out / "alignment_edges.parquet"
    df = pd.DataFrame([e.to_dict() for e in edges], columns=_EDGE_FIELDS)
    df = df.astype(_EDGE_DTYPES)
    df.to_parquet(dest, index=False)
    return dest


def _write_null_summary(out: Path, rows: list[dict[str, Any]]) -> Path:
    dest = out / "alignment_null_summary.tsv"
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_NULL_SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return dest


def align_registry(registry_dir: str | os.PathLike[str], out_dir: str | os.PathLike[str],
                   *, null_shuffles: int = DEFAULT_NULL_SHUFFLES, seed: int = 0,
                   min_overlap_bp: int = DEFAULT_MIN_OVERLAP_BP,
                   min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
                   ) -> tuple[AlignmentRunSummary, list[AlignmentEvidence]]:
    """Register every pair of motifs in a registry, calibrate a null for each,
    and write both the edge table and the null summary.

    `null_shuffles` and `seed` are threaded through as the provenance the
    non-negotiable constraints require: every emitted edge carries both
    (`AlignmentEvidence.null_shuffles` / `.seed`), not just the run as a whole.
    """
    from motifmultiverse import guards
    from motifmultiverse.ingest import load_registry
    from motifmultiverse.provenance import record

    if null_shuffles < 1:
        # Fail fast on a run-level parameter, before opening the registry or
        # writing anything: every pair would otherwise hit this same refusal
        # one at a time inside the loop below (see calibrate_pair_null).
        raise AlignmentError(f"--null-shuffles must be >= 1; got {null_shuffles}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Provenance is written before the registry is even opened, same as
    # compile's pattern: a run that fails to load its registry still leaves a
    # record of what was attempted (T-09).
    prov = record("align", seed=seed)
    prov.write(out)

    meta, nodes, arrays = load_registry(registry_dir)
    try:
        edges: list[AlignmentEvidence] = []
        null_rows: list[dict[str, Any]] = []
        n_excluded = 0
        n_considered = 0
        for a, b in combinations(nodes, 2):
            a_id, b_id = a["node_id"], b["node_id"]
            a_arrays, b_arrays = arrays[a_id], arrays[b_id]
            n_considered += 1
            if "ppm" not in a_arrays or "ppm" not in b_arrays:
                # Never averaged, never guessed: a node with no PPM at all has
                # no unsigned content to register on, so the pair is excluded
                # rather than silently registered on CWM alone.
                n_excluded += 1
                continue
            a_ppm, b_ppm = a_arrays["ppm"][:], b_arrays["ppm"][:]
            a_cwm = a_arrays["cwm"][:] if "cwm" in a_arrays else None
            b_cwm = b_arrays["cwm"][:] if "cwm" in b_arrays else None
            try:
                evidence = register_pair(
                    a_ppm, b_ppm, source_cwm=a_cwm, target_cwm=b_cwm,
                    min_overlap_bp=min_overlap_bp, min_overlap_frac=min_overlap_frac,
                )
            except AlignmentError:
                n_excluded += 1
                continue
            p_value, null_scores = calibrate_pair_null(
                a_ppm, b_ppm, null_shuffles=null_shuffles, seed=seed,
                min_overlap_bp=min_overlap_bp, min_overlap_frac=min_overlap_frac,
            )
            evidence = replace(
                evidence, source_node_id=a_id, target_node_id=b_id,
                empirical_p_value=p_value, null_shuffles=null_shuffles, seed=seed,
            )
            edges.append(evidence)
            null_rows.append({
                "source_node_id": a_id, "target_node_id": b_id,
                "null_shuffles": null_shuffles, "seed": seed,
                "empirical_p_value": p_value,
                "null_mean": sum(null_scores) / len(null_scores),
                "null_min": min(null_scores), "null_max": max(null_scores),
                "observed_ppm_similarity": evidence.ppm_similarity,
                "registration_rule_version": evidence.registration_rule_version,
            })
    finally:
        arrays.close()

    guards.sign_alignment([e.to_dict() for e in edges]).raise_if_failed()

    edges_path = _write_edges(out, edges)
    null_summary_path = _write_null_summary(out, null_rows)

    summary = AlignmentRunSummary(
        n_nodes=len(nodes),
        n_pairs_considered=n_considered,
        n_edges=len(edges),
        n_pairs_excluded=n_excluded,
        null_shuffles=null_shuffles,
        seed=seed,
        registration_rule_version=REGISTRATION_RULE_VERSION,
        edges_path=str(edges_path),
        null_summary_path=str(null_summary_path),
    )
    return summary, edges


#: The stage entry point named in `__all__`; kept as a plain alias so a caller
#: importing `align.run` gets exactly `align_registry`, not a wrapper that
#: could drift from it.
run = align_registry
