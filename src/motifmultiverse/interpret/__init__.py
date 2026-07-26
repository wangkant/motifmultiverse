"""Describe what is inside a peak set, at the strength its selection licenses.

This is the first module in the package with a real body, and it was chosen for
that on purpose: it is the only one of the nine that needs neither TF-MoDISco nor
a hit-caller backend. It consumes a **frozen** hit table and answers subset
queries over it, so it runs end to end with no external tool installed and serves
as the interface template for the rest.

Three things happen, in this order, and the order is the design:

1. **Resolve the output mode from the declared selection provenance.** Before any
   number is computed. A peak set chosen by the same signal that is about to be
   measured can produce a statistically valid, semantically circular result
   (``BA-16``), so what the query is *allowed* to emit is settled first.
2. **Compute three health numbers.** Intersection coverage, blocks spanned, and
   the fraction the frozen lexicon explains. If any falls below its
   pre-registered floor, the reading is **suppressed** -- not annotated. A
   disclaimer beside an effect size does not travel; the effect size does.
3. **Emit at the licensed strength.** Full inference, held-out inference, or a
   descriptive decomposition with no interval and no *p* value.

Nothing here re-runs a caller. Every specification is a subset of the one frozen
run the hit table came from (``FP-17``).
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from motifmultiverse import guards
from motifmultiverse.schema import (
    IMPLEMENTED_ESTIMATORS,
    MISSING_SENTINEL,
    Estimator,
    HealthFloors,
    HitRecord,
    Missingness,
    OutputMode,
    PeakSetQuery,
    SelectionProvenance,
)

__all__ = [
    "InterpretError", "HealthReport", "FamilyComposition", "FamilyEffect",
    "Interpretation", "read_hit_table", "read_peak_set", "peak_universe",
    "health_report", "compose", "estimate_effects", "interpret_query",
    "ESTIMATOR", "DEFAULT_BLOCK_SIZE", "DEFAULT_BOOTSTRAP",
]

#: Estimator actually implemented here. ``FP-15`` specifies a BCa paired block
#: bootstrap for intervals and a block-level wild cluster bootstrap-t for p
#: values; this is the percentile block bootstrap, which is weaker. It is named in
#: every result rather than described as "block bootstrap", because the gap
#: between what was specified and what ran is exactly the thing that goes missing.
#: The full set of recognised values travels with every result
#: (:class:`schema.Estimator`), so a caller can branch on it now and keep working
#: when the specified estimators arrive.
ESTIMATOR = Estimator.PERCENTILE_BLOCK_BOOTSTRAP.value

DEFAULT_BLOCK_SIZE = 1_000_000
DEFAULT_BOOTSTRAP = 2000


class InterpretError(ValueError):
    """A query cannot be answered at the strength it asked for."""


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _coerce_row(row: dict[str, Any]) -> HitRecord:
    coeff = row.get("hit_coefficient")
    if coeff in ("", None, MISSING_SENTINEL):
        coeff = None
    return HitRecord(
        region_id=str(row["region_id"]),
        chrom=str(row["chrom"]),
        start=int(row["start"]),
        end=int(row["end"]),
        missingness=Missingness(str(row["missingness"])),
        input_scale=int(row["input_scale"]),
        lexicon_id=str(row["lexicon_id"]),
        variant_id=str(row.get("variant_id") or MISSING_SENTINEL),
        family_id=str(row.get("family_id") or MISSING_SENTINEL),
        hit_coefficient=None if coeff is None else float(coeff),
    )


def read_hit_table(path: str | os.PathLike[str]) -> list[HitRecord]:
    """Read a frozen hit table (``.tsv``/``.txt`` or ``.parquet``).

    The parquet backend is imported lazily so that a missing dependency produces a
    sentence rather than a traceback.
    """
    p = Path(path)
    if p.suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise InterpretError(
                "reading a parquet hit table needs pandas + pyarrow; "
                "install them or supply the table as TSV"
            ) from exc
        rows: Iterable[dict[str, Any]] = pd.read_parquet(p).to_dict("records")
    else:
        with open(p, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
    records = [_coerce_row(dict(r)) for r in rows]
    if not records:
        raise InterpretError(f"{p} contains no rows")
    scales = {r.input_scale for r in records}
    if len(scales) != 1:
        raise InterpretError(
            f"{p} spans {len(scales)} input scales {sorted(scales)}. The hit caller is not "
            "input-scale invariant, so a table assembled from more than one run is not a "
            "substrate (FP-17)."
        )
    lexicons = {r.lexicon_id for r in records}
    if len(lexicons) != 1:
        raise InterpretError(f"{p} mixes lexicons {sorted(lexicons)}; freeze one lexicon per table")
    return records


def read_peak_set(path: str | os.PathLike[str]) -> list[str]:
    """Read a peak set as region ids: a BED (name column, else coordinates) or a list."""
    p = Path(path)
    out: list[str] = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "track", "browser")):
                continue
            fields = stripped.split("\t") if "\t" in stripped else stripped.split()
            if len(fields) >= 4:
                out.append(fields[3])
            elif len(fields) == 3:
                out.append(f"{fields[0]}:{fields[1]}-{fields[2]}")
            else:
                out.append(fields[0])
    if not out:
        raise InterpretError(f"{p} contains no regions")
    return out


# --------------------------------------------------------------------------- #
# Peak-level view of the substrate
# --------------------------------------------------------------------------- #
@dataclass
class Peak:
    """One peak, aggregated from its hit rows (``BA-07``: instances are not samples).

    Occupancy (``family_hit_count``) and signed coefficient mass
    (``family_coefficient_sum``) are tracked separately: opposite-signed hits in
    the same family cancel the mass to zero without erasing the occupancy, which
    is the scientific point of separating them. ``family_abs_coefficient_sum`` is
    the unsigned counterpart, so magnitude is not lost to cancellation either.
    """

    region_id: str
    chrom: str
    start: int
    end: int
    block: tuple[str, int]
    searched: bool = True
    family_hit_count: dict[str, int] = field(default_factory=dict)
    family_coefficient_sum: dict[str, float] = field(default_factory=dict)
    family_abs_coefficient_sum: dict[str, float] = field(default_factory=dict)

    @property
    def has_used_hit(self) -> bool:
        return any(n > 0 for n in self.family_hit_count.values())

    def add_used_hit(self, family_id: str, coefficient: float) -> None:
        self.family_hit_count[family_id] = self.family_hit_count.get(family_id, 0) + 1
        self.family_coefficient_sum[family_id] = (
            self.family_coefficient_sum.get(family_id, 0.0) + coefficient
        )
        self.family_abs_coefficient_sum[family_id] = (
            self.family_abs_coefficient_sum.get(family_id, 0.0) + abs(coefficient)
        )


def peak_universe(hits: Sequence[HitRecord], block_size: int) -> dict[str, Peak]:
    """Aggregate hit rows to peaks.

    The universe is every peak the table mentions, **including** peaks whose only
    row records that nothing was found. Deriving the universe from called hits
    alone would drop exactly those peaks and inflate every ratio computed from it
    (``BA-01``, ``BA-10``).

    ``NOT_SEARCHED`` is not a zero: such peaks are excluded from every denominator
    rather than counted as having no motif. ``NO_SEQUENCE_MATCH`` and
    ``HIT_BELOW_FLOOR`` *are* measurements, and contribute 0.

    A region is rejected outright, before any hit is aggregated, if its rows
    disagree about what the region *is*: ``NOT_SEARCHED`` mixed with any measured
    state means the table cannot decide whether the region was looked at, and
    coordinates that move mean it is not clear it is the same region at all.
    """
    by_region: dict[str, list[HitRecord]] = {}
    for h in hits:
        by_region.setdefault(h.region_id, []).append(h)

    peaks: dict[str, Peak] = {}
    for region_id, rows in by_region.items():
        first = rows[0]
        for h in rows[1:]:
            if (h.chrom, h.start, h.end) != (first.chrom, first.start, first.end):
                raise InterpretError(
                    f"{region_id}: inconsistent coordinates across its rows "
                    f"({first.chrom}:{first.start}-{first.end} vs {h.chrom}:{h.start}-{h.end})"
                )
        states = {h.missingness for h in rows}
        if Missingness.NOT_SEARCHED in states and len(states) > 1:
            raise InterpretError(
                f"{region_id}: mixes NOT_SEARCHED and measured rows; a region cannot be both "
                "unsearched and observed"
            )
        peak = Peak(
            region_id=region_id, chrom=first.chrom, start=first.start, end=first.end,
            block=first.block(block_size), searched=Missingness.NOT_SEARCHED not in states,
        )
        for h in rows:
            if h.missingness is Missingness.USED:
                peak.add_used_hit(h.family_id, float(h.hit_coefficient))
        peaks[region_id] = peak
    return peaks


# --------------------------------------------------------------------------- #
# Step 2: the three health numbers
# --------------------------------------------------------------------------- #
@dataclass
class HealthReport:
    """Three numbers, each with its denominator, produced before any effect."""

    n_submitted: int
    n_in_universe: int
    intersection_coverage: float | None
    n_blocks: int
    block_size: int
    n_searched: int
    n_with_used_hit: int
    explained_fraction: float | None
    floors: dict[str, float]
    floor_failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.floor_failures


def health_report(peaks: dict[str, Peak], submitted: Sequence[str],
                  floors: HealthFloors, block_size: int) -> HealthReport:
    """Coverage, blocks and explained fraction -- each reported with its denominator."""
    submitted_unique = list(dict.fromkeys(submitted))
    present = [peaks[r] for r in submitted_unique if r in peaks]
    coverage = len(present) / len(submitted_unique) if submitted_unique else None
    blocks = {p.block for p in present}
    searched = [p for p in present if p.searched]
    with_hit = [p for p in searched if p.has_used_hit]
    explained = len(with_hit) / len(searched) if searched else None

    failures: list[str] = []
    if coverage is None or coverage < floors.min_intersection_coverage:
        shown = "undefined" if coverage is None else round(coverage, 6)
        failures.append(
            f"intersection_coverage={shown} < floor {floors.min_intersection_coverage}"
        )
    if len(blocks) < floors.min_blocks:
        failures.append(f"n_blocks={len(blocks)} < floor {floors.min_blocks}")
    if explained is None or explained < floors.min_explained_fraction:
        shown = "undefined" if explained is None else round(explained, 6)
        failures.append(f"explained_fraction={shown} < floor {floors.min_explained_fraction}")

    return HealthReport(
        n_submitted=len(submitted_unique),
        n_in_universe=len(present),
        intersection_coverage=coverage,
        n_blocks=len(blocks),
        block_size=block_size,
        n_searched=len(searched),
        n_with_used_hit=len(with_hit),
        explained_fraction=explained,
        floors={
            "min_intersection_coverage": floors.min_intersection_coverage,
            "min_blocks": float(floors.min_blocks),
            "min_explained_fraction": floors.min_explained_fraction,
        },
        floor_failures=failures,
    )


# --------------------------------------------------------------------------- #
# Step 3a: descriptive decomposition
# --------------------------------------------------------------------------- #
@dataclass
class FamilyComposition:
    family_id: str
    n_peaks_with_family: int
    n_peaks_searched: int
    peak_share: float
    mean_coefficient_per_peak: float


def compose(peaks: dict[str, Peak], region_ids: Sequence[str]) -> list[FamilyComposition]:
    """Per-family occupancy over the queried peaks. Descriptive: no interval, no test."""
    searched = [peaks[r] for r in dict.fromkeys(region_ids) if r in peaks and peaks[r].searched]
    if not searched:
        return []
    families = sorted({fam for p in searched for fam in p.family_hit_count})
    out = []
    for fam in families:
        vals = [p.family_coefficient_sum.get(fam, 0.0) for p in searched]
        n_with = sum(1 for p in searched if p.family_hit_count.get(fam, 0) > 0)
        out.append(FamilyComposition(
            family_id=fam,
            n_peaks_with_family=n_with,
            n_peaks_searched=len(searched),
            peak_share=n_with / len(searched),
            mean_coefficient_per_peak=sum(vals) / len(vals),
        ))
    return sorted(out, key=lambda c: -c.mean_coefficient_per_peak)


# --------------------------------------------------------------------------- #
# Step 3b: inference, when the selection licenses it
# --------------------------------------------------------------------------- #
@dataclass
class FamilyEffect:
    id: str
    family_id: str
    comparator_id: str
    is_cross_condition: bool
    effect: float
    ci: tuple[float, float] | None
    p_value: float | None
    q_value: float | None
    n_query_peaks: int
    n_comparator_peaks: int
    n_blocks: int
    n_bootstrap: int
    n_bootstrap_valid: int
    block_size: int
    random_seed: int
    estimator: str = ESTIMATOR


def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _bh(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg over the families tested in this call, and only those."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    q = [0.0] * m
    running = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        running = min(running, p_values[i] * m / (m - rank + 1))
        q[i] = min(1.0, running)
    return q


def estimate_effects(peaks: dict[str, Peak], query_ids: Sequence[str],
                     comparator_ids: Sequence[str], comparator_id: str,
                     n_bootstrap: int, seed: int, block_size: int) -> list[FamilyEffect]:
    """Per-family difference in mean per-peak coefficient, query minus comparator.

    Whole genomic blocks are the resampling unit, not peaks: peaks within a block
    are not independent, and a peak-level bootstrap would report an interval far
    narrower than the data support (``BA-07``, ``FP-15``).
    """
    q_peaks = [peaks[r] for r in dict.fromkeys(query_ids) if r in peaks and peaks[r].searched]
    c_peaks = [peaks[r] for r in dict.fromkeys(comparator_ids) if r in peaks and peaks[r].searched]
    if not q_peaks or not c_peaks:
        raise InterpretError(
            f"effects need searched peaks on both sides: query={len(q_peaks)}, "
            f"comparator={len(c_peaks)}"
        )
    families = sorted({fam for p in (*q_peaks, *c_peaks) for fam in p.family_coefficient_sum})
    if not families:
        return []

    by_block: dict[tuple[str, int], tuple[list[Peak], list[Peak]]] = {}
    for p in q_peaks:
        by_block.setdefault(p.block, ([], []))[0].append(p)
    for p in c_peaks:
        by_block.setdefault(p.block, ([], []))[1].append(p)
    blocks = sorted(by_block)

    rng = random.Random(seed)
    replicates: list[list[float]] = [[] for _ in families]
    for _ in range(n_bootstrap):
        drawn = [by_block[blocks[rng.randrange(len(blocks))]] for _ in blocks]
        q_draw = [p for qs, _ in drawn for p in qs]
        c_draw = [p for _, cs in drawn for p in cs]
        if not q_draw or not c_draw:
            continue
        for k, fam in enumerate(families):
            replicates[k].append(
                _mean([p.family_coefficient_sum.get(fam, 0.0) for p in q_draw])
                - _mean([p.family_coefficient_sum.get(fam, 0.0) for p in c_draw])
            )

    p_values: list[float] = []
    effects: list[FamilyEffect] = []
    for k, fam in enumerate(families):
        point = (_mean([p.family_coefficient_sum.get(fam, 0.0) for p in q_peaks])
                 - _mean([p.family_coefficient_sum.get(fam, 0.0) for p in c_peaks]))
        reps = sorted(replicates[k])
        n_valid = len(reps)
        if n_valid:
            lo = reps[max(0, int(math.floor(0.025 * n_valid)) - 1)]
            hi = reps[min(n_valid - 1, int(math.ceil(0.975 * n_valid)) - 1)]
            n_le = sum(1 for r in reps if r <= 0.0)
            n_ge = sum(1 for r in reps if r >= 0.0)
            # Floored at 1/(B+1): a bootstrap cannot resolve a p value finer than
            # its own replicate count, and reporting one that looks finer is a
            # claim about resolution the procedure does not have.
            p = max(min(1.0, 2.0 * min(n_le, n_ge) / n_valid), 1.0 / (n_valid + 1))
            ci: tuple[float, float] | None = (lo, hi)
        else:
            p, ci = float("nan"), None
        p_values.append(p)
        effects.append(FamilyEffect(
            id=f"{fam}_vs_{comparator_id}",
            family_id=fam,
            comparator_id=comparator_id,
            is_cross_condition=True,
            effect=point,
            ci=ci,
            p_value=None if math.isnan(p) else p,
            q_value=None,
            n_query_peaks=len(q_peaks),
            n_comparator_peaks=len(c_peaks),
            n_blocks=len(blocks),
            n_bootstrap=n_bootstrap,
            n_bootstrap_valid=n_valid,
            block_size=block_size,
            random_seed=seed,
        ))
    if all(not math.isnan(p) for p in p_values):
        for eff, q in zip(effects, _bh(p_values), strict=True):
            eff.q_value = q
    return effects


# --------------------------------------------------------------------------- #
# The whole query
# --------------------------------------------------------------------------- #
@dataclass
class Interpretation:
    query_id: str
    selection_provenance: str
    output_mode: str
    emitted_order: list[str]
    health: dict[str, Any]
    floor_failures: list[str]
    interpretation_emitted: bool
    suppression_reason: str | None
    composition: list[dict[str, Any]] | None
    effects: list[dict[str, Any]] | None
    notes: list[str]
    input_scale: int
    lexicon_id: str
    estimator: str = ESTIMATOR
    #: Every recognised value, and the subset that exists in this release. A
    #: consumer branches on `estimator` against this list rather than against a
    #: literal, so adding FP-15's estimators does not change the caller.
    estimators_defined: list[str] = field(
        default_factory=lambda: [e.value for e in Estimator])
    estimators_implemented: list[str] = field(
        default_factory=lambda: sorted(e.value for e in IMPLEMENTED_ESTIMATORS))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, out_dir: str | os.PathLike[str]) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "interpretation.json"
        dest.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return dest


_UNVERIFIABLE = OutputMode.DESCRIPTIVE_ONLY_UNVERIFIABLE_CONDITIONING


def interpret_query(hits: Sequence[HitRecord], query: PeakSetQuery,
                    floors: HealthFloors | None = None,
                    block_size: int = DEFAULT_BLOCK_SIZE,
                    n_bootstrap: int = DEFAULT_BOOTSTRAP,
                    seed: int = 0) -> Interpretation:
    """Answer one peak-set query at the strength its selection provenance licenses."""
    floors = floors or HealthFloors()
    mode = query.output_mode                       # (1) before any number is computed
    notes: list[str] = []
    if query.selection_provenance is SelectionProvenance.DECLARATION_MISSING:
        notes.append(
            "selection_provenance was not declared; recorded as DECLARATION_MISSING and run in "
            "the most conservative mode. That is not the same as EXTERNAL."
        )
    if mode is _UNVERIFIABLE:
        notes.append(
            "the conditioning set of this selection cannot be verified: it cannot be shown that "
            "downstream information was not already visible when the peak set was chosen"
        )

    peaks = peak_universe(hits, block_size)
    region_ids = list(query.region_ids)
    comparator_ids = list(query.comparator_region_ids)
    if mode is OutputMode.FULL_INFERENCE_HELD_OUT:
        held_out = set(query.held_out_region_ids)
        if not held_out:
            raise InterpretError(
                "CLUSTERED_WITH_SPLIT claims a held-out half but none was supplied. The grade's "
                "entire licence is the split; without it the query is CLUSTERED_NO_SPLIT."
            )
        region_ids = [r for r in region_ids if r in held_out]
        comparator_ids = [r for r in comparator_ids if r in held_out]
        notes.append(f"inference restricted to the held-out half ({len(region_ids)} query peaks)")

    if not region_ids:
        raise InterpretError(
            f"{query.query_id}: the query is empty, so no ratio it produces has a denominator"
        )

    health = health_report(peaks, region_ids, floors, block_size)   # (2) health, always
    emitted = ["health"]

    composition: list[dict[str, Any]] | None = None
    effects: list[dict[str, Any]] | None = None
    suppression: str | None = None

    if not health.passed:
        # (3) Suppression, not annotation. A caveat beside a number does not travel.
        suppression = (
            "reading suppressed: " + "; ".join(health.floor_failures)
            + ". The health numbers above stand; no composition and no effect is reported."
        )
    else:
        composition = [asdict(c) for c in compose(peaks, region_ids)]
        emitted.append("composition")
        if mode in (OutputMode.FULL_INFERENCE, OutputMode.FULL_INFERENCE_HELD_OUT):
            if query.comparator_id == MISSING_SENTINEL or not comparator_ids:
                raise InterpretError(
                    "a cross-condition effect needs a named baseline peak set. One set of "
                    "measurements once supported both 'replicates exactly' and 'four times "
                    "stronger, prediction falsified', differing only in the comparator; no "
                    "baseline, no number (BA-18)."
                )
            effects = [asdict(e) for e in estimate_effects(
                peaks, region_ids, comparator_ids, query.comparator_id,
                n_bootstrap=n_bootstrap, seed=seed, block_size=block_size)]
            emitted.append("effects")
            guards.comparator_declared(effects).raise_if_failed()
        else:
            notes.append(
                f"{mode.value}: descriptive decomposition only. No interval and no p value is "
                "reported, because the selection criterion cannot be separated from the signal."
            )

    result = Interpretation(
        query_id=query.query_id,
        selection_provenance=query.selection_provenance.value,
        output_mode=mode.value,
        emitted_order=emitted,
        health=asdict(health),
        floor_failures=health.floor_failures,
        interpretation_emitted=composition is not None,
        suppression_reason=suppression,
        composition=composition,
        effects=effects,
        notes=notes,
        input_scale=hits[0].input_scale,
        lexicon_id=hits[0].lexicon_id,
    )

    # The guards run on the finished record, not on intermediate state, so that
    # what is checked is exactly what a reader would receive.
    guards.selection_provenance_declared([{
        "query_id": query.query_id,
        "selection_provenance": query.selection_provenance.value,
        "output_mode": mode.value,
    }]).raise_if_failed()
    guards.health_before_effect({
        "health": result.health,
        "emitted_order": result.emitted_order,
        "floor_failures": result.floor_failures,
        "effects": result.effects or [],
        "interpretation_emitted": result.interpretation_emitted,
    }).raise_if_failed()
    guards.single_scale([{"input_scale": h.input_scale} for h in hits]).raise_if_failed()
    return result
