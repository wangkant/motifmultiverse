"""Describe what is inside a peak set, at the strength its selection licenses.

This is the first module in the package with a real body, and it was chosen for
that on purpose: it is the only one of the nine that needs neither TF-MoDISco nor
a hit-caller backend. It consumes a **frozen** hit table and answers subset
queries over it, so it runs end to end with no external tool installed and serves
as the interface template for the rest.

Three things happen, in this order, and the order is the design:

1. **Resolve two independent things from the declared selection provenance.**
   Before any number is computed: ``statistical_license`` (may this query
   support inference at all) and ``claim_scope`` (what the resulting number can
   be a claim about). A peak set chosen by the same signal that is about to be
   measured can produce a result that is statistically valid *and*
   semantically circular (``BA-16``) -- the two questions do not covary, so
   they are resolved separately (``schema.identity.resolve_query_permissions``)
   rather than as one grade, and what the query is *allowed* to emit is settled
   first.
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
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from sys import intern
from typing import Any

from motifmultiverse import guards, infer
from motifmultiverse.schema import (
    ESTIMATOR_CAPABILITY,
    HIT_TABLE_COLUMNS,
    IMPLEMENTED_ESTIMATORS,
    MISSING_SENTINEL,
    ClaimScope,
    Estimator,
    HealthFloors,
    HitRecord,
    InferenceCapability,
    Missingness,
    PeakSetQuery,
    SelectionProvenance,
    StatisticalLicense,
)

__all__ = [
    "InterpretError", "HealthReport", "ContrastHealth", "FamilyComposition",
    "FamilyEffect", "Interpretation", "read_hit_table", "read_peak_set",
    "peak_universe", "health_report", "contrast_health_report", "compose",
    "estimate_effects", "two_part_effects", "interpret_query",
    "ESTIMATOR", "ESTIMATOR_PERCENTILE", "ESTIMATOR_BCA_WILD", "ESTIMATOR_CHOICES",
    "CAPABILITY", "DEFAULT_BLOCK_SIZE", "DEFAULT_BOOTSTRAP", "MIN_PERCENTILE_REPLICATES",
]

#: The weaker of the two estimator paths: a percentile block bootstrap. It is
#: named in every result rather than described as "block bootstrap", because the
#: gap between what was specified and what ran is exactly the thing that goes
#: missing. The full set of recognised values travels with every result
#: (:class:`schema.Estimator`), so a caller branches on it rather than on a
#: literal.
ESTIMATOR_PERCENTILE = Estimator.PERCENTILE_BLOCK_BOOTSTRAP.value

#: ``FP-15``'s specified pair, in one selectable path: a BCa paired
#: genomic-block bootstrap interval (:func:`infer.bca_paired_block_interval`)
#: and a block-level wild cluster bootstrap-*t* p value
#: (:func:`infer.wild_cluster_bootstrap_t`). The recorded value names the half
#: that decides the result's capability -- the *test* -- because that is what a
#: reader must not be wrong about; the interval half is named in the run's
#: notes. This is the only value in :class:`schema.Estimator` licensed
#: ``INTERVAL_AND_TEST``.
ESTIMATOR_BCA_WILD = Estimator.WILD_CLUSTER_BOOTSTRAP_T.value

#: Estimator used when a caller asks for none. Deliberately the conservative
#: one: a default that emits p values is a default that emits them to callers
#: who never decided they wanted a hypothesis test.
ESTIMATOR = ESTIMATOR_PERCENTILE

#: What ``--estimator`` accepts, mapped to the ``schema.Estimator`` value the
#: result records. The command-line spelling names both halves of the pair
#: (``bca-wild-cluster``) so that what runs is legible from the command line;
#: the recorded value names the capability-licensing half. The table is public
#: so the CLI does not restate the choices as literals and drift from them.
ESTIMATOR_CHOICES: dict[str, str] = {
    "percentile": ESTIMATOR_PERCENTILE,
    "bca-wild-cluster": ESTIMATOR_BCA_WILD,
}

#: What ``ESTIMATOR`` -- the *default* -- is licensed to emit
#: (``schema.ESTIMATOR_CAPABILITY``). The percentile block bootstrap's replicate
#: tail is not a calibrated hypothesis test, so this is ``ESTIMATION_ONLY``:
#: effects on that path carry a point estimate and a percentile interval, and
#: never a p or q value. A run that selects ``ESTIMATOR_BCA_WILD`` resolves its
#: own capability from the same table rather than reading this constant, so a
#: caller branches on the capability *in the result* -- never on this module
#: attribute, which describes only the default.
CAPABILITY = ESTIMATOR_CAPABILITY[Estimator(ESTIMATOR)]

DEFAULT_BLOCK_SIZE = 1_000_000
DEFAULT_BOOTSTRAP = 2000

#: Fewest replicates a 95% percentile interval may be computed from, derived
#: rather than chosen: the finest tail probability ``B`` replicates can resolve
#: is ``1/(B+1)``, so a 2.5% tail needs ``B + 1 >= 1/0.025``, i.e. ``B >= 39``.
#: Below it the two endpoints are the extreme replicates and the interval's real
#: coverage is ``1 - 2/(B+1)``, not 0.95 -- at ``B = 1`` that is zero, and the
#: estimator emitted ``[x, x]``: a zero-width 95% interval, printed beside its
#: point estimate, which reads as infinite precision rather than as one draw.
#: The percentile path is refused below this rather than degraded, for the same
#: reason `infer` refuses below `MIN_ESTIMABLE_BLOCKS`: an interval that cannot
#: be computed must not be reported narrower than one that can.
MIN_PERCENTILE_REPLICATES = 39


class InterpretError(ValueError):
    """A query cannot be answered at the strength it asked for."""


def _resolve_estimator(name: str) -> str:
    """Map a caller's estimator spelling onto the ``schema.Estimator`` recorded.

    An unrecognised name is refused, never mapped onto the default: silently
    running the weaker estimator for a caller who asked for the stronger one
    produces a result whose ``estimator`` field is true and whose *provenance in
    the caller's head* is false, which is the failure this project exists to
    prevent.
    """
    if name in ESTIMATOR_CHOICES:
        return ESTIMATOR_CHOICES[name]
    if name in set(ESTIMATOR_CHOICES.values()):
        return name
    accepted = sorted(set(ESTIMATOR_CHOICES) | set(ESTIMATOR_CHOICES.values()))
    raise InterpretError(
        f"unknown estimator {name!r}; this release implements {accepted}. "
        "Refusing to fall back to another estimator than the one requested."
    )


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _text(value: Any, column: str, *, default: str | None = None) -> str:
    """Read one text cell, treating every spelling of "no value" as absent.

    ``x or SENTINEL`` does not do this, and the reason is specific: pandas returns
    a null in an object column as float ``NaN``, and ``NaN`` is **truthy**. So the
    fallback was skipped and ``str(nan)`` produced the literal string ``"nan"`` --
    an identifier that is not the sentinel, so every check written against the
    sentinel let it through. A parquet table with a null family on some rows was
    accepted and reported a family called ``nan``, while the byte-equivalent TSV
    was correctly refused.

    ``default=None`` means the column is required: a null there is an error, not a
    sentinel, because a peak or a chromosome named ``NA`` is a fabricated row
    rather than a recorded absence.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if default is None:
            raise InterpretError(
                f"hit table has an empty {column}; it is required on every row and "
                "cannot be defaulted, because the default is indistinguishable from "
                "a real value"
            )
        return default
    text = str(value)
    if text == "":
        if default is None:
            raise InterpretError(f"hit table has an empty {column}; it is required on every row")
        return default
    return text


def _coerce_row(row: dict[str, Any]) -> HitRecord:
    """Build one HitRecord, sharing the strings that repeat down the column.

    A frozen hit table is one row per (peak, variant), and most of its string
    columns are near-constant: on a real K562 substrate of 576,589 rows,
    ``substrate_id`` and ``lexicon_id`` had **one** distinct value each, ``chrom``
    15, ``variant_id`` 17, ``family_id`` 12. Storing a separate 64-character
    ``substrate_id`` object per row is 576,589 copies of the same digest.

    ``sys.intern`` makes each distinct value one object. With ``slots=True`` on
    HitRecord this took the same table from 824 MB to the figure quoted in
    interpret/README.md; the point is not the constant factor but the ceiling --
    a genome-wide table has to fit at all.
    """
    coeff = row.get("hit_coefficient")
    if coeff in ("", None, MISSING_SENTINEL):
        coeff = None
    elif isinstance(coeff, float) and math.isnan(coeff):
        # NaN is how parquet and pandas spell "no value in this float column"; a
        # TSV spells the same thing as an empty field. Without this the two
        # encodings of one table disagreed: the TSV read fine and the parquet
        # raised on its first non-USED row, which is every realistic table -- the
        # upstream opportunity table is 4.48M rows of mostly SEARCHED_NOT_RETAINED.
        # NaN is also not a measurement: left as a float it reached HitRecord on
        # USED rows, poisoned every family mean it entered, and was serialised as
        # a bare `NaN` token that is not valid JSON. Absence takes no number, so
        # it becomes None here and a USED row carrying it is refused downstream
        # by the rule that a used hit must have a coefficient.
        coeff = None
    return HitRecord(
        region_id=intern(_text(row["region_id"], "region_id")),
        chrom=intern(_text(row["chrom"], "chrom")),
        start=int(row["start"]),
        end=int(row["end"]),
        missingness=Missingness(_text(row["missingness"], "missingness")),
        input_scale=int(row["input_scale"]),
        lexicon_id=intern(_text(row["lexicon_id"], "lexicon_id")),
        substrate_id=intern(_text(row.get("substrate_id"), "substrate_id",
                                  default=MISSING_SENTINEL)),
        variant_id=intern(_text(row.get("variant_id"), "variant_id",
                                default=MISSING_SENTINEL)),
        family_id=intern(_text(row.get("family_id"), "family_id",
                               default=MISSING_SENTINEL)),
        hit_coefficient=None if coeff is None else float(coeff),
    )


def _require_hit_table_columns(present: list[str], path: Path) -> None:
    """Refuse a hit table that does not carry every declared column.

    ``schema.HIT_TABLE_COLUMNS`` is the documented contract (``interpret/README.md``
    says the table has "columns from ``schema.HIT_TABLE_COLUMNS``") but nothing used
    to check it on read. Two failures followed, and both are worse than a refusal:

    * a table with no ``family_id`` was accepted and every row took the sentinel,
      so every family-level share, effect and CI was computed for one fabricated
      family named ``NA`` -- a plausible number that is wrong, which is the exact
      category this tool exists to prevent;
    * a table with a missing or renamed positional column raised a bare
      ``KeyError`` out of ``_coerce_row``, i.e. a traceback and an exit code
      outside the documented contract, in the module whose stated principle is
      that a bad input produces a sentence.
    """
    missing = [c for c in HIT_TABLE_COLUMNS if c not in present]
    if missing:
        raise InterpretError(
            f"{path} is missing required hit-table column(s): {', '.join(missing)}. "
            f"A frozen hit table must carry all of {', '.join(HIT_TABLE_COLUMNS)}; "
            "absent columns cannot be defaulted because the default is "
            "indistinguishable from a real value."
        )


def verify_against_manifest(hits: Sequence[HitRecord], manifest: Any, action: str) -> None:
    """Refuse a hit table that is not the whole frozen run its manifest describes.

    Checking ``substrate_id`` alone is not enough. The id travels with the rows, so
    a *truncated* table keeps it: drop regions and the table still claims to be the
    frozen run. Nothing downstream can notice, because every coverage figure is a
    fraction of the universe that was handed in -- shrink the universe and a query
    that the run never covered reports ``intersection_coverage = 1.0``. The manifest
    is the only independent statement of how large the run actually was, so it has
    to be compared against, not merely matched on identity.

    ``action`` is the caller's own verb, so the two call sites cannot drift into
    each other's message -- which is exactly what had happened: ``interpret`` said
    "refusing to infer" and ``infer`` said "refusing to interpret".
    """
    if hits[0].substrate_id != manifest.substrate_id:
        raise InterpretError(
            "hit table substrate_id does not match --substrate-manifest; "
            f"refusing to {action} over a different frozen caller run"
        )
    n_regions = len({h.region_id for h in hits})
    if n_regions != manifest.n_regions:
        raise InterpretError(
            f"hit table covers {n_regions} regions but --substrate-manifest declares "
            f"{manifest.n_regions}. A truncated substrate keeps its substrate_id and "
            "silently shrinks the universe, so every peak set looks fully present and "
            f"intersection_coverage reports 1.0 for regions the run never covered. "
            f"Refusing to {action} over a partial substrate."
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
        frame = pd.read_parquet(p)
        _require_hit_table_columns(list(frame.columns), p)
        # Column lists, then one transient dict per row. `to_dict("records")`
        # would materialise every row at once and hold it alongside the records
        # being built from it.
        columns = {name: frame[name].tolist() for name in HIT_TABLE_COLUMNS}
        del frame
        records = [
            _coerce_row({name: values[i] for name, values in columns.items()})
            for i in range(len(next(iter(columns.values()), ())))
        ]
        del columns
    else:
        with open(p, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            _require_hit_table_columns(list(reader.fieldnames or []), p)
            # Streamed on purpose. `list(reader)` held one dict per row and
            # `_coerce_row(dict(r))` then copied each one, so a table briefly
            # existed three times over: on a real 576,589-row substrate that was
            # the difference between fitting and not.
            records = [_coerce_row(row) for row in reader]
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
    substrate_ids = {r.substrate_id for r in records}
    if MISSING_SENTINEL in substrate_ids:
        raise InterpretError(f"{p} has rows without a substrate_id; frozen hit tables must name one")
    if len(substrate_ids) != 1:
        raise InterpretError(
            f"{p} mixes substrates {sorted(substrate_ids)}; a subset query must use one frozen run"
        )
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

    ``families_measured`` is the wider set: every family this peak's rows say
    something about, whether or not anything was retained. Only
    ``add_used_hit`` writes the three dictionaries above, so a family the caller
    searched here and found nothing for leaves no trace in them -- and a family
    with no trace is indistinguishable from a family that was never searched,
    which is the absence-versus-zero distinction ``schema.Missingness`` exists
    to keep. ``NO_SEQUENCE_MATCH`` and ``HIT_BELOW_FLOOR`` rows name their
    family, so the measured zero is recorded here and can be reported as one.
    """

    region_id: str
    chrom: str
    start: int
    end: int
    block: tuple[str, int]
    searched: bool = True
    families_measured: set[str] = field(default_factory=set)
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
            # Every row that is a measurement records the family it measured,
            # including the ones that retained nothing. The sentinel is excluded
            # on purpose: an unnamed family is not a measured zero, and admitting
            # it here would put `NA` in a composition table as though it were a
            # family (the failure `HitRecord.__post_init__` refuses for USED rows).
            if (h.missingness is not Missingness.NOT_SEARCHED
                    and h.family_id != MISSING_SENTINEL):
                peak.families_measured.add(h.family_id)
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
    searched = [p for p in present if p.searched]
    # NOT_SEARCHED peaks are not evidence of absence, so a block that contains
    # only NOT_SEARCHED peaks does not contribute to the effective sample size
    # either -- counting it would let an unsearched region inflate n_blocks.
    blocks = {p.block for p in searched}
    with_hit = [p for p in searched if p.has_used_hit]
    explained = len(with_hit) / len(searched) if searched else None

    failures: list[str] = []
    if coverage is None or coverage < floors.min_intersection_coverage:
        shown = "undefined" if coverage is None else round(coverage, 6)
        # Zero overlap is a different diagnosis from thin overlap, and saying
        # only "below the floor" sends the reader to look for missing peaks.
        # Region ids are matched by exact string equality, so not one submitted
        # id being in the universe means the two sides are keyed differently --
        # a 3-column BED read as `chr1:120939636-120940065` against a table that
        # spells the same peak `peak_000001` reports 0.0 here and nothing else.
        mismatch = (
            " -- not one submitted id is in the frozen universe, which is a key mismatch "
            "rather than thin coverage: ids are matched by exact string equality on "
            "region_id, never by interval overlap"
            if submitted_unique and not present else ""
        )
        failures.append(
            f"intersection_coverage={shown} < floor {floors.min_intersection_coverage}{mismatch}"
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


@dataclass
class ContrastHealth:
    """Both sides of a contrast, so an effect is never one-sided evidence.

    A comparator built from its own peak set can fail exactly the floors a
    query can: too few peaks in the universe, too few blocks, too little of it
    explained by the frozen lexicon. Checking only the query's health and then
    differencing against an unexamined comparator can silently produce an
    effect size from a comparator that would itself have been refused as a
    query. ``comparator`` is ``None`` when no comparator was submitted at all
    (as opposed to one that was submitted and failed), so the two absences are
    not conflated.

    ``n_shared_peaks`` counts peaks submitted on **both** sides, and it is here
    because ``shared_blocks`` cannot stand in for it. On a real K562 substrate a
    cluster measured against every peak in the universe and the same cluster
    measured against the universe *minus itself* spanned 283 and 282 shared
    blocks -- one apart -- while the first contrast put 8,277 peaks on both
    sides of the difference and the second none. A reader who went looking for
    the overlap in the block counts would have concluded there was none.
    """

    query: HealthReport
    comparator: HealthReport | None
    shared_blocks: int
    union_blocks: int
    n_shared_peaks: int
    passed: bool
    floor_failures: list[str]


def contrast_health_report(peaks: dict[str, Peak], query_ids: Sequence[str],
                           comparator_ids: Sequence[str], floors: HealthFloors,
                           block_size: int) -> ContrastHealth:
    """Health of the query and, if one was submitted, of the comparator.

    Failures are prefixed ``query:`` or ``comparator:`` so a reader can tell
    which side is responsible without re-deriving it. ``passed`` requires both
    sides to clear their floors; a comparator that was never submitted does
    not count against it, since there is then no contrast to gate.
    """
    query_health = health_report(peaks, query_ids, floors, block_size)
    comparator_health = (
        health_report(peaks, comparator_ids, floors, block_size) if comparator_ids else None
    )

    q_ids = list(dict.fromkeys(query_ids))
    c_ids = list(dict.fromkeys(comparator_ids))
    q_blocks = {peaks[r].block for r in q_ids if r in peaks and peaks[r].searched}
    c_blocks = {peaks[r].block for r in c_ids if r in peaks and peaks[r].searched}

    failures = [f"query: {f}" for f in query_health.floor_failures]
    if comparator_health is not None:
        failures += [f"comparator: {f}" for f in comparator_health.floor_failures]

    return ContrastHealth(
        query=query_health,
        comparator=comparator_health,
        shared_blocks=len(q_blocks & c_blocks),
        union_blocks=len(q_blocks | c_blocks),
        n_shared_peaks=len(set(q_ids) & set(c_ids)),
        passed=query_health.passed and (comparator_health is None or comparator_health.passed),
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
    """Per-family occupancy over the queried peaks. Descriptive: no interval, no test.

    The rows are the families the query peaks were **measured for**, not the ones
    that produced a hit. A family every query peak searched and found nothing for
    is a measured zero and gets a row at ``peak_share`` 0.0; taking the family
    list from ``family_hit_count`` instead dropped that row entirely, and a
    missing row is indistinguishable from a family that was never searched.
    Verified on a real K562 substrate: a 2,176-peak query with no CTCF and no
    GATA hit emitted a ten-row composition beside a twelve-row effects table
    (``estimate_effects`` unions the families over both sides, so it still
    reported them), one record disagreeing with itself about how many families
    exist.
    """
    searched = [peaks[r] for r in dict.fromkeys(region_ids) if r in peaks and peaks[r].searched]
    if not searched:
        return []
    families = sorted({
        fam for p in searched for fam in (*p.families_measured, *p.family_hit_count)
    })
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
    inference_capability: str = CAPABILITY.value


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


def _mean_difference(query: Sequence[float], comparator: Sequence[float]) -> float:
    """The statistic both estimator paths estimate: mean(query) - mean(comparator)."""
    return _mean(query) - _mean(comparator)


#: One family's per-peak coefficient sums, arranged by genomic block: the shared
#: input both estimator paths read. `(query values, comparator values)` per block.
_FamilyFrame = dict[tuple[str, int], tuple[list[float], list[float]]]


def _effect_frame(peaks: dict[str, Peak], query_ids: Sequence[str],
                  comparator_ids: Sequence[str]) -> tuple[
                      list[Peak], list[Peak], list[str],
                      dict[tuple[str, int], tuple[list[Peak], list[Peak]]],
                      list[tuple[str, int]]]:
    """The resampling frame shared by every estimator: peaks, families, blocks.

    Built once and handed to whichever estimator runs, so that changing the
    estimator cannot change *what is being estimated over* -- the block frame is
    the union of blocks either side touches, and it is the same union no matter
    which bootstrap consumes it.

    A peak submitted on both sides is refused here, once, rather than in each
    estimator: it is a property of the frame, not of how uncertainty is computed.
    """
    q_peaks = [peaks[r] for r in dict.fromkeys(query_ids) if r in peaks and peaks[r].searched]
    c_peaks = [peaks[r] for r in dict.fromkeys(comparator_ids) if r in peaks and peaks[r].searched]
    if not q_peaks or not c_peaks:
        raise InterpretError(
            f"effects need searched peaks on both sides: query={len(q_peaks)}, "
            f"comparator={len(c_peaks)}"
        )
    # A peak on both sides is subtracted from itself, and nothing downstream can
    # see it: the point estimate, the interval and every count stay in range, so
    # the result is a plausible number that answers a question nobody asked.
    # Measured on a real K562 substrate: one island against *all* peaks reported
    # all twelve families at exactly 0.7560 of their value against all peaks
    # minus that island -- the disjoint fraction of the comparator, 25,640/33,917
    # -- with the interval shifted to match. `shared_blocks` did not show it
    # either (283 overlapping vs 282 disjoint). At complete overlap, comparator
    # == query, every effect is exactly 0.0 with a zero-width interval, which
    # reads as "measured, and there is no difference".
    shared = {p.region_id for p in q_peaks} & {p.region_id for p in c_peaks}
    if shared:
        example = ", ".join(sorted(shared)[:3])
        raise InterpretError(
            f"query and comparator share {len(shared)} peak(s) ({example}"
            f"{', ...' if len(shared) > 3 else ''}). A peak on both sides of the difference "
            "is subtracted from itself, which attenuates every family's effect by exactly "
            "the comparator's disjoint fraction and shifts its interval to match, silently. "
            "Submit the comparator with the query peaks removed; 'query vs everything' is "
            "spelled 'query vs everything except the query'."
        )
    families = sorted({fam for p in (*q_peaks, *c_peaks) for fam in p.family_coefficient_sum})

    by_block: dict[tuple[str, int], tuple[list[Peak], list[Peak]]] = {}
    for p in q_peaks:
        by_block.setdefault(p.block, ([], []))[0].append(p)
    for p in c_peaks:
        by_block.setdefault(p.block, ([], []))[1].append(p)
    return q_peaks, c_peaks, families, by_block, sorted(by_block)


def estimate_effects(peaks: dict[str, Peak], query_ids: Sequence[str],
                     comparator_ids: Sequence[str], comparator_id: str,
                     n_bootstrap: int, seed: int, block_size: int,
                     estimator: str = ESTIMATOR) -> list[FamilyEffect]:
    """Per-family difference in mean per-peak coefficient, query minus comparator.

    Whole genomic blocks are the resampling unit, not peaks: peaks within a block
    are not independent, and a peak-level bootstrap would report an interval far
    narrower than the data support (``BA-07``, ``FP-15``). That holds on both
    estimator paths -- what ``estimator`` selects is how the uncertainty around
    the same point estimate is computed, never what the point estimate is:

    * ``ESTIMATOR_PERCENTILE`` -- percentile block bootstrap, interval only. No
      p or q value: the proportion of replicates crossing zero looks like a
      two-sided p value and is not a calibrated one.
    * ``ESTIMATOR_BCA_WILD`` -- ``FP-15``'s specified pair, licensed
      ``INTERVAL_AND_TEST``: a BCa paired genomic-block interval and a
      block-level wild cluster bootstrap-*t* p value, with q values by
      Benjamini-Hochberg over the families in *this call* and no others.

    An unknown ``estimator`` is refused rather than mapped onto the default.
    """
    estimator = _resolve_estimator(estimator)
    q_peaks, c_peaks, families, by_block, blocks = _effect_frame(peaks, query_ids, comparator_ids)
    if not families:
        return []
    if estimator == ESTIMATOR_BCA_WILD:
        return _effects_bca_wild(
            q_peaks, c_peaks, families, by_block, blocks, comparator_id,
            n_bootstrap=n_bootstrap, seed=seed, block_size=block_size)
    return _effects_percentile(
        q_peaks, c_peaks, families, by_block, blocks, comparator_id,
        n_bootstrap=n_bootstrap, seed=seed, block_size=block_size)


def _effects_percentile(q_peaks: list[Peak], c_peaks: list[Peak], families: list[str],
                        by_block: dict[tuple[str, int], tuple[list[Peak], list[Peak]]],
                        blocks: list[tuple[str, int]], comparator_id: str,
                        *, n_bootstrap: int, seed: int,
                        block_size: int) -> list[FamilyEffect]:
    """Percentile block bootstrap: a point estimate and an interval, and nothing else."""
    if n_bootstrap < MIN_PERCENTILE_REPLICATES:
        raise InterpretError(
            f"n_bootstrap={n_bootstrap} is below the preregistered floor of "
            f"{MIN_PERCENTILE_REPLICATES} for a 95% percentile interval: {n_bootstrap} "
            f"replicates resolve a tail no finer than 1/{n_bootstrap + 1}, so both "
            "endpoints are extreme replicates and the interval is not the 95% one it "
            "would be labelled. Refusing to report an interval narrower than the "
            "replicates support."
        )
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

    # No p or q value is computed on THIS path. The proportion of bootstrap
    # replicates crossing zero looks like a two-sided p value but is not a
    # calibrated one (this estimator is ESTIMATION_ONLY; see
    # schema.ESTIMATOR_CAPABILITY) -- a number that looks like a p value but is
    # not one is worse than no number. `_bh()` is correspondingly never called
    # here: a q value derived from an invalid p value is also invalid. Both are
    # emitted by `_effects_bca_wild`, whose p value comes from FP-15's specified
    # test rather than from this replicate tail.
    effects: list[FamilyEffect] = []
    for k, fam in enumerate(families):
        point = (_mean([p.family_coefficient_sum.get(fam, 0.0) for p in q_peaks])
                 - _mean([p.family_coefficient_sum.get(fam, 0.0) for p in c_peaks]))
        reps = sorted(replicates[k])
        n_valid = len(reps)
        if n_valid:
            lo = reps[max(0, int(math.floor(0.025 * n_valid)) - 1)]
            hi = reps[min(n_valid - 1, int(math.ceil(0.975 * n_valid)) - 1)]
            ci: tuple[float, float] | None = (lo, hi)
        else:
            ci = None
        effects.append(FamilyEffect(
            id=f"{fam}_vs_{comparator_id}",
            family_id=fam,
            comparator_id=comparator_id,
            is_cross_condition=True,
            effect=point,
            ci=ci,
            p_value=None,
            q_value=None,
            n_query_peaks=len(q_peaks),
            n_comparator_peaks=len(c_peaks),
            n_blocks=len(blocks),
            n_bootstrap=n_bootstrap,
            n_bootstrap_valid=n_valid,
            block_size=block_size,
            random_seed=seed,
            estimator=ESTIMATOR_PERCENTILE,
            inference_capability=ESTIMATOR_CAPABILITY[
                Estimator(ESTIMATOR_PERCENTILE)].value,
        ))
    return effects


def _effects_bca_wild(q_peaks: list[Peak], c_peaks: list[Peak], families: list[str],
                      by_block: dict[tuple[str, int], tuple[list[Peak], list[Peak]]],
                      blocks: list[tuple[str, int]], comparator_id: str,
                      *, n_bootstrap: int, seed: int,
                      block_size: int) -> list[FamilyEffect]:
    """``FP-15``'s specified pair: BCa block intervals and wild cluster bootstrap-*t* p.

    The two halves read the *same* per-family, per-block data, in two shapes the
    two estimators need:

    * the interval half (:func:`infer.bca_paired_block_interval`) resamples whole
      blocks and needs each block's per-peak values on each side, so it receives
      them unreduced;
    * the test half (:func:`infer.wild_cluster_bootstrap_t`) needs ONE scalar per
      block, so each block is reduced to its contribution to the peak-level mean
      difference:

      ``e_g = (G / N_q) * sum_q(g) - (G / N_c) * sum_c(g)``

      with ``G`` blocks and ``N_q`` / ``N_c`` searched peaks per side. That
      scaling is not cosmetic: it makes ``mean(e_g)`` **equal to the reported
      point estimate**, so the hypothesis the p value tests
      (``mean(e_g) == 0``) is a hypothesis about the number beside it, rather
      than about a differently-weighted quantity that merely resembles it. A
      block a side never touches contributes 0 from that side and stays in the
      frame -- dropping it would change ``G`` and silently reweight every other
      block.

    The block is the resampling unit in both halves by construction: whole
    blocks in the BCa draw, one Rademacher weight per block in the wild
    bootstrap. Every family in the call is given the same ``seed``, so all
    families see the same block resampling -- the families are contrasts over
    one substrate, not independent experiments.

    Refusals, both of which are "computed from too little data" rather than
    best-effort numbers:

    * ``infer.InferError`` propagates when the block frame or the estimable
      replicate count is below ``infer.MIN_ESTIMABLE_BLOCKS``. It is deliberately
      not caught and re-labelled: the health floors are caller-adjustable
      (``--floor-blocks``) and this one is not, so a caller that lowered a floor
      must still be told the estimator refused on its own terms.
    * ``InterpretError`` when a family's wild bootstrap leaves too few estimable
      replicates -- constant per-block effects give a degenerate reference
      distribution. The whole interpretation is refused rather than that one
      family being annotated: a p value computed from nothing must not travel
      beside the valid ones, where a reader would compare them.
    """
    n_q, n_c = len(q_peaks), len(c_peaks)
    g = len(blocks)

    rows: list[tuple[str, float, tuple[float, float], float, int]] = []
    for fam in families:
        query_values: dict[tuple[str, int], list[float]] = {}
        comparator_values: dict[tuple[str, int], list[float]] = {}
        block_effects: dict[tuple[str, int], float] = {}
        for b in blocks:
            qs, cs = by_block[b]
            q_vals = [p.family_coefficient_sum.get(fam, 0.0) for p in qs]
            c_vals = [p.family_coefficient_sum.get(fam, 0.0) for p in cs]
            if q_vals:
                query_values[b] = q_vals
            if c_vals:
                comparator_values[b] = c_vals
            block_effects[b] = (g / n_q) * sum(q_vals) - (g / n_c) * sum(c_vals)

        ci = infer.bca_paired_block_interval(
            query_values, comparator_values, statistic=_mean_difference,
            n_bootstrap=n_bootstrap, seed=seed)
        p_value, n_valid = infer.wild_cluster_bootstrap_t(
            block_effects, n_bootstrap=n_bootstrap, seed=seed)
        if n_valid < infer.MIN_ESTIMABLE_BLOCKS:
            raise InterpretError(
                f"{fam}: the wild cluster bootstrap-t reference distribution is degenerate "
                f"({n_valid} of {n_bootstrap} replicates estimable, below the preregistered "
                f"floor of {infer.MIN_ESTIMABLE_BLOCKS}). The per-block effects carry no "
                "variance to resample, so no p value is reported for any family in this "
                "interpretation."
            )
        point = (_mean([p.family_coefficient_sum.get(fam, 0.0) for p in q_peaks])
                 - _mean([p.family_coefficient_sum.get(fam, 0.0) for p in c_peaks]))
        rows.append((fam, point, ci, p_value, n_valid))

    # BH over the families tested in THIS call and no others. Reached only here:
    # a q value is a statement about a family of hypotheses, and there is no
    # family of hypotheses on the estimation-only path.
    q_values = _bh([p for _, _, _, p, _ in rows])
    capability = ESTIMATOR_CAPABILITY[Estimator(ESTIMATOR_BCA_WILD)].value
    return [
        FamilyEffect(
            id=f"{fam}_vs_{comparator_id}",
            family_id=fam,
            comparator_id=comparator_id,
            is_cross_condition=True,
            effect=point,
            ci=ci,
            p_value=p_value,
            q_value=q_value,
            n_query_peaks=n_q,
            n_comparator_peaks=n_c,
            n_blocks=g,
            n_bootstrap=n_bootstrap,
            #: The wild bootstrap-t's estimable replicate count -- the one that
            #: licenses the p value. The BCa half refuses outright below the same
            #: floor rather than reporting a reduced count, so one number here is
            #: not hiding a second, smaller one.
            n_bootstrap_valid=n_valid,
            block_size=block_size,
            random_seed=seed,
            estimator=ESTIMATOR_BCA_WILD,
            inference_capability=capability,
        )
        for (fam, point, ci, p_value, n_valid), q_value in zip(rows, q_values, strict=True)
    ]


# --------------------------------------------------------------------------- #
# Step 3c: two-part usage summaries (occupancy and intensity, never one number)
# --------------------------------------------------------------------------- #
def _peak_usage(peak: Peak, family_id: str) -> infer.PeakUsage:
    """One peak, in the shape :func:`infer.two_part_summary` reads.

    Occupancy comes from ``family_hit_count`` and signed mass from
    ``family_coefficient_sum`` -- the two Task 1 separated, kept separate all the
    way through. ``searched`` carries the only missingness distinction a
    denominator needs; `infer` applies the domain rule, so it lives in exactly
    one place rather than being re-derived here.
    """
    return infer.PeakUsage(
        searched=peak.searched,
        hit_count=peak.family_hit_count.get(family_id, 0),
        coefficient_sum=peak.family_coefficient_sum.get(family_id, 0.0),
        abs_coefficient_sum=peak.family_abs_coefficient_sum.get(family_id, 0.0),
        peak_abs_coefficient_sum=sum(peak.family_abs_coefficient_sum.values()),
    )


def two_part_effects(peaks: dict[str, Peak], query_ids: Sequence[str],
                     comparator_ids: Sequence[str], *,
                     usage_definition: infer.UsageDefinition,
                     usage_threshold: infer.UsageThreshold | None = None,
                     ) -> list[infer.TwoPartEffect]:
    """Per-family two-part summary of the same contrast :func:`estimate_effects` estimates.

    A family used in more query peaks but less intensely where it is used has a
    total effect near zero; reported as one number that reads as "no
    difference", reported as two it reads as what it is. This produces the two.

    Descriptive by construction: no interval and no p value, because the split
    is about *what was measured*, not about how uncertain it is.
    ``usage_definition`` has no default here either -- it is passed straight
    through to `infer`, which refuses anything but an explicit member.
    """
    q_peaks = [peaks[r] for r in dict.fromkeys(query_ids) if r in peaks]
    c_peaks = [peaks[r] for r in dict.fromkeys(comparator_ids) if r in peaks]
    families = sorted({
        fam
        for p in (*q_peaks, *c_peaks)
        if p.searched
        for fam in p.family_hit_count
    })
    return [
        infer.two_part_summary(
            [_peak_usage(p, fam) for p in q_peaks],
            [_peak_usage(p, fam) for p in c_peaks],
            family_id=fam,
            usage_definition=usage_definition,
            usage_threshold=usage_threshold,
        )
        for fam in families
    ]


def _serialize_two_part(effect: infer.TwoPartEffect) -> dict[str, Any]:
    """`asdict` plus the enum flattened, so the JSON carries "ANY_HIT" rather
    than depending on `StrEnum` happening to serialise as its value."""
    payload = asdict(effect)
    payload["usage_definition"] = effect.usage_definition.value
    return payload


# --------------------------------------------------------------------------- #
# The whole query
# --------------------------------------------------------------------------- #
@dataclass
class Interpretation:
    query_id: str
    selection_provenance: str
    #: T-07 (Task 7): may this query's numbers support inference at all, read
    #: independently of what they may be a claim about. See `schema.identity`.
    statistical_license: str
    #: T-07 (Task 7): what the resulting number can be a claim about, read
    #: independently of `statistical_license` -- a held-out attribution cluster
    #: is `HELD_OUT_INFERENCE` *and* `SUBSTRATE_CIRCULAR` in the same record.
    claim_scope: str
    #: Deprecated compatibility view of the two fields above, retained for one
    #: release. Derived, not a second source of truth: see
    #: `schema._output_mode_from_permissions`. `SUBSTRATE_CIRCULAR` has no
    #: representation here, which is exactly why it is not the field to read.
    output_mode: str
    emitted_order: list[str]
    query_health: dict[str, Any]
    comparator_health: dict[str, Any] | None
    contrast_health: dict[str, Any] | None
    floor_failures: list[str]
    interpretation_emitted: bool
    suppression_reason: str | None
    composition: list[dict[str, Any]] | None
    effects: list[dict[str, Any]] | None
    #: Occupancy and conditional intensity, reported separately (`FP-15`, Task
    #: 17). `None` means no `usage_definition` was configured, NOT that a
    #: default one was applied and found nothing: "used" is a scientific
    #: definition and this record never invents one. When present, every entry
    #: names the definition it was computed under.
    two_part_effects: list[dict[str, Any]] | None
    notes: list[str]
    input_scale: int
    lexicon_id: str
    substrate_id: str
    estimator: str = ESTIMATOR
    #: Every recognised value, and the subset that exists in this release. A
    #: consumer branches on `estimator` against this list rather than against a
    #: literal, so adding FP-15's estimators does not change the caller.
    estimators_defined: list[str] = field(
        default_factory=lambda: [e.value for e in Estimator])
    estimators_implemented: list[str] = field(
        default_factory=lambda: sorted(e.value for e in IMPLEMENTED_ESTIMATORS))
    #: Deprecated alias of `query_health`, retained for one release so an
    #: existing reader keyed on `health` (attribute or `to_dict()` key) does
    #: not break. See docs/DATA_MODEL.md. Slated for removal after this release.
    health: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, out_dir: str | os.PathLike[str]) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "interpretation.json"
        dest.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return dest


def interpret_query(hits: Sequence[HitRecord], query: PeakSetQuery,
                    floors: HealthFloors | None = None,
                    block_size: int = DEFAULT_BLOCK_SIZE,
                    n_bootstrap: int = DEFAULT_BOOTSTRAP,
                    seed: int = 0,
                    estimator: str = ESTIMATOR,
                    usage_definition: infer.UsageDefinition | None = None,
                    usage_threshold: infer.UsageThreshold | None = None,
                    ) -> Interpretation:
    """Answer one peak-set query at the strength its selection provenance licenses.

    ``estimator`` selects how uncertainty is computed (``ESTIMATOR_CHOICES``) and
    therefore what the result is licensed to carry: the default percentile block
    bootstrap is ``ESTIMATION_ONLY`` and withholds p and q values, while
    ``ESTIMATOR_BCA_WILD`` is ``INTERVAL_AND_TEST`` and emits them. It is
    resolved before anything is computed, so an unrecognised name costs nothing
    and never silently runs a different estimator.

    ``usage_definition`` adds the two-part usage summary (:func:`two_part_effects`)
    beside the effects. It has **no default**: omitting it leaves
    ``two_part_effects`` at ``None``, which records that no definition of "used"
    was chosen -- not that one was chosen for the caller and produced nothing.
    """
    floors = floors or HealthFloors()
    # Resolved first: a refusal that depends on no data should not wait behind a
    # bootstrap, and every downstream capability decision reads this one value.
    estimator = _resolve_estimator(estimator)
    capability = ESTIMATOR_CAPABILITY[Estimator(estimator)]
    substrate_ids = {h.substrate_id for h in hits}
    if MISSING_SENTINEL in substrate_ids:
        raise InterpretError("interpretation has records without a substrate_id")
    if len(substrate_ids) != 1:
        raise InterpretError("interpretation mixes substrates; subset queries require one frozen run")
    substrate_id = next(iter(substrate_ids))
    if (not isinstance(substrate_id, str)
            or len(substrate_id) != 64
            or any(c not in "0123456789abcdef" for c in substrate_id)):
        raise InterpretError("interpretation has an invalid substrate_id")
    # (1) before any number is computed. Two independent reads: `statistical_license`
    # decides what the query is allowed to compute, `claim_scope` decides what the
    # result may be evidence about. `mode` is kept only as the deprecated
    # compatibility view emitted alongside them -- see PeakSetQuery.output_mode.
    statistical_license = query.statistical_license
    claim_scope = query.claim_scope
    mode = query.output_mode
    notes: list[str] = []
    if query.selection_provenance is SelectionProvenance.DECLARATION_MISSING:
        notes.append(
            "selection_provenance was not declared; recorded as DECLARATION_MISSING and run in "
            "the most conservative mode. That is not the same as EXTERNAL."
        )
    if claim_scope is ClaimScope.CONDITIONING_UNVERIFIABLE:
        notes.append(
            "the conditioning set of this selection cannot be verified: it cannot be shown that "
            "downstream information was not already visible when the peak set was chosen"
        )
    if claim_scope is ClaimScope.SUBSTRATE_CIRCULAR:
        # The one outcome the deprecated `output_mode` cannot express, so it is
        # also the one a reader following that field will never be told about:
        # a fully licensed FULL_INFERENCE run selected on `hit_coefficient`
        # printed exactly what an EXTERNAL_STRUCTURE run printed. The note is
        # not a caveat beside a suppressed number -- the number is licensed, and
        # what the note names is what it may be a claim ABOUT.
        selected_on = ", ".join(query.selection_feature_names) or "an attribution-derived feature"
        notes.append(
            f"claim_scope is SUBSTRATE_CIRCULAR: this peak set was selected on {selected_on}, "
            "which is derived from the same attribution surface these numbers describe. The "
            "statistical license is unaffected -- what is limited is what the result can be "
            "evidence about: the model's own attribution surface, not structure external to "
            "it. output_mode cannot represent this, so read claim_scope."
        )

    peaks = peak_universe(hits, block_size)
    region_ids = list(query.region_ids)
    comparator_ids = list(query.comparator_region_ids)
    if statistical_license is StatisticalLicense.HELD_OUT_INFERENCE:
        held_out = set(query.held_out_region_ids)
        if not held_out:
            raise InterpretError(
                "CLUSTERED_WITH_SPLIT claims a held-out half but none was supplied. The grade's "
                "entire licence is the split; without it the query is CLUSTERED_NO_SPLIT."
            )
        # The split is applied to BOTH sides. Restricting only the query would
        # contrast held-out peaks against a comparator the clustering had already
        # seen, which is not the contrast the split licenses. The comparator's
        # count is reported for the same reason: the filter used to be announced
        # as a query-side restriction only, so a comparator it had emptied left
        # no trace, and the run then failed with "a cross-condition effect needs
        # a named baseline peak set" at a caller who had named one.
        n_comparator_submitted = len(comparator_ids)
        region_ids = [r for r in region_ids if r in held_out]
        comparator_ids = [r for r in comparator_ids if r in held_out]
        notes.append(
            f"inference restricted to the held-out half ({len(region_ids)} query peaks, "
            f"{len(comparator_ids)} comparator peaks)"
        )
        if n_comparator_submitted and not comparator_ids:
            raise InterpretError(
                f"the held-out set retains none of the {n_comparator_submitted} comparator "
                "peaks submitted. CLUSTERED_WITH_SPLIT restricts both sides of the contrast "
                "to the held-out half, so a --held-out list naming only query peaks leaves "
                "no baseline to difference against. Name the held-out half of the comparator "
                "in it as well, or declare the selection at a grade that does not split."
            )

    if not region_ids:
        raise InterpretError(
            f"{query.query_id}: the query is empty, so no ratio it produces has a denominator"
        )

    # (2) health, always -- both sides, before composition or effects, so a
    # comparator built from an unhealthy peak set cannot silently license an
    # effect the query side alone would have earned.
    contrast = contrast_health_report(peaks, region_ids, comparator_ids, floors, block_size)
    emitted = ["health"]

    composition: list[dict[str, Any]] | None = None
    effects: list[dict[str, Any]] | None = None
    two_part: list[dict[str, Any]] | None = None
    suppression: str | None = None

    # The operative failures: what actually caused *this* interpretation to be
    # suppressed. A comparator that was never relevant to this mode (descriptive
    # modes never touch it) must not appear here just because it happens to be
    # unhealthy -- that would report a failure for nothing that was suppressed.
    # `contrast.floor_failures` (unconditional, both sides) still travels in
    # full inside `contrast_health` below, for transparency.
    floor_failures: list[str] = [f"query: {f}" for f in contrast.query.floor_failures]

    if not contrast.query.passed:
        # (3) Suppression, not annotation. A caveat beside a number does not travel.
        suppression = (
            "reading suppressed: " + "; ".join(floor_failures)
            + ". The health numbers above stand; no composition and no effect is reported."
        )
    else:
        floor_failures = []
        composition = [asdict(c) for c in compose(peaks, region_ids)]
        emitted.append("composition")
        if statistical_license in (StatisticalLicense.FULL_INFERENCE,
                                   StatisticalLicense.HELD_OUT_INFERENCE):
            if query.comparator_id == MISSING_SENTINEL or not comparator_ids:
                raise InterpretError(
                    "a cross-condition effect needs a named baseline peak set. One set of "
                    "measurements once supported both 'replicates exactly' and 'four times "
                    "stronger, prediction falsified', differing only in the comparator; no "
                    "baseline, no number (BA-18)."
                )
            # `contrast.passed` is the single computed source of truth for "both
            # sides healthy" (`query.passed and (comparator is None or
            # comparator.passed)`). We are inside the `else` of `if not
            # contrast.query.passed`, so `contrast.query.passed` is already True
            # here, which makes `not contrast.passed` exactly equivalent to "a
            # comparator was submitted and it failed" -- read from the one field
            # instead of re-deriving that AND inline a second time.
            if not contrast.passed:
                # The query side earned composition, but an effect needs both sides
                # healthy: a comparator that would itself have been refused as a
                # query must not license a difference against it either.
                comparator_failures = [f"comparator: {f}" for f in contrast.comparator.floor_failures]
                floor_failures = comparator_failures
                suppression = (
                    "effects suppressed: " + "; ".join(comparator_failures)
                    + ". Composition above stands; no effect is reported because the "
                    "comparator side does not clear its floors."
                )
            else:
                effects = [asdict(e) for e in estimate_effects(
                    peaks, region_ids, comparator_ids, query.comparator_id,
                    n_bootstrap=n_bootstrap, seed=seed, block_size=block_size,
                    estimator=estimator)]
                emitted.append("effects")
                if usage_definition is not None:
                    # Same gate as the effects it sits beside, and for the same
                    # reason: this is a query-minus-comparator contrast, so a
                    # comparator that would have been refused as a query must
                    # not license it either. Descriptive within that gate --
                    # occupancy and intensity, no interval and no p value.
                    two_part = [
                        _serialize_two_part(e) for e in two_part_effects(
                            peaks, region_ids, comparator_ids,
                            usage_definition=usage_definition,
                            usage_threshold=usage_threshold)
                    ]
                    emitted.append("two_part_effects")
                    notes.append(
                        f"usage is defined as {usage_definition.value}; occupancy "
                        "(probability of use) and conditional intensity are reported "
                        "separately, because a family used more often and less intensely "
                        "than its comparator has a total effect near zero that describes "
                        "neither margin."
                    )
                guards.comparator_declared(effects).raise_if_failed()
                # Once per interpretation, not once per family: which estimator
                # ran, and what it therefore may emit, is a property of the run.
                # `effects` can legally be `[]` (query and comparator share no
                # family at all), and both notes describe effects that exist --
                # with none, they have nothing to describe and must not fire.
                if effects and capability is InferenceCapability.ESTIMATION_ONLY:
                    notes.append(
                        "The implemented percentile block bootstrap supports estimation only. "
                        "Hypothesis-test p and q values are withheld until the preregistered "
                        "wild cluster bootstrap-t estimator is used."
                    )
                elif effects:
                    # Names BOTH halves: the result's `estimator` field records
                    # the test, because that is the half whose absence a reader
                    # must not assume away, and the interval half would otherwise
                    # be invisible in the record.
                    notes.append(
                        "Intervals are BCa paired genomic-block bootstrap intervals and p "
                        "values are block-level wild cluster bootstrap-t values (FP-15's "
                        "specified pair, licensed INTERVAL_AND_TEST). q values are "
                        "Benjamini-Hochberg over the families in this interpretation and no "
                        "others."
                    )
        else:
            notes.append(
                f"{mode.value}: descriptive decomposition only. No interval and no p value is "
                "reported, because the selection criterion cannot be separated from the signal."
            )

    query_health = asdict(contrast.query)
    comparator_health = asdict(contrast.comparator) if contrast.comparator is not None else None
    contrast_health = asdict(contrast) if contrast.comparator is not None else None

    result = Interpretation(
        query_id=query.query_id,
        selection_provenance=query.selection_provenance.value,
        statistical_license=statistical_license.value,
        claim_scope=claim_scope.value,
        output_mode=mode.value,
        emitted_order=emitted,
        query_health=query_health,
        comparator_health=comparator_health,
        contrast_health=contrast_health,
        floor_failures=floor_failures,
        interpretation_emitted=composition is not None,
        suppression_reason=suppression,
        composition=composition,
        effects=effects,
        two_part_effects=two_part,
        notes=notes,
        input_scale=hits[0].input_scale,
        lexicon_id=hits[0].lexicon_id,
        substrate_id=substrate_id,
        estimator=estimator,
        health=query_health,   # deprecated alias of query_health; see docs/DATA_MODEL.md
    )

    # The guards run on the finished record, not on intermediate state, so that
    # what is checked is exactly what a reader would receive. health_before_effect
    # reads the query view only: it gates composition, which is a query-only
    # reading, not the bilateral contrast that gates effects.
    guards.selection_provenance_declared([{
        "query_id": query.query_id,
        "selection_provenance": query.selection_provenance.value,
        "output_mode": mode.value,
    }]).raise_if_failed()
    guards.health_before_effect({
        "health": result.query_health,
        "emitted_order": result.emitted_order,
        "floor_failures": contrast.query.floor_failures,
        "effects": result.effects or [],
        "interpretation_emitted": result.interpretation_emitted,
    }).raise_if_failed()
    guards.single_scale([{"input_scale": h.input_scale} for h in hits]).raise_if_failed()
    return result
