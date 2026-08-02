"""Run the declared grid of specifications, and refuse to average across questions.

The package is named for this and did not do it. Every release until now ran one
specification per inference run -- one query, one baseline, one lexicon, one
estimator -- and a reader who wanted to know whether a conclusion survived a
different defensible choice had to run the tool again and diff two directories by
hand. Nothing recorded that the second run existed, so nothing recorded when it
disagreed.

**The point of a multiverse is not width.** It is that the choices are declared
before the results are seen, that every declared cell appears in the output
whether or not it worked, and that the axes are kept apart so the summary does not
average an answer to one question with an answer to another. A grid without those
three is a way of finding a specification that agrees with you.

So the three kinds of choice are three *types*, not three keys in one options dict:

:class:`Estimand`
    The question. Query, **baseline population**, its type and its construction
    rule, and the selection rule that produced the query. Changing it changes what
    is being estimated.
:class:`Measurement`
    How that question is measured: which lexicon, which frozen hit table. Changing
    it changes the measurement of one question, not the question.
:class:`StatisticalChoice`
    How uncertainty is computed: estimator, block size, replicates, seed, floors.
    Changing it changes neither the question nor the measurement.

The type separation is load-bearing rather than tidy. Represented as one flat bag
of knobs, "average over the knobs" is the natural thing to write, and it produces
the generic robustness score that hides the only finding worth reporting: *which*
choice the conclusion was sensitive to. Here the estimand is the key of every
summary, and :func:`~motifmultiverse.guards.no_cross_estimand_pooling` refuses a
summary that spans two of them -- reading each cell's estimand from the manifest
written before the run, not from the summariser that grouped them.

This module computes no statistics. It enumerates, binds, calls
:func:`interpret.interpret_query`, and writes down what came back, including from
the cells that failed. A second implementation of a block bootstrap that agrees
with the first 99% of the time is worse than none, because the 1% arrives as a
number nobody can trace.

See ``docs/MULTIVERSE_DESIGN.md`` for the identity question this design had to
answer first -- what "a single frozen dataset" can mean when varying the lexicon
necessarily varies the frozen hit-calling run.
"""
from __future__ import annotations

import hashlib
import json
import os
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from motifmultiverse import guards, interpret
from motifmultiverse.guard_log import GuardLog
from motifmultiverse.schema import (
    MISSING_SENTINEL,
    PeakSetQuery,
    SchemaError,
    SelectionProvenance,
)
from motifmultiverse.substrate import read_opportunity_ledger

__all__ = [
    "MULTIVERSE_SCHEMA_VERSION", "MultiverseError", "CellStatus",
    "Estimand", "Measurement", "StatisticalChoice", "Specification",
    "MultiverseDesign", "CellResult", "MultiverseResult",
    "read_design", "plan", "run_multiverse", "stability_within_estimand",
    "FamilyCellStatus", "family_cell_states", "content_digest",
    "ESTIMAND_INPUT_ROLES", "MEASUREMENT_INPUT_ROLES", "NOT_DECLARED",
    "NOT_ESTIMABLE_MARKER", "NO_PREREGISTERED_THRESHOLD",
]

#: This artifact family says which version it is, like every other one here.
MULTIVERSE_SCHEMA_VERSION = "1"

#: What a per-family cell entry holds when the cell produced no estimate. A
#: literal token and never ``0.0``: a family that was not estimable and a family
#: whose effect was zero are the two things this package exists to keep apart, and
#: collapsing them is its founding failure in a new place.
NOT_ESTIMABLE_MARKER = "NOT_ESTIMABLE"

#: What the threshold field says when no threshold was preregistered -- which is
#: the normal case, and is not the same as a threshold of zero or of "any".
NO_PREREGISTERED_THRESHOLD = "NONE_PREREGISTERED"


class MultiverseError(SchemaError):
    """A design cannot be read, or a grid cannot be run as declared."""


class FamilyCellStatus:
    """What became of one (cell, family) pair. Six states, and none of them zero.

    The audit could already say that five families existed only under one lexicon,
    but only by *inference*: those cells had no row in `family_effects.tsv`, and a
    reader had to work out that an absent row meant an absent family rather than
    an absent effect. That is the founding failure's shape one level up -- an
    absence standing in for a measurement -- so the states are written down
    per pair instead of left to be deduced.

    ``MEASURED_ZERO`` is separate from ``ESTIMATED`` on purpose. An effect of
    exactly 0.0 is a real measurement and the one value most likely to be confused
    with a fill, so it is labelled rather than left to look like any other
    estimate.

    ``NOT_SEARCHED`` is evaluated against *this cell's* peaks -- the query and the
    baseline together -- not against the whole substrate: a family searched
    elsewhere in the frozen run and nowhere in this contrast was not searched
    here, whatever the rest of the table says.
    """

    ESTIMATED = "ESTIMATED"
    MEASURED_ZERO = "MEASURED_ZERO"
    NOT_IN_LEXICON = "NOT_IN_LEXICON"
    NOT_SEARCHED = "NOT_SEARCHED"
    NOT_ESTIMABLE = "NOT_ESTIMABLE"
    CELL_REFUSED = "CELL_REFUSED"

    ALL = (ESTIMATED, MEASURED_ZERO, NOT_IN_LEXICON, NOT_SEARCHED, NOT_ESTIMABLE,
           CELL_REFUSED)


class CellStatus:
    """What became of one planned cell. Every planned cell gets exactly one.

    ``ERROR`` exists so that an unexpected failure is a recorded cell rather than
    a traceback that ends the run: a grid that stops at its first surprise reports
    a subset of itself, and the subset is not random.
    """

    SUCCESS = "SUCCESS"
    REFUSED_GUARD = "REFUSED_GUARD"
    REFUSED_SCHEMA = "REFUSED_SCHEMA"
    NOT_ESTIMABLE = "NOT_ESTIMABLE"
    ERROR = "ERROR"

    ALL = (SUCCESS, REFUSED_GUARD, REFUSED_SCHEMA, NOT_ESTIMABLE, ERROR)


#: What a cell's identity binds, beyond the strings a design declares. Each is a
#: file whose *contents* decide what the cell measured, and hashing the path
#: instead would let the contents be swapped underneath a stable id -- the failure
#: mode this whole package is organised against, in the identity system itself. A
#: design that renames `peaksets/query.txt` and rewrites it produces a different
#: cell; a design that moves the same bytes to a new path produces the same one.
ESTIMAND_INPUT_ROLES = ("query_regions", "baseline_regions")
MEASUREMENT_INPUT_ROLES = ("hit_table", "lexicon_manifest", "opportunity_ledger")

#: Recorded for an input a design declares but no file backs -- the optional ones,
#: which are absent rather than empty. A literal token rather than "" so that a
#: declared-but-missing file cannot hash the same as one nobody declared.
NOT_DECLARED = "NOT_DECLARED"


def content_digest(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file's bytes, streamed.

    Streamed because a hit table is routinely hundreds of megabytes and an
    identity function that needs the whole file in memory is one people work
    around.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    """A deterministic id over the canonical JSON of ``payload``.

    Sorted keys and no whitespace, so the same declared grid produces the same ids
    on any machine and in any dict order -- which is what makes a cell id usable
    as the join key between the manifest, the effects table and a later re-run.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]}"


# --------------------------------------------------------------------------- #
# The three axes.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Estimand:
    """One scientific question: a query, and the population it is asked against.

    ``baseline_population_type`` and ``baseline_construction_rule`` are both
    required and neither has a default. The reference failure this comes from had
    the same data supporting both "replicates exactly" and "4x stronger,
    prediction falsified", differing only in which population the comparison was
    against -- and the artifacts recorded neither, so the disagreement could not
    be diagnosed from them. A baseline whose construction is not written down is
    not a baseline anyone can reproduce, so a design that omits the rule is
    refused rather than defaulted.
    """

    query_id: str
    query_regions: str
    baseline_id: str
    baseline_population_type: str
    baseline_construction_rule: str
    baseline_regions: str
    selection_provenance: str
    selection_rule: str
    selection_feature_names: tuple[str, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("query_id", "query_regions", "baseline_id", "baseline_regions",
                     "baseline_population_type", "baseline_construction_rule",
                     "selection_provenance", "selection_rule"):
            if not str(getattr(self, name) or "").strip():
                raise MultiverseError(
                    f"estimand {self.query_id or '<unnamed>'}: {name} is required. "
                    "An estimand is the question being asked; a question missing any of "
                    "these is not one a reader can reconstruct."
                )
        try:
            SelectionProvenance(self.selection_provenance)
        except ValueError as exc:
            raise MultiverseError(
                f"estimand {self.query_id}: selection_provenance "
                f"{self.selection_provenance!r} is not one of "
                f"{[p.value for p in SelectionProvenance]}"
            ) from exc

    def identity(self, digests: Mapping[str, str] | None = None) -> str:
        """The estimand's id, over its declaration AND its peak sets' contents.

        ``digests`` is required in practice and defaulted only so that a caller
        reasoning about a declaration alone can still get a stable string. A
        declaration-only id is not an identity: two runs whose `query.txt` differs
        are two different questions, and an id that cannot tell them apart makes
        the manifest's whole promise -- that a cell id names the science -- false.
        """
        return _stable_id("est", {"declared": asdict(self),
                                  "inputs": dict(sorted((digests or {}).items()))})

    @property
    def estimand_id(self) -> str:
        """Declaration-only id. Prefer :meth:`identity`; see its docstring."""
        return self.identity()


@dataclass(frozen=True)
class Measurement:
    """One way of measuring: a lexicon, and the frozen hit table called with it.

    ``substrate_id`` and ``lexicon_content_hash`` are declared here and checked
    against the artifacts at run time. Declaring them is what makes the check
    possible: a run that reads whatever hit table it is pointed at and reports the
    id it found cannot detect being pointed at the wrong one.
    """

    measurement_id: str
    lexicon_id: str
    substrate_id: str
    hit_table: str
    lexicon_content_hash: str
    lexicon_manifest: str = ""
    #: The `substrate.OpportunityLedger` for this frozen run, written by the
    #: program that froze it. Declared here rather than per-cell because it is a
    #: property of the measurement: the same hit table has the same retained
    #: coverage in every cell that reads it. Where it is given, every cell using
    #: this measurement records `four_state_missingness` among its guard outcomes;
    #: where it is not, those cells are unchecked on that axis and their outcome
    #: list says so by not containing it.
    opportunity_ledger: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("measurement_id", "lexicon_id", "substrate_id", "hit_table",
                     "lexicon_content_hash"):
            if not str(getattr(self, name) or "").strip():
                raise MultiverseError(
                    f"measurement {self.measurement_id or '<unnamed>'}: {name} is required"
                )


@dataclass(frozen=True)
class StatisticalChoice:
    """One way of computing uncertainty. Nothing here changes what is measured."""

    statistical_id: str
    estimator: str = interpret.ESTIMATOR
    block_size: int = interpret.DEFAULT_BLOCK_SIZE
    n_bootstrap: int = interpret.DEFAULT_BOOTSTRAP
    seed: int = 0
    floor_coverage: float = 0.9
    floor_blocks: float = 30.0
    floor_explained: float = 0.5
    label: str = ""

    def __post_init__(self) -> None:
        # Accepted exactly as `interpret` accepts it -- both the command-line
        # spelling and the recorded value -- by asking `interpret` rather than
        # restating the rule. A second acceptance table here would be free to
        # drift, and would refuse a design naming an estimator that runs.
        accepted = set(interpret.ESTIMATOR_CHOICES) | set(interpret.ESTIMATOR_CHOICES.values())
        if self.estimator not in accepted:
            raise MultiverseError(
                f"statistical choice {self.statistical_id}: estimator {self.estimator!r} "
                f"is not one of {sorted(accepted)}"
            )


@dataclass(frozen=True)
class Specification:
    """One cell: one of each axis, and the id that identifies it forever.

    "Forever" is the claim ``input_digests`` exists to make true. Without it the
    ids hash declared *strings* -- including paths -- so replacing the bytes at
    `peaksets/query.txt` leaves every id unchanged and two different analyses
    become indistinguishable in the manifest, the effects table and any
    comparison built on them. Same id must mean same science, and only content can
    promise that.
    """

    estimand: Estimand
    measurement: Measurement
    statistical: StatisticalChoice
    #: role -> SHA-256 of the file that role named, or ``NOT_DECLARED``. Empty
    #: only for a specification built outside :meth:`MultiverseDesign.resolve`,
    #: which is a declaration and not yet a cell.
    input_digests: Mapping[str, str] = field(default_factory=dict)
    #: The version of the design vocabulary these strings were written in, so a
    #: field that changes meaning between releases changes the id with it.
    design_schema_version: str = MULTIVERSE_SCHEMA_VERSION

    @property
    def estimand_id(self) -> str:
        return self.estimand.identity(
            {role: self.input_digests[role] for role in ESTIMAND_INPUT_ROLES
             if role in self.input_digests})

    @property
    def cell_id(self) -> str:
        return _stable_id("cell", {
            "estimand": asdict(self.estimand),
            "measurement": asdict(self.measurement),
            "statistical": asdict(self.statistical),
            "inputs": dict(sorted(self.input_digests.items())),
            "design_schema_version": self.design_schema_version,
        })

    def to_dict(self) -> dict[str, Any]:
        """The full identity of the cell, flattened for a manifest row.

        Every field requirement 5 asks a result to carry is here, at the level
        where it is declared, so that a row of the effects table can be traced to
        its specification without reading any other file.
        """
        return {
            "cell_id": self.cell_id,
            "estimand_id": self.estimand_id,
            "measurement_id": self.measurement.measurement_id,
            "statistical_id": self.statistical.statistical_id,
            "query_id": self.estimand.query_id,
            "baseline_id": self.estimand.baseline_id,
            "baseline_population_type": self.estimand.baseline_population_type,
            "baseline_construction_rule": self.estimand.baseline_construction_rule,
            "selection_provenance": self.estimand.selection_provenance,
            "selection_rule": self.estimand.selection_rule,
            "selection_feature_names": list(self.estimand.selection_feature_names),
            "lexicon_id": self.measurement.lexicon_id,
            "lexicon_content_hash": self.measurement.lexicon_content_hash,
            "substrate_id": self.measurement.substrate_id,
            "estimator": self.statistical.estimator,
            "block_size": self.statistical.block_size,
            "n_bootstrap": self.statistical.n_bootstrap,
            "seed": self.statistical.seed,
            "floor_coverage": self.statistical.floor_coverage,
            "floor_blocks": self.statistical.floor_blocks,
            "floor_explained": self.statistical.floor_explained,
        }


@dataclass(frozen=True)
class MultiverseDesign:
    """The declared grid, before anything has been run.

    ``preregistered_threshold`` defaults to :data:`NO_PREREGISTERED_THRESHOLD`
    and is not a number. A threshold invented after the effects are visible is
    the specification search this module exists to make impossible, so the only
    thresholds it will apply are the ones a design declared before the run.
    """

    multiverse_id: str
    estimands: tuple[Estimand, ...]
    measurements: tuple[Measurement, ...]
    statistical_choices: tuple[StatisticalChoice, ...]
    peak_universe_id: str = ""
    preregistered_threshold: str = NO_PREREGISTERED_THRESHOLD
    root: Path = field(default=Path("."), compare=False)

    def __post_init__(self) -> None:
        for name, values in (("estimands", self.estimands),
                             ("measurements", self.measurements),
                             ("statistical_choices", self.statistical_choices)):
            if not values:
                raise MultiverseError(
                    f"a design needs at least one entry on every axis; {name} is empty"
                )
        for label, ids in (
            ("measurement_id", [m.measurement_id for m in self.measurements]),
            ("statistical_id", [s.statistical_id for s in self.statistical_choices]),
        ):
            if len(set(ids)) != len(ids):
                raise MultiverseError(f"duplicate {label} in the design: {sorted(ids)}")

    def resolve_path(self, declared: str) -> Path:
        p = Path(declared)
        return p if p.is_absolute() else self.root / p

    def input_digests(self, estimand: Estimand, measurement: Measurement) -> dict[str, str]:
        """Hash every file this cell's identity depends on.

        A declared path that does not exist is an error here rather than a
        ``NOT_DECLARED``: the design named it, so a missing file is a broken
        design and not an absent input. Only the *optional* roles -- the lexicon
        manifest and the opportunity ledger -- may legitimately be undeclared, and
        those get the token so that "declared and missing" can never hash the same
        as "never declared".
        """
        digests: dict[str, str] = {}
        for role in ESTIMAND_INPUT_ROLES:
            declared = getattr(estimand, role)
            try:
                digests[role] = content_digest(self.resolve_path(declared))
            except OSError as exc:
                raise MultiverseError(
                    f"estimand {estimand.query_id} vs {estimand.baseline_id} declares "
                    f"{role} {declared!r}, which cannot be read ({exc}). The cell's "
                    "identity depends on this file's contents, so there is no cell "
                    "without it."
                ) from exc
        for role in MEASUREMENT_INPUT_ROLES:
            declared = getattr(measurement, role)
            if not declared:
                digests[role] = NOT_DECLARED
                continue
            path = self.resolve_path(declared)
            try:
                digests[role] = content_digest(path)
            except OSError as exc:
                raise MultiverseError(
                    f"measurement {measurement.measurement_id} declares {role} "
                    f"{declared!r}, which cannot be read ({exc}). A declared input that "
                    "is not there is a broken design, not an absent input."
                ) from exc
        return digests

    def specifications(self) -> list[Specification]:
        """Every planned cell, in a deterministic order, with its inputs hashed.

        The order is the declaration order of the three axes, so a design read
        twice plans the same cells in the same sequence -- which is what lets a
        re-run be compared to an earlier one row by row.

        Reading files here rather than in the dataclass is deliberate: an id that
        depends on file contents cannot be computed from a declaration alone, and
        pretending otherwise is how a stable-looking id ends up naming two
        different analyses. Digests are cached per (estimand, measurement) pair, so
        a 36-cell grid over 2 hit tables hashes each table once rather than 18
        times.
        """
        cache: dict[tuple[int, int], dict[str, str]] = {}
        specs = []
        for e_index, estimand in enumerate(self.estimands):
            for m_index, measurement in enumerate(self.measurements):
                key = (e_index, m_index)
                if key not in cache:
                    cache[key] = self.input_digests(estimand, measurement)
                for statistical in self.statistical_choices:
                    specs.append(Specification(
                        estimand=estimand, measurement=measurement,
                        statistical=statistical, input_digests=cache[key]))
        return specs


def read_design(path: str | os.PathLike[str]) -> MultiverseDesign:
    """Read a design from JSON. Paths inside it are relative to the design file."""
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise MultiverseError(f"{p} is not a readable multiverse design ({exc})") from exc
    if not isinstance(payload, Mapping):
        raise MultiverseError(f"{p}: a design is an object, this is {type(payload).__name__}")

    def _build(cls, key):
        rows = payload.get(key)
        if not isinstance(rows, list):
            raise MultiverseError(f"{p}: '{key}' must be a list of objects")
        built = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise MultiverseError(f"{p}: every entry of '{key}' must be an object")
            known = {f for f in cls.__dataclass_fields__}
            unknown = set(row) - known
            if unknown:
                raise MultiverseError(
                    f"{p}: {key} entry has field(s) this release does not define: "
                    f"{sorted(unknown)}. A design is refused rather than partly applied, "
                    "because a silently ignored axis is a specification nobody ran."
                )
            row = dict(row)
            if "selection_feature_names" in row and row["selection_feature_names"] is not None:
                row["selection_feature_names"] = tuple(row["selection_feature_names"])
            built.append(cls(**row))
        return tuple(built)

    return MultiverseDesign(
        multiverse_id=str(payload.get("multiverse_id") or p.stem),
        peak_universe_id=str(payload.get("peak_universe_id") or ""),
        preregistered_threshold=str(
            payload.get("preregistered_threshold") or NO_PREREGISTERED_THRESHOLD),
        estimands=_build(Estimand, "estimands"),
        measurements=_build(Measurement, "measurements"),
        statistical_choices=_build(StatisticalChoice, "statistical_choices"),
        root=p.resolve().parent,
    )


# --------------------------------------------------------------------------- #
# Planning: every cell is written down before any of them runs.
# --------------------------------------------------------------------------- #
def plan(design: MultiverseDesign) -> dict[str, Any]:
    """The specification manifest: every planned cell, with its deterministic ids.

    Written to disk *before* the grid runs. That ordering is the whole of it: a
    manifest produced afterwards can only contain the cells that survived, and
    "which cells were planned" is exactly the question a reader of a multiverse
    needs answered to know whether the reported ones are a selection.
    """
    specs = design.specifications()
    return {
        "schema_version": MULTIVERSE_SCHEMA_VERSION,
        "multiverse_id": design.multiverse_id,
        "peak_universe_id": design.peak_universe_id,
        "preregistered_threshold": design.preregistered_threshold,
        "n_planned_cells": len(specs),
        "input_digests_by_cell": {
            spec.cell_id: dict(sorted(spec.input_digests.items())) for spec in specs
        },
        "axes": {
            "estimands": [{**asdict(e), "estimand_id": e.estimand_id} for e in design.estimands],
            "measurements": [asdict(m) for m in design.measurements],
            "statistical_choices": [asdict(s) for s in design.statistical_choices],
        },
        "specifications": [spec.to_dict() for spec in specs],
    }


# --------------------------------------------------------------------------- #
# Running.
# --------------------------------------------------------------------------- #
@dataclass
class CellResult:
    """What became of one planned cell, whether or not it produced an estimate."""

    cell_id: str
    estimand_id: str
    status: str
    reason: str = ""
    guard_id: str = ""
    effects: list[dict[str, Any]] = field(default_factory=list)
    n_query_peaks: int | None = None
    n_baseline_peaks: int | None = None
    claim_scope: str = ""
    statistical_license: str = ""
    #: (families in this cell's frozen table, family -> regions where it was
    #: searched). Carried so that a (cell, family) pair can distinguish "this
    #: lexicon has no such family" from "it has one and this contrast never
    #: searched it" -- two absences that look identical as a missing effect row.
    measurement_coverage: tuple[set[str], dict[str, set[str]]] | None = None
    #: Query and baseline regions together: the peaks this cell actually compared.
    contrast_regions: set[str] = field(default_factory=set)
    #: What the guards returned *while this cell ran*. `guard_outcomes.json`
    #: records every outcome in the directory but has no cell to attribute them
    #: to -- 144 entries from a 36-cell grid, none of which a reader can join to
    #: the effect it licensed. A result is required to carry its guard outcomes,
    #: so the cell keeps the slice of the log it produced.
    guard_outcomes: list[dict[str, Any]] = field(default_factory=list)

    def row(self, spec_row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **spec_row,
            "status": self.status,
            "reason": self.reason,
            "guard_id": self.guard_id,
            "n_families_estimated": len(self.effects),
            "n_query_peaks": self.n_query_peaks,
            "n_baseline_peaks": self.n_baseline_peaks,
            "claim_scope": self.claim_scope,
            "statistical_license": self.statistical_license,
            "n_guards_run": len(self.guard_outcomes),
            # Not "all passed": with zero guards run that would be vacuously true,
            # and a cell that refused before reaching a guard is exactly the case
            # where a reader must not read reassurance into an empty list.
            "n_guards_failed": sum(1 for g in self.guard_outcomes if not g["passed"]),
        }


@dataclass
class MultiverseResult:
    manifest: dict[str, Any]
    cells: list[CellResult]
    summaries: list[dict[str, Any]]

    @property
    def by_status(self) -> dict[str, int]:
        return {s: sum(1 for c in self.cells if c.status == s) for s in CellStatus.ALL}


def _verify_measurement(hits: Sequence[Any], measurement: Measurement,
                        root: Path) -> None:
    """Refuse a hit table that is not the one the design declared.

    Two refusals here, not three. A table that *mixes* substrates -- rows from two
    frozen hit callers, which is not a measurement of anything -- is already
    refused by :func:`interpret.read_hit_table` before this is reached, and
    re-checking it here would be a second implementation of a rule that already
    has one: it would be unreachable, and unreachable code that reads as a refusal
    is worse than no code, because a reader counts it as protection.

    What is left is what only a *declaration* makes checkable. A run that reads
    whatever table it is pointed at and reports the id it found cannot notice
    being pointed at the wrong table; a design that declares the id up front can.
    So: the frozen run is the declared one, the vocabulary is the declared one,
    and where a compiled lexicon's manifest is named, its content hash agrees with
    the declaration.
    """
    substrates = {getattr(h, "substrate_id", None) for h in hits}
    found = substrates.pop() if substrates else None
    if found != measurement.substrate_id:
        raise MultiverseError(
            f"{measurement.measurement_id}: declared substrate_id "
            f"{measurement.substrate_id} but the table carries {found}"
        )
    lexicons = {getattr(h, "lexicon_id", None) for h in hits}
    if lexicons != {measurement.lexicon_id}:
        raise MultiverseError(
            f"{measurement.measurement_id}: declared lexicon_id "
            f"{measurement.lexicon_id!r} but the table carries {sorted(map(str, lexicons))}"
        )
    if measurement.lexicon_manifest:
        path = Path(measurement.lexicon_manifest)
        path = path if path.is_absolute() else root / path
        try:
            recorded = json.loads(path.read_text(encoding="utf-8")).get("lexicon_content_hash")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise MultiverseError(
                f"{measurement.measurement_id}: lexicon manifest {path.name} could not "
                f"be read ({exc}); a declared content hash with nothing to check it "
                "against is not verified"
            ) from exc
        if recorded != measurement.lexicon_content_hash:
            raise MultiverseError(
                f"{measurement.measurement_id}: declared lexicon_content_hash "
                f"{measurement.lexicon_content_hash} but {path.name} records {recorded}"
            )


def run_multiverse(design: MultiverseDesign, out_dir: str | os.PathLike[str],
                   guard_log: GuardLog | None = None) -> MultiverseResult:
    """Run every planned cell, record every one, and summarise within estimands.

    The manifest is written first, the cells are run in planned order, and each
    one is reduced to exactly one :class:`CellResult` -- including the ones that
    refused, which is why the loop catches broadly and files the traceback rather
    than letting the first surprise end the grid.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = guard_log if guard_log is not None else GuardLog("multiverse", out)

    manifest = plan(design)
    (out / "specification_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    specs = design.specifications()
    spec_rows = {spec.cell_id: spec.to_dict() for spec in specs}
    hit_cache: dict[str, list[Any]] = {}
    peak_cache: dict[str, list[str]] = {}
    # Per measurement: which families the frozen table contains at all, and for
    # each, the regions where it was actually searched. Built once per table --
    # this is what lets a (cell, family) pair say NOT_IN_LEXICON and NOT_SEARCHED
    # apart, rather than both arriving as a missing row.
    coverage_cache: dict[str, tuple[set[str], dict[str, set[str]]]] = {}
    cells: list[CellResult] = []

    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else design.root / p

    def _peaks(rel: str) -> list[str]:
        if rel not in peak_cache:
            peak_cache[rel] = interpret.read_peak_set(_resolve(rel))
        return peak_cache[rel]

    for spec in specs:
        cell = CellResult(cell_id=spec.cell_id, estimand_id=spec.estimand_id,
                          status=CellStatus.ERROR, reason="not run")
        # Where this cell's outcomes start in the shared log. The log is bound to
        # the directory and shared by every cell, so the only way to attribute an
        # outcome to a cell is to note the boundary before the cell runs.
        outcomes_before = len(log.outcomes)
        try:
            key = spec.measurement.measurement_id
            if key not in hit_cache:
                hits = interpret.read_hit_table(_resolve(spec.measurement.hit_table))
                _verify_measurement(hits, spec.measurement, design.root)
                hit_cache[key] = hits
            hits = hit_cache[key]
            if key not in coverage_cache:
                families: set[str] = set()
                searched: dict[str, set[str]] = {}
                for hit in hits:
                    family = str(hit.family_id)
                    # The sentinel is not a family, and this index is the grid's
                    # answer to "which families exist to be asked about". Admitting
                    # it puts a row named `NA` in the coverage tables as though a
                    # family had been measured -- the same fabrication
                    # `interpret.peak_universe` excludes it for, and the one
                    # `HitRecord.__post_init__` already refuses on USED rows. A
                    # measured row that names no family is a gap in the family
                    # assignment; it is not a family.
                    if family == MISSING_SENTINEL:
                        continue
                    families.add(family)
                    if str(hit.missingness) != "not_searched":
                        searched.setdefault(family, set()).add(hit.region_id)
                coverage_cache[key] = (families, searched)
            if spec.measurement.opportunity_ledger:
                # Re-checked per cell rather than once per measurement, because a
                # guard outcome belongs to the cell it licensed: a reader asking
                # "was this effect computed over a substrate whose coverage was
                # verified" must not have to know which cell happened to be first.
                interpret.verify_missingness_against_ledger(
                    hits,
                    read_opportunity_ledger(
                        _resolve(spec.measurement.opportunity_ledger),
                        substrate_id=spec.measurement.substrate_id),
                    guard_log=log)

            query_regions = _peaks(spec.estimand.query_regions)
            baseline_regions = _peaks(spec.estimand.baseline_regions)
            cell.n_query_peaks = len(query_regions)
            cell.n_baseline_peaks = len(baseline_regions)

            query = PeakSetQuery(
                query_id=spec.estimand.query_id,
                region_ids=query_regions,
                selection_provenance=SelectionProvenance(spec.estimand.selection_provenance),
                selection_rule=spec.estimand.selection_rule,
                selection_feature_names=list(spec.estimand.selection_feature_names),
                comparator_id=spec.estimand.baseline_id,
                comparator_region_ids=baseline_regions,
            )
            result = interpret.interpret_query(
                hits, query,
                floors=interpret.HealthFloors(
                    min_intersection_coverage=spec.statistical.floor_coverage,
                    min_blocks=spec.statistical.floor_blocks,
                    min_explained_fraction=spec.statistical.floor_explained),
                block_size=spec.statistical.block_size,
                n_bootstrap=spec.statistical.n_bootstrap,
                seed=spec.statistical.seed,
                estimator=spec.statistical.estimator,
                guard_log=log,
            )
            cell.claim_scope = result.claim_scope
            cell.statistical_license = result.statistical_license
            if not result.interpretation_emitted or not result.effects:
                cell.status = CellStatus.NOT_ESTIMABLE
                cell.reason = (result.suppression_reason
                               or (", ".join(result.floor_failures) if result.floor_failures
                                   else "no effects were emitted and no reason was recorded"))
            else:
                cell.status = CellStatus.SUCCESS
                cell.reason = ""
                cell.effects = [dict(e) for e in result.effects]
        except guards.GuardError as exc:
            cell.status, cell.reason = CellStatus.REFUSED_GUARD, str(exc)
            cell.guard_id = str(exc).split(":", 1)[0]
        except (MultiverseError, SchemaError, interpret.InterpretError) as exc:
            cell.status, cell.reason = CellStatus.REFUSED_SCHEMA, str(exc)
        except Exception as exc:                                   # noqa: BLE001
            # Recorded, never swallowed and never fatal: a grid that stops at its
            # first surprise reports a subset of itself, and the subset is exactly
            # the cells that did not surprise it.
            cell.status = CellStatus.ERROR
            cell.reason = f"{type(exc).__name__}: {exc}"
            (out / f"error_{spec.cell_id}.txt").write_text(
                traceback.format_exc(), encoding="utf-8")
        cell.guard_outcomes = [o.to_dict() for o in log.outcomes[outcomes_before:]]
        cell.measurement_coverage = coverage_cache.get(spec.measurement.measurement_id)
        cell.contrast_regions = (set(peak_cache.get(spec.estimand.query_regions, ()))
                                 | set(peak_cache.get(spec.estimand.baseline_regions, ())))
        cells.append(cell)

    summaries = stability_within_estimand(cells, manifest)
    log.record(
        guards.no_cross_estimand_pooling(summaries, {s["cell_id"]: s
                                                     for s in manifest["specifications"]}),
        subject=(f"{len(summaries)} family stability summaries over "
                 f"{len(cells)} planned cells, against the specification manifest"),
    ).raise_if_failed()

    result = MultiverseResult(manifest=manifest, cells=cells, summaries=summaries)
    _write_outputs(result, spec_rows, out, design)
    return result


# --------------------------------------------------------------------------- #
# Summarising -- descriptively, within one estimand.
# --------------------------------------------------------------------------- #
def stability_within_estimand(cells: Sequence[CellResult],
                              manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One summary per (estimand, family), never per family alone.

    The grouping *is* the scientific claim of this module. Grouping by family
    alone would put an effect against baseline A beside an effect against baseline
    B and call the spread "robustness"; here two baselines are two estimands and
    produce two summaries, so a conclusion that changes between them shows up as
    two rows that disagree rather than one row with a wide interval.

    Everything reported is descriptive: counts, the sign pattern, and the range.
    No cell is scored, and no family is classified robust -- see requirement 8 in
    ``docs/MULTIVERSE_DESIGN.md``. The one derived word, ``sign_agreement``, is a
    statement about the signs actually observed and carries its own denominator.
    """
    by_cell = {s["cell_id"]: s for s in manifest.get("specifications", ())}
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    planned_by_estimand: dict[str, list[str]] = {}
    for cell in cells:
        planned_by_estimand.setdefault(cell.estimand_id, []).append(cell.cell_id)

    for cell in cells:
        if cell.status != CellStatus.SUCCESS:
            continue
        for effect in cell.effects:
            family = effect.get("family_id")
            key = (cell.estimand_id, str(family))
            entry = grouped.setdefault(key, {
                "estimand_id": cell.estimand_id,
                "family_id": family,
                "group_key": f"{cell.estimand_id}|{family}",
                "cell_ids": [],
                "effects": [],
                "measurement_ids": [],
                "statistical_ids": [],
            })
            entry["cell_ids"].append(cell.cell_id)
            entry["effects"].append(effect.get("effect"))
            spec = by_cell.get(cell.cell_id, {})
            entry["measurement_ids"].append(spec.get("measurement_id"))
            entry["statistical_ids"].append(spec.get("statistical_id"))

    summaries: list[dict[str, Any]] = []
    for (estimand_id, family), entry in sorted(grouped.items()):
        values = [v for v in entry["effects"] if isinstance(v, (int, float))]
        n_planned = len(planned_by_estimand.get(estimand_id, ()))
        positive = sum(1 for v in values if v > 0)
        negative = sum(1 for v in values if v < 0)
        summaries.append({
            "estimand_id": estimand_id,
            "family_id": family,
            "group_key": entry["group_key"],
            "cell_ids": list(entry["cell_ids"]),
            "n_cells_planned_in_estimand": n_planned,
            "n_cells_with_estimate": len(values),
            # Not a rate and not a score: the two counts, and the reader divides
            # if they want to. A single "fraction stable" is the generic
            # robustness number this module exists to not produce.
            "n_effects_positive": positive,
            "n_effects_negative": negative,
            "sign_agreement": (f"{max(positive, negative)}/{len(values)} share a sign"
                               if values else NOT_ESTIMABLE_MARKER),
            "effect_min": min(values) if values else NOT_ESTIMABLE_MARKER,
            "effect_median": _median(values) if values else NOT_ESTIMABLE_MARKER,
            "effect_max": max(values) if values else NOT_ESTIMABLE_MARKER,
            "measurement_ids": sorted({m for m in entry["measurement_ids"] if m}),
            "statistical_ids": sorted({s for s in entry["statistical_ids"] if s}),
        })
    return summaries


def family_cell_states(cells: Sequence[CellResult]) -> list[dict[str, Any]]:
    """One row per (cell, family) over the union of families the grid ever saw.

    The union is taken across every measurement, which is what makes
    ``NOT_IN_LEXICON`` a statement rather than a silence: a family that only one
    lexicon contains gets an explicit row in the cells of the lexicon that does
    not, instead of a gap a reader has to interpret.

    Nothing here recomputes an effect. The status is read off what the cell
    returned and what its frozen table contains, and no absent value is given a
    number -- which is the property the whole table exists to make checkable at
    the family x specification level rather than only at the cell level.
    """
    universe: set[str] = set()
    for cell in cells:
        if cell.measurement_coverage:
            universe |= cell.measurement_coverage[0]
        universe |= {str(e.get("family_id")) for e in cell.effects}

    rows: list[dict[str, Any]] = []
    for cell in cells:
        by_family = {str(e.get("family_id")): e for e in cell.effects}
        families, searched = cell.measurement_coverage or (set(), {})
        for family in sorted(universe):
            effect = by_family.get(family)
            if cell.status != CellStatus.SUCCESS:
                status, value = FamilyCellStatus.CELL_REFUSED, ""
            elif family not in families:
                status, value = FamilyCellStatus.NOT_IN_LEXICON, ""
            elif not (searched.get(family, set()) & cell.contrast_regions):
                status, value = FamilyCellStatus.NOT_SEARCHED, ""
            elif effect is None or effect.get("effect") is None:
                status, value = FamilyCellStatus.NOT_ESTIMABLE, ""
            elif effect.get("effect") == 0:
                status, value = FamilyCellStatus.MEASURED_ZERO, effect.get("effect")
            else:
                status, value = FamilyCellStatus.ESTIMATED, effect.get("effect")
            rows.append({
                "cell_id": cell.cell_id,
                "estimand_id": cell.estimand_id,
                "family_id": family,
                "status": status,
                "effect": value,
                "reason": cell.reason if status == FamilyCellStatus.CELL_REFUSED else "",
            })
    return rows


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2)


# --------------------------------------------------------------------------- #
# Writing.
# --------------------------------------------------------------------------- #
def _tsv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(
            "" if row.get(c) is None else str(row.get(c)).replace("\t", " ").replace("\n", " ")
            for c in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


CELL_COLUMNS = (
    "cell_id", "estimand_id", "measurement_id", "statistical_id", "status", "reason",
    "guard_id", "query_id", "baseline_id", "baseline_population_type",
    "baseline_construction_rule", "selection_provenance", "selection_rule",
    "selection_feature_names", "lexicon_id", "lexicon_content_hash", "substrate_id",
    "estimator", "block_size", "n_bootstrap", "seed", "floor_coverage", "floor_blocks",
    "floor_explained", "n_query_peaks", "n_baseline_peaks", "n_families_estimated",
    "claim_scope", "statistical_license", "n_guards_run", "n_guards_failed",
)

CELL_GUARD_COLUMNS = (
    "cell_id", "estimand_id", "guard_id", "passed", "detail", "subject",
)

EFFECT_COLUMNS = (
    "cell_id", "estimand_id", "family_id", "effect", "ci_low", "ci_high", "p_value",
    "q_value", "inference_capability", "n_blocks", "n_bootstrap_valid",
    "query_id", "baseline_id", "baseline_population_type", "baseline_construction_rule",
    "selection_provenance", "selection_rule", "selection_feature_names",
    "lexicon_id", "lexicon_content_hash", "substrate_id",
    "estimator", "block_size", "n_bootstrap", "seed",
)

FAMILY_CELL_COLUMNS = (
    "cell_id", "estimand_id", "family_id", "status", "effect", "reason",
)

SUMMARY_COLUMNS = (
    "estimand_id", "family_id", "n_cells_planned_in_estimand", "n_cells_with_estimate",
    "n_effects_positive", "n_effects_negative", "sign_agreement",
    "effect_min", "effect_median", "effect_max", "measurement_ids", "statistical_ids",
)


def _write_outputs(result: MultiverseResult, spec_rows: Mapping[str, Mapping[str, Any]],
                   out: Path, design: MultiverseDesign) -> None:
    cell_rows = [cell.row(spec_rows[cell.cell_id]) for cell in result.cells]
    _tsv(out / "cells.tsv", CELL_COLUMNS, cell_rows)
    _tsv(out / "dropped_cells.tsv", CELL_COLUMNS,
         [r for r in cell_rows if r["status"] != CellStatus.SUCCESS])
    # The same outcomes as `guard_outcomes.json`, with the cell they licensed.
    # The JSON stays the authority -- this is a join, not a second record, and it
    # copies `passed` and `detail` without touching either.
    _tsv(out / "cell_guard_outcomes.tsv", CELL_GUARD_COLUMNS, [
        {"cell_id": cell.cell_id, "estimand_id": cell.estimand_id, **outcome}
        for cell in result.cells for outcome in cell.guard_outcomes
    ])

    effect_rows = []
    for cell in result.cells:
        if cell.status != CellStatus.SUCCESS:
            continue
        spec = spec_rows[cell.cell_id]
        for effect in cell.effects:
            ci = effect.get("ci") or [None, None]
            effect_rows.append({
                **spec,
                "cell_id": cell.cell_id,
                "family_id": effect.get("family_id"),
                "effect": effect.get("effect"),
                "ci_low": ci[0], "ci_high": ci[1],
                "p_value": effect.get("p_value"),
                "q_value": effect.get("q_value"),
                "inference_capability": effect.get("inference_capability"),
                "n_blocks": effect.get("n_blocks"),
                "n_bootstrap_valid": effect.get("n_bootstrap_valid"),
            })
    _tsv(out / "family_effects.tsv", EFFECT_COLUMNS, effect_rows)
    _tsv(out / "family_cells.tsv", FAMILY_CELL_COLUMNS, family_cell_states(result.cells))
    _tsv(out / "stability_by_estimand.tsv", SUMMARY_COLUMNS, result.summaries)
    (out / "specification_curve.md").write_text(
        _render_curve(result, design), encoding="utf-8")


def _render_curve(result: MultiverseResult, design: MultiverseDesign) -> str:
    """The document a person reads. Descriptive, and explicit about what is missing."""
    counts = result.by_status
    lines = [
        f"# Specification curve — {design.multiverse_id}",
        "",
        f"Schema version {MULTIVERSE_SCHEMA_VERSION}. "
        f"{counts[CellStatus.SUCCESS]} of {len(result.cells)} planned cells produced "
        "an estimate.",
        "",
        "## 1. Every planned cell",
        "",
        "| status | cells |",
        "|---|---|",
    ]
    lines += [f"| `{status}` | {counts[status]} |" for status in CellStatus.ALL]
    lines += [
        "",
        "Every planned cell appears in `cells.tsv` exactly once, whether or not it "
        "produced an estimate; the non-`SUCCESS` ones are repeated in "
        "`dropped_cells.tsv` with their reason. A cell that could not be estimated "
        "is a finding about the design, most often that a baseline is too small for "
        "the block bootstrap — and a grid reporting only the cells that worked has "
        "selected on its outcome.",
        "",
        "## 2. The axes, kept apart",
        "",
        f"- **Estimands** ({len(design.estimands)}): the questions. A different "
        "baseline population is a different question, never a different measurement "
        "of one.",
        f"- **Measurements** ({len(design.measurements)}): lexicon and frozen hit "
        "table.",
        f"- **Statistical choices** ({len(design.statistical_choices)}): estimator, "
        "block size, replicates, seed, floors.",
        "",
        "## 3. Stability, within one estimand at a time",
        "",
    ]
    if design.preregistered_threshold == NO_PREREGISTERED_THRESHOLD:
        lines += [
            "**No threshold was preregistered for this design, so nothing below is "
            "classified as robust.** What is reported is the count of cells that "
            "produced an estimate, the signs observed, and the range. A threshold "
            "chosen after seeing these numbers would be the specification search this "
            "report exists to make visible.",
            "",
        ]
    else:
        lines += [f"Preregistered threshold: `{design.preregistered_threshold}`.", ""]

    by_estimand: dict[str, list[dict[str, Any]]] = {}
    for summary in result.summaries:
        by_estimand.setdefault(summary["estimand_id"], []).append(summary)
    for estimand in design.estimands:
        rows = by_estimand.get(estimand.estimand_id, [])
        lines += [
            f"### {estimand.query_id} vs {estimand.baseline_id}",
            "",
            f"- baseline population type: `{estimand.baseline_population_type}`",
            f"- construction rule: {estimand.baseline_construction_rule}",
            f"- selection: `{estimand.selection_provenance}` — {estimand.selection_rule}",
            f"- estimand id: `{estimand.estimand_id}`",
            "",
        ]
        if not rows:
            lines += ["No cell in this estimand produced an estimate. "
                      "See `dropped_cells.tsv` for why.", ""]
            continue
        lines += ["| family | cells with an estimate | signs | min | median | max |",
                  "|---|---|---|---|---|---|"]
        for row in sorted(rows, key=lambda r: str(r["family_id"])):
            lines.append(
                f"| {row['family_id']} | {row['n_cells_with_estimate']}"
                f"/{row['n_cells_planned_in_estimand']} | {row['sign_agreement']} | "
                f"{_fmt(row['effect_min'])} | {_fmt(row['effect_median'])} | "
                f"{_fmt(row['effect_max'])} |")
        lines.append("")

    lines += [
        "## 4. What this report does not do",
        "",
        "It does not combine estimands. There is no number here that averages an "
        "effect against one baseline with an effect against another: those are "
        "answers to two questions, and `guards.no_cross_estimand_pooling` refuses a "
        "summary that spans them, reading each cell's estimand from the manifest "
        "written before the run rather than from the code that grouped them.",
        "",
        "It does not fill a missing cell with zero. A family that was not estimable "
        f"in a cell is recorded as `{NOT_ESTIMABLE_MARKER}` and is absent from that "
        "cell's counts, never present as a measured zero.",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return value if isinstance(value, str) else f"{value:.6g}"
