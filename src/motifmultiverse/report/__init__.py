"""Render one recorded interpretation as an audit report -- and nothing else.

See ``README.md`` in this directory for rule / failure / check. The rule is that
every number rendered carries its denominator, its baseline population and its
provenance; the failure that produced it is that the same data supported both
"replicates exactly" and "4x stronger, prediction falsified", differing only in
whether the baseline was the unselected universe or a residual subset, and that a
bootstrap resolution floor was printed as though it were a measured *p* value.

Four properties hold everywhere in this module, and each one is a failure that
already happened somewhere:

1. **Nothing is recomputed.** Every figure is ``str()`` of a field a stage
   recorded. Where the rule wants a denominator, the denominator is NAMED
   (``n_in_universe = 8277 of n_submitted = 8277``) rather than divided into a
   ratio the renderer invented. A report that recomputes is a second
   implementation of the estimator, and a second implementation can disagree
   with the first while looking authoritative.
2. **Nothing is defaulted.** A field the record does not carry is a refusal
   (:class:`ReportError`) or the literal token ``NOT RECORDED`` -- never a
   value.

   ``baseline_population`` is the case that matters, and it needs saying
   exactly, because the rule in ``README.md`` names a field no artifact in this
   package carries. Two things are true and they are not the same thing:

   * **Which peak set** an effect is against IS recorded, as ``comparator_id``
     beside ``n_comparator_peaks``, and a cross-condition effect that carries
     neither is REFUSED by :func:`_section_effects` -- not drawn with an empty
     cell. That refusal runs on every render and uses
     ``guards.comparator_declared``'s own predicate.
   * **What kind of population** it is -- an unselected universe, or a residual
     subset with the relevant peaks already removed -- is NOT recorded anywhere.
     That distinction is the whole founding failure: the same measurements read
     as "replicates exactly" against one and "four times stronger, prediction
     falsified" against the other. So §10 prints ``baseline_population: NOT
     RECORDED`` on every report, unconditionally, and names the field that would
     have said it.

   ``figures=True`` additionally refuses outright, for a caller who will not
   accept a document whose baseline kind is unrecorded. It is off by default
   because turning it on makes every artifact this package currently produces
   unrenderable, and a report that cannot be run is not a check.
3. **An absence never becomes a positive fact.** If the record does not say
   something, this report says it does not know, and names the field that would
   have said it. In particular ``guards.GUARDS_AWAITING_INPUT`` lists guards
   that have no call site; a guard's *absence* from that list says nothing about
   whether it ran, and this module prints exactly that rather than inferring the
   opposite. (An earlier attempt inferred the opposite and named a call site.
   That is fabrication, and it is why this paragraph exists.)

   What this report may now say about a guard changed, and only this much: a run
   writes :mod:`motifmultiverse.guard_log`'s ``guard_outcomes.json`` beside its
   result, so where that file exists §11 renders the outcomes it holds --
   verbatim, the guard's own ``passed`` and its own ``detail`` -- and §10 stops
   saying no artifact records one. Where it does *not* exist, which is every
   artifact produced before that file did, §10 says so in the old terms. The
   inference that stays forbidden is the reverse one: a guard **absent** from a
   present record is not thereby recorded as not having run, because a stage that
   binds no output directory writes no entries at all.
4. **A missing block is never inferred from a failed one.** Section rendering
   branches on ``composition is None`` / ``effects is None`` /
   ``two_part_effects is None`` and on the record's own ``emitted_order`` --
   never on ``floor_failures`` being non-empty. A comparator-side floor failure
   does not make a query's composition unpublishable, and collapsing the two
   made a run whose composition was perfectly good render as though it had
   produced no numbers at all.

Markdown only. ``--html`` / ``--docx`` are refused rather than silently served
markdown, on the precedent of ``cli._run_validate`` refusing ``--fimo-heldout``
as "a semantic no-op": the gap between what was specified and what ran is the
thing this package exists to close, and quietly answering a different request is
that gap.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from motifmultiverse import guard_log, guards
from motifmultiverse.schema import SchemaError

__all__ = [
    "ReportError", "NOT_RECORDED", "NOT_COMPUTED", "BIAS_AXIS_IDS",
    "BIAS_LEDGER_COLUMNS", "BIAS_LEDGER_RESOURCE", "GuardRecord",
    "packaged_bias_ledger_path", "read_bias_ledger", "read_guard_record",
    "render_markdown", "run",
]

#: A field no artifact in this package carries. Printed as a token, never as a
#: value, so "the record is silent" cannot be read as "the record says no".
NOT_RECORDED = "NOT RECORDED"

#: A field that exists in a schema but that no producer in this release writes
#: (``validate.StabilityResult.affected_interval``; any ``ci`` under an estimator
#: that emits none). Rendered, never omitted -- an omitted column reads as a
#: column that was not applicable.
NOT_COMPUTED = "NOT COMPUTED"

#: ``bias_ledger.tsv`` is the authoritative twin of ``docs/BIAS_LEDGER.md`` (that
#: file says so itself, where its English gloss differs), and the rule in this
#: directory's README names the TSV. Its shape is asserted rather than trusted: a
#: ledger that silently lost an axis still reads as a complete accounting of
#: biases.
BIAS_LEDGER_COLUMNS = ("axis_id", "bias", "mechanism", "control")
BIAS_AXIS_IDS = tuple(f"BA-{i:02d}" for i in range(1, 21))

#: The ledger ships *inside* this module, beside the code that reads it, and is
#: listed in ``[tool.setuptools.package-data]``. It used to live in ``docs/`` and
#: be resolved by walking up from ``__file__``, which is the same defect
#: ``adjudicate.packaged_criteria_path`` was written for: the path resolved under
#: an editable install and pointed above ``site-packages`` under a wheel, so
#: ``motifmultiverse report`` -- whose default this is -- refused on every plain
#: ``pip install``. Rendering the report needs the ledger, so the ledger is part
#: of the distribution rather than of the repository around it.
BIAS_LEDGER_RESOURCE = "bias_ledger.tsv"


class ReportError(SchemaError):
    """This report cannot be rendered as asked.

    A subclass of ``SchemaError`` so ``cli.main`` already maps it to exit **4**
    -- "a refusal is not a crash... the message says which rule declined it" --
    without ``report`` needing its own arm in that handler.
    """


# --------------------------------------------------------------------------- #
# Reading recorded values. Nothing on this path computes anything.
# --------------------------------------------------------------------------- #
def _field(record: Any, key: str, where: str) -> Any:
    """Read ``key`` off a recorded mapping, or refuse.

    There is no ``default`` parameter and there will not be one. A default here
    is indistinguishable, in the rendered output, from a value the run actually
    produced -- which is the failure this module was written against.
    """
    if not isinstance(record, dict):
        raise ReportError(
            f"{where} is not a record; refusing to render fields off a {type(record).__name__}"
        )
    if key not in record:
        raise ReportError(
            f"{where} carries no {key!r}. Refusing to render it: this report has no default for a "
            "field a stage did not record, because a default is indistinguishable from a result."
        )
    return record[key]


def _s(value: Any) -> str:
    """The recorded value as text. No arithmetic anywhere on this path.

    ``None`` renders as ``null`` because the record is JSON and ``null`` is the
    token it contains. Fields whose absence the contract gives a specific
    sentence to (``two_part_effects``, ``p_value``, ``q_value``, ``ci``) are
    branched on before they reach here: beside an effect size, ``null`` still
    reads as a small number that failed to print.
    """
    return "null" if value is None else str(value)


def _cell(value: Any) -> str:
    """``_s`` for a markdown table cell; escapes the cell delimiter only."""
    return _s(value).replace("|", r"\|")


def _load_json(path: Path, role: str) -> Any:
    """Read one recorded artifact, or refuse naming it.

    Both of this stage's inputs are JSON written by another stage, and both can
    arrive truncated: a run killed mid-write, a file edited by hand, a directory
    assembled by a pipeline that copied only part of it. That is a refusal --
    exit 4, the message naming the file and the rule -- and not the traceback and
    undocumented exit 1 that an unwrapped ``JSONDecodeError`` produced, because a
    caller distinguishing "the tool declined" from "the tool broke" reads the exit
    code. ``compile`` already wraps its decisions payload for the same reason.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(
            f"{str(path)!r} is not readable {role} JSON ({exc}). This report renders recorded "
            "fields and reconstructs none of them, so a record it cannot parse is a report it "
            "cannot render."
        ) from exc


def _find_anywhere(node: Any, key: str) -> list[Any]:
    """Every value recorded under ``key``, at any nesting depth.

    Used for ``baseline_population``, which no artifact in this release carries
    at any level. Searching, rather than asserting the absence from a belief
    written here, means the day a producer starts writing that field this report
    renders it instead of continuing to print ``NOT RECORDED``.
    """
    found: list[Any] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                found.append(v)
            found.extend(_find_anywhere(v, key))
    elif isinstance(node, list):
        for item in node:
            found.extend(_find_anywhere(item, key))
    return found


def read_bias_ledger(path: str | os.PathLike[str]) -> list[list[str]]:
    """Read the bias ledger TSV verbatim, or refuse.

    Verbatim: the cells reach the report as the TSV spells them, and nothing on
    this path rewrites, translates or reflows them -- ``docs/BIAS_LEDGER.md``
    records that the TSV is authoritative wherever that file's own prose differs
    from it. The "enforced here" column of that Markdown file is this
    repository's annotation about itself, not ledger content; it is not in the
    TSV and so is not in the report.
    """
    p = Path(path)
    if not p.is_file():
        raise ReportError(
            f"bias ledger {str(p)!r} is absent. The rule in src/motifmultiverse/report/README.md "
            "names the bias ledger TSV as the source of section 9; a report rendered without it "
            "would present an unaccounted analysis as an accounted one."
        )
    with p.open(newline="", encoding="utf-8") as fh:
        rows = [row for row in csv.reader(fh, delimiter="\t") if row]
    if not rows or tuple(rows[0]) != BIAS_LEDGER_COLUMNS:
        raise ReportError(
            f"bias ledger {str(p)!r} has header {rows[0] if rows else []}; expected exactly "
            f"{list(BIAS_LEDGER_COLUMNS)}."
        )
    body = rows[1:]
    bad_width = [r for r in body if len(r) != len(BIAS_LEDGER_COLUMNS)]
    if bad_width:
        raise ReportError(
            f"bias ledger {str(p)!r} has {len(bad_width)} row(s) that are not "
            f"{len(BIAS_LEDGER_COLUMNS)} columns wide; refusing to render a ledger whose columns "
            "cannot be told apart."
        )
    axis_ids = tuple(r[0] for r in body)
    if axis_ids != BIAS_AXIS_IDS:
        raise ReportError(
            f"bias ledger {str(p)!r} lists axes {list(axis_ids)}; expected exactly "
            f"{list(BIAS_AXIS_IDS)}. A ledger missing an axis still reads as a complete accounting."
        )
    return body


class GuardRecord:
    """What the directory says about the guards that ran in it -- in three states.

    ``ABSENT``, ``UNREADABLE`` and ``PRESENT`` are kept apart all the way to the
    page, because collapsing any two of them is the founding failure's shape in
    miniature. An absent record means *nothing was written down*, which is the
    state of every artifact produced before ``guard_outcomes.json`` existed and of
    every stage run through the library with no output directory bound; it is not
    evidence that no guard ran. An unreadable record is not an absent one -- a
    truncated log in a directory that recorded outcomes must not render as a
    directory that recorded none. Only ``PRESENT`` licenses the sentence "this
    guard returned this".

    :meth:`partition` answers the further question the states above do not
    answer: several runs legitimately write one ``--out``, so an outcome in the
    file is not automatically an outcome of the run that wrote
    ``interpretation.json``. Each entry records how many provenance records
    existed when it was written, and ``run_status.artifacts_are_from`` records the
    same number for the run whose artifacts are lying here. Where both are
    present the join is exact; where either is missing this class says the
    outcomes cannot be attributed rather than guessing that the newest ones are
    the right ones.
    """

    ABSENT = "ABSENT"
    PRESENT = "PRESENT"
    UNREADABLE = "UNREADABLE"

    def __init__(self, state: str, outcomes: list[dict[str, Any]] | None = None,
                 detail: str = "", artifact_run: Any = None) -> None:
        self.state = state
        self.outcomes = outcomes or []
        self.detail = detail
        #: The `provenance_records` index of the run that wrote the artifacts in
        #: this directory, or None when `run_status.json` does not say.
        self.artifact_run = artifact_run

    @property
    def attributable(self) -> bool:
        return self.state == self.PRESENT and isinstance(self.artifact_run, int)

    def partition(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """(outcomes of the run that wrote the artifacts here, outcomes of others).

        Everything lands in the second list when the join cannot be made, so a
        caller that renders only the first can never present another run's
        outcome as this artifact's.
        """
        if not self.attributable:
            return [], list(self.outcomes)
        mine, others = [], []
        for row in self.outcomes:
            (mine if row.get("provenance_records") == self.artifact_run else others).append(row)
        return mine, others


def read_guard_record(project: str | os.PathLike[str]) -> GuardRecord:
    """Read ``guard_outcomes.json`` and the run status that says whose it is.

    Never raises. This is the one input of the report whose *absence is the
    normal case* -- every artifact written before the log existed has none -- so
    an unreadable or missing file here downgrades what the page may say and does
    not refuse the page. (``interpretation.json`` and ``provenance.json`` still
    refuse: without those there is nothing to render at all.)
    """
    directory = Path(project)
    try:
        outcomes = guard_log.read_guard_outcomes(directory)
    except guard_log.GuardLogError as exc:
        # The message names the file by absolute path, and a report is a document
        # that gets sent to people. `redaction_policy` exempts the provenance
        # `command` string by name and nothing else, so this one is reduced to the
        # file name: the reader is looking at the directory it is in.
        detail = str(exc).replace(str(directory / guard_log.GUARD_OUTCOMES_FILENAME),
                                  guard_log.GUARD_OUTCOMES_FILENAME)
        return GuardRecord(GuardRecord.UNREADABLE, detail=detail)
    if outcomes is None:
        return GuardRecord(GuardRecord.ABSENT)
    from motifmultiverse.run_status import read_run_status

    status = read_run_status(directory) or {}
    origin = status.get("artifacts_are_from")
    artifact_run = origin.get("provenance_records") if isinstance(origin, dict) else None
    return GuardRecord(GuardRecord.PRESENT, outcomes=outcomes,
                       artifact_run=artifact_run if isinstance(artifact_run, int) else None)


def packaged_bias_ledger_path() -> Path:
    """Locate the bias ledger that ships with the package.

    This is ``report``'s default and therefore the CLI's default, so it has to
    resolve wherever the package is installed. It used to be
    ``Path(__file__).parents[3] / "docs" / "bias_ledger.tsv"`` -- the repository
    layout -- which meant ``motifmultiverse report interpretation/`` worked from a
    checkout and refused from a wheel, for a file the wheel had never contained.
    That refusal was correct about the file being absent and wrong about whose
    fault that was: a rendering rule the distribution cannot satisfy is a
    packaging defect, not an honest refusal, and
    ``adjudicate.packaged_criteria_path`` records the same failure for the
    criterion registry.

    :func:`read_bias_ledger`'s refusal still stands for a ledger the *caller*
    named with ``--bias-ledger`` and that is absent or malformed. What no longer
    happens is the default refusing on a correctly installed package.
    """
    from importlib.resources import as_file, files

    resource = files(__package__).joinpath(BIAS_LEDGER_RESOURCE)
    with as_file(resource) as concrete:
        return Path(concrete)


# --------------------------------------------------------------------------- #
# Sections. Each returns markdown lines; none computes a number.
# --------------------------------------------------------------------------- #
def _health_block(name: str, record: Any) -> list[str]:
    """One health record, every figure beside the denominator it was taken over.

    The two rates are printed as recorded, with their counts named next to them.
    The counts are not divided here and the result is not checked against the
    recorded rate: a renderer that re-derives a recorded ratio is a second
    estimator, and when the two disagree the reader cannot tell which one the run
    used.
    """
    floors = _field(record, "floors", f"`{name}`")
    return [
        f"#### `{name}`",
        "",
        f"- `intersection_coverage` = {_s(_field(record, 'intersection_coverage', name))} "
        f"(`n_in_universe` = {_s(_field(record, 'n_in_universe', name))} of "
        f"`n_submitted` = {_s(_field(record, 'n_submitted', name))})",
        f"- `n_blocks` = {_s(_field(record, 'n_blocks', name))} at "
        f"`block_size` = {_s(_field(record, 'block_size', name))}",
        f"- `explained_fraction` = {_s(_field(record, 'explained_fraction', name))} "
        f"(`n_with_used_hit` = {_s(_field(record, 'n_with_used_hit', name))} of "
        f"`n_searched` = {_s(_field(record, 'n_searched', name))})",
        "- `floors` = min_intersection_coverage "
        f"{_s(_field(floors, 'min_intersection_coverage', f'{name}.floors'))}, min_blocks "
        f"{_s(_field(floors, 'min_blocks', f'{name}.floors'))}, min_explained_fraction "
        f"{_s(_field(floors, 'min_explained_fraction', f'{name}.floors'))}",
        f"- `floor_failures` = {_s(_field(record, 'floor_failures', name))}",
        "",
    ]


def _section_identity(interp: dict[str, Any]) -> list[str]:
    w = "interpretation.json"
    return [
        "## 1. RUN IDENTITY",
        "",
        f"- `query_id` = {_s(_field(interp, 'query_id', w))}",
        f"- `substrate_id` = {_s(_field(interp, 'substrate_id', w))}",
        f"- `lexicon_id` = {_s(_field(interp, 'lexicon_id', w))} "
        "(a declared string; see section 10 -- it is not a lexicon content hash)",
        f"- `input_scale` = {_s(_field(interp, 'input_scale', w))}",
        f"- `estimator` = {_s(_field(interp, 'estimator', w))}",
        f"- `estimators_implemented` = {_s(_field(interp, 'estimators_implemented', w))}",
        f"- `estimators_defined` = {_s(_field(interp, 'estimators_defined', w))}",
        "",
        "`estimator` is printed against the recorded `estimators_implemented` set rather than "
        "against a literal held by this renderer, so a consumer reads the run's own notion of "
        "which estimators exist. `estimators_defined` is the larger set: an estimator can be named "
        "by the schema and not be runnable.",
        "",
    ]


def _section_provenance(records: Any) -> list[str]:
    """Every provenance record, in the order the append-log recorded them."""
    if not isinstance(records, list):
        raise ReportError(
            "provenance.json is not a list; the recorder writes an append-log, and a report that "
            "read one record would silently drop every earlier invocation."
        )
    lines = [
        "## 2. PROVENANCE",
        "",
        f"`provenance.json` is an append-log of {len(records)} record(s); all are rendered.",
        "",
    ]
    for i, rec in enumerate(records):
        where = f"provenance.json[{i}]"
        lines += [
            f"### Record {i}",
            "",
            f"- `subcommand` = {_s(_field(rec, 'subcommand', where))}",
            f"- `timestamp_utc` = {_s(_field(rec, 'timestamp_utc', where))}",
            f"- `substrate_id` = {_s(_field(rec, 'substrate_id', where))}",
            f"- `input_scale` = {_s(_field(rec, 'input_scale', where))}",
            f"- `random_seed` = {_s(_field(rec, 'random_seed', where))}",
            f"- `software` = {_s(_field(rec, 'software', where))}",
            f"- `schema_version` = {_s(_field(rec, 'schema_version', where))}",
            f"- `redaction_policy` = {_s(_field(rec, 'redaction_policy', where))}",
            "",
            "`command`, verbatim -- the recorder's redaction policy names it as the one unredacted "
            "field. Nothing in this report is parsed out of it: a flag read off a command line is a "
            "claim about the run that no field of the result supports.",
            "",
            "```",
            _s(_field(rec, "command", where)),
            "```",
            "",
            "| input role | name | sha256 |",
            "| --- | --- | --- |",
        ]
        inputs = _field(rec, "inputs", where)
        if not isinstance(inputs, dict):
            raise ReportError(f"{where}.inputs is not a role-keyed map")
        for key in sorted(inputs):
            role, _, name = key.partition(":")
            lines.append(f"| `{_cell(role)}` | {_cell(name)} | `{_cell(inputs[key])}` |")
        roles = ", ".join(f"`{k}`" for k in sorted(inputs)) or "(none)"
        lines += ["", f"This record names the input roles: {roles}.", ""]
        # A statement about the RECORD, derived from the keys the record has --
        # not a statement about what the run verified. That distinction is the
        # point: `interpret.verify_against_manifest` may or may not have run, and
        # no field anywhere says which.
        manifest_roles = sorted(k for k in inputs if k.split(":", 1)[0] == "substrate_manifest")
        if manifest_roles:
            named = ", ".join(f"`{k}`" for k in manifest_roles)
            lines.append(
                f"This record names {len(manifest_roles)} `substrate_manifest:` input ({named}). "
                "Whether `interpret.verify_against_manifest` ran against it is still not recorded "
                "by any field."
            )
        else:
            lines.append(
                "This record names no `substrate_manifest:` input; whether "
                "`interpret.verify_against_manifest` ran is not recorded by any field."
            )
        lines.append("")
    return lines


def _section_health(interp: dict[str, Any]) -> list[str]:
    lines = [
        "## 3. HEALTH",
        "",
        "Rendered before any effect, because an effect size travels and the disclaimer beside it "
        "does not.",
        "",
    ]
    query_health = _field(interp, "query_health", "interpretation.json")
    # `health` is the DEPRECATED alias of `query_health`. It is not rendered as a
    # second block -- the same numbers under two names is how a reader comes to
    # believe two things were measured. It is compared, and a disagreement is a
    # refusal rather than a silent choice of which one to believe (R14).
    if "health" in interp and interp["health"] != query_health:
        raise ReportError(
            "interpretation.json carries a deprecated `health` that differs from `query_health`. "
            "Refusing to render either: this report cannot choose which of two disagreeing health "
            "records the run's floors were actually applied to."
        )
    lines += _health_block("query_health", query_health)

    comparator_health = _field(interp, "comparator_health", "interpretation.json")
    if comparator_health is None:
        lines += [
            "#### `comparator_health`",
            "",
            "`comparator_health` is null: this record carries no comparator health block. That is "
            "a statement about the record, not a statement that no comparator was used.",
            "",
        ]
    else:
        lines += _health_block("comparator_health", comparator_health)

    contrast = _field(interp, "contrast_health", "interpretation.json")
    lines += ["#### `contrast_health`", ""]
    if contrast is None:
        lines += [
            "`contrast_health` is null: this record carries no query/comparator contrast block, so "
            "this report does not know how far the two peak sets overlap.",
            "",
        ]
    else:
        lines += [
            f"- `n_shared_peaks` = {_s(_field(contrast, 'n_shared_peaks', 'contrast_health'))}",
            f"- `shared_blocks` = {_s(_field(contrast, 'shared_blocks', 'contrast_health'))}",
            f"- `union_blocks` = {_s(_field(contrast, 'union_blocks', 'contrast_health'))}",
            f"- `passed` = {_s(_field(contrast, 'passed', 'contrast_health'))}",
            f"- `floor_failures` = {_s(_field(contrast, 'floor_failures', 'contrast_health'))}",
            "",
            # No numerals in this renderer's own prose. Every figure a reader sees
            # must be interpolated from the record, or the report becomes a second
            # place where numbers live and the two can drift -- which is the whole
            # thing it exists to prevent. The point survives without the figures.
            "`shared_blocks` cannot stand in for `n_shared_peaks`: two peak sets can be the same "
            "set while their block counts differ by one, so nearly-equal block counts read as "
            "ordinary overlap while every peak is shared. Only `n_shared_peaks` answers "
            "disjointness.",
            "",
        ]
    # Printed unconditionally, side by side. docs/DATA_MODEL.md allows these two
    # to diverge -- the top-level list is the operative one, the contrast list an
    # unconditional union -- so collapsing them into one line makes a divergence
    # unreadable exactly when it matters.
    top = _s(_field(interp, "floor_failures", "interpretation.json"))
    contrast_failures = (NOT_RECORDED if contrast is None
                         else _s(_field(contrast, "floor_failures", "contrast_health")))
    lines += [
        f"`floor_failures` (top level, operative) = {top} ; "
        f"`contrast_health.floor_failures` (unconditional union) = {contrast_failures}",
        "",
    ]
    return lines


def _section_composition(composition: Any) -> list[str]:
    lines = ["## 4. COMPOSITION", ""]
    if composition is None:
        # Branching on `composition is None` -- never on floor_failures. A
        # comparator-side floor failure has nothing to do with whether this block
        # was emitted, and conflating the two suppressed a perfectly good
        # composition once already.
        return lines + [
            "`composition` is null: this run emitted no composition table. That is not 'a "
            "composition was computed and came out empty'.",
            "",
        ]
    lines += [
        "Descriptive. No interval and no *p* value is attached to any figure below, because none "
        "was computed for one. The denominator is `n_peaks_searched`, carried on every row.",
        "",
        "| family_id | n_peaks_with_family | n_peaks_searched | peak_share | mean_coefficient_per_peak |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, row in enumerate(composition):
        w = f"composition[{i}]"
        lines.append(
            f"| {_cell(_field(row, 'family_id', w))} "
            f"| {_cell(_field(row, 'n_peaks_with_family', w))} "
            f"| {_cell(_field(row, 'n_peaks_searched', w))} "
            f"| {_cell(_field(row, 'peak_share', w))} "
            f"| {_cell(_field(row, 'mean_coefficient_per_peak', w))} |"
        )
    lines += ["", f"{len(composition)} row(s).", ""]
    return lines


#: Every field of a recorded effect row, rendered. The list is the row's own key
#: set, in reading order -- an effects table with a column dropped is an effect
#: size whose N, whose estimator or whose comparator went unread.
_EFFECT_COLUMNS = (
    "id", "family_id", "effect", "ci", "p_value", "q_value", "inference_capability",
    "estimator", "comparator_id", "n_query_peaks", "n_comparator_peaks", "n_blocks",
    "n_bootstrap", "n_bootstrap_valid", "block_size", "random_seed", "is_cross_condition",
)


def _section_effects(effects: Any, baseline_block: list[str]) -> list[str]:
    lines = ["## 5. EFFECTS", ""]
    if effects is None:
        return lines + [
            "`effects` is null: this run emitted no effect estimates. That is not 'effects were "
            "estimated and none was distinguishable from zero'.",
            "",
        ]
    # The specified check, on the field that actually plays the role the rule
    # names. No artifact carries a literal `baseline_population`; what carries it
    # is `comparator_id` (which population) beside `n_comparator_peaks` (how
    # large), and §10 says so. A cross-condition effect without one is the exact
    # figure this module's founding failure was made of -- the same measurements
    # read as "replicates exactly" or "four times stronger, prediction falsified"
    # depending on a baseline nobody had written down. So it is refused, not
    # drawn with a NOT RECORDED cell: a rendered effect size is read long after
    # the caveat beside it is forgotten.
    #
    # The predicate is `guards.comparator_declared`'s own, deliberately not a
    # second one written here: two implementations of "is this baseline named"
    # can disagree, and then the report and the guard disagree about the same row.
    unnamed = [
        _s(r.get("id", f"effects[{i}]"))
        for i, r in enumerate(effects)
        if r.get("is_cross_condition", True) and r.get("comparator_id") in (None, "", "NA")
    ]
    if unnamed:
        raise ReportError(
            f"{len(unnamed)} cross-condition effect(s) carry no comparator_id "
            f"({', '.join(unnamed[:5])}): the rule this renderer exists for is that every "
            "figure carries its baseline population, and `comparator_id` is the field that "
            "names it. Refusing to render an effect size whose baseline is not recorded -- "
            "the number outlives any caveat printed beside it."
        )
    comparator_ids = sorted({_s(_field(r, "comparator_id", f"effects[{i}]"))
                             for i, r in enumerate(effects)})
    lines += [
        "Baseline identity, as recorded on the rows: `comparator_id` = "
        f"{', '.join(comparator_ids) if comparator_ids else NOT_RECORDED}.",
        "",
    ]
    lines += baseline_block
    lines += [
        "| " + " | ".join(_EFFECT_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _EFFECT_COLUMNS) + " |",
    ]
    for i, row in enumerate(effects):
        w = f"effects[{i}]"
        cells = []
        for col in _EFFECT_COLUMNS:
            value = _field(row, col, w)
            if col == "ci" and value is None:
                # Never blank: a blank interval column reads as a point estimate
                # that needed no interval.
                cells.append(NOT_COMPUTED)
            elif col in ("p_value", "q_value") and value is None:
                # Never blank and never `n.s.`. A blank reads as "no evidence of
                # an effect"; `n.s.` reads as a test that ran and did not reject.
                # Neither happened. `cli._effect_estimate_rows` writes `NA` in the
                # flat TSV for exactly this reason; here the capability the record
                # itself states is named, so the reader sees WHY it is withheld
                # rather than only that it is.
                cells.append(
                    "WITHHELD -- inference_capability = "
                    f"{_cell(_field(row, 'inference_capability', w))}"
                )
            else:
                cells.append(_cell(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"{len(effects)} row(s).", ""]
    return lines


def _section_two_part(two_part: Any) -> list[str]:
    lines = ["## 6. TWO-PART EFFECTS", ""]
    if two_part is None:
        # Rendered, not skipped. A skipped section is indistinguishable from a
        # section that ran and had nothing to report, and those are different
        # facts about the run.
        return lines + [
            "`two_part_effects` is null: no `usage_definition` was configured; nobody chose a "
            'definition of "used". This is not "computed and found nothing".',
            "",
        ]
    lines += [
        "`conditional_intensity_effect` = null means a side never used the family. It is not 0.0, "
        "and the two must not be read alike.",
        "",
    ]
    for i, row in enumerate(two_part):
        w = f"two_part_effects[{i}]"
        lines += [f"### `{_s(_field(row, 'family_id', w))}`", ""]
        for col in ("probability_effect", "conditional_intensity_effect", "total_effect",
                    "n_used_query", "n_used_comparator", "n_measured_query",
                    "n_measured_comparator", "usage_definition", "usage_threshold",
                    "usage_threshold_source"):
            lines.append(f"- `{col}` = {_s(_field(row, col, w))}")
        lines.append("")
    return lines


def _section_permissions(interp: dict[str, Any]) -> list[str]:
    w = "interpretation.json"
    return [
        "## 7. PERMISSIONS AND CLAIM STRENGTH",
        "",
        "Both axes, always, never one. They do not covary: a query can be statistically licensed "
        "to compute an interval and semantically unable to support the claim the interval looks "
        "like.",
        "",
        f"- `statistical_license` = {_s(_field(interp, 'statistical_license', w))} "
        "-- may this query support inference at all",
        f"- `claim_scope` = {_s(_field(interp, 'claim_scope', w))} "
        "-- what the resulting number may be a claim about",
        f"- `selection_provenance` = {_s(_field(interp, 'selection_provenance', w))} "
        "-- how the peak set was chosen",
        f"- **DEPRECATED** `output_mode` = {_s(_field(interp, 'output_mode', w))} "
        "-- one collapsed grade that cannot represent `SUBSTRATE_CIRCULAR` at all. It is printed "
        "because it is recorded, and labelled because a reader who takes it for the claim-strength "
        "field will read a circular decomposition as a licensed one. `claim_scope` is the field to "
        "read.",
        "",
    ]


def _section_notes(interp: dict[str, Any]) -> list[str]:
    """Contract section 8, rendered ABOVE every number it qualifies.

    Its position is the point, not a layout preference. This report's founding
    failure was a bootstrap resolution floor printed as a measured *p* value; the
    note that discharges it says the implemented estimator supports estimation
    only. Printed below the effects table it would be a footnote to figures
    already read.
    """
    w = "interpretation.json"
    lines = [
        "## 8. CLAIM STRENGTH -- NOTES AND SUPPRESSION",
        "",
        "*(Contract section 8, rendered here, above every number it qualifies.)*",
        "",
        f"- `interpretation_emitted` = {_s(_field(interp, 'interpretation_emitted', w))}",
        f"- `suppression_reason` = {_s(_field(interp, 'suppression_reason', w))}",
        "",
    ]
    notes = _field(interp, "notes", w)
    if notes is None:
        return lines + [f"`notes` = {NOT_RECORDED}", ""]
    if not notes:
        return lines + ["`notes` is an empty list: this run attached no claim-strength note.", ""]
    lines += ["These are warnings about how the numbers below may be read, not footnotes to them:", ""]
    for note in notes:
        lines += [f"> {_s(note)}", ">"]
    lines.append("")
    return lines


def _section_bias_ledger(rows: list[list[str]], source: Path) -> list[str]:
    lines = [
        "## 9. BIAS LEDGER",
        "",
        f"Rendered verbatim from `{source}`: {len(rows)} axes, {BIAS_AXIS_IDS[0]}..{BIAS_AXIS_IDS[-1]}. "
        "The TSV is authoritative where the English gloss in `docs/BIAS_LEDGER.md` differs from it, "
        "so the TSV's own wording appears here, untranslated. That file's \"enforced here\" column "
        "is this repository's annotation about itself and is not ledger content, so it is absent.",
        "",
        "| " + " | ".join(BIAS_LEDGER_COLUMNS) + " |",
        "| --- | --- | --- | --- |",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    lines.append("")
    return lines


def _baseline_block(interp: dict[str, Any], provenance: Any) -> list[str]:
    """What plays the role of ``baseline_population`` -- and why none of it is that field.

    ``comparator_id`` names WHICH FILE the baseline came from.
    ``baseline_population`` (an axis of ``config/specifications.example.yaml``,
    read by no code, belonging to a specification multiverse this release does
    not implement) names WHAT KIND of population it is: an unselected universe or
    a matched background. Those are different claims, and the difference between
    them is exactly the founding failure -- the same data reading as "replicates
    exactly" under one and "4x stronger, prediction falsified" under the other.
    So the field is reported absent, and the four things that do exist are named
    as what they are rather than promoted into it.
    """
    found = (_find_anywhere(interp, "baseline_population")
             + _find_anywhere(provenance, "baseline_population"))
    if found:
        return [f"- `baseline_population` = {_s(found[0])} (recorded).", ""]

    effects = interp.get("effects")
    comparator_ids = (sorted({_s(r.get("comparator_id")) for r in effects})
                      if isinstance(effects, list) and effects else [])
    ch = interp.get("comparator_health")
    contrast = interp.get("contrast_health")

    def _ch(key: str) -> str:
        return NOT_RECORDED if not isinstance(ch, dict) or key not in ch else _s(ch[key])

    comparator_files: list[str] = []
    if isinstance(provenance, list):
        for rec in provenance:
            if isinstance(rec, dict) and isinstance(rec.get("inputs"), dict):
                comparator_files += [f"`{k}` = `{v}`" for k, v in sorted(rec["inputs"].items())
                                     if k.split(":", 1)[0] == "comparator"]
    return [
        f"- `baseline_population`: {NOT_RECORDED}. No artifact in this package carries this field "
        "at any nesting level. It names the KIND of population the baseline is (unselected "
        "universe vs matched background); the configuration axis that would define it is read by "
        "no code in this release. Four recorded things play parts of its role, and none of them is "
        "it:",
        "",
        "  - IDENTITY (which file): `comparator_id` = "
        f"{', '.join(comparator_ids) if comparator_ids else NOT_RECORDED} -- a caller-supplied "
        "string, the sentinel `NA` when undeclared.",
        f"  - DENOMINATORS (how many): `comparator_health.n_submitted` = {_ch('n_submitted')}, "
        f"`n_in_universe` = {_ch('n_in_universe')}, `n_searched` = {_ch('n_searched')}, "
        f"`n_with_used_hit` = {_ch('n_with_used_hit')}.",
        "  - DISJOINTNESS (does it overlap the query): `contrast_health.n_shared_peaks` = "
        + (NOT_RECORDED if not isinstance(contrast, dict) or "n_shared_peaks" not in contrast
           else _s(contrast["n_shared_peaks"])) + ".",
        "  - FILE PROVENANCE (which bytes): "
        + (", ".join(comparator_files) if comparator_files else NOT_RECORDED) + ".",
        "",
        "  Together these say which file the baseline came from, how large it was and that it is "
        "disjoint from the query. They do not say what kind of population it is, and that is the "
        "claim the rule asks for.",
        "",
    ]


def _guard_outcomes_bullet(record: GuardRecord) -> list[str]:
    """The §10 entry for guard outcomes -- one sentence per state, never a default.

    The absent case is the sentence this report carried unconditionally until a
    producer existed to make it false, kept word for word where it is still true.
    """
    if record.state == GuardRecord.ABSENT:
        return [
            f"- Guard outcomes: {NOT_RECORDED}. This directory carries no "
            f"`{guard_log.GUARD_OUTCOMES_FILENAME}`, which is the state of every artifact "
            "written before this package recorded guard outcomes and of any stage run with no "
            "output directory bound. This report may name which guards "
            "`interpret.interpret_query` invokes, as facts about the code path; for THIS "
            "artifact it does not and cannot state that any guard passed. Nothing in the "
            "guards sections below should be read as a guard that passed.",
            "",
        ]
    if record.state == GuardRecord.UNREADABLE:
        return [
            f"- Guard outcomes: a `{guard_log.GUARD_OUTCOMES_FILENAME}` is present here and "
            f"could not be read ({_s(record.detail)}). That is not the same as an absent "
            "record, and this report will not treat it as one: outcomes were written in this "
            "directory and this page cannot say what they were.",
            "",
        ]
    mine, others = record.partition()
    lines = [
        f"- Guard outcomes: recorded. `{guard_log.GUARD_OUTCOMES_FILENAME}` holds "
        f"{len(record.outcomes)} outcome(s) and they are rendered in section 11. Three things "
        "that record still does not establish, and this report does not imply any of them:",
        "",
        "  - It is written by the same run that ran the guard, so it attests **what the guard "
        "returned**, not that the guard was right to return it. A guard applied to a claim "
        "computed by the code it audits corroborates itself, and no log can repair that.",
        "  - A guard **absent** from the record is not recorded as not having run. Entries "
        "exist only for calls made by a stage that bound an output directory; "
        "`compile.probe_backend`'s call belongs to the environment `status` reports on and to "
        "no directory at all.",
        "  - Each outcome describes only the input its `subject` names. It says nothing about "
        "any other artifact in this directory.",
        "",
    ]
    if not record.attributable:
        lines += [
            "  This report additionally cannot tell which run in this directory produced those "
            "outcomes: `run_status.json` records no `artifacts_are_from.provenance_records` to "
            "join them to. They are rendered in section 11 as outcomes recorded HERE, not as "
            "outcomes of the run that wrote this interpretation.",
            "",
        ]
    elif others:
        lines += [
            f"  {len(others)} of them were written by other runs into this same directory and "
            "are listed separately in section 11; they are not this artifact's.",
            "",
        ]
    return lines


def _section_guard_outcomes(record: GuardRecord) -> list[str]:
    """Section 11: the recorded outcomes, verbatim.

    Verbatim in the strict sense this module means everywhere else: ``passed`` is
    the guard's own boolean and ``detail`` is the guard's own sentence, copied.
    This renderer runs no guard, re-derives no verdict, and has no opinion about
    whether a recorded pass was deserved.
    """
    lines = ["## 11. GUARD OUTCOMES RECORDED IN THIS DIRECTORY", ""]
    if record.state == GuardRecord.ABSENT:
        return lines + [
            f"No `{guard_log.GUARD_OUTCOMES_FILENAME}` accompanies this artifact, so there is "
            "no recorded outcome to render. This is the absence of a RECORD and not a record "
            "of an absence: it does not say that no guard ran, and nothing below may be read "
            "as evidence either way.",
            "",
        ]
    if record.state == GuardRecord.UNREADABLE:
        return lines + [
            f"A `{guard_log.GUARD_OUTCOMES_FILENAME}` is present in this directory and could "
            f"not be read: {_s(record.detail)}",
            "",
            "It is not rendered as absent. A directory that recorded outcomes and lost them is "
            "a different state from one that recorded none.",
            "",
        ]
    mine, others = record.partition()

    def _table(rows: list[dict[str, Any]]) -> list[str]:
        out = [
            "| guard_id | stage | passed | subject | detail |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            out.append("| " + " | ".join([
                f"`{_cell(row.get('guard_id', NOT_RECORDED))}`",
                _cell(row.get("stage", NOT_RECORDED)),
                _cell(row.get("passed", NOT_RECORDED)),
                _cell(row.get("subject", NOT_RECORDED)),
                _cell(row.get("detail", NOT_RECORDED)),
            ]) + " |")
        return out

    lines += [
        f"Rendered verbatim from `{guard_log.GUARD_OUTCOMES_FILENAME}`. `passed` is the "
        "guard's own boolean and `detail` its own sentence; this renderer runs no guard and "
        "recomputes no verdict. `subject` is what the stage handed the guard, in the stage's "
        "own words.",
        "",
    ]
    if record.attributable:
        lines += [
            "### Outcomes of the run that wrote the artifacts here (`provenance_records` = "
            f"{_s(record.artifact_run)})",
            "",
        ]
        lines += _table(mine) if mine else [
            "This run recorded no guard outcome. The file exists and holds outcomes of other "
            "runs only; nothing here says a guard ran for this artifact.",
        ]
        lines.append("")
        if others:
            lines += [
                "### Outcomes recorded in this directory by OTHER runs",
                "",
                "Listed because deleting them would hide that this directory was written more "
                "than once; they are not this artifact's and no claim above rests on them.",
                "",
            ]
            lines += _table(others)
            lines.append("")
    else:
        lines += [
            "### Outcomes recorded in this directory, run unattributed",
            "",
            "`run_status.json` does not record an `artifacts_are_from.provenance_records` for "
            "this directory, so these outcomes cannot be joined to the run that wrote "
            "`interpretation.json`. They are what was recorded HERE, by some run.",
            "",
        ]
        lines += _table(others)
        lines.append("")
    lines += [
        "A guard that does not appear above is **not** thereby recorded as not having run: "
        "only calls made by a stage with an output directory bound leave an entry.",
        "",
    ]
    return lines


def _section_unknown(interp: dict[str, Any], provenance: Any,
                     guard_record: GuardRecord) -> list[str]:
    lines = [
        "## 10. WHAT THIS REPORT DOES NOT KNOW",
        "",
        "Mandatory and always present. A suppressed, unverified or unrecorded thing must be "
        "impossible to mistake for a verified one, and the only way to guarantee that is to name "
        "each one, and the field that would have said otherwise.",
        "",
    ]
    lines += _baseline_block(interp, provenance)
    lines += [
        "- Lexicon citation: this record cites its lexicon as `lexicon_id` = "
        f"{_s(_field(interp, 'lexicon_id', 'interpretation.json'))}, a declared string carried on "
        "the hit rows. It is **not** `compile.LexiconManifest.lexicon_content_hash`, which is what "
        "`FP-11` requires a family-level number to state, and no field in `interpretation.json` "
        "joins to a compile manifest. So this report cannot say which compiled lexicon these "
        "families came from.",
        "",
        "- `selection_rule` and `selection_feature_names` are fields of `schema.PeakSetQuery` and "
        "are **not** emitted on `interpret.Interpretation`. So the executable rule that "
        "`PROGRAMMATIC_RULE` requires, and the features that decide `SUBSTRATE_CIRCULAR` vs "
        "`INTERNAL_DECOMPOSITION`, are not readable from this artifact. They survive only inside "
        "the provenance `command` string, which is rendered verbatim above and parsed nowhere.",
        "",
    ]
    lines += _guard_outcomes_bullet(guard_record)
    lines += [
        "- Alignment denominators: `align.AlignmentRunSummary` (`n_nodes`, `n_pairs_considered`, "
        "`n_edges`, `n_pairs_excluded`) is returned from `align.run` and printed to stdout by "
        "`cli._run_align`, and written to no file. An alignment edge table therefore has no "
        "recorded denominator on disk, and this report would say so rather than count its rows.",
        "",
    ]
    return lines


def _section_guards() -> list[str]:
    lines = [
        "## 12. GUARDS AWAITING INPUT",
        "",
        "Rendered verbatim from `guards.GUARDS_AWAITING_INPUT`. Each entry is a guard with **no "
        "call site** in this release, with the artifact that comes nearest, why that artifact is "
        "not a call site, and what would close it.",
        "",
    ]
    for gid in sorted(guards.GUARDS_AWAITING_INPUT):
        pending = guards.GUARDS_AWAITING_INPUT[gid]
        lines += [
            f"### `{gid}`",
            "",
            f"- `nearest_artifact`: {pending.nearest_artifact}",
            f"- `why_not_a_call_site`: {pending.why_not_a_call_site}",
            f"- `closes_when`: {pending.closes_when}",
            "",
        ]
    # THE INVERSION THIS SUBSECTION EXISTS TO PREVENT. A guard's absence from
    # GUARDS_AWAITING_INPUT is not evidence that it is wired: the registry
    # records guards known to LACK a call site and says nothing about the rest.
    # A previous attempt read that absence as a call site and named one. Reading
    # a positive fact out of a silence is the error this whole module is built
    # against, and it must not be committed by the module itself.
    others = sorted(g for g in guards.ALL_GUARDS if g not in guards.GUARDS_AWAITING_INPUT)
    lines += [
        "### Not listed as awaiting input; this report does not know whether it has a call site",
        "",
        f"These {len(others)} guards are absent from `GUARDS_AWAITING_INPUT`. That absence is not "
        "a record that they are wired, and no artifact rendered above says whether any of them "
        "ran. They are listed by name and nothing is concluded from the listing.",
        "",
    ]
    lines += [f"- `{gid}`" for gid in others]
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
#: Body sections whose position comes from the record's own `emitted_order`
#: rather than from an order this renderer holds. A run that emitted composition
#: before health and one that did the reverse are different runs; a fixed order
#: here would render them identically.
_ORDERED_SECTIONS = ("health", "composition", "effects")


def render_markdown(interpretation: dict[str, Any], provenance: Any,
                    bias_ledger_rows: list[list[str]], bias_ledger_source: Path,
                    guard_record: GuardRecord | None = None) -> str:
    """Render one recorded interpretation to markdown. Computes nothing.

    ``guard_record`` defaults to the ABSENT state rather than to "assume the
    caller read one": a renderer handed no guard record must produce the page it
    produced before guard outcomes existed, not a page that quietly implies there
    were none to read.
    """
    if not isinstance(interpretation, dict):
        raise ReportError("interpretation.json is not a record")
    guard_record = guard_record if guard_record is not None else GuardRecord(GuardRecord.ABSENT)
    baseline = _baseline_block(interpretation, provenance)
    lines = [
        f"# Audit report -- `{_s(_field(interpretation, 'query_id', 'interpretation.json'))}`",
        "",
        "Every figure below is `str()` of a field recorded by the stage that produced it. This "
        "renderer performs no arithmetic: where a rate appears, its numerator and denominator are "
        "named beside it rather than divided. A field the artifacts do not carry is printed as "
        f"`{NOT_RECORDED}`, never as a value. Section order for health / composition / effects is "
        "the record's own `emitted_order`.",
        "",
    ]
    lines += _section_identity(interpretation)
    lines += _section_provenance(provenance)
    lines += _section_notes(interpretation)

    emitted_order = _field(interpretation, "emitted_order", "interpretation.json")
    if not isinstance(emitted_order, list):
        raise ReportError(
            "interpretation.json `emitted_order` is not a list; refusing to impose an order the "
            "record did not state."
        )
    rendered: set[str] = set()
    for name in emitted_order:
        if name in rendered:
            continue
        rendered.add(name)
        if name == "health":
            lines += _section_health(interpretation)
        elif name == "composition":
            lines += _section_composition(
                _field(interpretation, "composition", "interpretation.json"))
        elif name == "effects":
            lines += _section_effects(
                _field(interpretation, "effects", "interpretation.json"), baseline)
        else:
            # Named but unrenderable: said out loud rather than dropped, because a
            # dropped section is indistinguishable from one that was never emitted.
            lines += [
                f"## (recorded in `emitted_order`: `{_s(name)}`)",
                "",
                f"`emitted_order` names `{_s(name)}`, for which this renderer has no section. Its "
                "absence below is this renderer's limitation, not the run's.",
                "",
            ]
    # A block that WAS emitted but that `emitted_order` does not name still
    # renders. The order comes from the record; what exists comes from the
    # record's own fields, and the test is `is None` -- never floor_failures.
    if "health" not in rendered:
        lines += _section_health(interpretation)
    for name in _ORDERED_SECTIONS:
        if name in rendered or name == "health" or interpretation.get(name) is None:
            continue
        lines += [
            f"*(`{name}` is non-null in the record but is not named by `emitted_order`; rendered "
            "here so that an emitted block cannot go missing.)*",
            "",
        ]
        lines += (_section_composition(interpretation[name]) if name == "composition"
                  else _section_effects(interpretation[name], baseline))

    lines += _section_two_part(_field(interpretation, "two_part_effects", "interpretation.json"))
    lines += _section_permissions(interpretation)
    lines += _section_bias_ledger(bias_ledger_rows, bias_ledger_source)
    lines += _section_unknown(interpretation, provenance, guard_record)
    lines += _section_guard_outcomes(guard_record)
    lines += _section_guards()
    return "\n".join(lines).rstrip() + "\n"


def run(project: str | os.PathLike[str],
        out_dir: str | os.PathLike[str] = "report/",
        *,
        html: bool = False,
        docx: bool = False,
        figures: bool = False,
        bias_ledger: str | os.PathLike[str] | None = None,
        ) -> Path:
    """Render ``<project>/interpretation.json`` and ``provenance.json`` to markdown.

    Returns the path written.

    ``html`` and ``docx`` are refused rather than served markdown under another
    name, on the precedent of ``cli._run_validate`` refusing ``--fimo-heldout``
    as "a semantic no-op": rendering one format while the caller asked for
    another is the specified-versus-ran gap this package exists to close.

    ``figures`` is refused for a different reason, and that refusal is this
    directory's stated check -- "the renderer refuses a figure with no
    denominator and no ``baseline_population`` field". No artifact in this
    package records ``baseline_population``, so the refusal is the rule working
    rather than a gap in it.

    ``bias_ledger`` defaults to :func:`packaged_bias_ledger_path`, the ledger
    shipped inside this package, so the default resolves wherever the package is
    installed. Passing one explicitly overrides it and is checked exactly as
    strictly.
    """
    for flag, enabled in (("--html", html), ("--docx", docx)):
        if enabled:
            raise ReportError(
                f"{flag} is not yet a renderer output; refusing a semantic no-op. This release "
                "renders markdown only, and handing markdown to a caller who asked for another "
                "format is the specified-versus-ran gap this package exists to close."
            )
    project_dir = Path(project)
    interp_path = project_dir / "interpretation.json"
    prov_path = project_dir / "provenance.json"
    if not interp_path.is_file():
        raise ReportError(
            f"{str(interp_path)!r} is absent. `report` renders a recorded interpretation and does "
            "not reconstruct one: `effect_estimates.tsv` carries no health block, no notes and no "
            "suppression reason, so it is not independently renderable."
        )
    if not prov_path.is_file():
        raise ReportError(
            f"{str(prov_path)!r} is absent. Every number rendered carries its provenance; a report "
            "without the append-log would present figures whose inputs are unnamed."
        )
    interpretation = _load_json(interp_path, "interpretation")
    provenance = _load_json(prov_path, "provenance")

    if figures and not (_find_anywhere(interpretation, "baseline_population")
                        + _find_anywhere(provenance, "baseline_population")):
        raise ReportError(
            "refusing to render a figure: no artifact here carries a `baseline_population` field. "
            "`comparator_id` names which FILE the baseline came from, not what KIND of population "
            "it is -- and the same data, read against an unselected universe and against a "
            "residual subset, supported both 'replicates exactly' and '4x stronger, prediction "
            "falsified'."
        )

    ledger_source = (Path(bias_ledger) if bias_ledger is not None
                     else packaged_bias_ledger_path())
    rows = read_bias_ledger(ledger_source)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "report.md"
    target.write_text(
        render_markdown(interpretation, provenance, rows, ledger_source,
                        read_guard_record(project_dir)),
        encoding="utf-8")
    return target
