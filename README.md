# motifmultiverse

**Bias-aware harmonization and robust inference of attribution-derived regulatory
motifs across models and methods.**

![status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange)
![version: 0.1.0.dev0](https://img.shields.io/badge/version-0.1.0.dev0-blue)
![API: unstable](https://img.shields.io/badge/API-unstable-red)

> **Pre-alpha.** What this is today, stated as precisely as the code supports: an
> **auditable motif lexicon compiler**, and a **single-specification inference
> reference implementation**. All nine analysis modules are implemented — `report`
> was the last skeleton — and no subcommand raises `NotImplementedError` any more.
> Implemented is not complete, and two gaps matter before the name misleads you:
> the specification **multiverse** does not exist — `infer` estimates ONE
> specification and says so on every run — and the adjudication criteria this
> package ships leave `TRUE_DUPLICATE` and `FRAGMENT_MATCH` undefined, so the
> default pipeline defers every duplicate and compiles an **undeduplicated**
> lexicon by design. The collapse path is implemented and tested; what is missing
> is a validated merge policy, which is a scientific decision nobody has frozen.
> `report` renders markdown only and prints `NOT RECORDED` for the fields no
> artifact carries. The API is unstable and the per-module table in
> [Implementation status](#implementation-status) is generated from the CLI
> dispatch table rather than maintained by hand, because it was wrong twice before
> it was generated. Read that section before investing time.

## Contents

[Overview](#overview) · [Installation](#installation) · [Usage](#usage) ·
[Design and its empirical basis](#design-and-its-empirical-basis) ·
[Implementation status](#implementation-status) ·
[Executable constraints](#executable-constraints) ·
[Scope and limitations](#scope-and-limitations) ·
[Repository layout](#repository-layout) · [Citation](#citation) ·
[Licence](#licence)

## Overview

The ecosystem already solves the neighbouring steps: **TF-MoDISco** discovers
motifs from attributions, **FIMO** scans sequence, **FiNeMo** calls instances,
**HOMER** and **TomTom** annotate and compare.

What is still project-local is the step between them — *how do you compile several
local TF-MoDISco outputs into one shared lexicon that is safe to compare across
models?* That is normally a pile of per-project scripts plus human judgement, and
it is where the decisions that matter get made without a record.

The missing piece is not another database. It is an **attribution lexicon
compiler**: something that turns "should these merge", "what is this called",
"does it enter the main lexicon" and "does this conclusion survive" into auditable
decisions that carry their evidence, their uncertainty and their downstream
validation. Concretely, the package normalises heterogeneous discovery output into
one registry, builds an alignment evidence graph over it, adjudicates merges,
compiles tiered content-addressed lexicons, validates those merges downstream, and
answers subset queries over a frozen hit table at the strength the query's
provenance licenses.

It is **not** another motif clustering algorithm, **not** a motif database, and
**not** an enrichment tool. See [`docs/CONCEPT.md`](docs/CONCEPT.md) for the
self-contained statement of purpose and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for the seven-layer, nine-module decomposition.

## Installation

Requires Python >= 3.11.

```bash
pip install -e ".[dev,finemo]"
```

The `finemo` extra is what makes the `compile` round trip **run** instead of skip:
round-trip verification calls the real hit-caller loader rather than re-reading
this package's own assumptions about the format. The extra is optional by design —
a missing backend must produce a clear "backend not installed" message, never an
`ImportError` traceback — but a run without it verifies less.

To make that absence loud rather than quiet, set
`MOTIFMULTIVERSE_REQUIRE_FINEMO=1`: a missing backend then fails the run instead of
shrinking it. The round trip used to skip wherever the backend was absent —
including in CI, where the extra named a distribution that does not exist on PyPI,
so the documented way to stop it skipping could not be run. *A skipped test is
unverified, not verified*, which is the same shape as a guard that can never fail.
See [`compile/README.md`](src/motifmultiverse/compile/README.md) for what the round
trip proves on a real TF-MoDISco lexicon, and for the incompatibility the skip was
hiding.

Development checks:

```bash
ruff check src tests
pytest
```

## Usage

```bash
motifmultiverse --help
```

A minimal path from discovery output through adjudication to a frozen lexicon, and
from a frozen hit table to an audit report:

```bash
# 1. normalise the TF-MoDISco outputs a project declares into one registry
motifmultiverse ingest project.yaml --out registry/

# 2. pairwise registration and similarity, with a persisted per-pair null
motifmultiverse align registry/ --out evidence/

# 3. retain competing database-label candidates for later adjudication
motifmultiverse annotate evidence/ --registry registry/ --tomtom \
    --databases config/db.yaml --out evidence/

# 4. decide merges, refusals and deferrals; emit a human review file
motifmultiverse adjudicate evidence/ --registry registry/ --out adjudication/

# 5. compile tiered, content-addressed lexicons
motifmultiverse compile registry/ \
    --decisions adjudication/merge_decisions.json --out lexicons/

# 6. check that a merge survives downstream
motifmultiverse validate lexicons/ --before-hits before.parquet --after-hits after.parquet \
    --substrate-manifest substrate.manifest.json \
    --split-manifest split-manifest.json --decision-artifact decision-split.json \
    --validation-artifact validation-split.json --out validation/

# 7. answer a subset query over one frozen hit table
motifmultiverse interpret hits.tsv --out interpretation/ \
    --peaks island_5.txt --comparator gc_matched.txt --comparator-id gc_matched \
    --selection-provenance CLUSTERED_WITH_SPLIT --held-out heldout.txt

# 8. render that interpretation and its provenance log as one markdown audit report
motifmultiverse report interpretation/ --out report/
```

Notes on individual steps:

- **`align`** is quadratic in the registry and re-registers the full pipeline 1000
  times per pair (`--null-shuffles`). `--workers N` spreads the pairs over N
  processes; each pair's null is drawn from the run seed alone, so the written
  tables are byte-identical at every worker count and only wall-clock changes.
  Progress goes to stderr, not stdout.
- **`annotate --tomtom`** reads *precomputed* TomTom output named by `--databases`;
  it does not run TomTom. `config/db.yaml` is site-specific and not shipped — copy
  the shape from `config/db.example.yaml`. Without it the backend is logged
  `UNVERIFIED`, not skipped.
- **`validate`** requires frozen, standardized before/after hit tables and the exact
  manifest-bound decision/validation split artifacts; see `validate --help`.
- **`report`** renders markdown only. `--html` and `--docx` refuse (exit 4) rather
  than emitting markdown under another name.

### Exit codes

| exit | meaning |
|---|---|
| `0` | success |
| `2` | usage error, or a named input does not exist |
| `3` | the module is a skeleton; the message names its README — **no module in this release exits `3`**, and the code path is kept (and tested) against a future one |
| `4` | **refusal** — the tool declined to produce a number, and the message says which rule declined it |

`4` is part of the behaviour contract, not an implementation detail. A refusal is a
designed outcome: an undeclared peak set, a missing baseline, a health floor that
did not clear. It is deliberately distinct from `3`, which means nobody has written
the code yet.

### The exit code is also written into the output directory

By the time anyone opens a results folder the exit code is gone, so every run that
names an `--out` leaves a **`run_status.json`** there: `SUCCESS` / `REFUSED` /
`UNIMPLEMENTED` / `INPUT_MISSING` / `CRASHED`, the exit code, and — for a refusal —
the sentence that refused. Its `artifacts_are_from` field names the last run that
actually succeeded in that directory, so a refusal that ran into a directory holding
an earlier result **labels** that result instead of deleting it: destroying a real
result to prevent a misreading of it is the worse trade. Provenance is still written
*before* the body runs, because a record that arrives only on success is a record the
runs you most want to explain never get; this file is the other half, written when the
outcome is known.

**A downstream reader's rule is one line: trust the artifacts only when
`status == "SUCCESS"`.**

### What individual modules guarantee

Running any subcommand writes a provenance record — input checksums, command line,
software versions, seed, timestamp — **before** the body runs, including before it
raises `NotImplementedError`. Provenance is the most expensive thing to add
retroactively, so it is here from the first commit.

**`ingest`** reads the discovery HDF5s a project declares into one registry, with a
checksum per input. A metacluster group that contributes no patterns is recorded as
one of `group_absent` / `group_empty` / `not_searched` — three different claims, and
none of them is "no motifs".

**`compile`** writes one hit-caller-compatible lexicon per tier, **in the order the
loader emits**, each with a content hash and an explicit statement of what the tier
contrast does and does not vary. Round-trip verification calls the real loader; without
that backend installed it says so instead of claiming success.

**`interpret`** answers subset queries over one frozen hit table. It was implemented
first because it is the only module that needs neither TF-MoDISco nor a hit-caller
backend, so it runs with nothing else installed and fixes the interface for the rest.
What it does, in order: resolve what the query is *allowed* to emit from its declared
selection provenance; compute three health numbers (intersection coverage, blocks
spanned, fraction the frozen lexicon explains); then emit at that strength — full
inference, held-out inference, or a descriptive decomposition with no interval and no
*p* value. **If a health number is below its pre-registered floor the reading is
suppressed, not annotated.** A caveat next to an effect size does not travel; the
effect size does. Undeclared provenance is a recorded state (`DECLARATION_MISSING`)
that costs the query its inference, and never resolves to the permissive grade.

**`report`** renders one markdown audit report from what a stage recorded — an
`interpretation.json` and the `provenance.json` log beside it — plus the
`bias_ledger.tsv` that ships **inside the package**, so the default resolves from an
installed wheel and not only from a checkout. Every number printed is the recorded field itself, beside the
denominator the stage recorded and *named* (`n_submitted = 8277`); nothing is
recomputed here, because a renderer that recomputes is a second implementation of the
statistics that can disagree with the first. Nothing absent is defaulted: a withheld
*p* value prints as `WITHHELD — inference_capability = ESTIMATION_ONLY` rather than
blank, and a field no artifact carries — `baseline_population`, the lexicon *content*
hash, `selection_rule`, every guard outcome — prints as `NOT RECORDED` in a mandatory
*What this report does not know* section that names the field that would have said it.
This is the module whose founding failure was a bootstrap resolution floor printed as
though it were a measured *p* value.

## Design and its empirical basis

### Three design principles

- **Representation and identity are separate.** A CWM belongs to one model and
  readout; `family_id` / `variant_id` are the ontology that crosses them.
- **Merges are validated downstream, not by similarity.** Two motifs are the same
  when collapsing them does not degrade reconstruction.
- **The output is stability and uncertainty**, not one absolute lexicon.

### Four findings from the reference implementation

This tool exists because of these four results. All are from the reference
implementation, not from this repository; figures and sources are in
[`docs/CONCEPT.md`](docs/CONCEPT.md).

**1. The hit caller is not input-scale invariant.** The same regions produce
different *discrete* retention decisions depending on which other regions share
the input. The onset was bracketed to **(6,460, 7,085] regions** on a 6,460-region
base — a **9.67%** increase in scale suffices. Consequence: every specification
must be a subset of **one frozen run**. That is an architecture constraint, not an
implementation detail.

**2. Numeric and discrete divergence are independent axes.** Permuting input order
produced the largest coefficient displacement measured (**2.07×** the median
coefficient) and **zero** discrete flips. Measure and gate them separately; never
let one license the other.

**3. Writing a rule down is not executing it.** A four-state missingness encoding
was specified and still silently destroyed by a table pivot returning `0.0` for an
all-undefined group — and the coverage figure, computed after that fill, reported
`1.000000`, so the error corroborated itself. Constraints must be executable
assertions.

**4. A guard must be proven capable of failing.** Of five framework guards that all
reported passing, **2 still passed** under a row-shifted *and* a permuted lexicon
index, and **none of the five** detected a reordered index. A guard that has never
failed is not evidence.

### The conclusion most worth carrying to other projects

> ### A frozen lexicon transfers; a comparator does not.
>
> The same measurements supported both *"replicates exactly"* and *"four times
> stronger, prediction falsified"* — differing only in whether the baseline was the
> unselected universe or a residual subset from which the relevant peaks had
> already been removed. Nothing about the data changed.
>
> **Any cross-condition motif claim that does not state its baseline population is
> uninterpretable.**

## Implementation status

Each module directory carries a README stating its rule, the failure that produced
the rule, and how the rule is checked. The table below is generated — a module counts
as implemented when its CLI subcommand dispatches to a real runner, which is a fact
about the code rather than an opinion about it:

<!-- BEGIN GENERATED STATUS -->
<!-- generated by `python -m motifmultiverse.status --render-readme README.md`; schema 1. Do not edit by hand. -->

| module | status |
|---|---|
| `ingest` | IMPLEMENTED |
| `align` | IMPLEMENTED |
| `annotate` | IMPLEMENTED |
| `adjudicate` | IMPLEMENTED |
| `compile` | IMPLEMENTED |
| `validate` | IMPLEMENTED |
| `infer` | IMPLEMENTED |
| `report` | IMPLEMENTED |
| `interpret` | IMPLEMENTED |

Optional-backend verification and the three test counts (passed / skipped /
failed, never summed) are per-run facts, not repository facts: see
`implementation_status.json`, which CI regenerates and uploads on every run.
<!-- END GENERATED STATUS -->

### What CI verifies, and what a `VERIFIED` backend means

Three jobs, because "the tests passed" is three different claims and it used to be
one. **`core`** runs lint and the suite on Python 3.11 *and* 3.12 — the classifiers
claimed 3.12 and nothing ran it. **`roundtrip`** installs `.[dev,finemo]` and runs
the suite under `MOTIFMULTIVERSE_REQUIRE_FINEMO=1`, so the one check that proves what
`compile` exists to guarantee fails when the backend is missing instead of skipping;
it had skipped in every CI run this repository has ever had. **`wheel`** builds the
distribution, installs it into a clean environment, and runs the CLI from a directory
that is not a checkout — which is how a `report` default that resolved only from a
source tree stayed invisible.

In `implementation_status.json`, a backend is `VERIFIED` only when a **capability
probe** ran on that machine: `compile.probe_backend()` compiles a one-motif lexicon
and reads it back with the real loader, comparing the order with the same guard
`verify_roundtrip` uses. Importable is not usable — finemo 0.40 renamed an argument,
and every installation of it imported cleanly and could not read a lexicon back. An
installed-but-incapable backend is `UNVERIFIED`, with the reason.

### What does not run

The specification *multiverse* inside `infer`: `infer` estimates ONE specification
with the `FP-15` estimators and says so; sweeping specification axes and reporting
the dropped cells with reasons is still M4. `report` renders markdown only, and
`--html` and `--docx` refuse.

The implemented middle path persists alignment evidence, retains competing annotation
candidates, records collapse/refusal/deferred adjudications, and validates a merge on
the affected subset of frozen standardized hit tables. An all-peak delta remains only
a dilution diagnostic: fewer than 30 affected peaks are recorded as
`LOW_RISK_RARE_NOT_VALIDATED`, without an interval or equivalence claim.

Undefined scientific thresholds remain explicitly deferred, and this is the second
thing to know before the name misleads you: the criteria this package **ships**
(`adjudicate/criteria.v1.yaml`) leave `TRUE_DUPLICATE` and `FRAGMENT_MATCH` at
`CRITERION_NOT_YET_DEFINED`, so the **default** run defers every duplicate and every
fragment and compiles an undeduplicated lexicon rather than guess a merge. The
collapse path is implemented and exercised end to end, but against thresholds the
caller supplies. That makes this release a strict **adjudication framework**, not a
harmonizer with a validated merge policy — an accurate description of where the
science is, not a gap in the code.

[`docs/ROADMAP.md`](docs/ROADMAP.md) states milestones M0–M5 by completion criteria
rather than dates.

## Executable constraints

### Guards

Fifteen executable constraints live in `src/motifmultiverse/guards/`. Each has a
positive test **and a falsification test that must make it fail** — the direct answer
to finding 4 above. A meta-test walks the guard registry and fails if any guard ships
without one.

`single_scale` · `variant_id_unique` · `no_key_parsing` · `four_state_missingness`† ·
`no_cross_model_cwm_avg`† · `sign_alignment` · `interaction_required`† ·
`estimability_floor`† · `stratum_parity`† · `short_motif_flag` · `single_family_layer`† ·
`selection_provenance_declared` · `health_before_effect` · `comparator_declared` ·
`index_order_matches_loader`

† **No call site in this release.** Six of the fifteen are defined and
falsification-tested but have never been put in front of an artifact, because this
release emits nothing that carries the thing they check. Counting them as protection
is the same error the guards exist to prevent, so they are marked here rather than
left for a reader to discover by grepping. `guards.GUARDS_AWAITING_INPUT` records for
each one the artifact that comes closest, what would go wrong if the guard were
pointed at it anyway — self-corroboration, vacuity, or overriding a decision nobody
has made — and what would have to exist. A test accompanies each entry and **fails
when that thing appears**, so the list shortens by wiring a guard, never by
relabelling one.

### Frozen design principles

[`docs/CONSTRAINTS.md`](docs/CONSTRAINTS.md) lists the 25 frozen design principles
`FP-01`…`FP-25`, each labelled `ENFORCED` / `PARTIAL` / `DOC_ONLY`. The current tally
is **2 / 15 / 8**, and it is deliberately unflattering. It is a roadmap metric: the
list comes from the design, not from the code, so most of it is unenforced until the
modules that would enforce it exist. Every `DOC_ONLY` entry carries a drafted
criterion — the check is unimplemented, but what it would check is written down.
Labelling a prose-only rule as enforced is the same class of error the guards exist to
prevent.

## Scope and limitations

**Deliberately out of scope for v1:** computing attributions, discovering motifs,
re-implementing a hit caller, cross-model raw CWM averaging (a design prohibition,
not a gap), and mapping a motif to a specific protein — the default output is
family-level identity with a confidence.

**This repository contains no analysis data and no analysis code from the reference
implementation.** That is deliberate:

- Those scripts hard-code absolute paths (39 of 41) and assume naming conventions
  that **mis-join silently rather than erroring** when a tool version changes.
  Porting them would import both defects.
- The reference line's own archive is a separate artifact with a separate purpose.

The reference implementation appears here in exactly two forms: **regression
fixtures** (`tests/fixtures/`, currently download recipes rather than data — see
that README for licensing) and **case citations** in the module READMEs explaining
which failure produced which rule.

You cannot clone this and reproduce that analysis. You can read what it learned and
run the checks.

## Repository layout

| path | what it is |
|---|---|
| [`docs/CONCEPT.md`](docs/CONCEPT.md) | what the tool is for and why, self-contained |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | seven layers, nine modules, two architecture-level constraints |
| [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) | node / edge schema and the four rules from real failures |
| [`docs/BIAS_LEDGER.md`](docs/BIAS_LEDGER.md) | 20 bias axes `BA-01`…`BA-20`, mechanism and control; the authoritative TSV `report` renders ships in the package, at `src/motifmultiverse/report/bias_ledger.tsv` |
| [`docs/CONSTRAINTS.md`](docs/CONSTRAINTS.md) | 25 principles `FP-01`…`FP-25`, honest enforcement status, criterion drafts |
| [`docs/LESSONS.md`](docs/LESSONS.md) | every architecture constraint, indexed back to the failure that produced it |
| [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md) | stop-condition handoff protocol — the most portable file here |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | M0–M5, defined by completion criteria not dates |
| `src/motifmultiverse/` | package: schema, guards, provenance, CLI, nine analysis modules |
| `config/` | example project, specification and database configs |
| `tests/` | schema, guard (incl. falsification), CLI and provenance tests |

## Citation

[`CITATION.cff`](CITATION.cff) carries the author and ORCID. It deliberately carries
**no DOI**: a placeholder resolves to nothing while rendering on GitHub as a citable
record, which is a stronger claim than an unpublished pre-alpha can support. When a
preprint exists its DOI goes in under `preferred-citation` so citations reach the
paper; a Zenodo DOI for the software itself can only be minted once this repository
is public, so it comes after that rather than before.

Two tests keep the file honest — `CITATION.cff`'s version must equal
`motifmultiverse.__version__`, and its `license` must match `LICENSE` — because a
citation record naming the wrong version cites something else, and every other
hand-maintained claim in this repository has drifted at least once.

## Licence

MIT. See [`LICENSE`](LICENSE).
