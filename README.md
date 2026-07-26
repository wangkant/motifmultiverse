# motifmultiverse

**Bias-aware harmonization and robust inference of attribution-derived regulatory
motifs across models and methods.**

![status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange)
![version: 0.1.0.dev0](https://img.shields.io/badge/version-0.1.0.dev0-blue)
![API: unstable](https://img.shields.io/badge/API-unstable-red)

> **This is mostly a skeleton.** Three of the nine analysis modules — `ingest`,
> `compile` and `interpret` — are implemented; the other six raise
> `NotImplementedError`. What else runs today is the schema, the guards, the
> provenance recorder, the CLI and the test suite. See
> [Current state](#current-state) before investing time.

## The problem

The ecosystem already solves the neighbouring steps: **TF-MoDISco** discovers
motifs from attributions, **FIMO** scans sequence, **FiNeMo** calls instances,
**HOMER** / **TomTom** annotate and compare.

What is still project-local is the step between them — *how do you compile several
local TF-MoDISco outputs into one shared lexicon that is safe to compare across
models?* That is normally a pile of per-project scripts plus human judgement, and
it is where the decisions that matter get made without a record.

The missing piece is not another database. It is an **attribution lexicon
compiler**: something that turns "should these merge", "what is this called",
"does it enter the main lexicon" and "does this conclusion survive" into auditable
decisions that carry their evidence, their uncertainty and their downstream
validation.

## Three design principles

- **Representation and identity are separate.** A CWM belongs to one model and
  readout; `family_id` / `variant_id` are the ontology that crosses them.
- **Merges are validated downstream, not by similarity.** Two motifs are the same
  when collapsing them does not degrade reconstruction.
- **The output is stability and uncertainty**, not one absolute lexicon.

## Four things the reference implementation learned

This tool exists because of these. All four are from that line, not from this
repository; figures and sources are in [`docs/CONCEPT.md`](docs/CONCEPT.md).

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

## The one conclusion worth carrying to other projects

> ### A frozen lexicon transfers; a comparator does not.
>
> The same measurements supported both *"replicates exactly"* and *"four times
> stronger, prediction falsified"* — differing only in whether the baseline was the
> unselected universe or a residual subset from which the relevant peaks had
> already been removed. Nothing about the data changed.
>
> **Any cross-condition motif claim that does not state its baseline population is
> uninterpretable.**

## Current state

Nine modules, six implemented. Each module directory carries a README stating its
rule, the failure that produced the rule, and how the rule is checked.

**What actually runs today:**

```bash
pip install -e ".[dev]"

motifmultiverse --help              # 9 subcommands with real arguments

# a minimal path from discovery output through adjudication to a frozen lexicon:
motifmultiverse ingest project.yaml --out registry/
motifmultiverse align registry/ --out evidence/
motifmultiverse annotate evidence/ --registry registry/ --tomtom --out evidence/
motifmultiverse adjudicate evidence/ --registry registry/ --out adjudication/
motifmultiverse compile registry/ \
    --decisions adjudication/merge_decisions.json --out lexicons/

# and a query over a frozen hit table:
motifmultiverse interpret hits.tsv \
    --peaks island_5.txt --comparator gc_matched.txt --comparator-id gc_matched \
    --selection-provenance CLUSTERED_WITH_SPLIT --held-out heldout.txt

ruff check src tests
pytest
```

**That one skip matters.** The `compile` round-trip test calls the real hit-caller
loader and skips wherever the `finemo` backend is absent — including in CI. *A
skipped test is unverified, not verified*, and this is the same shape as a guard that
can never fail, so it is written here instead of being left to a green check mark. It
has been verified manually against a real TF-MoDISco output; see
[`compile/README.md`](src/motifmultiverse/compile/README.md) for the evidence and for
what exactly was and was not exercised.

| exit | meaning |
|---|---|
| `0` | success |
| `2` | usage error, or a named input does not exist |
| `3` | the module is a skeleton; the message names its README |
| `4` | **refusal** — the tool declined to produce a number, and the message says which rule declined it |

`4` is part of the behaviour contract, not an implementation detail. A refusal is a
designed outcome: an undeclared peak set, a missing baseline, a health floor that did
not clear. It is deliberately distinct from `3`, which means nobody has written the
code yet.

**`ingest`** reads the discovery HDF5s a project declares into one registry, with a
checksum per input. A metacluster group that contributes no patterns is recorded as
one of `group_absent` / `group_empty` / `not_searched` — three different claims, and
none of them is "no motifs".

**`compile`** writes one hit-caller-compatible lexicon per tier, **in the order the
loader emits**, each with a content hash and an explicit statement of what the tier
contrast does and does not vary. Round-trip verification calls the real loader rather
than re-reading our own assumptions; without that backend installed it says so
instead of claiming success.

Running any subcommand writes a provenance record — input checksums, command line,
software versions, seed, timestamp — **before** the body runs, including before it
raises `NotImplementedError`. Provenance is the most expensive thing to add
retroactively, so it is here from the first commit.

**`interpret`** answers subset queries over one frozen hit table. It was
implemented first because it is the only module that needs neither TF-MoDISco nor
a hit-caller backend, so it runs with nothing else installed and fixes the
interface for the rest. What it does, in order: resolve what the query is
*allowed* to emit from its declared selection provenance; compute three health
numbers (intersection coverage, blocks spanned, fraction the frozen lexicon
explains); then emit at that strength — full inference, held-out inference, or a
descriptive decomposition with no interval and no *p* value. **If a health number
is below its pre-registered floor the reading is suppressed, not annotated.** A
caveat next to an effect size does not travel; the effect size does.

Undeclared provenance is a recorded state (`DECLARATION_MISSING`) that costs the
query its inference. It never resolves to the permissive grade.

**What does not run:** validate, infer and report. The implemented middle path
persists alignment evidence, retains competing annotation candidates, and records
collapse/refusal/deferred adjudications. Undefined scientific thresholds remain
explicitly deferred, so a run may intentionally compile an undeduplicated lexicon
rather than guess a merge.

**Deliberately out of scope for v1:** computing attributions, discovering motifs,
re-implementing a hit caller, cross-model raw CWM averaging (a design prohibition,
not a gap), and mapping a motif to a specific protein (the default output is
family-level identity with a confidence).

## Guards

Fifteen executable constraints in `src/motifmultiverse/guards/`. Each has a positive
test **and a falsification test that must make it fail** — the direct answer to
finding 4 above. A meta-test walks the guard registry and fails if any guard ships
without one.

`single_scale` · `variant_id_unique` · `no_key_parsing` · `four_state_missingness` ·
`no_cross_model_cwm_avg` · `sign_alignment` · `interaction_required` ·
`estimability_floor` · `stratum_parity` · `short_motif_flag` · `single_family_layer` ·
`selection_provenance_declared` · `health_before_effect` · `comparator_declared` ·
`index_order_matches_loader`

[`docs/CONSTRAINTS.md`](docs/CONSTRAINTS.md) lists the 25 frozen design principles,
each labelled `ENFORCED` / `PARTIAL` / `DOC_ONLY` with the current tally — which is
**4 / 13 / 8**, and deliberately unflattering. It is a roadmap metric: the list comes
from the design, not from the code, so most of it is unenforced until the modules
that would enforce it exist. Every `DOC_ONLY` entry carries a drafted criterion — the
check is unimplemented, but what it would check is written down. Labelling a
prose-only rule as enforced is the same class of error the guards exist to prevent.

## Repository layout

| path | what it is |
|---|---|
| [`docs/CONCEPT.md`](docs/CONCEPT.md) | what the tool is for and why, self-contained |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | seven layers, nine modules, two architecture-level constraints |
| [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) | node / edge schema and the four rules from real failures |
| [`docs/BIAS_LEDGER.md`](docs/BIAS_LEDGER.md) | 20 bias axes `BA-01`…`BA-20`, mechanism and control (+ `bias_ledger.tsv`) |
| [`docs/CONSTRAINTS.md`](docs/CONSTRAINTS.md) | 25 principles `FP-01`…`FP-25`, honest enforcement status, criterion drafts |
| [`docs/LESSONS.md`](docs/LESSONS.md) | every architecture constraint, indexed back to the failure that produced it |
| [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md) | stop-condition handoff protocol — the most portable file here |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | M1–M5, defined by completion criteria not dates |
| `src/motifmultiverse/` | package: schema, guards, provenance, CLI, nine module skeletons |
| `config/` | example project, specification and database configs |
| `tests/` | schema, guard (incl. falsification), CLI and provenance tests |

## Reproducibility, honestly

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

## Licence

MIT (`LICENSE`).

There is deliberately no `CITATION.cff` yet. A citation file with placeholder
authors renders on GitHub as a claim that the project is ready to be cited, which
is a stronger statement than this repository can support. It goes in at
publication, with real names.
