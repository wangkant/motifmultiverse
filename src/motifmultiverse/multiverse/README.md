# `multiverse` — run the declared grid, and refuse to average across questions

The package was named for this stage and, until this release, did not have one.
Every run answered a single specification, and a reader who wanted to know whether
a conclusion survived a different defensible choice ran the tool again and
compared two directories by hand. Nothing recorded that the second run happened,
so nothing recorded when it disagreed.

## What makes a grid worth reading

Width is the easy part and the least of it. Three properties do the work, and each
is enforced rather than described:

**The cells are declared before they are run.** `specification_manifest.json` is
written to the output directory before the first cell executes. A manifest
produced afterwards can only contain the cells that survived, and "were the
reported cells the planned cells?" is the first question a reader of a multiverse
needs answered.

**No cell disappears.** Every planned cell appears in `cells.tsv` exactly once
with one of `SUCCESS`, `REFUSED_GUARD`, `REFUSED_SCHEMA`, `NOT_ESTIMABLE` or
`ERROR`, and the non-`SUCCESS` ones are repeated in `dropped_cells.tsv` with the
reason. A cell that could not be estimated is a finding about the design — most
often that a baseline is too small for the block bootstrap — and a grid that
reports only the cells that worked has selected on its outcome. Even an unexpected
exception becomes a recorded cell rather than ending the run, because a grid that
stops at its first surprise reports a non-random subset of itself.

**Stability is summarised within one estimand, never across.** This is the whole
scientific claim of the module.

## Why the axes are three types

| type | varies | changing it changes |
|---|---|---|
| `Estimand` | query, **baseline population**, its type and construction rule, selection rule and features | **what is being estimated** |
| `Measurement` | lexicon id, lexicon content hash, substrate id, hit table | how that question is **measured** |
| `StatisticalChoice` | estimator, block size, replicates, seed, floors | how **uncertainty** is computed |

Represented as one flat bag of options, "average over the options" is the natural
thing to write, and it produces a generic robustness score that conceals the only
finding worth having: *which* choice the conclusion was sensitive to. An effect
against baseline A and an effect against baseline B are answers to two questions.
Averaged, they become a number that is an answer to neither.

So the estimand is the key of every summary, and
`guards.no_cross_estimand_pooling` refuses one that spans two. It takes the
summaries **and** the manifest, and resolves each cell's estimand from the
manifest — written earlier, by the code that enumerated the grid, not by the code
that grouped the results. A summariser that groups wrongly cannot also certify
that it grouped rightly.

## What this module does not do

It computes no statistics. `interpret.interpret_query` estimates every effect; this
module enumerates, binds, calls, and records. A second implementation of a block
bootstrap that agrees with the first 99% of the time is worse than none, because
the 1% arrives as a number nobody can trace. `test_the_module_computes_no_statistics_of_its_own`
walks this module's AST and fails on a function whose name suggests otherwise.

It invents no threshold. Nothing here is classified "robust". The summary reports
the number of cells that produced an estimate against the number planned, the
signs observed, and the range. Where a design preregistered a threshold it is
recorded and named; where none was — the normal case — the report says so and
stays descriptive. A threshold chosen after the effects are visible is the
specification search this module exists to make visible.

It fills nothing with zero. A family not estimable in a cell is
`NOT_ESTIMABLE`, absent from that cell's counts and never present as a measured
zero.

## Identity: what "one frozen dataset" means

`substrate_id` is the SHA-256 of a frozen hit caller's output; `lexicon_id` names
the vocabulary it used. These are not independent — two lexicons over the same
peaks are two hit-calling runs, so two substrates. A cell therefore binds exactly
one `(substrate_id, lexicon_content_hash)` pair and the grid spans several: what
is held fixed is the peak universe and the frozen run family, and what varies on
the measurement axis is which frozen table is read. Mixing substrates *inside* a
cell is refused — by `interpret.read_hit_table`, before this module compares
anything, which is why there is no second check for it here.

Declaring the ids in the design is what makes them checkable at all. A run that
reads whatever table it is pointed at and reports the id it finds cannot notice
being pointed at the wrong table.

## Usage

```
motifmultiverse multiverse design.json --out multiverse/
```

Paths inside the design are resolved relative to the design file. See
`docs/MULTIVERSE_DESIGN.md` for the design's shape and
`docs/K562_MULTIVERSE_AUDIT.md` for a real 36-cell grid over frozen K562
artifacts, including what it found to be baseline-sensitive.

## Outputs

| file | what it is |
|---|---|
| `specification_manifest.json` | every planned cell with its deterministic ids, written first |
| `cells.tsv` | one row per planned cell, with status |
| `family_effects.tsv` | family-level effects for successful cells, each carrying its full identity |
| `dropped_cells.tsv` | every non-`SUCCESS` cell and why |
| `stability_by_estimand.tsv` | descriptive stability within each fixed estimand |
| `specification_curve.md` | the document a person reads |
| `guard_outcomes.json` | what each guard returned, as everywhere else here |
