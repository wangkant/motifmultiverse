# The specification multiverse

The package was named for an analysis it did not perform. Every release until now
ran **one** specification per inference run — one query, one comparator, one
lexicon, one estimator — and a reader wanting to know whether a conclusion
survived a different defensible choice had to run the tool again and compare two
directories by hand. This document is the design for running the declared grid
instead, and for the constraint that makes such a grid worth running rather than
merely wide.

## 1. What a "single frozen dataset" is here

One peak universe (the K562 33,917-peak ALLPEAKS universe) and the frozen
hit-calling runs over it. This matters because of an identity question that has to
be answered before any grid can be drawn.

`substrate_id` is the SHA-256 of a frozen hit caller's own output. `lexicon_id`
names the motif vocabulary that run used. In the real archive these are not
independent: `CBP-2114__core_final` and `CBP-2114__expanded` are two FiNeMo runs
over the *same peaks* with *different lexicons*, so they carry two `substrate_id`s.

So "vary the lexicon" and "hold the substrate fixed" cannot both be literal, and
the design has to say which it means:

> **A cell binds exactly one `(substrate_id, lexicon_content_hash)` pair, and the
> multiverse spans several.** What is held fixed across the grid is the *peak
> universe* and the *frozen discovery run family*; what varies on the measurement
> axis is which frozen hit table is read. Mixing substrates **inside** a cell is
> refused, which is the failure the refusal is actually for: an effect computed by
> pooling rows from two hit callers is not an effect.

The alternative reading — one substrate, therefore one lexicon, therefore no
measurement axis — would make requirement 2 unsatisfiable. This one is stated in
the manifest so that a reader sees which substrate each cell rests on rather than
inferring that they share one.

## 2. Three axes, kept apart by type

The generic-robustness failure does not begin at the averaging step. It begins
where the choices are represented as one flat bag of knobs, because from there
"average over the knobs" is the natural thing to write. So the three kinds are
three types, and the type of the estimand is the key of every summary:

| Type | What it varies | Why it is not the others |
|---|---|---|
| `Estimand` | query, **baseline population**, its type and construction rule, selection rule and features | Changing it changes **what is being estimated**. Two estimands are two different questions and their answers are not two measurements of one thing. |
| `Measurement` | lexicon id, lexicon content hash, substrate id, hit table | Changing it changes **how the same question is measured**. |
| `StatisticalChoice` | estimator, block size, bootstrap replicates, seed, health floors | Changing it changes **how uncertainty is computed**, and nothing about the question or the measurement. |

A `Specification` is one of each. Its `cell_id` is a deterministic hash over the
canonical JSON of all three, so the same declared grid produces the same ids on
any machine, and a result can be traced to the exact triple that produced it.

## 3. The rule this exists to enforce

**Stability is summarised within a fixed estimand and never across estimands.**

A number that averages an effect against baseline *A* with the same effect against
baseline *B* is not a robustness statistic; it is two answers to two questions,
added. Where the baseline is the thing that moved, the honest report is that the
conclusion is **baseline-sensitive**, and naming which baseline gives which answer
— not a single score that conceals both.

This is enforced, not asserted. `guards.no_cross_estimand_pooling` takes each
summary's group key and its member cell ids, resolves every cell back through the
**manifest** — written before the run, by a different function than the one that
grouped — and refuses any group spanning more than one `estimand_id`. It is the
`verify_against_manifest` shape: the claim comes from one producer and the
recomputation from another's bytes.

## 4. No cell disappears

Every planned cell is written to the manifest **before** anything runs, and
appears exactly once in `cells.tsv` afterwards with one of:

`SUCCESS` · `REFUSED_GUARD` · `REFUSED_SCHEMA` · `NOT_ESTIMABLE` · `ERROR`

with the reason and, where a guard refused, the guard's own sentence. A cell that
could not be estimated is a finding about the design — usually that a baseline is
too small for the block bootstrap — and a grid that quietly reports only the cells
that worked is a grid that has selected on its outcome.

## 5. Reuse, not reimplementation

`multiverse` computes no statistics. It enumerates specifications, binds each to
its inputs, calls `interpret.interpret_query`, and records what came back. A
second implementation of a block bootstrap that agrees with the first 99% of the
time is worse than no second implementation, and the 1% surfaces as a number
nobody can trace.

## 6. Descriptive only

No threshold is invented. The stability summary reports counts, the sign pattern,
and the range across cells within an estimand; it does not classify a family as
"robust". Where a threshold was preregistered it is recorded and applied by name;
where none was, the report says so and stays descriptive.

## 7. Outputs

| File | What it is |
|---|---|
| `specification_manifest.json` | every planned specification with its deterministic ids, written before the run |
| `cells.tsv` | one row per planned cell, with status |
| `family_effects.tsv` | family-level effects for successful cells, each carrying its full identity |
| `dropped_cells.tsv` | every non-`SUCCESS` cell and why |
| `stability_by_estimand.tsv` | descriptive stability within each fixed estimand |
| `specification_curve.md` | the report a person reads |
| `guard_outcomes.json` | what each guard returned, as everywhere else here |

## 8. A design

Paths are relative to the design file. `test_the_design_documented_here_parses`
reads this block out of this document and builds it, so it cannot go stale
silently.

<!-- example-design:begin -->
```json
{
  "multiverse_id": "example",
  "peak_universe_id": "EXAMPLE_UNIVERSE",
  "preregistered_threshold": "NONE_PREREGISTERED",
  "estimands": [
    {
      "query_id": "island",
      "query_regions": "peaksets/island.txt",
      "baseline_id": "complement",
      "baseline_population_type": "COMPLEMENT_WITHIN_UNIVERSE",
      "baseline_construction_rule": "every peak of the frozen universe not in the query",
      "baseline_regions": "peaksets/complement.txt",
      "selection_provenance": "PROGRAMMATIC_RULE",
      "selection_rule": "leiden res0.5 == 5",
      "selection_feature_names": ["leiden_res0.5"]
    },
    {
      "query_id": "island",
      "query_regions": "peaksets/island.txt",
      "baseline_id": "sibling",
      "baseline_population_type": "SIBLING_CLUSTER",
      "baseline_construction_rule": "one other cluster at the same resolution, taken whole",
      "baseline_regions": "peaksets/sibling.txt",
      "selection_provenance": "PROGRAMMATIC_RULE",
      "selection_rule": "leiden res0.5 == 5",
      "selection_feature_names": ["leiden_res0.5"]
    }
  ],
  "measurements": [
    {
      "measurement_id": "core",
      "lexicon_id": "MODEL__core",
      "substrate_id": "0000000000000000000000000000000000000000000000000000000000000000",
      "hit_table": "substrate_core.tsv",
      "lexicon_content_hash": "1111111111111111111111111111111111111111111111111111111111111111",
      "lexicon_manifest": "lexicon_manifests/core.manifest.json"
    }
  ],
  "statistical_choices": [
    {"statistical_id": "pct_1Mb", "estimator": "percentile", "block_size": 1000000, "n_bootstrap": 100, "seed": 0},
    {"statistical_id": "bca_1Mb", "estimator": "bca-wild-cluster", "block_size": 1000000, "n_bootstrap": 100, "seed": 0}
  ]
}
```
<!-- example-design:end -->

Two estimands and two statistical choices is four cells; the grid is the product
of the three axes, so it grows quickly and a design is a thing to write
deliberately rather than to widen because widening is cheap.
