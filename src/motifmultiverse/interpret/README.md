# `interpret`

## The rule

A peak set may only be described at the strength its **selection provenance**
licenses, and the three health numbers come before any effect — if one is below its
pre-registered floor the reading is suppressed, not annotated. Describing what is
inside a cluster is a different question from testing how robust a conclusion is;
both share the statistics, neither may re-run the caller.

## The failure that produced it

Descriptive cluster summaries and robustness checks were run through the same
ad-hoc scripts, so an exploratory listing and a preregistered test became
indistinguishable in the record. Separately, one clustering resolution was chosen
by an automated agent with no criterion retained — the conditioning set cannot be
reconstructed, so it cannot be shown that downstream information was not already
visible when the peak set was chosen.

## How to check it

`guards.selection_provenance_declared` (an undeclared query takes the most
conservative mode, never a permissive default), `guards.health_before_effect`
(health first; a failed floor suppresses the reading), `guards.comparator_declared`
(no baseline, no number) and `guards.single_scale`. Interpret outputs are labelled
descriptive and may not be cited as a test result.

---

## Status: **implemented**

The first module in the package with a real body. It was chosen for that because it
is the only one of the nine that needs neither TF-MoDISco nor a hit-caller backend:
it consumes a frozen hit table and answers subset queries over it, so it runs end to
end with nothing else installed.

```bash
motifmultiverse interpret hits.tsv \
    --peaks island_5.txt \
    --comparator gc_matched.txt --comparator-id gc_matched_negatives \
    --selection-provenance EXTERNAL \
    --block-size 1000000 --bootstrap 2000 --seed 42 \
    --out interpretation/
```

### Input

| input | what it is |
|---|---|
| hit table | one **frozen** full-universe run, `.tsv` or `.parquet`, columns from `schema.HIT_TABLE_COLUMNS` |
| peak set | BED (4th column is the region id) or one region id per line |
| selection provenance | one of the six grades below; omitting it is recorded, not defaulted |

The hit table must carry a row for **every peak in the universe**, including peaks
that were searched and produced nothing (`missingness = no_sequence_match`) and
peaks that were never searched (`not_searched`). This is not bookkeeping: a table
containing only called hits has silently lost its zero-hit peaks, and every ratio
computed against it is inflated. `hit_coefficient` is empty for those rows —
`schema.HitRecord` raises if it is `0.0`.

`not_searched` peaks are **excluded from denominators**, never counted as zeros.
`no_sequence_match` and `hit_below_floor` *are* measurements and contribute 0.

### The three health numbers

Each is reported with its denominator.

| number | definition | default floor |
|---|---|---|
| `intersection_coverage` | submitted peaks found in the universe / submitted peaks | 0.90 |
| `n_blocks` | distinct genomic blocks the query spans | 30 |
| `explained_fraction` | searched query peaks with ≥1 hit from the frozen lexicon / searched query peaks | 0.50 |

`n_blocks` is a health number because for a clustered peak set the **effective**
sample size is the block count, not the peak count. The default of 30 is the same
floor `guards.estimability_floor` applies to N.

Floors are arguments (`--floor-coverage`, `--floor-blocks`, `--floor-explained`),
declared before the run and recorded in the result. A floor chosen after the
numbers are visible is not a floor.

`ContrastHealth` reports a fourth number that has no floor because it has no
tolerable value: `n_shared_peaks`, the peaks submitted on **both** sides. A peak
on both sides of the difference is subtracted from itself, so an effect is
refused outright when the comparator overlaps the query. "Cluster vs all peaks"
attenuated every family by exactly the comparator's disjoint fraction (0.7560 on
a real K562 island) with the interval shifted to match; at `comparator == query`
every effect was exactly 0.0 with a zero-width interval. `shared_blocks` does not
show it — 283 blocks overlapping against 282 disjoint. Spell it "query vs
everything except the query".

**Suppression here is wider than the design report requires**, and that is this
repository's own tightening. The report suppresses the *interpretation section*; a
failed floor here also withholds the descriptive composition. The reasoning: at 20%
intersection coverage a per-family composition table is itself a misleading reading,
and leaving it visible invites exactly the quotation the suppression exists to
prevent. If a project wants the looser rule, that is one condition in
`interpret_query`.

### Output modes

| selection provenance | output |
|---|---|
| `EXTERNAL` | full inference: effect and interval (*p* and *q* only under `--estimator bca-wild-cluster`) |
| `PROGRAMMATIC_RULE` | full inference; the rule text is required and recorded |
| `CLUSTERED_WITH_SPLIT` | full inference **on the held-out half only** |
| `CLUSTERED_NO_SPLIT` | descriptive decomposition; no interval, no *p* |
| `EYEBALLED` | descriptive decomposition |
| `MODEL_SELECTED_NO_TRANSCRIPT` | descriptive decomposition, **plus** a note that the conditioning set cannot be verified |
| *(undeclared)* | recorded as `DECLARATION_MISSING`; runs in the most conservative mode |

`MODEL_SELECTED_NO_TRANSCRIPT` is stricter than `EYEBALLED` on purpose. A human
selector can testify afterwards to what they looked at; an agent's conditioning set
cannot be reconstructed, and in particular cannot be shown to exclude downstream
information. Asking the agent later does not repair it.

### What the estimator actually is

Two paths, selected by `--estimator`. Both resample **whole genomic blocks**, both
are paired between query and comparator, and both store the block size, replicate
count `B` and seed beside every interval. What the flag changes is how the
uncertainty around the same point estimate is computed, and therefore what the
result is licensed to carry.

| `--estimator` | recorded as | interval | *p* / *q* |
|---|---|---|---|
| `percentile` *(default)* | `percentile_block_bootstrap` | percentile block bootstrap | **withheld** — `ESTIMATION_ONLY` |
| `bca-wild-cluster` | `wild_cluster_bootstrap_t` | BCa paired genomic-block bootstrap | wild cluster bootstrap-*t*, *q* by Benjamini-Hochberg over the families in that call — `INTERVAL_AND_TEST` |

`FP-15`'s specified pair is the second row, and it is implemented
(`infer.bca_paired_block_interval`, `infer.wild_cluster_bootstrap_t`). It is not
the default: a default that emits *p* values emits them to callers who never
decided they wanted a hypothesis test. The default path emits no *p* and no *q* at
all, because the proportion of bootstrap replicates crossing zero looks like a
two-sided *p* value and is not a calibrated one — and a *q* derived from it would
inherit the invalidity.

The default path refuses below `interpret.MIN_PERCENTILE_REPLICATES` (39) rather
than degrading: `B` replicates resolve a tail no finer than 1/(B+1), so a 2.5%
tail needs `B ≥ 39`, and below it both endpoints are the extreme replicates. At
`B = 1` that produced `[x, x]` — a zero-width 95% interval printed beside its
point estimate, which reads as infinite precision rather than as one draw.

Every result names its estimator rather than saying "block bootstrap", because the
distance between what was specified and what ran is exactly the thing that
disappears from a methods section. The recorded value names the half that decides
the result's *capability* — the test — with the interval half named in the run's
notes.

Every result also carries `estimators_defined` and `estimators_implemented`
(`schema.Estimator`), so a consumer branches on the recognised set rather than on a
string literal and keeps working when the specified estimators arrive. Label
permutation is absent from that enum on purpose: it is not unimplemented, it is
**abandoned** — under block-correlated structure it understates the variance.

### Exit codes

`0` success · `2` usage or missing input · `4` refusal — the tool declined to
produce a number, and the message says which rule declined it.

## Getting an `interpret`-ready hit table out of `compile`

The stages in this package stop at a compiled lexicon. `interpret` and `infer`
start from a *hit table*, and nothing here produces one, because producing one
means running a hit caller over your peaks — an external backend (`finemo-gpu`),
declared optional on purpose. The seam is real work, not a missing function call,
so it is written down rather than stubbed:

1. `compile` writes `<tier>.h5` plus `<tier>.manifest.json`. The manifest's
   `index` maps each `pattern_tag` in the H5 to its `node_id` and `variant_id`.
2. Run the hit caller with `<tier>.h5` over the peak universe you intend to query.
   **One frozen run over the whole universe** — the caller is not input-scale
   invariant, so every later query has to be a subset of that one run, never a
   re-run per subset.
3. Join the caller's per-hit `pattern_tag` back through the manifest to
   `variant_id`, and emit the columns in `schema.HIT_TABLE_COLUMNS`.
   `substrate_id` identifies that frozen run; `input_scale` is its region count.
4. `missingness` is four states, not a flag. A peak the caller searched and
   retained nothing for is `no_sequence_match` or `hit_below_floor`, never a row
   with `hit_coefficient` 0.0 and never an absent row.

### `family_id` is not derivable here, and that is deliberate

Step 3 cannot be completed from this package's outputs alone: `compile` emits
`variant_id`, never `family_id`. Family assignment is the slot
[`../annotate/README.md`](../annotate/README.md) records as **never specified** —
what a family assignment must satisfy has not been decided, so there is nothing to
implement yet, and inventing one to close the seam is precisely the failure that
README is about.

Until it is decided, `family_id` has to come from your own ontology, declared and
versioned like any other input. It cannot be left at the sentinel: every number
`interpret` reports is grouped by family, so a sentinel family is reported *as* a
family rather than as the missing assignment it is. `HitRecord` therefore refuses
a `used` row that does not name one — the same rule it already applied to
`variant_id`.
