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
| `EXTERNAL` | full inference: effect, interval, *q* |
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

Percentile **block** bootstrap over whole genomic blocks, paired between query and
comparator, with the block size, replicate count `B` and seed stored beside every
interval. The *p* value is floored at 1/(B+1) and *q* is Benjamini-Hochberg over
the families tested in that call.

`FP-15` specifies a **BCa** paired block bootstrap and a block-level wild cluster
bootstrap-*t*. Neither is implemented. Every result names its estimator
(`percentile_block_bootstrap`) rather than saying "block bootstrap", because the
distance between what was specified and what ran is exactly the thing that
disappears from a methods section.

Every result also carries `estimators_defined` and `estimators_implemented`
(`schema.Estimator`), so a consumer branches on the recognised set rather than on a
string literal and keeps working when the specified estimators arrive. Label
permutation is absent from that enum on purpose: it is not unimplemented, it is
**abandoned** — under block-correlated structure it understates the variance.

### Exit codes

`0` success · `2` usage or missing input · `4` refusal — the tool declined to
produce a number, and the message says which rule declined it.
