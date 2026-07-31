# `compile`

## The rule

Discovery support and analysis admission are two fields, not one; every lexicon is
content-addressed; the written index is in the order the **loader** emits, compared
by name; and a tier contrast that varies nothing must say so.

## The failure that produced it

A single `tier` conflated how strongly a pattern was discovered with whether it
should enter the analysis lexicon as an independent detector, so excluding a weak
variant silently changed both questions at once. Separately, a frozen index was
sorted by metacluster ascending (`neg` before `pos`) while the loader emits
positives first — every positional read against it was wrong, and it looked correct
only because the model in question had no negative motifs at all. And `core` and
`expanded` held identical positive sets, so a sensitivity analysis that appeared to
vary lexicon width varied only the negative half, with nothing in the artifact
saying so.

## How to check it

`schema.MotifNode` requires `tier_reason` whenever `discovery_tier !=
analysis_tier`; `guards.variant_id_unique` enforces 1:1 identity;
`guards.index_order_matches_loader` compares the written index to what the real
loader returns, **by name**; and each manifest's `comparisons` block states whether
the positive and negative sets differ from every other tier's.

### ⚠ The loader round trip is not covered by CI

`tests/test_ingest_compile.py::test_roundtrip_against_the_real_loader` calls the real
loader — and it **skips** wherever the `finemo` backend is not installed, which
includes this repository's CI. *A skipped test is unverified, not verified.* Treating
the two as equivalent is the same error as a guard that can never fail (`FP-25`), and
that error is what this project exists to avoid, so it is stated here rather than
left to be inferred from a green check mark.

What has been verified, manually, on a **real** ChromBPNet TF-MoDISco output
(22 positive + 11 negative patterns), outside CI:

```
loader returned 33 motifs; cwms (33, 4, 50); trim_masks (33, 50)
manifest == loader order: True
9..11 : ['pos_patterns.pattern_9', 'pos_patterns.pattern_10', 'pos_patterns.pattern_11']
```

The middle line is the claim; the third shows it holding across the 9 → 10 boundary,
where a lexicographic sort and the loader's numeric one diverge.

**Two things that run were not the same two things.** The verifying environment has
`finemo` but runs Python 3.10, and this package requires 3.11 (`StrEnum`), so the
loader was called **directly on the compiled HDF5** rather than through
`compile.load_back`. So: *the artifact is proven loader-compatible and its order
matches the manifest; the ten-line `load_back` wrapper has never been executed.*
Those are different statements and merging them into "the round trip passes" would be
the writing-a-rule-down-is-not-executing-it failure in miniature.

---

## Status: **implemented**

```bash
motifmultiverse compile registry/ --decisions decisions.json \
    --tiers core,expanded,sensitivity --verify-roundtrip auto --out lexicons/
```

Per tier: `<tier>.h5` (hit-caller layout: `pos_patterns` / `neg_patterns`, each
pattern carrying `contrib_scores`, `hypothetical_contribs`, `sequence`) and
`<tier>.manifest.json`, plus one `manifest.tsv` across tiers.

### Order comes from the loader, not from us

The loader walks `['pos_patterns', 'neg_patterns']` in that fixed order and sorts
within a group by the **integer suffix** of `pattern_N`. So `compile` *renumbers*:
patterns are assigned `pattern_0..N` in the order they should come back, which means
the loader's own sort reproduces the manifest exactly — including across the 9 → 10
boundary, where a lexicographic sort and a numeric one diverge.

Note the direction of that: we never recover the loader's order by parsing a name
(which would be the `no_key_parsing` failure). We assign names such that the
loader's rule yields our order, and the manifest carries **both** the new
`pattern_tag` and the source `node_id` so the mapping is a table, not an inference.

### Verification is behavioural

`--verify-roundtrip auto` reads each lexicon back with the **real** loader and
compares by name. Asserting that the file contains the groups we just wrote would
prove only that this package can read its own output.

The loader lives in the `finemo` backend, which is an optional dependency. Without
it, `auto` writes the lexicon and says plainly that no round trip was performed;
`require` fails instead; `skip` never tries. A skipped verification means
**unverified**, not verified.

One compatibility constraint falls out of reading that loader: it stacks every motif
into a single array, so a lexicon whose patterns differ in length cannot be read back
at all. `compile` refuses to write one.

### Publication is all-or-nothing

Tiers are written to a staging directory and moved into `--out` only once every
one of them has been written and verified. A compile that wrote `core` and then
refused `expanded` used to leave both files behind, and `validate` binds a lexicon
set by globbing `*.manifest.json` — so the wreckage of a refused compile read
downstream as a perfectly valid one-tier lexicon. What a refusal does leave is
`provenance.json`: a rejected compile still records what was attempted (`T-09`).

### Tiers, and the threshold that is deliberately absent

`core` ⊆ `expanded`; `sensitivity` is `expanded` with some merges left split
(`FP-06`). **Which merges is decided by three named triggers, not by a number:**

| trigger | fires when |
|---|---|
| `merge_confidence_not_high` | `merge_confidence` is anything other than `HIGH`, **including undeclared** |
| `family_ambiguity` | the decision declares it |
| `threshold_sensitive` | the decision declares it |

Any one of them is enough, and each manifest records which fired, per cluster.

There is no numeric cut-off anywhere in this module for merge confidence, and its
absence is the point. The design lists "moderate-confidence merge" as one trigger
for a sensitivity lexicon but **never defines it and never says it is a scalar**; in
the reference implementation the value was produced by
`"moderate" if family == "ZNF76" else "high"` — a lookup by name over a hard-coded
singleton. Picking 0.8, or 0.7, would invent a continuous quantity that the design
does not contain, and a downstream gate really did read that non-measure as though it
were one. `schema.MergeConfidence` is therefore a `StrEnum`, which also means
`grade < 0.8` raises `TypeError` rather than quietly comparing.

**What earns each grade is not decided yet.** `schema.MERGE_CONFIDENCE_CRITERIA`
records `CRITERION_NOT_YET_DEFINED` for all three, because assigning the grade
belongs to `adjudicate`, whose criteria are a preregistration item (`FP-13`).
`compile` only *dispatches* on a declared grade; it never assigns one. Note that
undeclared fires the trigger — "not stated" is not "high", and being wrong that way
costs one extra sensitivity lexicon rather than an unchecked merge.

A collapse names a representative, and `compile` refuses one that is not among its
own members — that is what "observed medoid, never a constructed average" means at
the artifact level (`FP-05`).

### `motif_type` is the loader's vocabulary

`motif_type` is handed verbatim to the backend named by `loader_backend`. finemo's
loader dispatches on `cwm` / `hcwm` / `pfm` / `pfm_softmax` and has no else-branch,
so an unrecognised value leaves its motif locals unbound and raises
`UnboundLocalError` from inside the backend rather than being refused here. The
compiler's table once keyed that third entry `ppm` — *our* name for the array —
and `compile(motif_type="ppm")` therefore wrote a lexicon that the very loader its
manifest declared could not read.

### `lexicon_content_hash`

`FP-11` requires every family-level number to state the lexicon it was computed
under. The hash is over the ordered content (pattern tags, node ids, **variant
ids**, arrays), not over the file, so it is stable across rewrites. Nothing yet
checks that a downstream number cites it; the thing to cite now exists.

`variant_id` is in there because it is what downstream binds to. The hit caller
returns a `pattern_tag`; this manifest's `index` is the only table that resolves a
tag to the semantic identity every family-level number is grouped by. Hashing tag,
node id and arrays alone let two lexicons that resolve the same tags to entirely
different variant assignments share one hash — so the citation `FP-11` asks for
named a lexicon it could not tell apart from another one.
