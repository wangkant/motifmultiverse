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
loader returns, **by name**, and what it returned per tier is written to
`guard_outcomes.json` beside the lexicons (`motifmultiverse.guard_log`) — with no entry
for a tier no backend could read back, because no guard ran there; and each manifest's
`comparisons` block states whether the positive and negative sets differ from every
other tier's.

### The loader round trip runs

`tests/test_ingest_compile.py::test_roundtrip_against_the_real_loader` calls the real
loader. It used to **skip** wherever the `finemo` backend was not installed, which
included this repository's CI — and it skipped there for a reason nobody had checked:
the `finemo` extra named the distribution `finemo-gpu`, which does not exist on PyPI
(`pip install finemo-gpu` → *No matching distribution found*). The documented way to
make the test runnable could not be run. The distribution is `finemo`; the extra now
says so, and `test_the_declared_finemo_extra_names_the_distribution_that_provides_the_backend`
holds it to that.

**What the skip was hiding.** Installing the backend turned 31 tests red. finemo 0.40
renamed the loader's `trim_threshold` argument to `trim_threshold_default` and added
`trim_coords` / `trim_thresholds`; `load_back` passed its arguments by keyword inside
a `try` that caught only `ImportError`, so every call raised `TypeError: unexpected
keyword argument 'trim_threshold'` — and because `--verify-roundtrip auto` is the
default and catches only `BackendMissing`, that aborted the compile outright and no
lexicon was written at all. `_loader_call_kwargs` now binds this package's settings to
the *installed* signature, and refuses with `BackendIncompatible` (a `BackendMissing`
subclass, so `auto` writes and claims nothing while `require` fails) rather than
guessing when it recognises neither spelling.

**Verified on a real lexicon**, compiled by `ingest` + `compile` from the ChromBPNet
TF-MoDISco outputs for two AG-2048 islands (`promoter_cl5`: 16 pos + 6 neg;
`distal_cl8`: 4 pos + 3 neg) and read back through `compile.load_back`:

```
ingested 29 nodes from 2 real modisco.h5 files
compiled core: 29 patterns, hash 31833bc41a8c, motif_type=cwm, trim_threshold=0.3
loader returned 29 motifs; cwms (29, 4, 50); trim_masks (29, 50)
manifest == loader order: True
9..11 : ['pos_patterns.pattern_9', 'pos_patterns.pattern_10', 'pos_patterns.pattern_11']
index sorted lexicographically matches the loader:          False
index sorted by metacluster ascending matches the loader:   False
```

The third line is the claim. The fourth shows it holding across the 9 → 10 boundary,
and the last two are what make it a claim at all: on this lexicon both of the orders
one would naturally write instead are wrong, so a passing round trip here is
discriminating rather than vacuous.

Both loader generations were exercised on that same file: finemo **0.41** (PyPI,
Python 3.13) through `compile.load_back`, and finemo **0.30** (an older
installation on Python 3.10 — this package needs 3.11, so the loader was called
directly there with the arguments `_loader_call_kwargs` binds for the 0.30
signature). Both returned the same 29 names in the manifest's order.

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

The loader lives in the `finemo` backend, which is an optional dependency
(`pip install -e ".[finemo]"`). Without it — or with a release whose loader this
package cannot call, which is the same thing from the artifact's point of view —
`auto` writes the lexicon and says plainly that no round trip was performed;
`require` fails instead; `skip` never tries. A skipped verification means
**unverified**, not verified. Set `MOTIFMULTIVERSE_REQUIRE_FINEMO=1` in any run that
must not be allowed to skip it.

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

### The operations log

`combination_operations.json` says, for every motif in every emitted tier, what it
was built from: `copy`, `select_representative` (a collapse that chose an observed
member), or `mean`. `guards.no_cross_model_cwm_avg` reads it before anything is
published and refuses a `mean` that does not hold model, readout and metacluster
fixed.

**Where the log comes from is the whole of it.** The obvious version is one the
collapse writes about itself — append `{"op": "medoid"}` next to the code that
picked the medoid — and a guard reading that audits a claim made by the code it is
auditing. It passes for exactly as long as somebody keeps the claim current by
hand, which is the unchecked invariant the guard replaces, one level in.
`compile.operations_log` therefore asks nothing: it opens the lexicon that was
just written and classifies each motif's matrices against the registry arrays of
the nodes it stands for. A future stage that starts averaging produces a `mean`
entry without editing the classifier, because the classification is of the file.

Four limits, since a pass here should not be read as wider than it is, and the
first is the largest. **The check reaches back to the registry and no further.**
The classifier's reference set is what `ingest` wrote, so a cross-model mean
performed *before* that — precisely where a "meta-analysed CWM across models" stage
would live — arrives as an ordinary registry motif, classifies as `copy`, and
passes; `test_a_cross_model_mean_made_upstream_of_the_registry_passes` builds one
and asserts exactly that, so the gap is pinned rather than described. A pass
therefore means "nothing downstream of the registry averaged", never "this lexicon
contains no cross-model average", and the guard's own sentence says so, because it
is persisted verbatim in `guard_outcomes.json` and printed verbatim by `report`.
An average over inputs whose matrices are identical *is* another of them, and no
reader of the artifact can see that it happened. The inputs of a collapse are taken to be its
representative plus the decision's other members that this tier did not emit, so a
member dropped for an unrelated reason widens the set a mean is checked against —
which can only cause a refusal to classify, never a silent pass. And a
representative averaged *within* one model holds all three axes fixed and passes
the guard (`FP-05`); what changes is that the log records it as `mean`, so it is
visible in a shipped artifact instead of nowhere.

An emitted motif that is neither its inputs nor their mean is refused, not filed
under the nearer label: an operation this package cannot name is one it cannot
audit for the axes it held fixed.

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
