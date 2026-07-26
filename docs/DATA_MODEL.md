# Data model

Implemented in `src/motifmultiverse/schema/`. Every rule below exists because its
absence caused a specific, documented failure.

## Motif node — six field groups

| group | fields |
|---|---|
| **identity** | `node_id`, `model`, `readout`, `context`, `metacluster`, `denovo_pattern_id`, `variant_id`, `family_id` |
| **representation** | `cwm`, `hypothetical_cwm`, `ppm`, `trimmed_core`, `motif_length` |
| **support** | `seqlet_count`, `core_ic`, `source_peak_count` |
| **annotation** | `annotation_matches` (TomTom / HOMER / JASPAR / HOCOMOCO), `putative_tf_label`, `low_confidence_annotation` |
| **uncertainty** | `family_assignment_source`, `family_assignment_confidence`, `discovery_tier`, `analysis_tier`, `tier_reason`, `exclusion_reason` |
| **provenance** | input checksum, software version, command, random seed, timestamp |

Representation belongs to one `model` x `readout`. Identity (`family_id`,
`variant_id`) is the ontology that crosses them. That separation is why
averaging CWMs across models is prohibited rather than merely discouraged.

Not every field is populated by every stage. `ingest` fills identity,
representation and support from the discovery HDF5; `source_peak_count` has no
source in a TF-MoDISco file and stays `None` until a stage that knows the peak set
fills it. Annotation and uncertainty stay at their sentinels until `annotate`
exists — which it does not (see its README).

### Known gap: one `family_id`, and disagreement is the normal state

`MotifNode` holds **one** `family_id`. Cross-model identity travels by that field
(`FP-03`), so two analyses proposing different families for the same motif is not an
edge case — it is the ordinary situation before adjudication, and the reason an
adjudication stage exists at all. The model cannot currently hold two candidates, so
the second assignment can only overwrite the first, silently, and whichever stage
runs last wins.

Closing this probably needs a `candidate_assignments` structure: one entry per
proposing analysis with its own source and confidence, leaving `family_id` as the
*adjudicated* result rather than the only slot. **Not implemented — this is a design
decision awaiting a ruling**, and it is recorded here rather than solved so that the
first caller to need it does not decide it by accident. See
`src/motifmultiverse/annotate/README.md`, question 4.

## Metacluster presence — three ways to be absent

Implementing `ingest` showed the model had no way to say *why* a metacluster
contributes no patterns, so `schema.MetaclusterState` was added:

| state | what it claims |
|---|---|
| `present` | the group exists and has patterns |
| `group_absent` | discovery ran; the group never formed. Evidence about the **admission gate**, whose seqlet threshold is absolute and therefore scale-dependent (`BA-12`) |
| `group_empty` | the group exists and contains no patterns. Discovery looked and found nothing |
| `not_searched` | this run never looked. No evidence either way |

In the reference implementation four discovery leaves had **no negative group at
all** in the HDF5. Reported as "no repressive motifs", all three absences become
the same false statement — the discovery-stage form of `BA-01`.

## Registry — what `ingest` emits

`schema.RegistryMetadata`: the project, the peak universe id, the analysis list,
`n_models`, the per-analysis `metacluster_states`, the trim threshold, and
`cross_model_claims_restricted`.

That last field is the N ≥ 3 rule travelling **with the data** instead of being
something a later step must remember to ask. It is derived, and the constructor
refuses a value that disagrees with `n_models`.

## Lexicon manifest — what `compile` emits

`schema.LexiconManifest`: the tier, the `lexicon_content_hash`, the pattern order
**in the order the loader emits**, the node ids behind those patterns, and a
`comparisons` block stating for every other tier whether the positive and negative
sets are identical.

Two fields exist for specific failures. The content hash is what `FP-11` requires
every family-level number to cite; without it there is nothing to cite. The
comparison block exists because a tier contrast that changes nothing must say so —
in the reference implementation `core` and `expanded` held identical positive
sets, so a sensitivity analysis that appeared to vary lexicon width varied only the
negative half, and no artifact recorded that.

`compile` renumbers patterns into loader order, so a manifest row carries **both**
the new `pattern_tag` and the source `node_id`. The manifest is the translation
table; nothing recovers the mapping by parsing a name.

## Evidence edge — six classes

| class | what it asserts |
|---|---|
| `alignment` | two nodes register at an offset with a similarity and a null |
| `sequence_hit` | a motif matches a sequence interval |
| `attribution_hit` | an instance was called in a region with a coefficient |
| `downstream_sensitivity` | a reconstruction changed (or did not) when the lexicon changed |
| `external_biology` | outside evidence (ChIP, conservation, perturbation) |
| `decision` | a human or policy decision, with rationale and decider |

Every edge carries a `missingness` state and, where applicable, an uncertainty
interval. An edge whose state is not `used` may not carry a statistic of `0`.

## The four rules (T-12)

### 1. `variant_id` is the only stable semantic identity, and is marked as such

In the reference implementation `variant_id` was the one identifier that was 1:1
with the discovery key with zero collisions — and **nothing in the codebase said
so**, so downstream code kept reaching for less stable keys. Here the format is
validated at construction and `guards.variant_id_unique` checks the 1:1 property.

### 2. No semantics may be parsed out of an identifier string

A hit-caller row number was once matched against a discovery manifest pattern id.
Nothing errored. One factor's evidence was simply filed under another's name.

Identifiers are `NamespacedId`; crossing a boundary requires `translate()` with an
explicit table that **raises on unknown keys** rather than dropping the row.
`guards.no_key_parsing` walks the AST and rejects slicing, `split`, `startswith`
and friends applied to an identifier.

### 3. Missingness is four-state and never collapses to 0

`not_searched` / `no_sequence_match` / `hit_below_floor` / `used`.

This encoding had already been written down in the reference implementation. It
was still destroyed, because a table pivot with `aggfunc="sum"` returns `0.0` for
an all-undefined group — so undefined values entered arm means as zeros. The
coverage figure was then computed *after* that fill and reported perfect coverage,
which made the error look like a validation.

Hence two checks, not one: undefined values may not be `0`, **and** coverage must
be computed pre-fill.

### 4. A decision must express a refusal; confidence must be a measure

In the reference implementation every row of the merge table was `collapse`.
Refusals existed only as *absent rows* and were therefore indistinguishable from
"never considered" — including for the single most consequential adjudication in
the project. And `merge_confidence` was a per-family name lookup returning one of
two labels, which a downstream re-tiering gate really did read.

`Decision` therefore includes `REFUSE_MERGE`, every record requires a `rationale`
and a `decided_by`, and a `confidence` outside `[0, 1]` is rejected.

## Interpretation record — three health views, one deprecated alias

`interpret.Interpretation` (see `src/motifmultiverse/interpret/README.md`) carries three
separate health records rather than one:

| field | what it reports health of |
|---|---|
| `query_health` | the submitted query peak set alone |
| `comparator_health` | the submitted comparator peak set alone, or `None` if none was submitted |
| `contrast_health` | both sides together (`interpret.ContrastHealth`: `query`, `comparator`, `shared_blocks`, `union_blocks`, `passed`, `floor_failures`), or `None` if there was no comparator to contrast against |

An effect is emitted only when **both** `query_health` and `comparator_health` pass their
floors — checking the query alone and differencing against an unexamined comparator can
silently produce an effect size from a comparator that would itself have been refused as a
query. Failures in `contrast_health.floor_failures` are prefixed `query:` or `comparator:` so
a reader does not have to re-derive which side is responsible.

### `floor_failures` at the top level is not the same list as `contrast_health.floor_failures`

`Interpretation.floor_failures` (top level) and `contrast_health.floor_failures` (nested,
`interpret.ContrastHealth`) are allowed to diverge, and the divergence is intentional rather
than a bug to be "cleaned up":

- `contrast_health.floor_failures` is the **unconditional union**: whatever `query_health` and
  `comparator_health` individually failed, prefixed and concatenated, regardless of whether the
  query's selection provenance ever licenses an effect against that comparator. It exists for
  full transparency about both peak sets as submitted.
- `Interpretation.floor_failures` (top level) is the **operative** list: only the failures that
  actually caused *this* interpretation to withhold something. A comparator can be declared but
  irrelevant — a `DESCRIPTIVE_ONLY`/`EYEBALLED` query never touches its comparator for
  inference — and an unhealthy-but-irrelevant comparator must not appear in the operative list,
  because nothing was suppressed on its account. Concretely: on full suppression (the query
  itself fails) the operative list holds the `query:`-prefixed failures; when the query passes
  and a comparator is checked (`FULL_INFERENCE`/`FULL_INFERENCE_HELD_OUT` with a declared
  comparator) and it fails, the operative list holds the `comparator:`-prefixed failures instead;
  otherwise it is empty.

A caller checking "was anything suppressed?" should read the top-level field, not the nested
one — `contrast_health.floor_failures` being non-empty does not imply `suppression_reason` is
set.

**Deprecated:** `Interpretation.health` is kept for one release as an alias of
`query_health` — readable both as an attribute (`result.health`) and as a `to_dict()` key
(`blob["health"]`) — so existing readers keyed on the old, single ambiguous `health` field do
not break. It will be removed in a later release; new readers should use `query_health`.

## Two tiers, not one (T-13)

`discovery_tier` answers *how strongly was this pattern discovered?*
`analysis_tier` answers *should it enter the analysis lexicon as an independent
detector?* When they differ, `tier_reason` is mandatory. A single tier field
silently answers both questions with one value.
