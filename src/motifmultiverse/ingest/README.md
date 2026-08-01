# `ingest`

## The rule

Every input carries its own checksum, software version and namespace before anything
reads it; identifiers are opaque join tokens; and a metacluster that contributes no
patterns records **which** of the three absences occurred.

## The failure that produced it

A hit-caller row number was matched against a discovery manifest pattern id, filing
one transcription factor's evidence under another's name. Nothing errored; the join
simply produced the wrong answer. Separately, four discovery leaves had **no negative
group at all** in their HDF5 — not an empty one — and the difference between "the
metacluster never formed" and "we looked and found nothing" was not representable.

## How to check it

`guards.no_key_parsing` (static, and run over this module's own source in the test
suite) and `schema.translate` (runtime) reject identifiers used across a namespace
boundary without an explicit table. `schema.MetaclusterState` makes the three
absences three values, and `schema.RegistryMetadata` refuses any fourth.

---

## Status: **implemented**

```bash
motifmultiverse ingest project.yaml --out registry/
```

Reads every discovery HDF5 named by the project config and writes:

| file | contents |
|---|---|
| `registry.json` | `RegistryMetadata` + one record per motif node, six field groups |
| `arrays.h5` | `cwm`, `hypothetical_cwm`, `ppm` per node |
| `provenance.json` | checksum of every input, command, software, seed, timestamp |

### What the config must declare, and why it cannot be inferred

Each analysis declares `id`, `model`, `readout`, `union_id`, `context` and
`modisco_h5`. **`union_id` is declared, never derived.** Deriving it from a filename,
or by slicing the analysis id, would be parsing semantics out of an identifier — the
failure above. It must be alphanumeric, so a value like `CBP_2048` is refused: that
reference-implementation key said 2048 while the real input width was 2114, and it
was harmless only because nothing ever read the digits.

`search_metaclusters: {pos_patterns: true, neg_patterns: false}` declares which
groups this run looked for at all. That declaration is the only thing that can
separate `not_searched` from the two absences the file itself can show.

### A layout this reader cannot read is refused, not reported as two absences

`group_absent` claims discovery ran and the group never formed — evidence about
the admission gate. An **original** (pre-`tfmodisco-lite`) TF-MoDISco file keeps
its patterns under `metacluster_idx_to_submetacluster_results`, so this reader saw
neither `pos_patterns` nor `neg_patterns` and recorded that claim twice, with exit
0 and an empty registry, for a file that may hold dozens of patterns. That is a
measurement invented out of the reader's own blindness. The three absences are
unchanged; what is refused is answering with any of them about a file that was
never read.

`cross_model_claims_restricted` is written into the registry whenever fewer than
three distinct models are present, so the N ≥ 3 rule travels with the data instead
of being something a later step has to remember to ask.

### Analyses that share one attribution array

`shared_attribution_groups: [[a, b], [c, d]]` declares which analyses read **one**
attribution array. They are one observation, not two, and the registry carries the
declaration plus the derived `n_attribution_sources` — one per group, one per
ungrouped analysis. A group naming an unknown id, naming an analysis already in
another group, or naming fewer than two analyses is refused: a one-member group
changes no count, so it is a declaration that cannot alter anything.

Omitting the key records **never declared** and leaves `n_attribution_sources`
`null`. That is not a claim of independence, and `len(analyses)` may not be
substituted for it — the difference between the two is the reason the field is
recorded at all.

**Nothing downstream reads either field yet**, and this page says so rather than
implying otherwise. `MotifNode.cross_context_recurrence` is the data model's one
count-of-corroboration field; it is always `None` today and `adjudicate` already
uses it as a medoid tie-break. Whatever eventually populates it must count distinct
attribution sources, and `n_attribution_sources` is where that denominator comes
from.

### Ordering is not established here

Patterns are read in the file's own key order, which is **not** the hit caller's
order — the loader sorts by a pattern's numeric suffix, and recovering that here
would mean parsing the digits out of a name. `compile` establishes loader order by
*assigning* names rather than by parsing them, and its manifest is the translation
table.

### Not populated

`source_peak_count` has no source in a TF-MoDISco file. Annotation and uncertainty
fields hold their sentinels: `family_id` is `NA`, and the middle segment of
`variant_id` reads `UNASSIGNED` because `annotate` does not exist. That segment is
decorative — `family_id` is the authoritative field.
