# `adjudicate`

## The rule

Connectivity proposes components; it never licenses a merge. Every proposed
component is recorded as `collapse`, `refuse_merge`, an explicit curator
override, or `deferred`. A collapse requires a versioned executable criterion,
complete evidence for every member-to-medoid gate, and an observed
representative from the authoritative motif registry.

Family conflicts always refuse a merge. Components missing any pairwise
registration defer before structural criteria are evaluated, preventing
single-linkage transitivity from becoming evidence. Missing criterion evidence
also always defers.

**Recorded limitation.** Component proposal reads edge *presence* only: an
alignment edge whose own null says the registration is unremarkable joins its
two nodes exactly as a compelling one does. That is unrestricted single linkage,
the open half of `FP-05`, and closing it means declaring the distance ceiling
`FP-13` reserves to the design. Expect large proposed components on a real
registry, and read them as proposals — the criterion gate below is what stands
between a component and a collapse.

## The failure that produced it

The earlier merge table contained only collapse rows. Three refusals existed as
absent rows, indistinguishable from relationships that had never been
considered. Representative selection also had no persisted, authoritative
metadata contract.

## Inputs and outputs

The command requires the evidence directory from `annotate` and `--registry`
pointing to the versioned registry from `ingest`. It reads the packaged **default**
criterion registry, `criteria.v1.yaml`, unless `--criteria` names another
versioned registry.

**The default removes no motifs.** In `criteria.v1.yaml` both `TRUE_DUPLICATE`
and `FRAGMENT_MATCH` are `CRITERION_NOT_YET_DEFINED`, so every duplicate and
every fragment is recorded as `deferred` and `compile` emits an undeduplicated
lexicon. The asymmetry that decides this: an under-deduplicated lexicon carries a
duplicate the reader can see and merge downstream, while an over-deduplicated one
has lost a motif and does not say which, so deletion is opted into rather than
administered.

`criteria.v2.yaml` also ships and is the registry that *does* deduplicate — a
preregistered `FROZEN_DECLARED_HEURISTIC` `TRUE_DUPLICATE` whose two declared
magnitudes, known costs and fired falsifiers are set out in
`docs/MERGE_CRITERION_PREREGISTRATION.md`. Pass it to `--criteria` (or reach it
in code with `adjudicate.packaged_v2_criteria_path()`) after reading that
document.

It writes:

- `ontology_decisions.parquet`, with artifact identity and provenance in
  Parquet file metadata even when there are zero decision rows;
- `merge_decisions.json`, the identity-bearing compile handoff whose content ID
  covers decisions, tier overrides, and provenance;
- `review.yaml` (or `--review`), the human audit surface.

## How to check it

Run `pytest tests/test_adjudicate.py`. The adversarial controls cover family
conflicts, non-transitive components, missing evidence, representative
tie-breaking, manual overrides, provenance tampering, and empty artifacts.

---

Status: **implemented**.
