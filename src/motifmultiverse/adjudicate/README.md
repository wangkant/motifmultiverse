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

## The failure that produced it

The earlier merge table contained only collapse rows. Three refusals existed as
absent rows, indistinguishable from relationships that had never been
considered. Representative selection also had no persisted, authoritative
metadata contract.

## Inputs and outputs

The command requires the evidence directory from `annotate` and `--registry`
pointing to the versioned registry from `ingest`. It reads the packaged frozen
criterion registry unless `--criteria` names another versioned registry.

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
