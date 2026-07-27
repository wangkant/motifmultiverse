# `infer`

## The rule

Every specification is a subset of ONE frozen hit substrate, and a specificity claim needs an interaction interval excluding zero.

## The failure that produced it

The hit caller is not input-scale invariant: the same regions produced different discrete retention decisions depending on which other regions shared the input, with onset measured under 10% growth on the base set. Re-calling per specification would confound the specification with the caller.

## How to check it

`guards.single_scale`, `guards.interaction_required`, `guards.estimability_floor`, `guards.stratum_parity`.

---

Status: **partial**. `FP-15`'s two specified estimators are implemented and tested —
`bca_paired_block_interval` (BCa paired genomic-block interval) and
`wild_cluster_bootstrap_t` (block-level wild cluster bootstrap-*t* *p* value); the
block, never the peak, is the resampling and jackknife unit in both. `interpret
--estimator bca-wild-cluster` runs them as one `INTERVAL_AND_TEST` path, while the
default estimator stays the conservative `ESTIMATION_ONLY` percentile bootstrap.

`run` — the module's own pipeline entry point — is still a skeleton and raises
`NotImplementedError`. See `docs/ROADMAP.md` (M4a done; the `infer` subcommand is
wired in Task 18).
