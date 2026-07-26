# `infer`

## The rule

Every specification is a subset of ONE frozen hit substrate, and a specificity claim needs an interaction interval excluding zero.

## The failure that produced it

The hit caller is not input-scale invariant: the same regions produced different discrete retention decisions depending on which other regions shared the input, with onset measured under 10% growth on the base set. Re-calling per specification would confound the specification with the caller.

## How to check it

`guards.single_scale`, `guards.interaction_required`, `guards.estimability_floor`, `guards.stratum_parity`.

---

Status: **skeleton**. The body raises `NotImplementedError`. See `docs/ROADMAP.md`.
