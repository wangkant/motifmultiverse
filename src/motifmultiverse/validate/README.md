# `validate`

## The rule

Downstream stability validates a merge decision; a merge is never justified by similarity alone.

## The failure that produced it

A preregistered geometric rule failed to reproduce a split that the pipeline was nonetheless enforcing via a string special-case, and the downstream reconstruction agreed with the geometry -- median delta-NLL of exactly 0.0000. The split survived only as curation.

## How to check it

Paired reconstruction NLL on the affected subset, with the affected-subset size
reported so that a null is distinguishable from dilution. The all-peak median is
persisted only as a labelled diagnostic and never licenses an equivalence claim.
At fewer than 30 affected peaks the result is
`LOW_RISK_RARE_NOT_VALIDATED`: it carries a frequency-limited power statement and
no interval.

---

Status: **implemented**. `validate` accepts frozen standardized before/after hit
tables (`peak_id`, `hit_id`, `coefficient`, `reconstruction`) and writes
`stability_results.parquet` plus `backend_verification.tsv`. Both artifacts bind
the exact `PeakSplitManifest` and the Task 13 decision/validation split artifacts.
Optional unavailable backends are persisted as `UNVERIFIED`, never silently
converted to a verified result.
