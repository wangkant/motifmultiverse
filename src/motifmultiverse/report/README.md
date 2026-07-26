# `report`

## The rule

Every number rendered carries its denominator, its baseline population, and its provenance.

## The failure that produced it

The same data supported both `replicates exactly` and `4x stronger, prediction falsified`, differing only in whether the baseline was the unselected universe or a residual subset. A bootstrap resolution floor was also printed as though it were a measured p-value.

## How to check it

The renderer refuses a figure with no denominator and no `baseline_population` field; the bias ledger is rendered from `docs/bias_ledger.tsv`.

---

Status: **skeleton**. The body raises `NotImplementedError`. See `docs/ROADMAP.md`.
