# `adjudicate`

## The rule

A decision record must be able to express a REFUSAL, and its confidence must be a measure.

## The failure that produced it

The merge table's every row was `collapse`. Three refusals existed only as absent rows, indistinguishable from never-considered. `merge_confidence` was a per-family name lookup, and a downstream re-tiering gate really did read it.

## How to check it

`schema.DecisionRecord` requires a rationale and `decided_by` for every decision including refusals, and rejects a confidence outside [0, 1].

---

Status: **skeleton**. The body raises `NotImplementedError`. See `docs/ROADMAP.md`.
