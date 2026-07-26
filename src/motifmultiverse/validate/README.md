# `validate`

## The rule

Downstream stability validates a merge decision; a merge is never justified by similarity alone.

## The failure that produced it

A preregistered geometric rule failed to reproduce a split that the pipeline was nonetheless enforcing via a string special-case, and the downstream reconstruction agreed with the geometry -- median delta-NLL of exactly 0.0000. The split survived only as curation.

## How to check it

Paired reconstruction NLL on the affected subset, with the affected-subset size reported so that a null is distinguishable from a dilution.

---

Status: **skeleton**. The body raises `NotImplementedError`. See `docs/ROADMAP.md`.
