# `align`

## The rule

Registration is established on UNSIGNED PPM similarity; signed similarity is a separate statistic, never the thing being maximised.

## The failure that produced it

An aligner maximising signed CWM cosine is structurally blind to a sign-flipped motif: at the true registration the signed cosine is near -1, so that offset can never win. Run that way, a sign survey returned no flips -- a false negative manufactured entirely by the instrument.

## How to check it

`guards.sign_alignment` rejects any alignment whose registration used signed similarity. Per-pair null p-values must be persisted, not asserted in prose.

---

Status: **skeleton**. The body raises `NotImplementedError`. See `docs/ROADMAP.md`.
