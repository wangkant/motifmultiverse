# `align`

## The rule

Registration is established on UNSIGNED PPM similarity; signed similarity is a separate statistic, never the thing being maximised.

## The failure that produced it

An aligner maximising signed CWM cosine is structurally blind to a sign-flipped motif: at the true registration the signed cosine is near -1, so that offset can never win. Run that way, a sign survey returned no flips -- a false negative manufactured entirely by the instrument.

## How to check it

`guards.sign_alignment` rejects any alignment whose registration used signed similarity. Per-pair null p-values must be persisted, not asserted in prose.

---

Status: **implemented**. `register_pair` searches offset x orientation on
unsigned PPM cosine under a bilateral overlap floor (`overlap_bp` and each
side's overlap fraction), then measures signed CWM similarity once, at that
registration only. `calibrate_pair_null` re-runs the full search on every
shuffle. `align_registry` (exported as `run`) registers every pair in a
registry, calibrates each pair's null, and writes `alignment_edges.parquet` +
`alignment_null_summary.tsv`, with `null_shuffles`/`seed`/`registered_on`/the
registration rule version carried on every edge. See `docs/ROADMAP.md` and
`tests/test_align.py`.
