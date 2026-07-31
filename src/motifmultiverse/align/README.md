# `align`

## The rule

Registration is established on UNSIGNED PPM similarity; signed similarity is a separate statistic, never the thing being maximised.

## The failure that produced it

An aligner maximising signed CWM cosine is structurally blind to a sign-flipped motif: at the true registration the signed cosine is near -1, so that offset can never win. Run that way, a sign survey returned no flips -- a false negative manufactured entirely by the instrument.

## How to check it

`guards.sign_alignment` rejects any alignment whose registration used signed similarity. Per-pair null p-values must be persisted, not asserted in prose.

## Recorded limitations

**An edge is emitted for every pair that clears the overlap floor**, whatever
its similarity and whatever its null says. Suppressing the unremarkable ones
would mean declaring a similarity or p-value ceiling, and `docs/CONSTRAINTS.md`
reserves that parameter to the design (`FP-13`) — it is the open half of `FP-05`
("single linkage is admissible only with a declared distance ceiling"). So the
p-value is recorded on the edge and `adjudicate`'s criterion, not the graph, is
what may gate a collapse. Consequence to expect: connectivity proposes large
components, because registrability rather than evidence is what joins nodes.

**The null costs one full registration per shuffle per pair**, and the pair
count is quadratic in the registry — single-threaded, with no progress output.
Measured on 29 real ChromBPNet patterns (406 pairs), one registration of a
4–30bp core costs about 135 µs, so at the default 1000 shuffles a twelve-run,
240-node registry extrapolates to roughly half a CPU-hour; on the untrimmed
windows this stage used to register it was about ten times that. Nothing is
cached across pairs and nothing is rescored at a remembered offset, because
every cheaper null on offer answers the easier question (see the module
docstring). `tests/test_align.py` pins the registration count so a later
"optimisation" cannot quietly buy speed with a weaker null; parallelising across
pairs is the one lever that would not, and it is not taken here.

---

Status: **implemented**. `register_pair` searches offset x orientation on
unsigned PPM cosine under a bilateral overlap floor (`overlap_bp` and each
side's overlap fraction), then measures signed CWM similarity once, at that
registration only. `align_registry` trims every matrix to the node's declared
`trimmed_core` first, so the search never sees the near-uniform background
TF-MoDISco pads a fixed-width pattern window with. `calibrate_pair_null` re-runs
the full search on every
shuffle. `align_registry` (exported as `run`) registers every pair in a
registry, calibrates each pair's null, and writes `alignment_edges.parquet` +
`alignment_null_summary.tsv`, with `null_shuffles`/`seed`/`registered_on`/the
registration rule version carried on every edge. See `docs/ROADMAP.md` and
`tests/test_align.py`.
