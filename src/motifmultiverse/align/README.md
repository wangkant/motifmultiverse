# `align`

## The rule

Registration is established on UNSIGNED PPM similarity; signed similarity is a separate statistic, never the thing being maximised.

## The failure that produced it

An aligner maximising signed CWM cosine is structurally blind to a sign-flipped motif: at the true registration the signed cosine is near -1, so that offset can never win. Run that way, a sign survey returned no flips -- a false negative manufactured entirely by the instrument.

## How to check it

`guards.sign_alignment` rejects any alignment whose registration used signed similarity, and what it returned is written to `guard_outcomes.json` beside the edge table (`motifmultiverse.guard_log`) — an edge table used to carry no statement that the question had been asked. Per-pair null p-values must be persisted, not asserted in prose.

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
count is quadratic in the registry. Measured on 29 real ChromBPNet patterns (406
pairs, 194 of them registrable), one registration of a 4–30bp core costs about
128 µs; the whole stage at the default 1000 shuffles takes 24.9 s on that
registry single-threaded, so a twelve-run, 240-node registry extrapolates to
roughly half a CPU-hour, and on the untrimmed windows this stage used to
register it was about ten times that. Nothing is cached across pairs and nothing
is rescored at a remembered offset, because every cheaper null on offer answers
the easier question (see the module docstring). `tests/test_align.py` pins the
registration count so a later "optimisation" cannot quietly buy speed with a
weaker null.

**Parallelism across pairs is the one lever that does not weaken the null**, and
it is now taken: `--workers` / `workers=` runs the pair loop in that many
processes, defaulting to 1 so no existing invocation changes. It is admissible
only because it cannot reach the arithmetic — each pair's null generator is
built from the run seed alone inside `calibrate_pair_null`, nothing is carried
between pairs, and outcomes are reassembled by pair order rather than by finish
time. Measured on the same 29-pattern registry at 1000 shuffles, two runs of the
sweep: 24.9–25.3 s at 1 worker, 13.1–13.2 s at 2 (1.9×), 6.8–7.5 s at 4
(3.4–3.7×), 3.7 s at 8 (6.7–6.8×) — with
`alignment_edges.parquet` and `alignment_null_summary.tsv` byte-for-byte
identical at all four worker counts, and identical to what this stage wrote for
the same registry before the parameter existed. The equality is the point, not
the speed, so it is a test rather than a claim:
`test_align_registry_writes_byte_identical_tables_at_every_worker_count`
compares the written bytes, and
`test_align_null_is_a_pure_function_of_the_seed_and_the_pair` pins the mechanism
against an independent recomputation from the seed — introducing a generator
shared between pairs fails both.

Progress goes to **stderr**, never to stdout, which carries the counts and the
`written:` paths a caller parses. `align_registry` itself writes to no stream at
all: it calls a `progress(completed, total)` callback once per registrable pair,
and the CLI is what decides that those become stderr lines — at most one every
two seconds, with the first and last pair always printed, so a run that takes
hours says where it is and a run that takes a second still says what it did.

**RECORDED, not fixed: the null generator is seeded per run, not per pair.** Two
pairs whose targets have the same trimmed-core length therefore draw the same
sequence of row permutations, and their nulls are positively dependent — which
matters to any later procedure that treats these p-values as independent tests.
Seeding per pair would decorrelate them and would change every p-value this rule
version has produced; that is a decision about the null, not about scheduling,
so it is written down here rather than made under cover of a performance change.
Parallelism reproduces the existing correlated draws exactly.

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
