# Lessons: every constraint, indexed back to the failure that produced it

This page exists to answer one question: **why does this rule exist, and what
reopens if you remove it?**

It is not a summary of the reference implementation's science. Each entry names a
constraint in this repository, the specific failure that produced it, and — the
part that matters for a future contributor — **what the constraint currently
blocks**. A rule whose cost is visible and whose benefit is not gets deleted in
the first refactor that finds it inconvenient.

Figures below are from the reference line. This repository contains none of that
data.

---

## 1. Single-scale hit substrate

**The constraint.** Every specification is computed as a subset of *one* frozen
full-universe run. The hit caller is never re-run per specification. `input_scale`
travels as a provenance field on every result and `guards.single_scale` rejects a
result set that spans more than one. (`FP-17`, `BA-14`)

**The failure.** The hit caller is not input-scale invariant. The same regions
produced different **discrete** retention decisions depending on which *other*
regions shared the input. A bisection bracketed the onset to **(6,460, 7,085]
regions** on a 6,460-region base — a **9.67%** increase in scale is enough. On the
second cell line the onset was never localised; it is only known to lie somewhere
in **(13,277, 33,917]**.

**The two intervals are not equal evidence.** The first is the result of an actual
bisection: intermediate scales were run and the divergence was cornered between two
adjacent measured points. The second is merely the gap between the two scales that
happened to be run — the bisection was skipped for resource reasons and, by the
governing specification, no substitute scales were run either. Written side by side
they look alike; one is a measurement and the other is an absence of measurement.

**What it blocks.** Without it, a specification curve confounds the specification
axis with the instrument: two cells of the multiverse differ both in the analysis
choice under study and in which peaks the caller happened to retain, and nothing
in the output distinguishes the two. There is **no measured safe zone** — the
bracket is a threshold, not a gradient, and "my subsets are only a bit smaller" is
not an argument. Relaxing this means re-deriving the onset first, on your own
caller and your own data.

**Re-measured here, and the lesson changes shape.**
[`INPUT_SCALE_INVARIANCE.md`](INPUT_SCALE_INVARIANCE.md) re-ran the caller on K562
ALLPEAKS with one frozen lexicon. The instability reproduced; **the attribution to
scale did not.** Growing the input 9.67% and 100%, with each region left at its
original row index, changed 0 decisions out of 219,203 hits. Re-ordering the same
6,460 regions changed 876, and changing only `--batch-size` changed 1,557. The
caller's own `peaks_qc.tsv` shows why: those manipulations change how many solver
iterations a region receives (up to 497 steps' difference for the same region),
and growing the input does not. So the constraint above is *right* but its stated
reason is wrong — the danger is **re-calling at all**, not re-calling at a
different size, and "there is no measured safe zone" is if anything an
understatement. It also means `guards.single_scale`, which compares a region
*count*, cannot see the effect that actually exists.

**And the meta-lesson survives its own test.** The paragraph above about the two
intervals — one a measurement, one an absence of one — was the right instinct
aimed at the wrong target. Both intervals were reported in the same units and with
the same confidence, and neither was a scale threshold at all.

## 2. Two axes, two gates

**The constraint.** Rerun invariance and batch-scale invariance are measured
separately; discrete identity and numeric tolerance get separate gates; neither
predicate may be inferred from the other's measurement, nor written as `== 0`
without one. (`FP-18`)

**The failure.** Permuting the input order produced the **largest coefficient
displacement of any perturbation measured — 2.07× the median coefficient — and
exactly zero discrete flips.** The two axes moved independently, in opposite
directions, in the same run.

**Both numbers are contradicted by the re-measurement**
([`INPUT_SCALE_INVARIANCE.md`](INPUT_SCALE_INVARIANCE.md)): permuting order at
fixed scale moved the discrete axis by **876 flips** across 4.69% of regions, and
the largest displacement was **104×** the median coefficient. The constraint is
unaffected — separate gates are exactly what let the two be reported as different
answers here rather than one being read off the other — but this section's
illustration is no longer a case of "one axis moved and the other did not". A
better reading of the same evidence: the numeric axis is so noisy that it fires on
a run against itself, which is precisely why the discrete gate has to be separate.

**What it blocks.** A single "reproducible" verdict. Whichever axis you measure
will look clean and license the other, and which one you happen to measure decides
what you conclude. Collapsing the two gates back into one restores exactly that.

## 3. Missingness is four-state, and the encoding is an assertion

**The constraint.** `not_searched`, `no_sequence_match`, `hit_below_floor` and
`used` are distinct; an undefined value takes an explicit sentinel and never `0`;
coverage is computed **before** any fill. `schema.HitRecord` refuses a zero
standing in for an undefined value on every row that is read.
`guards.four_state_missingness` refuses one too -- and would also catch the
coverage figure itself -- but it has **no call site** in this release
(`guards.GUARDS_AWAITING_INPUT`): no artifact here claims a coverage
independently of the code that would recompute it, and a guard that checks a
claim against the code that produced it corroborates itself, which is this very
failure. (`FP-22`, `BA-10`, `BA-17`)

**The failure.** The four-state encoding had already been specified — in prose. A
`pivot_table(aggfunc="sum")` silently returned **`0.0`** for a group in which every
value was missing, and those zeros entered the arm means as real measurements. The
coverage figure, computed *after* that fill, reported **`1.000000`**. The error
produced its own corroboration.

**What it blocks.** The most dangerous class of defect in this domain: one that
makes the audit statistic look better. Writing the rule in a principles table does
nothing — that is precisely what had been done. Only the executable assertion
catches it, which is why `DOC_ONLY` in `CONSTRAINTS.md` is a status and not a
resting place.

**Where the assertion sits matters too.** `schema.HitRecord` **raises** when a row
whose missingness is anything but `used` carries `hit_coefficient = 0.0`. That is
one line, and it stops `BA-01` and `BA-10` at the same point: a peak that was never
searched cannot enter a table looking like a peak with no motif. The reference
implementation caught this class of error far downstream, in an arm mean — by which
point the zeros were indistinguishable from measurements.

The same reasoning produced `schema.MetaclusterState`: `group_absent`,
`group_empty` and `not_searched` are three claims, and only the first is evidence
about the discovery gate. Collapsed into "no negative motifs", all three become one
false sentence.

**A smaller repeat, from this repository.** `Mapping` was used in annotations in
`guards/` and `schema/` while never being imported. Because
`from __future__ import annotations` defers annotation evaluation, the missing name
is never looked up at runtime — the suite stayed green, and `ruff` found it in a
static pass rather than a test. A passing suite means the assertions that ran held;
it does not mean the code was executed. Same failure family as the `pivot_table`
zeros above, in its cheapest reproducible form: the defect supplied its own evidence
of correctness.

## 4. A guard must be proven capable of failing

**The constraint.** Every guard ships with a falsification test that makes it
fail, and `tests/test_guards.py::test_every_guard_has_a_falsification_test` walks
the guard registry and fails if any guard lacks one. (`FP-25`)

**The failure.** Five framework guards all reported passing. A later falsification
pass fed them a **row-shifted** and a **permuted** lexicon index: **two still
passed under both**, and **none of the five** detected a reordered index. In the
report, those two were indistinguishable from the three that worked.

**What it blocks.** Guard theatre — a green check that means only that nobody has
tried to break it. Note that the meta-test is the load-bearing part, not the
individual tests: without it, the next guard added is the next vacuous one, and
nothing in CI says so. It was not in any specification; it is here because the
failure above is cheap to repeat.

## 5. A decision must be able to express a refusal

**The constraint.** `schema.Decision` includes `REFUSE_MERGE`, and every decision
record requires a `rationale` and a `decided_by`. (`FP-04`)

**The failure.** In the reference implementation's merge table, **every row was a
collapse**. Three merges had in fact been considered and refused — but a refusal
was recorded by *not writing a row*, so in the artifact it was indistinguishable
from a pair that was never examined at all.

**What it blocks.** The silent erasure of negative decisions, which is what makes
an adjudication log unauditable: a reader cannot tell diligence from omission.
Related: a curator override is a `KEEP_SEPARATE_CURATOR_OVERRIDE` with a written
reason, never an unexplained gate output — `FP-04` requires the override to be
*visible as* an override.

## 6. Never parse semantics out of an identifier

**The constraint.** Identifiers are wrapped (`schema.NamespacedId`), translated
through explicit tables that raise on unknown keys, and never sliced.
`guards.no_key_parsing` is an AST check, and it is run over this repository's own
source in the test suite. (`BA-11`)

**The failure.** A FiNeMo **row number** was matched against a discovery manifest
**pattern id**. The join succeeded, silently. One factor's evidence was filed
under another's name — **CEBPG's evidence recorded against ETV4**.

**What it blocks.** Joins that cannot fail loudly. Two producers' identifier
spaces look alike, so a mis-join produces a plausible table rather than an error,
and the mistake surfaces — if ever — as a biological conclusion about the wrong
transcription factor.

**The positive form of this rule: assigning is a construction, parsing is an
inference.** The hit-caller loader orders patterns by the integer suffix of their
names, so a lexicon has to come back in a known order. There are two ways to get
one, and only one of them is allowed.

- `ingest` reads patterns in the file's own key order and **does not** try to
  reproduce loader order, because doing so would mean reading the digits out of
  `pattern_10` — a parse, and therefore a claim that the name means something.
- `compile` **assigns** names: patterns are numbered in the order they should come
  back, so the loader's own rule reproduces that order. Nothing is read out of a
  name; a name is written such that someone else's rule yields the answer.

The manifest then carries both the assigned `pattern_tag` and the source `node_id`,
so the mapping is a table rather than something to re-derive. Verified end to end
against the real loader on a real TF-MoDISco output, including across the 9 → 10
boundary where the two sort orders diverge.

That distinction is worth carrying beyond this repository. Parsing an identifier
asserts that its text encodes a fact; constructing one asserts only that you wrote
it. The first can be wrong about the world — `CBP_2048_...` said 2048 while the real
input width was 2114 — and the second cannot.

## 7. A comparator does not transfer

**The constraint.** Every cross-condition effect carries the identity of the peak
set it is measured against; `guards.comparator_declared` refuses a claim with no
baseline, and refuses to let one effect be reported against two. (`BA-18`)

**The failure.** One set of measurements supported both *"replicates exactly"* and
*"four times stronger, prediction falsified"*. Nothing about the data differed —
only whether the baseline was the unselected universe or a residual subset from
which the relevant peaks had already been removed.

**What it blocks.** The single most portable error in this domain. A frozen
lexicon transfers between projects; a comparator does not, and a number quoted
without its baseline is not weakly supported — it is uninterpretable.

## 8. Selection provenance decides the output mode

**The constraint.** A peak-set query declares how it was chosen, and that grade —
not the analyst — decides whether it may emit an interval. An undeclared query is
recorded as `DECLARATION_MISSING` and takes the most conservative mode.
`guards.selection_provenance_declared` enforces both halves. (`FP-20`, `BA-16`)

**The failure.** A clustering resolution was selected by an automated agent with
**no criterion retained**. What the agent could see when it chose cannot be
reconstructed, and in particular it cannot be shown that downstream information
was not already visible. This is why `MODEL_SELECTED_NO_TRANSCRIPT` is *stricter*
than `EYEBALLED`, which reads backwards until you notice that a human selector can
at least testify afterwards to what they looked at.

**What it blocks.** Circular inference that is statistically impeccable. If the
criterion that selected the peaks came from the signal being measured, the *p*
value is honest and the claim is empty. It also blocks the specific regression of
a permissive default: `dict.get(grade, EXTERNAL)` is how an undeclared query
quietly becomes a published effect.

## 9. Health numbers come before effects, and a floor suppresses the reading

**The constraint.** Intersection coverage, blocks spanned and the fraction the
frozen lexicon explains are computed before any effect. If one is below its
pre-registered floor, the reading is **suppressed**, not annotated.
`guards.health_before_effect` checks both the ordering and the suppression.
(`FP-12`, `FP-24`, `BA-20`)

**The failure.** The general one: a caveat attached to an effect size does not
travel with it. The number gets quoted; the sentence beside it does not. The
specific one: an interval wide enough to contain both zero and the reference
estimate was read as a direction, because a direction was what the surrounding
prose was about.

**What it blocks.** The disclaimer as a substitute for a decision. Adding "note:
low coverage" beside a number and shipping it is not a control; withholding the
number is. Also note *why* the block count is a health number at all: for a
clustered peak set the effective sample size is the number of genomic blocks, not
the number of peaks (`BA-07`), so a query over 40,000 peaks in four blocks has
n≈4.

## 10. Instances are not samples

**The constraint.** Analysis is peak-level, and the bootstrap resamples whole
genomic blocks. (`FP-15`, `BA-07`)

**The failure.** Multiple motif instances inside one peak, and multiple peaks
inside one genomic neighbourhood, are not independent draws. Treating them as such
narrows an interval by whatever factor the clustering happens to imply.

**What it blocks.** Confidence intervals that are narrow for arithmetic reasons.
This is also why `interpret` refuses to accept a peak list without being able to
place each peak in a block: without coordinates there is no unit of resampling,
only an assumption.

---

## What this repository still cannot check

Eleven of the twenty-five frozen principles are `DOC_ONLY`. Each now carries a
drafted criterion in [`CONSTRAINTS.md`](CONSTRAINTS.md), and the drafts are the
point: the danger of an unimplemented constraint is not that it is unimplemented,
it is that **nobody knows what implementing it would check.** In the reference
implementation the alignment stage sat in a principles list, prose-only, for more
than a year — and because its criterion was never written down, there is still no
answer to what it should have checked. That gap is where the sign-blindness
failure lived: an aligner maximising *signed* similarity is structurally blind to a
sign-flipped motif, so the survey it produced returned "no flips" as a property of
the instrument.

Criterion first, implementation second.
