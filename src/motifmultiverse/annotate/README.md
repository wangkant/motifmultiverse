# `annotate`

## The rule

A database match is evidence about a label, never an identity; short or weakly supported motifs are flagged and reported alongside their chance expectation.

## The failure that produced it

In the reference implementation `annotate` had no stage at all -- family assignment was a hand-curated prefix dictionary embedded in an adjudication script, and a TomTom label was overridden by sequence in one special case with no record of the rule.

## How to check it

`guards.short_motif_flag`: PWM <= 6 bp, or TomTom q > 0.05, or seqlet count < 100 requires `low_confidence_annotation`. What it returned is written to `guard_outcomes.json` beside the candidate table (`motifmultiverse.guard_log`).

---

## Status: candidate retention is implemented; **family assignment was never specified**

The stage runs: it loads the registry, retains database-label candidates with their
evidence, flags weak ones, and writes its artifact and provenance. What it does *not* do
is decide a family, and that is not a coding gap. In the reference implementation
`annotate` **had no stage at all** — no design, no criterion, no artifact; family
assignment happened as a side effect inside an adjudication script.

Two different things can be missing, and they wait on different work:

| state | what it is waiting for |
|---|---|
| unimplemented (the specification *multiverse*, `docs/ROADMAP.md` M4) | **implementation** — the rule and its check are written |
| never specified (family assignment inside `annotate`) | **design** — what a family assignment must satisfy has not been decided, so there is nothing to implement yet |

> The per-module status is derived, not typed: `motifmultiverse --help` and
> `implementation_status.json` render it from the CLI dispatch table. An earlier version
> of this section listed seven modules as unimplemented when six of them had been
> implemented, which is the same class of stale hand-written claim that
> [`../../../src/motifmultiverse/status.py`](../status.py) exists to prevent — and this
> file ships inside the wheel, so the claim reached users.

Treating the second as the first is how a stage gets built to whatever the first caller
happened to need. So the open questions are written down here as questions, because a
blank module's danger is not the blank — it is that the first caller fills it in.

## The design questions a human has to answer

**1. What adjudicates family membership — matrix similarity, database annotation, or a
person?** The reference implementation used all three, interchangeably, with no record
of which decided any given case. These are not interchangeable: similarity is a
property of the representation, a database hit is evidence about a label, and a curator
is a decision. Whatever the answer, the *source* has to be a field
(`family_assignment_source` exists and is unpopulated), because a family assignment
whose basis is unrecoverable cannot be audited later.

**2. At what confidence may a database match become a `family_id`?** In the reference
implementation a match with **q = 1.000** — a *q* value that excludes nothing — acquired
a biological family name, and that variant was then retained in over 99% of peaks,
dominating its layer. `guards.short_motif_flag` flags such a motif today, but flagging
is not the same as deciding: the question is whether a flagged match may name a family
at all, and if so under what ceiling. Note the trap in the obvious answer — a threshold
on *q* is a threshold on a statistic that is itself a function of motif length
(`BA-15`), so "q < 0.05" silently favours long motifs.

**3. What does a curator override look like as a record?** `FP-04` requires an
override to be *visible as* an override and never dressed as the output of a
multi-evidence gate. `schema.Decision.KEEP_SEPARATE_CURATOR_OVERRIDE` with a required
`rationale` and `decided_by` covers the merge case. Family assignment has no
equivalent: there is no field for "a person overruled the database here, for this
reason". Until there is, an override and a computed assignment are indistinguishable
in the artifact — which is the same shape as the merge table in which every row was a
collapse.

**4. What happens when two analyses assign the same motif to different families?**
This surfaced while writing the first three and is the most pressing of the four.
Cross-model identity travels by `family_id` (`FP-03`), so disagreement is not an edge
case — it is the **normal state** before adjudication, and the whole point of having
an adjudication stage. Yet `MotifNode` holds exactly one `family_id`, so the normal
state is currently **unrepresentable**: the second assignment can only overwrite the
first, silently, and whichever stage runs last wins.

Nothing here should be inferred from that silence. Closing it probably needs a
`candidate_assignments` structure — one entry per proposing analysis, each with its
own `family_assignment_source` and confidence, with `family_id` becoming the
*adjudicated* result rather than the only slot. That is a design decision, not an
implementation one, so it is recorded as a known gap in
[`docs/DATA_MODEL.md`](../../../docs/DATA_MODEL.md) and **not implemented pending a
ruling**. Building it to whatever the first caller needs is precisely the failure
mode this whole README is about.

The *family-assignment* slot raises nothing and decides nothing: it is left unfilled
pending that ruling, rather than defaulted. See `docs/ROADMAP.md` and `FP-19` in
`docs/CONSTRAINTS.md`.

## A recorded limitation: a dropped backend is visible here and nowhere downstream

A backend that returns even one row naming a node outside the run is not trusted for
any of its rows. That is deliberate — output disagreeing with the registry cannot be
verified row by row — and the drop is recorded, not silent: the backend's entry in
`annotation_backend_logs.json` is `UNVERIFIED` and its `detail` names the row that
caused it, which the CLI prints.

The consequence is not recorded. `adjudicate` reads `annotation_candidates.parquet`
and nothing else, so a component whose family conflict lived in the dropped rows
adjudicates as a duplicate rather than refusing the merge, on annotation evidence it
cannot tell is partial. `tests/test_annotate.py::test_a_dropped_backend_names_its_reason_but_the_adjudicator_cannot_see_it`
pins that chain end to end.

It stays open because both repairs invent a rule this design has not decided.
Retaining the good rows contradicts the retention rule above. Deferring every
component whenever a backend is `UNVERIFIED` would defer whole runs at any site
without HOMER, because a backend that failed cannot say which nodes it *would* have
covered — so "the annotation evidence for this component is incomplete" has to be
specified as an adjudication input, with its own criterion, before it can be one.
That is design question 5, and it waits on the same ruling as the other four.
