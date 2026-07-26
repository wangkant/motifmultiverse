# Roadmap

Milestones have **definitions of done**, not dates.

## M0 — peak-set queries over a frozen substrate  ← **done**

*Done when:* a subset query over one frozen hit table produces three health numbers
before any effect, dispatches its output mode from the declared selection provenance,
and suppresses the reading when a pre-registered floor fails. **`interpret` does this**
(`src/motifmultiverse/interpret/`). Remaining: BCa intervals and the wild cluster
bootstrap-*t* required by `FP-15`.

`annotate` now implements the deliberately narrow candidate-evidence contract:
database matches and confidence flags survive as competing rows for adjudication;
they do not silently assign a family.

## M1 — standardised ingest and provenance  ← **done**

*Done when:* a project config with N heterogeneous discovery outputs produces one
registry in which every node carries its checksum, software version, seed and
timestamp; the four schema rules are enforced; and a regression fixture round-trips.

**`ingest` does this.** Remaining within M1: `source_peak_count` has no source in a
TF-MoDISco file and stays unpopulated, and the regression fixtures are still download
recipes rather than data.

## M1b — lexicon compilation against a real loader  ← **done**

*Done when:* tiered lexicons are written in the order the hit caller emits, verified
by reading them back with the real loader, each content-addressed, and each declaring
what its tier contrast does and does not vary.

**`compile` does this**, with one honest gap: verification needs the `finemo`
backend, and when it is absent the tool says the round trip did not happen rather
than assuming it would have passed.

Note this milestone sits *before* M2 in effort and *after* it in meaning: compile
can consume the strict, identity-bearing adjudication handoff, while an explicitly
deferred relationship remains uncollapsed.

## M2 — evidence graph and lexicon compiler  ← **implemented; reference-fixture validation pending**

*Done when:* alignment, adjudication and compilation reproduce the reference
implementation's alignment audit, medoid selection and `core` / `expanded` / split
lexicons on the regression fixtures, with refusals recorded as refusals.

**`align`, candidate-only `annotate`, `adjudicate`, and `compile` implement this
path.** Alignment persists the full-search null; adjudication records refusals and
selects observed medoids from authoritative registry metadata. The remaining M2
work is validation against the external reference regression fixtures. Criteria
whose scientific magnitude was never frozen remain `DEFERRED`, by design.

## M3 — stability adapters (FIMO / HOMER / FiNeMo)

*Done when:* held-out coverage, instance calling and paired reconstruction run
through backend adapters that fail with a clear "backend not installed" message,
and the single-scale substrate constraint is enforced end to end.

## M4 — robust statistics and specification curve

*Done when:* interaction estimates carry block-valid uncertainty, the specification
curve reports dropped cells with reasons, and every claim states its baseline
population and lexicon version.

### M4a — the estimators `FP-15` actually specifies

A named item, because the gap is currently carried in prose on every result.
`interpret` ships a **percentile** block bootstrap. `FP-15` specifies:

- **BCa paired block bootstrap** for intervals — bias-corrected and accelerated, so
  a skewed sampling distribution is not reported as a symmetric interval;
- **block-level wild cluster bootstrap-*t*** for *p* values;
- **label permutation stays abandoned.** It is not a fallback: under
  block-correlated structure it understates the variance.

Until the wild cluster bootstrap-*t* exists, `interpret` withholds `p_value` and
`q_value` outright rather than reporting the percentile bootstrap's replicate tail
as one: every result carries `inference_capability` (`schema.InferenceCapability`),
today always `ESTIMATION_ONLY`, and only a result licensed `INTERVAL_AND_TEST` may
carry a hypothesis test. A number that looks like a *p* value but is not one is
worse than no number.

*Done when:* both are implemented, `schema.IMPLEMENTED_ESTIMATORS` contains all
three recognised values, and a caller that branched on `estimator` needs no change —
which is why the enumeration ships now rather than with the implementation.

## M5 — cross-project generalisation and public release

*Done when:* the tool runs on a **second dataset** that shares no code path with
the reference implementation, a benchmark exists, and a container image is
published.

M5 is where the N>=3 ceiling is genuinely tested: with two models there is no
between-model variance to estimate, only sign consistency and
leave-one-model-out.
