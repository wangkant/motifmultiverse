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

### M4a — the estimators `FP-15` actually specifies — **DONE**

A named item, because the gap used to be carried in prose on every result.
`FP-15` specifies:

- **BCa paired block bootstrap** for intervals — bias-corrected and accelerated, so
  a skewed sampling distribution is not reported as a symmetric interval.
  `infer.bca_paired_block_interval`;
- **block-level wild cluster bootstrap-*t*** for *p* values.
  `infer.wild_cluster_bootstrap_t`;
- **label permutation stays abandoned.** It is not a fallback: under
  block-correlated structure it understates the variance.

Both now exist and `interpret --estimator bca-wild-cluster` runs them as one path,
licensed `INTERVAL_AND_TEST`. The **default is still `percentile`**, which is
`ESTIMATION_ONLY` and withholds `p_value` and `q_value` outright rather than
reporting the percentile bootstrap's replicate tail as one: a number that looks
like a *p* value but is not one is worse than no number. Every result carries
`inference_capability` (`schema.InferenceCapability`), and only a result licensed
`INTERVAL_AND_TEST` may carry a hypothesis test or a BH `q_value`.

*Done when:* both are implemented ✓, `schema.IMPLEMENTED_ESTIMATORS` contains all
three recognised values ✓, and a caller that branched on `estimator` needs no
change ✓ — which is why the enumeration shipped before the implementation.

*Still open:* the default estimator remains the conservative one. Switching it is a
separate decision with its own changelog entry, not a side effect of the
implementation landing.

### M4b — the report that renders these claims — **renderer done; the fields are not**

`report` is implemented (`src/motifmultiverse/report/`), and it was the last skeleton
body in the package. It renders one markdown document from an `interpretation.json`,
the `provenance.json` log beside it and `docs/bias_ledger.tsv`: every number is the
recorded field printed beside the denominator the producing stage recorded and named,
nothing is recomputed, and the bias ledger is rendered from the TSV the module's rule
names.

What it does **not** close is the rest of M4's *done when*. "Every claim states its
baseline population and lexicon version" is a claim about the *artifacts*, and they do
not yet carry it:

- no record at any nesting level carries `baseline_population`;
- `lexicon_id` is a declared string travelling on the hit rows, not
  `LexiconManifest.lexicon_content_hash` — which is the thing `FP-11` requires a
  family-level number to cite — and no field on `interpret.Interpretation` joins the
  result to a compile manifest;
- `selection_rule` and `selection_feature_names` are fields of `schema.PeakSetQuery`
  and are not emitted on `interpret.Interpretation`, so the executable rule that
  `PROGRAMMATIC_RULE` asserts, and the features that decide `SUBSTRATE_CIRCULAR`
  against `INTERNAL_DECOMPOSITION`, cannot be read back off a result;
- nothing in this package persists a `guards.GuardResult`, so no artifact records that
  a guard ran, let alone that it passed.

The report prints each of these as `NOT RECORDED` and names the field that would have
said it. *Done when:* the producing stages carry those fields — **not** when the
renderer stops mentioning them.

## M5 — cross-project generalisation and public release

*Done when:* the tool runs on a **second dataset** that shares no code path with
the reference implementation, a benchmark exists, and a container image is
published.

M5 is where the N>=3 ceiling is genuinely tested: with two models there is no
between-model variance to estimate, only sign consistency and
leave-one-model-out.
