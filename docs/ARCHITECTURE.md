# Architecture

Seven layers, nine modules. The two architecture-level constraints in §3 are not
implementation preferences — they change what the tool is allowed to compute.

## Seven layers

| # | layer | main input | main output |
|---|---|---|---|
| 1 | Input and provenance | TF-MoDISco HDF5, contribution and one-hot arrays, region BEDs, project config | motif registry with per-input checksums, software versions, seed, timestamp |
| 2 | Evidence graph | registry | typed nodes and edges: alignment, sequence hit, attribution hit, downstream sensitivity, external biology, decision |
| 3 | Ontology and adjudication | evidence graph | `family_id` / `variant_id` assignments, merge decisions **including refusals**, human review file |
| 4 | Lexicon compilation | adjudicated decisions | tiered lexicons (`core`, `expanded`, sensitivity variants), content-addressed |
| 5 | Downstream stability | lexicons + one frozen hit substrate | held-out coverage, instance calling, paired reconstruction deltas |
| 6 | Robust statistics | stability outputs | interaction estimates with block-valid uncertainty, specification curves |
| 7 | Audit report | everything above | HTML / DOCX with the bias ledger, provenance and every denominator rendered |

## Nine modules

| module | responsibility |
|---|---|
| **A. ingest** | normalise heterogeneous discovery outputs; attach provenance; establish namespaces — **implemented** |
| **B. align** | pairwise registration and similarity, with a persisted per-pair null — **implemented** |
| **C. annotate** | retain competing database-label candidates and low-confidence flags — **implemented** |
| **D. adjudicate** | merge / refuse / deferred decisions with rationale and decider; emit human review — **implemented** |
| **E. compile** | build tiered lexicons; separate discovery support from analysis admission — **implemented** |
| **F. validate** | downstream stability: affected-subset reconstruction and backend verification — **implemented** |
| **G. infer** | effect estimates for ONE specification over a frozen substrate — **implemented**; the specification *multiverse* is not |
| **H. report** | render the audit trail, denominators and bias ledger |
| **I. interpret** | describe what is inside a cluster (descriptive, not a test) — **implemented** |

Each module directory carries a README with its rule, the failure that produced
the rule, and how the rule is checked. Eight of the nine are implemented; `report`
still raises `NotImplementedError`. The current status is generated from the CLI
dispatch table rather than restated here — see the table in `README.md` and
`implementation_status.json`, because this paragraph has been wrong twice.

**The evidence middle now runs.** `ingest` → `align` → `annotate` →
`adjudicate` → `compile` is a real path from discovery output to a frozen,
content-addressed lexicon. Annotation retains candidates rather than silently
assigning identity, and adjudication defers when the frozen design does not state
a required threshold; implemented does not mean an unknown scientific criterion
was filled with a plausible number.

### 2.1 A loader contract, discovered by reading the loader

The hit caller walks its pattern groups in a fixed order — positives, then negatives
— and sorts within a group by the **integer suffix** of the pattern name. `compile`
therefore assigns names so that the loader's own rule reproduces the manifest's
order, and `guards.index_order_matches_loader` compares the two **by name**.

Note the asymmetry with `ingest`, which reads patterns in the file's key order and
does **not** try to reproduce loader order: doing so would mean parsing digits out of
a name, which is forbidden (`BA-11`). Assigning names is a construction; parsing them
is an inference. Only one of the two is allowed.

## 3. Two architecture-level constraints

### 3.1 Single-scale hit substrate

**All specifications are computed as subsets of one frozen full-universe run. The
hit caller is never re-run per specification.**

This exists because the hit caller is **not input-scale invariant**: the same
regions produce different discrete retention decisions depending on which *other*
regions share the input. In the reference implementation the onset of divergence
was bracketed to under 10% growth on the base set (see `CONCEPT.md`). Re-calling
per specification would confound the specification axis with the caller.

Consequences, stated because they are expensive:

- `input_scale` is a provenance field on **every** result, and `guards.single_scale`
  rejects a set of results that spans more than one scale.
- **The peak universe must be frozen before the specification set is frozen.**
- **Adding peaks invalidates every specification.** There is no incremental
  top-up: a new universe means re-running all of them.

### 3.2 `interpret` and `infer` are two consumers, not two names for one thing

They share the statistical machinery and have **separate entry points**:

- **`infer`** answers *"how robust is this conclusion to analysis choices?"* — a
  preregistered question over a specification multiverse. **Today it estimates ONE
  specification** with `FP-15`'s estimators and emits `effect_estimates.tsv`; the
  axis sweep, and the dropped cells with reasons, are still M4. It runs
  `interpret`'s code rather than a second copy of the statistics.
- **`interpret`** answers *"what is in this cluster?"* — descriptive.

Neither may trigger a caller re-run. Keeping them separate is what stops an
exploratory listing and a preregistered test from becoming indistinguishable in
the record, which is what happens when both run through the same ad-hoc script.

## 4. Stages that were weak by inheritance, and what closed each gap

Two stages arrived weak, in different ways. Both are now implemented; the history
stays here because what each one was is why each one is shaped as it is.

- **`align`** was **prose-only** in the reference implementation. Its key artifact,
  a per-pair null p-value, was never persisted, so the constraint had no
  executable check. Worse, the aligner maximised *signed* similarity, which makes
  it structurally blind to sign-flipped motifs. Both gaps are closed:
  registration is chosen on **unsigned PPM** content and signed CWM similarity is
  measured only at the winning registration (`guards.sign_alignment`), and each
  pair's null re-runs the whole registration on freshly shuffled data rather than
  rescoring the observed offset. `alignment_edges.parquet` carries the null
  shuffle count and seed on every edge.
- **`annotate`** had **no stage at all**. Family assignment lived as a hand-curated
  prefix dictionary inside an adjudication script, with at least one label
  overridden by sequence in a special case that left no record of the rule. It was
  **not unimplemented — it was never specified**, which is a different position on
  the roadmap. The design it now has is deliberately minimal: backends *propose*
  candidates and nothing assigns identity, so two backends disagreeing produce two
  rows rather than one overwriting the other, and `MotifNode.family_id` stays
  unset until adjudication decides.

What has **not** closed: no optional annotation backend (TomTom, HOMER) is
verified in CI, so `annotation_candidates.parquet` is legitimately empty there.
That is recorded as `UNVERIFIED` per backend in `implementation_status.json`
rather than left to a green check mark.

## 5. What this tool does not do

It consumes discovery output; it does not compute attributions, discover motifs,
or re-implement a hit caller. Cross-model raw CWM averaging is a **design
prohibition**, not a missing feature (`guards.no_cross_model_cwm_avg` states it;
nothing in this release combines CWMs at all, so the guard has **no call site** and
the prohibition is structural rather than checked), because the
CWM belongs to a specific model and readout while the ontology is what crosses
them. And the default annotation output is family-level identity with a
confidence — mapping a motif to a specific protein is not promised.
