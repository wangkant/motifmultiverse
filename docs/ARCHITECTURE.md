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
| **B. align** | pairwise registration and similarity, with a persisted per-pair null |
| **C. annotate** | database matches, family assignment, confidence and low-confidence flagging |
| **D. adjudicate** | merge / refuse decisions with rationale and decider; emit human review |
| **E. compile** | build tiered lexicons; separate discovery support from analysis admission — **implemented** |
| **F. validate** | downstream stability: does the merge survive reconstruction? |
| **G. infer** | robust inference across a specification multiverse |
| **H. report** | render the audit trail, denominators and bias ledger |
| **I. interpret** | describe what is inside a cluster (descriptive, not a test) — **implemented** |

Each module directory carries a README with its rule, the failure that produced
the rule, and how the rule is checked. `ingest`, `compile` and `interpret` are
implemented; the other six raise `NotImplementedError`.

**What runs today is the two ends, not the middle.** `ingest` → `compile` is a real
path from discovery output to a frozen, content-addressed lexicon, and `interpret`
consumes a frozen hit table at the other end. Between them, `align`, `annotate` and
`adjudicate` do not exist, so what `compile` currently emits is undeduplicated
discovery output. Those three were left for later on purpose: their central criteria
depend on the two design questions that are still open (`annotate` was never
specified; `align`'s per-pair null was never persisted), and implementing them now
would mean inventing the answers.

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
  preregistered question over a specification multiverse.
- **`interpret`** answers *"what is in this cluster?"* — descriptive.

Neither may trigger a caller re-run. Keeping them separate is what stops an
exploratory listing and a preregistered test from becoming indistinguishable in
the record, which is what happens when both run through the same ad-hoc script.

## 4. Known weak stages, labelled rather than smoothed over

Two stages are weak **by inheritance**, and they are weak in different ways:

- **`align`** was **prose-only** in the reference implementation. Its key artifact,
  a per-pair null p-value, was never persisted, so the constraint had no
  executable check. Worse, the aligner maximised *signed* similarity, which makes
  it structurally blind to sign-flipped motifs. `guards.sign_alignment` closes the
  second gap; the persisted null is still to be built. It is **unimplemented**: the
  rule exists and the code does not.
- **`annotate`** had **no stage at all**. Family assignment lived as a hand-curated
  prefix dictionary inside an adjudication script, with at least one label
  overridden by sequence in a special case that left no record of the rule. It is
  **not unimplemented — it was never specified**, which is a different position on
  the roadmap: it is waiting for a design decision, not for code. See its README.

Both raise `NotImplementedError` here. Presenting either as complete would be the
same class of error that `CONSTRAINTS.md` exists to prevent — and treating the
second as the first is how a stage gets built to whatever its first caller happened
to need.

## 5. What this tool does not do

It consumes discovery output; it does not compute attributions, discover motifs,
or re-implement a hit caller. Cross-model raw CWM averaging is a **design
prohibition**, not a missing feature (`guards.no_cross_model_cwm_avg`), because the
CWM belongs to a specific model and readout while the ontology is what crosses
them. And the default annotation output is family-level identity with a
confidence — mapping a motif to a specific protein is not promised.
