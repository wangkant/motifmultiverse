# Constraints

The **twenty-five frozen design principles** (`FP-01` … `FP-25`), each labelled by how
far enforcement in *this repository* actually goes (T-18). The machine-readable twin is
[`constraints.tsv`](constraints.tsv), which carries the principle text verbatim.

> **Provenance.** The principles are **transcribed from design report v0.8**; the table
> was supplied as a TSV and the `frozen_principle` text is verbatim (Chinese). The
> English wording below is a **gloss for readers of this repository**, not report text —
> where gloss and TSV differ, the TSV is right. The `enforcement`, `enforced_by` and
> `criterion_draft` columns are **this repository's own annotation and are not in the
> report**.

**Status tally: `ENFORCED` 4 · `PARTIAL` 14 · `DOC_ONLY` 7, of 25.**

## The annotation rule

A row is labelled by this rule and no other, so that a future contributor can label
a new row without asking anyone:

> **`ENFORCED`** — every clause that names a property of an artifact fails a guard or
> a schema check when violated, **and** a test proves that check can fail.
> **`PARTIAL`** — at least one such clause is mechanised and at least one is not; the
> `enforced_by` column says which is which.
> **`DOC_ONLY`** — no clause is mechanised. A `criterion_draft` is written anyway:
> **the criterion is written, the check is not implemented.**

Clauses that are guidance about how to *interpret* a result, rather than properties
of an artifact, do not affect the label; they are noted in `enforced_by` instead.

**A prerequisite is not the check.** A row does not move up because something it
needs now exists. `FP-10` is the standing example: `ingest` records *which* of the
three metacluster absences occurred, which that row's criterion needs as an **input**
— but the criterion is about what an absence *assertion* must carry (the subset's
peak count, the per-peak rate, or the literal `UNEXPLAINED`), and nothing checks
that. Promoting it because a prerequisite landed is exactly the error this table
exists to prevent, so it stays `DOC_ONLY`.

## Why this tally is not comparable to the earlier 20/3/3

Before the design report's tables were supplied, this file carried a list of
twenty-six principles labelled `ENFORCED` 20 / `PARTIAL` 3 / `DOC_ONLY` 3. **The two
tallies are not two counts of the same thing**, and reading the drop from 20 to 4 as
a regression in the code would be exactly backwards.

The old list was reverse-engineered *from the code that already existed*, so it was
almost bound to come out green: every entry was written by looking at a guard and
describing it. These twenty-five are the **design's** principles for the whole
analysis protocol — discovery, alignment, adjudication, instance calling, statistics,
governance — most of which lives in modules that are still skeletons. A list derived
from the implementation measures the implementation against itself; a list derived
from the design measures how much of the design exists.

So the number is a **roadmap metric**. It should rise as `align`, `adjudicate`,
`validate`, `infer` and `report` are implemented, and each rise should come with a
guard and a falsification test rather than with a relabelling. It went 4/10/11 →
4/13/8 in one round by implementing `ingest` and `compile`; that is what progress
looks like here.

## The twenty-five principles

| principle_id | requirement (gloss; verbatim text in `constraints.tsv`) | enforcement | enforced_by | criterion_draft |
|---|---|---|---|---|
| `FP-01` | Promoter and distal each get their own local TF-MoDISco run; a genome-wide catalog is not a substitute. | `PARTIAL` | `ingest` refuses an analysis that declares no `context` or whose discovery file is absent, and checksums every discovery HDF5 it reads; the catalog-substitution clause has no field and no check | *(remaining clause)* For every (model, readout, sequence_context) declared in the project config, the registry contains ≥1 discovery run whose provenance names that context and carries its own input-region checksum, and no admitted node has a catalog scan as its only support. **FAIL** if a declared context has no discovery run, or if an admitted node's sole support is a catalog scan. |
| `FP-02` | Merge promoter+distal *within* a model/readout; never average CWMs across models. | `PARTIAL` | `guards.no_cross_model_cwm_avg` enforces the prohibition; the within-analysis union is `compile/`, a skeleton | — |
| `FP-03` | Cross-model identity travels by `family_id`/`variant_id`; a specific TF label is recorded separately with a confidence. | `ENFORCED` | `schema.MotifNode` keeps the ontology separate from `putative_tf_label` and rejects a confidence outside [0,1]; `guards.variant_id_unique` | — |
| `FP-04` | Only complete duplicates collapse; an annotation-based exception is a recorded manual override, never dressed as a multi-evidence gate. | `PARTIAL` | `schema.Decision.KEEP_SEPARATE_CURATOR_OVERRIDE` + required `rationale`/`decided_by` make the override recordable and distinguishable from a gate product; the duplicate criterion is `adjudicate/`, a skeleton | — |
| `FP-05` | The representative is an observed medoid; no default CWM averaging; no unrestricted single linkage. | `PARTIAL` | `compile` refuses a collapse whose representative is not one of its own members, which is what "observed medoid, never a constructed average" means at the artifact level; the linkage clause has no check, and a *within*-model averaged representative still passes `no_cross_model_cwm_avg` | *(remaining clause)* Every collapsed cluster names a representative that appears in its own member list; no representative is produced by an averaging operation; every clustering step records its linkage, and single linkage is admissible only with a declared distance ceiling. **FAIL** on a non-member representative, an averaged representative, or unrestricted single linkage. |
| `FP-06` | `core` is the conservative specification, `expanded` the sensitivity one; a moderate merge auto-generates a split lexicon; discovery and analysis tier are separate fields; duplicate support sums only within one model/readout/metacluster. | `PARTIAL` | `schema.MotifNode` carries both tiers and requires `tier_reason` when they differ; `compile` auto-generates the split lexicon from three named triggers (`merge_confidence != HIGH`, `family_ambiguity`, `threshold_sensitive`) and records which fired. **What makes a merge moderate is undecided** — see `FP-13` — so `compile` dispatches on a declared grade and never assigns one. The summation scope is unchecked | — |
| `FP-07` | Attribution equivalence, real loader and matched-peak pilot all precede the formal instance-calling run. | `DOC_ONLY` | `validate/` records backend verification but has no orchestrator that orders or checksums the three prerequisites | An instance-calling run is admissible only if its provenance names three prerequisite artifacts by checksum — attribution equivalence, loader identity, matched-peak pilot — each timestamped earlier than the run. **FAIL** if any is absent, unchecksummed, or later than the run it licenses. |
| `FP-08` | Positional novelty is not coverage evidence; a redundancy claim needs both coefficient share and reconstruction gain. | `DOC_ONLY` | `validate/` emits an affected coefficient share and reconstruction delta, but does not yet validate a redundancy claim's estimation methods or prohibit positional novelty standing alone | A collapse justified as redundancy carries both a coefficient-share field and a reconstruction-gain field, each naming its estimation method; a positional-novelty field may never be the only evidence present. **FAIL** if either is absent or if positional novelty stands alone. |
| `FP-09` | Registration is established on unsigned sequence content; signed similarity is a separate statistic; sign separation holds within one readout only. | `PARTIAL` | `guards.sign_alignment` rejects signed registration and signed similarity choosing the offset; the cross-readout clause is unchecked | — |
| `FP-10` | Metacluster formation carries an absolute count gate; report per-peak rates, check subset size, and record `UNEXPLAINED` until the ultimate cause is proven. | `DOC_ONLY` | nothing checks the assertion. `ingest` now records **which** of the three absences occurred (`schema.MetaclusterState`), which the criterion below needs as an input but is not itself the check; `guards.single_scale` governs the *hit substrate*, a different instrument | Any assertion that a motif class is absent from a subset carries the subset's peak count, the per-peak rate in the comparison subset, and either a rate-difference result or the literal label `UNEXPLAINED`. **FAIL** if absence is asserted from a presence/absence field alone, or if a rate difference is attributed to peak count without a measured rate. |
| `FP-11` | Family effects depend on lexicon composition; aggregation safety is judged by the dominant variant's coefficient share; every family-level number states its lexicon version. | `PARTIAL` | `schema.LexiconManifest` requires a `lexicon_content_hash` and `compile` cannot emit a lexicon without one, so the thing a family-level number must cite now exists; nothing yet checks that a downstream number cites it, and the coefficient-share clause needs `validate` | *(remaining clause)* Every family-level quantitative field carries the lexicon identifier (content hash) it was computed under, and every family-level aggregation carries the dominant variant's coefficient share. **FAIL** if a family-level number is emitted without a lexicon identifier, or if aggregation safety is asserted without the share field. |
| `FP-12` | Compute the affected peak count before deciding; an estimate below the floor, or an interval containing both zero and the reference, is `NOT_ESTIMABLE_UNDERPOWERED` with no direction. | `ENFORCED` | `guards.estimability_floor` | — |
| `FP-13` | Merge/split rule parameters are written and checksummed before the pair's result is seen, and must be executable without judgement. | `DOC_ONLY` | nothing; `GOVERNANCE.md` states the legality test but there is no preregistration store. **Recorded design gap:** what evidence earns each `MergeConfidence` grade is undecided, and `schema.MERGE_CONFIDENCE_CRITERIA` holds `CRITERION_NOT_YET_DEFINED` for all three rather than a plausible fill. The alternative was a numeric cut-off, which would have invented a scalar the design does not contain — and a tier-membership threshold is exactly the parameter this principle says must be written and checksummed **by the design**, before results are seen, not chosen by an implementer | Every merge/split rule parameter set is stored with a checksum whose recorded timestamp precedes the earliest timestamp of any evidence artifact it reads, and every rule clause is an executable predicate over named fields with an explicit numeric threshold. **FAIL** if the checksum is missing or later than the evidence, or if any clause carries no magnitude threshold. |
| `FP-14` | The merged-vs-split Δ is computed on the affected subset with its size reported; a full-set Δ is reference only, labelled diluted. | `PARTIAL` | `validate.evaluate_stability` computes both deltas from strict frozen hit-table identity/coefficient/reconstruction columns, assigns decisions from the affected subset, and refuses interval/equivalence claims below 30 affected peaks. Adapter-specific biological reconstruction methods remain project configuration. | Every merged-vs-split delta carries the count and identifier of the affected subset it was computed over; a full-set delta carries an explicit dilution flag and may not be emitted without its affected-subset twin. **FAIL** on a delta with no affected-subset identifier, or a full-set delta reported alone. |
| `FP-15` | Peak-level primary; positive and negative separate; effect per model then meta-analysis; BCa paired block bootstrap for intervals, wild cluster bootstrap-*t* for *p*; block size, B and seed saved with the result. | `PARTIAL` | `interpret` aggregates to peak level, resamples whole genomic blocks, and stores block size, B and seed beside every interval; `AnalysisConfig` refuses heterogeneity below three models. Both specified estimators exist (`infer.bca_paired_block_interval`, `infer.wild_cluster_bootstrap_t`) and run together under `interpret --estimator bca-wild-cluster`, licensed `INTERVAL_AND_TEST`; label permutation stays abandoned. **The default estimator is still the percentile block bootstrap, which withholds *p* and *q*, and the cross-model effect-then-meta-analysis half of `FP-15` is not implemented** | — |
| `FP-16` | Co-occurrence and spacing are a screen; a pair claim needs in-silico ablation. | `DOC_ONLY` | nothing; in-silico ablation is out of scope for this tool, so only the labelling half is checkable here | Every pair-level claim carries an evidence kind of either `screen` or `ablation`; a screen-level claim may not carry an interaction estimate or use interaction wording. **FAIL** if a screen-level claim is phrased as an interaction. |
| `FP-17` | Every specification is a subset of one full-universe run; input scale travels as provenance; the peak universe is frozen before the specification set is. | `PARTIAL` | `guards.single_scale` + `provenance.ProvenanceRecord.input_scale`; the freeze-ordering clause needs an orchestrator and is unchecked | — |
| `FP-18` | Rerun invariance and batch-scale invariance are measured separately; discrete identity and numeric tolerance are gated separately; neither predicate may be derived from the other, nor written as `== 0` unmeasured. | `DOC_ONLY` | nothing; the two-axis rule is stated in `guards/` and `LESSONS.md`, but this repository runs no determinism ladder | The determinism report carries two independently measured values — a discrete identity count across reruns and a numeric displacement across reruns — each with its own gate, and neither gate's value is derived from the other's measurement. **FAIL** if either axis has a gate but no measurement record, or if a predicate is written as a constant without one. |
| `FP-19` | Length ≤ 6 bp, TomTom *q* > 0.05 or seqlet count < 100 ⇒ flagged, shown only by `variant_id`, reported beside its chance expectation; a single-family layer's share is `NOT_ESTIMABLE`. | `PARTIAL` | `guards.short_motif_flag` and `guards.single_family_layer`; the two reporting clauses (variant_id-only display, chance expectation alongside) are unchecked | — |
| `FP-20` | Every peak-set analysis declares its selection source, which determines the output mode; a missing field takes the most conservative mode; block splitting precedes clustering and assigns whole blocks. | `PARTIAL` | `guards.selection_provenance_declared` + the `interpret` dispatch table. **"Most conservative" is read here as `DESCRIPTIVE_ONLY_UNVERIFIABLE_CONDITIONING`, not `EYEBALLED`**: the report's §9.8 table names `EYEBALLED` as the default for a missing field, but it was written before `MODEL_SELECTED_NO_TRANSCRIPT` existed, and once a stricter grade exists the old default is no longer the most conservative one. The two differ only in the note that the conditioning set cannot be verified — which is true of an undeclared selection *a fortiori*, since an undeclared one is the least verifiable of all. The block-split-before-clustering and feature-disjointness clauses are unchecked | — |
| `FP-21` | Specificity is licensed by an interaction interval excluding zero, never by a difference of two significance statuses. | `ENFORCED` | `guards.interaction_required` | — |
| `FP-22` | Four-state missingness needs an executable assertion, not a line in a principles table; undefined values take an explicit sentinel; coverage is computed before any fill. | `ENFORCED` | `schema.Missingness` + `schema.MISSING_SENTINEL` + `guards.four_state_missingness` | — |
| `FP-23` | A stop condition halts, reports at the top, and waits; work done under a halt is labelled; a sealed criterion may only be changed if the change could have been written without knowing the result. | `DOC_ONLY` | nothing; `GOVERNANCE.md` states the protocol, but enforcing it needs an orchestrator that can see artifact timestamps across steps | For every halt record, no artifact of a later step carries a timestamp between the halt and the recorded human decision; every conclusion emitted in that window carries `CHARACTERISATION_UNDER_HALT`; and every edit to a sealed criterion carries a written justification that the change could have been written without knowing the result. **FAIL** on any later-step artifact inside the window, or on an unlabelled conclusion. |
| `FP-24` | A stratifying variable entering an interaction uses one definition for all units; a different definition is a named sensitivity check; the cross-tabulation comes before any effect is visible. | `PARTIAL` | `guards.stratum_parity` for the one-rule clause; the ordering clause is unchecked (`guards.health_before_effect` covers the health numbers, not the stratum table) | — |
| `FP-25` | Every guard ships a failure proof on a shifted or permuted mapping; a guard that never fails counts as unverified; guards may not read their own output or hard-code upstream row counts. | `PARTIAL` | `tests/test_guards.py::test_every_guard_has_a_falsification_test` walks `guards.ALL_GUARDS`; the two authoring clauses are reviewed by hand, not checked | — |

## Why the drafts exist, and what they must satisfy

Every `DOC_ONLY` row above carries a `criterion_draft`, and so does every row promoted
to `PARTIAL` whose remaining clause is still unchecked — marked *(remaining clause)*.
**The criterion is written; the check is not implemented.** That distinction is the
whole point of the column.

The danger of a `DOC_ONLY` constraint is not that it is unimplemented — it is that
*nobody knows what implementing it would check*. In the reference implementation the
alignment stage sat in a principles list, prose-only, for more than a year; because its
criterion was never written down, there is still no answer to what it should have
checked, and that gap is exactly where the sign-blindness failure lived. Criterion
first, implementation second.

Each draft is written to satisfy three rules:

1. **Decidable without looking at results.** It reads structure — presence of fields,
   ordering of timestamps, membership — never whether a number came out favourably.
2. **No self-reference.** A criterion never reads the artifact its own check produces
   (`FP-25`).
3. **No hard-coded upstream counts.** Where a count matters it is read from the upstream
   artifact and compared, never pinned in the criterion (`FP-25`).

None of the eleven drafted this way had to be recorded `NOT_YET_DECIDABLE`.

## Guards that enforce a bias-ledger control rather than a frozen principle

Four of the fifteen guards have no `FP-` home, because the rule they enforce lives in
the bias ledger:

| guard | enforces |
|---|---|
| `guards.no_key_parsing` | `BA-11` — never parse semantics out of a key string |
| `guards.index_order_matches_loader` | `BA-11` — a frozen index is compared to the loader **by name**; a positional read is the same defect wearing a different hat |
| `guards.comparator_declared` | `BA-18` — a comparator carries its basis and source |
| `guards.health_before_effect` | `BA-16` / `BA-20` — health numbers precede any effect, and a floor failure suppresses the reading |

`schema.AnalysisConfig.assert_between_model_heterogeneity_estimable` (N ≥ 3) is likewise
this repository's own constraint, tightening `BA-05`.

## Reconciliation with this repository's earlier reconstruction

The `A01`…`A26` list this file previously carried was reconstructed before the report
tables were supplied. It is superseded. Nothing in it was dropped without a home: each
former entry is either a clause of one of the twenty-five principles above, a
bias-ledger control (the three guards in the table just above), or the N ≥ 3 constraint.
The count itself was wrong — twenty-six was a miscount that included the header row.
