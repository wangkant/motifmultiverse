# Bias ledger

**Twenty axes.** Each is a way a number in this domain can be right-looking and wrong.
The machine-readable twin is
[`src/motifmultiverse/report/bias_ledger.tsv`](../src/motifmultiverse/report/bias_ledger.tsv),
which the `report` module renders (T-14). Every axis is referenced by its `axis_id`
(`BA-01` … `BA-20`) so that this file and the table can be cross-checked line by line.

It lives **inside the package**, not in this directory, because `report` reads it at
run time and a wheel carries package data rather than the repository around it: while it
sat here, `motifmultiverse report interpretation/` refused on every non-editable install,
naming a file the distribution had never contained. There is exactly one copy — a second
one kept in `docs/` for convenience is the drift this file's "the TSV is authoritative"
rule exists to prevent, and `tests/test_packaging.py` fails if one appears.

> **Provenance.** These axes are **transcribed from design report v0.8**, which supplied
> the table as a TSV in Chinese; the TSV's `bias` / `mechanism` / `control` text has
> since been rendered into English, structure-preserving — same four columns, same
> twenty `axis_id`s, no axis added or dropped. It remains the authoritative twin: where
> the prose in the table below differs from the TSV, **the TSV is right**, and `report`
> renders the TSV rather than this file. The `enforced here` column is this repository's
> own annotation, is not in the TSV, and does not appear in the report.

## The twenty axes

| axis_id | bias | mechanism, in brief | control | enforced here |
|---|---|---|---|---|
| `BA-01` | Lexicon ascertainment | A motif absent from the local lexicon was never searched for, and its absence is read as zero contribution. | Shared union within a model/readout; `not-searched` encoded explicitly. | `schema.Missingness.NOT_SEARCHED` |
| `BA-02` | Redundancy | Near-duplicate CWMs are either double-counted or split one coefficient between them. | family/variant ontology; merged-vs-split sensitivity. | ontology fields plus split-bound affected-subset sensitivity in `validate/`; interpretation of redundancy remains criterion-dependent |
| `BA-03` | Alignment | Short overlaps and offset maximisation manufacture high similarity. | Two-sided overlap; null calibration; PPM and CWM on the same window. | — (the per-pair null is unbuilt; see `align/`) |
| `BA-04` | Annotation | The top database hit is read as a specific TF identity. | Keep family separate from putative TF; carry source and confidence fields. | `schema.MotifNode` (`putative_tf_label`, `family_assignment_confidence`) |
| `BA-05` | Model scale | Raw contribution units are not comparable between models. | Estimate the effect within each model; meta-analyse at a second level. | `AnalysisConfig` refuses heterogeneity below three models; `guards.no_cross_model_cwm_avg` runs at `compile`, over `compile.operations_log` -- a classification of each emitted lexicon against the registry arrays, so combination performed between the registry and the emitted file is checked on the bytes rather than resting on nothing here combining CWMs; combination performed upstream of `ingest`, which is where a meta-analysis across models would sit, is outside it and passes |
| `BA-06` | Selection | The same peaks serve as both discovery and validation set. | Held-out chromosome, cross-fitting, or external data. | `interpret` refuses inference on a `CLUSTERED_NO_SPLIT` query |
| `BA-07` | Pseudo-replication | Multiple instances in one peak are counted as independent samples. | Peak-level aggregation, or a cluster-robust bootstrap. | `interpret` aggregates to peak level and resamples whole blocks |
| `BA-08` | Threshold / specification | Lambda, FIMO *p*, trimming and lexicon tier each move the conclusion. | Multiverse / specification curve. | `infer/` is a skeleton |
| `BA-09` | Multiple testing | Family, variant, pair and method are tested at once. | Three layers that may not substitute for one another; the test count is fixed at freeze time; specification grids report a joint null, not per-cell BH. | `interpret` floors its bootstrap *p* at 1/(B+1) and reports B; hierarchical FDR is `infer/` |
| `BA-10` | Missingness | *No sequence*, *no hit*, *not searched* and *below floor* all collapse into zero. | Multi-state encoding and a two-part model. | `schema.Missingness`, checked on every row read; `guards.four_state_missingness` has **no call site** -- no artifact claims a coverage independently of the code that recomputes it |
| `BA-11` | Identifier namespace | An index that is only meaningful inside one namespace is used across a boundary, filing evidence against the wrong motif or peak. | Explicit translation tables that raise on unknown keys; never parse semantics out of a key string; assert on the identity of the object retrieved. | `schema.NamespacedId` / `translate` + `guards.no_key_parsing` |
| `BA-12` | Cluster-size | Metacluster formation carries an absolute seqlet-count gate, so composition changes with subset size; whether the per-peak *rate* also differs must be measured, not assumed. | Report per-peak rates, not presence/absence; check subset size before comparing composition; record `UNEXPLAINED` until the ultimate cause is proven. | — (see `FP-10`, `DOC_ONLY`, criterion drafted) |
| `BA-13` | Attribution sign | The same element can land in opposite metaclusters at different readout widths: sign is a property of the model×readout pair, not necessarily of the element. | Positive/negative separation holds within one readout; verify explicitly before comparing signs across readouts. | `guards.sign_alignment` covers registration; the cross-readout clause is unchecked |
| `BA-14` | Instrument context dependence | The hit caller's discrete output moves with input scale: the same peak can be retained in one run and dropped in another that differs only in which other peaks were present. | One full-universe run shared by every specification; never re-run the caller between specifications; carry the input scale as provenance. | `guards.single_scale` + `provenance.ProvenanceRecord.input_scale` |
| `BA-15` | Annotation false salience | A short motif picks up a database match and a biological family name at chance level, then dominates a layer through a very high retention rate. | Three flag rules (length, *q*, seqlet count); chance expectation reported beside the observed rate; flagged variants appear only by `variant_id`. | `guards.short_motif_flag` (the two reporting clauses are unchecked) |
| `BA-16` | Selection provenance | If the criterion that selected a peak set came from the same signal being measured, the inference can be statistically valid and semantically circular. | Provenance grade determines the output mode; attribution-derived clustering is `SUBSTRATE_CIRCULAR`; block-level splitting, done before clustering. | `guards.selection_provenance_declared` + the `interpret` dispatch table |
| `BA-17` | Missingness coercion | A library default (a pivot aggregation) silently turns an all-missing group into zero, bypassing a four-state encoding that was already written down. | Explicit sentinel for undefined values; an executable assertion against zero-filling; coverage computed before any fill. | `schema.MISSING_SENTINEL` and `interpret._coerce_row` keep an absent value out of the numbers; `guards.four_state_missingness` is the executable assertion the axis asks for and has **no call site** -- see `guards.GUARDS_AWAITING_INPUT` for the claim it is waiting for |
| `BA-18` | Comparator defect propagation | When a previous result is used as a gate or a baseline, that result's own defect becomes this decision's bias. | Label the comparator's basis and source; cite the source table's path and column, never a rounded prose interval. | `guards.comparator_declared` |
| `BA-19` | Stratum definition parity | One stratifying variable is generated by different rules in different subsets, so the interaction term absorbs a definitional difference. | One definition for all cells in the main analysis; any second definition is a separately named sensitivity check; produce the cross-tabulation before looking at any effect. | `guards.stratum_parity` states the one-rule clause and has **no call site** (nothing here emits strata); the ordering clause is unchecked either way |
| `BA-20` | Estimability floor | Below a sample-size or event-count floor, an interval wide enough to be uninformative gets written up as a direction. | Pre-registered floor; an interval containing both zero and the reference estimate is `NOT_ESTIMABLE_UNDERPOWERED`, with no direction recorded. | `guards.estimability_floor` states the floor and has **no call site**; what runs instead is `schema.HealthFloors.min_blocks`, floored before any effect, and `validate.StabilityResult`, which refuses a result below 30 affected peaks that is not `LOW_RISK_RARE_NOT_VALIDATED` |

## Four that were learned the hard way

**`BA-14`** and the two-axis rule behind it (`FP-18`) came from the same investigation:
the hit caller is not input-scale invariant, and the onset was bracketed to a window
under 10% of the base set's size. In the same ladder, permuting input order produced
the largest coefficient displacement measured and **zero** discrete flips. Two axes,
neither predicting the other. See [`LESSONS.md`](LESSONS.md).

**`BA-17`** is the reason the schema has a four-state enum rather than a nullable float.
The encoding had already been written down; a table operation coerced an all-undefined
group to `0.0` anyway, and the coverage figure computed after that coercion reported
perfect coverage — so the error corroborated itself.

**`BA-16`** is why `interpret` dispatches its output mode from a declared grade rather
than from a flag the analyst sets, and why an undeclared grade lands in the most
conservative mode rather than the most permissive.

**`BA-11`** is why identifiers are wrapped rather than passed as strings: a hit-caller
row number was once matched against a discovery manifest id, filing one factor's
evidence under another's name.

## Reconciliation with this repository's earlier reconstruction

Before the report tables were supplied, this file held a 21-axis list reconstructed from
the reference implementation's own record. Nine of those axes have no `BA-` counterpart.
They are **not** discarded — some are covered by a frozen principle instead of an axis,
and some may be genuinely absent from the report:

| reconstructed axis | disposition |
|---|---|
| Numeric vs discrete divergence | covered as a frozen principle (`FP-18`), not as a bias axis |
| Single-family layer | covered as a frozen principle (`FP-19`) |
| Vacuous guard | covered as a frozen principle (`FP-25`) |
| Aggregation attenuation | covered as a frozen principle (`FP-11`); adjacent to `BA-02` |
| Competitive-fit redistribution | covered as a frozen principle (`FP-08`) |
| Comparator dependence (baseline choice flips the reading) | **no counterpart.** `BA-18` is a *defect* propagating through a comparator; this is a sound comparator, differently chosen. Retained here and enforced by `guards.comparator_declared` |
| Attribution asymmetry (bias-corrected vs uncorrected) | **no counterpart.** Retained as an open question |
| Readout confound (which cell type the readout reports on) | **no counterpart.** Adjacent to `BA-13`, but about cell type rather than sign. Retained as an open question |
| Shared weights across analyses (two analyses, one attribution array) | **no counterpart.** Retained as an open question; it is why analyses are counted by distinct attribution source, not by row |

The four marked **no counterpart** are this repository's annotations, not report content,
and are listed here so that the report's authors can decide whether they belong in the
ledger. The report remains the design authority.
