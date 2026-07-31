# Changelog

All notable changes to this project are documented here.
Format follows Keep a Changelog; this project uses semantic versioning once it reaches 0.1.0.

## [0.1.0.dev0] - unreleased

**Pre-alpha. The API is not stable, and no module produces a lexicon yet.**

### Added
- Package skeleton for nine modules (ingest, align, annotate, adjudicate, compile,
  validate, infer, interpret, report), each with a README stating its rule, the failure
  that produced the rule, and how the rule is checked.
- **`ingest` is implemented**: discovery HDF5s named by a project config become one
  registry, with a checksum per input. A metacluster contributing no patterns records
  which of three absences occurred (`group_absent` / `group_empty` / `not_searched`),
  and `cross_model_claims_restricted` travels with the data when N < 3 models.
- **`compile` is implemented**: one hit-caller-compatible lexicon per tier
  (`core` / `expanded` / `sensitivity`), written **in the order the loader emits**,
  each with a `lexicon_content_hash` and a `comparisons` block stating whether the
  tier contrast varies the positive set, the negative set, or neither. Round-trip
  verification calls the real loader; without that backend it says the round trip did
  not happen rather than assuming it would have passed.
- `guards.index_order_matches_loader`, with a falsification test on a
  metacluster-ascending index.
- `schema`: `MetaclusterState`, `RegistryMetadata`, `LexiconManifest`, `Estimator` /
  `IMPLEMENTED_ESTIMATORS`, `UNION_ID_RE`.
- `provenance`: a `redaction_policy` field, so the policy is data rather than a source
  comment for anything that later bundles records for release.
- `schema/`: motif node, evidence edge, decision record and analysis config, encoding
  four rules derived from real failures (stable `variant_id`, no key parsing, four-state
  missingness, refusable decisions with measured confidence).
- `guards/`: fourteen executable constraints, each with a falsification test, plus a
  meta-test that fails if any guard ships without one.
- `provenance/`: input checksums, command line, software versions, seed and timestamp,
  recorded by every subcommand including the unimplemented ones.
- CLI with nine subcommands wired to real arguments and real `--help`.
- **`interpret` is implemented**: subset queries over one frozen hit table, with the
  output mode dispatched from the declared selection provenance, three health numbers
  computed before any effect, and suppression (not annotation) when a pre-registered
  floor fails. Percentile block bootstrap with block size, `B` and seed stored beside
  every interval.
- `schema`: `SelectionProvenance`, `OutputMode`, `HitRecord`, `PeakSetQuery` and
  `HealthFloors`, added because implementing `interpret` showed the hit table had no
  schema — including no way to represent a peak that was searched and produced nothing.
- `guards`: `selection_provenance_declared`, `health_before_effect`,
  `comparator_declared`.
- `docs/LESSONS.md`: every architecture constraint indexed back to the failure that
  produced it, and to what it currently blocks.
- Exit code `4` for a refusal — the tool declined to produce a number and says which
  rule declined it.

### Changed
- `docs/bias_ledger.tsv` and `docs/constraints.tsv` are now **transcribed** from design
  report v0.8 (20 axes `BA-01`…`BA-20`; 25 principles `FP-01`…`FP-25`) rather than
  reconstructed. The earlier counts of 21 and 26 were miscounts that included the header
  row. The `enforcement` column is this repository's own annotation.
- Enforcement re-annotated against the 25 transcribed principles: `ENFORCED` 4,
  `PARTIAL` 10, `DOC_ONLY` 11. The earlier 20/3/3 was written backwards from what the
  code already did.
- Every `DOC_ONLY` principle now carries a `criterion_draft`: the check is
  unimplemented, but what it would check is written down.
- Enforcement moved to `ENFORCED` 4 / `PARTIAL` 13 / `DOC_ONLY` 8 as `ingest` and
  `compile` landed (`FP-01`, `FP-05`, `FP-11` left `DOC_ONLY`). `CONSTRAINTS.md` now
  states the labelling rule itself, so a new row can be labelled without asking.
- `interpret` results enumerate every recognised estimator and the subset implemented,
  so a caller does not change when `FP-15`'s estimators arrive.
- **`FP-15`'s two specified estimators are implemented** (`docs/ROADMAP.md` M4a):
  `infer.bca_paired_block_interval` (BCa paired genomic-block bootstrap interval) and
  `infer.wild_cluster_bootstrap_t` (block-level wild cluster bootstrap-*t* *p* value,
  null-imposed Rademacher weights, finite-sample `p = (extreme + 1) / (B + 1)`). The
  block, never the peak, is the resampling and jackknife unit in both.
  `interpret --estimator bca-wild-cluster` runs them as one path licensed
  `INTERVAL_AND_TEST`: an effect then carries a BCa interval, a *p* value, and a
  Benjamini-Hochberg *q* value over the families in that interpretation and no others.
  `schema.IMPLEMENTED_ESTIMATORS` now contains all three recognised values, so a
  result can no longer name an estimator its own enumeration calls unavailable.
  **The default is unchanged** — `--estimator percentile`, `ESTIMATION_ONLY`, *p* and
  *q* withheld. Switching the default is a separate decision and will get its own
  entry here; an unrecognised `--estimator` is refused rather than mapped onto the
  default.
- **Two-part usage summaries** (`infer.two_part_summary`, `interpret.two_part_effects`).
  A family used more often and less intensely than its comparator has a one-part mean
  difference of about zero, which reads as "no difference" for two large opposite
  effects. Occupancy (`Peak.family_hit_count`) and conditional intensity
  (`Peak.family_coefficient_sum`) are now reported separately, with the total they
  multiply to beside them. `infer.UsageDefinition` (`ANY_HIT`, `CONTRIBUTION_FLOOR`,
  `BUDGET_FRACTION`) has **no default anywhere**: `Interpretation.two_part_effects` is
  `null` when nobody chose one, and the thresholded definitions require an
  `infer.UsageThreshold` carrying the null calibration it came from — a cut-off without
  a named null is refused, not assumed. `NOT_SEARCHED` leaves every denominator;
  `NO_SEQUENCE_MATCH`, `HIT_BELOW_FLOOR` and sub-threshold hits stay in it as measured
  non-use. `conditional_intensity_effect` is `null`, never `0.0`, when a side never uses
  the family.
- **`infer` is implemented** and no longer exits 3. It reads one frozen hit table and
  writes `inference/effect_estimates.tsv` -- one row per family, carrying the effect,
  the interval, `inference_capability`, the estimator that produced it and the
  substrate identity -- beside the full `interpretation.json`. An undefined value is
  written as `NA`, never as a blank or a zero. It runs `interpret`'s code rather than a
  second copy of the statistics. **Contract change:** the skeleton's
  `infer instances/ --unit --multiverse` arguments are gone; the specification
  *multiverse* is not implemented, and arguments that implied it were removed rather
  than left to suggest a sweep that does not happen.
- **`implementation_status.json`**, generated by `python -m motifmultiverse.status`.
  Module status is derived from the CLI dispatch table, optional backends are
  `VERIFIED`/`UNVERIFIED` with no third value, and test counts are three separate
  numbers that are absent (`NOT_RUN`) rather than zero when nothing ran. The README's
  module table is rendered from it and a test fails if it drifts; the stale
  "163 passed, 1 skipped" line is gone, and no test count is committed to prose at all.
  CI uploads the document as an artifact.
- `tests/test_end_to_end.py`: the declared path (`ingest -> align -> annotate ->
  adjudicate -> compile -> validate -> infer`) run on synthetic inputs, asserting every
  promised artifact exists and parses, that a sign-flipped representation registers at
  the same offset with a negative signed CWM similarity, and that adjudication emits
  collapse, refusal and deferral in one run once downstream stability evidence exists.
- **Contract change:** `compile` takes `compile registry/ --decisions decisions.json`
  instead of the skeleton's `compile review.yaml`, since `adjudicate` — which would
  have produced that review file — does not exist. No migration path is offered at
  `0.1.0.dev0`.
- Sensitivity-lexicon membership is decided by three named triggers
  (`merge_confidence != HIGH`, `family_ambiguity`, `threshold_sensitive`) and each
  manifest records which fired. The numeric `MODERATE_MERGE_CONFIDENCE` threshold is
  **removed**: the design never defined "moderate-confidence merge" and never made it
  a scalar, so any cut-off invented a continuous quantity that does not exist.
  `schema.MergeConfidence` is a categorical grade — `grade < 0.8` raises `TypeError` —
  and `MERGE_CONFIDENCE_CRITERIA` records `CRITERION_NOT_YET_DEFINED` for all three
  levels rather than filling them in.
- **Breaking: `lexicon_content_hash` now covers loader configuration, not only motif
  arrays.** `compile_lexicons()` gained `trim_threshold`, `motif_type`, `include_rc`,
  `loader_backend` and `loader_parameters` — the settings that used to be hard-coded
  inside `load_back()` and therefore invisible to the hash. Two lexicons built from
  identical arrays but meant to be read back under different loader settings load
  differently and must not share an identity; now they don't. Every
  previously-computed `lexicon_content_hash` changes as a result — there is no
  migration, because nothing has consumed a hash as a stored value yet. `load_back()`
  and `verify_roundtrip()` were updated to read back under the same configuration the
  hash covers, instead of `load_back`'s own fixed defaults, and behaviourally
  equivalent spellings of `loader_parameters` (`None`, `{}`, an explicit
  `{"motif_lambda_default": 0.7}`) are resolved to one canonical form before hashing
  so they still address identically.

- **`align` runs its pair loop in parallel on request**: `align_registry(..., workers=N)`
  and `motifmultiverse align --workers N`, **defaulting to 1**, so no existing
  invocation changes. The null is untouched — still one full re-registration per
  shuffle per pair, still pinned by `test_align_null_re_registers_from_scratch_for_every_shuffle`
  — because parallelism is the one speed lever that does not answer an easier question.
  It is admissible only because it cannot reach the arithmetic: each pair's null
  generator is built from the run seed alone inside `calibrate_pair_null`, nothing is
  carried between pairs, and outcomes are reassembled by pair order rather than by
  finish time. Measured on the 29-pattern ChromBPNet registry at 1000 shuffles (two runs
  of the sweep): 24.9–25.3 s at 1 worker, 13.1–13.2 s at 2, 6.8–7.5 s at 4, 3.7 s at 8,
  with `alignment_edges.parquet` and
  `alignment_null_summary.tsv` byte-identical at every worker count and identical to
  what the stage wrote before the parameter existed. That equality is a test, not a
  claim (`test_align_registry_writes_byte_identical_tables_at_every_worker_count`), and
  `test_align_null_is_a_pure_function_of_the_seed_and_the_pair` pins the mechanism it
  rests on against an independent recomputation from the seed.
- `align` reports progress: `align_registry` calls a `progress(completed, total)`
  callback once per registrable pair and writes to no stream itself; the CLI turns that
  into **stderr** lines, at most one every two seconds. Nothing new is written to
  stdout, which callers parse for the counts and the `written:` paths.

### Removed
- `CITATION.cff`. A citation file with placeholder authors renders as a claim that the
  project is ready to be cited; it goes in at publication, with real names.

### Known gaps
- `report` is the one module body that still raises `NotImplementedError`, and `infer`
  estimates one specification rather than a multiverse.
- The shipped `config/criteria.v1.yaml` leaves `TRUE_DUPLICATE` and `FRAGMENT_MATCH`
  `CRITERION_NOT_YET_DEFINED`, so the shipped pipeline **defers every duplicate and
  every fragment** and `compile` emits an undeduplicated lexicon by design. That is a
  statement about the science -- no frozen document says how much reconstruction loss a
  collapse may cost -- not about the code: the collapse path is implemented and
  exercised end to end against a criteria file supplied by the caller.
- `compile`'s round-trip verification is **skipped, not passed**, when the `finemo`
  backend is absent — which includes CI. Verified manually against a real TF-MoDISco
  output; `compile.load_back` itself has never been executed, because the environment
  that has the backend runs Python 3.10 and this package requires 3.11.
- What earns each `MergeConfidence` grade is undecided (`CRITERION_NOT_YET_DEFINED`),
  so `compile` dispatches on a declared grade and never assigns one.
- `MotifNode` holds one `family_id`, so two analyses disagreeing about a motif's
  family — the normal state before adjudication — cannot be represented. See
  `docs/DATA_MODEL.md`.
- `align` is prose-only by inheritance; `annotate` was **never specified** at all, which
  is a different position on the roadmap — it is waiting for a design, not an
  implementation.
- `align`'s null generator is seeded **per run, not per pair**, so two pairs whose
  targets have the same trimmed-core length draw the same sequence of row permutations
  and their nulls are positively dependent. Anything later that treats these p-values
  as independent tests inherits that. Seeding per pair would decorrelate them and would
  change every p-value this registration rule version has produced — a decision about
  the null, which is not one to make inside a performance change, so it is recorded
  here instead. Parallelism reproduces the existing draws exactly and neither causes
  nor is blocked by this.
- `interpret`'s **default** intervals are percentile block bootstrap and carry no *p*
  or *q* value. `FP-15`'s BCa interval and wild cluster bootstrap-*t* now exist behind
  `--estimator bca-wild-cluster`; the cross-model effect-then-meta-analysis half of
  `FP-15` does not.
- Between-model heterogeneity is refused at runtime for N < 3 models.
