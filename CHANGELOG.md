# Changelog

All notable changes to this project are documented here.
Format follows Keep a Changelog; this project uses semantic versioning once it reaches 0.1.0.

Nothing has been released yet, so the section below describes the **net state** of the
unreleased tree rather than the order in which it was reached. A file added, removed and
added again is not three entries; the specific accidents live in the commit messages and
in `docs/LESSONS.md`, which is where a reader looking for them expects to find them.

## [0.1.0.dev0] - unreleased

**Pre-alpha. The API is not stable.** All ten modules are implemented, `compile`
produces tiered lexicons, and `multiverse` runs a predeclared grid of specifications
over one frozen dataset — recording every planned cell, keeping estimands,
measurement definitions and statistical choices on separate axes, and refusing to
summarise across baseline populations. Read this as an auditable lexicon compiler
plus a specification-multiverse inference reference implementation.

### Added
- `examples/input_scale_invariance.py` + `docs/INPUT_SCALE_INVARIANCE.md`: README
  finding 1 -- the load-bearing one, previously marked *externally sourced, not
  reproducible from this repository* -- turned into a measurement on K562 ALLPEAKS
  with one frozen lexicon. **The instability reproduces; its stated cause does
  not.** 3.4-6.3% of shared regions change their discrete hit decisions, against a
  0-1 change floor from a same-input repeat (independently re-run from a rebuilt
  input on a third GPU: 0 changes, Jaccard 1.000000). But growing the input by
  9.67% -- the reference's own bracket -- or by 100%, with each region left at its
  row index, changes **zero** decisions; re-ordering the same 6,460 regions changes
  876, and changing only `--batch-size` changes 1,557. The caller's own `num_steps`
  moves the same way, so the operative variable is the solver schedule, not the
  region count. **The "(6,460, 7,085]" onset and the "9.67% suffices" figure are
  withdrawn as scale claims** in `README.md`, `docs/CONCEPT.md`, `docs/LESSONS.md`,
  `docs/ARCHITECTURE.md`, `docs/BIAS_LEDGER.md` and `infer/README.md`. README
  finding 2's "zero discrete flips under permutation" and "2.07x displacement" are
  contradicted outright (876 flips; 104x). The architecture constraint that rests
  on all this -- every specification is a subset of one frozen run -- is unchanged
  and better supported, since re-calling now demonstrably perturbs decisions at
  identical scale. `guards.single_scale` is recorded as **under-keyed**: it
  compares a region *count*, and two runs with the same count disagree on 876 hits.
  Fixing the guard's contract is left as a separate decision.
- `adjudicate/criteria.v2.yaml`: a preregistered, frozen `TRUE_DUPLICATE` --
  `FROZEN_DECLARED_HEURISTIC`, two declared magnitudes (`ppm_similarity ge 0.90`,
  `overlap_bp ge 8`), checksummed in `docs/MERGE_CRITERION_PREREGISTRATION.md` before the
  held-out run, with its known costs stated in its own `declared_rationale` and its exit
  conditions in `replacement_evidence`. It ships in the wheel and is reached by
  `--criteria` or `adjudicate.packaged_v2_criteria_path()`. It is deliberately **NOT the
  default**: it collapses, and deletion is the one error direction a reader of a compiled
  lexicon cannot undo — an under-deduplicated lexicon carries a duplicate that can still
  be seen and merged, an over-deduplicated one has lost a motif and does not record which.
  On its own preregistered held-out set it fired two of its own falsifiers. A criterion in
  that state is one to offer, not one to administer. `tests/test_default_removes_no_motifs.py`
  pins the resulting invariant as a property rather than as a file name: a default run
  removes no motifs, checked on a fixture `criteria.v2.yaml` demonstrably does delete from.
- `adjudicate.packaged_v1_criteria_path()` / `packaged_v2_criteria_path()`: each packaged
  registry is reachable **by name**. `packaged_criteria_path()` keeps its own meaning --
  "whatever `--criteria` defaults to" -- and `packaged_legacy_criteria_path()` is gone,
  because "legacy" named the file that is now the default. Callers that want a specific
  criterion's behaviour must say which criterion; reading it off the default is what made
  a default change look like a regression in tests that were never about the default.
- `multiverse`: the predeclared specification grid over one frozen dataset. Three
  axes as three types — `Estimand` (query and **baseline population**, with its type
  and construction rule), `Measurement` (lexicon and frozen hit table), and
  `StatisticalChoice` (estimator, block size, replicates, seed, floors) — because a
  flat bag of options invites averaging over them, which is the generic robustness
  score that hides which choice the conclusion was sensitive to. The manifest of
  every planned cell is written *before* the first cell runs; every planned cell
  appears in `cells.tsv` exactly once with `SUCCESS`, `REFUSED_GUARD`,
  `REFUSED_SCHEMA`, `NOT_ESTIMABLE` or `ERROR`; stability is summarised within each
  estimand and never across. No statistics of its own: every effect comes from
  `interpret.interpret_query`, checked structurally by a test. Validated on a
  36-cell grid over frozen K562 artifacts (`docs/K562_MULTIVERSE_AUDIT.md`), which
  found `cl3`'s AP-1/bZIP effect to change sign with the baseline population.
- `guards.no_cross_estimand_pooling`: refuses a stability summary that spans two
  estimands, resolving each cell's estimand from the manifest written before the run
  rather than from the code that grouped the results.
- `substrate.OpportunityLedger` and `interpret.verify_missingness_against_ledger`:
  the call site `guards.four_state_missingness` — the guard for this project's
  founding failure — waited four releases for. The ledger is written by the program
  that FROZE the run, so the claim comes from outside this package and the guard can
  fail for the reason it exists; every coverage this package could previously have
  offered it was one this package had itself computed from the same rows. Verified
  on the frozen K562 substrate: the truthful ledger passes, and one claiming a
  coverage of 1.0 is refused at exit 4 with the failing outcome on disk. Supplied
  with `--opportunity-ledger`; a run without one is unchecked on this axis and does
  not pretend otherwise. Four guards remain uncalled, down from seven.
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
- `compile.operations_log` and the `combination_operations.json` it publishes: for every
  motif in every emitted tier, whether its matrices are the registry's own (`copy` /
  `select_representative`) or the mean of the ones it stands for (`mean`), together with
  the axes that operation held fixed. It is a classification of the written lexicon
  against the registry arrays, not the compiler's account of its own behaviour, which is
  what makes it admissible evidence for the guard that reads it.
- `interpret.FamilyEffect.estimator_min_blocks`: the block floor the effect's own
  estimator enforced, or `None` where it enforced none. The `bca-wild-cluster` path
  refuses below `infer.MIN_ESTIMABLE_BLOCKS`; the percentile path floors replicates and
  not blocks, and the artifact did not say so.
- Exit code `4` for a refusal — the tool declined to produce a number and says which
  rule declined it.

### Changed
- `guards.no_cross_model_cwm_avg` has a call site: `compile.compile_lexicons` runs it over
  the operations log before publication, so a lexicon holding a CWM averaged across
  model, readout or metacluster is refused rather than written. Five guards, not six, now
  have no call site; `guards.GUARDS_AWAITING_INPUT` and the README daggers follow.
- `guards.GUARDS_AWAITING_INPUT["single_family_layer"]` now states *which* answer it
  reached about `interpret.FamilyComposition.peak_share` — defined, and out of the
  guard's scope, because its denominator is searched peaks and being the only family
  constrains it to nothing (0.310670 for CTCF/CTCFL-like and 0.998143 for AP-1/bZIP on
  the real K562 substrate) — rather than leaving a future round to re-derive it. The
  guard stays unwired, still blocked on the `BUDGET_FRACTION` decision.
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
- **`report` is implemented** and no longer exits 3 — it was the last skeleton body, so
  nothing in this release does. `motifmultiverse report <interpretation>/ --out report/`
  renders one **markdown** document from the `interpretation.json` in that directory,
  the `provenance.json` log beside it, and `docs/bias_ledger.tsv`. Every number is
  `str()` of a recorded field beside the denominator the producing stage recorded and
  *named* (`n_submitted = 8277`); the renderer computes no ratio, because a second
  implementation of the statistics can disagree with the first. Composition and effects
  are branched on `composition is None` / `effects is None`, never on `floor_failures`
  being non-empty — a comparator-side failure withholds effects, not a composition that
  never depended on it. A withheld *p* value prints as `WITHHELD —
  inference_capability = ESTIMATION_ONLY`, never blank and never `n.s.`; this module's
  founding failure was a bootstrap resolution floor printed as though it were a measured
  *p* value, and the note that says so is rendered **above** the effects table.
  **Contract change:** the skeleton's `report project/` positional is now
  `report <interpretation>/` — a directory of stage artifacts, which is what is
  rendered — plus `--bias-ledger` (default `docs/bias_ledger.tsv`). `--html` and
  `--docx` are kept as **refusals** (exit 4), not silently downgraded to markdown:
  rendering one form while the caller asked for another is the specified-versus-ran gap
  this package exists to close. No migration path is offered at `0.1.0.dev0`.
- **`report` names what it does not know rather than defaulting it.** A mandatory
  *What this report does not know* section renders the literal token `NOT RECORDED` for
  `baseline_population` (carried by no artifact in this package, though the module's own
  rule demands it), states that `lexicon_id` is a declared string on the hit rows and
  **not** `LexiconManifest.lexicon_content_hash` (what `FP-11` requires a family-level
  number to cite), and records that `selection_rule` / `selection_feature_names` are
  fields of `schema.PeakSetQuery` that `interpret.Interpretation` does not emit. No
  artifact in this package persists a `guards.GuardResult`, so the report may name which
  guards `interpret.interpret_query` invokes as facts about the code path and **must not**
  state that any guard passed on this artifact; a guard's absence from
  `guards.GUARDS_AWAITING_INPUT` is not evidence that it has a call site.
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

- **The distribution carries the data its code reads.** `docs/bias_ledger.tsv` moved to
  `src/motifmultiverse/report/bias_ledger.tsv` and is declared package data, and
  `report.packaged_bias_ledger_path()` resolves it through `importlib.resources` the way
  `adjudicate.packaged_criteria_path()` already resolved the criterion registry.
  `report --bias-ledger` now defaults to the packaged ledger rather than to
  `docs/bias_ledger.tsv`: that default resolved from a checkout and not from a wheel, so
  `pip install motifmultiverse && motifmultiverse report interpretation/` refused, naming
  a file the distribution had never contained — a packaging defect wearing a refusal's
  clothes. `tests/test_packaging.py` checks every declared resource through the accessor
  the code calls, in both directions, and CI's `wheel` job installs the built wheel into
  a clean environment and renders a report from a directory with no repository in it.
- **Every run records its outcome where the run wrote**
  (`motifmultiverse.run_status`, `run_status.json`). A refusal used to append its
  provenance record and produce nothing else, so an output directory could hold an
  earlier run's result, a later run's refusal record, and nothing relating the two; this
  was documented in `cli`'s docstring as a known limitation whose advice was "read the
  exit code, not the directory". The exit code is gone by the time anyone opens the
  folder. `run_status.json` states `SUCCESS` / `REFUSED` / `UNIMPLEMENTED` /
  `INPUT_MISSING` / `CRASHED` with the exit code and the refusal's own sentence, and
  `artifacts_are_from` carries the last successful run forward across later failures, so
  a stale result is labelled rather than removed. Nothing is deleted: destroying a real
  result to prevent a misreading of it was never the repair. A run refused before it
  writes anything still creates no output directory.
- **Every run records what its guards returned** (`motifmultiverse.guard_log`,
  `guard_outcomes.json`). A tool whose thesis is that decisions carry their evidence
  recorded the outcome of none of its own executable constraints: `report` had to print,
  in *what this report does not know*, that no artifact persisted a `guards.GuardResult`,
  so the strongest thing it could say was which guards the **code path** calls. Every
  stage that runs a guard — `interpret`, `infer`, `align`, `annotate`, `compile` — now
  appends the guard's own pass/fail, its own sentence, and what it was handed, and it is
  written *as the guard returns*, so a run refused **by** a guard records which one
  refused it even though it produces no result artifact at all. It is its own file rather
  than a field of `provenance.json` (written before the body runs, by contract) or of the
  result (which a failing guard prevents). `report` renders it, joined to the run through
  `run_status.artifacts_are_from.provenance_records`, and keeps the old sentence for an
  artifact that carries no such record. Two inferences are refused in code and in prose:
  a guard **absent** from a record is not recorded as not having run, and a recorded pass
  is what the guard returned, not evidence that it was right to return it. It closes no
  entry in `guards.GUARDS_AWAITING_INPUT` — a record of what a guard returned is not an
  independent claim for a guard to check — and `four_state_missingness`'s entry now says
  so, along with the two things that do stand between it and a call site.
- **`report` refuses an unparseable input instead of raising through the CLI.** Both
  `interpretation.json` and `provenance.json` were read with a bare `json.loads`, so a
  truncated or hand-edited record escaped as a `JSONDecodeError`: a traceback and exit 1,
  a code this CLI's contract does not define. It is now a `ReportError` — exit 4, naming
  the file — as `compile` already did for its decisions payload.
- **`status` verifies a backend by using it, not by importing it.**
  `status.BackendProbe` carries the capability a `VERIFIED` claims and an executable
  check of it; `compile.probe_backend()` compiles a one-motif lexicon and reads it back
  with the real loader, comparing the order with the same guard `verify_roundtrip` uses.
  `backend_status` reported `VERIFIED` as soon as `import_module` returned, which this
  package's own code contradicts — `BackendMissing` and `BackendIncompatible` exist
  because finemo 0.40 renamed an argument and every importable installation of it could
  not read a lexicon back. An installed-but-incapable backend is now `UNVERIFIED` with
  the reason. Still two-valued; there is still no third, comfortable value.
- **CI is three jobs, and one of them is the round trip.** `core` runs lint and the suite
  on Python 3.11 **and 3.12** (the classifiers claimed 3.12 and nothing ran it);
  `roundtrip` installs `.[dev,finemo]`, requires the capability probe to pass, and runs
  the suite with `MOTIFMULTIVERSE_REQUIRE_FINEMO=1` so a missing backend fails rather
  than skips — that check had skipped in every CI run this repository has ever had;
  `wheel` builds the distribution, installs it into a clean environment and runs the CLI
  from a directory that is not a checkout.
- Release metadata in `pyproject.toml`: an author, and absolute `Homepage` /
  `Repository` / `Documentation` / `Issues` / `Changelog` URLs. The single previous entry
  was `Documentation = "./docs/"`, a relative path resolved against pypi.org.
- `CITATION.cff`, with a real author and ORCID and deliberately **no DOI**: a placeholder
  DOI resolves to nothing while rendering as a citable record. Two tests keep it honest —
  its version must equal `motifmultiverse.__version__`, and its licence must match
  `LICENSE`.

### Known gaps
- Every module body is now implemented and nothing exits `3`; the exit-3 path and
  `cli._not_implemented` are kept, and tested against a subcommand pushed back to the
  skeleton dispatcher, rather than deleted with the last skeleton.
- `infer` estimates one specification rather than a multiverse.
- `report` renders **markdown only**; `--html` and `--docx` refuse. What it renders is
  bounded by what the artifacts carry: `baseline_population` is carried by nothing in
  this package, no artifact persists a guard outcome, and `interpret.Interpretation`
  emits neither `selection_rule` nor `selection_feature_names` — so a report over a real
  run states those as unknown rather than as checked, and `docs/ROADMAP.md` M4 does not
  close until the producing stages carry them.
- The **default** criterion registry is `adjudicate/criteria.v1.yaml`, which leaves
  `TRUE_DUPLICATE` and `FRAGMENT_MATCH` `CRITERION_NOT_YET_DEFINED`, so a default run
  defers every duplicate and every fragment, **removes no motifs**, and `compile` emits
  an undeduplicated lexicon by design. That is a statement about the science -- no frozen
  document says how much reconstruction loss a collapse may cost -- not about the code:
  the collapse path is implemented and exercised end to end, but against a criterion the
  caller asks for, so this release is a strict adjudication framework rather than a
  harmonizer with a validated merge policy. `adjudicate/criteria.v2.yaml` is the one
  frozen exception and is not on the default path (see *Added*).
- `compile`'s round-trip verification is **skipped, not passed**, wherever the `finemo`
  backend is absent, and a skipped test is unverified. It is no longer absent in CI: the
  `roundtrip` job installs the backend, requires `status`'s capability probe to pass, and
  runs the suite under `MOTIFMULTIVERSE_REQUIRE_FINEMO=1`, so a missing backend fails
  that job instead of shrinking it.
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
