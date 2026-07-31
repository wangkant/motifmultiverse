# A 36-cell multiverse over frozen K562 artifacts

Validation of `multiverse` on real data, not a fixture: the frozen K562 ALLPEAKS
universe (33,917 peaks) and two frozen ChromBPNet hit-calling runs over it. Nothing
was re-called for this audit; both hit tables are materialisations of runs that
already existed.

## The grid

| axis | levels | what moves |
|---|---|---|
| **Estimand** | 6 | two queries (`cl5`, `cl3`) × three baseline populations: the complement within the universe, and two sibling Leiden clusters (`cl4`, `cl0`) taken whole |
| **Measurement** | 2 | `CBP-2114__core_final` and `CBP-2114__expanded` — two lexicons over the same peaks, so two frozen substrates |
| **Statistical** | 3 | percentile block bootstrap at 1 Mb and at 250 kb; wild-cluster bootstrap-*t* at 1 Mb. 100 replicates, seed 0 throughout |

6 × 2 × 3 = **36 planned cells. 36 ran and produced estimates; none was refused
and none was non-estimable**, so the dropped-cell path is exercised in the test
suite rather than here — this grid does not demonstrate it and does not claim to.
522 family-level effects, and 102 stability summaries across 6 estimands, each
summary inside one estimand. 145 guard outcomes: four per cell, plus the
`no_cross_estimand_pooling` call over the finished summaries. Each is joined to the
cell it licensed in `cell_guard_outcomes.tsv` — the directory-level
`guard_outcomes.json` records them all but has no cell to attribute them to, which
for a 36-cell grid is 144 entries a reader cannot place.

## 1. The statistical axis moves the interval, not the estimate

Across all 174 (estimand, family, lexicon) combinations the point estimate is
**bit-identical** under all three statistical choices — maximum spread `0.0`. That
is the correct behaviour and worth stating plainly, because a specification curve
that plots point estimates against statistical choices would show three identical
rows and invite the reading that the result is "robust to the estimator". It is
not robust to it; it is *independent* of it. What the choice moves is the interval:

| statistical choice | median CI width | capability | *p* values |
|---|---|---|---|
| `pct_1Mb` | 6.9e-05 | `ESTIMATION_ONLY` | 0 of 174 |
| `pct_250kb` | 6.3e-05 | `ESTIMATION_ONLY` | 0 of 174 |
| `bca_1Mb` | 6.3e-05 | `INTERVAL_AND_TEST` | 174 of 174 |

Quartering the block size narrows the interval by ~6% at the median (ratio range
0.67–1.31), which is the expected direction: more, smaller blocks resample more
independent units and understate dependence if the true correlation length is
longer than the block.

**Where that matters: 8 of 174 combinations have an interval that includes zero
under one statistical choice and excludes it under another.** Among them
`cl3 / cl0 / ZBTB17` under both lexicons, and `cl3 / cl0 / AP-1/bZIP` under
`core_final`. Any statement of the form "this family's interval excludes zero"
is, for those eight, a statement about the block size as much as about the data.

## 2. The baseline is the axis that changes the answer

Sign agreement **within** an estimand (across both lexicons and all three
statistical choices) holds for 101 of 102 family summaries. Sign agreement
**across** baselines does not: 8 of 34 (query, family) pairs flip sign when the
baseline population changes.

Four of those eight flip at magnitudes of 1e-7 to 5e-7, against a median absolute
effect of 7.5e-05 over the whole grid — `RFX1` and `SRY` in `cl3`, `HIC1` and
`SRY` in `cl5`. Those are sign changes in a quantity indistinguishable from zero
at this resolution and are **not** reported here as baseline-sensitivity; naming
them as findings would be reading noise. The remaining four are real:

| query | family | vs complement | vs `cl4` | vs `cl0` |
|---|---|---|---|---|
| `cl3` | **AP-1/bZIP** | **+1.84e-04** | **+1.86e-04** | **−4.05e-05** |
| `cl3` | **NDF2** | +4.76e-05 | −8.17e-06 | **−2.03e-04** |
| `cl3` | SNAI1 | +2.00e-05 | +1.69e-05 | −8.78e-06 |
| `cl3` | ZNF524 | −1.48e-05 | +3.63e-08 | −2.58e-05 |

**`AP-1/bZIP` in `cl3` is the clearest result this grid produces.** Against the
complement and against `cl4` it is enriched, at effectively the same magnitude;
against `cl0` it is depleted. A single-specification run against any one of those
baselines would have reported a clean answer, and two of the three would have
agreed with each other. The conclusion is not "AP-1 is enriched in `cl3`" — it is
that `cl3` and `cl0` are alike in AP-1 content while `cl3` and `cl4` are not, and
which of those the complement comparison reflects depends on how much of the
complement is `cl0`.

This is the case the module's central refusal is built for. Averaged into one
"robustness across specifications" score, `cl3`'s AP-1 effect would come out
positive with a wide interval, and the disagreement — the only informative part —
would be invisible. `guards.no_cross_estimand_pooling` refuses the summary that
would produce it.

**Direction can be stable while magnitude is not.** `CTCF/CTCFL-like` in `cl5` is
depleted against all three baselines, so its sign is not baseline-sensitive at
all — but the magnitude spans 30×: −4.3e-05 against `cl0`, −2.7e-04 against `cl4`,
−1.3e-03 against the complement. "CTCF is depleted in `cl5`" survives every
specification in this grid. "CTCF is depleted in `cl5` by X" is not a property of
`cl5`; it is a property of the pair.

## 3. Lexicon sensitivity, and a category the grid makes visible

Between the two lexicons, within a fixed estimand, **1 of 72 (estimand, family)
pairs flips sign** — `cl3 / cl0 / AP-1/bZIP`, which is the same comparison already
flagged as baseline-sensitive, and it is the one MIXED-sign summary in §2. The
largest relative shifts are `NFY` in `cl3 / cl0` (3.6× between lexicons) and
`AP-1/bZIP` across several estimands (0.22–1.41×).

The larger lexicon effect is not on the numbers but on **which families exist to
be asked about**. Five families are estimable only under `CBP-2114__expanded` and
have no cell at all under `CBP-2114__core_final`: `HIC1`, `NDF2`, `RFX1`, `SNAI1`,
`SRY`. Nothing is estimable only under core.

That asymmetry is worth naming because it is invisible to any single-lexicon
analysis, and because two of those five (`NDF2`, `SNAI1`) are among the four real
baseline-sensitive results in §2. Their entire evidential basis is one measurement
definition; under the other, the question cannot be posed. A summary that filled
their missing cells with zero would have reported them as "measured, and equal to
nothing" under core — which is the collapse this package was written against, and
why `NOT_ESTIMABLE` is a token here and never `0.0`.

## 4. Verdict, in the four categories the audit was asked for

- **Stable across every axis in this grid.** The sign of 101 of 102 within-estimand
  summaries, and specifically the depletion of `CTCF/CTCFL-like` and enrichment of
  `SP_KLF`, `ETS`, `NRF1`, `NFY` in `cl5` — same direction under both lexicons, all
  three statistical choices, and all three baselines. Magnitudes are not stable and
  are not claimed to be.
- **Baseline-sensitive.** `AP-1/bZIP` and `NDF2` in `cl3` (sign flips at real
  magnitude); `SNAI1` and `ZNF524` in `cl3` marginally. And, as magnitude rather
  than sign, `CTCF/CTCFL-like` in `cl5` across a 30× range.
- **Lexicon-sensitive.** `cl3 / cl0 / AP-1/bZIP` by sign; `NFY` in `cl3 / cl0` by a
  factor of 3.6. More importantly, five families exist only under the expanded
  lexicon.
- **Not estimable.** No *cell* was non-estimable in this grid. At the family level,
  the five expanded-only families are not estimable under core, and are recorded as
  such rather than as zero.

## 5. What this audit does not establish

It uses one cell type, one model backbone (`CBP-2114`), two lexicons that are
nested rather than independent, and two queries. Nothing here says how a grid
behaves when the measurement axis spans *models* — which, under this package's
identity rules, would also span peak universes and is a different design.

No threshold was preregistered, so nothing above is classified "significant" and
the *p* values from the `bca_1Mb` cells are reported as available rather than
interpreted. Sign agreement is a description of the signs observed in this grid;
it carries its denominator everywhere it appears and is not a test.

## Reproducing

The design, the peak sets, the two frozen hit tables and the full output live with
the analysis, not in this repository — the audit's inputs are several gigabytes and
one of them is a 96 MB TSV. The design's shape is `docs/MULTIVERSE_DESIGN.md`; the
command is:

```
motifmultiverse multiverse multiverse_design.json --out multiverse_out/
```
