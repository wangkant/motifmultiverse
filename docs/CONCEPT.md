# Concept

An English summary of the design document, self-contained: you should not need
the original to understand the architecture.

> **Provenance.** Mixed, and the difference matters.
>
> - The **bias ledger** and the **frozen principles** are now *transcribed* from
>   design report v0.8, whose tables were supplied as TSVs. See
>   [`BIAS_LEDGER.md`](BIAS_LEDGER.md) (20 axes, `BA-01`…`BA-20`) and
>   [`CONSTRAINTS.md`](CONSTRAINTS.md) (25 principles, `FP-01`…`FP-25`).
> - **Everything else on this page** — the architecture narrative, the layer and
>   module decomposition, the four findings — is still *reconstructed* from the
>   reference implementation's own record, not translated from the report.
>
> Do not read this file as verified as a whole. The report is the design
> authority; where the two disagree, the report wins and this file is wrong.

## What the tool is for

> Bias-aware harmonization and robust inference of attribution-derived regulatory
> motifs across models and methods.
>
> — concept report §4.1, "Working name and one-sentence positioning"

The ecosystem already solves the neighbouring problems: **TF-MoDISco** discovers
motifs from attributions, **FIMO** scans sequence, **FiNeMo** calls instances,
**HOMER** and **TomTom** annotate and compare. What remains project-local is the
step in between — *how do you compile several local TF-MoDISco outputs into one
shared lexicon that is safe to compare across models?* That step is usually a pile
of per-project scripts plus human judgement.

This tool is that step: an **attribution lexicon compiler**. It turns "should
these merge", "what is this called", "does it enter the main lexicon" and "does
this conclusion hold" into auditable decisions carrying evidence, uncertainty and
downstream validation.

It is **not** another motif clustering algorithm, **not** a motif database, and
**not** an enrichment tool.

## Three design principles

1. **Representation and identity are separate.** A CWM belongs to a specific model
   and readout. `family_id` / `variant_id` are the ontology that crosses them.
   Averaging CWMs across models is therefore prohibited, not merely discouraged.
2. **A merge is validated downstream, not by similarity.** Two motifs are "the
   same" when collapsing them does not degrade reconstruction — a testable claim.
3. **The output is stability and uncertainty, not one absolute lexicon.** Tiers,
   sensitivity lexicons and specification curves are the product.

## Seven layers, nine modules

Input and provenance → evidence graph → ontology and adjudication → lexicon
compilation → downstream stability → robust statistics → audit report; realised
as ingest, align, annotate, adjudicate, compile, validate, infer, report and
interpret. See `ARCHITECTURE.md`.

The **evidence graph** is the central data structure: typed nodes (motifs) and six
classes of typed edge (alignment, sequence hit, attribution hit, downstream
sensitivity, external biology, decision), each carrying uncertainty and a
missingness state. See `DATA_MODEL.md`.

The **bias ledger** enumerates **20 axes** (`BA-01`…`BA-20`) along which a number in
this domain can look right and be wrong, each with its mechanism and control
strategy. It is both prose and a machine-readable table, because the report renders
it. The twenty-five **frozen principles** (`FP-01`…`FP-25`) are its companion. See
`BIAS_LEDGER.md` and `CONSTRAINTS.md`.

The **specification curve** reports a conclusion across the analysis choices that
could reasonably have been made. Cells are pruned only for reasons fixed in
advance — currently `NOT_ESTIMABLE_UNDERPOWERED` (N below floor, or an interval
containing both zero and the reference) and `SINGLE_FAMILY_LAYER` (a within-peak
share that is 1 by construction). **Dropped cells are reported, never silently
absent.**

## The four findings that motivate the design

These come from the reference implementation and are the reason specific
constraints are architectural rather than optional. Figures below are from that
line; this repository contains no analysis data of its own.

**1. The hit caller is not input-scale invariant.** Identical regions produced
different *discrete* retention decisions depending on which other regions shared
the input. A bisection bracketed the onset to **(6,460, 7,085] regions** on a
6,460-region base — a **9.67%** increase in scale is enough. There is no usable
safe zone, so all specifications must be subsets of one frozen run.

> **Re-measured in this repository; the scale attribution did not survive.**
> [`INPUT_SCALE_INVARIANCE.md`](INPUT_SCALE_INVARIANCE.md) re-ran the caller over
> nested peak sets on K562 ALLPEAKS. The instability is real (3.4–4.7% of shared
> regions, against a 0-change floor on a repeat run), but **scale is not what
> drives it**: +9.67% and +100% growth with every region left at its row index
> changed 0 decisions, while re-ordering the same 6,460 regions changed 876 and
> changing only the batch size changed 1,557. The bracket and the 9.67% figure
> should not be cited as a scale threshold. "All specifications must be subsets of
> one frozen run" is unaffected and, if anything, more strongly supported.

**2. Numeric and discrete divergence are independent axes.** Permuting input order
produced the **largest** coefficient displacement of any perturbation measured
(**2.07×** the median coefficient) while flipping **zero** retention decisions.
Neither axis predicts the other; both are measured and gated separately.

> **Both figures contradicted by the same re-measurement.** At fixed scale,
> permuting order gave **876** discrete flips (4.69% of regions), not zero, and a
> largest displacement of **104×** the median coefficient, not 2.07×. Gating the
> two axes separately remains the right practice; these numbers do not transfer.

**3. Writing a rule down is not executing it.** A four-state missingness encoding
was specified and still silently destroyed by a table pivot that coerces an
all-undefined group to `0.0`. The coverage figure, computed *after* that fill,
reported `1.000000` — the error corroborated itself. Constraints must therefore be
executable assertions.

**4. A guard must be shown to be capable of failing.** Of five framework guards
that all reported passing, a falsification pass found **2 still passed** under a
row-shifted *and* a permuted lexicon index, and **none of the five** could detect a
reordered index. In a report, a vacuous guard and a correct guard are
indistinguishable.

## The most transferable conclusion

> **A frozen lexicon transfers; a comparator does not.**

The same measurements supported both "replicates exactly" and "four times
stronger, prediction falsified" — differing only in whether the baseline was the
unselected universe or a residual subset from which the relevant peaks had already
been removed. Nothing about the data changed.

Any cross-condition motif claim that does not state its baseline population is
uninterpretable.

## The N=2 ceiling

The reference implementation had exactly two models, and that number was baked
into its cross-model reasoning. This tool keeps the analysis list unbounded, and
**refuses at runtime** to estimate between-model heterogeneity below three models,
reporting sign consistency and leave-one-model-out instead. That is a conceptual
limit, not a configuration one: adding a model name does not supply the missing
variance.
