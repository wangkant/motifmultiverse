# Input-scale invariance of the hit caller — an in-repository measurement

Script: `examples/input_scale_invariance.py`
Result tables: `examples/results/scale_invariance_results.tsv`,
`examples/results/scale_invariance_stability.tsv`

**Verdict: the finding half reproduces, and its stated cause does not.** (The
same run also contradicts README finding 2 — see below.) The hit caller's
discrete output does move when the input composition changes —
3.4–6.3% of shared regions, far above a 0.015% instrument floor. But growing the
input while leaving each region's *position* alone changes nothing at all, and
the instability is already at full magnitude at **0%** growth. There is no
onset between 6,460 and 7,085 regions to bracket on this substrate, because
there is nothing to onset *from*: the effect is at full size as soon as the
caller's *schedule* changes, and absent when it does not — changing only
`--batch-size`, which is not a property of the input at all, moves more decisions
than any scale step. The architectural consequence the finding is cited for
survives intact and is, if anything, strengthened.

## Why this experiment

README finding 1 is marked *externally sourced, not reproducible from this
repository*:

> The hit caller is not input-scale invariant. The same regions produce
> different discrete retention decisions depending on which other regions share
> the input. The onset was bracketed to (6,460, 7,085] regions on a 6,460-region
> base — a 9.67% increase in scale suffices. Consequence: every specification
> must be a subset of one frozen run.

The consequence is load-bearing. It is why `guards.single_scale` exists
(`src/motifmultiverse/guards/__init__.py:65`, whose docstring restates the
6,460/7,085 bracket), why `provenance.ProvenanceRecord.input_scale` rides on
every result, and why `docs/BIAS_LEDGER.md` `BA-14` forbids re-calling the
caller between specifications.

## What is actually on disk

| Artifact | Path | Contents |
| --- | --- | --- |
| Lexicon | `…/merge/casestudy/lexicons/core.h5` | 139 motifs; `lexicon_content_hash` `4beaf836a35957eb9f603d80f28c6f4712befa84c0c80edc3aeca47d29115340` in `core.manifest.json`; file sha256 `13d6ff0c83d9b3066279a358d9b050899f9ed2d0214fc9d9e3fceb6acf68455a`. FiNeMo expands it to 278 CWMs (each motif plus its reverse complement): `hits/before_core/parameters.json` `num_motifs: 278`. |
| Case-study hit run | `…/merge/casestudy/hits/before_core/` | 93,661 hits over 2,639 regions; `parameters.json` records `-M pp -t 0.3 -l 0.7 -b 2000`. |
| Case-study regions | `…/region_v4/04_cbp_only_islands/allclusters/cl0/finemo_regions.npz` | **2,639** regions — one Leiden cluster, not the universe. |
| **Full universe** | `…/region_v3/k562/interpret_cbp_par_hyp/finemo_regions.npz` | **33,917** regions × 4 × 2,114, ChromBPNet DeepSHAP hypothetical contributions, unique `peak_name`. |

The case study's own FiNeMo run used a 2,639-peak subset, smaller than the
6,460-region base the finding is stated on, so that file cannot carry this
experiment. The full ALLPEAKS universe can:
`interpret_cbp_par_hyp/finemo_regions.npz` holds all 33,917 peaks in FiNeMo's
own `.npz` format, and `allclusters/prep_allclusters.py` documents that the
per-cluster region files are *row subsets* of exactly this array — ChromBPNet
attribution is per-peak independent and seed-pinned, so a row subset is
identical to a fresh attribution run on that subset. That is precisely the
property the experiment needs: subsetting the universe changes the caller's
input **set** and nothing about any individual region's contributions.

## Method

One fixed lexicon (`core.h5`, hash above). One frozen universe. A single seeded
permutation (`seed=20260731`, the case study's seed) defines a **nested**
ladder: step *N* is the permutation's first *N* rows, so every step provably
contains the base set (verified: the base row set is a subset of every step).
Each step is called with the case study's frozen settings,
`finemo call-hits -M pp -t 0.3 -l 0.7 -b 2000`.

### Two arms, because "which other regions share the input" is two questions

- **`append`** — the superset is the base rows *in base order*, then the extras.
  Every base region keeps its row index. This isolates **scale**: the only
  change is that more regions exist.
- **`sorted`** — base plus extras re-sorted into universe order, so extras
  interleave and every base region's row index moves. This is what enlarging a
  peak universe actually looks like; it varies scale and position together.

`append`-base against `sorted`-base is then a third comparison for free: the
same 6,460 regions, the same count, a different row order — **position at fixed
scale**. This mattered because FiNeMo optimises a rolling buffer of
`--batch-size 2000` regions and refills converged slots from the input in
order, so row position is an a-priori candidate mechanism that a count-based
experiment cannot see.

### What counts as a decision change

A hit **key present in one run and absent in the other**. Nothing else.

The key is `(peak_name, motif_name, start_untrimmed, strand)` — all four
properties of the genome and the lexicon. The caller's own `peak_id` is a row
index into the input file and means different things in two runs of different
composition; it is never used as an identity here.

Coefficient movement on a key present in *both* runs is measured and reported
but is **not** a decision change. The floor is not invented for this
experiment: the package already measured a device null — identical lexicon,
identical regions, GPU against CPU — at max |coefficient delta| **3.63e-07**
over 93,661 shared keys, as
`motifmultiverse.validate.DEVICE_NULL_ABS_COEFFICIENT_DELTA`
(`src/motifmultiverse/validate/__init__.py:95`). The script imports that
constant and reports how many shared keys exceed it.

### The comparison set comes from the inputs, not the outputs

The compared region set is the base set's `peak_name`s read from the row-index
files that built the `.npz`, never the intersection of the two hit tables. A
region whose every hit vanished in one run has no rows in that table, so an
intersection of hit-table peak names would delete exactly the regions the
experiment hunts for and score the largest possible decision change as no
change at all.

### Controls

1. **Negative control** — the base set called twice, same scale, same GPU, same
   input file (`ctrl` reuses `base`'s `.npz`; writing a second copy would test
   the writer). Run before any superset.
2. **Device control** — the `sorted` base re-called on the *other* arm's GPU.
   The two arms were pinned to two different GPUs, so without this the
   order-only comparison would confound row order with GPU instance.
3. **Independent replication, end to end** — both base `.npz` rebuilt from
   scratch (the seeded permutation re-derived and checked equal to the stored row
   indices) and both re-called on a *third* GPU. This is the control the whole
   experiment rests on, so it was reproduced from the inputs rather than trusted
   once:

   | re-run | vs | decision changes | Jaccard |
   | --- | --- | ---: | ---: |
   | `append_base` rebuilt, third GPU | original `append_base` | **0** / 219,203 | 1.000000 |
   | `sorted_base` rebuilt, third GPU | original `sorted_base` | **0** / 219,376 | 1.000000 |
   | rebuilt `append` | rebuilt `sorted`, same GPU (**order only**) | **875**, 302 regions (4.675%) | 0.996018 |

   The order-only effect comes back at 875/302 against the original run's
   876/303, `num_steps` differs for the same 742 of 6,460 regions, and
   `global_scale` is bitwise identical across the pair. The instrument is
   deterministic and the effect is not an artefact of one set of runs.
4. **Batch-size control** — the base input, the base order, one GPU, only
   `--batch-size` changed. Scale and order are both pinned, so anything that
   moves here is attributable to neither.

## Results

`examples/results/scale_invariance_results.tsv`. Base = 6,460 regions,
219,203 hits (`append`) / 219,376 hits (`sorted`) over those regions.

| comparison | what varies | N | decision changes | regions changed | % regions | changes on hits ≥1 | Jaccard |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `append:base_vs_ctrl` | **nothing** (repeat run) | 6,460 | **0** | 0 | 0.000% | 0 | 1.000000 |
| `sorted:base_vs_ctrl` | **nothing** (repeat run) | 6,460 | **1** | 1 | 0.015% | 0 | 0.999995 |
| `sorted_base_vs_sorted_base_other_gpu` | **GPU instance only** | 6,460 | **1** | 1 | 0.015% | 0 | 0.999995 |
| `append:base_vs_sup_7085` | **scale only**, +9.67% | 7,085 | **0** | 0 | 0.000% | 0 | 1.000000 |
| `append:base_vs_sup_12920` | **scale only**, +100% | 12,920 | **0** | 0 | 0.000% | 0 | 1.000000 |
| `append_base_vs_sorted_base_same_gpu` | **order only**, +0% | 6,460 | **876** | 303 | 4.690% | 195 | 0.996013 |
| `append_base_vs_sorted_base` | order + GPU, +0% | 6,460 | 875 | 302 | 4.675% | 195 | 0.996018 |
| `sorted:base_vs_sup_6525` | order + scale, +1.01% | 6,525 | **960** | 222 | 3.437% | 228 | 0.995635 |
| `sorted:base_vs_sup_6783` | order + scale, +5.00% | 6,783 | **732** | 236 | 3.653% | 210 | 0.996670 |
| `sorted:base_vs_sup_7085` | order + scale, +9.67% | 7,085 | **984** | 245 | 3.793% | 242 | 0.995527 |
| `append_base_vs_batch_1000` | **batch size only**, +0% | 6,460 | **1,557** | 408 | 6.316% | 354 | 0.992932 |

### The batch-size row settles the attribution

`--batch-size 2000 → 1000`, on the base `.npz`, in the base row order: the region
set, the region order and the region count are all held fixed and **1,557
decisions move across 408 of 6,460 regions — more than order does, and far more
than scale does, which is nothing.** Batch size cannot be a property of the input
at all. It is a property of how the caller schedules the input, which is what the
row-order arm was already pointing at and what the next section measures directly.

(That run was called on a third GPU. The device control bounds a GPU change at ≤1
decision, and an independent re-call of the base input on that same third GPU
reproduced `append_base` with **0** decision changes and Jaccard exactly
1.000000, so the 1,557 is not a device effect.)

### The mechanism, measured rather than assumed

FiNeMo's `peaks_qc.tsv` records `num_steps` per region — how many solver
iterations that region actually received. It is the schedule made observable, and
the results table now carries it (`n_regions_num_steps_differs`,
`max_abs_num_steps_delta`, `n_changed_regions_also_resolved_differently`).

| comparison | what varies | regions whose `num_steps` differs | max Δ steps | decision changes |
| --- | --- | ---: | ---: | ---: |
| `append:base_vs_ctrl` | nothing | 3 / 6,460 | 2 | 0 |
| `append:base_vs_sup_7085` | **scale only**, +9.67% | 2 / 6,460 | 1 | 0 |
| `append:base_vs_sup_12920` | **scale only**, +100% | **0** / 6,460 | **0** | 0 |
| `append_base_vs_sorted_base_same_gpu` | **order only** | **742** / 6,460 | 464 | 876 |
| `append_base_vs_batch_1000` | **batch size only** | **1,122** / 6,460 | 497 | 1,557 |

The two columns move together and they separate the arms the same way the
decisions do. Doubling the input leaves *every one* of the 6,460 base regions on
the identical iteration count; re-ordering or re-batching perturbs 11–17% of them
by as much as 497 steps. Of the regions whose decisions changed, 61–64% are also
regions whose iteration count changed.

This is no longer "a hypothesis the experiment is consistent with". The operative
variable is where a region lands in the solver's schedule; growing the input by
appending does not move any region's place in that schedule, and re-ordering and
re-batching both do.

Both floor-level changes are the same kind of object — a single
`neg_patterns.pattern_5` hit of coefficient **5.62e-06** (device control,
`Peak_57851`) and **8.67e-06** (`sorted` repeat, `Peak_43640`), i.e. dust
flickering across the caller's sparsity boundary. By contrast the "changes on
hits ≥1" column counts changes on hits at or above the base run's median hit
coefficient (1.189): 195–242 of them in every arm where the regions moved.
Those are motif calls appearing and disappearing, not rounding.

### Integrity of the order-only comparison

The order-only row carries the weight of the reading, so it was checked rather
than assumed. Between `append_base` and `sorted_base_on_append_gpu` — same
6,460 regions, same GPU, different row order — the caller's own `peaks_qc.tsv`
reports:

- the same 6,460 `peak_name`s on both sides;
- identical `peak_region_start` for all 6,460;
- **bitwise-identical `global_scale` for all 6,460** (max |difference| exactly
  0.0).

`global_scale` is the per-region normalisation the caller derives from that
region's own contributions. Its being bitwise equal on every region proves the
two runs were fed the same data for the same regions, and that the only thing
that differed was the order those regions arrived in.

### This also bears on README finding 2, and contradicts it

Finding 2 states:

> Permuting input order produced the largest coefficient displacement measured
> (**2.07×** the median coefficient) and **zero** discrete flips.

The order-only comparison here *is* that experiment, and on this substrate both
halves come out differently:

| | finding 2 | measured here |
| --- | --- | --- |
| discrete flips under permutation | **zero** | **876**, across 303 of 6,460 regions (4.69%) |
| largest coefficient displacement | 2.07× median | **104.27×** median (max \|delta\| 123.95, median coefficient 1.189) |

36.6% of the 218,851 shared keys move by more than the device null under
permutation, and 2.78% move by more than 1% of the median coefficient. So the
"numeric and discrete divergence are independent axes" claim is not supported
here in the direction it was stated: permutation moved both, and moved the
discrete axis by nearly three orders of magnitude more than the 0–1 flip
instrument floor. Whether the two axes are *independent* is a separate question this
experiment does not settle; what it does settle is that on this substrate
permutation does not produce zero discrete flips.

### Coefficients are not a usable channel here, and the device null understates the noise

On an **identical** re-run of the same file on the same GPU, 422 of 219,203
shared keys move by more than `DEVICE_NULL_ABS_COEFFICIENT_DELTA`, with a max
|delta| of **0.104** — 2.9e5× the 3.63e-07 constant.
The constant was measured GPU-against-CPU on the case study's 2,639-region run;
it does not bound this caller's *same-device repeat* noise on 6,460 regions.

This is visible in the package's own evaluator. Running
`validate.evaluate_stability` at its default tolerance
(`scale_invariance_stability.tsv`):

| comparison | affected peaks | affected hits | hit_jaccard |
| --- | ---: | ---: | ---: |
| `append:base_vs_ctrl` (identical input) | 158 | 5,482 | **1.000000** |
| `append:base_vs_sup_7085` (+9.67%, scale only) | 170 | 6,134 | **1.000000** |
| `append:base_vs_sup_12920` (+100%, scale only) | 167 | 5,765 | **1.000000** |
| `sorted:base_vs_ctrl` (identical input) | 157 | 5,472 | 0.999817 |
| `sorted:base_vs_sup_6525` (+1.01%) | 2,043 | 69,840 | 0.986254 |
| `sorted:base_vs_sup_6783` (+5.00%) | 2,065 | 70,623 | 0.989635 |
| `sorted:base_vs_sup_7085` (+9.67%) | 2,220 | 75,910 | 0.987037 |

The `hit_jaccard` column — hit identity, no tolerance — separates the arms
cleanly. The `affected` columns do not: they flag 158 peaks on a run against
itself. That is the coefficient tolerance firing on the caller's own repeat
noise, and it is why "decision change" is defined here as hit-set change only.

## Reading

**Reproduced.** The same regions do produce different discrete retention
decisions depending on how the caller was invoked on them. The magnitude is
3.4–6.3% of shared regions and 876–1,557 hits in ~219,000, against an instrument
floor of 0–1 change (0.000–0.015%) established by a same-device repeat, a
cross-device repeat, and an independent re-call from a rebuilt input on a third
GPU (0 changes, Jaccard 1.000000). The floor is three orders of magnitude below
the effect, so the instrument can tell them apart. The changes are not confined
to negligible hits: 195–354 of them are on hits of coefficient ≥ 1.0, against a
base-run median hit coefficient of 1.189.

**Not reproduced: the cause, and therefore the onset.** Four results say the
operative variable is not the number of regions:

1. Growing the input by **9.67%** — the reference's own bracket — with every
   base region left in its original row position produces **0** decision
   changes out of 219,203 hits and `hit_jaccard` exactly 1.0.
2. **Doubling** the input to 12,920 regions, same construction, also produces
   **0** decision changes and `hit_jaccard` exactly 1.0. Ten times the
   reference's increment, still nothing.
3. Changing **only the row order**, at exactly 6,460 regions on one GPU,
   produces **876** decision changes across 4.69% of regions.
4. Changing **only `--batch-size`**, at exactly 6,460 regions in exactly the
   base order, produces **1,557** decision changes across 6.32% of regions —
   more than order does. Batch size is not a property of the input set at all,
   so no reading on which "which regions are in the input" is the operative
   variable can accommodate it.

So on this substrate the caller is input-*scale* invariant and input-*schedule*
sensitive. A bisection for an onset in (6,460, 7,085] would not have converged
here: in the `sorted` arm the effect is already at essentially full magnitude
at +1.01% (960 changes) and at +0% (876), and in the `append` arm it is absent
at +9.67% and at +100%. There is no threshold; there is a switch, and the count
is not what throws it.

The mechanism is measured, not merely consistent. FiNeMo holds a rolling buffer
of `--batch-size` regions, refills each converged slot from the input in order,
and each region's optimisation runs in its slot. Appending rows after the base
leaves every base region in the slot and at the step index it had before, so its
trajectory is untouched no matter how many rows follow; interleaving new rows, or
resizing the buffer, moves base regions into different slots at different times.
The caller's own `peaks_qc.tsv` exposes the consequence per region as `num_steps`,
and it behaves exactly as that account predicts: **0** of 6,460 base regions
change iteration count when the input is doubled, against 742 under re-ordering
and 1,122 under re-batching, and 61–64% of the regions whose decisions changed are
regions whose iteration count also changed. What remains untested is the step from
"different iteration count" to "different hit set" for any individual region; the
association is measured, the per-region causation is not.

Whether the reference implementation's bisection was measuring the same
mechanism cannot be settled from here — it ran on different data and its
region-ordering convention is not recorded in this repository. What can be said
is that on the K562 ALLPEAKS substrate with this lexicon and these caller
settings, a scale increase alone does not move a single decision, and the
9.67% figure does not reproduce as a scale threshold.

**The architectural consequence is unaffected, and one guard is now known to be
under-keyed.** "Every specification must be a subset of one frozen run" is
exactly the right rule — this measurement makes it *more* necessary, because
re-calling produces a 4.7% decision shift even when the region count is held
identical. But `guards.single_scale` enforces agreement of `input_scale`, a
*count*. Two different runs of 6,460 regions in different orders carry the same
`input_scale`, pass the guard, and disagree on 876 hits across 303 regions.
The count is not an identity for the run. A content identity (the substrate's
hash, which `validate` already threads around as `lexicon_identity` and which
`compile` already computes for lexicons) would key it; the count cannot.
Flagging, not fixing: changing a guard's contract is not this experiment's
call.

## Reproducing

Neither the region universe nor the lexicon ships with this repository, so the
script has no built-in paths to them — pass them, or set the two environment
variables. The values the table above was produced with are recorded in
`examples/results/run_manifest.json`:

    export MMV_SCALE_UNIVERSE_NPZ=.../region_v3/k562/interpret_cbp_par_hyp/finemo_regions.npz
    export MMV_SCALE_LEXICON_H5=.../casestudy/lexicons/core.h5   # sha256 13d6ff0c83d9…
    export MMV_FINEMO=.../bin/finemo

    python examples/input_scale_invariance.py build --out RUNDIR
    python examples/input_scale_invariance.py call  --out RUNDIR --arm append --gpu <UUID-A>
    python examples/input_scale_invariance.py call  --out RUNDIR --arm sorted --gpu <UUID-B>
    python examples/input_scale_invariance.py call-one --out RUNDIR \
        --regions regions/sorted_base.npz --label sorted_base_on_append_gpu --gpu <UUID-A>
    # the batch-size control: same input, same order, different schedule.
    # any label matching batch_* becomes a `batch_size_only` row in compare.
    python examples/input_scale_invariance.py call-one --out RUNDIR \
        --regions regions/append_base.npz --label batch_1000 \
        --batch-size 1000 --gpu <UUID-A>
    python examples/input_scale_invariance.py compare   --out RUNDIR
    python examples/input_scale_invariance.py stability --out RUNDIR

`call` skips steps whose hit table exists, so the arms are resumable and can run
concurrently on two pinned GPUs. At `--batch-size 2000` each run holds ~44 GB of
GPU memory, so one run per GPU.

## What was not run

Time, not data, was the limit. Runtime is wildly non-linear in *N*: 6,460
regions took ~525 s, 6,525 took 533 s, but 6,783 took 2,979 s, 7,085 took
2,985–3,077 s and 12,920 took 3,084 s. The full planned ladder was
`base, ctrl, +1.01%, +5.00%, +9.67%, +50%, +100%` in both arms. What was run is
exactly the ten rows of `scale_invariance_results.tsv`; **the `sorted` arm's
+50% and +100% steps and the `append` arm's +1.01%, +5.00% and +50% steps were
not run and no number is reported for them.**

The truncation was chosen, not accidental. In the `append` arm the +9.67% and
+100% answers are both exactly zero, so intermediate steps in that arm can only
interpolate between zero and zero. GPU time went instead to the device control
and to `append` +100%, the two runs that decide the reading. In the `sorted`
arm three steps (+1.01%, +5.00%, +9.67%) already show a flat 3.4–3.8%, so the
missing larger steps would test whether the fraction grows with scale — a
question this document does not answer.

The batch-size arm has one point, `--batch-size 1000`. `--batch-size 4000` was
attempted and died in `finemo.hitcaller.fit_contribs` allocating
`importance_scale_batch` — at ~44 GB for 2,000 regions the buffer does not fit
twice over on this GPU. So the batch axis is measured at 2,000 against 1,000 and
nowhere else; whether the effect grows, saturates or reverses with buffer size is
not answered here.

Also not done: no attempt was made to reproduce the reference's bisection
procedure, and the reference implementation's region-ordering convention is not
recorded in this repository, so it cannot be checked whether its ladder was an
`append`-style or a `sorted`-style ladder. That is the single largest
uncertainty in reading this result against the original.
