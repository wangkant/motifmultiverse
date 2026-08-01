# Preregistration — `TRUE_DUPLICATE` v2

**Status: frozen and checksummed. Not yet run on the validation dataset.**

| | |
|---|---|
| Criterion | `TRUE_DUPLICATE`, version `2`, status `FROZEN_DECLARED_HEURISTIC` |
| Frozen file | `src/motifmultiverse/adjudicate/criteria.v2.yaml` |
| **sha256** | `8af0a87963c9f7af276f9a84092f83d5bdc265d1414b4ecd03cd573e57ab319a` |
| Tuning set | K562 ALLPEAKS, 13 Leiden clusters, one ChromBPNet counts-head MoDISco run each; 139 registry nodes |
| Validation set | GM12878, `region_v3/gm12878/interpret_cbp_par_hyp/modisco.h5` and siblings |
| Written | before any GM12878 artifact was opened, and before any number in this document was computed from one |
| In force by default? | **No.** The default registry is `criteria.v1.yaml`; this criterion applies only when `--criteria` names `criteria.v2.yaml`. It collapses motifs, and deletion is the one error direction a reader of a compiled lexicon cannot undo. |

> **Note on one stale line inside the frozen file.** `criteria.v2.yaml`'s own
> header comment still reads "This is the registry the package ships and loads by
> default". It was true when the file was sealed and is not true now. It is left
> uncorrected **on purpose**: the sha256 above covers the file's bytes, comments
> included, so editing that sentence would invalidate this registration and
> require re-registering the criterion — which `FP-13` reserves for a change to
> the *rule*, not to a comment. The authoritative statement of what is default is
> the row above, `adjudicate/README.md`, and `adjudicate.CRITERIA_RESOURCE`.

This document exists because of a specific failure, named plainly in the next
section. Everything after it is either a number computed on the tuning set, a
prediction about the validation set, or a statement of what would count as the
prediction being wrong.

---

## 1. The failure this document is answering

`FP-13` requires that

> the parameters of any merge/split decision rule must be written down and
> checksummed **before that pair's result is seen** … and the parameters may not
> be retuned after the fact.

The criterion's two declared magnitudes — `ppm_similarity ge 0.90` and
`overlap_bp ge 8` — were **anchored on the K562 run's own distribution** (the
q90 of its registered edges, and the median of its trimmed cores) and then
**validated by sweeping collapse outcomes on that same run**. The rule was
adjusted until the answers looked stable, and the stability of those answers was
then offered as the reason to believe the rule. The criterion file cited `FP-13`
in its header while doing this and never confronted the clause it was violating.

That is the standard this package applies to other people's work. So:

> **Nothing measured on K562 is evidence that this criterion transfers.**
> Section 5 lists exactly what is contaminated. The transfer claim is settled in
> Section 3 against GM12878, a cell type this criterion has never been run on,
> or it is not settled at all.

A second thing has to be said in the same breath, because it limits what the
GM12878 run can deliver even if it goes perfectly. See Section 6: the experiment
can falsify **transfer**. It cannot establish **correctness**, and no experiment
currently in this package can.

---

## 2. The criterion, as frozen

The executable content, complete. `basis` prose is omitted from this table and
reproduced verbatim in Appendix A; the sha256 above covers the file byte for
byte.

| # | field | operator | value | provenance | resolves from |
|---|---|---|---|---|---|
| 1 | `at_alignment_null_floor` | `is_true` | — | *(no magnitude)* | `empirical_p_value <= 1/(null_shuffles+1)` |
| 2 | `overlap_frac_source` | `ge` | `1.0` | `derived` | `evidence_domain: overlap_frac_source, endpoint: max` |
| 3 | `overlap_frac_target` | `ge` | `1.0` | `derived` | `evidence_domain: overlap_frac_target, endpoint: max` |
| 4 | `signed_cwm_similarity` | `ge` | `0.0` | `derived` | `evidence_domain: signed_cwm_similarity, endpoint: sign_boundary` |
| 5 | `ppm_similarity` | `ge` | `0.90` | **`declared`** | — chosen by a maintainer |
| 6 | `overlap_bp` | `ge` | `8` | **`declared`** | — chosen by a maintainer |

- `insufficient_evidence_action: deferred` — missing evidence never licenses a
  collapse and never asserts a refusal.
- `decision_if_matched: collapse`.
- All six predicates must hold. Predicates 1–6 are also re-read by
  `adjudicate.edge_admits_duplicate_candidate` as the edge filter that proposes
  connected components (`FP-05`'s declared distance ceiling). **This matters for
  Section 4.3** — the rule is not only a decision procedure, it is also what
  decides which pairs are considered together, so its response is not monotone.

### 2.1 What `derived` now means

Three of the four magnitudes are labelled `derived`. Until this change that label
was pure author self-report: a review wrote

```yaml
- field: ppm_similarity
  operator: ge
  value: 0.7314159
  provenance: derived
  basis: "It follows from the structure of the problem."
```

and the loader accepted it, inside a criterion the package then reported as
`FROZEN`. It now refuses it. A `derived` threshold must carry a `derived_from`
naming a landmark of its own field's validated range, the loader **recomputes**
the value from `schema.EVIDENCE_FIELD_DOMAINS` — the same table
`align.AlignmentEvidence` validates the data against, not a second copy — and a
mismatch refuses the file.

**What that establishes, exactly:** the number was not freely chosen; it is an
endpoint, or the sign boundary, of the interval the field is held to.
**What it does not establish:** that gating there is the right rule.
`overlap_frac_source ge 1.0` is checkably the top of `[0.0, 1.0]`; whether full
bilateral containment means two motifs are the same motif is an argument, and the
argument lives in `basis` and is read by a person. A magnitude that cannot be
written as such a landmark is not derived and must be `declared` — and
`declared` claims nothing.

Tests: `tests/test_derived_provenance.py` (12).

---

## 3. Predictions for GM12878

### 3.1 How these are stated

I have not opened any GM12878 artifact, so I do not know its registry size. Every
prediction is therefore given **as a rate**, with the K562 value it is
extrapolated from; absolute counts are given only under an assumption that is
labelled as one.

Write `N` for the GM12878 registry's node count and `E` for its registered edge
count, both measured by whoever runs this. K562: `N = 139`, `E = 5,171`, from
`9,591` unordered pairs.

> **Assumption A (for the absolute columns only):** the GM12878 registry is built
> the same way — per-Leiden-cluster MoDISco on one ChromBPNet counts-head model —
> and `N` lands in `[70, 280]`, i.e. within a factor of two of K562's 139. If `N`
> falls outside that, read only the rate columns. Assumption A being wrong is not
> the criterion failing.

### 3.2 The edge funnel

Computed by me on K562 from `alignment_edges.parquet` at the shipped operating
point:

| stage | K562 | as a share | predicted GM12878 share |
|---|---|---|---|
| registered edges | 5,171 / 9,591 pairs | 53.9% | 35 – 70% |
| … at the null floor | 272 | 5.3% of `E` | 2 – 12% |
| … bilateral (both `overlap_frac` = 1.0) | 312 | 6.0% of `E` | 3 – 12% |
| … bilateral **and** at floor | 107 | 2.07% of `E` | 0.8 – 4% |
| … **and** `signed_cwm ≥ 0` | 106 | 2.05% of `E` | 0.8 – 4% |
| … **and** `overlap_bp ≥ 8` | 92 | 1.78% of `E` | 0.5 – 3.5% |
| … **and** `ppm_similarity ≥ 0.90` | 92 | 1.78% of `E` | 0.5 – 3.5% |

### 3.3 Decisions

| quantity | K562 | predicted GM12878 (rate) | predicted (absolute, under Assumption A) |
|---|---|---|---|
| adjudication decisions | 9 | — | **4 – 16** |
| `collapse` | 4 | ~1 per 35 nodes | **1 – 9** |
| `refuse_merge` | 4 | — | **1 – 9** |
| `deferred` | 1 | — | **1 – 3** |
| motifs removed | 9 | 6.5% of `N` | **3 – 12% of `N`** |
| core lexicon | 139 → 130 | — | `N` → `N` − (3–12% of `N`) |
| largest collapse component | 5 members | — | **2 – 7 members** |

### 3.4 Structural predictions

**P1 — Exactly one giant deferred component.** K562 produced one 115-node
component (82.7% of the registry) deferred for non-transitivity: single linkage
over a dense similarity graph fuses almost everything, and the transitivity check
refuses to let connectivity alone license a merge. Predict **exactly 1** such
component in GM12878, containing **55 – 95%** of `N`.

**P2 — `ppm_similarity` is inert.** At `overlap_bp ≥ 8` on K562, the
`ppm_similarity` gate removes **zero** edges: the three surviving edges below 0.90
all have `overlap_bp = 7` and are already excluded. The collapse set is
bit-identical for every `S ∈ [0.70, 0.95]`. Predict the same in GM12878 — the
collapse set is invariant over `S ∈ [0.70, 0.95]` at `overlap_bp = 8`.
*Confidence is lower than it sounds:* the minimum `ppm_similarity` among K562's 92
fully-passing edges is **0.90118**, which clears the declared 0.90 by 0.0012. One
edge sitting that close means inertness is a property of this dataset, not of the
rule.

**P3 — The sign gate fires rarely.** On K562 it removes 84 of 312 bilateral
edges, but only **1 of the 107** that also reach the null floor. Predict it
removes **0 – 3** edges at the operating point in GM12878, and that **no collapse
mixes a pos-metacluster node with a neg-metacluster one** (0, structurally
enforced — see F5).

**P4 — Which families collapse.** CTCF is the most reproducibly rediscovered
motif in this class of run and accounted for two of K562's four collapses
(a 5-way component, and the CTCF/CTCFL pair). Predict:

- **CTCF is among the collapsed families in GM12878** — the single highest-confidence
  prediction here.
- The collapsed set is dominated by **constitutive / ubiquitous** factors — CTCF,
  GC-box SP/KLF, ETS, NFY, CpG-promoter-type motifs — rather than by
  GM12878-lineage-specific factors (EBF1, SPI1/PU.1, IRF4, NF-κB, POU2F2, RUNX3).
  Rationale: a motif has to be rediscovered in ≥2 clusters to be a duplicate
  candidate at all, and constitutive factors are the ones that recur across
  clusters. A lineage factor concentrated in one cluster cannot produce a
  cross-cluster duplicate.
- **At least one collapse merges a paralog pair or two same-family members that
  TomTom annotates separately.** This is the criterion's known destructive
  failure mode (`KNOWN COST 1`), and it is not fixed. Predicting it is not a
  prediction of success.

**P5 — `overlap_bp` plateau.** K562 has a plateau: `bp ∈ {8, 10, 12}` give an
identical collapse set. Predict GM12878 also has a plateau containing `bp = 8`,
i.e. at least `bp = 8` and `bp = 10` agree.

---

## 4. The K562 measurements these predictions extrapolate from

Everything in this section is the tuning set. It is here so the predictions are
auditable, **not** as evidence that the criterion is right.

### 4.1 Operating point

`ppm_similarity ≥ 0.90`, `overlap_bp ≥ 8`, on 139 nodes:

- 9 decisions: 4 `collapse`, 4 `refuse_merge`, 1 `deferred`
- 9 motifs removed, 139 → 130
- collapse-membership fingerprint `f4362c36dbd3`
- 17 node pairs merged (transitive closure of the 4 components)

### 4.2 Full sweep — 54 cells

Each cell is a complete re-run through the real `adjudicate` and `compile`
stages on this HEAD. Values are **motifs removed** (139 minus core lexicon size);
letters mark distinct collapse-membership fingerprints.

| `ppm \ bp` | 6 | 7 | **8** | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| 0.70 | 13 ᵃ | 11 ᵇ | **9 ᶜ** | 9 ᶜ | 9 ᶜ | 1 ᵈ |
| 0.80 | 13 ᵃ | 11 ᵇ | **9 ᶜ** | 9 ᶜ | 9 ᶜ | 1 ᵈ |
| **0.90** | 13 ᵃ | 11 ᵇ | **9 ᶜ ← shipped** | 9 ᶜ | 9 ᶜ | 1 ᵈ |
| 0.93 | 13 ᵃ | 11 ᵇ | **9 ᶜ** | 9 ᶜ | 9 ᶜ | 1 ᵈ |
| 0.95 | 13 ᵃ | 11 ᵇ | **9 ᶜ** | 9 ᶜ | 9 ᶜ | 1 ᵈ |
| **0.96** | **14** ᵉ | **12** ᶠ | **10 ᵍ** | 9 ᶜ | 9 ᶜ | 1 ᵈ |
| **0.97** | **14** ᵉ | **12** ᶠ | **10 ᵍ** | 9 ᶜ | 9 ᶜ | 1 ᵈ |
| 0.98 | 11 ʰ | 9 ⁱ | 7 ʲ | 6 ᵏ | 6 ᵏ | 0 |
| 0.99 | 5 ˡ | 4 ᵐ | 2 ⁿ | 1 ᵒ | 1 ᵒ | 0 |

Fingerprints: ᶜ = `f4362c36dbd3`, ᵍ = `37ed841cb2bb`.

Two counts that are easy to conflate, so both are stated. The **declared box**
`ppm [0.70, 0.95] × bp [8, 12]` is **15 cells**, and that is the flatness claim
the criterion file makes. Fingerprint `f4362c36dbd3` occurs in **19** cells —
those 15 plus `(0.96, 10)`, `(0.96, 12)`, `(0.97, 10)`, `(0.97, 12)`, which lie
outside the box. The larger number is not the stronger claim: the box is the
region the declared pair was argued to be safe to move within, and it is the one
being registered.

### 4.3 The response is NOT monotone, and an earlier table hid it

**Raising `ppm_similarity` from 0.95 to 0.96 removes MORE motifs — 10, against
the plateau's 9.** An earlier version of this table reported only
`0.99 → 2` and omitted 0.96 and 0.97 entirely, which reads as "stricter is safer"
and is false here.

The cause is structural, not incidental. The criterion's predicates are also the
component-proposal edge filter (Section 2), so raising a threshold does not
merely drop candidate pairs — it **re-partitions the graph**. Traced exactly:

- At `S = 0.95`, the pair `cl2::pos_patterns.pattern_6` /
  `cl3::pos_patterns.pattern_6` sits inside an **11-node component** that is
  refused for `family_conflict` (`AMBIGUOUS_CROSS_FAMILY`).
- At `S = 0.96`, the edges lost to the higher threshold **split that component**
  into a 9-node refusal plus this pair alone. Alone, it has no family conflict
  left to refuse it, and it collapses.
- The other four collapses are unchanged. Component sizes go
  `[115 d, 8 r, 11 r, 5 c, 7 r, 3 c, 3 c, 2 c]` → `[115 d, 8 r, 9 r, 5 c, 7 r, 3 c, 2 c, 3 c, 2 c]`.

**Consequence for a reader:** flatness *inside* the box
`ppm [0.70, 0.95] × bp [8, 12]` is real (15 identical cells). Monotonicity at the
box's edge is not, and only the first was ever measured. A reviewer who reasons
"0.96 would be more conservative than 0.90" would be wrong.

### 4.4 A case-study number changed under the criterion

`overlap_bp = 6` removed **12** motifs when the criterion was written and removes
**13** on this HEAD. The criterion did not change. The short-motif annotation
rule now measures the trimmed core rather than the padded MoDISco window, 68 of
6,430 annotation candidates changed their `low_confidence_annotation` flag, and
one family resolution moved. This is the ordinary reason a case-study figure must
name the commit it was measured on, and it is why the sweep above was recomputed
rather than copied.

---

## 5. What is tuned on K562 and therefore cannot be evidence on K562

**Contaminated — chosen or validated by looking at K562 answers:**

1. `ppm_similarity ge 0.90` — anchored on K562's own q90 of registered edges
   (0.8721).
2. `overlap_bp ge 8` — anchored on K562's own median trimmed-core length
   (q50 = 8).
3. The claim that the answer is *flat over a wide box* — the box was found by
   sweeping K562 collapse outcomes.
4. The choice of **8** rather than 10 or 12 — a reading of where K562's plateau
   starts, described in the file as "the aggressive end of the plateau". A
   preference, expressed over measured outcomes.
5. The "every predicate is load-bearing" demonstration — each perturbation in
   `tests/test_true_duplicate_criterion.py` is a K562 pair chosen because the
   criterion decides it a particular way.

**Not contaminated — fixed before any outcome was seen, and machine-resolvable:**

6. `overlap_frac_source ge 1.0`, `overlap_frac_target ge 1.0` — the max of the
   field's validated domain, and the boundary `adjudicate` already used to split
   `TRUE_DUPLICATE` from `FRAGMENT_MATCH`.
7. `signed_cwm_similarity ge 0.0` — the sign boundary of a `[-1.0, 1.0]` field.
8. `at_alignment_null_floor` — the instrument's own resolution,
   `1/(null_shuffles+1)`; it moves with `--null-shuffles` and carries no chosen
   alpha.

The split in 1–8 lines up exactly with the `declared` / `derived` labels, which
is the one thing the labelling scheme was supposed to buy.

---

## 6. What would falsify this — and what would not

### 6.1 Falsifiers

Any of these, on GM12878, means the criterion does not transfer. They are ordered
most to least decisive.

| | outcome | what it would mean |
|---|---|---|
| **F1** | **0 collapse decisions**, or motifs removed < 1% of `N` | The rule is K562-specific. `overlap_bp ≥ 8` is anchored on K562's median core; if GM12878's cores run shorter the gate zeroes the criterion out, and the package is back to shipping an undeduplicated lexicon while claiming a frozen rule. |
| **F2** | Motifs removed **> 25% of `N`**, or a collapse component with **> 10 members** | The geometry gates do not discriminate on GM12878. Since the error direction is deletion, this is the expensive failure. |
| **F3** | The collapse set is **not** invariant over `ppm ∈ [0.70, 0.95]` at `bp = 8` | Contradicts P2. The declared magnitude would be doing work in GM12878 that it demonstrably never did in K562, so "flat over a wide box" was a K562 artifact and the number was tuned, not merely inert. |
| **F4** | `bp = 8` and `bp = 10` give **different** collapse sets | Contradicts P5. "8 is the lower edge of a plateau" would then be a K562 fact and not a property of the rule, and there is no principled reason left to prefer 8. |
| **F5** | **Any** collapse mixes a pos-metacluster and a neg-metacluster node | Falsifies the *implementation*, not the tuning. Predicate 4 is supposed to make this unreachable. If it happens, the sign gate is wrong. |
| **F6** | **0 `refuse_merge` decisions** | The package's entire measured contribution on K562 is refusal (Section 7). A run that refuses nothing is a run where the criterion is a similarity threshold with extra steps. |
| **F7** | GM12878's giant deferred component is **absent**, or there are **≥ 2** of them | Contradicts P1, and would mean the connectivity structure the transitivity check was written against does not recur. |

### 6.2 What would NOT count against it, stated so it cannot be claimed later

- **Assumption A failing** (`N` outside `[70, 280]`) — that is about the registry,
  not the rule. Read the rate columns.
- **Different families collapsing than P4 predicts**, other than CTCF's absence.
  P4 is a biological expectation, weakly held; only "CTCF is among them" is
  offered with confidence.
- **A paralog merge occurring** (P4, third bullet). That is a *predicted known
  failure*, already recorded as `KNOWN COST 1`. It confirms the criterion behaves
  as documented; it does not make the criterion good.
- Counts landing just outside the Section 3.3 ranges. Those ranges are
  extrapolations from a single dataset and are stated wide on purpose; a miss by
  one or two is uninformative, which is why F1/F2 are set at the qualitative
  boundaries instead.

### 6.3 The part that is not falsifiable, and I am not inventing a test for it

Every falsifier above tests **transfer**: does the rule behave on new data the way
it behaved on the data it was tuned on. None of them tests **correctness**: are
the motifs it merges actually the same motif.

That is not an oversight, and it cannot be fixed by running GM12878:

- There is no ground truth for "these two MoDISco patterns are the same motif" in
  GM12878 any more than in K562. No labelled same/different pair set exists in
  this package; the criterion's own `replacement_evidence` names its construction
  as the outstanding item.
- The one downstream test that could stand in for ground truth — reconstruction
  loss — **was measured and has no power**. See Section 7.
- So a GM12878 run that satisfies every prediction in Section 3 establishes that
  the criterion is *consistent*. It does not establish that it is *right*, and
  no result from it may be reported as though it did.

If that reads as a weak claim, it is the accurate one. The alternative was to
name a GM12878 outcome as a correctness test when nothing about GM12878 makes it
one.

---

## 7. The naive baseline — the criterion's contribution is refusal

A reader deciding whether to trust this needs this section, and it is not
flattering.

**The package beats the obvious alternative.** Plain TomTom all-against-all
single-linkage merging at the conventional `q < 0.05` collapses the same 139
K562 motifs to **19** entries, absorbing 120, with a largest merged group of 65
held together by 3 bridge edges. It loses two thirds of the hit table
(93,661 → 33,934 hits, Jaccard 0.047) and makes **99.89%** of peaks reconstruct
worse. The criterion does not do that.

**But its contribution is entirely refusal.** At the shipped operating point,
computed by me on the 4 collapse components:

- 17 node pairs merged by the criterion.
- **0** of them are pairs TomTom would not also have merged at `q < 0.05` — in
  either direction, and in fact in both.
- **674** pairs TomTom merges that the criterion does not.

The criterion proposes nothing TomTom does not propose. Every merge it makes is a
strict subset of the naive baseline's. What it contributes on this registry is
**declining 674 of the naive baseline's 691 merges**, for stated reasons
(`FRAGMENT_MATCH` geometry, family conflict, sign flip, non-transitivity). That
is a real and valuable contribution. It is not motif discovery, and this package
should not be described as if the merging were the clever part.

**And the reconstruction test cannot see the difference between its merges and
arbitrary deletions.** Measured on a 134-entry configuration (5 motifs absorbed):

| lexicon | hit Jaccard | ΔNLL mean | 95% CI |
|---|---|---|---|
| package, 5 absorbed | 0.9508 | +5.42e-4 | (−8.86e-4, +1.87e-3) |
| **5 random motifs deleted**, comparable mass | 0.9679 | **+2.41e-4** | (−6.63e-4, +1.19e-3) |
| naive TomTom, 19 entries | 0.0466 | +3.54e-2 | (+3.43e-2, +3.66e-2) |

Deleting five random motifs of comparable mass (737 hits, 0.787%, against the
package's 576, 0.615%) scores **better** on the point estimate and on Jaccard,
with a confidence interval spanning zero — statistically indistinguishable from
the criterion's own merge. The naive 19-entry lexicon is separated from both by
two orders of magnitude, so the test is not inert; it simply has no resolution at
the scale the criterion operates at.

> **Therefore: "no detectable reconstruction loss" is a necessary condition for
> duplication, not a sufficient one.** Nothing in this package may be written as
> though passing that test is evidence that a merge was correct. It is evidence
> that the merge was small.

**Two caveats on the table above, so it is not over-read.** (i) It was computed
on a 134-entry configuration — the criterion with reconstruction evidence fed
back — not on the shipped 130-entry one, so the ΔNLL row is not the shipped
operating point's. The strict-subset finding above **is** at the shipped
operating point; I recomputed it. (ii) All hit-level evidence on this run is
cluster 0 only, 2,639 of ~33,917 K562 peaks.

---

## 8. The two known costs, carried here from the criterion file

**Cost 1 — it merges paralogs, and the error direction is deletion.** One of the
four K562 collapses merges CTCF with CTCFL/BORIS. Both resolve to
`C2H2_ZINC_FINGER_FACTORS` so `family_conflict` never fires, and the pair clears
every predicate comfortably (`overlap_bp` 14, both overlap fracs 1.0,
`ppm_similarity` 0.97413, `signed_cwm_similarity` 0.981337, at the null floor).
In K562, CTCFL is a real, separately regulated factor. That is 25% of the
criterion's output, and a deleted motif is irreversible for the reader of the
lexicon in a way an inflated one is not. TomTom separates the two by five orders
of magnitude, but no evidence field carries that, so no predicate can read it.
Pinned as a passing test in `tests/test_true_duplicate_criterion.py`.

**Cost 2 — an `FP-08` regression, previously unmentioned.** `FP-08` requires a
redundancy claim to carry **both** a coefficient-share field and a
reconstruction-gain field, and its audit test fails if either is absent. v1's
`TRUE_DUPLICATE` required both `paired_delta_reconstruction_affected` and
`affected_coefficient_share`. **v2 requires neither.** Dropping reconstruction is
argued — Section 7 shows it lacks power. Dropping `affected_coefficient_share`
was never argued and was silently unmentioned.

The honest account is mechanical rather than principled, and I measured it:
putting **either** field back into `required_evidence` takes the criterion from
**4 collapses to 0** on K562 — every component defers for missing evidence and
the lexicon stays at 139, which is exactly v1's state. `validate` binds a
`StabilityResult` to a single `decision_id` and this run has one such row,
matching none of these components. So v2 collapses motifs on geometry alone,
having checked neither piece of downstream redundancy evidence the design says a
collapse needs. That is the price of the criterion functioning at all, and the
per-pair `decision_id` named in `replacement_evidence` is what would let a future
version pay it properly.

---

## 9. Protocol for the validation run

1. Do not modify `criteria.v2.yaml`. Verify its sha256 matches the value in the
   header of this document before running anything.
2. Run `ingest` → `align` → `annotate` → `adjudicate` → `compile` on GM12878 with
   `--criteria` pointing at the packaged `criteria.v2.yaml` and
   `--null-shuffles 1000`. **Not the default criteria**: the default is
   `criteria.v1.yaml`, which defers every duplicate, so a default run would test
   nothing in this document.
3. Record: `N`, `E`, the edge funnel of Section 3.2, the decision counts of
   Section 3.3, and the collapse membership of every component.
4. Re-run `adjudicate` at `ppm ∈ {0.70, 0.80, 0.90, 0.95}` × `bp ∈ {8, 10}` to
   settle F3 and F4.
5. Report every falsifier in Section 6.1 as fired or not fired, **before**
   interpreting anything.
6. **If a falsifier fires, that is the result.** `FP-13`: when the rule returns an
   answer contrary to the prior, that answer is the conclusion, and the
   parameters may not be retuned. A GM12878 run that triggers F1 or F3 means the
   declared magnitudes do not transfer — it does not mean GM12878 needs different
   ones.

---

## Appendix A — the predicate block, verbatim

Reproduced byte for byte from `src/motifmultiverse/adjudicate/criteria.v2.yaml`
(sha256 in the header). Comments and `basis` prose included.

<!-- BEGIN VERBATIM PREDICATE BLOCK -->
```yaml
  - criterion_id: TRUE_DUPLICATE
    version: "2"
    status: FROZEN_DECLARED_HEURISTIC
    relationship: TRUE_DUPLICATE
    required_evidence:
      - at_alignment_null_floor
      - overlap_frac_source
      - overlap_frac_target
      - overlap_bp
      - signed_cwm_similarity
      - ppm_similarity
    predicates:
      # --- DERIVED: the instrument's own resolution, not a chosen alpha --------
      # No provenance/basis keys, and that is deliberate: `is_true` is not an
      # ordered comparison and carries no magnitude, so the loader REFUSES
      # provenance on it. The derivation lives in the helper that computes the
      # field -- adjudicate.at_alignment_null_floor, which is
      # `empirical_p_value <= 1/(null_shuffles+1)`, i.e. "the observed alignment
      # was not matched by any shuffle of this exact pair". Writing
      # `empirical_p_value le 0.001` instead would have been a chosen alpha that
      # silently means something different at a different --null-shuffles; on
      # this run the two select the identical 272 edges
      # (casestudy/distributions.tsv reports 272 at the 1/1001 floor).
      - field: at_alignment_null_floor
        operator: is_true

      # --- DERIVED: structural definitions, not tuned cuts --------------------
      - field: overlap_frac_source
        operator: ge
        value: 1.0            # 1.0 = "the whole trimmed core is inside the alignment"; the
                              # boundary adjudicate ALREADY uses to split TRUE_DUPLICATE from
                              # FRAGMENT_MATCH. 312/5171 edges meet the bilateral form
                              # (casestudy/distributions.tsv).
        provenance: derived
        derived_from: {evidence_domain: overlap_frac_source, endpoint: max}
        basis: >-
          RESOLVED: 1.0 is the top of the interval align validates this field into
          ([0.0, 1.0], schema.EVIDENCE_FIELD_DOMAINS), so the loader recomputes it
          rather than taking this label on trust. What that check establishes is
          only that the number was not chosen -- it is the field's own ceiling.
          The ARGUMENT, which no loader can check, is that the ceiling is the
          right place to gate: 1.0 is the definition of "the whole of this motif's
          trimmed core is inside the alignment", and adjudicate_component already
          uses exactly this boundary to separate TRUE_DUPLICATE from
          FRAGMENT_MATCH (`overlap_frac_source < 1.0 or overlap_frac_target < 1.0`
          -> FRAGMENT_MATCH), so restating it here introduces no number the
          package did not already contain. On the K562 run 312 of 5171 registered
          edges satisfy the bilateral form (casestudy/distributions.tsv).
      - field: overlap_frac_target
        operator: ge
        value: 1.0            # same definition, other side. Source: as above.
        provenance: derived
        derived_from: {evidence_domain: overlap_frac_target, endpoint: max}
        basis: >-
          Bilateral form of the same definition, and resolved the same way against
          its own field's ceiling. Required separately because a one-sided 1.0 is a
          fragment inside a longer parent, which is FRAGMENT_MATCH's case, not this
          one.
      - field: signed_cwm_similarity
        operator: ge
        value: 0.0            # the SIGN boundary, not a magnitude. q25 of this field over
                              # the 5171 edges is -0.4641 (casestudy/distributions.tsv), so
                              # sign-flipped pairs are common and this gate is load-bearing:
                              # it removes 84 of the 312 bilateral edges (merge/sweep.py).
        provenance: derived
        derived_from: {evidence_domain: signed_cwm_similarity, endpoint: sign_boundary}
        basis: >-
          RESOLVED: this field is validated into [-1.0, 1.0]
          (schema.EVIDENCE_FIELD_DOMAINS), a range that straddles zero, so 0.0 is
          its sign boundary and the loader recomputes it. The `sign_boundary`
          endpoint is refused on a field that never changes sign, which is what
          stops this from being a floor dressed up as a sign test.
          Zero is the sign boundary, not a magnitude. align registers on UNSIGNED
          ppm precisely so that sign-flipped pairs stay visible rather than being
          silently matched (align: registered_on='unsigned_ppm'); q25 of
          signed_cwm_similarity over the 5171 registered edges is -0.4641
          (casestudy/distributions.tsv). Collapsing across the sign merges a
          driver with a repressor. On the 312 bilateral edges this gate removes 84
          (312 -> 228), and it is why no collapse in the rule sweep mixes a
          pos-metacluster node with a neg-metacluster one.
          Its MARGINAL effect at the shipped operating point is much smaller and
          is stated here so the 84 cannot be read as this gate's contribution:
          once the null-floor gate has also been applied, 107 bilateral edges
          remain and the sign gate removes exactly ONE of them (107 -> 106). That
          one edge is the cl0/pattern_11 -- cl7/neg pattern_1 pair, 0.977 similar
          on unsigned ppm and -0.976 on the signed CWM, pinned in
          tests/test_true_duplicate_criterion.py. A gate that fires once is still
          worth having when the thing it prevents is merging a driver into a
          repressor, but it is not doing 84 edges' worth of work.

      # --- DECLARED: chosen by a maintainer -----------------------------------
      - field: ppm_similarity
        operator: ge
        value: 0.90           # DECLARED. Anchored to this run's q90 = 0.8721 over all 5171
                              # edges (casestudy/distributions.tsv). MEASURED CAVEAT: the run
                              # is bit-identical for any S in [0.70, 0.95] at overlap_bp=8 --
                              # and NOT monotone above it: S=0.96 removes MORE motifs (10)
                              # than the plateau (9). Full table and mechanism in
                              # docs/MERGE_CRITERION_PREREGISTRATION.md.
        provenance: declared
        basis: >-
          DECLARED. Anchored to the run's own distribution -- 0.90 sits just above
          q90 of all 5171 registered edges (0.8721,
          casestudy/distributions.tsv), so it reads "top decile of what alignment
          registered at all". Measured caveat, not assumed: this predicate is
          nearly INERT at the shipped operating point, because the derived
          null-floor gate above already implies high similarity. At overlap_bp=8
          the whole run is bit-identical for every value in [0.70, 0.95] --
          collapse_set_fingerprint f4362c36dbd3, 139 -> 130 core motifs, for S in
          {0.70, 0.80, 0.90, 0.93, 0.95}.
          NOT MONOTONE, and this was omitted from an earlier draft of this file:
          raising S past the plateau removes MORE motifs, not fewer. S=0.96 and
          S=0.97 each remove 10 (139 -> 129, fingerprint 37ed841cb2bb) against the
          plateau's 9. The cause is structural rather than incidental --
          `adjudicate.edge_admits_duplicate_candidate` re-reads these same
          predicates as the component-proposal edge filter (FP-05's distance
          ceiling), so raising a threshold does not merely drop candidate pairs, it
          RE-PARTITIONS the graph. At S=0.95 the pair (cl2/pattern_6,
          cl3/pattern_6) sits inside an 11-node component refused for
          family_conflict; at S=0.96 the lost edges split that component into a
          9-node refusal plus this pair alone, which then has no conflict left to
          refuse it and collapses. A reader must not assume a stricter-looking
          threshold is a more conservative one. Above that, S=0.98 removes 7 and
          S=0.99 removes 2. All 54 swept cells are in
          docs/MERGE_CRITERION_PREREGISTRATION.md, re-measured end to end through
          the real adjudicate and compile stages on this HEAD.
          0.90 is kept as a floor that does not depend on shuffle count, so a
          future run with a coarser null is still gated -- not because it is doing
          work here. A release that promoted it to `derived` on the strength of
          this run would be promoting an untested number, and the loader would
          refuse it anyway: 0.90 is not a landmark of ppm_similarity's [-1.0, 1.0]
          domain, which is exactly the check that makes `derived` mean something.
      - field: overlap_bp
        operator: ge
        value: 8              # DECLARED, and the only chosen number that moves the answer
                              # WITHIN the plateau. 8 = median registry trimmed-core length
                              # (q50 = 8, casestudy/distributions.tsv "registry: trimmed_core
                              # length"), above align's 6 bp bilateral admission floor.
                              # MEASURED at S=0.90: B=6 -> 13 motifs removed, B=7 -> 11,
                              # B=8/10/12 -> 9, B=14 -> 1. 8 is the LOWER, more aggressive
                              # edge of the B=8..12 plateau.
        provenance: declared
        basis: >-
          DECLARED. Under the two bilateral 1.0 predicates, overlap_bp IS the
          shared trimmed-core length, so a `ge` expresses a minimum core length
          with no new operator. 8 bp is the median trimmed core in the K562
          registry (q50 = 8, casestudy/distributions.tsv) and sits above align's
          6 bp bilateral admission floor, which the case study flags as written
          for longer motifs. Measured through the real adjudicate + compile stages
          with THIS criterion at S=0.90, on this HEAD: B=6 -> 13 motifs removed,
          B=7 -> 11, B=8/10/12 -> 9, B=14 -> 1
          (docs/MERGE_CRITERION_PREREGISTRATION.md, sweep table). The extra
          collapses at B=6-7 are 6-7 bp ETS cores (bare GGAA half-sites), where
          the criterion would be merging detectors the package cannot show are the
          same. There is a plateau at B=8..12; 8 is its LOWER, more aggressive
          edge, and a reviewer preferring 10 or 12 would be reading the same
          plateau more conservatively. It is reasoning, not a derivation.
          The B=6 figure was 12 in an earlier draft of this file. It is 13 on this
          HEAD, because the short-motif annotation rule now measures the trimmed
          core rather than the padded window and 68 of 6,430 annotation candidates
          changed their low_confidence flag as a result, which moved one family
          resolution. The number changed under the criterion without the criterion
          changing -- which is the ordinary reason a case-study figure has to name
          the commit it was measured on.
    insufficient_evidence_action: deferred
    decision_if_matched: collapse
```
<!-- END VERBATIM PREDICATE BLOCK -->
