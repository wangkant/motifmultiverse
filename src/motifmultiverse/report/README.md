# `report`

## The rule

Every number rendered carries its denominator, its baseline population, and its provenance.

## The failure that produced it

The same data supported both `replicates exactly` and `4x stronger, prediction falsified`, differing only in whether the baseline was the unselected universe or a residual subset. A bootstrap resolution floor was also printed as though it were a measured p-value.

## How to check it

The renderer refuses a figure with no denominator and no `baseline_population` field; the bias ledger is rendered from the `bias_ledger.tsv` packaged beside this module (`report.packaged_bias_ledger_path`).

---

## Status: implemented, markdown only

`motifmultiverse report <interpretation>/ --out report/` renders one markdown document
from the artifacts a stage actually wrote — the `interpretation.json` in that directory
and the `provenance.json` log beside it — plus the packaged `bias_ledger.tsv`. Section order
comes from the interpretation's own `emitted_order`, not from a list the renderer holds.

**What it renders.** Every number is `str()` of a recorded field, printed beside the
denominator the producing stage recorded and *named* (`n_submitted = 8277`). No ratio is
recomputed here: a renderer that recomputes is a second implementation of the statistics
that can disagree with the first. Composition and effects are branched on `composition is
None` and `effects is None` — never on `floor_failures` being non-empty, which is how an
earlier draft made a run whose composition was perfectly good render as though it had no
numbers. A `p_value` of `None` renders as `WITHHELD — inference_capability =
ESTIMATION_ONLY`, never blank and never `n.s.`, because a blank reads as *no evidence of
an effect*. `two_part_effects = None` renders as "nobody chose a definition of 'used'",
which is not "computed and found nothing". Both permission axes always print: `claim_scope`,
and `output_mode` under an explicit DEPRECATED label saying it cannot represent
`SUBSTRATE_CIRCULAR`. The top-level `floor_failures` and `contrast_health.floor_failures`
are both shown rather than collapsed; they are allowed to diverge (`docs/DATA_MODEL.md`).

**What it refuses** (exit `4`): `--html` and `--docx`, because rendering markdown for a
caller who asked for HTML is the gap between what was specified and what ran that this
package exists to close; an artifact whose deprecated `health` alias disagrees with
`query_health`; and a bias ledger that is absent, is not 4 columns, or whose `axis_id`
column is not exactly `BA-01`…`BA-20`.

**What it will not do.** No absent field gets a default. `baseline_population` — which
the rule above demands — is carried by no artifact in this package, so it renders as the
literal token `NOT RECORDED` in a mandatory *What this report does not know* section,
beside the lexicon-citation gap (`lexicon_id` is a declared string on the hit rows, not
`LexiconManifest.lexicon_content_hash`, which is what `FP-11` requires a family-level
number to cite), the absence of `selection_rule` and `selection_feature_names` from
`interpret.Interpretation`, and the fact that **no artifact in this package persists a
`guards.GuardResult`** — so the report names which guards `interpret.interpret_query`
invokes as source facts about the code path and never states that a guard passed on this
artifact. A guard's absence from `guards.GUARDS_AWAITING_INPUT` is *not* evidence that it
has a call site, and is reported as not known.

See `docs/ROADMAP.md` (M4b) for what remains: the fields above have to be carried by the
stages that produce them before this report can stop saying it does not know them.
