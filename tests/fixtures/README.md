# Regression fixtures

Five boundary cases from the reference implementation. Each fixture is
**input + expected verdict + the reason for that verdict** -- the reason is part of
the fixture, because a fixture that only pins an output cannot tell a maintainer
whether a change is a fix or a regression.

**No fixture data is committed yet.** See "Licensing and size" below.

## Format convention

```
tests/fixtures/<case_id>/
  input/              minimal TF-MoDISco-shaped inputs (see docs/DATA_MODEL.md)
  expected.yaml       verdict + reason + the guard(s) that must fire
  PROVENANCE.md       where the case came from and what it demonstrates
```

`expected.yaml`:

```yaml
case_id: ctcf_duplicate
verdict: collapse
reason: >-
  The same motif discovered once per context. Collapsing removes duplicate
  discovery; it does not merge different factors.
guards_expected_to_pass: [variant_id_unique, no_cross_model_cwm_avg]
guards_expected_to_fail: []
```

## The five cases

| case_id | what it pins | expected verdict |
|---|---|---|
| `ctcf_duplicate` | one motif discovered independently in two contexts | `collapse` |
| `gata_tal1_flank` | a split whose geometric basis fails: the non-overlapping flank carries a small fraction of the core's information, and downstream reconstruction shows no change | `refuse_merge` by curation only, never by geometry |
| `nfyb_sp_fragment` | a fragment/partial pattern that must NOT be merged into the full motif | `keep_separate` |
| `znf76_merged_split` | a merge accepted by explicit override rather than by the gate | `collapse` with `decided_by` recorded |
| `rest_sign_flip` | a sign-flipped pair, invisible to an aligner that maximises signed similarity | registration must succeed on unsigned PPM |

`rest_sign_flip` is the fixture that would have caught the instrument failure
described in `src/motifmultiverse/align/README.md`.

## Licensing and size

Fixture inputs derive from published chromatin accessibility data and a reference
genome. Before any fixture data is committed:

1. confirm the redistribution terms of each source, and
2. keep every file well under the repository's large-file limit.

Where either check fails, the fixture ships as a **download recipe** (accession +
checksum + the exact command that produces the input) rather than as data. That
is the current state for all five.
