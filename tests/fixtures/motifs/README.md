# Synthetic motif-array fixtures for `tests/test_align.py`

These are small, deterministic, **synthetic** numpy arrays -- not real
genomic/TF-MoDISco data -- so none of the licensing/size concerns in
`tests/fixtures/README.md` apply here. They exist to give the align-stage
unit tests exact, hand-verified numeric behaviour instead of asserting
against whatever a random seed happens to produce.

Base order convention: PPM/CWM columns are `(A, C, G, T)` (see
`BASE_ORDER` in `src/motifmultiverse/align/__init__.py`); this is a numeric
axis convention, not a parsed identifier, so it does not conflict with
Rule 2 (`no_key_parsing`).

- `shared_ppm.npy`: a length-14 one-hot PPM, `(14, 4)`. Used both as the
  self-alignment substrate for the sign-flip test (paired with itself, plus
  opposite-signed CWMs) and as the "source" half of the short-overlap
  adversarial test.
- `short_overlap_target_ppm.npy`: a length-14 one-hot PPM whose content is
  independent of `shared_ppm` except its last 4 bases are forced equal to
  `shared_ppm`'s first 4 -- a "spike" that reaches cosine 1.0 only at the one
  offset where exactly those 4 positions overlap (`overlap_bp=4`, below the
  bilateral floor of 6bp / 0.5 frac). A brute-force scan over every
  offset/orientation (see `tests/test_align.py`) confirms every offset with
  `overlap_bp>=6` scores <=0.5 in both orientations, so excluding the 4bp
  spike is what makes the outcome different, not incidental.
- `sign_flip_cwm.npy`: an arbitrary nonzero `(14, 4)` contribution-score
  matrix. The "opposite" half of the sign-flip pair is `-sign_flip_cwm`,
  computed in the test rather than stored a second time.

This is not the `rest_sign_flip` regression case named in
`tests/fixtures/README.md` (that one is real REST-motif data, still pending
a download recipe) -- it is a synthetic stand-in that exercises the same
failure mode the README describes.
