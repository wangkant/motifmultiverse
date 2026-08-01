"""align stage: unsigned-PPM registration and a full-pipeline-per-shuffle null.

Two constraints carry the science here, and each has a dedicated adversarial
test rather than a happy-path one:

1. Registration is chosen on UNSIGNED PPM content; signed CWM similarity is
   measured only AT that registration, never re-optimised (``align/README.md``:
   the sign-flip false negative). ``test_sign_flip_*`` below is the synthetic
   stand-in for the ``rest_sign_flip`` case named in
   ``tests/fixtures/README.md``.
2. The bilateral overlap requirement must actually exclude a short, spuriously
   perfect match -- not just be present as an unused parameter.
   ``test_short_overlap_*`` constructs a pair where the highest-scoring window
   of all is a 4bp spike, and asserts it is never selected.

Fixture arrays live in ``tests/fixtures/motifs/`` and are hand-verified there
(see the README in that directory) rather than trusted from a random seed.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from motifmultiverse import guards
from motifmultiverse.align import (
    DEFAULT_MIN_OVERLAP_BP,
    DEFAULT_MIN_OVERLAP_FRAC,
    AlignmentError,
    AlignmentEvidence,
    align_registry,
    calibrate_pair_null,
    register_pair,
    run,
)
from motifmultiverse.schema import REGISTRY_SCHEMA_VERSION

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

FIXTURES = Path(__file__).parent / "fixtures" / "motifs"


def _load(name: str):
    return np.load(FIXTURES / name)


@pytest.mark.parametrize(
    "changes",
    [
        {"orientation": "forward"},
        {"registered_on": "signed_cwm"},
        {"registration_rule_version": "forged"},
        {"overlap_bp": 0},
        {"overlap_frac_source": -0.1},
        {"overlap_frac_target": 1.1},
        {"ppm_similarity": float("inf")},
        {"signed_cwm_similarity": float("nan")},
        {"empirical_p_value": -0.01},
        {"empirical_p_value": 1.01},
        {"null_shuffles": 0},
        {"seed": 1.5},
    ],
)
def test_alignment_evidence_refuses_corrupted_fields(changes):
    evidence = AlignmentEvidence(
        source_node_id="a",
        target_node_id="b",
        orientation="+",
        offset=0,
        overlap_bp=10,
        overlap_frac_source=1.0,
        overlap_frac_target=1.0,
        ppm_similarity=0.9,
        signed_cwm_similarity=0.8,
        empirical_p_value=0.01,
        null_shuffles=100,
        seed=7,
    )

    with pytest.raises(AlignmentError):
        replace(evidence, **changes)


def _revcomp(mat):
    """Independent reimplementation of the BASE_ORDER=ACGT reverse-complement.

    `align._reverse_complement` is a single vectorised fancy-indexing
    expression (`mat[::-1, [3, 2, 1, 0]]`). This is deliberately NOT that
    expression with the names changed: it walks rows with an explicit
    Python loop, reverses position by index arithmetic (`n - 1 - i`) rather
    than a slice, and looks up each base's complement from an explicit
    ``{base: complement}`` map rather than a column-permutation list. A bug
    in the production function's row-reversal or column-permutation (e.g. an
    off-by-one, or A/G swapped instead of A/T) would not be reproduced here,
    so this genuinely cross-checks the result rather than restating it.
    """
    n_rows = mat.shape[0]
    complement_of = {0: 3, 1: 2, 2: 1, 3: 0}
    out = np.empty_like(mat)
    for i in range(n_rows):
        source_row = mat[n_rows - 1 - i]
        for base, comp_base in complement_of.items():
            out[i, comp_base] = source_row[base]
    return out


@pytest.fixture
def shared_ppm():
    return _load("shared_ppm.npy")


@pytest.fixture
def short_overlap_target():
    return _load("short_overlap_target_ppm.npy")


@pytest.fixture
def sign_flip_cwm():
    return _load("sign_flip_cwm.npy")


# --------------------------------------------------------------------- Step 1
def test_sign_flip_same_registration_and_signed_similarity_below_neg0_9(
    shared_ppm, sign_flip_cwm,
):
    """Same PPM, opposite CWM: registration must land on the trivial self
    alignment (offset 0, forward), and the signed CWM similarity measured
    there -- not re-optimised -- must be near -1.

    This is exactly the failure `align/README.md` documents: an aligner that
    maximises *signed* cosine could never find this offset at all, because the
    signed cosine there is close to -1, so a sign survey run that way would
    report no flip -- a false negative manufactured by the instrument.
    """
    evidence = register_pair(
        shared_ppm, shared_ppm, source_cwm=sign_flip_cwm, target_cwm=-sign_flip_cwm,
    )
    assert evidence.orientation == "+"
    assert evidence.offset == 0
    assert evidence.overlap_bp == shared_ppm.shape[0]
    assert evidence.ppm_similarity == pytest.approx(1.0, abs=1e-9)
    assert evidence.registered_on == "unsigned_ppm"
    assert evidence.signed_cwm_similarity is not None
    assert evidence.signed_cwm_similarity < -0.9


def test_sign_alignment_guard_passes_the_evidence_dict(shared_ppm, sign_flip_cwm):
    """The guard named in the README as "how to check it" must actually accept
    what register_pair emits (a falsification twin lives in tests/test_guards.py).
    """
    evidence = register_pair(
        shared_ppm, shared_ppm, source_cwm=sign_flip_cwm, target_cwm=-sign_flip_cwm,
    )
    result = guards.sign_alignment([evidence.to_dict()])
    assert result.passed, result.detail


# --------------------------------------------------------------------- Step 2
def test_short_overlap_spike_is_excluded_by_bilateral_requirement(
    shared_ppm, short_overlap_target,
):
    """`short_overlap_target_ppm.npy` is built so the single highest-scoring
    window in the ENTIRE search space (any offset, either orientation) is a
    4bp spike with cosine 1.0 -- and every window with overlap_bp>=6 scores
    <=0.5 (verified by brute-force scan when the fixture was built; see
    tests/fixtures/motifs/README.md). If register_pair ever selected on raw
    score without enforcing the bilateral floor, it would return that 4bp
    match. It must not.
    """
    evidence = register_pair(shared_ppm, short_overlap_target)
    assert evidence.overlap_bp >= DEFAULT_MIN_OVERLAP_BP
    assert evidence.overlap_frac_source >= DEFAULT_MIN_OVERLAP_FRAC
    assert evidence.overlap_frac_target >= DEFAULT_MIN_OVERLAP_FRAC
    # The excluded spike scores 1.0; every valid window scores <=0.5, so a
    # correct implementation is bounded well away from the spike's score.
    assert evidence.ppm_similarity <= 0.6
    assert not (evidence.overlap_bp == 4 and evidence.ppm_similarity == pytest.approx(1.0))


def test_register_pair_raises_when_no_offset_meets_the_bilateral_floor():
    """Two motifs short enough that no offset can reach the overlap floor at
    all: there is no registration to report, so this must refuse rather than
    silently return the best of a set of invalid candidates.
    """
    tiny_source = np.eye(4)[[0, 1, 2]]           # length 3
    tiny_target = np.eye(4)[[3, 2, 1]]            # length 3
    with pytest.raises(AlignmentError, match="bilateral overlap"):
        register_pair(tiny_source, tiny_target, min_overlap_bp=6, min_overlap_frac=0.5)


# --------------------------------------------------------------------- Step 3
def test_null_pipeline_reruns_full_registration_and_does_not_reuse_the_offset(
    shared_ppm, short_overlap_target, monkeypatch,
):
    """Every null shuffle must call register_pair fresh -- offset AND
    orientation re-optimised on the shuffled data -- never just recompute a
    score at the observed offset.
    """
    import motifmultiverse.align as align_mod

    calls: list[tuple[bytes, bytes]] = []
    real_register_pair = align_mod.register_pair

    def spy(source_ppm, target_ppm, *args, **kwargs):
        calls.append((np.asarray(source_ppm).tobytes(), np.asarray(target_ppm).tobytes()))
        return real_register_pair(source_ppm, target_ppm, *args, **kwargs)

    monkeypatch.setattr(align_mod, "register_pair", spy)

    null_shuffles = 8
    p_value, null_scores = align_mod.calibrate_pair_null(
        shared_ppm, short_overlap_target, null_shuffles=null_shuffles, seed=3,
    )

    # One observed call plus one call per shuffle -- no shortcut that skips
    # re-registering some shuffles.
    assert len(calls) == null_shuffles + 1
    assert len(null_scores) == null_shuffles
    assert 0.0 < p_value <= 1.0

    observed_target_bytes = calls[0][1]
    shuffle_target_bytes = [c[1] for c in calls[1:]]
    # The shuffles must actually differ from the observed target (a shuffled
    # permutation of 14 non-constant rows essentially never reproduces the
    # original by chance) -- proof the offset-holder was not just re-scored.
    assert any(b != observed_target_bytes for b in shuffle_target_bytes)
    # And the offset is not reused: at least one shuffle call's target bytes
    # differ from every other shuffle call's, i.e. this is not one cached
    # permutation replayed null_shuffles times.
    assert len(set(shuffle_target_bytes)) > 1


def test_null_summary_and_p_value_bounds(shared_ppm, short_overlap_target):
    p_value, null_scores = calibrate_pair_null(
        shared_ppm, short_overlap_target, null_shuffles=25, seed=11,
    )
    assert len(null_scores) == 25
    assert 0.0 < p_value <= 1.0
    assert all(isinstance(s, float) for s in null_scores)


def test_calibrate_pair_null_is_reproducible_with_the_same_seed(shared_ppm, short_overlap_target):
    p1, scores1 = calibrate_pair_null(shared_ppm, short_overlap_target, null_shuffles=10, seed=5)
    p2, scores2 = calibrate_pair_null(shared_ppm, short_overlap_target, null_shuffles=10, seed=5)
    assert p1 == p2
    assert scores1 == pytest.approx(scores2)


def test_calibrate_pair_null_differs_across_seeds(shared_ppm, short_overlap_target):
    _, scores_a = calibrate_pair_null(shared_ppm, short_overlap_target, null_shuffles=10, seed=1)
    _, scores_b = calibrate_pair_null(shared_ppm, short_overlap_target, null_shuffles=10, seed=2)
    assert scores_a != scores_b


def test_calibrate_pair_null_rejects_zero_shuffles(shared_ppm, short_overlap_target):
    """`null_shuffles=0` must fail fast with a clear message, not fall through
    to the loop's "every null shuffle failed" refusal (a review finding: that
    message is about a genuinely unreachable case, not this one).
    """
    with pytest.raises(AlignmentError, match="null_shuffles"):
        calibrate_pair_null(shared_ppm, short_overlap_target, null_shuffles=0, seed=1)


# --------------------------------------------------------------------- Step 4
def test_register_pair_detects_reverse_complement_orientation(shared_ppm):
    rc_target = _revcomp(shared_ppm)
    evidence = register_pair(shared_ppm, rc_target)
    assert evidence.orientation == "-"
    assert evidence.offset == 0
    assert evidence.overlap_bp == shared_ppm.shape[0]
    assert evidence.ppm_similarity == pytest.approx(1.0, abs=1e-9)


def test_register_pair_rejects_a_non_length4_ppm(shared_ppm):
    bad = shared_ppm[:, :3]
    with pytest.raises(AlignmentError, match="4"):
        register_pair(shared_ppm, bad)


def test_register_pair_rejects_cwm_ppm_length_mismatch(shared_ppm, sign_flip_cwm):
    short_cwm = sign_flip_cwm[:-1]
    with pytest.raises(AlignmentError, match="length"):
        register_pair(shared_ppm, shared_ppm, source_cwm=short_cwm, target_cwm=sign_flip_cwm)


def test_registered_on_defaults_to_unsigned_ppm(shared_ppm, short_overlap_target):
    evidence = register_pair(shared_ppm, short_overlap_target)
    assert evidence.registered_on == "unsigned_ppm"
    assert evidence.registration_rule_version


# --------------------------------------------------------------------- Step 5
def _registry_arrays_h5(tmp_path, motifs: dict[str, dict[str, np.ndarray]],
                        cores: dict[str, list[int] | None] | None = None):
    """A minimal on-disk registry (registry.json + arrays.h5) with just enough
    structure for align_registry to read: node_id, trimmed_core, and per-node
    ppm/cwm arrays.

    `trimmed_core` defaults to the whole matrix, which is the fixture arrays'
    honest core: they carry no background padding to trim. `cores` overrides it
    per node, including with None for a node that declares none at all.
    """
    import json

    h5py = pytest.importorskip("h5py")
    cores = cores or {}
    reg_dir = tmp_path / "registry"
    reg_dir.mkdir()
    nodes = []
    with h5py.File(reg_dir / "arrays.h5", "w") as h5:
        for node_id, arrays in motifs.items():
            grp = h5.create_group(node_id)
            for name, arr in arrays.items():
                grp.create_dataset(name, data=arr)
            full = next(iter(arrays.values())).shape[0]
            nodes.append({
                "node_id": node_id, "model": "m", "readout": "r", "context": "c",
                "metacluster": "pos", "denovo_pattern_id": node_id,
                "variant_id": f"U_FAM_{len(nodes):02d}", "family_id": "FAM",
                "trimmed_core": cores.get(node_id, [0, full]),
            })
    payload = {
        "registry_metadata": {
            "project": "p", "peak_universe_id": "u", "analyses": [],
            "n_models": 1, "cross_model_claims_restricted": True,
            "metacluster_states": {}, "trim_threshold": 0.3,
            "schema_version": REGISTRY_SCHEMA_VERSION,
        },
        "nodes": nodes,
    }
    (reg_dir / "registry.json").write_text(json.dumps(payload))
    return reg_dir


def test_align_registry_writes_edges_and_null_summary_with_required_provenance(
    tmp_path, shared_ppm, short_overlap_target, sign_flip_cwm,
):
    registry = _registry_arrays_h5(tmp_path, {
        "a": {"ppm": shared_ppm, "cwm": sign_flip_cwm},
        "b": {"ppm": short_overlap_target, "cwm": -sign_flip_cwm},
    })
    out = tmp_path / "evidence"
    summary, edges = align_registry(registry, out, null_shuffles=5, seed=9)

    assert len(edges) == 1
    edges_path = out / "alignment_edges.parquet"
    null_path = out / "alignment_null_summary.tsv"
    assert edges_path.exists() and null_path.exists()

    df = pd.read_parquet(edges_path)
    assert len(df) == 1
    row = df.iloc[0]
    # Required provenance: null_shuffles and seed on every emitted edge, plus
    # every overlap field and the registration rule version.
    for field in ("null_shuffles", "seed", "registration_rule_version",
                  "overlap_bp", "overlap_frac_source", "overlap_frac_target",
                  "empirical_p_value", "registered_on"):
        assert field in df.columns, field
    assert row["null_shuffles"] == 5
    assert row["seed"] == 9
    assert row["registered_on"] == "unsigned_ppm"

    null_df = pd.read_csv(null_path, sep="\t")
    assert len(null_df) == 1
    assert null_df.iloc[0]["null_shuffles"] == 5
    assert null_df.iloc[0]["seed"] == 9
    assert (out / "provenance.json").exists()


def test_align_registry_edges_pass_the_sign_alignment_guard(
    tmp_path, shared_ppm, short_overlap_target,
):
    registry = _registry_arrays_h5(tmp_path, {
        "a": {"ppm": shared_ppm, "cwm": shared_ppm},
        "b": {"ppm": short_overlap_target, "cwm": short_overlap_target},
    })
    _, edges = align_registry(registry, tmp_path / "evidence", null_shuffles=3, seed=1)
    result = guards.sign_alignment([e.to_dict() for e in edges])
    assert result.passed, result.detail


def test_align_registry_excludes_nodes_without_a_ppm(tmp_path, shared_ppm, sign_flip_cwm):
    registry = _registry_arrays_h5(tmp_path, {
        "a": {"ppm": shared_ppm, "cwm": sign_flip_cwm},
        "b": {"cwm": sign_flip_cwm},   # no "sequence"/ppm dataset at all
    })
    summary, edges = align_registry(registry, tmp_path / "evidence", null_shuffles=2, seed=1)
    assert edges == []
    assert summary.n_pairs_excluded == 1

    # A zero-edge run must still write a typed (not all-object) parquet: a
    # downstream reader concatenating edge tables across runs should never see
    # the schema depend on whether this particular run happened to find any
    # edges.
    df = pd.read_parquet(tmp_path / "evidence" / "alignment_edges.parquet")
    assert len(df) == 0
    assert str(df["offset"].dtype) == "int64"
    assert str(df["ppm_similarity"].dtype) == "float64"
    assert str(df["null_shuffles"].dtype) == "int64"


def test_align_registry_rejects_zero_null_shuffles(tmp_path, shared_ppm, short_overlap_target):
    """`align_registry` has its OWN fail-fast guard for `null_shuffles < 1`,
    checked before the output directory is even created -- before
    `out.mkdir`, before provenance is written, before the registry is
    opened. `calibrate_pair_null` (called once per pair, deep in the loop
    below all of that) has its own separately-tested guard for the same
    condition, with a message that also contains "null_shuffles". So
    asserting only on message text cannot tell the two guards apart:
    deleting `align_registry`'s own guard and falling through to that
    backstop would still raise a message-matching AlignmentError, just from
    inside the loop, after `out_dir` and `provenance.json` already exist.

    Assert on the one thing only the early guard can produce: `out_dir`
    must never come into existence at all.
    """
    registry = _registry_arrays_h5(tmp_path, {
        "a": {"ppm": shared_ppm}, "b": {"ppm": short_overlap_target},
    })
    out = tmp_path / "evidence"
    with pytest.raises(AlignmentError, match="null.shuffles"):
        align_registry(registry, out, null_shuffles=0, seed=1)
    assert not out.exists(), (
        "out_dir was created before align_registry's own null_shuffles guard "
        "fired -- this can only happen if that guard was removed and "
        "calibrate_pair_null's backstop guard (which runs after out.mkdir, "
        "the provenance write, and the registry load) caught it instead"
    )


def test_run_is_the_orchestrator_entry_point():
    assert run is align_registry


# ----------------------------------------------------------------------- CLI
def test_cli_align_runs_end_to_end(tmp_path, capsys, shared_ppm, short_overlap_target, sign_flip_cwm):
    from motifmultiverse.cli import main

    registry = _registry_arrays_h5(tmp_path, {
        "a": {"ppm": shared_ppm, "cwm": sign_flip_cwm},
        "b": {"ppm": short_overlap_target, "cwm": -sign_flip_cwm},
    })
    out = tmp_path / "evidence"
    rc = main(["align", str(registry), "--null-shuffles", "4", "--seed", "2", "--out", str(out)])
    assert rc == 0
    df = pd.read_parquet(out / "alignment_edges.parquet")
    assert len(df) == 1
    assert df.iloc[0]["null_shuffles"] == 4
    assert df.iloc[0]["seed"] == 2
    printed = capsys.readouterr().out
    assert "alignment_edges.parquet" in printed or "written" in printed


# --- the cosine fast path must be the same arithmetic --------------------------
def test_align_cosine_matches_the_linalg_norm_formulation():
    """`_cosine` replaced np.linalg.norm with sqrt(v @ v) for speed.

    A profile of a 29-pattern registry put 110,313 calls in `_cosine` and 220,626
    inside np.linalg.norm, which on vectors of a few dozen floats spends its time
    dispatching. The two formulations are the same quantity; this pins that, so a
    later "optimisation" cannot quietly change the number instead of the cost.
    """
    np = pytest.importorskip("numpy")

    from motifmultiverse.align import _cosine

    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 24))
        a = rng.normal(size=(n, 4))
        b = rng.normal(size=(n, 4))
        fa, fb = a.ravel(), b.ravel()
        reference = float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb)))
        reference = max(-1.0, min(1.0, reference))
        assert _cosine(a, b) == pytest.approx(reference, abs=1e-12)


def test_align_cosine_still_returns_zero_for_a_zero_vector():
    np = pytest.importorskip("numpy")

    from motifmultiverse.align import _cosine

    zero = np.zeros((4, 4))
    other = np.ones((4, 4))
    assert _cosine(zero, other) == 0.0
    assert _cosine(other, zero) == 0.0


@pytest.mark.parametrize("scale", [1e-320, 1e-300, 1e-200, 1e-160, 1.0,
                                   1e160, 1e200, 1e300, 1e308])
def test_align_cosine_is_exact_at_every_representable_magnitude(scale):
    """Including the band where the clamp used to turn NaN into +1.0.

    np.linalg.norm rescales internally; np.dot does not. Above ~1e153 the
    numerator overflows, the quotient is NaN, and `max(-1.0, min(1.0, nan))`
    yields **+1.0**, because a NaN comparison is False and the clamp keeps its
    first argument. Measured before the fix: two exactly anti-correlated windows
    at 1e155 reported +1 -- a sign-flipped pair read as a perfect positive match,
    in the module whose whole docstring is about not doing that. Below ~1e165 the
    numerator underflows instead and the answer was 0.0.
    """
    np = pytest.importorskip("numpy")

    from motifmultiverse.align import _cosine

    x = np.full((6, 4), scale)
    assert _cosine(x, x) == pytest.approx(1.0, abs=1e-9)
    assert _cosine(x, -x) == pytest.approx(-1.0, abs=1e-9)


def test_align_cosine_keeps_zero_for_a_genuinely_zero_window():
    np = pytest.importorskip("numpy")

    from motifmultiverse.align import _cosine

    assert _cosine(np.zeros((4, 4)), np.ones((4, 4))) == 0.0
    assert _cosine(np.ones((4, 4)), np.zeros((4, 4))) == 0.0


def test_align_cosine_keeps_zero_for_genuinely_orthogonal_windows():
    np = pytest.importorskip("numpy")

    from motifmultiverse.align import _cosine

    assert _cosine(np.array([[1.0, 0, 0, 0]]), np.array([[0, 1.0, 0, 0]])) == 0.0


# ------------------------------------------- registration is on the TRIMMED CORE
def _informative_core(seed: int, length: int):
    """A core with one dominant base per position -- real motif content."""
    rng = np.random.default_rng(seed)
    matrix = np.full((length, 4), 0.02)
    for i in range(length):
        matrix[i, rng.integers(0, 4)] = 0.94
    return matrix / matrix.sum(axis=1, keepdims=True)


def _padded_pattern(core, left_pad: int, right_pad: int, seed: int):
    """A TF-MoDISco-shaped pattern: an informative core inside a fixed-width
    window whose flanks are the near-uniform background EVERY pattern in a run
    carries. This is what the discovery HDF5 actually contains, and it is why
    registering on the window instead of the core measures shared padding.
    """
    rng = np.random.default_rng(seed)
    window = rng.normal(0.25, 0.008, size=(left_pad + core.shape[0] + right_pad, 4))
    window = window.clip(0.05)
    window[left_pad:left_pad + core.shape[0]] = core
    return window / window.sum(axis=1, keepdims=True)


def test_align_registry_registers_on_the_core_not_the_padded_window(tmp_path):
    """The same 8bp motif, padded into two 40bp windows, must register as an 8bp
    identity -- not as a 35bp near-identity of the two windows' shared background.

    Measured before the fix: `overlap_bp=35`, `ppm_similarity=0.9996`. The whole
    registration was carried by flanks that say nothing about either motif, and
    the bilateral overlap floor -- the module's own protection against a window
    scoring high on content it does not have -- passed on the strength of that
    same padding. `ingest` had already recorded where each core is; align simply
    never asked.
    """
    core = _informative_core(7, 8)
    registry = _registry_arrays_h5(
        tmp_path,
        {"a": {"ppm": _padded_pattern(core, 16, 16, 101)},
         "b": {"ppm": _padded_pattern(core, 21, 11, 202)}},
        cores={"a": [16, 24], "b": [21, 29]},
    )
    _, edges = align_registry(registry, tmp_path / "evidence", null_shuffles=3, seed=1)

    assert len(edges) == 1
    edge = edges[0]
    assert edge.overlap_bp == 8, (
        "registration covered more than the 8bp core, so it was scored on the "
        "padded window rather than on the span ingest recorded as trimmed_core"
    )
    assert edge.overlap_frac_source == pytest.approx(1.0)
    assert edge.overlap_frac_target == pytest.approx(1.0)
    assert edge.ppm_similarity == pytest.approx(1.0, abs=1e-12)


def test_align_registry_slices_the_cwm_to_the_same_core_as_the_ppm(tmp_path):
    """Signed CWM similarity is measured AT the PPM's registration, so the two
    matrices have to describe the same positions. A CWM read at full width
    against a core-width PPM registration is not a sign statistic about the
    motif -- it is a sign statistic about the padding, which is the failure this
    module exists to prevent, one field over.
    """
    core = _informative_core(7, 8)
    cwm = (core - 0.25) * 3.0
    flank_noise = np.random.default_rng(5).normal(0, 0.4, size=(16, 4))
    source_cwm = np.vstack([flank_noise, cwm, flank_noise])
    # The flanks are NOT negated: only the core is a true sign flip, so a signed
    # similarity of exactly -1 can only come from a core-width comparison.
    target_cwm = np.vstack([flank_noise, -cwm, flank_noise])
    registry = _registry_arrays_h5(
        tmp_path,
        {"a": {"ppm": _padded_pattern(core, 16, 16, 101), "cwm": source_cwm},
         "b": {"ppm": _padded_pattern(core, 16, 16, 101), "cwm": target_cwm}},
        cores={"a": [16, 24], "b": [16, 24]},
    )
    _, edges = align_registry(registry, tmp_path / "evidence", null_shuffles=3, seed=1)

    assert len(edges) == 1
    assert edges[0].signed_cwm_similarity == pytest.approx(-1.0, abs=1e-12)


def test_align_registry_excludes_a_node_that_declares_no_trimmed_core(tmp_path):
    """A registry that declares no core for a node is refused for that node's
    pairs, not quietly registered on its padded window.

    Falling back would put a padding-driven score into the edge table under the
    same `registration_rule_version` as a trimmed one, where no downstream
    reader could tell the two apart -- the same "recorded as if it were the real
    measurement" failure the module docstring is about.
    """
    core = _informative_core(7, 8)
    registry = _registry_arrays_h5(
        tmp_path,
        {"a": {"ppm": _padded_pattern(core, 16, 16, 101)},
         "b": {"ppm": _padded_pattern(core, 16, 16, 101)}},
        cores={"a": [16, 24], "b": None},
    )
    summary, edges = align_registry(registry, tmp_path / "evidence", null_shuffles=3, seed=1)

    assert edges == []
    assert summary.n_pairs_considered == 1
    assert summary.n_pairs_excluded == 1


def test_align_registry_separates_components_by_core_width_not_window_width(tmp_path):
    """Four patterns in one 40bp window: two share an 8bp core, two share a 20bp
    core, and the two cores are unrelated. That is one component of two and
    another of two -- not one component of four.

    Registered on the window, all four are 40bp long, so every pair clears the
    bilateral overlap floor and single linkage hands adjudication ONE proposed
    cluster containing every motif in the project. Measured on 29 real
    ChromBPNet patterns (50bp windows, cores 4-30bp): 406 of 406 pairs
    registered and every one of them landed in a single component. Registered on
    the core, 8bp covers 0.4 of 20bp, the floor excludes the cross pairs, and
    the two real groups survive as two.
    """
    from motifmultiverse.adjudicate import adjudicate_all, packaged_criteria_path
    from motifmultiverse.schema.criteria import load_criteria

    short, long = _informative_core(7, 8), _informative_core(9, 20)
    registry = _registry_arrays_h5(
        tmp_path,
        {"s0": {"ppm": _padded_pattern(short, 16, 16, 101)},
         "s1": {"ppm": _padded_pattern(short, 16, 16, 202)},
         "l0": {"ppm": _padded_pattern(long, 10, 10, 303)},
         "l1": {"ppm": _padded_pattern(long, 10, 10, 404)}},
        cores={"s0": [16, 24], "s1": [16, 24], "l0": [10, 30], "l1": [10, 30]},
    )
    summary, edges = align_registry(registry, tmp_path / "evidence", null_shuffles=3, seed=1)

    assert summary.n_pairs_considered == 6
    assert {tuple(sorted((e.source_node_id, e.target_node_id))) for e in edges} == {
        ("l0", "l1"), ("s0", "s1"),
    }

    decisions = adjudicate_all(
        edges, [], [], load_criteria(packaged_criteria_path()), "test",
    )
    assert sorted(decision.node_ids for decision in decisions) == [
        ("l0", "l1"), ("s0", "s1"),
    ]


# ----------------------------------------------- recorded limitations, pinned
def test_align_emits_an_edge_for_every_registrable_pair_however_unremarkable(tmp_path):
    """RECORDED LIMITATION, not an accident: registrability -- not similarity and
    not the null -- decides which pairs reach adjudication.

    Two unrelated cores of the same width clear the bilateral overlap floor, so
    align emits an edge whose own null says the registration is unremarkable,
    and `adjudicate_all` proposes them as a component on the strength of that
    edge existing. That is unrestricted single linkage, and `docs/CONSTRAINTS.md`
    already carries it as the open half of FP-05: "single linkage is admissible
    only with a declared distance ceiling", enforcement PARTIAL, "the linkage
    clause has no check".

    Closing it here would mean choosing the ceiling -- a similarity or p-value
    cut-off -- and FP-13 reserves exactly that parameter to the design, which is
    why `adjudicate/criteria.v1.yaml` leaves TRUE_DUPLICATE and FRAGMENT_MATCH
    `CRITERION_NOT_YET_DEFINED` rather than guess one. So the edge is recorded
    WITH its p-value, connectivity only ever proposes, and the criterion is the
    only thing that may gate a collapse. This test exists so that the gap is a
    decision on the record rather than something a reader has to infer from an
    edge table where nothing was ever filtered out.
    """
    from motifmultiverse.adjudicate import adjudicate_all, packaged_criteria_path
    from motifmultiverse.schema.criteria import load_criteria

    registry = _registry_arrays_h5(
        tmp_path,
        {"a": {"ppm": _padded_pattern(_informative_core(7, 12), 14, 14, 101)},
         "b": {"ppm": _padded_pattern(_informative_core(9, 12), 14, 14, 202)}},
        cores={"a": [14, 26], "b": [14, 26]},
    )
    _, edges = align_registry(registry, tmp_path / "evidence", null_shuffles=99, seed=1)

    assert len(edges) == 1, "an unremarkable registration is still recorded"
    assert edges[0].empirical_p_value > 0.05, (
        "fixture no longer exercises the limitation: these two cores are "
        "supposed to be unrelated enough that the null does not reject"
    )
    decisions = adjudicate_all(
        edges, [], [], load_criteria(packaged_criteria_path()), "test",
    )
    assert [d.node_ids for d in decisions] == [("a", "b")], (
        "component proposal reads edge presence only; an edge the null did not "
        "single out still proposes a cluster"
    )


def test_align_null_re_registers_from_scratch_for_every_shuffle(tmp_path):
    """RECORDED LIMITATION with a guarantee on the other side of it: the null
    costs one full registration per shuffle per pair, and that cost is the
    guarantee.

    Counting registrations pins both halves. `align_registry` registers each
    considered pair once; `calibrate_pair_null` then registers the observed pair
    again plus once per shuffle, on freshly permuted data. Nothing is cached
    across pairs and nothing is rescored at a remembered offset -- a null that
    did either would answer "how similar are these two at one fixed alignment"
    instead of "how surprising is it to find SOME alignment this good", and
    would inflate every p-value in the table.

    The price is quadratic in the registry: measured on 29 real ChromBPNet
    patterns, one registration of a 4-30bp core costs about 128 microseconds, so
    a twelve-run 240-node registry at the default 1000 shuffles extrapolates to
    roughly half a CPU-hour of registrations. That total is recorded rather than
    traded away, because every cheaper null on offer is a weaker one.

    Counted here on the default single worker, because this counter is a
    monkeypatch in THIS process and a worker process would never see it. That is
    a limit of the counting, not a gap in the pin: what shows the parallel path
    does the same registrations is
    `test_align_registry_writes_byte_identical_tables_at_every_worker_count` --
    a null of 25 re-registrations per pair cannot come out bit-identical from a
    process that skipped or cached any of them.
    """
    import motifmultiverse.align as align_module

    core = _informative_core(7, 12)
    registry = _registry_arrays_h5(
        tmp_path,
        {"a": {"ppm": _padded_pattern(core, 14, 14, 101)},
         "b": {"ppm": _padded_pattern(core, 14, 14, 202)},
         "c": {"ppm": _padded_pattern(core, 14, 14, 303)}},
        cores={"a": [14, 26], "b": [14, 26], "c": [14, 26]},
    )
    calls = []
    real_register_pair = align_module.register_pair

    def counting_register_pair(*args, **kwargs):
        calls.append(1)
        return real_register_pair(*args, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(align_module, "register_pair", counting_register_pair)
    try:
        summary, edges = align_registry(
            registry, tmp_path / "evidence", null_shuffles=7, seed=1,
        )
    finally:
        monkey.undo()

    assert summary.n_pairs_considered == 3 and len(edges) == 3
    # 3 observed registrations in align_registry, then per registered pair one
    # more observed registration inside calibrate_pair_null plus one per shuffle.
    assert len(calls) == 3 + 3 * (1 + 7)


# --------------------------------------------- parallelism must not be arithmetic
def _parallel_registry(tmp_path, n_nodes: int = 8):
    """A registry with enough registrable pairs to spread over several workers.

    Every core is the same width, so no pair is excluded by the overlap floor and
    every worker has real work; the cores themselves are unrelated, so the
    p-values are spread rather than all pinned at the floor.
    """
    windows = {f"n{i}": _padded_pattern(_informative_core(i + 1, 12), 14, 14, 100 + i)
               for i in range(n_nodes)}
    return _registry_arrays_h5(
        tmp_path,
        {node_id: {"ppm": window, "cwm": (window - 0.25) * 3.0}
         for node_id, window in windows.items()},
        cores={node_id: [14, 26] for node_id in windows},
    )


def test_align_null_is_a_pure_function_of_the_seed_and_the_pair(tmp_path):
    """The property that makes parallelising this stage admissible at all.

    A pair's null must depend on the run seed and on that pair's own matrices,
    and on nothing else -- not on how many pairs were calibrated before it, and
    not on which worker happened to take it. Two things are checked, because
    either alone is weak: the same pair calibrated in two different call orders
    must give the identical null, AND the null scores must be reproducible from
    `default_rng(seed)` by an independent loop that permutes and re-registers
    here in the test. The second is what actually pins the mechanism: if the
    generator were ever hoisted to module level or shared between pairs, the
    draws would depend on how many pairs preceded this one and this
    recomputation -- which starts its stream from the seed alone -- would stop
    matching. That failure would be invisible in a serial run and would make
    `--workers` change the p-value table.
    """
    from motifmultiverse.align import calibrate_pair_null, register_pair

    a = _informative_core(11, 12)
    b = _informative_core(12, 12)
    c = _informative_core(13, 12)
    seed, shuffles = 5, 12

    forward = [calibrate_pair_null(x, y, null_shuffles=shuffles, seed=seed)
               for x, y in ((a, b), (a, c), (b, c))]
    backward = [calibrate_pair_null(x, y, null_shuffles=shuffles, seed=seed)
                for x, y in ((b, c), (a, c), (a, b))][::-1]
    assert forward == backward, (
        "a pair's null changed with the order the pairs were calibrated in; "
        "some state is being carried between pairs"
    )

    for (source, target), (_p_value, null_scores) in zip(
        ((a, b), (a, c), (b, c)), forward, strict=True,
    ):
        rng = np.random.default_rng(seed)
        independent = [
            register_pair(source, target[rng.permutation(target.shape[0])]).ppm_similarity
            for _ in range(shuffles)
        ]
        assert independent == null_scores, (
            "this pair's null is not reproducible from default_rng(seed) alone, so "
            "it depends on something other than the seed and the pair itself"
        )


def test_align_registry_writes_byte_identical_tables_at_every_worker_count(tmp_path):
    """The point of `workers` is that it changes nothing but wall-clock.

    Not "the same p-values to some tolerance" -- the same bytes. Row order is
    part of that: outcomes are reassembled by each pair's position in the pair
    order, so a table ordered by whichever worker finished first would fail here
    with every number in it correct, which is exactly the kind of difference that
    is impossible to explain to a reader six months later.

    A real ProcessPoolExecutor runs, so this also exercises the pickling path:
    the matrices reach the workers through the pool initialiser, and anything
    that failed to survive that round trip would change the numbers rather than
    raise.
    """
    serial_out = tmp_path / "serial"
    parallel_out = tmp_path / "parallel"
    registry = _parallel_registry(tmp_path)

    serial_summary, serial_edges = align_registry(
        registry, serial_out, null_shuffles=25, seed=3, workers=1)
    parallel_summary, parallel_edges = align_registry(
        registry, parallel_out, null_shuffles=25, seed=3, workers=3)

    assert serial_summary.n_edges == parallel_summary.n_edges == len(serial_edges)
    assert serial_summary.n_edges > 1, "fixture registers too few pairs to distribute"
    assert serial_edges == parallel_edges
    for name in ("alignment_edges.parquet", "alignment_null_summary.tsv"):
        assert (serial_out / name).read_bytes() == (parallel_out / name).read_bytes(), (
            f"{name} differs between worker counts"
        )
    assert serial_summary.workers == 1 and parallel_summary.workers == 3


def test_align_registry_really_runs_a_pool_at_the_requested_worker_count(tmp_path):
    """`workers` must not be a parameter that is accepted and ignored.

    The byte-identity test above passes trivially if `workers=3` quietly runs
    everything in this process -- identical output, no speedup, and a
    documented-but-absent feature. So the executor is recorded here: constructed
    once, with max_workers=3, and asked to run the jobs.
    """
    import motifmultiverse.align as align_module

    constructed = []
    real_executor = align_module.ProcessPoolExecutor

    def recording_executor(*args, **kwargs):
        constructed.append(kwargs.get("max_workers"))
        return real_executor(*args, **kwargs)

    registry = _parallel_registry(tmp_path, n_nodes=5)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(align_module, "ProcessPoolExecutor", recording_executor)
    try:
        summary, edges = align_registry(
            registry, tmp_path / "evidence", null_shuffles=5, seed=3, workers=3)
    finally:
        monkey.undo()

    assert constructed == [3], (
        "no process pool was created for workers=3, so the pair loop ran in this "
        "process and --workers is decorative"
    )
    assert len(edges) == summary.n_edges > 1


@pytest.mark.parametrize("workers", [0, -1, 1.5, "4", None, True])
def test_align_registry_refuses_a_worker_count_that_is_not_a_positive_integer(
    tmp_path, workers,
):
    """Refused before the output directory exists, like `null_shuffles`.

    `True` is in the list deliberately: `bool` is a subclass of `int`, so
    `workers=True` would otherwise be silently read as "one worker" -- i.e. as
    "no, serial" -- by a caller who wrote it meaning "yes, parallelise". A
    parameter that quietly means the opposite of what it was written to mean is
    worse than one that refuses.
    """
    registry = _parallel_registry(tmp_path, n_nodes=3)
    out = tmp_path / "evidence"
    with pytest.raises(AlignmentError, match="workers"):
        align_registry(registry, out, null_shuffles=3, seed=1, workers=workers)
    assert not out.exists(), (
        "the worker-count guard fired after the run had already started writing"
    )


def test_align_registry_is_byte_identical_under_the_spawn_start_method(tmp_path):
    """The same equality under the start method that pickles everything.

    `fork` hands a worker the parent's memory, so the matrices reach it without
    ever being serialised; `spawn` and `forkserver` pickle the pool's initargs
    and re-import this module in a fresh interpreter. Linux defaults to `fork`
    today and CPython is moving off it -- 3.14 makes `forkserver` the default --
    and macOS has been `spawn` for years, so the pickling path is not exotic: it
    is what most callers will get next. A float array that survived fork but
    round-tripped imprecisely, or a module that behaved differently on re-import,
    would move the p-values on those platforms only.

    Run in a subprocess because the start method is process-global state and
    forcing it here would leak into every other test in the session.
    """
    import subprocess
    import sys
    import textwrap

    registry = _parallel_registry(tmp_path, n_nodes=5)
    serial_out = tmp_path / "serial"
    spawned_out = tmp_path / "spawned"
    align_registry(registry, serial_out, null_shuffles=5, seed=3, workers=1)

    script = textwrap.dedent(
        """
        import multiprocessing, sys

        def main():
            multiprocessing.set_start_method("spawn", force=True)
            from motifmultiverse.align import align_registry
            align_registry(sys.argv[1], sys.argv[2], null_shuffles=5, seed=3, workers=2)

        if __name__ == "__main__":
            main()
        """
    )
    script_path = tmp_path / "spawn_align.py"
    script_path.write_text(script)
    completed = subprocess.run(
        [sys.executable, str(script_path), str(registry), str(spawned_out)],
        capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stderr

    for name in ("alignment_edges.parquet", "alignment_null_summary.tsv"):
        assert (serial_out / name).read_bytes() == (spawned_out / name).read_bytes(), (
            f"{name} differs between an in-process serial run and a spawn-method "
            "parallel run"
        )


def test_align_registry_reports_progress_and_prints_nothing_itself(tmp_path, capsys):
    """A multi-hour stage that says nothing is its own defect -- and a library
    that prints is a different one.

    `align_registry` reports through a callback and writes to no stream at all,
    so a caller parsing its stdout cannot be polluted by progress it did not ask
    for; the CLI is what decides that the lines go to stderr. The callback fires
    once per registrable pair, in order, with a running count and a fixed total,
    so a reader can tell a slow run from a hung one.
    """
    registry = _parallel_registry(tmp_path, n_nodes=4)
    seen: list[tuple[int, int]] = []

    summary, _edges = align_registry(
        registry, tmp_path / "evidence", null_shuffles=3, seed=1,
        progress=lambda done, total: seen.append((done, total)),
    )

    total = summary.n_pairs_considered - summary.n_pairs_excluded
    assert seen == [(i, total) for i in range(1, total + 1)]
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "", (
        "align_registry wrote to a stream of its own; stdout is a caller's to parse"
    )


def test_align_registry_reports_progress_from_workers_too(tmp_path):
    """The parallel path must report as it goes, not in one burst at the end.

    Progress exists for the run that takes hours, and that is the run someone
    will use `--workers` for. The counts must still arrive one per pair and in
    order, because the callback is a count of finished work, not of scheduled
    work.
    """
    registry = _parallel_registry(tmp_path, n_nodes=5)
    seen: list[tuple[int, int]] = []

    summary, _edges = align_registry(
        registry, tmp_path / "evidence", null_shuffles=3, seed=1, workers=2,
        progress=lambda done, total: seen.append((done, total)),
    )

    total = summary.n_pairs_considered - summary.n_pairs_excluded
    assert seen == [(i, total) for i in range(1, total + 1)]


def test_cli_align_workers_progress_goes_to_stderr_not_stdout(tmp_path, capsys):
    """The CLI's own stdout is parsed -- `written: <path>` lines and counts -- so
    progress goes to the other stream.

    Checked as an absence on stdout as well as a presence on stderr: a progress
    line that also appeared on stdout would satisfy "there is progress output"
    while breaking the caller it was supposed not to disturb.
    """
    from motifmultiverse.cli import main

    registry = _parallel_registry(tmp_path, n_nodes=4)
    out = tmp_path / "evidence"
    rc = main(["align", str(registry), "--null-shuffles", "3", "--seed", "1",
               "--workers", "2", "--out", str(out)])
    assert rc == 0

    captured = capsys.readouterr()
    assert "candidate pairs processed" in captured.err, "the stage ran silently"
    assert "candidate pairs processed" not in captured.out, (
        "progress reached stdout, where a caller parsing the result lines will read it"
    )
    assert "written:" in captured.out and "workers=2" in captured.out
    assert len(pd.read_parquet(out / "alignment_edges.parquet")) > 1


def test_align_records_what_its_guard_returned_beside_the_edge_table(
    tmp_path, shared_ppm, short_overlap_target, sign_flip_cwm,
):
    """`sign_alignment` is this stage's whole executable claim; it now leaves a trace.

    Registration on signed CWM cosine is structurally blind to a sign-flipped
    motif, so "no flips" can be an artefact of the instrument. The guard says the
    registration was unsigned; until this record existed, a directory holding an
    edge table could not say whether it had ever been asked.
    """
    import json

    from motifmultiverse import guard_log

    registry = _registry_arrays_h5(tmp_path, {
        "a": {"ppm": shared_ppm, "cwm": sign_flip_cwm},
        "b": {"ppm": short_overlap_target, "cwm": -sign_flip_cwm},
    })
    out = tmp_path / "evidence"
    align_registry(registry, out, null_shuffles=3, seed=4)

    recorded = json.loads((out / guard_log.GUARD_OUTCOMES_FILENAME).read_text())
    assert [row["guard_id"] for row in recorded] == ["sign_alignment"]
    assert recorded[0]["stage"] == "align" and recorded[0]["passed"] is True
    assert "unsigned" in recorded[0]["detail"]
    # Written after the provenance record, so the join to the run holds.
    assert recorded[0]["provenance_records"] == 1


def test_align_records_the_registry_bytes_it_read(
    tmp_path, shared_ppm, short_overlap_target, sign_flip_cwm,
):
    """`align` recorded no inputs at all, which made its edges untraceable.

    Every other stage checksums what it read. This one wrote `"inputs": {}`, so
    `alignment_edges.parquet` could not be tied to the registry that produced it
    -- the one thing the provenance discipline exists for. Found by running the
    pipeline on real data.

    The checksums are taken before the record is written, so T-09 still holds: a
    run that cannot load its registry leaves a record naming what it tried.
    """
    import json

    registry = _registry_arrays_h5(tmp_path, {
        "a": {"ppm": shared_ppm, "cwm": sign_flip_cwm},
        "b": {"ppm": short_overlap_target, "cwm": -sign_flip_cwm},
    })
    out = tmp_path / "evidence"
    align_registry(registry, out, null_shuffles=3, seed=4)

    records = json.loads((out / "provenance.json").read_text())
    assert len(records) == 1, "a second write would move the provenance_records join"
    inputs = records[0]["inputs"]
    assert inputs, "align recorded no inputs"
    assert {key.split(":", 1)[1] for key in inputs} >= {"registry.json"}
    assert all(len(digest) == 64 for digest in inputs.values())
