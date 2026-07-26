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

from pathlib import Path

import pytest

from motifmultiverse import guards
from motifmultiverse.align import (
    DEFAULT_MIN_OVERLAP_BP,
    DEFAULT_MIN_OVERLAP_FRAC,
    AlignmentError,
    align_registry,
    calibrate_pair_null,
    register_pair,
    run,
)

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

FIXTURES = Path(__file__).parent / "fixtures" / "motifs"


def _load(name: str):
    return np.load(FIXTURES / name)


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
def _registry_arrays_h5(tmp_path, motifs: dict[str, dict[str, np.ndarray]]):
    """A minimal on-disk registry (registry.json + arrays.h5) with just enough
    structure for align_registry to read: node_id, and per-node ppm/cwm arrays.
    """
    import json

    h5py = pytest.importorskip("h5py")
    reg_dir = tmp_path / "registry"
    reg_dir.mkdir()
    nodes = []
    with h5py.File(reg_dir / "arrays.h5", "w") as h5:
        for node_id, arrays in motifs.items():
            grp = h5.create_group(node_id)
            for name, arr in arrays.items():
                grp.create_dataset(name, data=arr)
            nodes.append({
                "node_id": node_id, "model": "m", "readout": "r", "context": "c",
                "metacluster": "pos", "denovo_pattern_id": node_id,
                "variant_id": f"U_FAM_{len(nodes):02d}", "family_id": "FAM",
            })
    payload = {
        "registry_metadata": {
            "project": "p", "peak_universe_id": "u", "analyses": [],
            "n_models": 1, "cross_model_claims_restricted": True,
            "metacluster_states": {}, "trim_threshold": 0.3,
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
