"""``compile.validate_compiled_lexicon`` refusing, one corruption at a time.

Branch coverage over the suite as it stood found the function running its whole
happy path on every run and **28 of its 44 refusals never once** -- among them
the ``lexicon_content_hash`` gate, which is the only reason ``FP-11`` can require
a family-level number to name the lexicon it was computed under. The sixteen
that did fire were reached sideways, through ``validate.load_lexicon_binding``,
not by anything testing this function. A refusal nothing exercises is a claim,
not a check: a corrupt lexicon may well have loaded clean, and the manifest would
still have said the hash matched. Measured again with this file present, all 44
fire, as do all 13 in the helpers it calls.

Every case here does two things, and the second is worthless without the first.
It validates the PRISTINE compiler output -- the same pair, freshly copied into
the case's own directory -- and only then corrupts exactly one thing and asserts
the refusal names that thing. A suite that skipped the pristine half would pass
just as happily against a validator that refused everything, which is how strict
loaders end up deleted for crying wolf; a suite that skipped the corrupt half
would pass against one that refused nothing, which is the state this file exists
to end.

The fixture is real compiler output -- ``ingest_project`` then
``compile_lexicons``, the same route ``test_ingest_compile`` takes. A
hand-rolled manifest would let the validator and the compiler drift apart and
would prove only that this file agrees with itself, which is precisely what the
function's docstring promises it does not do when it enumerates the whole HDF5
tree "rather than trusting a manifest-controlled subset".

Two refusals are unreachable through this door. They are documented below rather
than reached by contortion, because an unreachable refusal is a finding about
the validator, not a gap in the suite:

* the ``pattern_order`` half of the duplicate-name check -- see
  ``test_duplicate_pattern_order_is_refused_earlier_by_the_dataclass``;
* ``sensitivity_triggers``' ``type(cluster_id) is not str`` arm, which no JSON
  document can reach because JSON object keys are always strings. Only its
  non-empty arm is exercisable, and ``blank-sensitivity-trigger-key`` does that.

One more branch is reachable only under a non-default ``motif_type``: the
"lacks the dataset your motif_type requires" refusal is shadowed by the
unconditional ``contrib_scores`` check whenever ``motif_type == "cwm"``, so it
gets its own compiled pair (``hcwm_pair``) rather than a manifest edited to
disagree with the file beside it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from motifmultiverse import compile as compile_mod
from motifmultiverse import ingest

h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

MOTIF_LEN = 12


# --------------------------------------------------------------------------- #
# Fixture: one real compiled pair, built the way test_ingest_compile builds one.
# --------------------------------------------------------------------------- #
def _pattern(h5, group, name, seed):
    rng = np.random.default_rng(seed)
    grp = h5.require_group(group).create_group(name)
    cwm = rng.normal(size=(MOTIF_LEN, 4))
    grp.create_dataset("contrib_scores", data=cwm)
    grp.create_dataset("hypothetical_contribs", data=cwm * 0.5)
    ppm = np.abs(rng.normal(size=(MOTIF_LEN, 4)))
    grp.create_dataset("sequence", data=ppm / ppm.sum(axis=1, keepdims=True))
    grp.create_group("seqlets").create_dataset("n_seqlets", data=np.array(250 + seed))


def _modisco(path):
    with h5py.File(path, "w") as h5:
        for i in range(3):
            _pattern(h5, "pos_patterns", f"pattern_{i}", seed=i)
        for i in range(2):
            _pattern(h5, "neg_patterns", f"pattern_{i}", seed=100 + i)
    return path


def _registry(root: Path) -> Path:
    """A five-motif registry: three positives, two negatives, one analysis.

    Both metaclusters are populated on purpose. The refusals that walk the HDF5
    tree compare the manifest's *group universe* against the file's, and a
    lexicon with an empty ``neg_patterns`` would leave the negative half of that
    comparison untested.
    """
    config = {
        "project": "test-project",
        "peak_universe_id": "u1",
        "analyses": [{
            "id": "modelA_r1", "model": "modelA", "readout": "r1", "union_id": "MA",
            "context": "promoter", "modisco_h5": str(_modisco(root / "a.h5")),
        }],
    }
    (root / "project.json").write_text(json.dumps(config))
    ingest.ingest_project(root / "project.json", root / "registry")
    return root / "registry"


def _compile_pair(root: Path, **options) -> tuple[Path, Path]:
    lexicons = root / "lex"
    compile_mod.compile_lexicons(_registry(root), lexicons, verify="skip", **options)
    return lexicons / "core.manifest.json", lexicons / "core.h5"


@pytest.fixture(scope="session")
def compiled_core(tmp_path_factory) -> tuple[Path, Path]:
    """The one compiled pair every case corrupts a private copy of.

    Session-scoped because compiling is the expensive part and none of the cases
    may mutate the original -- each gets its own copy via ``pristine``.
    """
    return _compile_pair(tmp_path_factory.mktemp("compiled-core"))


@pytest.fixture(scope="session")
def compiled_hcwm(tmp_path_factory) -> tuple[Path, Path]:
    """A pair whose declared ``motif_type`` makes a *second* dataset mandatory."""
    return _compile_pair(tmp_path_factory.mktemp("compiled-hcwm"), motif_type="hcwm")


def _private_copy(source: tuple[Path, Path], destination: Path) -> tuple[Path, Path]:
    manifest = destination / "core.manifest.json"
    h5 = destination / "core.h5"
    shutil.copy2(source[0], manifest)
    shutil.copy2(source[1], h5)
    return manifest, h5


@pytest.fixture
def pristine(compiled_core, tmp_path) -> tuple[Path, Path]:
    return _private_copy(compiled_core, tmp_path)


@pytest.fixture
def hcwm_pair(compiled_hcwm, tmp_path) -> tuple[Path, Path]:
    return _private_copy(compiled_hcwm, tmp_path)


# --------------------------------------------------------------------------- #
# Corruption plumbing
# --------------------------------------------------------------------------- #
def _manifest_edit(mutate):
    """Adapt a payload mutator into a ``(manifest_path, h5_path)`` corruption."""
    def corrupt(manifest_path: Path, h5_path: Path) -> None:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutate(payload)
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return corrupt


def _h5_edit(mutate):
    """Adapt an open-file mutator into a ``(manifest_path, h5_path)`` corruption."""
    def corrupt(manifest_path: Path, h5_path: Path) -> None:
        with h5py.File(h5_path, "r+") as h5:
            mutate(h5)
    return corrupt


def _put(target, key, value):
    target[key] = value
    return target


def _drop(target, key):
    del target[key]
    return target


def _retag(payload, position, pattern_tag):
    """Rename one index row's pattern tag *and* the manifest's copy of it.

    The two have to move together or the index/pattern_order agreement check
    fires first and the branch under test is never reached -- which would make
    the case a duplicate of ``index-rows-swapped`` wearing a different name.
    """
    payload["index"][position]["pattern_tag"] = pattern_tag
    payload["pattern_order"][position] = pattern_tag
    return payload


def _reorder(payload, order):
    """Permute index, pattern_order and node_ids together, keeping ``index`` canonical.

    A permutation applied consistently leaves every per-row and cross-list check
    satisfied, so what is left to refuse is the loader-order rule itself.
    """
    rows = [dict(payload["index"][position]) for position in order]
    for position, row in enumerate(rows):
        row["index"] = position
    payload["index"] = rows
    payload["pattern_order"] = [row["pattern_tag"] for row in rows]
    payload["node_ids"] = [row["node_id"] for row in rows]
    return payload


def _flip_one_hex_character(payload):
    original = payload["lexicon_content_hash"]
    payload["lexicon_content_hash"] = ("0" if original[0] != "0" else "1") + original[1:]
    return payload


def _replace_dataset(h5, path, data):
    del h5[path]
    h5.create_dataset(path, data=data)
    return h5


def _shatter_h5(manifest_path: Path, h5_path: Path) -> None:
    h5_path.write_bytes(b"this is not an HDF5 superblock")


def _delete_h5(manifest_path: Path, h5_path: Path) -> None:
    h5_path.unlink()


def _delete_manifest(manifest_path: Path, h5_path: Path) -> None:
    manifest_path.unlink()


def _write_non_json_manifest(manifest_path: Path, h5_path: Path) -> None:
    manifest_path.write_text("{ this was truncated by a full disk", encoding="utf-8")


def _write_json_array_manifest(manifest_path: Path, h5_path: Path) -> None:
    manifest_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")


def _soft_link_a_motif(h5):
    """Point one motif name at another motif's object through a soft link.

    Compiler-emitted trees are local hard links only. A soft link is how a
    lexicon acquires a motif it does not physically contain -- the bytes the hash
    covers would then live somewhere the artifact does not name.
    """
    del h5["pos_patterns/pattern_0"]
    h5["pos_patterns/pattern_0"] = h5py.SoftLink("/pos_patterns/pattern_1")
    return h5


def _alias_two_motifs(h5):
    """Two pattern names, one object: the manifest claims five motifs, the file has four."""
    del h5["pos_patterns/pattern_0"]
    h5["pos_patterns/pattern_0"] = h5["pos_patterns/pattern_1"]
    return h5


#: ``(id, corruption, expected message fragment)``. The fragment is matched with
#: ``re.search`` against the ``CompileError``, and it is deliberately specific:
#: a case that merely asserts "something was refused" cannot tell the branch it
#: aimed at from an earlier branch that happened to fire first, and several of
#: these corruptions are only one ordering accident away from each other.
CASES = [
    # -- the manifest as a document ----------------------------------------- #
    ("manifest-is-not-json", _write_non_json_manifest, r"is not valid JSON"),
    ("manifest-file-absent", _delete_manifest, r"is not valid JSON"),
    ("manifest-is-a-json-array", _write_json_array_manifest, r"manifest must be a JSON object"),
    ("manifest-field-missing", _manifest_edit(lambda p: _drop(p, "project")),
     r"manifest schema is not exact: missing \['project'\]"),
    ("manifest-field-extra", _manifest_edit(lambda p: _put(p, "notes", "hand-edited")),
     r"manifest schema is not exact: unknown \['notes'\]"),

    # -- scalar field types -------------------------------------------------- #
    ("tier-is-a-number", _manifest_edit(lambda p: _put(p, "tier", 5)),
     r"manifest tier has type int"),
    ("n-motifs-is-a-bool", _manifest_edit(lambda p: _put(p, "n_motifs", True)),
     r"manifest n_motifs has type bool"),
    ("n-motifs-is-a-string", _manifest_edit(lambda p: _put(p, "n_motifs", "5")),
     r"manifest n_motifs has type str"),
    ("project-is-blank", _manifest_edit(lambda p: _put(p, "project", "   ")),
     r"manifest project must be non-empty"),
    ("loader-backend-is-blank", _manifest_edit(lambda p: _put(p, "loader_backend", "")),
     r"manifest loader_backend must be non-empty"),
    ("trim-threshold-is-a-string", _manifest_edit(lambda p: _put(p, "trim_threshold", "0.3")),
     r"manifest trim_threshold has type str"),
    ("trim-threshold-is-infinite",
     _manifest_edit(lambda p: _put(p, "trim_threshold", float("inf"))),
     r"manifest trim_threshold must be finite"),
    ("trim-threshold-is-nan", _manifest_edit(lambda p: _put(p, "trim_threshold", float("nan"))),
     r"manifest trim_threshold must be finite"),
    ("include-rc-is-an-integer", _manifest_edit(lambda p: _put(p, "include_rc", 1)),
     r"manifest include_rc must be a boolean"),
    ("cross-model-flag-is-a-string",
     _manifest_edit(lambda p: _put(p, "cross_model_claims_restricted", "yes")),
     r"manifest cross_model_claims_restricted must be a boolean"),
    ("loader-parameters-is-an-array", _manifest_edit(lambda p: _put(p, "loader_parameters", [])),
     r"manifest loader_parameters must be an object"),
    ("comparisons-is-an-array", _manifest_edit(lambda p: _put(p, "comparisons", [])),
     r"manifest comparisons must be an object"),
    ("sensitivity-triggers-is-an-array",
     _manifest_edit(lambda p: _put(p, "sensitivity_triggers", [])),
     r"manifest sensitivity_triggers must be an object"),
    ("pattern-order-is-an-object", _manifest_edit(lambda p: _put(p, "pattern_order", {})),
     r"manifest pattern_order must be an array"),
    ("node-ids-is-an-object", _manifest_edit(lambda p: _put(p, "node_ids", {})),
     r"manifest node_ids must be an array"),
    ("index-is-an-object", _manifest_edit(lambda p: _put(p, "index", {})),
     r"manifest index must be an array"),

    # `ppm` is our registry array's name; the loader dispatches on cwm/hcwm/pfm
    # and leaves its motif variables unbound for anything else. _MOTIF_TYPE_DATASET
    # once said `ppm`, so a lexicon was written that its own declared loader could
    # not read -- this refusal is the table that keeps the two vocabularies apart.
    ("motif-type-is-our-name-not-the-loaders",
     _manifest_edit(lambda p: _put(p, "motif_type", "ppm")),
     r"manifest motif_type must be one of \['cwm', 'hcwm', 'pfm'\]"),

    # motif_lambda_default is *resolved* at compile time and folded into the
    # hash, so a manifest that does not carry it cannot be re-verified at all.
    ("motif-lambda-default-missing",
     _manifest_edit(lambda p: _drop(p["loader_parameters"], "motif_lambda_default")),
     r"loader_parameters must contain the finite resolved motif_lambda_default"),
    ("motif-lambda-default-is-a-bool",
     _manifest_edit(lambda p: _put(p["loader_parameters"], "motif_lambda_default", True)),
     r"loader_parameters must contain the finite resolved motif_lambda_default"),
    ("motif-lambda-default-is-nan",
     _manifest_edit(lambda p: _put(p["loader_parameters"], "motif_lambda_default", float("nan"))),
     r"loader_parameters must contain the finite resolved motif_lambda_default"),

    # -- comparisons: the block that says what a tier contrast varies -------- #
    ("comparison-names-its-own-tier",
     _manifest_edit(lambda p: _put(p["comparisons"], "core", p["comparisons"]["expanded"])),
     r"manifest comparisons has an invalid tier key"),
    ("comparison-names-an-unknown-tier",
     _manifest_edit(lambda p: _put(p["comparisons"], "legacy", p["comparisons"]["expanded"])),
     r"manifest comparisons has an invalid tier key"),
    ("comparison-is-an-array",
     _manifest_edit(lambda p: _put(p["comparisons"], "expanded", [])),
     r"manifest comparisons\['expanded'\] must be an object"),
    ("comparison-has-an-extra-field",
     _manifest_edit(lambda p: _put(p["comparisons"]["expanded"], "n_shared", 3)),
     r"manifest comparisons\['expanded'\] schema is not exact"),
    ("comparison-is-missing-a-field",
     _manifest_edit(lambda p: _drop(p["comparisons"]["expanded"], "only_here")),
     r"manifest comparisons\['expanded'\] schema is not exact"),
    ("comparison-identity-flag-is-a-string",
     _manifest_edit(lambda p: _put(p["comparisons"]["expanded"], "positive_sets_identical", "yes")),
     r"comparisons\['expanded'\]\.positive_sets_identical must be a boolean"),
    ("comparison-identity-flag-is-null",
     _manifest_edit(lambda p: _put(p["comparisons"]["expanded"], "negative_sets_identical", None)),
     r"comparisons\['expanded'\]\.negative_sets_identical must be a boolean"),
    ("comparison-count-is-negative",
     _manifest_edit(lambda p: _put(p["comparisons"]["expanded"], "n_positive_here", -1)),
     r"comparisons\['expanded'\]\.n_positive_here must be a non-negative integer"),
    ("comparison-count-is-a-float",
     _manifest_edit(lambda p: _put(p["comparisons"]["expanded"], "n_negative_there", 2.0)),
     r"comparisons\['expanded'\]\.n_negative_there must be a non-negative integer"),
    ("comparison-only-here-is-unsorted",
     _manifest_edit(lambda p: _put(p["comparisons"]["expanded"], "only_here", ["b", "a"])),
     r"comparisons\['expanded'\]\.only_here must be a sorted unique string array"),
    ("comparison-only-there-has-duplicates",
     _manifest_edit(lambda p: _put(p["comparisons"]["expanded"], "only_there", ["a", "a"])),
     r"comparisons\['expanded'\]\.only_there must be a sorted unique string array"),
    ("comparison-only-here-holds-a-number",
     _manifest_edit(lambda p: _put(p["comparisons"]["expanded"], "only_here", [7])),
     r"comparisons\['expanded'\]\.only_here must be a sorted unique string array"),
    # The warning is the artifact's own admission that a contrast varies nothing
    # -- an empty one is worse than none, because the field looks answered.
    ("comparison-warning-is-blank",
     _manifest_edit(lambda p: _put(p["comparisons"]["expanded"], "warning", "  ")),
     r"comparisons\['expanded'\]\.warning must be a non-empty string"),

    # -- sensitivity_triggers: why a cluster stays split, by name not threshold #
    ("blank-sensitivity-trigger-key",
     _manifest_edit(lambda p: _put(p, "sensitivity_triggers", {" ": ["family_ambiguity"]})),
     r"manifest sensitivity_triggers keys must be non-empty strings"),
    ("sensitivity-trigger-is-not-a-named-trigger",
     _manifest_edit(lambda p: _put(p, "sensitivity_triggers", {"c1": ["merge_confidence_0_6"]})),
     r"sensitivity_triggers\['c1'\] must be a unique non-empty array of named compiler triggers"),
    ("sensitivity-trigger-list-is-empty",
     _manifest_edit(lambda p: _put(p, "sensitivity_triggers", {"c1": []})),
     r"sensitivity_triggers\['c1'\] must be a unique non-empty array of named compiler triggers"),
    ("sensitivity-trigger-list-repeats-itself",
     _manifest_edit(lambda p: _put(
         p, "sensitivity_triggers", {"c1": ["family_ambiguity", "family_ambiguity"]})),
     r"sensitivity_triggers\['c1'\] must be a unique non-empty array of named compiler triggers"),

    # -- manifest identity --------------------------------------------------- #
    ("pattern-order-shorter-than-n-motifs",
     _manifest_edit(lambda p: _put(p, "pattern_order", p["pattern_order"][:-1])),
     r"manifest is malformed"),
    ("schema-version-is-from-another-release",
     _manifest_edit(lambda p: _put(p, "schema_version", "2.0")),
     r"manifest schema_version must be '1\.0'"),
    ("tier-is-not-one-of-the-three",
     _manifest_edit(lambda p: _put(p, "tier", "legacy")),
     r"manifest has unknown tier 'legacy'"),
    ("n-motifs-is-zero",
     _manifest_edit(lambda p: _reorder(_put(_put(_put(
         p, "n_motifs", 0), "pattern_order", []), "node_ids", []), [])),
     r"manifest must describe at least one motif"),
    ("content-hash-is-uppercase",
     _manifest_edit(lambda p: _put(p, "lexicon_content_hash",
                                   p["lexicon_content_hash"].upper())),
     r"manifest has no lowercase SHA-256 content hash"),
    ("content-hash-is-truncated",
     _manifest_edit(lambda p: _put(p, "lexicon_content_hash",
                                   p["lexicon_content_hash"][:-1])),
     r"manifest has no lowercase SHA-256 content hash"),
    ("pattern-order-holds-a-number",
     _manifest_edit(lambda p: _put(p["pattern_order"], 0, 0)),
     r"manifest pattern_order must contain strings"),
    ("node-ids-holds-a-blank-string",
     _manifest_edit(lambda p: _put(p["node_ids"], 0, " ")),
     r"manifest node_ids must contain strings"),
    ("node-ids-repeats-an-id",
     _manifest_edit(lambda p: _put(p["node_ids"], 1, p["node_ids"][0])),
     r"manifest has duplicate node_ids"),

    # -- the index rows ------------------------------------------------------ #
    ("index-drops-a-row",
     _manifest_edit(lambda p: _put(p, "index", p["index"][:-1])),
     r"manifest index must describe every motif"),
    ("index-row-is-an-array",
     _manifest_edit(lambda p: _put(p["index"], 0, ["pos_patterns.pattern_0"])),
     r"manifest index row 0 must be an object"),
    ("index-row-has-an-extra-field",
     _manifest_edit(lambda p: _put(p["index"][0], "seqlet_count", 250)),
     r"manifest index row 0 schema is not exact: unknown \['seqlet_count'\]"),
    ("index-row-is-missing-a-field",
     _manifest_edit(lambda p: _drop(p["index"][2], "variant_id")),
     r"manifest index row 2 schema is not exact: missing \['variant_id'\]"),
    ("index-row-index-is-a-string",
     _manifest_edit(lambda p: _put(p["index"][1], "index", "1")),
     r"manifest index row 1 index must be an integer"),
    ("index-row-index-is-noncanonical",
     _manifest_edit(lambda p: _put(p["index"][3], "index", 7)),
     r"manifest index row 3 has noncanonical index 7"),
    ("index-row-node-id-is-blank",
     _manifest_edit(lambda p: _put(p["index"][2], "node_id", "")),
     r"manifest index row 2 node_id must be a non-empty string"),
    ("index-row-metacluster-is-a-number",
     _manifest_edit(lambda p: _put(p["index"][0], "metacluster", 1)),
     r"manifest index row 0 metacluster must be a non-empty string"),
    ("index-repeats-a-pattern-tag",
     _manifest_edit(lambda p: _put(p["index"][1], "pattern_tag",
                                   p["index"][0]["pattern_tag"])),
     r"manifest index has duplicate pattern identity"),
    ("index-repeats-a-node-id",
     _manifest_edit(lambda p: _put(p["index"][1], "node_id", p["index"][0]["node_id"])),
     r"manifest index has duplicate node identity"),
    # variant_id is the only stable semantic identity downstream groups by, and
    # it is in the content hash for that reason; a malformed one is not a name.
    ("index-variant-id-is-malformed",
     _manifest_edit(lambda p: _put(p["index"][0], "variant_id", "MA_UNASSIGNED_0")),
     r"manifest index has a malformed variant identity"),
    ("index-repeats-a-variant-id",
     _manifest_edit(lambda p: _put(p["index"][1], "variant_id",
                                   p["index"][0]["variant_id"])),
     r"manifest index has duplicate variant identity"),
    # Rows swapped while pattern_order/node_ids stay put: every positional read
    # against the manifest would then name the wrong motif.
    ("index-rows-swapped-against-pattern-order",
     _manifest_edit(lambda p: _put(p, "index", [
         {**p["index"][1], "index": 0}, {**p["index"][0], "index": 1}, *p["index"][2:]])),
     r"manifest index order does not match pattern_order/node_ids"),
    ("index-pattern-tag-names-an-unknown-group",
     _manifest_edit(lambda p: _retag(p, 0, "seqlets.pattern_0")),
     r"manifest index pattern_tag 'seqlets\.pattern_0' is not compiler-emitted"),
    ("index-pattern-tag-has-three-pieces",
     _manifest_edit(lambda p: _retag(p, 0, "pos_patterns.pattern_0.cwm")),
     r"manifest index pattern_tag 'pos_patterns\.pattern_0\.cwm' is not compiler-emitted"),
    ("index-pattern-numbering-skips",
     _manifest_edit(lambda p: _retag(p, 0, "pos_patterns.pattern_5")),
     r"manifest index pattern 'pattern_5' is not the next compiler-emitted name 'pattern_0'"),
    ("index-metacluster-contradicts-its-group",
     _manifest_edit(lambda p: _put(p["index"][0], "metacluster", "neg")),
     r"manifest index metacluster does not match pos_patterns"),
    # The loader walks pos_patterns then neg_patterns, whole groups at a time.
    # An index that interleaves them is internally consistent row by row and
    # still describes an order the loader will never return.
    ("index-interleaves-the-two-groups",
     _manifest_edit(lambda p: _reorder(p, [0, 1, 3, 2, 4])),
     r"manifest index is not in compiler loader order"),

    # -- the HDF5 tree ------------------------------------------------------- #
    ("h5-file-absent", _delete_h5, r"is not readable HDF5"),
    ("h5-file-is-not-hdf5", _shatter_h5, r"is not readable HDF5"),
    ("h5-has-an-extra-root-group",
     _h5_edit(lambda h5: h5.create_group("unindexed_patterns")),
     r"HDF5 root groups do not exactly match the manifest universe"),
    ("h5-root-group-is-a-dataset",
     _h5_edit(lambda h5: _replace_dataset(h5, "neg_patterns", np.zeros((2, 4)))),
     r"HDF5 neg_patterns must be a group"),
    ("h5-group-has-an-extra-motif",
     _h5_edit(lambda h5: h5.create_group("pos_patterns/pattern_9")),
     r"HDF5 motifs in pos_patterns do not exactly match the manifest universe"),
    ("h5-group-is-missing-a-motif",
     _h5_edit(lambda h5: _drop(h5, "pos_patterns/pattern_2")),
     r"HDF5 motifs in pos_patterns do not exactly match the manifest universe"),
    ("h5-motif-is-a-dataset",
     _h5_edit(lambda h5: _replace_dataset(h5, "pos_patterns/pattern_0",
                                          np.zeros((MOTIF_LEN, 4)))),
     r"HDF5 motif pos_patterns\.pattern_0 must be a group"),
    ("h5-motif-lacks-contrib-scores",
     _h5_edit(lambda h5: _drop(h5, "pos_patterns/pattern_1/contrib_scores")),
     r"HDF5 motif pos_patterns\.pattern_1 lacks contrib_scores"),
    # Anything the compiler did not write is unindexed: it is not in the hash,
    # so it is a payload the artifact's identity does not cover.
    ("h5-motif-has-an-extra-dataset",
     _h5_edit(lambda h5: h5.create_dataset("pos_patterns/pattern_0/trimmed_core",
                                           data=np.array([2, 10]))),
     r"HDF5 motif pos_patterns\.pattern_0 has unindexed dataset/group \['trimmed_core'\]"),
    ("h5-motif-has-an-extra-subgroup",
     _h5_edit(lambda h5: h5.create_group("pos_patterns/pattern_0/seqlets")),
     r"HDF5 motif pos_patterns\.pattern_0 has unindexed dataset/group \['seqlets'\]"),
    ("h5-contrib-scores-is-a-group",
     _h5_edit(lambda h5: (_drop(h5, "pos_patterns/pattern_0/contrib_scores"),
                          h5.create_group("pos_patterns/pattern_0/contrib_scores"))),
     r"HDF5 motif pos_patterns\.pattern_0/contrib_scores must be a dataset"),
    ("h5-dataset-has-five-columns",
     _h5_edit(lambda h5: _replace_dataset(h5, "pos_patterns/pattern_0/contrib_scores",
                                          np.zeros((MOTIF_LEN, 5)))),
     r"HDF5 motif pos_patterns\.pattern_0/contrib_scores must have shape \(width, 4\)"),
    ("h5-dataset-is-one-dimensional",
     _h5_edit(lambda h5: _replace_dataset(h5, "neg_patterns/pattern_0/sequence",
                                          np.zeros(MOTIF_LEN * 4))),
     r"HDF5 motif neg_patterns\.pattern_0/sequence must have shape \(width, 4\)"),
    ("h5-dataset-is-float32",
     _h5_edit(lambda h5: _replace_dataset(h5, "pos_patterns/pattern_2/hypothetical_contribs",
                                          np.zeros((MOTIF_LEN, 4), dtype=np.float32))),
     r"pattern_2/hypothetical_contribs must contain compiler-emitted float64 values"),
    # The loader stacks every motif into one array, so a lexicon whose patterns
    # differ in width cannot be read back at all -- the compiler refuses to write
    # one, and the validator has to refuse to accept one it did not write.
    ("h5-motifs-have-mixed-widths",
     _h5_edit(lambda h5: _replace_dataset(h5, "pos_patterns/pattern_1/contrib_scores",
                                          np.zeros((8, 4)))),
     r"HDF5 motifs have mixed widths \[8, 12\]"),
    ("h5-motif-is-a-soft-link", _h5_edit(_soft_link_a_motif),
     r"HDF5 pos_patterns\.pattern_0 uses a nonlocal or soft link"),
    ("h5-motifs-alias-one-object", _h5_edit(_alias_two_motifs),
     r"HDF5 pos_patterns\.pattern_1 aliases compiler object 'pos_patterns\.pattern_0'"),

    # -- the content hash gate ----------------------------------------------- #
    # The three ways a hash stops matching: the hash moves, the bytes move, or
    # the identity the bytes are hashed under moves. The third is the one that
    # motivated putting variant_id in the hash at all -- two lexicons resolving
    # the same tags to different variants must not share an identity, or the
    # FP-11 citation names a lexicon nothing can tell apart from another.
    ("content-hash-has-one-character-flipped", _manifest_edit(_flip_one_hex_character),
     r"lexicon_content_hash does not match the complete HDF5 motif universe"),
    ("h5-array-value-changed",
     _h5_edit(lambda h5: h5["neg_patterns/pattern_1/contrib_scores"].__setitem__(
         (0, 0), 1.5)),
     r"lexicon_content_hash does not match the complete HDF5 motif universe"),
    ("variant-id-reassigned-but-well-formed",
     _manifest_edit(lambda p: _put(p["index"][0], "variant_id", "MA_REASSIGNED_99")),
     r"lexicon_content_hash does not match the complete HDF5 motif universe"),
]


@pytest.mark.parametrize(
    "corrupt,expected",
    [pytest.param(corrupt, expected, id=case_id) for case_id, corrupt, expected in CASES],
)
def test_one_corruption_at_a_time_is_refused_by_name(pristine, corrupt, expected):
    """The pristine pair loads; the pair with one thing wrong is refused, by name.

    Both halves are the test. The pristine assertion is what stops this file
    passing against a validator that refuses everything -- a real risk for a
    function whose refusals had never once been observed, because nothing would
    have noticed if one of them had been over-broad. The message assertion is
    what stops a case claiming a branch that a *different*, earlier branch
    actually fired on; several of these corruptions sit one ordering accident
    apart from each other.
    """
    manifest_path, h5_path = pristine
    assert compile_mod.validate_compiled_lexicon(manifest_path).tier == "core"

    corrupt(manifest_path, h5_path)

    with pytest.raises(compile_mod.CompileError, match=expected):
        compile_mod.validate_compiled_lexicon(manifest_path)


def test_every_case_corrupts_something_the_validator_can_still_find(pristine):
    """No two cases share an id, and the table is not silently empty.

    Parametrised suites fail open: a table that lost its entries to a bad edit
    still reports green, and duplicate ids make pytest run only the last of the
    colliding cases while the report keeps showing the count it had before.
    """
    ids = [case_id for case_id, _corrupt, _expected in CASES]
    assert len(ids) == len(set(ids))
    assert len(ids) > 70
    assert compile_mod.validate_compiled_lexicon(pristine[0]).n_motifs == 5


def test_validation_is_reproducible_on_an_untouched_pair(pristine):
    """Validating twice returns the same manifest and mutates neither file.

    The validator opens the HDF5 read-only and recomputes the hash from it; if
    either call left a byte behind, every corruption case above would be racing
    its own fixture.
    """
    manifest_path, h5_path = pristine
    before = h5_path.read_bytes(), manifest_path.read_bytes()

    first = compile_mod.validate_compiled_lexicon(manifest_path)
    second = compile_mod.validate_compiled_lexicon(manifest_path, h5_path)

    assert first == second
    assert (h5_path.read_bytes(), manifest_path.read_bytes()) == before


def test_the_dataset_a_declared_motif_type_requires_is_separately_enforced(hcwm_pair):
    """``motif_type='hcwm'`` makes ``hypothetical_contribs`` mandatory too.

    This branch is unreachable under the default ``motif_type='cwm'``: the
    unconditional ``contrib_scores`` check fires first on the very same missing
    dataset, so the refusal that names the *declared* motif type only exists for
    lexicons compiled to be read back as hcwm or pfm. It gets a compiled pair of
    its own rather than a cwm manifest edited to claim hcwm, because a manifest
    that disagrees with the file beside it is a different defect and would reach
    this branch for the wrong reason.
    """
    manifest_path, h5_path = hcwm_pair
    assert compile_mod.validate_compiled_lexicon(manifest_path).motif_type == "hcwm"

    with h5py.File(h5_path, "r+") as h5:
        del h5["pos_patterns/pattern_0/hypothetical_contribs"]

    with pytest.raises(
        compile_mod.CompileError,
        match=r"lacks hypothetical_contribs required by motif_type='hcwm'",
    ):
        compile_mod.validate_compiled_lexicon(manifest_path)


def test_duplicate_pattern_order_is_refused_earlier_by_the_dataclass(pristine):
    """The validator's own duplicate-``pattern_order`` check cannot be reached.

    ``LexiconManifest.__post_init__`` refuses duplicate pattern names, and it is
    constructed before the validator's ``len(set(values)) != len(values)`` test
    runs -- so that test's ``pattern_order`` half is dead code and the refusal
    the caller actually sees is the wrapped ``SchemaError``. Recorded rather than
    contorted into a pass: the duplicate IS refused, which is what matters to a
    caller, but anyone reading the validator should know that one of its
    branches has no reachable input. The ``node_ids`` half of the same check is
    live (``node-ids-repeats-an-id`` above), because the dataclass checks
    ``node_ids`` for length only.
    """
    manifest_path, _h5_path = pristine
    assert compile_mod.validate_compiled_lexicon(manifest_path).tier == "core"

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["pattern_order"][1] = payload["pattern_order"][0]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        compile_mod.CompileError,
        match=r"manifest is malformed: .*duplicate pattern names",
    ):
        compile_mod.validate_compiled_lexicon(manifest_path)
