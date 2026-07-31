"""What `report` may render, checked against the one real interpretation on disk.

`report` is the only stage that produces something a person reads rather than a
program consumes, so it is the only stage whose failures are invisible to the
rest of the suite: every other module can be checked by reading back the file it
wrote, and a report can be wrong in a way that round-trips perfectly. The rule it
exists for -- every number carries its denominator, its baseline population and
its provenance -- came out of a run where the same data supported both
"replicates exactly" and "4x stronger, prediction falsified", differing only in
which population the comparison was against, and where a bootstrap's resolution
floor was printed as though it were a measured p value. Both of those documents
would have round-tripped.

So the tests here are of three kinds, and only the first is about output:

**The refusals fire.** A share with no denominator, and a cross-condition effect
with no named baseline, each raise `ReportError` rather than render. These are
the check the module's README specifies; if they cannot be made to fail there is
no module, only a formatter.

**Nothing is recomputed.** The recorded value is asserted to appear as
`str(value)`, including for values deliberately doctored to be arithmetically
impossible: a coverage of 0.42 recorded beside `n_in_universe == n_submitted`
must render as 0.42. A renderer that divides is a second implementation of every
stage it renders, free to disagree with the one that wrote the artifact, and the
disagreement would surface as a number no one could trace.

**Absence cannot be mistaken for a verified result.** A suppressed reading, a
comparator-side suppression that leaves a perfectly good composition, a
`two_part_effects` that is null because nobody chose a definition of "used", and
a guard with no call site are four different absences. Each is asserted to be
distinguishable from a licensed number and from each other. The specific
inversion tested at the end -- concluding a guard is wired because it is absent
from `GUARDS_AWAITING_INPUT` -- is fabrication of exactly the shape this package
is organised against, and a previous attempt at this module committed it.

The artifact rendered is a byte-for-byte content copy of a real run over the
576,589-row K562 substrate (`interpret`, 8,277 query peaks against 25,640
comparator peaks), embedded below rather than referenced by path so the suite
does not depend on a scratch directory, and rather than synthesised so that no
test can be satisfied by a fixture shaped to the renderer's convenience. Doctored
copies are derived from it in-test, one field at a time, so what each refusal
turns on is one visible edit.

`report.run`'s signature is discovered rather than assumed (`_call_run`): these
tests were written against the rendering contract, beside rather than after the
implementation, and a parameter name is not part of that contract. Everything
else asserted here is.
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from motifmultiverse import cli, guards, report

# --------------------------------------------------------------------------- #
# The real artifact.
#
# `/.../mmvps/ok/interpretation.json` (sha256 297fe9fb...4a36a, 13,036 bytes) and
# `provenance.json` (sha256 8fbc87ed...22c37, 1,352 bytes), minified and
# key-sorted. Values are unaltered; `test_the_embedded_artifact_is_the_real_one`
# re-checks the invariants that make it the real one rather than a plausible one.
# --------------------------------------------------------------------------- #
_REAL_INTERPRETATION_JSON = (
    '{"claim_scope":"INTERNAL_DECOMPOSITION","comparator_health":{"block_size":1000000,"explained'
    '_fraction":1.0,"floor_failures":[],"floors":{"min_blocks":30.0,"min_explained_fraction":0.5,'
    '"min_intersection_coverage":0.9},"intersection_coverage":1.0,"n_blocks":326,"n_in_universe":'
    '25640,"n_searched":25640,"n_submitted":25640,"n_with_used_hit":25640},"composition":[{"famil'
    'y_id":"SP_KLF","mean_coefficient_per_peak":0.0008582166330275271,"n_peaks_searched":8277,"n_'
    'peaks_with_family":7545,"peak_share":0.9115621602029721},{"family_id":"ETS","mean_coefficien'
    't_per_peak":0.00045697082540351535,"n_peaks_searched":8277,"n_peaks_with_family":7557,"peak_'
    'share":0.9130119608553824},{"family_id":"AP-1/bZIP","mean_coefficient_per_peak":0.0004452721'
    '3246693247,"n_peaks_searched":8277,"n_peaks_with_family":8258,"peak_share":0.997704482300350'
    '4},{"family_id":"NFY","mean_coefficient_per_peak":0.000331035231859328,"n_peaks_searched":82'
    '77,"n_peaks_with_family":2731,"peak_share":0.329950465144376},{"family_id":"NRF1","mean_coef'
    'ficient_per_peak":0.0002771372531038084,"n_peaks_searched":8277,"n_peaks_with_family":5910,"'
    'peak_share":0.7140268213120696},{"family_id":"ZNF76","mean_coefficient_per_peak":0.000203564'
    '04406064393,"n_peaks_searched":8277,"n_peaks_with_family":742,"peak_share":0.089646007007369'
    '83},{"family_id":"CTCF/CTCFL-like","mean_coefficient_per_peak":0.00017477080848166197,"n_pea'
    'ks_searched":8277,"n_peaks_with_family":1806,"peak_share":0.21819499818774918},{"family_id":'
    '"GATA","mean_coefficient_per_peak":0.00015619435425440675,"n_peaks_searched":8277,"n_peaks_w'
    'ith_family":5470,"peak_share":0.6608674640570255},{"family_id":"KAISO","mean_coefficient_per'
    '_peak":0.00012601775818301572,"n_peaks_searched":8277,"n_peaks_with_family":5756,"peak_share'
    '":0.6954210462728042},{"family_id":"ZBTB17","mean_coefficient_per_peak":0.000102067014789511'
    '36,"n_peaks_searched":8277,"n_peaks_with_family":4549,"peak_share":0.5495952639845355},{"fam'
    'ily_id":"YY1","mean_coefficient_per_peak":8.522571993346744e-05,"n_peaks_searched":8277,"n_p'
    'eaks_with_family":1585,"peak_share":0.19149450283919295},{"family_id":"ZNF524","mean_coeffic'
    'ient_per_peak":3.630695178205437e-05,"n_peaks_searched":8277,"n_peaks_with_family":313,"peak'
    '_share":0.037815633683701826}],"contrast_health":{"comparator":{"block_size":1000000,"explai'
    'ned_fraction":1.0,"floor_failures":[],"floors":{"min_blocks":30.0,"min_explained_fraction":0'
    '.5,"min_intersection_coverage":0.9},"intersection_coverage":1.0,"n_blocks":326,"n_in_univers'
    'e":25640,"n_searched":25640,"n_submitted":25640,"n_with_used_hit":25640},"floor_failures":[]'
    ',"n_shared_peaks":0,"passed":true,"query":{"block_size":1000000,"explained_fraction":1.0,"fl'
    'oor_failures":[],"floors":{"min_blocks":30.0,"min_explained_fraction":0.5,"min_intersection_'
    'coverage":0.9},"intersection_coverage":1.0,"n_blocks":283,"n_in_universe":8277,"n_searched":'
    '8277,"n_submitted":8277,"n_with_used_hit":8277},"shared_blocks":282,"union_blocks":327},"eff'
    'ects":[{"block_size":1000000,"ci":[-0.0008270086303810119,-0.0006694220451666969],"comparato'
    'r_id":"not_cl5","effect":-0.0007394514607583663,"estimator":"percentile_block_bootstrap","fa'
    'mily_id":"AP-1/bZIP","id":"AP-1/bZIP_vs_not_cl5","inference_capability":"ESTIMATION_ONLY","i'
    's_cross_condition":true,"n_blocks":327,"n_bootstrap":100,"n_bootstrap_valid":100,"n_comparat'
    'or_peaks":25640,"n_query_peaks":8277,"p_value":null,"q_value":null,"random_seed":0},{"block_'
    'size":1000000,"ci":[-0.001441147774015579,-0.0011684297611618333],"comparator_id":"not_cl5",'
    '"effect":-0.001297269840287245,"estimator":"percentile_block_bootstrap","family_id":"CTCF/CT'
    'CFL-like","id":"CTCF/CTCFL-like_vs_not_cl5","inference_capability":"ESTIMATION_ONLY","is_cro'
    'ss_condition":true,"n_blocks":327,"n_bootstrap":100,"n_bootstrap_valid":100,"n_comparator_pe'
    'aks":25640,"n_query_peaks":8277,"p_value":null,"q_value":null,"random_seed":0},{"block_size"'
    ':1000000,"ci":[0.00027945864710375476,0.000366211470547075],"comparator_id":"not_cl5","effec'
    't":0.00033096215917924457,"estimator":"percentile_block_bootstrap","family_id":"ETS","id":"E'
    'TS_vs_not_cl5","inference_capability":"ESTIMATION_ONLY","is_cross_condition":true,"n_blocks"'
    ':327,"n_bootstrap":100,"n_bootstrap_valid":100,"n_comparator_peaks":25640,"n_query_peaks":82'
    '77,"p_value":null,"q_value":null,"random_seed":0},{"block_size":1000000,"ci":[-0.00075130085'
    '33059579,-0.0006272618142089644],"comparator_id":"not_cl5","effect":-0.0006946033124298974,"'
    'estimator":"percentile_block_bootstrap","family_id":"GATA","id":"GATA_vs_not_cl5","inference'
    '_capability":"ESTIMATION_ONLY","is_cross_condition":true,"n_blocks":327,"n_bootstrap":100,"n'
    '_bootstrap_valid":100,"n_comparator_peaks":25640,"n_query_peaks":8277,"p_value":null,"q_valu'
    'e":null,"random_seed":0},{"block_size":1000000,"ci":[8.025993950036184e-05,0.000144884073015'
    '06596],"comparator_id":"not_cl5","effect":0.00011019582042908721,"estimator":"percentile_blo'
    'ck_bootstrap","family_id":"KAISO","id":"KAISO_vs_not_cl5","inference_capability":"ESTIMATION'
    '_ONLY","is_cross_condition":true,"n_blocks":327,"n_bootstrap":100,"n_bootstrap_valid":100,"n'
    '_comparator_peaks":25640,"n_query_peaks":8277,"p_value":null,"q_value":null,"random_seed":0}'
    ',{"block_size":1000000,"ci":[0.0002715027088550537,0.0003531840061181434],"comparator_id":"n'
    'ot_cl5","effect":0.00031239906723773853,"estimator":"percentile_block_bootstrap","family_id"'
    ':"NFY","id":"NFY_vs_not_cl5","inference_capability":"ESTIMATION_ONLY","is_cross_condition":t'
    'rue,"n_blocks":327,"n_bootstrap":100,"n_bootstrap_valid":100,"n_comparator_peaks":25640,"n_q'
    'uery_peaks":8277,"p_value":null,"q_value":null,"random_seed":0},{"block_size":1000000,"ci":['
    '0.0002157551769140619,0.0002893283010333846],"comparator_id":"not_cl5","effect":0.0002574269'
    '2958793495,"estimator":"percentile_block_bootstrap","family_id":"NRF1","id":"NRF1_vs_not_cl5'
    '","inference_capability":"ESTIMATION_ONLY","is_cross_condition":true,"n_blocks":327,"n_boots'
    'trap":100,"n_bootstrap_valid":100,"n_comparator_peaks":25640,"n_query_peaks":8277,"p_value":'
    'null,"q_value":null,"random_seed":0},{"block_size":1000000,"ci":[0.0006026155201172424,0.000'
    '7579228580088533],"comparator_id":"not_cl5","effect":0.0006946250526876728,"estimator":"perc'
    'entile_block_bootstrap","family_id":"SP_KLF","id":"SP_KLF_vs_not_cl5","inference_capability"'
    ':"ESTIMATION_ONLY","is_cross_condition":true,"n_blocks":327,"n_bootstrap":100,"n_bootstrap_v'
    'alid":100,"n_comparator_peaks":25640,"n_query_peaks":8277,"p_value":null,"q_value":null,"ran'
    'dom_seed":0},{"block_size":1000000,"ci":[5.7282615926451916e-05,0.00010284177090568415],"com'
    'parator_id":"not_cl5","effect":7.499845429323956e-05,"estimator":"percentile_block_bootstrap'
    '","family_id":"YY1","id":"YY1_vs_not_cl5","inference_capability":"ESTIMATION_ONLY","is_cross'
    '_condition":true,"n_blocks":327,"n_bootstrap":100,"n_bootstrap_valid":100,"n_comparator_peak'
    's":25640,"n_query_peaks":8277,"p_value":null,"q_value":null,"random_seed":0},{"block_size":1'
    '000000,"ci":[-3.5124960738040404e-06,1.622392781508249e-05],"comparator_id":"not_cl5","effec'
    't":7.240694349781954e-06,"estimator":"percentile_block_bootstrap","family_id":"ZBTB17","id":'
    '"ZBTB17_vs_not_cl5","inference_capability":"ESTIMATION_ONLY","is_cross_condition":true,"n_bl'
    'ocks":327,"n_bootstrap":100,"n_bootstrap_valid":100,"n_comparator_peaks":25640,"n_query_peak'
    's":8277,"p_value":null,"q_value":null,"random_seed":0},{"block_size":1000000,"ci":[5.3745858'
    '07266622e-06,6.652868128724842e-05],"comparator_id":"not_cl5","effect":3.0742440075624337e-0'
    '5,"estimator":"percentile_block_bootstrap","family_id":"ZNF524","id":"ZNF524_vs_not_cl5","in'
    'ference_capability":"ESTIMATION_ONLY","is_cross_condition":true,"n_blocks":327,"n_bootstrap"'
    ':100,"n_bootstrap_valid":100,"n_comparator_peaks":25640,"n_query_peaks":8277,"p_value":null,'
    '"q_value":null,"random_seed":0},{"block_size":1000000,"ci":[0.00013674199256505796,0.0002541'
    '955447203242],"comparator_id":"not_cl5","effect":0.00019355121440708302,"estimator":"percent'
    'ile_block_bootstrap","family_id":"ZNF76","id":"ZNF76_vs_not_cl5","inference_capability":"EST'
    'IMATION_ONLY","is_cross_condition":true,"n_blocks":327,"n_bootstrap":100,"n_bootstrap_valid"'
    ':100,"n_comparator_peaks":25640,"n_query_peaks":8277,"p_value":null,"q_value":null,"random_s'
    'eed":0}],"emitted_order":["health","composition","effects"],"estimator":"percentile_block_bo'
    'otstrap","estimators_defined":["percentile_block_bootstrap","bca_paired_block_bootstrap","wi'
    'ld_cluster_bootstrap_t"],"estimators_implemented":["bca_paired_block_bootstrap","percentile_'
    'block_bootstrap","wild_cluster_bootstrap_t"],"floor_failures":[],"health":{"block_size":1000'
    '000,"explained_fraction":1.0,"floor_failures":[],"floors":{"min_blocks":30.0,"min_explained_'
    'fraction":0.5,"min_intersection_coverage":0.9},"intersection_coverage":1.0,"n_blocks":283,"n'
    '_in_universe":8277,"n_searched":8277,"n_submitted":8277,"n_with_used_hit":8277},"input_scale'
    '":33917,"interpretation_emitted":true,"lexicon_id":"CBP-2114__core_final","notes":["The '
    'implemented percentile block bootstrap supports estimation only. Hypothesis-test p and q '
    'values are withheld until the preregistered wild cluster bootstrap-t estimator is used."],"o'
    'utput_mode":"FULL_INFERENCE","query_health":{"block_size":1000000,"explained_fraction":1.0,"'
    'floor_failures":[],"floors":{"min_blocks":30.0,"min_explained_fraction":0.5,"min_intersectio'
    'n_coverage":0.9},"intersection_coverage":1.0,"n_blocks":283,"n_in_universe":8277,"n_searched'
    '":8277,"n_submitted":8277,"n_with_used_hit":8277},"query_id":"cl5","selection_provenance":"P'
    'ROGRAMMATIC_RULE","statistical_license":"FULL_INFERENCE","substrate_id":"c804f947098de3688ef'
    '90054c919bc628fda78057abf392dfe0e7be130108377","suppression_reason":null,"two_part_effects":'
    'null}'
)

_REAL_PROVENANCE_JSON = (
    '[{"command":"motifmultiverse interpret /data1/test/leixiong/kant/new_experiment/region_v4/04'
    '_cbp_only_islands/mmv/substrate_CBP2114_core_final.tsv --peaks /tmp/claude-26399/-data1-test'
    '-leixiong-kant/27c8cefc-97b6-4bae-8516-e5742c85b242/scratchpad/mmvps/cl5.txt --comparator /t'
    'mp/claude-26399/-data1-test-leixiong-kant/27c8cefc-97b6-4bae-8516-e5742c85b242/scratchpad/mm'
    'vps/not_cl5.txt --comparator-id not_cl5 --query-id cl5 --selection-provenance '
    'PROGRAMMATIC_RULE --selection-rule leiden res0.5 == 5 --bootstrap 100 --out /tmp/claude-2639'
    '9/-data1-test-leixiong-kant/27c8cefc-97b6-4bae-8516-e5742c85b242/scratchpad/mmvps/ok","input'
    '_scale":33917,"inputs":{"comparator:not_cl5.txt":"ba7989ea84b0891552ec650ddfc98bdd39cb4ea0d8'
    '8c7dc21a29b8987c38710b","hits:substrate_CBP2114_core_final.tsv":"557cf50902ba78ac481b37ffe8d'
    'cacb200f1c7ec645fe42b8e4571daf6ccfbe1","peaks:cl5.txt":"ba759d16140262e87202ca23b06f4b39bc5c'
    'b889f4cfb6fab71e3217208d9a21"},"random_seed":0,"redaction_policy":"basenames_only_except_com'
    'mand","schema_version":"1","software":{"motifmultiverse":"0.1.0.dev0","python":"3.13.12"},"s'
    'ubcommand":"interpret","substrate_id":"c804f947098de3688ef90054c919bc628fda78057abf392dfe0e7'
    'be130108377","timestamp_utc":"2026-07-31T05:49:08Z"}]'
)


REAL_INTERPRETATION: dict = json.loads(_REAL_INTERPRETATION_JSON)
REAL_PROVENANCE: list = json.loads(_REAL_PROVENANCE_JSON)


# --------------------------------------------------------------------------- #
# Calling the renderer.
#
# The contract fixes what is rendered and what is refused; it does not fix how
# `run` is spelled. These helpers bind to whatever signature the module exposes
# and fail with the signature in the message when they cannot, so a mismatch
# reads as "this test could not call the module" rather than as a rendering
# defect.
# --------------------------------------------------------------------------- #
_OUT_PARAM_NAMES = ("out", "out_dir", "outdir", "output", "output_dir", "dest", "destination")
_LEDGER_PARAM_NAMES = (
    "bias_ledger", "bias_ledger_path", "bias_ledger_tsv", "ledger", "ledger_path", "docs_dir",
)


def _report_error() -> type[BaseException]:
    """The exception type the module refuses with."""
    err = getattr(report, "ReportError", None)
    if not (isinstance(err, type) and issubclass(err, BaseException)):
        pytest.fail(
            "motifmultiverse.report exports no `ReportError` exception type, so a refusal "
            "cannot be distinguished from a crash. Every other stage names its refusal "
            "(IngestError, InterpretError, ...) and cli.main maps that name to exit 4."
        )
    return err


def _call_run(project: Path, out: Path, **options):
    fn = getattr(report, "run", None)
    if not callable(fn):
        pytest.fail("motifmultiverse.report exposes no callable `run`.")
    sig = inspect.signature(fn)
    params = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    if not params:
        pytest.fail(f"report.run takes no arguments: {sig}")

    args: list[str] = [str(project)]
    kwargs: dict[str, object] = {}

    tail = params[1:]
    out_param = next((p.name for p in tail if p.name in _OUT_PARAM_NAMES), None)
    if out_param is None:
        required_tail = [p.name for p in tail if p.default is inspect.Parameter.empty]
        if len(required_tail) == 1:
            out_param = required_tail[0]
    if out_param is not None:
        kwargs[out_param] = str(out)

    for name, value in options.items():
        if name in sig.parameters or accepts_var_kw:
            kwargs[name] = value
        else:
            pytest.fail(f"report.run has no `{name}` parameter; its signature is {sig}")
    try:
        sig.bind(*args, **kwargs)
    except TypeError as exc:  # pragma: no cover - signature mismatch, not a defect
        pytest.fail(f"could not call report.run{sig}: {exc}")
    return fn(*args, **kwargs)


def render(project: Path, out: Path, **options) -> str:
    """Render `project` into `out` and return the markdown a reader would read."""
    out.mkdir(parents=True, exist_ok=True)
    result = _call_run(project, out, **options)

    written = sorted(out.rglob("*.md"))
    if len(written) > 1:
        preferred = [p for p in written if "report" in p.name.lower()]
        written = preferred or written[:1]
    if written:
        return written[0].read_text(encoding="utf-8")
    if isinstance(result, Path):
        return result.read_text(encoding="utf-8")
    if isinstance(result, str) and "\n" in result:
        return result
    pytest.fail(
        f"report.run wrote no .md under {out} and returned {type(result).__name__}; "
        "there is no document to check."
    )


def _write_project(
    tmp_path: Path,
    name: str,
    interpretation: dict | None = None,
    provenance: list | None = None,
) -> Path:
    """A project directory shaped like `interpret --out`: the two files it writes."""
    project = tmp_path / name
    project.mkdir(parents=True, exist_ok=True)
    (project / "interpretation.json").write_text(
        json.dumps(REAL_INTERPRETATION if interpretation is None else interpretation, indent=2),
        encoding="utf-8",
    )
    (project / "provenance.json").write_text(
        json.dumps(REAL_PROVENANCE if provenance is None else provenance, indent=2),
        encoding="utf-8",
    )
    return project


def doctored(**edits) -> dict:
    """A deep copy of the real interpretation with top-level keys replaced."""
    record = json.loads(_REAL_INTERPRETATION_JSON)
    record.update(edits)
    return record


def _flat(text: str) -> str:
    """Whitespace-normalised, so a long recorded sentence survives markdown wrapping."""
    return re.sub(r"\s+", " ", text)


def _within(doc: str, anchor: str, needle: str, window: int = 400) -> bool:
    """Is `needle` rendered within `window` characters of `anchor`, either side?

    Either side on purpose. A label that qualifies a field usually precedes it --
    the renderer writes `**DEPRECATED** \u0060output_mode\u0060` -- and an
    after-only window asserts a rendering order the contract never specified,
    which is a test failing on prose layout rather than on the claim it is about.
    """
    flat = _flat(doc)
    start = flat.find(anchor)
    if start == -1:
        return False
    return needle in flat[max(0, start - window):start + len(anchor) + window]


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "docs" / "bias_ledger.tsv").is_file():
            return parent
    pytest.fail("docs/bias_ledger.tsv is not above tests/; section 9 has nothing to render")


def _ledger_rows() -> list[list[str]]:
    text = (_repo_root() / "docs" / "bias_ledger.tsv").read_text(encoding="utf-8")
    return [line.split("\t") for line in text.splitlines() if line.strip()]


@pytest.fixture
def real_project(tmp_path: Path) -> Path:
    return _write_project(tmp_path, "ok")


# --------------------------------------------------------------------------- #
# 0. The fixture is the artifact.
# --------------------------------------------------------------------------- #
def test_the_embedded_artifact_is_the_real_one():
    """The invariants that make this the real run rather than a convenient one.

    A fixture built to suit the renderer would not have a deprecated alias that
    is byte-equal to the field it aliases, a `two_part_effects` that is null for
    a reason, or twelve effect rows that all withhold p and q while carrying an
    interval. Each of those is a case the renderer is required to handle, and
    each is here because a real run produced it.
    """
    record = REAL_INTERPRETATION
    assert set(record) == {
        "claim_scope", "comparator_health", "composition", "contrast_health", "effects",
        "emitted_order", "estimator", "estimators_defined", "estimators_implemented",
        "floor_failures", "health", "input_scale", "interpretation_emitted", "lexicon_id",
        "notes", "output_mode", "query_health", "query_id", "selection_provenance",
        "statistical_license", "substrate_id", "suppression_reason", "two_part_effects",
    }
    assert record["health"] == record["query_health"]  # the deprecated alias, byte-equal here
    assert record["emitted_order"] == ["health", "composition", "effects"]
    assert record["two_part_effects"] is None
    assert record["floor_failures"] == []
    assert len(record["composition"]) == 12
    assert len(record["effects"]) == 12
    assert all(row["p_value"] is None and row["q_value"] is None for row in record["effects"])
    assert all(row["inference_capability"] == "ESTIMATION_ONLY" for row in record["effects"])
    assert all(row["comparator_id"] == "not_cl5" for row in record["effects"])
    assert record["query_health"]["n_submitted"] == 8277
    assert record["comparator_health"]["n_submitted"] == 25640
    assert record["contrast_health"]["n_shared_peaks"] == 0
    # No artifact carries it. Section 10 exists because of this line.
    assert "baseline_population" not in json.dumps(REAL_INTERPRETATION)
    assert "baseline_population" not in json.dumps(REAL_PROVENANCE)
    assert len(REAL_PROVENANCE) == 1 and REAL_PROVENANCE[0]["subcommand"] == "interpret"
    assert "substrate_manifest" not in REAL_PROVENANCE[0]["inputs"]


# --------------------------------------------------------------------------- #
# 1. The refusals.
# --------------------------------------------------------------------------- #
def _drop_composition_denominator(record: dict) -> dict:
    record["composition"][0].pop("n_peaks_searched")
    return record


def _drop_effect_query_denominator(record: dict) -> dict:
    record["effects"][0].pop("n_query_peaks")
    return record


def _drop_effect_comparator_denominator(record: dict) -> dict:
    record["effects"][0].pop("n_comparator_peaks")
    return record


def _drop_health_denominator(record: dict) -> dict:
    # Dropped from the alias too, so the refusal cannot be R14's instead.
    record["query_health"].pop("n_submitted")
    record["health"].pop("n_submitted")
    return record


@pytest.mark.parametrize(
    ("mutate", "missing"),
    [
        (_drop_composition_denominator, "n_peaks_searched"),
        (_drop_effect_query_denominator, "n_query_peaks"),
        (_drop_effect_comparator_denominator, "n_comparator_peaks"),
        (_drop_health_denominator, "n_submitted"),
    ],
    ids=["composition_share", "effect_query", "effect_comparator", "health_coverage"],
)
def test_a_figure_with_no_denominator_is_refused(tmp_path, mutate, missing):
    """A share whose denominator is not on the record does not get rendered.

    `peak_share = 0.91` is a different claim depending on whether it is 0.91 of
    8,277 searched peaks or of some residual subset, and the renderer cannot
    tell which from the number alone -- which is why it may not print the number
    alone, and may not supply the denominator from anywhere but the record.
    """
    project = _write_project(tmp_path, "no_denominator", mutate(doctored()))
    with pytest.raises(_report_error()) as excinfo:
        render(project, tmp_path / "out")
    assert missing in str(excinfo.value), (
        "the refusal must name the field whose absence caused it, or a reader cannot "
        f"tell which denominator is missing; got: {excinfo.value}"
    )


def _strip_comparator_id(record: dict) -> dict:
    for row in record["effects"]:
        row.pop("comparator_id")
        row["id"] = row["family_id"]
    return record


def _undeclared_comparator_id(record: dict) -> dict:
    # "NA" is the sentinel an undeclared comparator leaves behind -- exactly what
    # `guards.comparator_declared` acts on.
    for row in record["effects"]:
        row["comparator_id"] = "NA"
        row["id"] = f"{row['family_id']}_vs_NA"
    return record


@pytest.mark.parametrize(
    "mutate", [_strip_comparator_id, _undeclared_comparator_id], ids=["absent", "NA_sentinel"]
)
def test_a_cross_condition_effect_with_no_named_baseline_is_refused(tmp_path, mutate):
    """The founding failure, in one field.

    `effect = -0.00074` against an unnamed population is the number that read as
    "replicates exactly" and as "4x stronger, prediction falsified" from the same
    data. `is_cross_condition` is true on every row here, so each of these rows
    is a claim about a difference; a difference from what is not optional, and
    the sentinel "NA" is not an answer to it.
    """
    record = mutate(doctored())
    assert all(row["is_cross_condition"] for row in record["effects"])
    project = _write_project(tmp_path, "no_baseline", record)
    with pytest.raises(_report_error()) as excinfo:
        render(project, tmp_path / "out")
    message = str(excinfo.value)
    assert "comparator_id" in message or "baseline" in message.lower(), (
        f"the refusal must say which baseline it wanted; got: {message}"
    )


def test_the_deprecated_health_alias_diverging_is_refused(tmp_path):
    """`health` and `query_health` disagreeing means one of them is not the run's.

    They are written from one object, so a divergence is not a stale copy to
    prefer around: it is evidence the record was edited after the fact, and
    rendering either one would be rendering a number whose provenance the file
    itself contradicts.
    """
    record = doctored()
    record["health"] = dict(record["health"], n_submitted=1)
    project = _write_project(tmp_path, "alias_divergence", record)
    with pytest.raises(_report_error()) as excinfo:
        render(project, tmp_path / "out")
    assert "health" in str(excinfo.value)


def test_html_and_docx_are_refused_and_markdown_is_what_is_written(tmp_path, capsys):
    """Rendering markdown for a caller who asked for HTML is the gap, in miniature.

    `cli._run_validate` refuses `--fimo-heldout` as "not yet an adapter input;
    refusing a semantic no-op" rather than accepting the flag and ignoring it,
    for the same reason: a flag that is accepted and does nothing puts what was
    specified and what ran into different documents. Exit 4 is the tool declining
    to produce a number, distinct from exit 3 (not implemented) and 2 (bad input).
    """
    project = _write_project(tmp_path, "formats")
    out = tmp_path / "out"

    for flag in ("--html", "--docx"):
        code = cli.main(["report", str(project), "--out", str(out / flag.strip("-"))])  # baseline
        assert code == 0, "the markdown path must work, or the refusal below proves nothing"
        code = cli.main(["report", str(project), "--out", str(out / flag.strip("-")), flag])
        assert code == 4, f"{flag} must be refused (exit 4), not rendered as markdown"
        stderr = capsys.readouterr().err
        assert "refused" in stderr and flag in stderr, stderr

    rendered = sorted(p.name for p in out.rglob("*") if p.is_file())
    assert any(name.endswith(".md") for name in rendered), rendered
    assert not [n for n in rendered if n.endswith((".html", ".htm", ".docx", ".pdf"))], (
        f"a refused format was rendered anyway: {rendered}"
    )


# --------------------------------------------------------------------------- #
# 2. Nothing is recomputed.
# --------------------------------------------------------------------------- #
def test_every_recorded_number_is_rendered_verbatim_beside_its_denominator(real_project, tmp_path):
    """`str()` of the recorded field, and the field's name beside it.

    Rendered as `str()` rather than formatted, because a renderer that chooses a
    precision is choosing which digits of a bootstrap interval a reader sees; and
    with the denominator NAMED rather than a ratio recomputed, because the ratio
    is already on the record and a second division is a second implementation.
    """
    doc = render(real_project, tmp_path / "out")
    record = REAL_INTERPRETATION

    for field in ("query_id", "substrate_id", "lexicon_id", "estimator", "statistical_license",
                  "claim_scope", "selection_provenance", "output_mode"):
        assert record[field] in doc, field
    assert str(record["input_scale"]) in doc
    # The recognised set, not just the literal that was used.
    for name in record["estimators_implemented"]:
        assert name in doc, name

    for block in ("query_health", "comparator_health"):
        health = record[block]
        for key in ("n_submitted", "n_in_universe", "n_searched", "n_with_used_hit",
                    "n_blocks", "block_size", "intersection_coverage", "explained_fraction"):
            assert key in doc, f"{block}.{key} is not named"
            assert str(health[key]) in doc, f"{block}.{key} = {health[key]}"
        for floor, value in health["floors"].items():
            assert floor in doc and str(value) in doc, floor
    assert str(record["comparator_health"]["n_submitted"]) in doc  # 25640, the baseline's size

    contrast = record["contrast_health"]
    for key in ("n_shared_peaks", "shared_blocks", "union_blocks"):
        assert key in doc, key
        assert str(contrast[key]) in doc, key
    # 283 query blocks against 282 shared blocks once hid 8,277 shared peaks: the
    # block counts must not be the only disjointness number on the page.
    assert _within(doc, "n_shared_peaks", str(contrast["n_shared_peaks"]), window=200)

    for row in record["composition"]:
        assert row["family_id"] in doc
        for key in ("n_peaks_with_family", "n_peaks_searched", "peak_share",
                    "mean_coefficient_per_peak"):
            assert str(row[key]) in doc, f"composition {row['family_id']}.{key}"

    for row in record["effects"]:
        for key in ("id", "family_id", "estimator", "comparator_id", "inference_capability"):
            assert row[key] in doc, f"effect {row['id']}.{key}"
        for key in ("effect", "n_query_peaks", "n_comparator_peaks", "n_blocks", "n_bootstrap",
                    "n_bootstrap_valid", "block_size", "random_seed"):
            assert str(row[key]) in doc, f"effect {row['id']}.{key} = {row[key]}"
        for bound in row["ci"]:
            assert str(bound) in doc, f"effect {row['id']} interval bound {bound}"

    record_prov = REAL_PROVENANCE[0]
    flat = _flat(doc)
    for key in ("subcommand", "timestamp_utc", "redaction_policy", "schema_version"):
        assert str(record_prov[key]) in doc, key
    assert _flat(record_prov["command"]) in flat, "the command is the one unredacted field"
    for role, checksum in record_prov["inputs"].items():
        assert role in doc, role
        assert checksum in doc, f"{role} checksum"
    for name, version in record_prov["software"].items():
        assert name in doc and version in doc
    assert str(record_prov["random_seed"]) in doc


def test_an_impossible_recorded_value_renders_as_recorded(tmp_path):
    """The test that separates rendering from recomputing.

    Each doctored value contradicts the counts beside it: a coverage of 0.42 with
    `n_in_universe == n_submitted == 8277`, a share of 42.0, an interval whose
    bounds are inverted. A renderer that derives numbers rather than reading them
    prints the arithmetically correct value and the contradiction never reaches
    the reader -- which is the failure mode, not the recovery.
    """
    record = doctored()
    # Doctored on every health block, so that the line-scoped check below holds
    # against a renderer that shows the comparator's coverage as well as the
    # query's -- the point is that no rendered coverage is a recomputed one.
    for block in (record["query_health"], record["comparator_health"],
                  record["contrast_health"]["query"], record["contrast_health"]["comparator"]):
        block["intersection_coverage"] = 0.4242424242
        block["explained_fraction"] = 7.5
    record["health"] = json.loads(json.dumps(record["query_health"]))
    record["composition"][0]["peak_share"] = 42.0
    record["effects"][0]["effect"] = 999.5
    record["effects"][0]["ci"] = [1.0, -1.0]

    doc = render(_write_project(tmp_path, "impossible", record), tmp_path / "out")

    assert "0.4242424242" in doc
    assert "7.5" in doc
    assert "42.0" in doc
    assert "999.5" in doc
    assert _within(doc, "intersection_coverage", "0.4242424242", window=200), (
        "the recorded coverage is not beside its own name"
    )
    # And the value a division would have produced is not offered in its place.
    for line in doc.splitlines():
        if "intersection_coverage" in line and "min_intersection_coverage" not in line:
            assert "0.4242424242" in line, line
    assert record["query_health"]["n_submitted"] == record["query_health"]["n_in_universe"] == 8277


# --------------------------------------------------------------------------- #
# 3. Absence is never a verified result.
# --------------------------------------------------------------------------- #
def test_emission_branches_on_none_and_never_on_floor_failures(tmp_path):
    """A comparator-side suppression leaves the composition intact. Render it.

    `interpret` writes exactly this record when the query clears its floors and
    the comparator does not: `composition` is a full twelve rows, `effects` is
    None, `floor_failures` is non-empty, and `interpretation_emitted` stays true.
    Branching on `floor_failures` instead of on the field that was actually
    emitted renders that run as though it had produced no numbers at all -- the
    defect this test exists to hold shut.
    """
    failures = ["comparator: explained_fraction 0.31 below floor 0.5"]
    record = doctored(
        effects=None,
        two_part_effects=None,
        emitted_order=["health", "composition"],
        floor_failures=failures,
        suppression_reason=(
            "effects suppressed: " + failures[0] + ". Composition above stands; no effect is "
            "reported because the comparator side does not clear its floors."
        ),
    )
    record["contrast_health"]["floor_failures"] = failures
    record["contrast_health"]["passed"] = False
    record["comparator_health"]["floor_failures"] = ["explained_fraction 0.31 below floor 0.5"]
    record["comparator_health"]["explained_fraction"] = 0.31

    doc = render(_write_project(tmp_path, "comparator_failed", record), tmp_path / "out")

    for row in REAL_INTERPRETATION["composition"]:
        assert row["family_id"] in doc, f"{row['family_id']} was dropped with the effects"
        assert str(row["peak_share"]) in doc
        assert str(row["n_peaks_searched"]) in doc
    assert _flat(failures[0]) in _flat(doc)
    assert _flat(record["suppression_reason"]) in _flat(doc)
    # ... and the effects that were not computed are not on the page as numbers.
    assert REAL_INTERPRETATION["effects"][0]["id"] not in doc
    assert str(REAL_INTERPRETATION["effects"][0]["effect"]) not in doc

    # The mirror image: effects recorded, composition not.
    other = doctored(composition=None, two_part_effects=None, emitted_order=["health", "effects"])
    doc2 = render(_write_project(tmp_path, "no_composition", other), tmp_path / "out2")
    for row in REAL_INTERPRETATION["effects"]:
        assert row["id"] in doc2, f"{row['id']} was dropped with the composition"
    assert "mean_coefficient_per_peak" not in doc2


def test_the_two_floor_failure_lists_are_both_shown(tmp_path):
    """They are allowed to diverge, so showing one is showing a number's opposite.

    `Interpretation.floor_failures` is the operative list (what suppressed
    something); `contrast_health.floor_failures` is the unconditional union of
    both sides. docs/DATA_MODEL.md keeps them apart deliberately: a comparator
    failure appears in the union while the top-level list is empty, and a reader
    shown only the empty one concludes both sides cleared.
    """
    union = ["comparator: intersection_coverage 0.42 below floor 0.9"]
    record = doctored(floor_failures=[])
    record["contrast_health"]["floor_failures"] = union

    doc = render(_write_project(tmp_path, "divergent_failures", record), tmp_path / "out")

    assert doc.count("floor_failures") >= 2, (
        "both lists must be rendered; one of them is the other's negation here"
    )
    assert _flat(union[0]) in _flat(doc)
    assert "contrast_health" in doc


def test_a_suppressed_reading_cannot_read_as_a_licensed_one(tmp_path):
    """Two documents from the same pipeline, and the difference must be unmissable."""
    failures = ["query: intersection_coverage 0.42 below floor 0.9"]
    suppressed_record = doctored(
        composition=None,
        effects=None,
        two_part_effects=None,
        emitted_order=["health"],
        interpretation_emitted=False,
        floor_failures=failures,
        suppression_reason=(
            "reading suppressed: " + failures[0] + ". The health numbers above stand; no "
            "composition and no effect is reported."
        ),
        notes=["DESCRIPTIVE_ONLY: descriptive decomposition only."],
        statistical_license="DESCRIPTIVE_ONLY",
    )
    suppressed_record["query_health"]["intersection_coverage"] = 0.42
    suppressed_record["query_health"]["floor_failures"] = [failures[0].removeprefix("query: ")]
    suppressed_record["health"] = json.loads(json.dumps(suppressed_record["query_health"]))

    licensed = render(_write_project(tmp_path, "licensed"), tmp_path / "out_ok")
    suppressed = render(
        _write_project(tmp_path, "suppressed", suppressed_record), tmp_path / "out_suppressed"
    )

    assert _flat(suppressed_record["suppression_reason"]) in _flat(suppressed)
    assert _flat(suppressed_record["suppression_reason"]) not in _flat(licensed)
    assert "reading suppressed" in _flat(suppressed)
    assert "DESCRIPTIVE_ONLY" in suppressed
    for row in REAL_INTERPRETATION["effects"]:
        assert row["id"] not in suppressed
        assert str(row["effect"]) not in suppressed
        assert row["id"] in licensed
    for row in REAL_INTERPRETATION["composition"]:
        assert str(row["mean_coefficient_per_peak"]) not in suppressed
    # The health numbers still stand -- suppression is not silence.
    assert "0.42" in suppressed and "n_submitted" in suppressed
    # And the licensed document says it was not suppressed rather than saying nothing.
    assert "suppression_reason" in licensed


def test_p_values_are_withheld_in_words_above_the_table_that_lacks_them(real_project, tmp_path):
    """A blank p column reads as "no evidence of an effect". It means neither.

    The recorded note is this run's discharge of the failure the module was
    written for -- a percentile bootstrap's resolution floor printed as a
    measured p -- so it goes above the table, where it is read before the
    numbers, not beneath it as a footnote. `cli._effect_estimate_rows` writes the
    literal `NA` into the TSV for the same reason.
    """
    doc = render(real_project, tmp_path / "out")
    note = REAL_INTERPRETATION["notes"][0]

    assert _flat(note) in _flat(doc), "the recorded note is not rendered verbatim"
    assert "WITHHELD" in doc
    assert doc.count("WITHHELD") >= len(REAL_INTERPRETATION["effects"]), (
        "every withheld p value must say so on its own row"
    )
    assert "ESTIMATION_ONLY" in doc
    assert "n.s." not in doc

    flat = _flat(doc)
    assert flat.index(_flat(note)) < flat.index(REAL_INTERPRETATION["effects"][0]["id"]), (
        "the note qualifying the estimator appears below the numbers it qualifies"
    )
    # The interval WAS computed and is not withheld with the test.
    assert str(REAL_INTERPRETATION["effects"][0]["ci"][0]) in doc


def test_a_null_two_part_effect_is_not_computed_and_found_nothing(real_project, tmp_path):
    """`two_part_effects is None` because nobody chose a definition of "used".

    Rendered as an absent input rather than an absent result: a reader who takes
    it for "measured, no difference" has been told something the run never said.
    """
    doc = render(real_project, tmp_path / "out")
    assert REAL_INTERPRETATION["two_part_effects"] is None
    flat = _flat(doc)
    assert "usage_definition" in flat
    assert "computed and found nothing" in flat
    assert "probability_effect" not in flat or "NOT COMPUTED" in flat


def test_section_order_follows_the_recorded_emitted_order(tmp_path):
    """The order is a fact about the run, so the renderer reads it rather than holding it."""
    composition_only = "mean_coefficient_per_peak"
    effects_only = "n_bootstrap_valid"

    doc = render(_write_project(tmp_path, "recorded_order"), tmp_path / "out")
    assert REAL_INTERPRETATION["emitted_order"] == ["health", "composition", "effects"]
    assert doc.index(composition_only) < doc.index(effects_only)

    swapped = doctored(emitted_order=["health", "effects", "composition"])
    doc2 = render(_write_project(tmp_path, "swapped_order", swapped), tmp_path / "out2")
    assert doc2.index(effects_only) < doc2.index(composition_only), (
        "the renderer holds its own section order instead of reading emitted_order"
    )
    # Health is first in both, because it gates everything after it.
    assert doc2.index("intersection_coverage") < doc2.index(effects_only)


def test_what_this_report_does_not_know_names_the_fields_that_would_have_said_it(
    real_project, tmp_path
):
    """The mandatory section, and the one place a missing field may be printed.

    `baseline_population` is on no artifact this package emits -- it is an axis of
    `config/specifications.example.yaml`, which no code reads. So it renders as
    the literal `NOT RECORDED`, never as `comparator_id` quietly standing in for
    it: `comparator_id` names which FILE the baseline came from, and
    `baseline_population` names what KIND of population it is, and the distance
    between those two is the distance between "replicates exactly" and
    "prediction falsified".
    """
    doc = render(real_project, tmp_path / "out")
    flat = _flat(doc)

    assert "NOT RECORDED" in doc
    assert _within(doc, "baseline_population", "NOT RECORDED", window=300)
    # What plays the role is rendered as itself, under its own name.
    assert "comparator_id" in doc and "not_cl5" in doc
    assert str(REAL_INTERPRETATION["effects"][0]["n_comparator_peaks"]) in doc

    # The lexicon is cited as a declared string, not as a content hash.
    assert REAL_INTERPRETATION["lexicon_id"] in doc
    assert "lexicon_content_hash" in flat
    # The rule PROGRAMMATIC_RULE requires, and the features SUBSTRATE_CIRCULAR
    # turns on, are fields of schema.PeakSetQuery and are not on this record.
    assert "selection_rule" in flat
    assert "selection_feature_names" in flat
    assert "SUBSTRATE_CIRCULAR" in flat
    # Both permission axes, always, and the deprecated one labelled.
    assert "statistical_license" in flat and "claim_scope" in flat
    assert _within(doc, "output_mode", "DEPRECATED", window=300)
    # The provenance record names three input roles and not a manifest.
    assert "substrate_manifest" in flat
    assert "verify_against_manifest" in flat
    # Nothing here may claim a guard ran.
    assert "GuardResult" in flat


# --------------------------------------------------------------------------- #
# 4. The bias ledger, and the guards.
# --------------------------------------------------------------------------- #
def test_the_bias_ledger_renders_from_the_tsv(real_project, tmp_path):
    """From `docs/bias_ledger.tsv`, which the rule names, verbatim.

    docs/BIAS_LEDGER.md carries an English gloss and an "enforced here" column
    that are this repository's annotation of the ledger rather than the ledger;
    where they differ the TSV is authoritative, so the TSV is what is read.
    """
    rows = _ledger_rows()
    assert rows[0] == ["axis_id", "bias", "mechanism", "control"]
    data = rows[1:]
    assert [row[0] for row in data] == [f"BA-{n:02d}" for n in range(1, 21)]
    assert all(len(row) == 4 for row in data)

    flat = _flat(render(real_project, tmp_path / "out"))
    for axis_id, bias, mechanism, control in data:
        assert axis_id in flat
        for cell in (bias, mechanism, control):
            assert _flat(cell) in flat, f"{axis_id}: {cell}"


def test_a_malformed_bias_ledger_is_refused(tmp_path, monkeypatch, real_project):
    """A ledger of the wrong shape is not a ledger, and a report without one is not this report."""
    sig = inspect.signature(report.run)
    param = next((name for name in _LEDGER_PARAM_NAMES if name in sig.parameters), None)
    attr = None
    if param is None:
        attr = next(
            (
                name for name, value in vars(report).items()
                if isinstance(value, (str, Path)) and str(value).endswith("bias_ledger.tsv")
            ),
            None,
        )
    if param is None and attr is None:
        pytest.fail(
            "the bias ledger path is neither a parameter of report.run nor a module attribute, "
            "so the refusal the rule specifies (absent, wrong column count, axis_id not "
            f"BA-01..BA-20) cannot be exercised. report.run{sig}"
        )

    def _render_with(path: Path) -> None:
        if param is not None:
            render(real_project, tmp_path / f"out_{path.name}", **{param: str(path)})
        else:
            monkeypatch.setattr(report, attr, str(path))
            render(real_project, tmp_path / f"out_{path.name}")

    good = _ledger_rows()

    absent = tmp_path / "absent.tsv"
    five_columns = tmp_path / "five.tsv"
    five_columns.write_text(
        "\n".join("\t".join([*row, "enforced here"]) for row in good) + "\n", encoding="utf-8"
    )
    renumbered = tmp_path / "renumbered.tsv"
    renumbered.write_text(
        "\n".join(
            ["\t".join(good[0])] + ["\t".join(["BA-99", *row[1:]]) for row in good[1:]]
        ) + "\n",
        encoding="utf-8",
    )

    for path in (absent, five_columns, renumbered):
        with pytest.raises(_report_error()):
            _render_with(path)


def test_a_guard_absent_from_guards_awaiting_input_is_not_thereby_wired(real_project, tmp_path):
    """The inversion that sank the previous attempt, held shut.

    `GUARDS_AWAITING_INPUT` records why six guards have no call site. It is not a
    complement: the nine guards it does not mention are not thereby known to be
    called, and no artifact in this package persists a `guards.GuardResult`, so
    nothing the report can read says any guard passed on this run. Concluding
    otherwise from an absence is the shape of fabrication the whole module exists
    to make impossible.
    """
    doc = render(real_project, tmp_path / "out")
    flat = _flat(doc)

    awaiting = guards.GUARDS_AWAITING_INPUT
    assert set(awaiting) == {
        "four_state_missingness", "no_cross_model_cwm_avg", "interaction_required",
        "estimability_floor", "stratum_parity", "single_family_layer",
    }
    for guard_id, pending in awaiting.items():
        assert guard_id in flat, guard_id
        for field in ("nearest_artifact", "why_not_a_call_site", "closes_when"):
            assert _flat(getattr(pending, field)) in flat, f"{guard_id}.{field}"

    unlisted = [gid for gid in guards.ALL_GUARDS if gid not in awaiting]
    assert unlisted, "the fixture assumption that some guards are unlisted no longer holds"
    for guard_id in unlisted:
        assert guard_id in flat, (
            f"{guard_id} is not in GUARDS_AWAITING_INPUT; the report must name it as a guard "
            "whose call site it cannot see, not omit it"
        )
    # Nothing on the page may put a guard id near a passing outcome, in either
    # order: no recorded text does (no entry of GUARDS_AWAITING_INPUT contains the
    # word "passed" or names another guard), and no artifact records one, so any
    # such sentence was written by the renderer out of an absence.
    for guard_id in guards.ALL_GUARDS:
        for pattern in (
            rf"{re.escape(guard_id)}.{{0,80}}?\bpassed\b",
            rf"\bpassed\b.{{0,80}}?{re.escape(guard_id)}",
        ):
            found = re.search(pattern, flat)
            assert found is None, (
                f"the report puts `{guard_id}` beside a passing outcome -- {found.group(0)!r} -- "
                "and no artifact in this package records a GuardResult"
            )
    assert not re.search(r"\b(all|every|the)\s+guards?\b.{0,40}?\bpass(ed|es)?\b", flat)
