"""The outcome record an output directory carries (`motifmultiverse.run_status`).

The subject is a claim about *whose* the files in a directory are, so most of
these assert what the record refuses to say: that it never reports a refusal as a
success, never invents a predecessor it did not read, and never quietly loses the
last successful run across a string of failures.
"""
from __future__ import annotations

import json

import pytest

from motifmultiverse import run_status as rs


def _write(out, status, command="motifmultiverse interpret hits.tsv", **kwargs):
    return rs.write_run_status(out, status=status, command=command,
                               subcommand="interpret", **kwargs)


def test_a_success_records_itself_as_the_run_the_artifacts_came_from(tmp_path):
    _write(tmp_path, "SUCCESS")
    document = json.loads((tmp_path / rs.RUN_STATUS_FILENAME).read_text())
    assert document["status"] == "SUCCESS" and document["exit_code"] == 0
    assert document["artifacts_are_from"]["command"] == document["command"]
    assert document["artifacts_are_from"]["finished_utc"] == document["finished_utc"]


def test_a_refusal_after_a_success_still_names_the_success(tmp_path):
    """The whole point: the files are the earlier run's, and the record says so."""
    _write(tmp_path, "SUCCESS", command="motifmultiverse interpret hits.tsv --out o/")
    _write(tmp_path, "REFUSED", command="motifmultiverse interpret hits.tsv --substrate-manifest m",
           detail="the manifest does not describe this substrate")
    document = json.loads((tmp_path / rs.RUN_STATUS_FILENAME).read_text())
    assert document["status"] == "REFUSED" and document["exit_code"] == 4
    assert document["detail"] == "the manifest does not describe this substrate"
    assert document["artifacts_are_from"]["command"].endswith("--out o/")


def test_the_last_success_survives_a_run_of_failures(tmp_path):
    """Carry-forward is transitive, or it stops answering after the second failure."""
    _write(tmp_path, "SUCCESS", command="the run that wrote these files")
    for status in ("REFUSED", "INPUT_MISSING", "CRASHED", "UNIMPLEMENTED", "REFUSED"):
        _write(tmp_path, status, command=f"a {status} run")
    document = json.loads((tmp_path / rs.RUN_STATUS_FILENAME).read_text())
    assert document["artifacts_are_from"]["command"] == "the run that wrote these files"


def test_a_failure_with_no_predecessor_says_no_run_succeeded_rather_than_null(tmp_path):
    _write(tmp_path, "REFUSED", detail="nothing here ever worked")
    document = json.loads((tmp_path / rs.RUN_STATUS_FILENAME).read_text())
    assert document["artifacts_are_from"] == rs.NO_SUCCESSFUL_RUN


def test_an_unreadable_predecessor_is_named_as_a_loss_not_as_an_absence(tmp_path):
    """Two different facts: nothing succeeded here, and this file cannot tell.

    Reporting the second as the first is the shape of error this package is
    organised against -- an absence of evidence rendered as evidence of absence.
    """
    (tmp_path / rs.RUN_STATUS_FILENAME).write_text("{not json", encoding="utf-8")
    _write(tmp_path, "REFUSED")
    document = json.loads((tmp_path / rs.RUN_STATUS_FILENAME).read_text())
    assert document["artifacts_are_from"] == "PREVIOUS_RUN_STATUS_UNREADABLE"
    assert document["status"] == "REFUSED", "a corrupt predecessor cost the run its own record"


def test_provenance_records_counts_the_log_beside_it(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps([{"subcommand": "interpret"}] * 3))
    _write(tmp_path, "SUCCESS")
    assert json.loads((tmp_path / rs.RUN_STATUS_FILENAME).read_text())["provenance_records"] == 3


def test_an_absent_or_unreadable_provenance_log_is_not_reported_as_zero_records(tmp_path):
    """Zero records nobody counted is not the claim "zero records"."""
    _write(tmp_path, "SUCCESS")
    assert json.loads((tmp_path / rs.RUN_STATUS_FILENAME).read_text())[
        "provenance_records"] == "NOT_RECORDED"
    (tmp_path / "provenance.json").write_text("[[[", encoding="utf-8")
    _write(tmp_path, "SUCCESS")
    assert json.loads((tmp_path / rs.RUN_STATUS_FILENAME).read_text())[
        "provenance_records"] == "NOT_RECORDED"


def test_an_unknown_status_is_refused_rather_than_written(tmp_path):
    """The vocabulary is closed, so a reader can branch on it exhaustively."""
    with pytest.raises(ValueError, match="unknown run status"):
        _write(tmp_path, "PROBABLY_FINE")
    assert not (tmp_path / rs.RUN_STATUS_FILENAME).exists()


def test_every_status_carries_the_exit_code_the_cli_returns_for_it(tmp_path):
    """The document must not contradict the process: `status` and `exit_code` are
    two spellings of one outcome, and a reader may use either."""
    assert rs.STATUSES == {"SUCCESS": 0, "INPUT_MISSING": 2, "UNIMPLEMENTED": 3,
                           "REFUSED": 4, "CRASHED": 1}
    for status, code in rs.STATUSES.items():
        _write(tmp_path, status)
        assert json.loads((tmp_path / rs.RUN_STATUS_FILENAME).read_text())["exit_code"] == code


def test_no_partial_file_is_left_behind(tmp_path):
    """Written through a staged sibling and `os.replace`, as the provenance log is:
    a directory whose problem was that nothing said anything must not acquire a
    status file that says half of something."""
    _write(tmp_path, "SUCCESS")
    assert sorted(p.name for p in tmp_path.iterdir()) == [rs.RUN_STATUS_FILENAME]
