

def test_a_directory_where_a_file_was_expected_is_a_named_refusal(tmp_path):
    """Found by running the quickstart, which is why the quickstart exists.

    Almost every `--flag` in this CLI takes a directory and `compile --decisions`
    takes a file, so passing the directory is the ordinary mistake. It used to
    raise `IsADirectoryError` from inside provenance -- an unhandled traceback at
    exit 1, in a package whose contract is that a refusal is named and exits 4.
    """
    import pytest

    from motifmultiverse.provenance import ProvenanceError, sha256_file

    with pytest.raises(ProvenanceError, match="is a directory"):
        sha256_file(tmp_path)

    file = tmp_path / "real.json"
    file.write_text("{}", encoding="utf-8")
    assert len(sha256_file(file)) == 64
