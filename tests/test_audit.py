import pytest
from dns_exfiltration.audit import audit_csv


def test_streaming_audit_counts_invalid_missing_and_infinite(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("feature,label\n1,benign\n,attack\nInfinity,attack\nword,benign\n")
    report = audit_csv(path, "label", chunk_rows=2)
    assert report["rows"] == 4
    assert report["raw_label_counts"] == {"benign": 2, "attack": 2}
    assert report["missing"]["feature"] == 1
    assert report["non_numeric_values"]["feature"] == 1
    assert report["infinite_values"]["feature"] == 1
    assert len(report["sha256"]) == 64


def test_duplicate_header_is_rejected_before_pandas_renames_it(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x,x,label\n1,2,benign\n")
    with pytest.raises(ValueError, match="duplicate"):
        audit_csv(path, "label")
