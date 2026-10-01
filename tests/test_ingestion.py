"""Tests for dsagent.ingestion.loader."""

from __future__ import annotations

import pytest

from dsagent.ingestion.loader import UnsupportedFileError, load_file


def test_loads_valid_csv():
    csv_bytes = b"a,b\n1,2\n3,4\n"
    result = load_file(csv_bytes, "data.csv")
    assert list(result.dataframe.columns) == ["a", "b"]
    assert len(result.dataframe) == 2
    assert result.detected_encoding is not None


def test_rejects_unsupported_extension():
    with pytest.raises(UnsupportedFileError):
        load_file(b"hello", "notes.txt")


def test_rejects_empty_csv():
    with pytest.raises(UnsupportedFileError):
        load_file(b"a,b\n", "empty.csv")


def test_samples_large_file():
    rows = "\n".join(f"{i},{i * 2}" for i in range(100))
    csv_bytes = f"a,b\n{rows}\n".encode()
    result = load_file(csv_bytes, "big.csv", max_rows=10)
    assert len(result.dataframe) == 10
    assert result.was_sampled is True


def test_excel_reports_available_sheets(tmp_path):
    import pandas as pd

    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"x": [1, 2]}).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame({"y": [3, 4]}).to_excel(writer, sheet_name="Sheet2", index=False)

    result = load_file(path.read_bytes(), "book.xlsx")
    assert result.available_sheets == ["Sheet1", "Sheet2"]
    assert result.sheet_name == "Sheet1"
    assert list(result.dataframe.columns) == ["x"]
