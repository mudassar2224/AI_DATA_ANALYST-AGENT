"""Load a user-uploaded file into a clean pandas DataFrame.

This is the ingestion specialist: pure deterministic parsing, no LLM
involvement. It handles CSV encoding detection and multi-sheet Excel
files, and samples down datasets that are too large to analyze in full
in this demo environment.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd
from charset_normalizer import from_bytes


class UnsupportedFileError(ValueError):
    """Raised when the uploaded file type or content can't be parsed."""


@dataclass
class LoadResult:
    """The loaded data plus a short record of how it was interpreted."""

    dataframe: pd.DataFrame
    source_filename: str
    sheet_name: str | None = None
    detected_encoding: str | None = None
    available_sheets: list[str] = field(default_factory=list)
    was_sampled: bool = False


def _detect_encoding(raw: bytes) -> str:
    result = from_bytes(raw).best()
    return result.encoding if result else "utf-8"


def load_file(
    file_bytes: bytes,
    filename: str,
    sheet_name: str | int | None = 0,
    max_rows: int | None = None,
) -> LoadResult:
    """Parse CSV or Excel bytes into a DataFrame.

    Raises UnsupportedFileError for anything else, or a file so malformed
    pandas can't make sense of it.
    """
    lower = filename.lower()

    if lower.endswith((".xlsx", ".xls")):
        try:
            excel = pd.ExcelFile(io.BytesIO(file_bytes))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as-is
            raise UnsupportedFileError(f"Could not open '{filename}' as Excel: {exc}") from exc
        chosen = sheet_name if isinstance(sheet_name, str) else excel.sheet_names[0]
        df = excel.parse(chosen)
        result = LoadResult(
            dataframe=df,
            source_filename=filename,
            sheet_name=chosen,
            available_sheets=excel.sheet_names,
        )
    elif lower.endswith(".csv"):
        encoding = _detect_encoding(file_bytes)
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise UnsupportedFileError(
                f"Could not parse '{filename}' as CSV (tried encoding '{encoding}')."
            ) from exc
        result = LoadResult(dataframe=df, source_filename=filename, detected_encoding=encoding)
    else:
        raise UnsupportedFileError(
            f"Unsupported file type for '{filename}'. Upload a .csv, .xlsx, or .xls file."
        )

    if result.dataframe.empty:
        raise UnsupportedFileError(f"'{filename}' loaded with zero rows.")

    if max_rows is not None and len(result.dataframe) > max_rows:
        result.dataframe = result.dataframe.sample(n=max_rows, random_state=42).reset_index(
            drop=True
        )
        result.was_sampled = True

    return result
