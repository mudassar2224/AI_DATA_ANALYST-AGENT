"""Shared pytest fixtures.

messy_df plants known issues at known locations so tests assert against
ground truth instead of just "it ran without crashing": one missing
value, one exact duplicate row, one disguised-missing token, and one
numeric outlier (planted large enough that it survives IQR quartile
shift with a small sample).
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def messy_df() -> pd.DataFrame:
    data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 8],
        "age": [25, 30, None, 40, 29, 33, 27, 31, 31],
        "city": [
            "Lahore",
            "Karachi",
            "Unknown",
            "Lahore",
            "Islamabad",
            "Lahore",
            "Karachi",
            "Lahore",
            "Lahore",
        ],
        "income": [50000, 52000, 48000, 51000, 49500, 53000, 47000, 999000, 999000],
    }
    return pd.DataFrame(data)
