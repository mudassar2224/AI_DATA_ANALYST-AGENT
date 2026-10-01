"""Generate data/sample/customers_sample.csv — a small, reproducible demo
dataset with planted issues so the "try a sample dataset" button in the
app always has something interesting to show: missing values, a few
duplicate rows, a disguised-missing token, and income outliers.

Run with: uv run python scripts/generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).parent.parent / "data" / "sample" / "customers_sample.csv"


def build() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 260

    cities = rng.choice(
        ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan"],
        size=n,
        p=[0.35, 0.30, 0.15, 0.12, 0.08],
    )
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.5, 0.3, 0.2])
    tenure_months = rng.integers(1, 72, size=n)
    age = rng.integers(18, 70, size=n)

    # monthly_charge carries a real, contract-dependent discount (a common,
    # realistic telecom pattern) so the categorical-numeric relationship
    # test has a genuine signal to find, not just noise
    contract_discount = np.select(
        [contract == "Two year", contract == "One year"], [-18.0, -8.0], default=0.0
    )
    monthly_charge = np.round((rng.normal(70, 14, size=n) + contract_discount).clip(15, 150), 2)

    # churn depends on short tenure and month-to-month contracts, with a
    # large enough affected group (~90 of 260 rows) to survive FDR
    # correction — the earlier version of this generator planted the same
    # idea in too few rows to be statistically detectable, which the
    # relationship engine correctly refused to report
    short_tenure = tenure_months < 18
    month_to_month = contract == "Month-to-month"
    churn_prob = 0.05 + 0.45 * short_tenure + 0.25 * month_to_month
    churn = rng.random(n) < churn_prob.clip(0, 0.95)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{1000 + i}" for i in range(n)],
            "age": age,
            "city": cities,
            "contract_type": contract,
            "tenure_months": tenure_months,
            "monthly_charge": monthly_charge,
            "signup_date": pd.date_range("2021-01-01", periods=n, freq="7D").strftime("%Y-%m-%d"),
            "churn": churn,
        }
    )

    # plant quality issues
    df.loc[rng.choice(n, size=15, replace=False), "age"] = None
    df.loc[rng.choice(n, size=10, replace=False), "monthly_charge"] = None
    df.loc[rng.choice(n, size=12, replace=False), "city"] = "Unknown"
    df.loc[[3, 4], "monthly_charge"] = 480.0  # outliers
    df = pd.concat([df, df.iloc[[10, 55]]], ignore_index=True)  # exact duplicate rows

    return df.sample(frac=1, random_state=1).reset_index(drop=True)


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    build().to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
