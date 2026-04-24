"""
Unit tests for the data validation logic.
Tests schema checks, missing value audits, target distribution,
and value range validations.
All tests use in-memory synthetic DataFrames (no Kaggle data required).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def make_tx_df(
    n: int = 500,
    fraud_rate: float = 0.035,
    seed: int = 42,
    include_v: bool = True,
) -> pd.DataFrame:
    """Create a minimal transaction DataFrame matching IEEE CIS schema."""
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_rate))
    n_legit = n - n_fraud
    labels = np.array([0] * n_legit + [1] * n_fraud)
    rng.shuffle(labels)

    df = pd.DataFrame(
        {
            "TransactionID": np.arange(1000, 1000 + n),
            "isFraud": labels,
            "TransactionDT": rng.integers(100, 15_000_000, size=n),
            "TransactionAmt": rng.lognormal(4.0, 1.5, size=n).round(2),
            "ProductCD": rng.choice(["W", "H", "C"], size=n),
        }
    )
    if include_v:
        for i in range(1, 6):
            vals = rng.standard_normal(n).astype(float)
            vals[rng.random(n) < 0.3] = np.nan
            df[f"V{i}"] = vals
    return df


# ------------------------------------------------------------------ #
# Validation logic (mirrors component logic, extracted for unit tests) #
# ------------------------------------------------------------------ #


def run_validation(tx_df: pd.DataFrame, id_df: pd.DataFrame = None) -> dict:
    """Run the same validation checks as data_validation component."""
    if id_df is None:
        if "TransactionID" in tx_df.columns:
            id_df = pd.DataFrame({"TransactionID": tx_df["TransactionID"].values[:10]})
        else:
            id_df = pd.DataFrame({"TransactionID": []})

    REQUIRED = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
    issues = []
    warnings_list = []

    # Check 1: schema
    missing = [c for c in REQUIRED if c not in tx_df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")

    # Check 2: binary labels
    if "isFraud" in tx_df.columns:
        unique = set(tx_df["isFraud"].dropna().unique())
        if not unique.issubset({0, 1}):
            issues.append(f"Non-binary isFraud values: {unique}")

    # Check 3: fraud rate
    if "isFraud" in tx_df.columns:
        rate = tx_df["isFraud"].mean()
        if rate < 0.001 or rate > 0.20:
            issues.append(f"Fraud rate {rate:.4f} out of range")

    # Check 4: TransactionAmt > 0
    if "TransactionAmt" in tx_df.columns:
        n_bad = int((tx_df["TransactionAmt"] <= 0).sum())
        if n_bad > 0:
            warnings_list.append(f"{n_bad} negative/zero TransactionAmt")

    # Check 5: TransactionDT > 0
    if "TransactionDT" in tx_df.columns:
        n_bad = int((tx_df["TransactionDT"] <= 0).sum())
        if n_bad > 0:
            warnings_list.append(f"{n_bad} non-positive TransactionDT")

    # Check 6: high missing columns
    missing_pct = tx_df.isnull().mean()
    high = missing_pct[missing_pct > 0.95]
    if not high.empty:
        warnings_list.append(f"{len(high)} cols >95% missing")

    # Check 7: duplicates
    if "TransactionID" in tx_df.columns:
        n_dup = int(tx_df["TransactionID"].duplicated().sum())
        if n_dup > 0:
            issues.append(f"{n_dup} duplicate TransactionIDs")

    return {
        "issues": issues,
        "warnings": warnings_list,
        "passed": len(issues) == 0,
    }


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #


class TestSchemaValidation:
    def test_valid_schema_passes(self):
        df = make_tx_df()
        result = run_validation(df)
        assert result["passed"], f"Expected pass but got issues: {result['issues']}"

    def test_missing_required_column_fails(self):
        df = make_tx_df().drop(columns=["isFraud"])
        result = run_validation(df)
        assert not result["passed"]
        assert any("isFraud" in iss for iss in result["issues"])

    def test_missing_transaction_id_fails(self):
        df = make_tx_df().drop(columns=["TransactionID"])
        result = run_validation(df)
        assert not result["passed"]

    def test_missing_transaction_amt_fails(self):
        df = make_tx_df().drop(columns=["TransactionAmt"])
        result = run_validation(df)
        assert not result["passed"]


class TestTargetDistribution:
    def test_normal_fraud_rate_passes(self):
        df = make_tx_df(fraud_rate=0.035)
        result = run_validation(df)
        assert result["passed"]

    def test_zero_fraud_fails(self):
        df = make_tx_df()
        df["isFraud"] = 0  # all legitimate
        result = run_validation(df)
        assert not result["passed"]
        assert any("Fraud rate" in iss for iss in result["issues"])

    def test_excessive_fraud_rate_fails(self):
        df = make_tx_df()
        df["isFraud"] = 1  # all fraud
        result = run_validation(df)
        assert not result["passed"]

    def test_non_binary_labels_fail(self):
        df = make_tx_df()
        df.loc[0, "isFraud"] = 2  # invalid label
        result = run_validation(df)
        assert not result["passed"]


class TestMissingValues:
    def test_normal_missing_passes(self):
        df = make_tx_df(include_v=True)
        result = run_validation(df)
        # V columns have 30% missing – that should just be a warning, not a failure
        assert result["passed"]

    def test_column_over_95pct_missing_is_warning(self):
        df = make_tx_df()
        df["almost_empty"] = np.nan  # 100% missing
        result = run_validation(df)
        # Should raise a warning but NOT a hard failure
        assert result["passed"]
        assert len(result["warnings"]) > 0

    def test_all_nan_column_flagged(self):
        df = make_tx_df()
        df["all_nan"] = np.nan
        result = run_validation(df)
        assert any("missing" in w for w in result["warnings"])


class TestValueRanges:
    def test_positive_transaction_amt_passes(self):
        df = make_tx_df()
        df["TransactionAmt"] = abs(df["TransactionAmt"]) + 0.01
        result = run_validation(df)
        assert len([w for w in result["warnings"] if "TransactionAmt" in w]) == 0

    def test_negative_transaction_amt_warns(self):
        df = make_tx_df()
        df.loc[0, "TransactionAmt"] = -100.0
        result = run_validation(df)
        # Should warn, may or may not fail depending on implementation
        assert any("TransactionAmt" in w for w in result["warnings"])

    def test_positive_transaction_dt_passes(self):
        df = make_tx_df()
        assert (df["TransactionDT"] > 0).all()
        result = run_validation(df)
        assert result["passed"]


class TestDuplicates:
    def test_no_duplicates_passes(self):
        df = make_tx_df()
        result = run_validation(df)
        assert result["passed"]

    def test_duplicate_ids_fail(self):
        df = make_tx_df()
        df.loc[1, "TransactionID"] = df.loc[0, "TransactionID"]  # create duplicate
        result = run_validation(df)
        assert not result["passed"]
        assert any("duplicate" in iss.lower() for iss in result["issues"])


class TestRowCount:
    def test_sufficient_rows_passes(self):
        df = make_tx_df(n=500)
        result = run_validation(df)
        assert result["passed"]

    def test_empty_dataframe_flags_issue(self):
        df = make_tx_df(n=1)
        df["isFraud"] = 0  # will fail fraud rate check
        result = run_validation(df)
        # At minimum fraud rate check should fail
        assert not result["passed"]
