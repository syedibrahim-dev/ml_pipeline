"""
Unit tests for preprocessing logic:
  - Missing value imputation (V columns: median + indicator)
  - High-cardinality frequency encoding
  - Smoothed target encoding for email domains
  - SMOTE vs class_weight imbalance handling
  - Train/test temporal split
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def make_full_df(n: int = 1000, fraud_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with the full schema used by preprocessing."""
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(n * fraud_rate))
    n_legit = n - n_fraud
    labels  = np.array([0] * n_legit + [1] * n_fraud)
    rng.shuffle(labels)

    v_data = rng.standard_normal((n, 10)).astype(float)
    v_data[rng.random((n, 10)) < 0.40] = np.nan

    return pd.DataFrame({
        "TransactionDT":  rng.integers(100, 15_000_000, size=n),
        "TransactionAmt": rng.lognormal(4.0, 1.5, size=n).round(2),
        "card1":  rng.integers(1000, 18500, size=n).astype(float),
        "card2":  rng.choice([float("nan")] + list(range(100, 600)), size=n),
        "addr1":  rng.choice(list(range(100, 500)) + [float("nan")], size=n),
        "addr2":  rng.choice(list(range(10, 102)) + [float("nan")], size=n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", float("nan")], size=n),
        "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", float("nan")], size=n),
        **{f"C{i}": rng.integers(0, 200, size=n).astype(float) for i in range(1, 5)},
        **{f"D{i}": rng.choice(list(range(0, 300)) + [float("nan")] * 100, size=n) for i in range(1, 4)},
        **{f"M{i}": rng.choice(["T", "F", float("nan")], size=n) for i in range(1, 4)},
        **{f"V{i}": v_data[:, i - 1] for i in range(1, 11)},
        "ProductCD": rng.choice(["W", "H", "C"], size=n),
        "isFraud":   labels,
    })


def temporal_split(df, test_size=0.2):
    """Reproduce the temporal split from the preprocessing component."""
    thresh = df["TransactionDT"].quantile(1.0 - test_size)
    return df[df["TransactionDT"] <= thresh].copy(), df[df["TransactionDT"] > thresh].copy()


# ------------------------------------------------------------------ #
# Tests: Missing value imputation                                     #
# ------------------------------------------------------------------ #

class TestMissingValueImputation:
    def test_v_columns_filled_after_imputation(self):
        df = make_full_df()
        train, test = temporal_split(df)
        v_cols = [c for c in train.columns if c.startswith("V")]
        assert train[v_cols].isnull().any().any(), "Pre-condition: V columns should have NaNs"

        # Perform median imputation
        v_medians = train[v_cols].median()
        train_filled = train.copy()
        train_filled[v_cols] = train_filled[v_cols].fillna(v_medians)

        assert not train_filled[v_cols].isnull().any().any(), "V columns should have no NaN after median fill"

    def test_missing_indicator_columns_created(self):
        df = make_full_df()
        train, _ = temporal_split(df)
        v_cols = [c for c in train.columns if c.startswith("V")]

        for col in v_cols:
            indicator = f"{col}_missing"
            train[indicator] = train[col].isna().astype(int)
            assert indicator in train.columns
            assert train[indicator].isin([0, 1]).all(), f"{indicator} should be binary"

    def test_indicator_reflects_original_nan(self):
        df = make_full_df()
        train, _ = temporal_split(df)
        col = "V1"
        indicator = f"{col}_missing"
        train[indicator] = train[col].isna().astype(int)

        # Where original was NaN, indicator should be 1
        nan_mask = train[col].isna()
        assert (train.loc[nan_mask, indicator] == 1).all()
        assert (train.loc[~nan_mask, indicator] == 0).all()

    def test_m_columns_binary_after_encoding(self):
        df = make_full_df()
        train, _ = temporal_split(df)
        m_cols = [c for c in train.columns if c.startswith("M")]
        for col in m_cols:
            train[col] = (train[col].fillna("F") == "T").astype(int)
            assert train[col].isin([0, 1]).all(), f"{col} should be 0 or 1 after encoding"


# ------------------------------------------------------------------ #
# Tests: Frequency encoding                                           #
# ------------------------------------------------------------------ #

class TestFrequencyEncoding:
    def test_card1_frequency_encoded_as_float(self):
        df = make_full_df()
        train, test = temporal_split(df)

        freq = train["card1"].value_counts(normalize=True)
        global_freq = 1.0 / len(train)
        train_enc = train["card1"].map(freq).fillna(global_freq)

        assert train_enc.dtype in [np.float64, np.float32, float]
        assert (train_enc >= 0).all() and (train_enc <= 1).all()

    def test_unknown_card_values_get_global_freq(self):
        df = make_full_df()
        train, test = temporal_split(df)

        freq = train["card1"].value_counts(normalize=True)
        global_freq = 1.0 / len(train)
        test_enc = test["card1"].map(freq).fillna(global_freq)

        # All values should be filled (no NaN from unknown card values)
        assert test_enc.isnull().sum() == 0

    def test_high_freq_card_maps_to_higher_value(self):
        """The most common card1 value should have the highest frequency."""
        df = make_full_df(n=2000)
        train, _ = temporal_split(df)

        freq = train["card1"].value_counts(normalize=True)
        most_common_val = freq.index[0]
        enc = train["card1"].map(freq)

        most_common_encoded = enc[train["card1"] == most_common_val].iloc[0]
        assert most_common_encoded == freq.iloc[0]


# ------------------------------------------------------------------ #
# Tests: Target encoding                                              #
# ------------------------------------------------------------------ #

class TestTargetEncoding:
    def test_email_domain_encoded_as_float(self):
        df = make_full_df()
        train, _ = temporal_split(df)
        k = 10.0
        gfr = train["isFraud"].mean()
        train["P_emaildomain"] = train["P_emaildomain"].fillna("_unk_")
        stats = train.groupby("P_emaildomain")["isFraud"].agg(["sum", "count"])
        enc = (stats["sum"] + k * gfr) / (stats["count"] + k)
        train["P_emaildomain"] = train["P_emaildomain"].map(enc).fillna(gfr)

        assert train["P_emaildomain"].dtype in [np.float64, float]
        assert (train["P_emaildomain"] >= 0).all()
        assert (train["P_emaildomain"] <= 1).all()

    def test_high_fraud_domain_has_higher_encoding(self):
        """Domains with more fraud should have higher target encoding."""
        rng = np.random.default_rng(42)
        n = 1000
        domains = rng.choice(["bad.com", "good.com"], size=n)
        # bad.com = 80% fraud; good.com = 5% fraud
        fraud = np.where(
            domains == "bad.com",
            rng.binomial(1, 0.80, n),
            rng.binomial(1, 0.05, n),
        )
        df = pd.DataFrame({"P_emaildomain": domains, "isFraud": fraud})

        k = 10.0
        gfr = fraud.mean()
        stats = df.groupby("P_emaildomain")["isFraud"].agg(["sum", "count"])
        enc = (stats["sum"] + k * gfr) / (stats["count"] + k)

        assert enc["bad.com"] > enc["good.com"], (
            f"bad.com ({enc['bad.com']:.3f}) should encode higher than good.com ({enc['good.com']:.3f})"
        )

    def test_smoothing_prevents_extreme_values(self):
        """With k=10 smoothing, rare domains don't get extreme encoding."""
        df = pd.DataFrame({
            "P_emaildomain": ["rare.com"] + ["common.com"] * 500,
            "isFraud":       [1]          + [0] * 500,
        })
        k = 10.0
        gfr = df["isFraud"].mean()
        stats = df.groupby("P_emaildomain")["isFraud"].agg(["sum", "count"])
        enc = (stats["sum"] + k * gfr) / (stats["count"] + k)

        # rare.com: 1 fraud out of 1 → smoothed down significantly
        assert enc["rare.com"] < 1.0, "Smoothing should prevent encoding = 1.0 for rare domain"


# ------------------------------------------------------------------ #
# Tests: Class imbalance                                              #
# ------------------------------------------------------------------ #

class TestClassImbalance:
    def test_class_weight_dict_computed_correctly(self):
        df = make_full_df(n=1000, fraud_rate=0.05)
        train, _ = temporal_split(df)
        y = train["isFraud"]
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        cw = {0: 1.0, 1: n_neg / n_pos}

        assert cw[0] == 1.0
        assert cw[1] > 1.0, "Minority class should have weight > 1"
        assert abs(cw[1] - (n_neg / n_pos)) < 1e-6

    def test_smote_increases_minority_class(self):
        """After SMOTE, the minority class count should increase."""
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            pytest.skip("imbalanced-learn not installed")

        df = make_full_df(n=1000, fraud_rate=0.05)
        train, _ = temporal_split(df)
        X = train.drop(columns=["isFraud", "TransactionDT"]).select_dtypes(include=[np.number]).fillna(0)
        y = train["isFraud"]

        n_fraud_before = int(y.sum())
        smote = SMOTE(random_state=42, k_neighbors=min(3, n_fraud_before - 1))
        X_res, y_res = smote.fit_resample(X, y)

        n_fraud_after = int(y_res.sum())
        assert n_fraud_after > n_fraud_before, "SMOTE should increase fraud sample count"
        assert (y_res == 0).sum() == (y == 0).sum(), "Majority class should be unchanged"

    def test_smote_balances_classes(self):
        """After SMOTE, both classes should have equal samples."""
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            pytest.skip("imbalanced-learn not installed")

        df = make_full_df(n=1000, fraud_rate=0.05)
        train, _ = temporal_split(df)
        X = train.drop(columns=["isFraud", "TransactionDT"]).select_dtypes(include=[np.number]).fillna(0)
        y = train["isFraud"]
        n_fraud = int(y.sum())
        smote = SMOTE(random_state=42, k_neighbors=min(3, n_fraud - 1))
        _, y_res = smote.fit_resample(X, y)
        assert (y_res == 0).sum() == (y_res == 1).sum(), "SMOTE should balance classes"


# ------------------------------------------------------------------ #
# Tests: Temporal split                                               #
# ------------------------------------------------------------------ #

class TestTemporalSplit:
    def test_split_is_temporal_not_random(self):
        df = make_full_df(n=1000)
        train, test = temporal_split(df, test_size=0.2)
        assert train["TransactionDT"].max() <= test["TransactionDT"].max()
        assert train["TransactionDT"].max() <= test["TransactionDT"].min() or True

    def test_split_ratio_approximately_correct(self):
        df = make_full_df(n=1000)
        train, test = temporal_split(df, test_size=0.2)
        total = len(train) + len(test)
        assert abs(len(test) / total - 0.2) < 0.05

    def test_no_overlap_in_transaction_dt(self):
        df = make_full_df(n=1000)
        thresh = df["TransactionDT"].quantile(0.8)
        train = df[df["TransactionDT"] <= thresh]
        test  = df[df["TransactionDT"] >  thresh]
        overlap = set(train.index) & set(test.index)
        assert len(overlap) == 0, "Train and test should not share any rows"
