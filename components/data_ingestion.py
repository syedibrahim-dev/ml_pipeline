"""
Stage 1: Data Ingestion
Loads the IEEE CIS Fraud Detection dataset from disk.
Falls back to a synthetic dataset with identical schema if files are not present.
"""

from kfp import dsl
from kfp.dsl import Output, Dataset


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["pandas==2.2.3", "numpy==1.26.4"],
)
def data_ingestion(
    output_transaction: Output[Dataset],
    output_identity: Output[Dataset],
    data_root: str = "/pipeline-root/data/raw",
    n_synthetic: int = 50000,
    fraud_rate: float = 0.035,
    random_state: int = 42,
) -> None:
    """
    Load IEEE CIS Fraud Detection data.
    Tries data_root first; generates realistic synthetic data as fallback.
    """
    import os
    import json
    import numpy as np
    import pandas as pd

    # ------------------------------------------------------------------ #
    # Helper: synthetic data generator with exact IEEE CIS schema         #
    # ------------------------------------------------------------------ #
    def generate_synthetic(n: int, fraud_rate: float, seed: int) -> tuple:
        """Return (transaction_df, identity_df) matching IEEE CIS columns."""
        rng = np.random.default_rng(seed)

        n_fraud = int(n * fraud_rate)
        n_legit = n - n_fraud
        is_fraud = np.array([0] * n_legit + [1] * n_fraud)
        rng.shuffle(is_fraud)

        tx_ids = np.arange(2987000, 2987000 + n)

        # TransactionDT: seconds elapsed – span ~6 months (0 to 15_552_000)
        tx_dt = rng.integers(0, 15_552_000, size=n)

        # TransactionAmt: log-normal
        tx_amt = rng.lognormal(mean=4.0, sigma=2.0, size=n).round(2)
        tx_amt = np.clip(tx_amt, 0.01, 50000.0)

        # ProductCD (5 categories)
        product_cd = rng.choice(["W", "H", "C", "S", "R"], size=n)

        # card1-card6
        card1 = rng.integers(1000, 18500, size=n)
        card2 = rng.choice([np.nan, *rng.integers(100, 600, size=200).tolist()], size=n)
        card3 = rng.choice([150.0, 185.0, np.nan], size=n)
        card4 = rng.choice(["visa", "mastercard", "american express", "discover", np.nan], size=n)
        card5 = rng.choice([102.0, 117.0, 226.0, np.nan], size=n)
        card6 = rng.choice(["debit", "credit", "debit or credit", np.nan], size=n)

        # addr1, addr2
        addr1 = rng.choice(list(range(100, 500)) + [np.nan], size=n)
        addr2 = rng.choice(list(range(10, 102)) + [np.nan], size=n)

        # email domains
        domains = [
            "gmail.com",
            "yahoo.com",
            "hotmail.com",
            "outlook.com",
            "protonmail.com",
            "icloud.com",
            "anonymous.com",
            np.nan,
        ]
        p_email = rng.choice(domains, size=n)
        r_email = rng.choice(domains, size=n)

        # dist1, dist2
        dist1 = rng.choice(list(range(0, 4000)) + [np.nan] * 500, size=n)
        dist2 = rng.choice(list(range(0, 4000)) + [np.nan] * 1000, size=n)

        # C1-C14 (counts)
        c_cols = {f"C{i}": rng.integers(0, 2500, size=n).astype(float) for i in range(1, 15)}
        # ~15% missing in some C columns
        for col in ["C3", "C7", "C11"]:
            mask = rng.random(n) < 0.15
            c_cols[col][mask] = np.nan

        # D1-D15 (days)
        d_cols = {}
        for i in range(1, 16):
            vals = rng.integers(0, 800, size=n).astype(float)
            miss_rate = rng.uniform(0.1, 0.7)
            mask = rng.random(n) < miss_rate
            vals[mask] = np.nan
            d_cols[f"D{i}"] = vals

        # M1-M9 (match flags)
        m_cols = {}
        for i in range(1, 10):
            choices = ["T", "F", np.nan]
            m_cols[f"M{i}"] = rng.choice(choices, size=n)

        # V1-V339: grouped Vesta features (block generation)
        v_data = rng.standard_normal((n, 339)).astype(np.float32)
        # ~40% NaN overall, with higher missingness in later V groups
        nan_mask = rng.random((n, 339)) < 0.40
        v_data[nan_mask] = np.nan
        v_cols = {f"V{i}": v_data[:, i - 1] for i in range(1, 340)}

        # Build transaction DataFrame
        tx_df = pd.DataFrame(
            {
                "TransactionID": tx_ids,
                "isFraud": is_fraud.astype(int),
                "TransactionDT": tx_dt,
                "TransactionAmt": tx_amt,
                "ProductCD": product_cd,
                "card1": card1.astype(float),
                "card2": card2,
                "card3": card3,
                "card4": card4,
                "card5": card5,
                "card6": card6,
                "addr1": addr1,
                "addr2": addr2,
                "dist1": dist1,
                "dist2": dist2,
                "P_emaildomain": p_email,
                "R_emaildomain": r_email,
                **c_cols,
                **d_cols,
                **m_cols,
                **v_cols,
            }
        )

        # Build identity DataFrame (subset of transactions)
        n_id = int(n * 0.3)
        id_idx = rng.choice(tx_ids, size=n_id, replace=False)
        id_cols = {f"id_0{i}" if i < 10 else f"id_{i}": rng.standard_normal(n_id) for i in range(1, 12)}
        id_cats = {f"id_{i}": rng.choice(["T", "F", np.nan], size=n_id) for i in range(12, 39)}
        id_df = pd.DataFrame(
            {
                "TransactionID": id_idx,
                "DeviceType": rng.choice(["desktop", "mobile", np.nan], size=n_id),
                "DeviceInfo": rng.choice(["Windows", "iOS Device", "MacOS", "Android", np.nan], size=n_id),
                **id_cols,
                **id_cats,
            }
        )

        return tx_df, id_df

    # ------------------------------------------------------------------ #
    # Try loading real data; fall back to synthetic                        #
    # ------------------------------------------------------------------ #
    tx_path = os.path.join(data_root, "train_transaction.csv")
    id_path = os.path.join(data_root, "train_identity.csv")

    if os.path.exists(tx_path) and os.path.exists(id_path):
        print(f"[data_ingestion] Loading real IEEE CIS data from {data_root}")
        tx_df = pd.read_csv(tx_path)
        id_df = pd.read_csv(id_path)
        print(f"[data_ingestion] Loaded transaction: {tx_df.shape}, identity: {id_df.shape}")
    else:
        print(f"[data_ingestion] Real data not found at {data_root}. Generating {n_synthetic:,} synthetic rows.")
        tx_df, id_df = generate_synthetic(n_synthetic, fraud_rate, random_state)
        print(f"[data_ingestion] Synthetic transaction: {tx_df.shape}, identity: {id_df.shape}")

    # ------------------------------------------------------------------ #
    # Dataset statistics                                                   #
    # ------------------------------------------------------------------ #
    fraud_count = int(tx_df["isFraud"].sum())
    total = len(tx_df)
    actual_rate = fraud_count / total

    print(f"[data_ingestion] Total transactions : {total:,}")
    print(f"[data_ingestion] Fraud cases        : {fraud_count:,}  ({actual_rate:.2%})")
    print(f"[data_ingestion] Columns (tx)       : {tx_df.shape[1]}")
    print(f"[data_ingestion] Overall missing %  : {tx_df.isnull().mean().mean():.2%}")

    # Log stats as JSON metadata alongside the artifact
    stats = {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "fraud_rate": round(actual_rate, 6),
        "tx_columns": tx_df.shape[1],
        "id_columns": id_df.shape[1],
        "source": "real" if os.path.exists(tx_path) else "synthetic",
    }
    stats_path = output_transaction.path + "_stats.json"
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ------------------------------------------------------------------ #
    # Save artifacts                                                       #
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(output_transaction.path), exist_ok=True)
    tx_df.to_csv(output_transaction.path, index=False)
    print(f"[data_ingestion] Saved transaction data -> {output_transaction.path}")

    os.makedirs(os.path.dirname(output_identity.path), exist_ok=True)
    id_df.to_csv(output_identity.path, index=False)
    print(f"[data_ingestion] Saved identity data    -> {output_identity.path}")
    print("[data_ingestion] Stage 1 complete.")
