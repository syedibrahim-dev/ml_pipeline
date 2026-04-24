"""
Stage 3: Data Preprocessing
Handles:
  - Advanced missing value imputation (median + missing-indicator for V features)
  - High-cardinality frequency encoding (card1, card2, addr1, addr2)
  - Smoothed target encoding for email domains
  - One-hot encoding for low-cardinality categoricals
  - Temporal train/test split (TransactionDT 80th percentile)
  - Class imbalance handling: SMOTE or class_weight
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.6.1",
        "imbalanced-learn==0.13.0",
        "joblib",
    ],
)
def preprocessing(
    input_transaction: Input[Dataset],
    input_identity: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    scaler_artifact: Output[Model],
    preprocessing_metrics: Output[Metrics],
    imbalance_method: str = "class_weight",
    test_size: float = 0.2,
    random_state: int = 42,
    target_encoding_smoothing: float = 10.0,
) -> None:
    """
    Full preprocessing pipeline for IEEE CIS fraud data.
    imbalance_method: "class_weight" | "smote"
    """
    import json
    import os
    import warnings
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")
    print("[preprocessing] Stage 3 – Preprocessing starting...")

    # ------------------------------------------------------------------ #
    # Load & merge                                                         #
    # ------------------------------------------------------------------ #
    tx_df = pd.read_csv(input_transaction.path)
    id_df = pd.read_csv(input_identity.path)
    df = tx_df.merge(id_df, on="TransactionID", how="left")
    print(f"[preprocessing] Merged shape: {df.shape}")

    # ------------------------------------------------------------------ #
    # Temporal train/test split (avoids data leakage)                     #
    # ------------------------------------------------------------------ #
    dt_threshold = df["TransactionDT"].quantile(1.0 - test_size)
    train_mask = df["TransactionDT"] <= dt_threshold
    test_mask  = df["TransactionDT"] >  dt_threshold

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df  = df[test_mask].copy().reset_index(drop=True)
    print(f"[preprocessing] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    # ------------------------------------------------------------------ #
    # Identify column groups                                               #
    # ------------------------------------------------------------------ #
    v_cols      = [c for c in df.columns if c.startswith("V")]
    c_cols      = [c for c in df.columns if c.startswith("C")]
    d_cols      = [c for c in df.columns if c.startswith("D")]
    m_cols      = [c for c in df.columns if c.startswith("M")]
    high_card_cols    = ["card1", "card2", "addr1", "addr2"]
    email_cols        = ["P_emaildomain", "R_emaildomain"]
    low_card_cat_cols = ["ProductCD", "card3", "card4", "card5", "card6", "DeviceType"]
    drop_cols         = ["TransactionID", "TransactionDT"]

    # ------------------------------------------------------------------ #
    # 1. Missing values: V features – median + missing indicator          #
    #    (Advanced strategy: indicator column captures "missingness" as    #
    #     a feature in its own right, since missingness in V cols is       #
    #     correlated with fraud patterns.)                                  #
    # ------------------------------------------------------------------ #
    print(f"[preprocessing] Imputing {len(v_cols)} V-feature columns with median + indicator...")
    v_medians = train_df[v_cols].median()
    new_indicator_cols = []
    for col in v_cols:
        indicator_col = f"{col}_missing"
        for split in [train_df, test_df]:
            split[indicator_col] = split[col].isna().astype(np.int8)
        new_indicator_cols.append(indicator_col)
    train_df[v_cols] = train_df[v_cols].fillna(v_medians)
    test_df[v_cols]  = test_df[v_cols].fillna(v_medians)
    print(f"[preprocessing] Added {len(new_indicator_cols)} missingness indicator columns")

    # 2. C/D columns – median imputation
    for col_list in [c_cols, d_cols]:
        for col in col_list:
            if col in train_df.columns:
                med = train_df[col].median()
                train_df[col] = train_df[col].fillna(med)
                test_df[col]  = test_df[col].fillna(med)

    # 3. M columns – fill NaN with "missing" string, then binary encode
    for col in m_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("missing")
            test_df[col]  = test_df[col].fillna("missing")
            train_df[col] = (train_df[col] == "T").astype(np.int8)
            test_df[col]  = (test_df[col] == "T").astype(np.int8)

    # 4. Card numeric cols – mode imputation
    for col in ["card1", "card2", "card3", "card5"]:
        if col in train_df.columns and train_df[col].dtype in [np.float64, np.int64]:
            mode_val = train_df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else 0
            train_df[col] = train_df[col].fillna(fill_val)
            test_df[col]  = test_df[col].fillna(fill_val)

    # 5. addr1, addr2 – median
    for col in ["addr1", "addr2"]:
        if col in train_df.columns:
            med = train_df[col].median()
            train_df[col] = train_df[col].fillna(med)
            test_df[col]  = test_df[col].fillna(med)

    # ------------------------------------------------------------------ #
    # High-cardinality frequency encoding                                 #
    # card1, card2, addr1, addr2 → replace value with its train frequency #
    # ------------------------------------------------------------------ #
    freq_maps = {}
    global_freq = 1.0 / len(train_df)
    for col in high_card_cols:
        if col not in train_df.columns:
            continue
        freq = train_df[col].value_counts(normalize=True)
        freq_maps[col] = freq.to_dict()
        train_df[col] = train_df[col].map(freq).fillna(global_freq)
        test_df[col]  = test_df[col].map(freq).fillna(global_freq)
    print(f"[preprocessing] Frequency-encoded: {high_card_cols}")

    # ------------------------------------------------------------------ #
    # Smoothed target encoding for email domains                          #
    # formula: (count_fraud_in_group + k*global_rate) / (count + k)      #
    # ------------------------------------------------------------------ #
    global_fraud_rate = train_df["isFraud"].mean()
    target_enc_maps = {}
    k = target_encoding_smoothing
    for col in email_cols:
        if col not in train_df.columns:
            continue
        train_df[col] = train_df[col].fillna("_unknown_")
        test_df[col]  = test_df[col].fillna("_unknown_")
        stats = train_df.groupby(col)["isFraud"].agg(["sum", "count"])
        enc = (stats["sum"] + k * global_fraud_rate) / (stats["count"] + k)
        target_enc_maps[col] = enc.to_dict()
        train_df[col] = train_df[col].map(enc).fillna(global_fraud_rate)
        test_df[col]  = test_df[col].map(enc).fillna(global_fraud_rate)
    print(f"[preprocessing] Target-encoded email domains: {email_cols}")

    # ------------------------------------------------------------------ #
    # Low-cardinality one-hot encoding                                    #
    # ------------------------------------------------------------------ #
    for col in low_card_cat_cols:
        if col not in train_df.columns:
            continue
        train_df[col] = train_df[col].fillna("unknown")
        test_df[col]  = test_df[col].fillna("unknown")

    # Fit encoder on train, align test to same columns
    train_encoded = pd.get_dummies(train_df[low_card_cat_cols], prefix=low_card_cat_cols, dummy_na=False)
    test_encoded  = pd.get_dummies(test_df[low_card_cat_cols],  prefix=low_card_cat_cols, dummy_na=False)
    # Align columns
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    train_df = train_df.drop(columns=low_card_cat_cols)
    test_df  = test_df.drop(columns=low_card_cat_cols)
    train_df = pd.concat([train_df.reset_index(drop=True), train_encoded.reset_index(drop=True)], axis=1)
    test_df  = pd.concat([test_df.reset_index(drop=True),  test_encoded.reset_index(drop=True)],  axis=1)
    print(f"[preprocessing] One-hot encoded: {low_card_cat_cols}")

    # ------------------------------------------------------------------ #
    # Identity numeric columns – median imputation                        #
    # ------------------------------------------------------------------ #
    id_cols_present = [c for c in train_df.columns
                       if (c.startswith("id_") and train_df[c].dtype in [np.float64, np.float32, np.int64])]
    for col in id_cols_present:
        med = train_df[col].median()
        train_df[col] = train_df[col].fillna(med)
        test_df[col]  = test_df[col].fillna(med)

    # Identity categorical columns – fill + binary
    id_cat_cols_present = [c for c in train_df.columns if c.startswith("id_") and c not in id_cols_present]
    for col in id_cat_cols_present:
        if col in train_df.columns:
            train_df[col] = (train_df[col].fillna("F") == "T").astype(np.int8)
            test_df[col]  = (test_df[col].fillna("F") == "T").astype(np.int8)

    # ------------------------------------------------------------------ #
    # Drop identifier columns                                             #
    # ------------------------------------------------------------------ #
    for col in drop_cols:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df  = test_df.drop(columns=[col])

    # Align columns (test may have extra/missing due to one-hot)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    # Ensure no remaining non-numeric columns
    obj_cols = train_df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        print(f"[preprocessing] Dropping remaining object columns: {obj_cols}")
        train_df = train_df.drop(columns=obj_cols)
        test_df  = test_df.drop(columns=[c for c in obj_cols if c in test_df.columns])

    # ------------------------------------------------------------------ #
    # Separate features and target                                        #
    # ------------------------------------------------------------------ #
    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"].astype(int)
    X_test  = test_df.drop(columns=["isFraud"])
    y_test  = test_df["isFraud"].astype(int)

    print(f"[preprocessing] Feature matrix: train {X_train.shape}, test {X_test.shape}")
    print(f"[preprocessing] Train fraud rate: {y_train.mean():.4f} | Test fraud rate: {y_test.mean():.4f}")

    # ------------------------------------------------------------------ #
    # Class imbalance handling                                            #
    # ------------------------------------------------------------------ #
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weight_dict = {0: 1.0, 1: n_neg / n_pos}

    if imbalance_method == "smote":
        from imblearn.over_sampling import SMOTE
        print(f"[preprocessing] Applying SMOTE (before: {n_pos} fraud, {n_neg} legit)...")
        smote = SMOTE(random_state=random_state, k_neighbors=min(5, n_pos - 1))
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"[preprocessing] After SMOTE: {y_train.sum()} fraud, {(y_train==0).sum()} legit")
    else:
        print(f"[preprocessing] Using class_weight={class_weight_dict} (no resampling)")

    # ------------------------------------------------------------------ #
    # Feature scaling (fit on train, transform both)                     #
    # ------------------------------------------------------------------ #
    feature_names = X_train.columns.tolist()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(np.float32))
    X_test_scaled  = scaler.transform(X_test.values.astype(np.float32))

    train_out = pd.DataFrame(X_train_scaled, columns=feature_names)
    train_out["isFraud"] = y_train.values

    test_out = pd.DataFrame(X_test_scaled, columns=feature_names)
    test_out["isFraud"] = y_test.values

    # ------------------------------------------------------------------ #
    # Save artifacts                                                       #
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(train_dataset.path), exist_ok=True)
    train_out.to_csv(train_dataset.path, index=False)
    print(f"[preprocessing] Saved train -> {train_dataset.path}")

    os.makedirs(os.path.dirname(test_dataset.path), exist_ok=True)
    test_out.to_csv(test_dataset.path, index=False)
    print(f"[preprocessing] Saved test  -> {test_dataset.path}")

    # Save scaler + encoding maps
    os.makedirs(scaler_artifact.path, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_artifact.path, "scaler.joblib"))
    with open(os.path.join(scaler_artifact.path, "freq_maps.json"), "w") as f:
        json.dump(freq_maps, f)
    with open(os.path.join(scaler_artifact.path, "target_enc_maps.json"), "w") as f:
        json.dump(target_enc_maps, f)
    with open(os.path.join(scaler_artifact.path, "class_weight.json"), "w") as f:
        json.dump(class_weight_dict, f)
    with open(os.path.join(scaler_artifact.path, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    print(f"[preprocessing] Saved scaler + maps -> {scaler_artifact.path}")

    # ------------------------------------------------------------------ #
    # Log metrics                                                          #
    # ------------------------------------------------------------------ #
    preprocessing_metrics.log_metric("train_rows", len(train_out))
    preprocessing_metrics.log_metric("test_rows", len(test_out))
    preprocessing_metrics.log_metric("n_features", len(feature_names))
    preprocessing_metrics.log_metric("train_fraud_rate", round(float(y_train.mean()), 6))
    preprocessing_metrics.log_metric("test_fraud_rate", round(float(y_test.mean()), 6))
    preprocessing_metrics.log_metric("class_weight_ratio", round(n_neg / n_pos, 2))
    preprocessing_metrics.log_metric("imbalance_method", 1 if imbalance_method == "smote" else 0)

    print("\n[preprocessing] ===== PREPROCESSING SUMMARY =====")
    print(f"  Features       : {len(feature_names)}")
    print(f"  V indicators   : {len(new_indicator_cols)}")
    print(f"  Train rows     : {len(train_out):,}")
    print(f"  Test rows      : {len(test_out):,}")
    print(f"  Imbalance      : {imbalance_method}")
    print("[preprocessing] Stage 3 complete.")
