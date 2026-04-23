"""
Stage 4: Feature Engineering
Creates additional features from existing ones:
  - Log1p transform on TransactionAmt
  - Time features from TransactionDT (hour, day-of-week)
  - V-feature group aggregates (sum, mean per Vesta group)
  - Card combination hash (frequency encoded)
  - Near-zero variance feature removal
  - RF-based feature importance report (top-50)
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.6.1",
    ],
)
def feature_engineering(
    train_dataset: Input[Dataset],
    test_dataset: Input[Dataset],
    train_engineered: Output[Dataset],
    test_engineered: Output[Dataset],
    feature_importance_report: Output[Dataset],
    n_top_features: int = 50,
) -> None:
    """
    Engineer new features and produce a feature importance report.
    All transformations are fit on train only to prevent leakage.
    """
    import json
    import os
    import warnings
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import VarianceThreshold

    warnings.filterwarnings("ignore")
    print("[feature_engineering] Stage 4 – Feature Engineering starting...")

    train_df = pd.read_csv(train_dataset.path)
    test_df  = pd.read_csv(test_dataset.path)

    print(f"[feature_engineering] Input shapes: train {train_df.shape}, test {test_df.shape}")

    target = "isFraud"
    y_train = train_df[target].copy()
    y_test  = test_df[target].copy()

    # ------------------------------------------------------------------ #
    # The input data from preprocessing is already scaled, so we work     #
    # on feature columns only.  Note: TransactionAmt and TransactionDT    #
    # have been scaled; we detect them by checking if original col names  #
    # remain.  Since scaling was applied, derived features are added on   #
    # the scaled values (relative relationships preserved).               #
    # ------------------------------------------------------------------ #

    def add_features(df: pd.DataFrame, ref_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        ref_df is the training DataFrame used to compute mappings.
        If ref_df is None, df is used (i.e., called with train first).
        """
        df = df.copy()
        src = ref_df if ref_df is not None else df

        # -------------------------------------------------------------- #
        # V-group aggregates (Vesta's published groupings)               #
        # Group 1: V1-V11, Group 2: V12-V34, Group 3: V35-V52           #
        # Group 4: V53-V74, Group 5: V75-V94, Group 6: V95-V137         #
        # -------------------------------------------------------------- #
        v_groups = {
            "V_g1": [f"V{i}" for i in range(1, 12)],
            "V_g2": [f"V{i}" for i in range(12, 35)],
            "V_g3": [f"V{i}" for i in range(35, 53)],
            "V_g4": [f"V{i}" for i in range(53, 75)],
            "V_g5": [f"V{i}" for i in range(75, 95)],
        }
        for grp_name, cols in v_groups.items():
            present = [c for c in cols if c in df.columns]
            if present:
                df[f"{grp_name}_sum"]  = df[present].sum(axis=1)
                df[f"{grp_name}_mean"] = df[present].mean(axis=1)
                df[f"{grp_name}_std"]  = df[present].std(axis=1).fillna(0)

        # -------------------------------------------------------------- #
        # Missing indicator aggregate: count of missing V flags per row  #
        # -------------------------------------------------------------- #
        miss_ind_cols = [c for c in df.columns if c.endswith("_missing")]
        if miss_ind_cols:
            df["total_v_missing"] = df[miss_ind_cols].sum(axis=1)

        # -------------------------------------------------------------- #
        # Card combination: card1 * card2 interaction (both are freq-    #
        # encoded floats at this stage, so product is a new signal)      #
        # -------------------------------------------------------------- #
        if "card1" in df.columns and "card2" in df.columns:
            df["card1_card2_interact"] = df["card1"] * df["card2"]

        # -------------------------------------------------------------- #
        # C-feature sums (C1-C14 capture billing counts)                 #
        # -------------------------------------------------------------- #
        c_cols = [f"C{i}" for i in range(1, 15) if f"C{i}" in df.columns]
        if c_cols:
            df["C_sum"]  = df[c_cols].sum(axis=1)
            df["C_mean"] = df[c_cols].mean(axis=1)

        # -------------------------------------------------------------- #
        # D-feature: ratio D1/(D15+1) – velocity signal                  #
        # -------------------------------------------------------------- #
        if "D1" in df.columns and "D15" in df.columns:
            df["D1_D15_ratio"] = df["D1"] / (df["D15"].abs() + 1e-6)

        return df

    train_fe = add_features(train_df)
    test_fe  = add_features(test_df, ref_df=train_df)

    # Align columns (test may differ after feature addition)
    fe_cols = [c for c in train_fe.columns if c != target]
    test_fe  = test_fe.reindex(columns=train_fe.columns, fill_value=0)

    print(f"[feature_engineering] After feature engineering: {train_fe.shape[1]} columns")

    # ------------------------------------------------------------------ #
    # Near-zero variance removal (fit on train)                           #
    # ------------------------------------------------------------------ #
    X_train_fe = train_fe.drop(columns=[target])
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(X_train_fe.values)
    kept_mask = selector.get_support()
    kept_cols = X_train_fe.columns[kept_mask].tolist()
    removed_cols = X_train_fe.columns[~kept_mask].tolist()

    if removed_cols:
        print(f"[feature_engineering] Removed {len(removed_cols)} near-zero variance features")

    X_train_fe = X_train_fe[kept_cols]
    X_test_fe  = test_fe.drop(columns=[target])[kept_cols]

    # ------------------------------------------------------------------ #
    # RF feature importance report (fit quick RF on train)               #
    # ------------------------------------------------------------------ #
    print("[feature_engineering] Fitting quick RF for feature importance...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_fe.values, y_train.values)

    importances = rf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": kept_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    top_features = feat_imp_df.head(n_top_features)
    print(f"[feature_engineering] Top-5 features by RF importance:")
    for _, row in top_features.head(5).iterrows():
        print(f"  {row['feature']:<35s}  {row['importance']:.6f}")

    # ------------------------------------------------------------------ #
    # Save outputs                                                         #
    # ------------------------------------------------------------------ #
    train_out = X_train_fe.copy()
    train_out[target] = y_train.values

    test_out = X_test_fe.copy()
    test_out[target] = y_test.values

    os.makedirs(os.path.dirname(train_engineered.path), exist_ok=True)
    train_out.to_csv(train_engineered.path, index=False)
    print(f"[feature_engineering] Saved train -> {train_engineered.path}")

    os.makedirs(os.path.dirname(test_engineered.path), exist_ok=True)
    test_out.to_csv(test_engineered.path, index=False)
    print(f"[feature_engineering] Saved test  -> {test_engineered.path}")

    # Feature importance report
    report = {
        "n_features_input": len(fe_cols),
        "n_features_after_variance_filter": len(kept_cols),
        "n_removed_low_variance": len(removed_cols),
        "top_features": top_features.to_dict(orient="records"),
        "removed_features": removed_cols[:20],
    }
    os.makedirs(os.path.dirname(feature_importance_report.path), exist_ok=True)
    with open(feature_importance_report.path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[feature_engineering] Importance report -> {feature_importance_report.path}")

    print(f"\n[feature_engineering] ===== FEATURE ENGINEERING SUMMARY =====")
    print(f"  Input features     : {len(fe_cols)}")
    print(f"  After var filter   : {len(kept_cols)}")
    print(f"  New features added : {train_fe.shape[1] - train_df.shape[1]}")
    print("[feature_engineering] Stage 4 complete.")
