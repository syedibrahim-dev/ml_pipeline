"""
Stage 2: Data Validation
Validates the ingested dataset for schema integrity, missing values,
target distribution, and value ranges before any transformation.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Metrics


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["pandas==2.2.3", "numpy==1.26.4"],
)
def data_validation(
    input_transaction: Input[Dataset],
    input_identity: Input[Dataset],
    validation_report: Output[Dataset],
    validation_metrics: Output[Metrics],
    min_fraud_rate: float = 0.001,
    max_fraud_rate: float = 0.20,
    max_missing_pct: float = 0.95,
) -> None:
    """
    Perform comprehensive data validation checks.
    Writes a JSON report and logs a pass/fail metric.
    Does NOT raise on failure – logs issues so the pipeline continues.
    """
    import json
    import os
    import pandas as pd

    print("[data_validation] Stage 2 – Data Validation starting...")

    # Required columns in the transaction table
    REQUIRED_TX_COLS = [
        "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    ]

    issues = []
    warnings = []
    checks_passed = 0
    checks_total = 0

    # ------------------------------------------------------------------ #
    # Load data                                                            #
    # ------------------------------------------------------------------ #
    tx_df = pd.read_csv(input_transaction.path)
    id_df = pd.read_csv(input_identity.path)
    print(f"[data_validation] Loaded tx: {tx_df.shape}, id: {id_df.shape}")

    # ------------------------------------------------------------------ #
    # Check 1: Required schema columns                                     #
    # ------------------------------------------------------------------ #
    checks_total += 1
    missing_cols = [c for c in REQUIRED_TX_COLS if c not in tx_df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        print(f"[data_validation] FAIL – Missing columns: {missing_cols}")
    else:
        checks_passed += 1
        print("[data_validation] PASS – All required columns present")

    # ------------------------------------------------------------------ #
    # Check 2: Target column (isFraud) is binary 0/1                      #
    # ------------------------------------------------------------------ #
    checks_total += 1
    if "isFraud" in tx_df.columns:
        unique_labels = set(tx_df["isFraud"].dropna().unique())
        if not unique_labels.issubset({0, 1}):
            issues.append(f"isFraud contains unexpected values: {unique_labels}")
        else:
            checks_passed += 1
            print("[data_validation] PASS – isFraud is binary")
    else:
        issues.append("isFraud column absent")

    # ------------------------------------------------------------------ #
    # Check 3: Fraud rate within acceptable range                         #
    # ------------------------------------------------------------------ #
    checks_total += 1
    if "isFraud" in tx_df.columns:
        fraud_rate = tx_df["isFraud"].mean()
        print(f"[data_validation] Fraud rate: {fraud_rate:.4f}")
        if fraud_rate < min_fraud_rate or fraud_rate > max_fraud_rate:
            issues.append(
                f"Fraud rate {fraud_rate:.4f} outside [{min_fraud_rate}, {max_fraud_rate}]"
            )
        else:
            checks_passed += 1
            print(f"[data_validation] PASS – Fraud rate {fraud_rate:.4f} within bounds")
    else:
        fraud_rate = None

    # ------------------------------------------------------------------ #
    # Check 4: TransactionAmt > 0                                         #
    # ------------------------------------------------------------------ #
    checks_total += 1
    if "TransactionAmt" in tx_df.columns:
        n_invalid_amt = int((tx_df["TransactionAmt"] <= 0).sum())
        if n_invalid_amt > 0:
            warnings.append(f"{n_invalid_amt} rows with TransactionAmt <= 0")
            print(f"[data_validation] WARN – {n_invalid_amt} rows with non-positive TransactionAmt")
        else:
            checks_passed += 1
            print("[data_validation] PASS – TransactionAmt all positive")
    else:
        issues.append("TransactionAmt column absent")

    # ------------------------------------------------------------------ #
    # Check 5: TransactionDT > 0                                          #
    # ------------------------------------------------------------------ #
    checks_total += 1
    if "TransactionDT" in tx_df.columns:
        n_invalid_dt = int((tx_df["TransactionDT"] <= 0).sum())
        if n_invalid_dt > 0:
            warnings.append(f"{n_invalid_dt} rows with TransactionDT <= 0")
        else:
            checks_passed += 1
            print("[data_validation] PASS – TransactionDT all positive")
    else:
        issues.append("TransactionDT column absent")

    # ------------------------------------------------------------------ #
    # Check 6: Missing value audit                                        #
    # ------------------------------------------------------------------ #
    checks_total += 1
    missing_pct = tx_df.isnull().mean()
    high_missing = missing_pct[missing_pct > max_missing_pct]
    if not high_missing.empty:
        warnings.append(
            f"{len(high_missing)} columns exceed {max_missing_pct:.0%} missing: "
            f"{high_missing.index.tolist()[:10]}..."
        )
        print(f"[data_validation] WARN – {len(high_missing)} columns >95% missing")
    else:
        checks_passed += 1
        print("[data_validation] PASS – No column exceeds 95% missing")

    # ------------------------------------------------------------------ #
    # Check 7: Duplicate TransactionIDs                                   #
    # ------------------------------------------------------------------ #
    checks_total += 1
    if "TransactionID" in tx_df.columns:
        n_dups = int(tx_df["TransactionID"].duplicated().sum())
        if n_dups > 0:
            issues.append(f"{n_dups} duplicate TransactionIDs found")
            print(f"[data_validation] FAIL – {n_dups} duplicate TransactionIDs")
        else:
            checks_passed += 1
            print("[data_validation] PASS – No duplicate TransactionIDs")

    # ------------------------------------------------------------------ #
    # Check 8: Row count sanity                                           #
    # ------------------------------------------------------------------ #
    checks_total += 1
    if len(tx_df) < 100:
        issues.append(f"Too few rows: {len(tx_df)}")
    else:
        checks_passed += 1
        print(f"[data_validation] PASS – Row count {len(tx_df):,} is adequate")

    # ------------------------------------------------------------------ #
    # Summary statistics for the report                                   #
    # ------------------------------------------------------------------ #
    top_missing = (
        missing_pct.sort_values(ascending=False).head(10).round(4).to_dict()
    )

    # V-feature group missing rates
    v_cols = [c for c in tx_df.columns if c.startswith("V")]
    v_missing = float(tx_df[v_cols].isnull().mean().mean()) if v_cols else None

    report = {
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "pass_rate": round(checks_passed / checks_total, 4) if checks_total > 0 else 0,
        "issues": issues,
        "warnings": warnings,
        "dataset_stats": {
            "tx_rows": len(tx_df),
            "tx_columns": tx_df.shape[1],
            "id_rows": len(id_df),
            "id_columns": id_df.shape[1],
            "fraud_rate": round(float(fraud_rate), 6) if fraud_rate is not None else None,
            "overall_missing_pct": round(float(tx_df.isnull().mean().mean()), 4),
            "v_feature_missing_pct": round(v_missing, 4) if v_missing is not None else None,
        },
        "top_missing_columns": top_missing,
        "validation_passed": len(issues) == 0,
    }

    # ------------------------------------------------------------------ #
    # Save report                                                          #
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(validation_report.path), exist_ok=True)
    with open(validation_report.path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[data_validation] Report saved -> {validation_report.path}")

    # ------------------------------------------------------------------ #
    # Log metrics                                                          #
    # ------------------------------------------------------------------ #
    validation_metrics.log_metric("checks_passed", checks_passed)
    validation_metrics.log_metric("checks_total", checks_total)
    validation_metrics.log_metric("pass_rate", round(checks_passed / checks_total, 4))
    validation_metrics.log_metric("issues_count", len(issues))
    validation_metrics.log_metric("warnings_count", len(warnings))
    validation_metrics.log_metric("validation_passed", int(len(issues) == 0))

    # Summary print
    print("\n[data_validation] ===== VALIDATION SUMMARY =====")
    print(f"  Checks passed : {checks_passed}/{checks_total}")
    print(f"  Issues        : {len(issues)}")
    print(f"  Warnings      : {len(warnings)}")
    for iss in issues:
        print(f"  [ISSUE]   {iss}")
    for wrn in warnings:
        print(f"  [WARNING] {wrn}")
    print("[data_validation] Stage 2 complete.")
