"""
Local Pipeline Runner – Assignment #4
======================================
Runs all 9 assignment tasks locally (no Kubernetes required).
Mirrors the pattern from mloopsasign#3/run_experiments.py.

Usage:
  python run_pipeline.py                # run all tasks
  python run_pipeline.py --task models  # run a specific task
  python run_pipeline.py --help

Tasks:
  all         – run everything (default)
  pipeline    – compile + run full 7-stage pipeline
  models      – compare XGBoost, LightGBM, RF-Hybrid
  imbalance   – compare SMOTE vs class_weight
  cost        – compare standard vs cost-sensitive training
  drift       – time-based drift simulation
  retrain     – retraining strategy comparison
  explain     – SHAP explainability analysis

Results are saved to results/metrics/, results/plots/, results/models/
"""

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR  = "results"
METRICS_DIR  = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR   = os.path.join(RESULTS_DIR, "models")

for d in [METRICS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------------ #
# Synthetic data generator (reused across tasks)                      #
# ------------------------------------------------------------------ #

def generate_synthetic_data(n: int = 30000, fraud_rate: float = 0.035, seed: int = 42):
    """Generate synthetic IEEE CIS–schema DataFrame for local testing."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud
    is_fraud = np.array([0] * n_legit + [1] * n_fraud)
    rng.shuffle(is_fraud)

    tx_dt = rng.integers(0, 15_552_000, size=n)
    tx_amt = rng.lognormal(mean=4.0, sigma=2.0, size=n).round(2)
    tx_amt = np.clip(tx_amt, 0.01, 50000.0)

    # V features (V1-V339) as block with ~40% NaN
    v_data = rng.standard_normal((n, 50)).astype(np.float32)  # 50 V cols for speed
    nan_mask = rng.random((n, 50)) < 0.40
    v_data[nan_mask] = np.nan

    c_data = rng.integers(0, 500, size=(n, 10)).astype(float)
    d_data = rng.integers(0, 300, size=(n, 5)).astype(float)
    d_data[rng.random((n, 5)) < 0.5] = np.nan

    df = pd.DataFrame({
        "TransactionDT":  tx_dt,
        "TransactionAmt": tx_amt,
        "card1": rng.integers(1000, 18500, size=n).astype(float),
        "card2": rng.choice([float("nan"), *rng.integers(100, 600, size=200).tolist()], size=n),
        "addr1": rng.choice(list(range(100, 500)) + [float("nan")], size=n),
        "addr2": rng.choice(list(range(10, 102)) + [float("nan")], size=n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", float("nan")], size=n),
        "R_emaildomain": rng.choice(["gmail.com", "outlook.com", float("nan")], size=n),
        **{f"C{i}": c_data[:, i-1] for i in range(1, 11)},
        **{f"D{i}": d_data[:, i-1] for i in range(1, 6)},
        **{f"M{i}": rng.choice(["T", "F", float("nan")], size=n) for i in range(1, 6)},
        **{f"V{i}": v_data[:, i-1] for i in range(1, 51)},
        "isFraud": is_fraud,
    })
    return df


def preprocess_local(df: pd.DataFrame, imbalance_method: str = "class_weight", seed: int = 42):
    """
    Lightweight local preprocessing (matches component logic).
    Returns (X_train, X_test, y_train, y_test, feature_names, class_weight).
    """
    from sklearn.preprocessing import StandardScaler

    # Temporal split
    dt_thresh = df["TransactionDT"].quantile(0.8)
    train_df = df[df["TransactionDT"] <= dt_thresh].copy()
    test_df  = df[df["TransactionDT"] >  dt_thresh].copy()

    # Missing value imputation
    v_cols = [c for c in df.columns if c.startswith("V")]
    for col in v_cols:
        med = train_df[col].median()
        train_df[col] = train_df[col].fillna(med)
        test_df[col]  = test_df[col].fillna(med)

    for col in ["addr1", "addr2"] + [f"C{i}" for i in range(1, 11)] + [f"D{i}" for i in range(1, 6)]:
        if col in train_df.columns:
            med = train_df[col].median()
            train_df[col] = train_df[col].fillna(med)
            test_df[col]  = test_df[col].fillna(med)

    for col in [f"M{i}" for i in range(1, 6)]:
        if col in train_df.columns:
            train_df[col] = (train_df[col].fillna("F") == "T").astype(int)
            test_df[col]  = (test_df[col].fillna("F") == "T").astype(int)

    # Frequency encode card/addr
    for col in ["card1", "card2"]:
        freq = train_df[col].value_counts(normalize=True)
        train_df[col] = train_df[col].map(freq).fillna(1.0 / len(train_df))
        test_df[col]  = test_df[col].map(freq).fillna(1.0 / len(train_df))

    # Email target encoding
    gfr = train_df["isFraud"].mean()
    for col in ["P_emaildomain", "R_emaildomain"]:
        train_df[col] = train_df[col].fillna("_unk_")
        test_df[col]  = test_df[col].fillna("_unk_")
        stats = train_df.groupby(col)["isFraud"].agg(["sum", "count"])
        enc = (stats["sum"] + 10 * gfr) / (stats["count"] + 10)
        train_df[col] = train_df[col].map(enc).fillna(gfr)
        test_df[col]  = test_df[col].map(enc).fillna(gfr)

    drop_cols = ["TransactionDT"]
    for col in drop_cols:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
            test_df  = test_df.drop(columns=[col])

    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"].astype(int)
    X_test  = test_df.drop(columns=["isFraud"])
    y_test  = test_df["isFraud"].astype(int)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    feature_names = X_train.columns.tolist()

    n_pos = y_train.sum()
    n_neg = (y_train == 0).sum()
    class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}

    if imbalance_method == "smote":
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=seed, k_neighbors=min(5, n_pos - 1))
        X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test  = pd.DataFrame(scaler.transform(X_test.reindex(columns=feature_names, fill_value=0)),
                           columns=feature_names)

    return X_train, X_test, y_train, y_test, feature_names, class_weight


def evaluate_model(model, X_test, y_test, model_type: str = "xgboost",
                   fn_cost: float = 10.0, fp_cost: float = 1.0):
    """Return metrics dict for a trained model."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
    )

    if isinstance(model, dict) and model.get("type") == "rf_hybrid":
        X_t = X_test.values[:, model["selected_indices"]]
        y_prob = model["xgb_model"].predict_proba(X_t)[:, 1]
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "model_type":   model_type,
        "precision":    round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":       round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":           round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "auc_roc":      round(float(roc_auc_score(y_test, y_prob)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "business_cost": round(fn_cost * fn + fp_cost * fp, 2),
    }


# ------------------------------------------------------------------ #
# Task runners                                                         #
# ------------------------------------------------------------------ #

def task_pipeline():
    """Compile the Kubeflow pipeline YAML."""
    print("\n" + "="*60)
    print("TASK 1: Compiling Kubeflow Pipeline YAML")
    print("="*60)
    try:
        from pipelines.fraud_pipeline import compile_pipeline
        yaml_path = compile_pipeline()
        print(f"[OK] Pipeline compiled -> {yaml_path}")
        return yaml_path
    except Exception as e:
        print(f"[ERROR] Pipeline compilation: {e}")
        return None


def task_models():
    """Task 3: Compare XGBoost, LightGBM, RF-Hybrid models."""
    print("\n" + "="*60)
    print("TASK 3: Model Comparison – XGBoost vs LightGBM vs RF-Hybrid")
    print("="*60)
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier

    df = generate_synthetic_data(n=20000, seed=42)
    X_train, X_test, y_train, y_test, feat_names, cw = preprocess_local(df)

    n_pos = y_train.sum()
    n_neg = (y_train == 0).sum()
    spw   = n_neg / max(n_pos, 1)

    results = []

    # XGBoost
    print("[model_comparison] Training XGBoost...")
    t0 = time.time()
    m_xgb = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                               scale_pos_weight=spw, random_state=42, verbosity=0,
                               use_label_encoder=False)
    m_xgb.fit(X_train, y_train)
    r = evaluate_model(m_xgb, X_test, y_test, "xgboost")
    r["time_s"] = round(time.time() - t0, 1)
    results.append(r)
    import joblib
    joblib.dump(m_xgb, os.path.join(MODELS_DIR, "xgboost_model.joblib"))

    # LightGBM
    print("[model_comparison] Training LightGBM...")
    t0 = time.time()
    m_lgb = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                num_leaves=31, class_weight="balanced",
                                random_state=42, verbose=-1)
    m_lgb.fit(X_train, y_train)
    r = evaluate_model(m_lgb, X_test, y_test, "lightgbm")
    r["time_s"] = round(time.time() - t0, 1)
    results.append(r)
    joblib.dump(m_lgb, os.path.join(MODELS_DIR, "lightgbm_model.joblib"))

    # RF-Hybrid (RF feature selection → XGBoost)
    print("[model_comparison] Training RF-Hybrid...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=50, max_depth=None, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    top_idx = np.argsort(rf.feature_importances_)[::-1][:30]
    top_idx_sorted = sorted(top_idx)
    X_tr_red = X_train.values[:, top_idx_sorted]
    X_te_red = X_test.values[:, top_idx_sorted]
    m_hybrid = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                  scale_pos_weight=spw, random_state=42, verbosity=0,
                                  use_label_encoder=False)
    m_hybrid.fit(X_tr_red, y_train)
    hybrid_model = {"type": "rf_hybrid", "rf_selector": rf, "xgb_model": m_hybrid,
                    "selected_indices": top_idx_sorted}
    r = evaluate_model(hybrid_model, X_test, y_test, "rf_hybrid")
    r["time_s"] = round(time.time() - t0, 1)
    results.append(r)
    joblib.dump(hybrid_model, os.path.join(MODELS_DIR, "rf_hybrid_model.joblib"))

    # Print table
    print("\n{:<15} {:>10} {:>10} {:>10} {:>10} {:>12} {:>8}".format(
        "Model", "Precision", "Recall", "F1", "AUC-ROC", "BizCost", "Time(s)"))
    print("-" * 80)
    for r in results:
        print("{:<15} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.0f} {:>8.1f}".format(
            r["model_type"], r["precision"], r["recall"], r["f1"],
            r["auc_roc"], r["business_cost"], r["time_s"]))

    # Save results
    with open(os.path.join(METRICS_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved -> {METRICS_DIR}/model_comparison.json")
    return results


def task_imbalance():
    """Task 2: Compare SMOTE vs class_weight imbalance handling."""
    print("\n" + "="*60)
    print("TASK 2: Imbalance Strategy Comparison – SMOTE vs class_weight")
    print("="*60)
    import xgboost as xgb

    df = generate_synthetic_data(n=20000, seed=42)
    results = []

    for method in ["class_weight", "smote"]:
        print(f"\n[imbalance] Method: {method}")
        X_train, X_test, y_train, y_test, feat_names, cw = preprocess_local(
            df.copy(), imbalance_method=method
        )
        n_pos = y_train.sum()
        n_neg = (y_train == 0).sum()
        spw   = n_neg / max(n_pos, 1) if method == "class_weight" else 1.0
        print(f"  Train class distribution: {(y_train==0).sum()} legit, {n_pos} fraud")

        t0 = time.time()
        m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                               scale_pos_weight=spw, random_state=42, verbosity=0,
                               use_label_encoder=False)
        m.fit(X_train, y_train)
        r = evaluate_model(m, X_test, y_test)
        r["method"] = method
        r["train_fraud_count"] = int(n_pos)
        r["time_s"] = round(time.time() - t0, 1)
        results.append(r)

    print("\n{:<15} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
        "Method", "Precision", "Recall", "F1", "AUC-ROC", "BizCost"))
    print("-" * 70)
    for r in results:
        print("{:<15} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.0f}".format(
            r["method"], r["precision"], r["recall"],
            r["f1"], r["auc_roc"], r["business_cost"]))

    winner = max(results, key=lambda x: x["recall"])
    print(f"\n[analysis] Best recall: '{winner['method']}' (recall={winner['recall']:.4f})")

    with open(os.path.join(METRICS_DIR, "imbalance_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved -> {METRICS_DIR}/imbalance_comparison.json")
    return results


def task_cost_sensitive():
    """Task 4: Compare standard vs cost-sensitive training."""
    print("\n" + "="*60)
    print("TASK 4: Cost-Sensitive Learning – Standard vs Cost-Sensitive")
    print("="*60)
    import xgboost as xgb

    df = generate_synthetic_data(n=20000, seed=42)
    X_train, X_test, y_train, y_test, feat_names, cw = preprocess_local(df)
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    base_spw = n_neg / max(n_pos, 1)

    results = []
    fn_cost, fp_cost = 10.0, 1.0

    for label, cost_sensitive, spw_val in [
        ("Standard (equal cost)",          False, base_spw),
        (f"Cost-Sensitive (FN={fn_cost}x)", True,  base_spw * fn_cost),
    ]:
        print(f"\n[cost] Training: {label}  (scale_pos_weight={spw_val:.2f})")
        m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                               scale_pos_weight=spw_val, random_state=42, verbosity=0,
                               use_label_encoder=False)
        m.fit(X_train, y_train)
        r = evaluate_model(m, X_test, y_test, fn_cost=fn_cost, fp_cost=fp_cost)
        r["variant"] = label
        r["scale_pos_weight"] = round(float(spw_val), 2)

        # Business impact analysis
        r["fraud_loss_saved"]  = round(r["tp"] * 500.0, 2)  # avg $500 per detected fraud
        r["false_alarm_cost"]  = round(r["fp"] * 20.0, 2)   # $20 per false alarm
        r["net_value"]         = round(r["fraud_loss_saved"] - r["false_alarm_cost"], 2)
        results.append(r)

    # Print comparison
    print("\n{:<35} {:>10} {:>10} {:>10} {:>12} {:>12}".format(
        "Variant", "Recall", "Precision", "AUC-ROC", "BizCost", "Net Value"))
    print("-" * 92)
    for r in results:
        print("{:<35} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.0f} {:>12.0f}".format(
            r["variant"], r["recall"], r["precision"],
            r["auc_roc"], r["business_cost"], r["net_value"]))

    delta_recall = results[1]["recall"] - results[0]["recall"]
    delta_cost   = results[0]["business_cost"] - results[1]["business_cost"]
    print(f"\n[analysis] Cost-sensitive improves recall by {delta_recall:+.4f}")
    print(f"[analysis] Reduces business cost by {delta_cost:.0f} units")

    with open(os.path.join(METRICS_DIR, "cost_sensitive_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved -> {METRICS_DIR}/cost_sensitive_comparison.json")
    return results


def task_drift():
    """Task 7: Drift simulation."""
    print("\n" + "="*60)
    print("TASK 7: Drift Simulation")
    print("="*60)
    from drift.simulate_drift import run_drift_simulation
    result = run_drift_simulation(output_dir=METRICS_DIR, plots_dir=PLOTS_DIR)
    print(f"[OK] Drift simulation complete.")
    return result


def task_retrain():
    """Task 8: Retraining strategy comparison."""
    print("\n" + "="*60)
    print("TASK 8: Retraining Strategy Comparison")
    print("="*60)
    from drift.retraining_strategy import compare_strategies
    result = compare_strategies(output_dir=METRICS_DIR)
    print(f"[OK] Retraining strategy comparison complete.")
    return result


def task_explain():
    """Task 9: SHAP explainability."""
    print("\n" + "="*60)
    print("TASK 9: SHAP Explainability Analysis")
    print("="*60)
    import shap
    import xgboost as xgb
    import matplotlib.pyplot as plt

    df = generate_synthetic_data(n=20000, seed=42)
    X_train, X_test, y_train, y_test, feat_names, _ = preprocess_local(df)
    n_pos = y_train.sum()
    n_neg = (y_train == 0).sum()
    spw   = n_neg / max(n_pos, 1)

    print("[explain] Training XGBoost for SHAP analysis...")
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                               scale_pos_weight=spw * 10, random_state=42, verbosity=0,
                               use_label_encoder=False)
    model.fit(X_train, y_train)

    # SHAP TreeExplainer on 1000-row subset
    sample_idx = np.random.default_rng(42).choice(len(X_test), min(1000, len(X_test)), replace=False)
    X_shap = X_test.values[sample_idx]

    print("[explain] Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feat_names[:len(mean_abs_shap)],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n[explain] Top-10 fraud prediction features (SHAP):")
    print("{:<35} {:>12}".format("Feature", "Mean |SHAP|"))
    print("-" * 50)
    for _, row in shap_df.head(10).iterrows():
        print(f"  {row['feature']:<33s}  {row['mean_abs_shap']:.6f}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    top20 = shap_df.head(20)
    ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance – Fraud Detection\n(Why is the model predicting fraud?)")
    fig.tight_layout()
    shap_plot_path = os.path.join(PLOTS_DIR, "shap_feature_importance.png")
    fig.savefig(shap_plot_path, dpi=120)
    plt.close(fig)
    print(f"[OK] SHAP plot saved -> {shap_plot_path}")

    # Save JSON report
    report = {
        "n_samples_explained": len(sample_idx),
        "top_20_features": shap_df.head(20).to_dict(orient="records"),
        "interpretation": (
            "The model flags a transaction as fraud primarily based on the "
            "features listed above. High SHAP values indicate that when these "
            "features deviate from the norm, they strongly push the model's "
            "prediction toward 'fraud'. Key signals are often transaction "
            "amount, card frequency patterns, and time-related V-features."
        ),
    }
    with open(os.path.join(METRICS_DIR, "shap_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] SHAP report saved -> {METRICS_DIR}/shap_report.json")
    return report


# ------------------------------------------------------------------ #
# Main entry point                                                     #
# ------------------------------------------------------------------ #

TASK_MAP = {
    "pipeline": task_pipeline,
    "models":   task_models,
    "imbalance": task_imbalance,
    "cost":     task_cost_sensitive,
    "drift":    task_drift,
    "retrain":  task_retrain,
    "explain":  task_explain,
}


def main():
    parser = argparse.ArgumentParser(description="Run fraud detection MLOps assignment tasks")
    parser.add_argument("--task", default="all",
                        choices=["all"] + list(TASK_MAP.keys()),
                        help="Which task to run (default: all)")
    args = parser.parse_args()

    print("\n" + "#"*60)
    print("# MLOps Assignment #4 – Fraud Detection System")
    print("# Running tasks locally (no Kubernetes required)")
    print("#"*60)

    all_results = {}
    start_total = time.time()

    tasks = list(TASK_MAP.keys()) if args.task == "all" else [args.task]

    for task_name in tasks:
        try:
            result = TASK_MAP[task_name]()
            all_results[task_name] = "OK"
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Task '{task_name}' failed: {e}")
            traceback.print_exc()
            all_results[task_name] = f"FAILED: {e}"

    elapsed = time.time() - start_total
    print("\n" + "#"*60)
    print(f"# All tasks completed in {elapsed:.1f}s")
    print("#"*60)
    print("\nTask Summary:")
    for task_name, status in all_results.items():
        icon = "[OK]  " if status == "OK" else "[FAIL]"
        print(f"  {icon} {task_name}")

    with open(os.path.join(METRICS_DIR, "run_summary.json"), "w") as f:
        json.dump({"tasks": all_results, "total_time_s": round(elapsed, 2)}, f, indent=2)


if __name__ == "__main__":
    main()
