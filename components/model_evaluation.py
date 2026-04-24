"""
Stage 6: Model Evaluation
Evaluates the trained model on the held-out test set.
Computes: Precision, Recall, F1-score, AUC-ROC, Confusion Matrix.
Generates PR curve and ROC curve plots.
Writes recall_value to OutputPath(float) for dsl.Condition in Stage 7.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, ClassificationMetrics


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.6.1",
        "xgboost==2.1.3",
        "lightgbm==4.5.0",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "joblib",
    ],
)
def model_evaluation(
    test_dataset: Input[Dataset],
    trained_model: Input[Model],
    eval_metrics: Output[Metrics],
    confusion_matrix_artifact: Output[ClassificationMetrics],
    plots_artifact: Output[Dataset],
    recall_output: Output[Dataset],
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    classification_threshold: float = 0.5,
) -> None:
    """
    Evaluate model on the test set.
    Writes fraud_recall to recall_output (JSON) for conditional deployment.
    """
    import json
    import os
    import warnings
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        average_precision_score,
        precision_recall_curve,
        roc_curve,
    )

    warnings.filterwarnings("ignore")
    print("[model_evaluation] Stage 6 – Model Evaluation starting...")

    # ------------------------------------------------------------------ #
    # Load data and model                                                  #
    # ------------------------------------------------------------------ #
    test_df = pd.read_csv(test_dataset.path)
    X_test = test_df.drop(columns=["isFraud"]).values.astype(np.float32)
    y_test = test_df["isFraud"].values.astype(int)

    model_path = os.path.join(trained_model.path, "model.joblib")
    meta_path = os.path.join(trained_model.path, "metadata.json")

    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    model_type = meta.get("model_type", "xgboost")
    print(f"[model_evaluation] Model type: {model_type}  | Test rows: {len(y_test):,}")

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #
    if model_type == "rf_hybrid":
        sel_idx = model["selected_indices"]
        X_test = X_test[:, sel_idx]
        pred_mdl = model["xgb_model"]
        y_prob = pred_mdl.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= classification_threshold).astype(int)

    # ------------------------------------------------------------------ #
    # Metrics (binary – fraud = positive class)                           #
    # ------------------------------------------------------------------ #
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc_roc = float(roc_auc_score(y_test, y_prob))
    avg_prec = float(average_precision_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / max(fp + tn, 1)

    # Business cost
    total_cost = fn_cost * fn + fp_cost * fp

    print("\n[model_evaluation] ===== TEST SET RESULTS =====")
    print(f"  Precision    : {precision:.4f}")
    print(f"  Recall       : {recall:.4f}  ← fraud recall (key metric)")
    print(f"  F1-score     : {f1:.4f}")
    print(f"  AUC-ROC      : {auc_roc:.4f}")
    print(f"  Avg Precision: {avg_prec:.4f}")
    print(f"  FPR          : {fpr_val:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Business cost (fn×{fn_cost}+fp×{fp_cost}): {total_cost:.0f}")

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    os.makedirs(plots_artifact.path, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Confusion Matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix – {model_type}\nRecall={recall:.3f}  AUC={auc_roc:.3f}", fontsize=13)
    cm_path = os.path.join(plots_artifact.path, "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    # 2. ROC Curve
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, color="steelblue", lw=2, label=f"AUC = {auc_roc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve – {model_type}", fontsize=13)
    ax.legend(loc="lower right")
    roc_path = os.path.join(plots_artifact.path, "roc_curve.png")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)

    # 3. Precision-Recall Curve
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_arr, prec_arr, color="darkorange", lw=2, label=f"AP = {avg_prec:.4f}")
    ax.axhline(y=y_test.mean(), color="gray", linestyle="--", lw=1, label="Baseline (no-skill)")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve – {model_type}", fontsize=13)
    ax.legend()
    pr_path = os.path.join(plots_artifact.path, "pr_curve.png")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=120)
    plt.close(fig)

    # 4. Prediction confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax_i, (label, mask) in enumerate([(0, y_test == 0), (1, y_test == 1)]):
        axes[ax_i].hist(
            y_prob[mask], bins=50, color="steelblue" if label == 0 else "tomato", edgecolor="white", alpha=0.8
        )
        axes[ax_i].set_title(f"Confidence Distribution – {'Legit' if label==0 else 'Fraud'}", fontsize=11)
        axes[ax_i].set_xlabel("Predicted Fraud Probability")
        axes[ax_i].set_ylabel("Count")
    fig.suptitle(f"Model: {model_type}", fontsize=13)
    fig.tight_layout()
    conf_path = os.path.join(plots_artifact.path, "confidence_distribution.png")
    fig.savefig(conf_path, dpi=120)
    plt.close(fig)

    print(f"[model_evaluation] Plots saved -> {plots_artifact.path}")

    # ------------------------------------------------------------------ #
    # Confusion Matrix KFP artifact                                       #
    # ------------------------------------------------------------------ #
    confusion_matrix_artifact.log_confusion_matrix(
        ["Legitimate", "Fraud"],
        [[int(tn), int(fp)], [int(fn), int(tp)]],
    )

    # ------------------------------------------------------------------ #
    # Write recall_output for dsl.Condition                               #
    # ------------------------------------------------------------------ #
    recall_data = {
        "fraud_recall": round(recall, 6),
        "auc_roc": round(auc_roc, 6),
        "f1": round(f1, 6),
        "model_type": model_type,
    }
    os.makedirs(os.path.dirname(recall_output.path), exist_ok=True)
    with open(recall_output.path, "w") as f:
        json.dump(recall_data, f)

    # ------------------------------------------------------------------ #
    # Log all metrics                                                      #
    # ------------------------------------------------------------------ #
    eval_metrics.log_metric("precision", round(precision, 4))
    eval_metrics.log_metric("recall", round(recall, 4))
    eval_metrics.log_metric("f1_score", round(f1, 4))
    eval_metrics.log_metric("auc_roc", round(auc_roc, 4))
    eval_metrics.log_metric("avg_precision", round(avg_prec, 4))
    eval_metrics.log_metric("false_positive_rate", round(fpr_val, 4))
    eval_metrics.log_metric("true_positives", int(tp))
    eval_metrics.log_metric("false_positives", int(fp))
    eval_metrics.log_metric("false_negatives", int(fn))
    eval_metrics.log_metric("true_negatives", int(tn))
    eval_metrics.log_metric("business_cost", round(total_cost, 2))
    eval_metrics.log_metric("fraud_rate_test", round(float(y_test.mean()), 6))

    print("[model_evaluation] Stage 6 complete.")
