"""
Stage 5: Model Training
Trains one of three model types:
  - xgboost:   XGBoost with cost-sensitive weighting + early stopping
  - lightgbm:  LightGBM with cost-sensitive weighting
  - rf_hybrid: Random Forest feature selection → XGBoost re-fit (hybrid)

Implements cost-sensitive learning (Task 4) via scale_pos_weight penalty.
Produces SHAP feature importance (Task 9) on a 1000-row subset.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.6.1",
        "xgboost==2.1.3",
        "lightgbm==4.5.0",
        "shap==0.46.0",
        "joblib",
    ],
)
def model_training(
    train_dataset: Input[Dataset],
    trained_model: Output[Model],
    train_metrics: Output[Metrics],
    shap_report: Output[Dataset],
    model_type: str = "xgboost",
    use_cost_sensitive: bool = True,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    random_state: int = 42,
    imbalance_method: str = "class_weight",
) -> None:
    """
    Train fraud detection model.

    model_type:        "xgboost" | "lightgbm" | "rf_hybrid"
    use_cost_sensitive: If True, multiply positive class weight by fn_cost/fp_cost.
    fn_cost:           Cost penalty for false negatives (missed fraud).
    fp_cost:           Cost penalty for false positives (false alarms).
    """
    import json
    import os
    import time
    import warnings
    import numpy as np
    import pandas as pd
    import joblib
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

    warnings.filterwarnings("ignore")
    print(
        f"[model_training] Stage 5 – Training '{model_type}' "
        f"(cost_sensitive={use_cost_sensitive}, fn_cost={fn_cost}) ..."
    )

    # ------------------------------------------------------------------ #
    # Load data                                                            #
    # ------------------------------------------------------------------ #
    train_df = pd.read_csv(train_dataset.path)
    X = train_df.drop(columns=["isFraud"]).values.astype(np.float32)
    y = train_df["isFraud"].values.astype(int)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    base_spw = n_neg / max(n_pos, 1)

    # Cost-sensitive weight: amplify the positive class by fn_cost/fp_cost
    cost_spw = base_spw * (fn_cost / fp_cost) if use_cost_sensitive else base_spw

    feature_names = train_df.drop(columns=["isFraud"]).columns.tolist()
    print(f"[model_training] Train shape: {X.shape}, Fraud rate: {y.mean():.4f}")
    print(f"[model_training] Base scale_pos_weight: {base_spw:.2f} | " f"Cost-sensitive: {cost_spw:.2f}")

    # Validation split for early stopping (last 15% of training rows)
    val_size = int(len(X) * 0.15)
    X_tr, X_val = X[:-val_size], X[-val_size:]
    y_tr, y_val = y[:-val_size], y[-val_size:]

    # ------------------------------------------------------------------ #
    # Train model                                                          #
    # ------------------------------------------------------------------ #
    start = time.time()

    if model_type == "xgboost":
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            scale_pos_weight=cost_spw,
            eval_metric="aucpr",
            early_stopping_rounds=50,
            random_state=random_state,
            verbosity=0,
            use_label_encoder=False,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_iter = model.best_iteration
        print(f"[model_training] XGBoost best iteration: {best_iter}")

    elif model_type == "lightgbm":
        import lightgbm as lgb

        if use_cost_sensitive:
            # Manual class weighting via is_unbalance=False + scale_pos_weight
            model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=63,
                scale_pos_weight=cost_spw,
                early_stopping_rounds=50,
                random_state=random_state,
                verbose=-1,
            )
        else:
            model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=63,
                class_weight="balanced",
                early_stopping_rounds=50,
                random_state=random_state,
                verbose=-1,
            )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

    elif model_type == "rf_hybrid":
        # Step 1: Quick RF to identify top features
        print("[model_training] RF Hybrid – Step 1: RF feature selection...")
        rf_selector = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        rf_selector.fit(X_tr, y_tr)

        # Select top-50 features by importance
        importances = rf_selector.feature_importances_
        top_idx = np.argsort(importances)[::-1][:50]
        top_features = [feature_names[i] for i in sorted(top_idx)]
        print(f"[model_training] RF Hybrid – Selected top {len(top_idx)} features")

        # Step 2: Re-fit XGBoost on reduced feature set
        import xgboost as xgb

        print("[model_training] RF Hybrid – Step 2: XGBoost on reduced features...")
        X_tr_reduced = X_tr[:, sorted(top_idx)]
        X_val_reduced = X_val[:, sorted(top_idx)]

        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=cost_spw,
            eval_metric="aucpr",
            early_stopping_rounds=50,
            random_state=random_state,
            verbosity=0,
            use_label_encoder=False,
        )
        xgb_model.fit(
            X_tr_reduced,
            y_tr,
            eval_set=[(X_val_reduced, y_val)],
            verbose=False,
        )

        # Wrap both steps in a pipeline-like dict for saving
        model = {
            "type": "rf_hybrid",
            "rf_selector": rf_selector,
            "xgb_model": xgb_model,
            "selected_indices": sorted(top_idx),
            "selected_features": top_features,
        }

        # For prediction, replace X with reduced version
        X = X[:, sorted(top_idx)]
        X_val = X_val[:, sorted(top_idx)]
        feature_names = top_features

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose: xgboost | lightgbm | rf_hybrid")

    elapsed = time.time() - start
    print(f"[model_training] Training completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------ #
    # Training-set predictions for cost analysis                          #
    # ------------------------------------------------------------------ #
    if model_type == "rf_hybrid":
        pred_model = model["xgb_model"]
        y_pred = pred_model.predict(X)
        y_prob = pred_model.predict_proba(X)[:, 1]
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

    recall = float(recall_score(y, y_pred, zero_division=0))
    precision = float(precision_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    auc = float(roc_auc_score(y, y_prob))

    # Business cost comparison
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    standard_cost = fn_cost * fn + fp_cost * fp
    print(
        f"[model_training] Train  – Recall: {recall:.4f} | AUC: {auc:.4f} "
        f"| FN: {fn} | FP: {fp} | Business Cost: {standard_cost:.0f}"
    )

    # ------------------------------------------------------------------ #
    # SHAP feature importance (Task 9)                                    #
    # Subsample to 1000 rows for memory efficiency                        #
    # ------------------------------------------------------------------ #
    print("[model_training] Computing SHAP values (1000-sample subset)...")
    shap_sample_size = min(1000, len(X))
    rng = np.random.default_rng(random_state)
    shap_idx = rng.choice(len(X), size=shap_sample_size, replace=False)
    X_shap = X[shap_idx]

    try:
        if model_type == "rf_hybrid":
            explainer = shap.TreeExplainer(model["xgb_model"])
        else:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (fraud)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame(
            {
                "feature": feature_names[: len(mean_abs_shap)],
                "mean_abs_shap": mean_abs_shap.tolist(),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        top_shap = shap_df.head(20)
        print("[model_training] Top-5 SHAP features (fraud prediction):")
        for _, row in top_shap.head(5).iterrows():
            print(f"  {row['feature']:<35s}  SHAP={row['mean_abs_shap']:.6f}")

        shap_report_data = {
            "model_type": model_type,
            "n_samples_explained": shap_sample_size,
            "top_20_features": top_shap.to_dict(orient="records"),
            "explanation": (
                "Mean absolute SHAP value per feature. Higher = more influential "
                "in fraud prediction decisions. Features are ranked by average "
                "impact magnitude across the 1000 explained samples."
            ),
        }
    except Exception as e:
        print(f"[model_training] SHAP computation failed: {e}. Falling back to native importances.")
        if model_type in ("xgboost", "lightgbm"):
            imps = model.feature_importances_
        elif model_type == "rf_hybrid":
            imps = model["xgb_model"].feature_importances_

        shap_df = pd.DataFrame(
            {
                "feature": feature_names[: len(imps)],
                "mean_abs_shap": imps.tolist(),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        shap_report_data = {
            "model_type": model_type,
            "n_samples_explained": 0,
            "top_20_features": shap_df.head(20).to_dict(orient="records"),
            "explanation": "Native feature importances (SHAP fallback)",
        }

    # ------------------------------------------------------------------ #
    # Save model                                                           #
    # ------------------------------------------------------------------ #
    os.makedirs(trained_model.path, exist_ok=True)
    model_path = os.path.join(trained_model.path, "model.joblib")
    joblib.dump(model, model_path)
    # Save metadata alongside model
    meta = {
        "model_type": model_type,
        "use_cost_sensitive": use_cost_sensitive,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "cost_scale_pos_weight": cost_spw,
        "n_features": len(feature_names),
        "feature_names": feature_names[:100],  # first 100 to keep file small
        "training_time_s": round(elapsed, 2),
    }
    with open(os.path.join(trained_model.path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[model_training] Model saved -> {model_path}")

    # Save SHAP report
    os.makedirs(os.path.dirname(shap_report.path), exist_ok=True)
    with open(shap_report.path, "w") as f:
        json.dump(shap_report_data, f, indent=2)
    print(f"[model_training] SHAP report -> {shap_report.path}")

    # ------------------------------------------------------------------ #
    # Log metrics                                                          #
    # ------------------------------------------------------------------ #
    train_metrics.log_metric("train_recall", round(recall, 4))
    train_metrics.log_metric("train_precision", round(precision, 4))
    train_metrics.log_metric("train_f1", round(f1, 4))
    train_metrics.log_metric("train_auc", round(auc, 4))
    train_metrics.log_metric("fn_count", int(fn))
    train_metrics.log_metric("fp_count", int(fp))
    train_metrics.log_metric("business_cost", round(standard_cost, 2))
    train_metrics.log_metric("training_time_s", round(elapsed, 2))
    train_metrics.log_metric("scale_pos_weight", round(cost_spw, 4))

    print("\n[model_training] ===== TRAINING SUMMARY =====")
    print(f"  Model          : {model_type}")
    print(f"  Cost-sensitive : {use_cost_sensitive} (fn_cost={fn_cost}x)")
    print(f"  Train recall   : {recall:.4f}")
    print(f"  Train AUC      : {auc:.4f}")
    print(f"  Business cost  : {standard_cost:.0f}")
    print(f"  Training time  : {elapsed:.1f}s")
    print("[model_training] Stage 5 complete.")
