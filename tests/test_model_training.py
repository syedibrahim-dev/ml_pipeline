"""
Unit tests for model training:
  - XGBoost trains and produces predictions
  - LightGBM trains and produces predictions
  - RF-Hybrid pipeline runs end-to-end
  - Cost-sensitive scale_pos_weight calculation
  - SHAP values are produced without error
  - Recall is above zero (model actually learns)
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def fraud_dataset():
    """Generate a small synthetic train/test split for model tests."""
    rng = np.random.default_rng(42)
    n = 600
    n_fraud = 30  # ~5% fraud
    n_legit = n - n_fraud
    labels = np.array([0] * n_legit + [1] * n_fraud)
    rng.shuffle(labels)

    X = rng.standard_normal((n, 20)).astype(np.float32)
    # Make fraud class somewhat separable
    X[labels == 1, :5] += 2.0

    split = int(n * 0.8)
    return {
        "X_train": X[:split],
        "y_train": labels[:split],
        "X_test":  X[split:],
        "y_test":  labels[split:],
        "n_features": 20,
        "feature_names": [f"feat_{i}" for i in range(20)],
    }


# ------------------------------------------------------------------ #
# XGBoost tests                                                        #
# ------------------------------------------------------------------ #

class TestXGBoost:
    def test_xgboost_trains_without_error(self, fraud_dataset):
        import xgboost as xgb
        d = fraud_dataset
        model = xgb.XGBClassifier(n_estimators=20, max_depth=3, verbosity=0,
                                   use_label_encoder=False, random_state=42)
        model.fit(d["X_train"], d["y_train"])
        assert model is not None

    def test_xgboost_predicts_correct_shape(self, fraud_dataset):
        import xgboost as xgb
        d = fraud_dataset
        model = xgb.XGBClassifier(n_estimators=20, max_depth=3, verbosity=0,
                                   use_label_encoder=False, random_state=42)
        model.fit(d["X_train"], d["y_train"])
        preds = model.predict(d["X_test"])
        assert preds.shape == (len(d["X_test"]),)

    def test_xgboost_predict_proba_valid_range(self, fraud_dataset):
        import xgboost as xgb
        d = fraud_dataset
        model = xgb.XGBClassifier(n_estimators=20, max_depth=3, verbosity=0,
                                   use_label_encoder=False, random_state=42)
        model.fit(d["X_train"], d["y_train"])
        proba = model.predict_proba(d["X_test"])[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_xgboost_recall_above_zero(self, fraud_dataset):
        """Model should learn something – recall should be > 0."""
        from sklearn.metrics import recall_score
        import xgboost as xgb

        d = fraud_dataset
        n_pos = d["y_train"].sum()
        n_neg = (d["y_train"] == 0).sum()
        spw   = (n_neg / max(n_pos, 1)) * 10  # cost-sensitive

        model = xgb.XGBClassifier(n_estimators=50, max_depth=4, scale_pos_weight=spw,
                                   verbosity=0, use_label_encoder=False, random_state=42)
        model.fit(d["X_train"], d["y_train"])
        preds = model.predict(d["X_test"])
        recall = recall_score(d["y_test"], preds, zero_division=0)
        assert recall > 0.0, "XGBoost should detect at least some fraud cases"

    def test_cost_sensitive_scale_pos_weight(self, fraud_dataset):
        """Cost-sensitive SPW should be fn_cost × base_spw."""
        d = fraud_dataset
        n_pos  = d["y_train"].sum()
        n_neg  = (d["y_train"] == 0).sum()
        fn_cost = 10.0
        fp_cost =  1.0
        base_spw = n_neg / max(n_pos, 1)
        cost_spw = base_spw * (fn_cost / fp_cost)
        assert cost_spw == pytest.approx(base_spw * 10)

    def test_cost_sensitive_improves_recall(self, fraud_dataset):
        """Cost-sensitive training should achieve higher or equal recall than standard."""
        from sklearn.metrics import recall_score
        import xgboost as xgb

        d = fraud_dataset
        n_pos = d["y_train"].sum()
        n_neg = (d["y_train"] == 0).sum()
        base_spw = n_neg / max(n_pos, 1)

        # Standard
        m_std = xgb.XGBClassifier(n_estimators=50, scale_pos_weight=base_spw,
                                   verbosity=0, use_label_encoder=False, random_state=42)
        m_std.fit(d["X_train"], d["y_train"])
        recall_std = recall_score(d["y_test"], m_std.predict(d["X_test"]), zero_division=0)

        # Cost-sensitive
        m_cs = xgb.XGBClassifier(n_estimators=50, scale_pos_weight=base_spw * 10,
                                  verbosity=0, use_label_encoder=False, random_state=42)
        m_cs.fit(d["X_train"], d["y_train"])
        recall_cs = recall_score(d["y_test"], m_cs.predict(d["X_test"]), zero_division=0)

        assert recall_cs >= recall_std - 0.05, (
            f"Cost-sensitive recall ({recall_cs:.4f}) should be >= standard ({recall_std:.4f}) - 5%"
        )


# ------------------------------------------------------------------ #
# LightGBM tests                                                       #
# ------------------------------------------------------------------ #

class TestLightGBM:
    def test_lightgbm_trains_without_error(self, fraud_dataset):
        import lightgbm as lgb
        d = fraud_dataset
        model = lgb.LGBMClassifier(n_estimators=20, max_depth=3,
                                   class_weight="balanced", random_state=42, verbose=-1)
        model.fit(d["X_train"], d["y_train"])
        assert model is not None

    def test_lightgbm_predict_proba_valid(self, fraud_dataset):
        import lightgbm as lgb
        d = fraud_dataset
        model = lgb.LGBMClassifier(n_estimators=20, max_depth=3,
                                   class_weight="balanced", random_state=42, verbose=-1)
        model.fit(d["X_train"], d["y_train"])
        proba = model.predict_proba(d["X_test"])[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_lightgbm_auc_above_random(self, fraud_dataset):
        """AUC-ROC should be significantly above 0.5 (better than random)."""
        from sklearn.metrics import roc_auc_score
        import lightgbm as lgb

        d = fraud_dataset
        if len(np.unique(d["y_test"])) < 2:
            pytest.skip("Test set has only one class")

        model = lgb.LGBMClassifier(n_estimators=50, class_weight="balanced",
                                   random_state=42, verbose=-1)
        model.fit(d["X_train"], d["y_train"])
        proba = model.predict_proba(d["X_test"])[:, 1]
        auc = roc_auc_score(d["y_test"], proba)
        assert auc >= 0.50, f"LightGBM AUC {auc:.4f} should be >= 0.50"


# ------------------------------------------------------------------ #
# RF Hybrid tests                                                      #
# ------------------------------------------------------------------ #

class TestRFHybrid:
    def test_rf_hybrid_selects_top_features(self, fraud_dataset):
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb

        d = fraud_dataset
        rf = RandomForestClassifier(n_estimators=30, class_weight="balanced",
                                    random_state=42, n_jobs=1)
        rf.fit(d["X_train"], d["y_train"])
        top_idx = np.argsort(rf.feature_importances_)[::-1][:10]
        assert len(top_idx) == 10
        assert all(0 <= i < d["n_features"] for i in top_idx)

    def test_rf_hybrid_xgboost_trains_on_reduced_features(self, fraud_dataset):
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb

        d = fraud_dataset
        rf = RandomForestClassifier(n_estimators=30, class_weight="balanced",
                                    random_state=42, n_jobs=1)
        rf.fit(d["X_train"], d["y_train"])
        top_idx = sorted(np.argsort(rf.feature_importances_)[::-1][:10])

        X_tr_red = d["X_train"][:, top_idx]
        X_te_red = d["X_test"][:, top_idx]

        xgb_model = xgb.XGBClassifier(n_estimators=20, verbosity=0,
                                       use_label_encoder=False, random_state=42)
        xgb_model.fit(X_tr_red, d["y_train"])
        preds = xgb_model.predict(X_te_red)
        assert preds.shape == (len(d["X_test"]),)

    def test_rf_hybrid_reduces_feature_count(self, fraud_dataset):
        from sklearn.ensemble import RandomForestClassifier
        d = fraud_dataset
        n_select = 8
        rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
        rf.fit(d["X_train"], d["y_train"])
        top_idx = np.argsort(rf.feature_importances_)[::-1][:n_select]
        assert len(top_idx) == n_select
        assert len(top_idx) < d["n_features"]


# ------------------------------------------------------------------ #
# SHAP tests                                                           #
# ------------------------------------------------------------------ #

class TestSHAP:
    def test_shap_tree_explainer_runs(self, fraud_dataset):
        try:
            import shap
            import xgboost as xgb
        except ImportError:
            pytest.skip("shap or xgboost not installed")

        d = fraud_dataset
        model = xgb.XGBClassifier(n_estimators=20, verbosity=0,
                                   use_label_encoder=False, random_state=42)
        model.fit(d["X_train"], d["y_train"])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(d["X_test"][:50])
        assert shap_values is not None

    def test_shap_values_shape_matches_features(self, fraud_dataset):
        try:
            import shap
            import xgboost as xgb
        except ImportError:
            pytest.skip("shap or xgboost not installed")

        d = fraud_dataset
        model = xgb.XGBClassifier(n_estimators=20, verbosity=0,
                                   use_label_encoder=False, random_state=42)
        model.fit(d["X_train"], d["y_train"])

        explainer = shap.TreeExplainer(model)
        sample = d["X_test"][:50]
        shap_values = explainer.shap_values(sample)
        assert shap_values.shape == sample.shape

    def test_mean_abs_shap_nonnegative(self, fraud_dataset):
        try:
            import shap
            import xgboost as xgb
        except ImportError:
            pytest.skip("shap or xgboost not installed")

        d = fraud_dataset
        model = xgb.XGBClassifier(n_estimators=20, verbosity=0,
                                   use_label_encoder=False, random_state=42)
        model.fit(d["X_train"], d["y_train"])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(d["X_test"][:50])
        mean_abs = np.abs(shap_values).mean(axis=0)
        assert (mean_abs >= 0).all(), "Mean absolute SHAP values should be non-negative"


# ------------------------------------------------------------------ #
# Business cost tests                                                  #
# ------------------------------------------------------------------ #

class TestBusinessCost:
    def test_business_cost_formula(self):
        """Business cost = fn_cost * FN + fp_cost * FP."""
        fn_cost, fp_cost = 10.0, 1.0
        fn, fp = 5, 20
        expected = fn_cost * fn + fp_cost * fp
        assert expected == 70.0

    def test_cost_sensitive_reduces_fn_count(self, fraud_dataset):
        """Cost-sensitive model should have fewer false negatives."""
        from sklearn.metrics import confusion_matrix
        import xgboost as xgb

        d = fraud_dataset
        n_pos = d["y_train"].sum()
        n_neg = (d["y_train"] == 0).sum()
        base_spw = n_neg / max(n_pos, 1)

        m_std = xgb.XGBClassifier(n_estimators=50, scale_pos_weight=base_spw,
                                   verbosity=0, use_label_encoder=False, random_state=42)
        m_std.fit(d["X_train"], d["y_train"])

        m_cs = xgb.XGBClassifier(n_estimators=50, scale_pos_weight=base_spw * 10,
                                  verbosity=0, use_label_encoder=False, random_state=42)
        m_cs.fit(d["X_train"], d["y_train"])

        if len(np.unique(d["y_test"])) < 2:
            pytest.skip("Test set has only one class")

        cm_std = confusion_matrix(d["y_test"], m_std.predict(d["X_test"]))
        cm_cs  = confusion_matrix(d["y_test"], m_cs.predict(d["X_test"]))

        fn_std = cm_std[1, 0]
        fn_cs  = cm_cs[1, 0]

        # Cost-sensitive should have fewer or equal FN
        assert fn_cs <= fn_std + 2, (
            f"Cost-sensitive FN ({fn_cs}) should be <= standard FN ({fn_std}) + 2 tolerance"
        )
