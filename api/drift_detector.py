"""
Drift detector using Kolmogorov-Smirnov test.
Compares current batch distributions to a reference (training) distribution.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats


REFERENCE_STATS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "metrics", "reference_distribution.json"
)


class KSDriftDetector:
    """
    Detects feature distribution drift using the KS two-sample test.
    Reference distribution is loaded from a JSON file that stores
    training set percentiles for each numeric feature.
    """

    def __init__(self, reference_path: str = REFERENCE_STATS_PATH):
        self.reference_path = reference_path
        self._reference = {}
        self._loaded = False
        self._load()

    def _load(self):
        if os.path.exists(self.reference_path):
            with open(self.reference_path) as f:
                self._reference = json.load(f)
            self._loaded = True
            print(f"[drift_detector] Loaded reference stats for {len(self._reference)} features")
        else:
            print(f"[drift_detector] No reference stats at {self.reference_path}. Drift detection disabled.")

    def save_reference(self, df: pd.DataFrame, feature_cols: list = None):
        """Compute and save reference statistics from a training DataFrame."""
        cols = feature_cols or df.select_dtypes(include=[np.number]).columns.tolist()
        reference = {}
        for col in cols[:50]:  # limit to first 50 features for speed
            vals = df[col].dropna().values
            if len(vals) == 0:
                continue
            reference[col] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "p5":   float(np.percentile(vals, 5)),
                "p25":  float(np.percentile(vals, 25)),
                "p50":  float(np.percentile(vals, 50)),
                "p75":  float(np.percentile(vals, 75)),
                "p95":  float(np.percentile(vals, 95)),
                "n":    len(vals),
                # Store 200 samples for KS test
                "sample": np.random.default_rng(42).choice(vals, min(200, len(vals)),
                                                            replace=False).tolist(),
            }
        os.makedirs(os.path.dirname(self.reference_path), exist_ok=True)
        with open(self.reference_path, "w") as f:
            json.dump(reference, f)
        self._reference = reference
        self._loaded = True
        print(f"[drift_detector] Saved reference for {len(reference)} features -> {self.reference_path}")

    def detect(self, current_df: pd.DataFrame, ks_pvalue_threshold: float = 0.05) -> dict:
        """
        Run KS test on each feature.
        Returns per-feature drift scores and overall drift flag.
        """
        if not self._loaded or not self._reference:
            return {"drift_detected": False, "reason": "No reference data", "features": {}}

        feature_scores = {}
        drifted_features = []

        for col, ref_stats in self._reference.items():
            if col not in current_df.columns:
                continue
            curr_vals = current_df[col].dropna().values
            if len(curr_vals) < 10:
                continue

            ref_sample = np.array(ref_stats.get("sample", []))
            if len(ref_sample) == 0:
                continue

            ks_stat, p_value = stats.ks_2samp(ref_sample, curr_vals[:200])
            is_drifted = p_value < ks_pvalue_threshold

            feature_scores[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value":      round(float(p_value), 6),
                "drifted":      bool(is_drifted),
                "ref_mean":     round(ref_stats["mean"], 4),
                "curr_mean":    round(float(np.mean(curr_vals)), 4),
            }
            if is_drifted:
                drifted_features.append(col)

        max_ks = max((v["ks_statistic"] for v in feature_scores.values()), default=0.0)
        drift_detected = len(drifted_features) > 0

        return {
            "drift_detected":  drift_detected,
            "n_features_checked": len(feature_scores),
            "n_drifted_features": len(drifted_features),
            "drifted_features": drifted_features[:10],
            "max_ks_statistic": round(max_ks, 4),
            "features": feature_scores,
        }


# Module-level singleton
detector = KSDriftDetector()
