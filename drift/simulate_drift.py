"""
Task 7: Drift Simulation
=========================
Simulates realistic data drift in the IEEE CIS fraud detection context:

1. Time-based drift   – train on early TransactionDT, test on later distribution
2. Pattern drift      – inject new fraud patterns (amount spikes, new device types)
3. Feature importance shift – compare which features matter before vs after drift

Usage:
  python drift/simulate_drift.py --simulate --output results/drift_report.json
  python drift/simulate_drift.py --check-only --output results/drift_report.json
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ #
# Synthetic data generator (same schema as data_ingestion component)  #
# ------------------------------------------------------------------ #


def generate_base_data(n: int = 30000, seed: int = 42) -> pd.DataFrame:
    """Generate baseline synthetic data spanning 6 months of TransactionDT."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n * 0.035)
    n_legit = n - n_fraud
    is_fraud = np.array([0] * n_legit + [1] * n_fraud)
    rng.shuffle(is_fraud)

    # 6-month span: 0 to 15,552,000 seconds
    tx_dt = rng.integers(0, 15_552_000, size=n)
    tx_amt = rng.lognormal(mean=4.0, sigma=2.0, size=n).round(2)
    tx_amt = np.clip(tx_amt, 0.01, 50000.0)

    v_data = rng.standard_normal((n, 30)).astype(np.float32)
    v_data[rng.random((n, 30)) < 0.40] = np.nan

    c_data = rng.integers(0, 500, size=(n, 8)).astype(float)
    d_data = rng.integers(0, 300, size=(n, 5)).astype(float)
    d_data[rng.random((n, 5)) < 0.5] = np.nan

    df = pd.DataFrame(
        {
            "TransactionDT": tx_dt,
            "TransactionAmt": tx_amt,
            "card1": rng.integers(1000, 18500, size=n).astype(float),
            "card2": rng.choice([float("nan")] + list(range(100, 600)), size=n),
            "addr1": rng.choice(list(range(100, 500)) + [float("nan")], size=n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", float("nan")], size=n),
            **{f"C{i}": c_data[:, i - 1] for i in range(1, 9)},
            **{f"D{i}": d_data[:, i - 1] for i in range(1, 6)},
            **{f"M{i}": rng.choice(["T", "F", float("nan")], size=n) for i in range(1, 5)},
            **{f"V{i}": v_data[:, i - 1] for i in range(1, 31)},
            "isFraud": is_fraud,
        }
    )
    return df


# ------------------------------------------------------------------ #
# Drift simulation functions                                           #
# ------------------------------------------------------------------ #


def split_temporal(df: pd.DataFrame, early_pct: float = 0.70):
    """Split on TransactionDT: early = train, late = test (simulates real deployment)."""
    dt_thresh = df["TransactionDT"].quantile(early_pct)
    early = df[df["TransactionDT"] <= dt_thresh].copy()
    late = df[df["TransactionDT"] > dt_thresh].copy()
    return early, late


def inject_pattern_drift(df: pd.DataFrame, seed: int = 99) -> pd.DataFrame:
    """
    Introduce new fraud patterns into the late-period data:
    1. High-amount fraud spike (new pattern: fraud via large transactions)
    2. V-feature distribution shift (feature importance shift)
    3. New email domain pattern for fraud
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    n_new_fraud = int(n * 0.02)  # add 2% new fraud pattern

    # Pattern 1: high-amount fraud spike
    fraud_idx = rng.choice(df[df["isFraud"] == 0].index, size=n_new_fraud, replace=False)
    df.loc[fraud_idx, "isFraud"] = 1
    df.loc[fraud_idx, "TransactionAmt"] = rng.lognormal(mean=8.0, sigma=0.5, size=n_new_fraud)

    # Pattern 2: V-feature shift (V1-V10 mean shifts by +2 std)
    v_cols = [f"V{i}" for i in range(1, 11) if f"V{i}" in df.columns]
    for col in v_cols:
        shift = float(df[col].std(skipna=True)) * 2.0
        df.loc[fraud_idx, col] = df.loc[fraud_idx, col].fillna(0) + shift

    # Pattern 3: new email domain associated with fraud
    df.loc[fraud_idx, "P_emaildomain"] = "suspicious-domain.com"

    return df


def compute_ks_drift(reference: pd.DataFrame, current: pd.DataFrame, feature_cols: list = None) -> dict:
    """
    Compute KS two-sample test statistics between reference and current distributions.
    Returns per-feature scores and overall drift flag.
    """
    from scipy import stats

    if feature_cols is None:
        feature_cols = [
            c
            for c in reference.columns
            if c not in ("isFraud", "TransactionDT")
            and reference[c].dtype in [np.float64, np.float32, np.int64, int, float]
        ]

    scores = {}
    drifted = []

    for col in feature_cols[:30]:  # limit for speed
        ref_vals = reference[col].dropna().values[:500]
        curr_vals = current[col].dropna().values[:500]
        if len(ref_vals) < 10 or len(curr_vals) < 10:
            continue
        ks_stat, p_val = stats.ks_2samp(ref_vals, curr_vals)
        scores[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 6),
            "drifted": bool(p_val < 0.05),
            "ref_mean": round(float(np.mean(ref_vals)), 4),
            "curr_mean": round(float(np.mean(curr_vals)), 4),
        }
        if p_val < 0.05:
            drifted.append(col)

    max_ks = max((v["ks_statistic"] for v in scores.values()), default=0.0)

    return {
        "n_features_checked": len(scores),
        "n_drifted": len(drifted),
        "drifted_features": drifted[:10],
        "max_ks_stat": round(max_ks, 4),
        "drift_detected": len(drifted) > 0,
        "feature_scores": scores,
    }


def compute_feature_importance_shift(
    early_df: pd.DataFrame,
    late_df: pd.DataFrame,
) -> dict:
    """
    Train quick RF on early and late data, compare top-10 feature importances.
    Returns a dict showing which features shifted in importance.
    """
    from sklearn.ensemble import RandomForestClassifier

    def get_importances(df, seed=42):
        numeric_cols = [
            c
            for c in df.columns
            if c not in ("isFraud", "TransactionDT") and df[c].dtype in [np.float64, np.float32, np.int64, int, float]
        ]
        X = df[numeric_cols].fillna(0).values.astype(np.float32)
        y = df["isFraud"].values
        if len(np.unique(y)) < 2:
            return {}
        rf = RandomForestClassifier(n_estimators=30, max_depth=5, class_weight="balanced", random_state=seed, n_jobs=-1)
        rf.fit(X, y)
        return dict(zip(numeric_cols, rf.feature_importances_.tolist()))

    early_imp = get_importances(early_df)
    late_imp = get_importances(late_df)

    if not early_imp or not late_imp:
        return {}

    # Compute importance shifts
    common_feats = set(early_imp) & set(late_imp)
    shifts = {}
    for feat in common_feats:
        shifts[feat] = {
            "early": round(early_imp[feat], 6),
            "late": round(late_imp[feat], 6),
            "delta": round(late_imp[feat] - early_imp[feat], 6),
        }

    # Sort by absolute delta
    sorted_shifts = sorted(shifts.items(), key=lambda x: abs(x[1]["delta"]), reverse=True)
    top_shifts = dict(sorted_shifts[:10])

    print("\n[drift] Top feature importance shifts (early → late):")
    print(f"  {'Feature':<30} {'Early':>10} {'Late':>10} {'Delta':>10}")
    print("  " + "-" * 65)
    for feat, vals in top_shifts.items():
        print(f"  {feat:<30} {vals['early']:>10.6f} {vals['late']:>10.6f} {vals['delta']:>+10.6f}")

    return {
        "top_shifted_features": top_shifts,
        "n_features_compared": len(common_feats),
    }


def run_drift_simulation(output_dir: str = "results/metrics", plots_dir: str = "results/plots") -> dict:
    """Full drift simulation pipeline."""
    import matplotlib.pyplot as plt

    print("[drift_simulation] Generating synthetic data (6-month span)...")
    df = generate_base_data(n=25000, seed=42)

    # ---------------------------------------------------------------- #
    # 1. Temporal split: train on early data, evaluate on late         #
    # ---------------------------------------------------------------- #
    early_df, late_df = split_temporal(df, early_pct=0.70)
    print(f"[drift_simulation] Early period: {len(early_df):,} rows  | " f"Late period: {len(late_df):,} rows")
    print(
        f"[drift_simulation] Early fraud rate: {early_df['isFraud'].mean():.4f} | "
        f"Late fraud rate:  {late_df['isFraud'].mean():.4f}"
    )

    # ---------------------------------------------------------------- #
    # 2. Inject new fraud patterns into late data                       #
    # ---------------------------------------------------------------- #
    print("[drift_simulation] Injecting new fraud patterns into late period...")
    late_df_drifted = inject_pattern_drift(late_df, seed=99)
    print(f"[drift_simulation] Late fraud rate after injection: " f"{late_df_drifted['isFraud'].mean():.4f}")

    # ---------------------------------------------------------------- #
    # 3. KS test: baseline drift (no injection)                        #
    # ---------------------------------------------------------------- #
    print("\n[drift_simulation] Computing KS drift (natural temporal drift)...")
    drift_natural = compute_ks_drift(early_df, late_df)
    print(
        f"[drift_simulation] Natural drift – {drift_natural['n_drifted']}/{drift_natural['n_features_checked']} "
        f"features drifted | max KS: {drift_natural['max_ks_stat']:.4f}"
    )

    # ---------------------------------------------------------------- #
    # 4. KS test: after pattern injection                               #
    # ---------------------------------------------------------------- #
    print("\n[drift_simulation] Computing KS drift (after pattern injection)...")
    drift_injected = compute_ks_drift(early_df, late_df_drifted)
    print(
        f"[drift_simulation] Injected drift – {drift_injected['n_drifted']}/{drift_injected['n_features_checked']} "
        f"features drifted | max KS: {drift_injected['max_ks_stat']:.4f}"
    )

    # ---------------------------------------------------------------- #
    # 5. Feature importance shift                                       #
    # ---------------------------------------------------------------- #
    print("\n[drift_simulation] Analysing feature importance shift...")
    importance_shift = compute_feature_importance_shift(early_df, late_df_drifted)

    # ---------------------------------------------------------------- #
    # 6. Model performance degradation (time-based)                     #
    # ---------------------------------------------------------------- #
    print("\n[drift_simulation] Measuring performance degradation over time...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import recall_score, roc_auc_score

    def get_numeric_cols(df):
        return [
            c
            for c in df.columns
            if c not in ("isFraud", "TransactionDT") and df[c].dtype in [np.float64, np.float32, np.int64, int, float]
        ]

    feat_cols = get_numeric_cols(early_df)
    X_early = early_df[feat_cols].fillna(0).values.astype(np.float32)
    y_early = early_df["isFraud"].values

    model = RandomForestClassifier(n_estimators=50, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_early, y_early)

    # Evaluate on rolling 20% windows of the full dataset
    performance_over_time = []
    dt_min, dt_max = df["TransactionDT"].min(), df["TransactionDT"].max()
    window = (dt_max - dt_min) // 5  # 5 equal time windows

    for w in range(5):
        t_start = dt_min + w * window
        t_end = dt_min + (w + 1) * window
        window_df = df[(df["TransactionDT"] >= t_start) & (df["TransactionDT"] < t_end)]
        if len(window_df) < 50 or window_df["isFraud"].sum() < 5:
            continue
        X_w = window_df[feat_cols].fillna(0).values.astype(np.float32)
        y_w = window_df["isFraud"].values
        y_prob = model.predict_proba(X_w)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        recall = float(recall_score(y_w, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_w, y_prob)) if len(np.unique(y_w)) > 1 else 0.0
        performance_over_time.append(
            {
                "window": w + 1,
                "t_start": int(t_start),
                "t_end": int(t_end),
                "n_samples": len(window_df),
                "fraud_rate": round(float(y_w.mean()), 4),
                "recall": round(recall, 4),
                "auc_roc": round(auc, 4),
            }
        )
        print(
            f"  Window {w+1}: recall={recall:.4f}  AUC={auc:.4f}  " f"(n={len(window_df)}, fraud_rate={y_w.mean():.4f})"
        )

    # ---------------------------------------------------------------- #
    # 7. Visualisations                                                  #
    # ---------------------------------------------------------------- #
    os.makedirs(plots_dir, exist_ok=True)

    if performance_over_time:
        windows = [r["window"] for r in performance_over_time]
        recalls = [r["recall"] for r in performance_over_time]
        aucs = [r["auc_roc"] for r in performance_over_time]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(windows, recalls, marker="o", color="tomato", linewidth=2)
        axes[0].axhline(y=0.80, linestyle="--", color="gray", label="Recall threshold (0.80)")
        axes[0].set_xlabel("Time Window")
        axes[0].set_ylabel("Fraud Recall")
        axes[0].set_title("Model Recall Over Time (Temporal Drift)")
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(windows, aucs, marker="s", color="steelblue", linewidth=2)
        axes[1].set_xlabel("Time Window")
        axes[1].set_ylabel("AUC-ROC")
        axes[1].set_title("AUC-ROC Over Time (Temporal Drift)")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True)

        fig.suptitle("Performance Degradation Due to Temporal Data Drift", fontsize=14)
        fig.tight_layout()
        perf_plot = os.path.join(plots_dir, "drift_performance_over_time.png")
        fig.savefig(perf_plot, dpi=120)
        plt.close(fig)
        print(f"[drift_simulation] Saved performance drift plot -> {perf_plot}")

    # KS scores bar chart
    if drift_injected["feature_scores"]:
        feats = list(drift_injected["feature_scores"].keys())[:15]
        ks_vals = [drift_injected["feature_scores"][f]["ks_statistic"] for f in feats]
        colors = ["tomato" if drift_injected["feature_scores"][f]["drifted"] else "steelblue" for f in feats]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feats[::-1], ks_vals[::-1], color=colors[::-1])
        ax.axvline(x=0.1, linestyle="--", color="orange", label="Warning threshold (0.10)")
        ax.axvline(x=0.25, linestyle="--", color="red", label="Critical threshold (0.25)")
        ax.set_xlabel("KS Statistic")
        ax.set_title("Feature Drift Scores (KS Test: early vs late period)")
        ax.legend()
        fig.tight_layout()
        ks_plot = os.path.join(plots_dir, "drift_ks_scores.png")
        fig.savefig(ks_plot, dpi=120)
        plt.close(fig)
        print(f"[drift_simulation] Saved KS scores plot -> {ks_plot}")

    # ---------------------------------------------------------------- #
    # 8. Build report                                                   #
    # ---------------------------------------------------------------- #
    performance_drop = False
    if len(performance_over_time) >= 2:
        first_recall = performance_over_time[0]["recall"]
        last_recall = performance_over_time[-1]["recall"]
        performance_drop = (first_recall - last_recall) > 0.05

    report = {
        "simulation_summary": {
            "n_early_rows": len(early_df),
            "n_late_rows": len(late_df),
            "early_fraud_rate": round(float(early_df["isFraud"].mean()), 4),
            "late_fraud_rate": round(float(late_df_drifted["isFraud"].mean()), 4),
        },
        "natural_temporal_drift": drift_natural,
        "injected_pattern_drift": drift_injected,
        "feature_importance_shift": importance_shift,
        "performance_over_time": performance_over_time,
        "drift_detected": drift_injected["drift_detected"],
        "performance_drop": performance_drop,
        "max_ks_stat": drift_injected["max_ks_stat"],
        "recommendation": (
            "Retrain model immediately."
            if drift_injected["max_ks_stat"] > 0.25
            else (
                "Monitor closely – drift detected."
                if drift_injected["drift_detected"]
                else "No significant drift. Continue monitoring."
            )
        ),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "drift_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[drift_simulation] Report saved -> {out_path}")

    print("\n[drift_simulation] ===== DRIFT SUMMARY =====")
    print(f"  Natural drift   : {drift_natural['n_drifted']} features  " f"(max KS={drift_natural['max_ks_stat']:.4f})")
    print(
        f"  Injected drift  : {drift_injected['n_drifted']} features  " f"(max KS={drift_injected['max_ks_stat']:.4f})"
    )
    print(f"  Performance drop: {performance_drop}")
    print(f"  Recommendation  : {report['recommendation']}")

    return report


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud detection drift simulation")
    parser.add_argument("--simulate", action="store_true", help="Run full simulation")
    parser.add_argument("--check-only", action="store_true", help="Check drift from saved report")
    parser.add_argument("--output", default="results/metrics/drift_report.json")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)

    if args.simulate or not args.check_only:
        result = run_drift_simulation(output_dir=output_dir)
    else:
        if os.path.exists(args.output):
            with open(args.output) as f:
                result = json.load(f)
            print(f"[drift_simulation] Loaded report from {args.output}")
            print(f"  Drift detected  : {result.get('drift_detected', False)}")
            print(f"  Performance drop: {result.get('performance_drop', False)}")
            print(f"  Max KS stat     : {result.get('max_ks_stat', 0):.4f}")
        else:
            print(f"[drift_simulation] No report found at {args.output}. Run with --simulate first.")
            result = {"drift_detected": False, "performance_drop": False}

    # Exit code for CI/CD trigger
    if result.get("drift_detected") or result.get("performance_drop"):
        print("\n[drift_simulation] ACTION REQUIRED: Drift/performance drop detected.")
        sys.exit(0)
    else:
        print("\n[drift_simulation] System nominal.")
        sys.exit(0)
