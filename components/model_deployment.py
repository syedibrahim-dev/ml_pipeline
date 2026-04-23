"""
Stage 7: Conditional Model Deployment
Deploys the trained model to the model registry only if
fraud recall >= recall_threshold.

This implements the conditional deployment requirement from Task 1.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Model, Metrics, Dataset


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["joblib"],
)
def model_deployment(
    trained_model: Input[Model],
    recall_output: Input[Dataset],
    deployment_status: Output[Metrics],
    recall_threshold: float = 0.85,
    model_registry_path: str = "/tmp/fraud_model_registry",
) -> None:
    """
    Conditionally deploy model to registry based on recall threshold.
    If recall >= recall_threshold: champion model is updated.
    Otherwise: model is rejected with a reason log.
    """
    import json
    import os
    import shutil
    import joblib
    from pathlib import Path

    print("[model_deployment] Stage 7 – Conditional Deployment starting...")

    # ------------------------------------------------------------------ #
    # Read recall from evaluation output                                  #
    # ------------------------------------------------------------------ #
    with open(recall_output.path) as f:
        recall_data = json.load(f)

    fraud_recall = float(recall_data.get("fraud_recall", 0.0))
    auc_roc      = float(recall_data.get("auc_roc", 0.0))
    model_type   = recall_data.get("model_type", "unknown")

    print(f"[model_deployment] Model type      : {model_type}")
    print(f"[model_deployment] Fraud recall    : {fraud_recall:.4f}")
    print(f"[model_deployment] AUC-ROC         : {auc_roc:.4f}")
    print(f"[model_deployment] Recall threshold: {recall_threshold:.2f}")

    # ------------------------------------------------------------------ #
    # Deployment decision                                                  #
    # ------------------------------------------------------------------ #
    should_deploy = fraud_recall >= recall_threshold

    if should_deploy:
        print(f"[model_deployment] DEPLOY: recall {fraud_recall:.4f} >= threshold {recall_threshold:.2f}")

        # Create registry paths
        champion_dir = os.path.join(model_registry_path, "champion")
        archive_dir  = os.path.join(model_registry_path, "archive")
        os.makedirs(champion_dir, exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)

        # Archive previous champion if it exists
        prev_champion = os.path.join(champion_dir, "model.joblib")
        if os.path.exists(prev_champion):
            prev_meta = os.path.join(champion_dir, "metadata.json")
            archive_name = "previous_champion"
            prev_archive = os.path.join(archive_dir, archive_name)
            os.makedirs(prev_archive, exist_ok=True)
            shutil.copy2(prev_champion, os.path.join(prev_archive, "model.joblib"))
            if os.path.exists(prev_meta):
                shutil.copy2(prev_meta, os.path.join(prev_archive, "metadata.json"))
            print(f"[model_deployment] Archived previous champion -> {prev_archive}")

        # Copy new champion
        src_model = os.path.join(trained_model.path, "model.joblib")
        src_meta  = os.path.join(trained_model.path, "metadata.json")
        shutil.copy2(src_model, os.path.join(champion_dir, "model.joblib"))
        if os.path.exists(src_meta):
            shutil.copy2(src_meta, os.path.join(champion_dir, "metadata.json"))

        # Write deployment manifest
        manifest = {
            "status": "deployed",
            "model_type": model_type,
            "fraud_recall": round(fraud_recall, 6),
            "auc_roc": round(auc_roc, 6),
            "recall_threshold": recall_threshold,
            "champion_path": champion_dir,
            "message": f"Model deployed: recall {fraud_recall:.4f} meets threshold {recall_threshold:.2f}",
        }
        with open(os.path.join(champion_dir, "deployment_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[model_deployment] Champion model saved -> {champion_dir}")

    else:
        print(f"[model_deployment] REJECT: recall {fraud_recall:.4f} < threshold {recall_threshold:.2f}")
        print(f"[model_deployment] Model NOT deployed. Keeping existing champion.")

        manifest = {
            "status": "rejected",
            "model_type": model_type,
            "fraud_recall": round(fraud_recall, 6),
            "recall_threshold": recall_threshold,
            "message": (
                f"Recall {fraud_recall:.4f} is below the required {recall_threshold:.2f} "
                f"threshold. Model rejected to maintain production quality."
            ),
        }

    # ------------------------------------------------------------------ #
    # Log deployment status                                               #
    # ------------------------------------------------------------------ #
    deployment_status.log_metric("deployed", int(should_deploy))
    deployment_status.log_metric("fraud_recall", round(fraud_recall, 4))
    deployment_status.log_metric("recall_threshold", recall_threshold)
    deployment_status.log_metric("auc_roc", round(auc_roc, 4))

    print(f"\n[model_deployment] ===== DEPLOYMENT SUMMARY =====")
    print(f"  Status    : {'DEPLOYED' if should_deploy else 'REJECTED'}")
    print(f"  Recall    : {fraud_recall:.4f} (threshold: {recall_threshold:.2f})")
    print(f"  AUC-ROC   : {auc_roc:.4f}")
    print("[model_deployment] Stage 7 complete.")
