"""
Fraud Detection Kubeflow Pipeline (7 stages)

Stages:
  1. Data Ingestion
  2. Data Validation
  3. Data Preprocessing
  4. Feature Engineering
  5. Model Training
  6. Model Evaluation
  7. Conditional Deployment (deploy only if recall >= threshold)

Features:
  - Retry mechanisms on every stage
  - dsl.Condition for Stage 7 deployment gate
  - Configurable namespace, resource requests, pipeline root
  - Compiles to YAML for submission to Kubeflow UI or CLI

Usage:
  python pipelines/fraud_pipeline.py            # compile only
  python pipelines/fraud_pipeline.py --submit   # compile + submit to KFP
"""

import os
import sys
import argparse

# Ensure project root is on path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kfp import dsl, compiler
from kfp.dsl import pipeline

from components.data_ingestion    import data_ingestion
from components.data_validation   import data_validation
from components.preprocessing     import preprocessing
from components.feature_engineering import feature_engineering
from components.model_training    import model_training
from components.model_evaluation  import model_evaluation
from components.model_deployment  import model_deployment


# ------------------------------------------------------------------ #
# Pipeline definition                                                 #
# ------------------------------------------------------------------ #

@dsl.pipeline(
    name="fraud-detection-pipeline",
    description=(
        "End-to-end fraud detection pipeline using IEEE CIS Fraud Detection data. "
        "7 stages: ingestion, validation, preprocessing, feature engineering, "
        "training, evaluation, and conditional deployment."
    ),
)
def fraud_detection_pipeline(
    # Data parameters
    data_root: str = "/pipeline-root/data/raw",
    n_synthetic: int = 50000,
    fraud_rate: float = 0.035,
    # Preprocessing parameters
    imbalance_method: str = "class_weight",   # "class_weight" | "smote"
    test_size: float = 0.2,
    # Model parameters
    model_type: str = "xgboost",              # "xgboost" | "lightgbm" | "rf_hybrid"
    use_cost_sensitive: bool = True,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    # Deployment parameters
    recall_threshold: float = 0.85,
    model_registry_path: str = "/tmp/fraud_model_registry",
    # Reproducibility
    random_state: int = 42,
) -> None:
    """
    7-stage fraud detection pipeline with conditional deployment.
    All stage have retry mechanisms to handle transient failures.
    Stage 7 only executes if fraud recall meets the deployment threshold.
    """

    # ---------------------------------------------------------------- #
    # Stage 1: Data Ingestion                                          #
    # ---------------------------------------------------------------- #
    ingest_task = data_ingestion(
        data_root=data_root,
        n_synthetic=n_synthetic,
        fraud_rate=fraud_rate,
        random_state=random_state,
    )
    ingest_task.set_display_name("1. Data Ingestion")
    ingest_task.set_retry(num_retries=3, backoff_duration="30s", backoff_factor=2.0)
    ingest_task.set_cpu_request("0.5").set_memory_request("1Gi")
    ingest_task.set_cpu_limit("2").set_memory_limit("4Gi")

    # ---------------------------------------------------------------- #
    # Stage 2: Data Validation                                         #
    # ---------------------------------------------------------------- #
    validate_task = data_validation(
        input_transaction=ingest_task.outputs["output_transaction"],
        input_identity=ingest_task.outputs["output_identity"],
    )
    validate_task.set_display_name("2. Data Validation")
    validate_task.set_retry(num_retries=2, backoff_duration="10s")
    validate_task.set_cpu_request("0.25").set_memory_request("512Mi")

    # ---------------------------------------------------------------- #
    # Stage 3: Preprocessing                                           #
    # ---------------------------------------------------------------- #
    preprocess_task = preprocessing(
        input_transaction=ingest_task.outputs["output_transaction"],
        input_identity=ingest_task.outputs["output_identity"],
        imbalance_method=imbalance_method,
        test_size=test_size,
        random_state=random_state,
    )
    preprocess_task.after(validate_task)   # explicitly wait for validation
    preprocess_task.set_display_name("3. Data Preprocessing")
    preprocess_task.set_retry(num_retries=2, backoff_duration="30s", backoff_factor=2.0)
    preprocess_task.set_cpu_request("1").set_memory_request("2Gi")
    preprocess_task.set_cpu_limit("4").set_memory_limit("8Gi")

    # ---------------------------------------------------------------- #
    # Stage 4: Feature Engineering                                     #
    # ---------------------------------------------------------------- #
    feat_eng_task = feature_engineering(
        train_dataset=preprocess_task.outputs["train_dataset"],
        test_dataset=preprocess_task.outputs["test_dataset"],
    )
    feat_eng_task.set_display_name("4. Feature Engineering")
    feat_eng_task.set_retry(num_retries=2, backoff_duration="10s")
    feat_eng_task.set_cpu_request("1").set_memory_request("2Gi")
    feat_eng_task.set_cpu_limit("4").set_memory_limit("8Gi")

    # ---------------------------------------------------------------- #
    # Stage 5: Model Training                                          #
    # ---------------------------------------------------------------- #
    train_task = model_training(
        train_dataset=feat_eng_task.outputs["train_engineered"],
        model_type=model_type,
        use_cost_sensitive=use_cost_sensitive,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        random_state=random_state,
        imbalance_method=imbalance_method,
    )
    train_task.set_display_name("5. Model Training")
    train_task.set_retry(num_retries=3, backoff_duration="60s", backoff_factor=2.0)
    train_task.set_cpu_request("2").set_memory_request("4Gi")
    train_task.set_cpu_limit("4").set_memory_limit("8Gi")

    # ---------------------------------------------------------------- #
    # Stage 6: Model Evaluation                                        #
    # ---------------------------------------------------------------- #
    eval_task = model_evaluation(
        test_dataset=feat_eng_task.outputs["test_engineered"],
        trained_model=train_task.outputs["trained_model"],
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )
    eval_task.set_display_name("6. Model Evaluation")
    eval_task.set_retry(num_retries=2, backoff_duration="30s")
    eval_task.set_cpu_request("1").set_memory_request("2Gi")
    eval_task.set_cpu_limit("2").set_memory_limit("4Gi")

    # ---------------------------------------------------------------- #
    # Stage 7: Conditional Deployment                                   #
    # Deploy only if evaluation recall_output JSON contains recall >=  #
    # threshold.  We use an unconditional deployment component here     #
    # that reads the recall from the JSON and decides internally.       #
    # (KFP v2 dsl.Condition works on pipeline output params; since     #
    # recall is in a JSON artifact, we deploy unconditionally and let  #
    # the component handle the gate logic.)                            #
    # ---------------------------------------------------------------- #
    deploy_task = model_deployment(
        trained_model=train_task.outputs["trained_model"],
        recall_output=eval_task.outputs["recall_output"],
        recall_threshold=recall_threshold,
        model_registry_path=model_registry_path,
    )
    deploy_task.set_display_name("7. Conditional Deployment")
    deploy_task.set_retry(num_retries=2, backoff_duration="30s")
    deploy_task.set_cpu_request("0.25").set_memory_request("256Mi")


# ------------------------------------------------------------------ #
# Compile + optional submit                                           #
# ------------------------------------------------------------------ #

def compile_pipeline(output_dir: str = "pipelines/compiled") -> str:
    """Compile the pipeline to YAML and return the output path."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fraud_detection_pipeline.yaml")
    compiler.Compiler().compile(fraud_detection_pipeline, output_path)
    print(f"[pipeline] Compiled -> {output_path}")
    return output_path


def submit_pipeline(
    pipeline_yaml: str,
    kfp_endpoint: str = "http://localhost:3000",
    experiment_name: str = "fraud-detection",
    run_name: str = "fraud-detection-run",
) -> None:
    """Submit a compiled pipeline to a running Kubeflow Pipelines instance."""
    try:
        import kfp
        client = kfp.Client(host=kfp_endpoint)
        run = client.create_run_from_pipeline_package(
            pipeline_file=pipeline_yaml,
            arguments={},
            run_name=run_name,
            experiment_name=experiment_name,
            namespace="fraud-detection",
        )
        print(f"[pipeline] Submitted run: {run.run_id}")
        print(f"[pipeline] View at: {kfp_endpoint}/#/runs/details/{run.run_id}")
    except Exception as e:
        print(f"[pipeline] Could not submit to KFP endpoint {kfp_endpoint}: {e}")
        print("[pipeline] Pipeline YAML is ready for manual upload to the Kubeflow UI.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and optionally submit fraud detection pipeline")
    parser.add_argument("--submit", action="store_true", help="Submit to KFP after compiling")
    parser.add_argument("--endpoint", default="http://localhost:3000", help="KFP endpoint URL")
    parser.add_argument("--output-dir", default="pipelines/compiled", help="Output directory for YAML")
    args = parser.parse_args()

    yaml_path = compile_pipeline(output_dir=args.output_dir)

    if args.submit:
        submit_pipeline(pipeline_yaml=yaml_path, kfp_endpoint=args.endpoint)
    else:
        print("[pipeline] To submit to Kubeflow, run with --submit --endpoint <KFP_URL>")
        print("[pipeline] Or upload the YAML manually via the Kubeflow Pipelines UI.")
